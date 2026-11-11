import asyncio
import json
import socket
import sys
import time
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

import main as main_module
from main import app, get_local_model_settings, is_ready_hybrid_model, provider
from resource_monitor import GPUUsageSampler


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_test_database(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "DB_PATH", tmp_path / "database" / "app.sqlite")
    monkeypatch.setattr(main_module, "EXPORTS_DIR", tmp_path / "exports")
    main_module.init_db()


def test_cpu_sampler_preserves_disk_activity_reading(monkeypatch):
    sampler = GPUUsageSampler()
    sampler._readings["disk"].update({
        "usage_percent": 42.5,
        "telemetry_error": None,
        "sampled_at": "disk-activity-sample",
    })
    cpu_calls = 0

    def sample_cpu(interval=None):
        nonlocal cpu_calls
        cpu_calls += 1
        if interval is not None:
            sampler._stop_event.set()
        return 18.0

    monkeypatch.setattr("resource_monitor.psutil.cpu_percent", sample_cpu)
    monkeypatch.setattr(
        "resource_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=60.0, used=12 * 1024**3, total=20 * 1024**3),
    )
    monkeypatch.setattr(
        "resource_monitor.psutil.disk_usage",
        lambda _path: SimpleNamespace(percent=70.0, used=700 * 1024**3, free=300 * 1024**3, total=1000 * 1024**3),
    )

    sampler._sample_cpu_loop()

    disk = sampler.get_reading()["disk"]
    assert disk["usage_percent"] == 42.5
    assert disk["telemetry_error"] is None
    assert disk["sampled_at"] == "disk-activity-sample"
    assert disk["storage_percent"] == 70.0
    assert disk["storage_sampled_at"] is not None


def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'


def test_brand_logo_is_served_without_embedded_metadata():
    response = client.get('/assets/localdev-ai-mark.svg')

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in response.text
    assert "<metadata>" not in response.text


def test_system_status_reports_measurable_resources(monkeypatch):
    monkeypatch.setattr(
        "main.gpu_usage_sampler.get_reading",
        lambda: {
            "cpu": {"usage_percent": 18.0, "sampled_at": "test"},
            "ram": {"usage_percent": 60.0, "used_gb": 12.0, "total_gb": 20.0, "sampled_at": "test"},
            "disk": {
                "usage_percent": 12.5,
                "storage_percent": 70.0,
                "used_gb": 700.0,
                "free_gb": 300.0,
                "total_gb": 1000.0,
                "telemetry_error": None,
                "sampled_at": "test",
            },
            "gpu": {"usage_percent": 17.0, "telemetry_error": None, "sampled_at": "test"},
        },
    )
    response = client.get('/api/system/status')
    assert response.status_code == 200
    payload = response.json()
    assert payload['cpu']['usage_percent'] == 18.0
    assert payload['ram']['usage_percent'] == 60.0
    assert payload['disk']['usage_percent'] == 12.5
    assert payload['disk']['storage_percent'] == 70.0
    assert payload['disk']['free_gb'] == 300.0
    assert payload['disk']['status'] == 'available'
    if payload['gpu']['telemetry_available']:
        assert 0 <= payload['gpu']['usage_percent'] <= 100
    else:
        assert payload['gpu']['usage_percent'] is None
    assert 'npu' not in payload


def test_system_usage_returns_cached_readings_without_model_metrics(monkeypatch):
    readings = {
        "cpu": {"usage_percent": 18.0, "sampled_at": "shared"},
        "ram": {"usage_percent": 60.0, "used_gb": 12.0, "total_gb": 20.0, "sampled_at": "shared"},
        "disk": {
            "usage_percent": 12.5,
            "storage_percent": 70.0,
            "used_gb": 700.0,
            "free_gb": 300.0,
            "total_gb": 1000.0,
            "telemetry_error": None,
            "sampled_at": "shared",
        },
        "gpu": {"usage_percent": 17.0, "telemetry_error": None, "sampled_at": "shared"},
    }
    monkeypatch.setattr("main.gpu_usage_sampler.get_reading", lambda: readings)
    monkeypatch.setattr(
        provider,
        "get_metrics",
        lambda: (_ for _ in ()).throw(AssertionError("resource polling must not inspect model metrics")),
    )

    response = client.get('/api/system/usage')

    assert response.status_code == 200
    assert response.json() == {
        "cpu": readings["cpu"],
        "gpu": readings["gpu"],
        "disk": readings["disk"],
        "ram": readings["ram"],
    }


def test_standalone_resource_monitor_lifecycle(monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        monitor_port = listener.getsockname()[1]
    monkeypatch.setattr(main_module, "RESOURCE_MONITOR_PORT", monitor_port)

    with TestClient(app) as lifespan_client:
        process = main_module.resource_monitor_process
        assert process is not None
        assert process.is_alive()

        deadline = time.monotonic() + 5
        readings = {}
        while time.monotonic() < deadline:
            with urlopen(main_module.resource_monitor_url("/api/system/usage"), timeout=2) as response:
                readings = json.loads(response.read())
            if readings["cpu"]["sampled_at"] is not None and readings["disk"]["storage_percent"] is not None:
                break
            time.sleep(0.1)
        assert set(readings) == {"cpu", "gpu", "disk", "ram"}
        assert readings["cpu"]["sampled_at"] is not None
        assert isinstance(readings["disk"]["storage_percent"], (int, float))
        assert readings["disk"]["total_gb"] > 0
        for resource in ("gpu", "disk"):
            usage = readings[resource]["usage_percent"]
            if usage is None:
                assert readings[resource]["telemetry_error"]
            else:
                assert 0 <= usage <= 100
        first_cpu_sample_at = readings["cpu"]["sampled_at"]
        time.sleep(1)
        with urlopen(main_module.resource_monitor_url("/api/system/usage"), timeout=2) as response:
            updated_readings = json.loads(response.read())
        assert updated_readings["cpu"]["sampled_at"] != first_cpu_sample_at

        preflight = Request(
            main_module.resource_monitor_url("/api/system/usage"),
            headers={
                "Origin": "http://127.0.0.1:8000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "content-type",
            },
            method="OPTIONS",
        )
        with urlopen(preflight, timeout=2) as response:
            assert response.status == 204
            assert response.headers["Access-Control-Allow-Origin"] == "*"

        shutdown_request = Request(
            main_module.resource_monitor_url("/shutdown"),
            data=b"{}",
            method="POST",
        )
        try:
            urlopen(shutdown_request, timeout=2)
            assert False, "The monitor must reject unauthenticated shutdown requests."
        except HTTPError as exc:
            assert exc.code == 403

        api_response = lifespan_client.get("/api/system/usage")
        assert api_response.status_code == 200
        assert set(api_response.json()) == set(readings)

    assert process is not None
    assert not process.is_alive()


def test_system_status_reports_hybrid_gpu_and_disk_usage(monkeypatch):
    monkeypatch.setattr(
        provider,
        "get_metrics",
        lambda: {"status": "loaded", "execution_mode": "hybrid_npu_igpu"},
    )
    monkeypatch.setattr(
        "main.gpu_usage_sampler.get_reading",
        lambda: {
            "cpu": {"usage_percent": 20.0, "sampled_at": "test"},
            "ram": {"usage_percent": 50.0, "used_gb": 10.0, "total_gb": 20.0, "sampled_at": "test"},
            "disk": {
                "usage_percent": 22.0,
                "storage_percent": 61.0,
                "used_gb": 610.0,
                "free_gb": 390.0,
                "total_gb": 1000.0,
                "telemetry_error": None,
                "sampled_at": "test",
            },
            "gpu": {"usage_percent": 42.5, "telemetry_error": None, "sampled_at": "test"},
        },
    )
    payload = client.get('/api/system/status').json()
    assert payload['gpu']['status'] == 'ready'
    assert payload['gpu']['usage_percent'] == 42.5
    assert payload['gpu']['telemetry_available'] is True
    assert payload['disk']['usage_percent'] == 22.0
    assert payload['disk']['storage_percent'] == 61.0
    assert 'npu' not in payload


def test_streaming_chat_endpoint_preserves_server_sent_events(monkeypatch):
    monkeypatch.setattr(provider, "stream_generate", lambda messages: iter(["hello", " world"]))

    response = client.post('/api/chat/stream', json={'message': 'Say hello'})

    assert response.status_code == 200
    assert response.text == (
        'data: {"chunk": "hello"}\n\n'
        'data: {"chunk": " world"}\n\n'
        'data: [DONE]\n\n'
    )


def test_resource_status_updates_during_model_generation(monkeypatch):
    generation_started = Event()

    def slow_generate(messages):
        generation_started.set()
        time.sleep(0.4)
        return "Concurrent test response"

    monkeypatch.setattr(provider, "generate", slow_generate)
    monkeypatch.setattr(
        "main.gpu_usage_sampler.get_reading",
        lambda: {
            "cpu": {"usage_percent": 20.0, "sampled_at": "test"},
            "ram": {"usage_percent": 50.0, "used_gb": 10.0, "total_gb": 20.0, "sampled_at": "test"},
            "disk": {
                "usage_percent": 22.0,
                "storage_percent": 61.0,
                "used_gb": 610.0,
                "free_gb": 390.0,
                "total_gb": 1000.0,
                "telemetry_error": None,
                "sampled_at": "test",
            },
            "gpu": {"usage_percent": 42.5, "telemetry_error": None, "sampled_at": "test"},
        },
    )

    async def verify_status_remains_responsive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            conversation_response = await async_client.post(
                "/api/conversations",
                json={"title": "Concurrent telemetry check"},
            )
            conversation_id = conversation_response.json()["id"]
            chat_task = asyncio.create_task(
                async_client.post(
                    "/api/chat",
                    json={"message": "Test usage while generating", "conversation_id": conversation_id},
                )
            )
            for _ in range(50):
                if generation_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert generation_started.is_set()
            await asyncio.sleep(0.03)
            assert not chat_task.done()

            started = time.perf_counter()
            status_response = await async_client.get("/api/system/status")
            elapsed = time.perf_counter() - started

            assert status_response.status_code == 200
            assert elapsed < 0.2
            assert status_response.json()["gpu"]["usage_percent"] == 42.5
            usage_started = time.perf_counter()
            usage_response = await async_client.get("/api/system/usage")
            usage_elapsed = time.perf_counter() - usage_started
            assert usage_response.status_code == 200
            assert usage_elapsed < 0.2
            assert usage_response.json()["gpu"]["usage_percent"] == 42.5
            assert (await chat_task).status_code == 200
            await async_client.delete(f"/api/conversations/{conversation_id}")

    asyncio.run(verify_status_remains_responsive())


def test_model_status_reports_local_model_configuration():
    response = client.get('/api/system/model')
    assert response.status_code == 200
    payload = response.json()
    assert payload['model_path']
    if Path(payload['model_path']).is_dir():
        assert payload['status'] == 'available'
    else:
        assert payload['status'] == 'not_found'
    assert payload['backend'] in {'transformers_local', 'onnxruntime_genai'}


def test_model_selection_requires_complete_hybrid_package():
    settings = get_local_model_settings()
    selected_path = Path(settings['model_path'])
    hybrid_path = ROOT.parent / 'models' / 'Qwen2.5-Coder-7B-Instruct-RyzenAI-Hybrid'
    if is_ready_hybrid_model(hybrid_path):
        assert selected_path == hybrid_path
    else:
        assert selected_path != hybrid_path


def test_conversation_roundtrip(monkeypatch):
    monkeypatch.setattr(provider, "generate", lambda messages: "Local test response")
    response = client.post('/api/conversations', json={'title': 'Demo chat', 'project_id': None})
    assert response.status_code == 200
    conversation = response.json()
    conversation_id = conversation['id']

    chat = client.post('/api/chat', json={'message': 'Tell me about the repo.', 'conversation_id': conversation_id})
    assert chat.status_code == 200
    payload = chat.json()
    assert 'answer' in payload
    assert payload['conversation_id'] == conversation_id
    assert payload['title'] == 'Demo chat'

    export_response = client.get(f'/api/conversations/{conversation_id}/export')
    assert export_response.status_code == 200

    read_response = client.get(f'/api/conversations/{conversation_id}')
    assert read_response.status_code == 200
    assert len(read_response.json()['messages']) >= 2


def test_delete_conversation_removes_database_rows_and_export(monkeypatch, tmp_path):
    database_path = tmp_path / "database" / "app.sqlite"
    exports_path = tmp_path / "exports"
    monkeypatch.setattr(main_module, "DB_PATH", database_path)
    monkeypatch.setattr(main_module, "EXPORTS_DIR", exports_path)
    main_module.init_db()
    monkeypatch.setattr(provider, "generate", lambda messages: "Local test response")

    create_response = client.post("/api/conversations", json={"title": "Delete test"})
    assert create_response.status_code == 200
    conversation_id = create_response.json()["id"]
    chat_response = client.post(
        "/api/chat",
        json={"message": "Test message storage", "conversation_id": conversation_id},
    )
    assert chat_response.status_code == 200
    export_response = client.get(f"/api/conversations/{conversation_id}/export")
    assert export_response.status_code == 200
    export_path = exports_path / f"{conversation_id}.json"
    assert export_path.is_file()

    with main_module.get_db_connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0] == 2

    delete_response = client.delete(f"/api/conversations/{conversation_id}")
    assert delete_response.status_code == 200
    with main_module.get_db_connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not export_path.exists()
    assert client.delete(f"/api/conversations/{conversation_id}").status_code == 404


def test_new_conversation_gets_title_from_first_message(monkeypatch):
    monkeypatch.setattr(provider, "generate", lambda messages: "Local test response")
    response = client.post('/api/chat', json={'message': 'Explain linear regression'})
    assert response.status_code == 200
    payload = response.json()
    assert payload['title'] == 'Explain linear regression'
    conversations = client.get('/api/conversations').json()
    matching = next(item for item in conversations if item['id'] == payload['conversation_id'])
    assert matching['title'] == 'Explain linear regression'


def test_project_indexing():
    project_root = str(Path(__file__).resolve().parent.parent)
    response = client.post('/api/projects', json={'name': 'project-under-test', 'root_path': project_root})
    assert response.status_code == 200
    payload = response.json()
    project_id = payload['id']

    index_response = client.post(f'/api/projects/{project_id}/index')
    assert index_response.status_code == 200
    status_payload = index_response.json()
    assert status_payload['status'] == 'indexed'

    delete_response = client.delete(f'/api/projects/{project_id}')
    assert delete_response.status_code == 200
    assert delete_response.json()['status'] == 'deleted'
