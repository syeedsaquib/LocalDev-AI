import gc
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from resource_monitor import GPUUsageSampler, RESOURCE_MONITOR_PORT, run_resource_monitor_server

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "local_data"
DB_PATH = DATA_DIR / "database" / "app.sqlite"
VECTOR_DIR = DATA_DIR / "vector_store"
PROJECT_METADATA_DIR = DATA_DIR / "project_metadata"
EXPORTS_DIR = DATA_DIR / "exports"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
    ".cache",
    "dist",
    "build",
    "site-packages",
}

SAFE_TEXT_EXTENSIONS = {
    ".py",
    ".pyi",
    ".js",
    ".jsx",
    ".mjs",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".sql",
    ".sh",
    ".ps1",
    ".toml",
    ".ini",
    ".xml",
    ".txt",
    ".rst",
    ".dockerfile",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_conversation_title(message: str) -> str:
    title = " ".join(message.split())
    if len(title) > 48:
        title = title[:45].rstrip() + "..."
    return title or "New chat"


def is_ready_hybrid_model(path: Path) -> bool:
    required_files = (
        "genai_config.json",
        "model_jit.onnx",
        "model_jit.onnx.data",
        "model_jit.bin",
        "model_jit.pb.bin",
        "tokenizer.json",
    )
    return all((path / name).is_file() for name in required_files)


def get_model_execution_mode(path: Path) -> Optional[str]:
    config_path = path / "genai_config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    provider_options = (
        config.get("model", {})
        .get("decoder", {})
        .get("session_options", {})
        .get("provider_options", [])
    )
    if any("RyzenAI" in options for options in provider_options):
        return "hybrid_npu_igpu"
    if any("dml" in options for options in provider_options):
        return "dml_gpu"
    return None


def get_local_model_settings() -> Dict[str, Any]:
    load_dotenv(APP_ROOT / ".env")
    models_dir = APP_ROOT.parent / "models"
    hybrid_model_path = models_dir / "Qwen2.5-Coder-7B-Instruct-RyzenAI-Hybrid"
    onnx_model_path = models_dir / "Qwen2.5-Coder-7B-Instruct-ONNX-DML"
    default_model_path = (
        hybrid_model_path
        if is_ready_hybrid_model(hybrid_model_path)
        else onnx_model_path
        if (onnx_model_path / "genai_config.json").is_file()
        else models_dir / "Qwen2.5-Coder-7B-Instruct"
    )
    return {
        "provider": os.getenv("LOCALDEV_MODEL_PROVIDER", "onnxruntime_genai"),
        "model_name": os.getenv("LOCALDEV_MODEL_NAME", "Qwen2.5-Coder-7B-Instruct"),
        "model_path": os.getenv("LOCALDEV_MODEL_PATH") or os.getenv("MODEL_PATH") or str(default_model_path),
        "execution_mode": os.getenv("LOCALDEV_EXECUTION_MODE", "auto"),
        "temperature": float(os.getenv("LOCALDEV_TEMPERATURE", "0.2")),
        "top_p": float(os.getenv("LOCALDEV_TOP_P", "0.9")),
        "max_new_tokens": int(os.getenv("LOCALDEV_MAX_NEW_TOKENS", "2048")),
        "streaming": os.getenv("LOCALDEV_STREAMING", "true").lower() == "true",
        "context_window": os.getenv("LOCALDEV_CONTEXT_WINDOW", "auto"),
    }


def ensure_directories() -> None:
    for path in [DATA_DIR, DB_PATH.parent, VECTOR_DIR, PROJECT_METADATA_DIR, EXPORTS_DIR, CACHE_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    ensure_directories()
    with get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'idle',
                indexed_file_count INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                last_indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS indexed_files (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                path TEXT NOT NULL,
                language TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_indexed_files_project_id ON indexed_files(project_id);

            CREATE TABLE IF NOT EXISTS project_memories (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


class LLMProvider:
    def load_model(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"status": "loaded", "provider": "base"}

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return ""

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs: Any):
        yield ""

    def unload_model(self) -> None:
        pass

    def health_check(self) -> bool:
        return True

    def get_metrics(self) -> Dict[str, Any]:
        return {"status": "idle", "execution_mode": "cpu_fallback"}


class FallbackLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self.config: Dict[str, Any] = {
            "provider": "fallback",
            "model_name": "local-fallback",
            "execution_mode": "cpu_fallback",
            "temperature": 0.2,
            "max_new_tokens": 512,
        }

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        user_text = messages[-1].get("content", "") if messages else ""
        if not user_text:
            return "No prompt received."
        base = user_text.strip()
        if len(base) > 220:
            base = base[:220] + "..."
        return (
            "Local fallback response: I cannot reach the AMD ONNX Runtime GenAI model in this environment, "
            f"so I am responding using a local safe fallback. The last user request was: {base}"
        )

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs: Any):
        response = self.generate(messages, **kwargs)
        for chunk in re.findall(r".{1,80}", response):
            yield chunk

    def get_metrics(self) -> Dict[str, Any]:
        return {"status": "fallback_active", "execution_mode": "cpu_fallback", "loaded_model": "local-fallback"}


class LocalTransformersProvider(LLMProvider):
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.genai = None
        self.genai_model = None
        self.genai_tokenizer = None
        self.config: Dict[str, Any] = get_local_model_settings()
        self._loaded = False
        self._available = False
        self._error: Optional[str] = None
        self._load_seconds: Optional[float] = None
        self._generation_seconds: Optional[float] = None
        self._generated_tokens = 0
        self._runtime_device_type: Optional[str] = None

    def _resolve_model_path(self) -> Optional[Path]:
        model_path = self.config.get("model_path")
        if not model_path:
            return None
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = (APP_ROOT / path).resolve()
        return path if path.exists() else None

    def load_model(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.config.update(config or {})
        self.config.update(get_local_model_settings())
        model_path = self._resolve_model_path()

        if model_path is None:
            self._available = False
            self._loaded = False
            self._error = "Local model directory not found. Set LOCALDEV_MODEL_PATH to the downloaded Qwen directory."
            return {
                "status": "not_found",
                "provider": "transformers_local",
                "reason": self._error,
                "model_name": self.config.get("model_name"),
                "model_path": self.config.get("model_path"),
            }

        self.config["model_path"] = str(model_path)
        started = datetime.now().timestamp()
        if (model_path / "genai_config.json").is_file():
            try:
                configured_mode = get_model_execution_mode(model_path)
                uses_ryzen_ai = configured_mode == "hybrid_npu_igpu"
                if uses_ryzen_ai and not is_ready_hybrid_model(model_path):
                    raise FileNotFoundError(
                        "AMD hybrid model package is incomplete. Required ONNX and Ryzen AI data files are missing."
                    )
                import onnxruntime_genai as og

                if uses_ryzen_ai and not og.is_qnn_available():
                    raise RuntimeError("The ONNX Runtime GenAI Ryzen AI provider is unavailable.")
                if not uses_ryzen_ai and not og.is_dml_available():
                    raise RuntimeError("ONNX Runtime GenAI DirectML provider is unavailable.")
                genai_model = og.Model(str(model_path))
                device_type = str(genai_model.device_type)
                if uses_ryzen_ai and "ryzenai" not in device_type.lower():
                    raise RuntimeError(
                        f"Expected Ryzen AI hybrid execution, but ONNX Runtime selected {device_type}."
                    )
                if not uses_ryzen_ai and "directml" not in device_type.lower():
                    raise RuntimeError(
                        f"Expected DirectML GPU execution, but ONNX Runtime selected {device_type}."
                    )
                genai_tokenizer = og.Tokenizer(genai_model)
                self.genai = og
                self.genai_model = genai_model
                self.genai_tokenizer = genai_tokenizer
                self._available = True
                self._loaded = True
                self._error = None
                self._load_seconds = datetime.now().timestamp() - started
                self._runtime_device_type = device_type
                self.config["execution_mode"] = (
                    "hybrid_npu_igpu" if uses_ryzen_ai else "dml_gpu"
                )
                return {
                    "status": "loaded",
                    "provider": "onnxruntime_genai",
                    "model_name": self.config.get("model_name"),
                    "execution_mode": self.config["execution_mode"],
                    "backend": "onnxruntime_genai",
                    "runtime_device_type": device_type,
                    "model_path": str(model_path),
                    "load_seconds": self._load_seconds,
                }
            except Exception as exc:
                self._available = False
                self._loaded = False
                self._error = f"Could not load the ONNX DirectML model: {exc}"
                return {
                    "status": "load_error",
                    "provider": "onnxruntime_genai",
                    "reason": self._error,
                    "model_name": self.config.get("model_name"),
                    "model_path": str(model_path),
                }

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not (model_path / "config.json").is_file() or not (model_path / "tokenizer.json").is_file():
                raise FileNotFoundError("Expected config.json and tokenizer.json in the local model directory.")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            self.tokenizer = tokenizer
            self.model = model
            self._available = True
            self._loaded = True
            self._error = None
            self._load_seconds = datetime.now().timestamp() - started
            self.config["execution_mode"] = "cuda" if device.type == "cuda" else "cpu"
            return {
                "status": "loaded",
                "provider": "transformers_local",
                "model_name": self.config.get("model_name"),
                "execution_mode": self.config["execution_mode"],
                "backend": "transformers",
                "model_path": str(model_path),
                "load_seconds": self._load_seconds,
            }
        except Exception as exc:  # pragma: no cover - environment-dependent
            self._available = False
            self._loaded = False
            self._error = str(exc)
            return {
                "status": "load_error",
                "provider": "transformers_local",
                "reason": str(exc),
                "model_name": self.config.get("model_name"),
                "model_path": str(model_path),
            }

    def generate(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        if not self._loaded:
            result = self.load_model(self.config)
            if result["status"] != "loaded":
                raise RuntimeError(result.get("reason", "Local model failed to load."))

        if self.genai_model is not None and self.genai_tokenizer is not None:
            prompt_parts = []
            if not messages or messages[0].get("role") != "system":
                prompt_parts.append(
                    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. "
                    "You are a helpful assistant.<|im_end|>\n"
                )
            for message in messages:
                role = message.get("role", "user")
                if role not in {"system", "user", "assistant"}:
                    continue
                prompt_parts.append(
                    f"<|im_start|>{role}\n{message.get('content', '')}<|im_end|>\n"
                )
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "".join(prompt_parts)
            input_tokens = self.genai_tokenizer.encode(prompt)
            params = self.genai.GeneratorParams(self.genai_model)
            temperature = float(kwargs.get("temperature", self.config.get("temperature", 0.2)))
            search_options: Dict[str, Any] = {
                "max_length": len(input_tokens)
                + int(kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 512))),
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                search_options["temperature"] = temperature
                search_options["top_p"] = float(kwargs.get("top_p", self.config.get("top_p", 0.9)))
            params.set_search_options(**search_options)
            generator = self.genai.Generator(self.genai_model, params)
            generator.append_tokens(input_tokens)
            started = datetime.now().timestamp()
            while not generator.is_done():
                generator.generate_next_token()
            self._generation_seconds = datetime.now().timestamp() - started
            sequence = generator.get_sequence(0)
            self._generated_tokens = max(0, len(sequence) - len(input_tokens))
            response = self.genai_tokenizer.decode(sequence[len(input_tokens):]).strip()
            if not response:
                raise RuntimeError("The DirectML model returned an empty response.")
            return response

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("The local model is not loaded.")

        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        temperature = float(kwargs.get("temperature", self.config.get("temperature", 0.2)))
        generation_options: Dict[str, Any] = {
            "max_new_tokens": int(kwargs.get("max_new_tokens", self.config.get("max_new_tokens", 512))),
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature
            generation_options["top_p"] = float(kwargs.get("top_p", self.config.get("top_p", 0.9)))

        started = datetime.now().timestamp()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_options)
        self._generation_seconds = datetime.now().timestamp() - started
        self._generated_tokens = int(output.shape[1] - input_length)
        response = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
        if not response:
            raise RuntimeError("The local model returned an empty response.")
        return response

    def stream_generate(self, messages: List[Dict[str, str]], **kwargs: Any):
        yield self.generate(messages, **kwargs)

    def health_check(self) -> bool:
        return self._loaded

    def get_metrics(self) -> Dict[str, Any]:
        model_path = self._resolve_model_path()
        configured_mode = get_model_execution_mode(model_path) if model_path else None
        execution_mode = (
            self.config.get("execution_mode")
            if self._loaded
            else configured_mode or self.config.get("execution_mode", "auto")
        )
        return {
            "status": "loaded" if self._loaded else "available" if self._resolve_model_path() else "not_found",
            "execution_mode": execution_mode,
            "execution_verified": self._loaded and self._runtime_device_type is not None,
            "loaded_model": self.config.get("model_name"),
            "model_path": self.config.get("model_path"),
            "backend": "onnxruntime_genai" if configured_mode else "transformers_local",
            "runtime_device_type": self._runtime_device_type,
            "accelerator_policy": (
                "AMD Ryzen AI hybrid: NPU prefill and iGPU decode"
                if execution_mode == "hybrid_npu_igpu"
                else "DirectML GPU"
                if execution_mode == "dml_gpu"
                else "Transformers selected device"
            ),
            "last_error": self._error,
            "load_seconds": self._load_seconds,
            "generation_seconds": self._generation_seconds,
            "generated_tokens": self._generated_tokens,
            "tokens_per_second": (
                round(self._generated_tokens / self._generation_seconds, 2)
                if self._generation_seconds
                else None
            ),
        }

    def unload_model(self) -> None:
        self.model = None
        self.tokenizer = None
        self.genai_model = None
        self.genai_tokenizer = None
        self.genai = None
        self._runtime_device_type = None
        self._loaded = False
        self._available = False
        gc.collect()


class ProjectManager:
    @staticmethod
    def create_project(name: str, root_path: str) -> Dict[str, Any]:
        project_id = str(uuid.uuid4())
        now = utcnow()
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO projects (id, name, root_path, created_at, updated_at, status) VALUES (?, ?, ?, ?, ?, 'idle')",
                (project_id, name, root_path, now, now),
            )
        return {"id": project_id, "name": name, "root_path": root_path, "status": "idle"}

    @staticmethod
    def list_projects() -> List[Dict[str, Any]]:
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM projects ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def get_project(project_id: str) -> Optional[Dict[str, Any]]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            if row is None:
                return None
            return dict(row)

    @staticmethod
    def delete_project(project_id: str) -> None:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    @staticmethod
    def scan_project(project_id: str, root_path: str) -> List[Dict[str, Any]]:
        root = Path(root_path).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise HTTPException(status_code=400, detail=f"Project path does not exist: {root_path}")

        records: List[Dict[str, Any]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in file_path.parts):
                continue
            if file_path.name.startswith(".") and file_path.name not in {".gitignore", ".editorconfig", ".env.example"}:
                continue
            suffix = file_path.suffix.lower()
            if suffix and suffix not in SAFE_TEXT_EXTENSIONS:
                continue
            if file_path.name.endswith(".env") or "secret" in file_path.name.lower():
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            short = file_path.relative_to(root)
            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "project_id": project_id,
                    "path": str(short).replace("\\", "/"),
                    "language": suffix.lstrip(".") or "text",
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "size": len(content.encode("utf-8")),
                    "content": content[:20000],
                    "updated_at": utcnow(),
                }
            )
        return records

    @staticmethod
    def index_project(project_id: str) -> Dict[str, Any]:
        project = ProjectManager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        root_path = project["root_path"]
        with get_db_connection() as conn:
            conn.execute("DELETE FROM indexed_files WHERE project_id = ?", (project_id,))
            conn.execute("UPDATE projects SET status = 'indexing', updated_at = ? WHERE id = ?", (utcnow(), project_id))

        records = ProjectManager.scan_project(project_id, root_path)
        with get_db_connection() as conn:
            for rec in records:
                conn.execute(
                    "INSERT INTO indexed_files (id, project_id, path, language, sha256, size, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec["id"],
                        rec["project_id"],
                        rec["path"],
                        rec["language"],
                        rec["sha256"],
                        rec["size"],
                        rec["updated_at"],
                    ),
                )
            conn.execute(
                "UPDATE projects SET status = 'indexed', indexed_file_count = ?, total_chunks = ?, last_indexed_at = ?, updated_at = ? WHERE id = ?",
                (len(records), len(records), utcnow(), utcnow(), project_id),
            )

        metadata_path = PROJECT_METADATA_DIR / f"{project_id}.json"
        metadata_path.write_text(json.dumps({"id": project_id, "name": project["name"], "root_path": root_path, "file_count": len(records), "updated_at": utcnow()}, indent=2), encoding="utf-8")
        return {"project_id": project_id, "indexed_file_count": len(records), "total_chunks": len(records), "status": "indexed"}

    @staticmethod
    def get_project_status(project_id: str) -> Dict[str, Any]:
        project = ProjectManager.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        return {
            "project_id": project_id,
            "status": project["status"],
            "indexed_file_count": project["indexed_file_count"],
            "total_chunks": project["total_chunks"],
            "last_indexed_at": project["last_indexed_at"],
        }


def retrieve_project_context(project_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT path, language, sha256, size, updated_at FROM indexed_files WHERE project_id = ? ORDER BY path ASC",
            (project_id,),
        ).fetchall()

    if not rows:
        return []

    tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    scored: List[Dict[str, Any]] = []
    for row in rows:
        path = row["path"]
        file_path = (Path(ProjectManager.get_project(project_id)["root_path"]) / path).resolve()
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = text.lower()
        score = 0
        for token in tokens:
            if token in lower:
                score += 2
            if token in path.lower():
                score += 5
        if not tokens or score <= 0:
            score = 1
        scored.append({"path": path, "score": score, "snippet": text[:400], "language": row["language"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    project_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class ConversationCreateRequest(BaseModel):
    project_id: Optional[str] = None
    title: str = "New conversation"


class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1)


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    root_path: str = Field(..., min_length=1)


class DeleteRequest(BaseModel):
    filename: str | None = None


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    start_resource_monitor()
    try:
        yield
    finally:
        stop_resource_monitor()
        provider.unload_model()


app = FastAPI(title="LocalDev AI", lifespan=app_lifespan)
app.mount("/assets", StaticFiles(directory=str(APP_ROOT / "assets")), name="assets")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
provider = LocalTransformersProvider()
gpu_usage_sampler = GPUUsageSampler()
resource_monitor_process: Optional[Process] = None
resource_monitor_shutdown_token: Optional[str] = None


def resource_monitor_url(path: str) -> str:
    return f"http://127.0.0.1:{RESOURCE_MONITOR_PORT}{path}"


def start_resource_monitor() -> None:
    global resource_monitor_process, resource_monitor_shutdown_token
    shutdown_token = secrets.token_urlsafe(32)
    process = Process(
        target=run_resource_monitor_server,
        args=(RESOURCE_MONITOR_PORT, shutdown_token),
        name="localdev-resource-monitor",
        daemon=True,
    )
    process.start()
    resource_monitor_process = process
    resource_monitor_shutdown_token = shutdown_token
    deadline = time.monotonic() + 5
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        if not process.is_alive():
            stop_resource_monitor()
            raise RuntimeError("The separate resource monitor process exited during startup.")
        try:
            with urlopen(resource_monitor_url("/api/system/usage"), timeout=0.5) as response:
                payload = json.loads(response.read())
            if not all(key in payload for key in ("cpu", "gpu", "disk", "ram")):
                stop_resource_monitor()
                raise RuntimeError("The resource monitor returned an invalid startup response.")
            return
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            time.sleep(0.1)
    stop_resource_monitor()
    raise RuntimeError(f"The separate resource monitor did not become ready: {last_error}")


def stop_resource_monitor() -> None:
    global resource_monitor_process, resource_monitor_shutdown_token
    process = resource_monitor_process
    shutdown_token = resource_monitor_shutdown_token
    resource_monitor_process = None
    resource_monitor_shutdown_token = None
    if process is None:
        return
    if process.is_alive():
        try:
            request = Request(
                resource_monitor_url("/shutdown"),
                data=b"{}",
                headers={"X-Resource-Monitor-Token": shutdown_token or ""},
                method="POST",
            )
            with urlopen(request, timeout=1):
                pass
        except (URLError, TimeoutError):
            pass
        process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)


def get_resource_readings() -> Dict[str, Dict[str, Any]]:
    process = resource_monitor_process
    if process is None:
        return gpu_usage_sampler.get_reading()
    if not process.is_alive():
        raise HTTPException(status_code=503, detail="The separate resource monitor process is not running.")
    try:
        with urlopen(resource_monitor_url("/api/system/usage"), timeout=1) as response:
            readings = json.loads(response.read())
    except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Could not read system usage: {exc}") from exc
    return readings


@app.get("/")
async def read_root() -> FileResponse:
    return FileResponse(str(APP_ROOT / "index.html"))


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "provider": provider.get_metrics(),
        "database": str(DB_PATH),
        "projects_dir": str(PROJECT_METADATA_DIR),
    }


@app.get("/api/conversations")
async def list_conversations() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        conversations = []
        for row in rows:
            conversation = dict(row)
            if conversation["title"] == "New conversation":
                first_message = conn.execute(
                    "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' ORDER BY created_at ASC LIMIT 1",
                    (conversation["id"],),
                ).fetchone()
                if first_message:
                    conversation["title"] = make_conversation_title(first_message["content"])
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (conversation["title"], conversation["id"]),
                    )
            conversations.append(conversation)
    return conversations


@app.post("/api/conversations")
async def create_conversation(request: ConversationCreateRequest) -> Dict[str, Any]:
    conversation_id = str(uuid.uuid4())
    now = utcnow()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, project_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, request.project_id, request.title or "New conversation", now, now),
        )
    return {"id": conversation_id, "title": request.title or "New conversation", "project_id": request.project_id}


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        conversation = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
    return {"conversation": dict(conversation), "messages": [dict(row) for row in rows]}


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: ConversationRenameRequest) -> Dict[str, Any]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (request.title, utcnow(), conversation_id))
    return {"status": "updated", "conversation_id": conversation_id, "title": request.title}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> Dict[str, Any]:
    with get_db_connection() as conn:
        conversation = conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    export_path = EXPORTS_DIR / f"{conversation_id}.json"
    if export_path.is_file():
        export_path.unlink()
    return {"status": "deleted", "conversation_id": conversation_id}


@app.delete("/api/conversations")
async def delete_all_conversations() -> Dict[str, Any]:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM conversations")
        conn.execute("DELETE FROM messages")
    return {"status": "all_deleted"}


@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str) -> Dict[str, Any]:
    convo = await get_conversation(conversation_id)
    export_path = EXPORTS_DIR / f"{conversation_id}.json"
    export_path.write_text(json.dumps(convo, indent=2), encoding="utf-8")
    return {"status": "exported", "path": str(export_path)}


@app.get("/api/projects")
async def list_projects() -> List[Dict[str, Any]]:
    return ProjectManager.list_projects()


@app.post("/api/projects")
async def create_project(request: ProjectCreateRequest) -> Dict[str, Any]:
    project = ProjectManager.create_project(request.name, request.root_path)
    return project


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> Dict[str, Any]:
    project = ProjectManager.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str) -> Dict[str, Any]:
    ProjectManager.delete_project(project_id)
    return {"status": "deleted", "project_id": project_id}


@app.post("/api/projects/{project_id}/index")
async def index_project(project_id: str) -> Dict[str, Any]:
    return ProjectManager.index_project(project_id)


@app.get("/api/projects/{project_id}/status")
async def project_status(project_id: str) -> Dict[str, Any]:
    return ProjectManager.get_project_status(project_id)


@app.post("/api/rag/search")
async def rag_search(request: dict) -> Dict[str, Any]:
    query = request.get("query") or ""
    project_id = request.get("project_id")
    if not project_id:
        return {"results": []}
    results = retrieve_project_context(project_id, query)
    return {"results": results}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, project_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, request.project_id, make_conversation_title(message), utcnow(), utcnow()),
            )

    context_items: List[Dict[str, Any]] = []
    if request.project_id:
        context_items = retrieve_project_context(request.project_id, message)

    system_prompt = "You are LocalDev AI, a local-first coding assistant. Answer using the project context if available and cite file paths when they are relevant."
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in (request.history or [])
    )
    context_block = "\n\n".join(
        f"File: {item['path']}\nSnippet: {item['snippet']}" for item in context_items
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": history_text or message},
    ]
    if context_block:
        messages.append({"role": "user", "content": f"Relevant project context:\n{context_block}"})
    messages.append({"role": "user", "content": message})

    try:
        reply = (await run_in_threadpool(provider.generate, messages)).strip()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Local model generation failed: {exc}") from exc
    if not reply:
        raise HTTPException(status_code=503, detail="The local model returned an empty response.")

    with get_db_connection() as conn:
        conversation = conn.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        conversation_title = (
            make_conversation_title(message)
            if conversation is None or conversation["title"] == "New conversation"
            else conversation["title"]
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, metadata) VALUES (?, ?, 'user', ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, message, utcnow(), json.dumps({"project_id": request.project_id})),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at, metadata) VALUES (?, ?, 'assistant', ?, ?, ?)",
            (str(uuid.uuid4()), conversation_id, reply, utcnow(), json.dumps({"citations": context_items})),
        )
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (conversation_title, utcnow(), conversation_id),
        )

    return {
        "conversation_id": conversation_id,
        "title": conversation_title,
        "answer": reply,
        "citations": context_items,
        "provider": provider.get_metrics(),
    }


@app.post("/api/chat/stream")
async def stream_chat(request: ChatRequest):
    def event_generator():
        for chunk in provider.stream_generate([
            {"role": "user", "content": request.message},
        ]):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        iterate_in_threadpool(event_generator()),
        media_type="text/event-stream",
    )


@app.get("/api/system/usage")
async def system_usage() -> Dict[str, Any]:
    readings = await run_in_threadpool(get_resource_readings)
    return {
        "cpu": readings["cpu"],
        "gpu": readings["gpu"],
        "disk": readings["disk"],
        "ram": readings["ram"],
    }


@app.get("/api/system/status")
async def system_status() -> Dict[str, Any]:
    model_metrics = provider.get_metrics()
    execution_mode = model_metrics.get("execution_mode")
    model_loaded = model_metrics.get("status") == "loaded"
    gpu_selected = execution_mode in {"hybrid_npu_igpu", "dml_gpu"}
    readings = await run_in_threadpool(get_resource_readings)
    gpu_reading = readings["gpu"]
    return {
        "cpu": {**readings["cpu"], "status": "available"},
        "gpu": {
            "usage_percent": gpu_reading["usage_percent"],
            "status": "ready" if model_loaded and gpu_selected else "not_selected",
            "telemetry_available": gpu_reading["usage_percent"] is not None,
            "telemetry_error": gpu_reading["telemetry_error"],
            "sampled_at": gpu_reading["sampled_at"],
            "measurement": "busiest_engine",
        },
        "disk": {**readings["disk"], "status": "available"},
        "ram": {**readings["ram"], "status": "available"},
        "model": model_metrics,
    }


@app.get("/api/system/model")
async def system_model() -> Dict[str, Any]:
    return provider.get_metrics()


@app.post("/api/system/model/load")
async def load_system_model() -> Dict[str, Any]:
    result = provider.load_model()
    if result["status"] != "loaded":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/api/system/model/unload")
async def unload_system_model() -> Dict[str, Any]:
    provider.unload_model()
    return provider.get_metrics()


@app.get("/api/system/metrics")
async def system_metrics() -> Dict[str, Any]:
    return provider.get_metrics()


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    file_path = DATA_DIR / "uploads" / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return {"filename": file.filename, "size": file_path.stat().st_size, "status": "uploaded"}


@app.get("/api/history")
async def get_history() -> List[Dict[str, Any]]:
    return await list_conversations()


@app.delete("/api/history")
async def delete_history() -> Dict[str, Any]:
    return await delete_all_conversations()


@app.get("/api/history/{conversation_id}")
async def get_history_entry(conversation_id: str) -> Dict[str, Any]:
    return await get_conversation(conversation_id)


@app.delete("/api/history/{conversation_id}")
async def delete_history_entry(conversation_id: str) -> Dict[str, Any]:
    return await delete_conversation(conversation_id)


@app.post("/api/agent/plan")
async def agent_plan(request: dict) -> Dict[str, Any]:
    return {
        "plan": [
            "Inspect the project structure.",
            "Locate likely implementation points.",
            "Create a focused patch and validate it.",
        ],
        "request": request,
    }


@app.post("/api/agent/execute")
async def agent_execute(request: dict) -> Dict[str, Any]:
    return {"status": "approved", "request": request}


@app.get("/api/rag/statistics")
async def rag_statistics() -> Dict[str, Any]:
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
    return {"indexed_files": total, "vector_dir": str(VECTOR_DIR)}


@app.post("/api/projects/{project_id}/reindex")
async def reindex_project(project_id: str) -> Dict[str, Any]:
    return ProjectManager.index_project(project_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)