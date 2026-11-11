import json
import os
import secrets
import subprocess
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import psutil

RESOURCE_MONITOR_PORT = 8766


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class GPUUsageSampler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._process_lock = threading.Lock()
        self._accelerator_process: Optional[subprocess.Popen[str]] = None
        self._threads: list[threading.Thread] = []
        self._readings: Dict[str, Dict[str, Any]] = {
            "cpu": {"usage_percent": None, "sampled_at": None},
            "ram": {"usage_percent": None, "used_gb": None, "total_gb": None, "sampled_at": None},
            "disk": {
                "usage_percent": None,
                "storage_percent": None,
                "used_gb": None,
                "free_gb": None,
                "total_gb": None,
                "storage_sampled_at": None,
                "telemetry_error": "Waiting for the first disk activity sample.",
                "sampled_at": None,
            },
            "gpu": {
                "usage_percent": None,
                "telemetry_error": "Waiting for the first GPU usage sample.",
                "sampled_at": None,
            },
        }

    def start(self) -> None:
        if any(thread.is_alive() for thread in self._threads):
            return
        self._stop_event.clear()
        self._threads = [
            threading.Thread(target=self._sample_cpu_loop, name="system-usage-sampler", daemon=True),
            threading.Thread(
                target=self._sample_windows_accelerators,
                name="accelerator-usage-sampler",
                daemon=True,
            ),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._process_lock:
            process = self._accelerator_process
        if process and process.poll() is None:
            process.terminate()
        for thread in self._threads:
            thread.join(timeout=3)
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        self._threads = []

    def get_reading(self) -> Dict[str, Any]:
        with self._lock:
            return {key: value.copy() for key, value in self._readings.items()}

    def _sample_cpu_loop(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            usage = psutil.cpu_percent(interval=0.25)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
            sampled_at = utcnow()
            with self._lock:
                self._readings["cpu"] = {"usage_percent": usage, "sampled_at": sampled_at}
                self._readings["ram"] = {
                    "usage_percent": memory.percent,
                    "used_gb": round(memory.used / (1024 ** 3), 1),
                    "total_gb": round(memory.total / (1024 ** 3), 1),
                    "sampled_at": sampled_at,
                }
                self._readings["disk"].update({
                    "storage_percent": disk.percent,
                    "used_gb": round(disk.used / (1024 ** 3), 1),
                    "free_gb": round(disk.free / (1024 ** 3), 1),
                    "total_gb": round(disk.total / (1024 ** 3), 1),
                    "storage_sampled_at": sampled_at,
                })

    def _set_accelerator_error(self, message: str) -> None:
        sampled_at = utcnow()
        with self._lock:
            self._readings["gpu"] = {
                "usage_percent": None,
                "telemetry_error": message,
                "sampled_at": sampled_at,
            }
            self._readings["disk"].update({
                "usage_percent": None,
                "telemetry_error": "Windows physical-disk active-time counters are unavailable.",
                "sampled_at": sampled_at,
            })

    def _sample_windows_accelerators(self) -> None:
        if os.name != "nt":
            self._set_accelerator_error("Windows accelerator performance counters are unavailable.")
            return

        script = r"""
$ErrorActionPreference = 'Stop'
function Update-UsageCounters([string]$categoryName, [hashtable]$counterMap) {
    $category = [System.Diagnostics.PerformanceCounterCategory]::new($categoryName)
    $activeInstances = @{}
    foreach ($instanceName in $category.GetInstanceNames()) {
        $activeInstances[$instanceName] = $true
        if (-not $counterMap.ContainsKey($instanceName)) {
            $counter = [System.Diagnostics.PerformanceCounter]::new(
                $categoryName, 'Utilization Percentage', $instanceName
            )
            [void]$counter.NextValue()
            $counterMap[$instanceName] = $counter
        }
    }
    foreach ($instanceName in @($counterMap.Keys)) {
        if (-not $activeInstances.ContainsKey($instanceName)) {
            $counterMap[$instanceName].Dispose()
            [void]$counterMap.Remove($instanceName)
        }
    }
}
function Get-MaximumUsage([hashtable]$counterMap) {
    if ($counterMap.Count -eq 0) { return $null }
    $usageByEngine = @{}
    foreach ($instanceName in $counterMap.Keys) {
        $engineName = [regex]::Replace([string]$instanceName, '^pid_\d+_', '')
        $value = [double]$counterMap[$instanceName].NextValue()
        if (-not $usageByEngine.ContainsKey($engineName)) {
            $usageByEngine[$engineName] = [double]0
        }
        $usageByEngine[$engineName] += $value
    }
    $maximum = [double]0
    foreach ($value in $usageByEngine.Values) {
        if ($value -gt $maximum) { $maximum = $value }
    }
    $bounded = [Math]::Min([double]100, [Math]::Max([double]0, [double]$maximum))
    return [Math]::Round([double]$bounded, 1)
}
$gpuCounters = @{}
Update-UsageCounters 'GPU Engine' $gpuCounters
if ($gpuCounters.Count -eq 0) { throw 'No GPU Engine performance counters were found.' }
$diskCounter = $null
$diskError = $null
try {
    $diskCounter = [System.Diagnostics.PerformanceCounter]::new(
        'PhysicalDisk', '% Disk Time', '_Total'
    )
    [void]$diskCounter.NextValue()
} catch {
    $diskCounter = $null
    $diskError = 'Windows physical-disk active-time counters are unavailable.'
}
$lastGpuRefresh = [DateTime]::UtcNow
try {
    while ($true) {
        Start-Sleep -Milliseconds 250
        if (([DateTime]::UtcNow - $lastGpuRefresh).TotalSeconds -ge 0.5) {
            Update-UsageCounters 'GPU Engine' $gpuCounters
            $lastGpuRefresh = [DateTime]::UtcNow
        }
        $gpuUsage = Get-MaximumUsage -counterMap $gpuCounters
        $gpuError = if ($null -eq $gpuUsage) { 'GPU Engine performance counters are unavailable.' } else { $null }
        $diskUsage = $null
        if ($null -ne $diskCounter) {
            try {
                $diskUsage = [Math]::Min([double]100, [Math]::Max([double]0, [double]$diskCounter.NextValue()))
                $diskError = $null
            } catch {
                $diskCounter.Dispose()
                $diskCounter = $null
                $diskError = 'Windows physical-disk active-time counters stopped reporting.'
            }
        }
        [PSCustomObject]@{
            gpu = $gpuUsage
            gpu_error = $gpuError
            disk = $diskUsage
            disk_error = $diskError
        } |
            ConvertTo-Json -Compress |
            ForEach-Object { [Console]::Out.WriteLine($_); [Console]::Out.Flush() }
    }
} finally {
    foreach ($counter in $gpuCounters.Values) { $counter.Dispose() }
    if ($null -ne $diskCounter) { $diskCounter.Dispose() }
}
"""
        while not self._stop_event.is_set():
            try:
                process = subprocess.Popen(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as exc:
                self._set_accelerator_error(f"Could not start Windows accelerator sampling: {exc}")
                self._stop_event.wait(3)
                continue

            with self._process_lock:
                self._accelerator_process = process
            try:
                while not self._stop_event.is_set():
                    line = process.stdout.readline() if process.stdout else ""
                    if not line:
                        error = process.stderr.read().strip() if process.stderr else ""
                        if not self._stop_event.is_set():
                            self._set_accelerator_error(
                                error or "Windows accelerator performance counter sampler stopped."
                            )
                        break
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._set_accelerator_error(f"Invalid accelerator sample from Windows: {exc}")
                        continue

                    sampled_at = utcnow()
                    with self._lock:
                        self._readings["gpu"] = {
                            "usage_percent": sample.get("gpu"),
                            "telemetry_error": sample.get("gpu_error"),
                            "sampled_at": sampled_at,
                        }
                        self._readings["disk"].update({
                            "usage_percent": sample.get("disk"),
                            "telemetry_error": sample.get("disk_error"),
                            "sampled_at": sampled_at,
                        })
            finally:
                if process.poll() is None:
                    process.terminate()
                with self._process_lock:
                    if self._accelerator_process is process:
                        self._accelerator_process = None
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
                process.wait()

            if not self._stop_event.is_set():
                self._stop_event.wait(1)


def run_resource_monitor_server(port: int, shutdown_token: str) -> None:
    sampler = GPUUsageSampler()
    sampler.start()

    class ResourceHandler(BaseHTTPRequestHandler):
        def _write_headers(self, status_code: int, content_length: int) -> None:
            self.send_response(status_code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(content_length))
            self.end_headers()

        def do_OPTIONS(self) -> None:
            self._write_headers(204, 0)

        def do_GET(self) -> None:
            if urlparse(self.path).path != "/api/system/usage":
                body = b'{"detail":"Not found"}'
                self._write_headers(404, len(body))
                self.wfile.write(body)
                return
            body = json.dumps(sampler.get_reading()).encode("utf-8")
            self._write_headers(200, len(body))
            self.wfile.write(body)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/shutdown":
                body = b'{"detail":"Not found"}'
                self._write_headers(404, len(body))
                self.wfile.write(body)
                return
            supplied_token = self.headers.get("X-Resource-Monitor-Token", "")
            if not secrets.compare_digest(supplied_token, shutdown_token):
                body = b'{"detail":"Forbidden"}'
                self._write_headers(403, len(body))
                self.wfile.write(body)
                return
            self._write_headers(200, 2)
            self.wfile.write(b"{}")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), ResourceHandler)
    server.daemon_threads = True
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        sampler.stop()
