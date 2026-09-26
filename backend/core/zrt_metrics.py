"""Live HP ZGX readings and the local ZRT / vLLM session histogram."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path

ZRT_METRICS_SOCK = Path(
    "/opt/hp/zrt/run/vllm-hf-mistralai-mistral-7b-instruct-v0-3-main.sock"
)

_BUCKET = re.compile(
    r'^vllm:(time_to_first_token_seconds|e2e_request_latency_seconds)_bucket\{[^}]*\ble="([^"]+)"[^}]*\} ([0-9eE+.\-]+)'
)
_TOKEN_SUM = re.compile(
    r'^vllm:(request_prompt_tokens_sum|request_generation_tokens_sum)\{[^}]*\} ([0-9eE+.\-]+)'
)
_SUCCESS = re.compile(
    r'^vllm:request_success_total\{[^}]*finished_reason="(stop|length)"[^}]*\} ([0-9eE+.\-]+)'
)


def histogram_quantile(buckets: list[tuple[float, float]], quantile: float) -> float | None:
    """Interpolate a Prometheus cumulative histogram. ``le`` of inf is ignored."""
    finite = [(le, count) for le, count in buckets if le != float("inf")]
    if not finite:
        return None
    total = finite[-1][1]
    if total <= 0:
        return None
    target = total * quantile
    previous_le = 0.0
    previous_count = 0.0
    for le, count in finite:
        if count >= target:
            span = count - previous_count
            if span <= 0:
                return previous_le
            fraction = (target - previous_count) / span
            return previous_le + fraction * (le - previous_le)
        previous_le = le
        previous_count = count
    return finite[-1][0]


def _empty_session() -> dict[str, float | int | None]:
    return {
        "ttft_p50_ms": None,
        "ttft_p90_ms": None,
        "ttft_p99_ms": None,
        "prompt_tokens_total": None,
        "generated_tokens_total": None,
        "requests_succeeded": None,
        "e2e_latency_p50_s": None,
    }


def _raw_from_text(text: str) -> dict[str, object]:
    buckets: dict[str, list[tuple[float, float]]] = {
        "time_to_first_token_seconds": [],
        "e2e_request_latency_seconds": [],
    }
    prompt_tokens = 0.0
    generated_tokens = 0.0
    succeeded = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:"):
            continue
        bucket = _BUCKET.match(line)
        if bucket:
            le = float("inf") if bucket.group(2) == "+Inf" else float(bucket.group(2))
            buckets[bucket.group(1)].append((le, float(bucket.group(3))))
            continue
        tokens = _TOKEN_SUM.match(line)
        if tokens:
            if tokens.group(1) == "request_prompt_tokens_sum":
                prompt_tokens += float(tokens.group(2))
            else:
                generated_tokens += float(tokens.group(2))
            continue
        success = _SUCCESS.match(line)
        if success:
            succeeded += float(success.group(2))
    return {
        "buckets": buckets,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "succeeded": succeeded,
    }


def _subtract_raw(current: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    current_buckets = current["buckets"]
    baseline_buckets = baseline["buckets"]
    delta_buckets: dict[str, list[tuple[float, float]]] = {}
    for name in ("time_to_first_token_seconds", "e2e_request_latency_seconds"):
        base_counts = {le: count for le, count in baseline_buckets.get(name, [])}
        delta_buckets[name] = [
            (le, max(count - base_counts.get(le, 0.0), 0.0))
            for le, count in current_buckets.get(name, [])
        ]
    return {
        "buckets": delta_buckets,
        "prompt_tokens": max(float(current["prompt_tokens"]) - float(baseline["prompt_tokens"]), 0.0),
        "generated_tokens": max(
            float(current["generated_tokens"]) - float(baseline["generated_tokens"]),
            0.0,
        ),
        "succeeded": max(float(current["succeeded"]) - float(baseline["succeeded"]), 0.0),
    }


def _session_from_raw(raw: dict[str, object]) -> dict[str, float | int | None]:
    buckets = raw["buckets"]
    succeeded = float(raw["succeeded"])
    ttft_p50 = histogram_quantile(buckets["time_to_first_token_seconds"], 0.50)
    if succeeded <= 0 and ttft_p50 is None:
        return _empty_session()
    ttft_p90 = histogram_quantile(buckets["time_to_first_token_seconds"], 0.90)
    ttft_p99 = histogram_quantile(buckets["time_to_first_token_seconds"], 0.99)
    e2e_p50 = histogram_quantile(buckets["e2e_request_latency_seconds"], 0.50)
    if succeeded <= 0:
        return _empty_session()
    return {
        "ttft_p50_ms": None if ttft_p50 is None else round(ttft_p50 * 1000.0, 1),
        "ttft_p90_ms": None if ttft_p90 is None else round(ttft_p90 * 1000.0, 1),
        "ttft_p99_ms": None if ttft_p99 is None else round(ttft_p99 * 1000.0, 1),
        "prompt_tokens_total": int(raw["prompt_tokens"]),
        "generated_tokens_total": int(raw["generated_tokens"]),
        "requests_succeeded": int(succeeded),
        "e2e_latency_p50_s": None if e2e_p50 is None else round(e2e_p50, 3),
    }


def parse_vllm_metrics(text: str) -> dict[str, float | int | None]:
    return _session_from_raw(_raw_from_text(text))


_BASELINE_LOCK = threading.Lock()
_BASELINE: dict[str, object] | None = None


def mark_session_baseline() -> None:
    """Hide ZRT counters collected before Clear dashboard was pressed."""
    global _BASELINE
    raw = _raw_from_text(_read_metrics_text())
    with _BASELINE_LOCK:
        _BASELINE = raw


def _session_since_clear(text: str) -> dict[str, float | int | None]:
    raw = _raw_from_text(text)
    with _BASELINE_LOCK:
        baseline = _BASELINE
    if baseline is None:
        return _session_from_raw(raw)
    return _session_from_raw(_subtract_raw(raw, baseline))


def _system_memory() -> dict[str, float | None]:
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                fields[key] = int(rest.strip().split()[0])
        total_kb = fields["MemTotal"]
        available_kb = fields["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return {
            "system_memory_used_gb": None,
            "system_memory_total_gb": None,
            "memory_percent": None,
        }
    if total_kb <= 0:
        return {
            "system_memory_used_gb": None,
            "system_memory_total_gb": None,
            "memory_percent": None,
        }
    used_kb = max(total_kb - available_kb, 0)
    used_gb = used_kb / (1024.0 * 1024.0)
    total_gb = total_kb / (1024.0 * 1024.0)
    return {
        "system_memory_used_gb": round(used_gb, 1),
        "system_memory_total_gb": round(total_gb, 1),
        "memory_percent": round((used_gb / total_gb) * 100.0, 1),
    }


def _cpu_percent() -> float | None:
    def sample() -> tuple[int, int] | None:
        try:
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
                if line.startswith("cpu "):
                    parts = [int(part) for part in line.split()[1:]]
                    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                    return idle, sum(parts)
        except (OSError, ValueError, IndexError):
            return None
        return None

    first = sample()
    time.sleep(0.1)
    second = sample()
    if first is None or second is None:
        return None
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return round((1.0 - idle_delta / total_delta) * 100.0, 1)


def _gpu_sample() -> dict[str, float | None]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        util_text, _, temp_text = output.strip().splitlines()[0].partition(",")
        util = float(util_text.strip())
        temp = float(temp_text.strip())
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"system_gpu_percent": None, "package_temp_c": None}
    return {
        "system_gpu_percent": round(util, 1),
        "package_temp_c": round(temp, 1),
    }


def _read_metrics_text() -> str:
    if not ZRT_METRICS_SOCK.exists():
        return ""
    try:
        return subprocess.check_output(
            [
                "curl",
                "-sf",
                "--max-time",
                "2",
                "--unix-socket",
                str(ZRT_METRICS_SOCK),
                "http://localhost/metrics",
            ],
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def collect_engine_telemetry() -> dict[str, float | int | None]:
    """Memory, CPU, GPU, and the current Mistral ZRT session counters."""
    payload: dict[str, float | int | None] = {}
    payload.update(_system_memory())
    payload.update(_session_since_clear(_read_metrics_text()))
    payload["system_cpu_percent"] = _cpu_percent()
    payload.update(_gpu_sample())
    return payload
