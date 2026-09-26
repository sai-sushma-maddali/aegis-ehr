"""Session telemetry for the HP ZGX Nano dashboard.

One tracker records each screened request. Latency, token counts, and GPU
memory come from the call that just finished. Dollar rates are the local
savings formula for tiers that never call the cloud.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

# GB10 unified memory specification used by the dashboard formula.
VRAM_TOTAL_GB = 128.0

# Avoided cloud token price for on-device routes.
INPUT_USD_PER_TOKEN = 0.0000015
OUTPUT_USD_PER_TOKEN = 0.000002

ROUTE_VECTOR = "EDGE_VECTOR_CACHE"
ROUTE_EDGE = "EDGE_GUARD_8B"
ROUTE_CLOUD = "CLOUD_DEFENDER_70B"

_SOURCE_ROUTES = {
    "EDGE_VECTOR_CACHE": ROUTE_VECTOR,
    "EDGE_INTENT_TRIAGE": ROUTE_EDGE,
    "CLOUD_DEFENDER_HOTPATCH": ROUTE_CLOUD,
}


def map_route(source_layer: str | None) -> str:
    """Map an EdgeGuard source layer onto the dashboard route name."""
    return _SOURCE_ROUTES.get(str(source_layer or ""), ROUTE_EDGE)


def estimate_tokens(text: str | None) -> int:
    """Approximate tokens when a model response has no usage field."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def query_cost_saved_usd(
    route: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Savings versus the token price. Cloud routes save nothing."""
    if route == ROUTE_CLOUD:
        return 0.0
    saved = (max(input_tokens, 0) * INPUT_USD_PER_TOKEN) + (
        max(output_tokens, 0) * OUTPUT_USD_PER_TOKEN
    )
    return round(saved, 6)


def data_residency_pct(phi_tokens_egressed: int) -> float:
    """Full residency when no PHI tokens left the device."""
    if phi_tokens_egressed <= 0:
        return 100.0
    return 0.0


def tokens_per_sec(output_tokens: int, generation_ms: float | None) -> float | None:
    """Output tokens divided by generation time. No generation yields None."""
    if output_tokens <= 0 or generation_ms is None or generation_ms <= 0:
        return None
    return round(output_tokens / (generation_ms / 1000.0), 3)


def read_vram_used_gb() -> float | None:
    """
    Unified memory in use.

    This device does not report a separate GPU framebuffer, so the reading
    is system memory currently unavailable to new allocations. Torch is not
    initialized here; doing so can fail while other models already hold the GPU.
    """
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                fields[key] = int(rest.strip().split()[0])
        total_kb = fields["MemTotal"]
        available_kb = fields["MemAvailable"]
    except (OSError, KeyError, ValueError, IndexError):
        return None
    if total_kb <= 0:
        return None
    used_kb = max(total_kb - available_kb, 0)
    return round(used_kb / (1024.0 * 1024.0), 3)


class TelemetryTracker:
    """In-process totals for the life of the API process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cumulative_savings_usd = 0.0
        self.queries_processed = 0
        self._latest: dict[str, Any] | None = None

    def record(
        self,
        *,
        source_layer: str | None,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        generation_ms: float | None = None,
        phi_tokens_egressed: int = 0,
        cache_hit: bool = False,
        cache_hit_latency_ms: float | None = None,
        signature_count: int | None = None,
    ) -> dict[str, Any]:
        route = map_route(source_layer)
        saved = query_cost_saved_usd(route, input_tokens, output_tokens)
        with self._lock:
            self.queries_processed += 1
            self.cumulative_savings_usd = round(
                self.cumulative_savings_usd + saved, 6
            )
            snapshot = self._payload(
                latency_ms=latency_ms,
                route=route,
                saved=saved,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_ms=generation_ms,
                phi_tokens_egressed=max(int(phi_tokens_egressed), 0),
                cache_hit=cache_hit,
                cache_hit_latency_ms=cache_hit_latency_ms,
                signature_count=signature_count,
            )
            self._latest = snapshot
            return dict(snapshot)

    def clear(self) -> None:
        """Drop the latest request and session totals. Memory is read live."""
        with self._lock:
            self.cumulative_savings_usd = 0.0
            self.queries_processed = 0
            self._latest = None

    def snapshot(self, *, signature_count: int | None = None) -> dict[str, Any]:
        """Latest request, with a fresh memory reading."""
        with self._lock:
            latest = dict(self._latest) if self._latest else self._empty()
            cumulative = self.cumulative_savings_usd
            queries = self.queries_processed
        latest["cumulative_savings_usd"] = cumulative
        latest["queries_processed"] = queries
        if signature_count is not None:
            latest["active_hotpatches_count"] = signature_count
        used = read_vram_used_gb()
        latest["vram_used_gb"] = used
        latest["vram_total_gb"] = VRAM_TOTAL_GB
        latest["vram_percent"] = (
            round((used / VRAM_TOTAL_GB) * 100.0, 3) if used is not None else None
        )
        return latest

    def _payload(
        self,
        *,
        latency_ms: float,
        route: str,
        saved: float,
        input_tokens: int,
        output_tokens: int,
        generation_ms: float | None,
        phi_tokens_egressed: int,
        cache_hit: bool,
        cache_hit_latency_ms: float | None,
        signature_count: int | None,
    ) -> dict[str, Any]:
        latency = round(float(latency_ms), 3)
        used = read_vram_used_gb()
        return {
            "latency_ms": latency,
            # This API does not stream, so first-token time is the full request.
            "ttft_ms": latency,
            "tokens_per_sec": tokens_per_sec(output_tokens, generation_ms),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "vram_used_gb": used,
            "vram_total_gb": VRAM_TOTAL_GB,
            "vram_percent": (
                round((used / VRAM_TOTAL_GB) * 100.0, 3) if used is not None else None
            ),
            "active_route": route,
            "query_cost_saved_usd": saved,
            "cumulative_savings_usd": self.cumulative_savings_usd,
            "queries_processed": self.queries_processed,
            "phi_tokens_egressed": phi_tokens_egressed,
            "data_residency_pct": data_residency_pct(phi_tokens_egressed),
            "cache_hit_latency_ms": (
                round(float(cache_hit_latency_ms), 3)
                if cache_hit and cache_hit_latency_ms is not None
                else None
            ),
            "active_hotpatches_count": signature_count,
        }

    def _empty(self) -> dict[str, Any]:
        used = read_vram_used_gb()
        return {
            "latency_ms": None,
            "ttft_ms": None,
            "tokens_per_sec": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "vram_used_gb": used,
            "vram_total_gb": VRAM_TOTAL_GB,
            "vram_percent": (
                round((used / VRAM_TOTAL_GB) * 100.0, 3) if used is not None else None
            ),
            "active_route": None,
            "query_cost_saved_usd": 0.0,
            "cumulative_savings_usd": 0.0,
            "queries_processed": 0,
            "phi_tokens_egressed": 0,
            "data_residency_pct": 100.0,
            "cache_hit_latency_ms": None,
            "active_hotpatches_count": None,
        }


_TRACKER = TelemetryTracker()


def get_tracker() -> TelemetryTracker:
    return _TRACKER
