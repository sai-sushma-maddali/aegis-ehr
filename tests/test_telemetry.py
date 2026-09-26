"""Telemetry math and the payload attached to a recorded request."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.telemetry_tracker import (  # noqa: E402
    ROUTE_CLOUD,
    ROUTE_EDGE,
    ROUTE_VECTOR,
    TelemetryTracker,
    data_residency_pct,
    map_route,
    query_cost_saved_usd,
    tokens_per_sec,
)


class TestTelemetryMath(unittest.TestCase):
    def test_route_names(self):
        self.assertEqual(map_route("EDGE_VECTOR_CACHE"), ROUTE_VECTOR)
        self.assertEqual(map_route("EDGE_INTENT_TRIAGE"), ROUTE_EDGE)
        self.assertEqual(map_route("CLOUD_DEFENDER_HOTPATCH"), ROUTE_CLOUD)

    def test_local_savings_and_cloud_zero(self):
        saved = query_cost_saved_usd(ROUTE_EDGE, 1000, 200)
        self.assertAlmostEqual(saved, 1000 * 0.0000015 + 200 * 0.000002)
        self.assertEqual(query_cost_saved_usd(ROUTE_CLOUD, 1000, 200), 0.0)

    def test_residency_is_full_only_when_no_phi_leaves(self):
        self.assertEqual(data_residency_pct(0), 100.0)
        self.assertEqual(data_residency_pct(4), 0.0)

    def test_tokens_per_sec_needs_generation_time(self):
        self.assertIsNone(tokens_per_sec(0, 100))
        self.assertIsNone(tokens_per_sec(20, None))
        self.assertAlmostEqual(tokens_per_sec(20, 500), 40.0)

    def test_session_totals_and_payload_shape(self):
        tracker = TelemetryTracker()
        first = tracker.record(
            source_layer="EDGE_VECTOR_CACHE",
            latency_ms=12.4,
            input_tokens=100,
            output_tokens=0,
            generation_ms=None,
            phi_tokens_egressed=0,
            cache_hit=True,
            cache_hit_latency_ms=3.8,
            signature_count=12,
        )
        self.assertEqual(first["active_route"], ROUTE_VECTOR)
        self.assertEqual(first["ttft_ms"], first["latency_ms"])
        self.assertEqual(first["phi_tokens_egressed"], 0)
        self.assertEqual(first["data_residency_pct"], 100.0)
        self.assertEqual(first["cache_hit_latency_ms"], 3.8)
        self.assertEqual(first["active_hotpatches_count"], 12)
        self.assertEqual(first["vram_total_gb"], 128.0)
        self.assertGreater(first["query_cost_saved_usd"], 0)

        second = tracker.record(
            source_layer="CLOUD_DEFENDER_HOTPATCH",
            latency_ms=800,
            input_tokens=50,
            output_tokens=10,
            generation_ms=700,
            phi_tokens_egressed=0,
            cache_hit=False,
            cache_hit_latency_ms=4.0,
            signature_count=13,
        )
        self.assertEqual(second["query_cost_saved_usd"], 0.0)
        self.assertIsNone(second["cache_hit_latency_ms"])
        self.assertAlmostEqual(
            second["cumulative_savings_usd"], first["query_cost_saved_usd"]
        )
        self.assertEqual(second["queries_processed"], 2)

        live = tracker.snapshot(signature_count=13)
        self.assertEqual(live["active_route"], ROUTE_CLOUD)
        self.assertEqual(live["queries_processed"], 2)
        self.assertEqual(live["active_hotpatches_count"], 13)
        self.assertNotIn("prompt", live)

        tracker.clear()
        cleared = tracker.snapshot(signature_count=8)
        self.assertIsNone(cleared["latency_ms"])
        self.assertIsNone(cleared["active_route"])
        self.assertEqual(cleared["cumulative_savings_usd"], 0.0)
        self.assertEqual(cleared["queries_processed"], 0)
        self.assertEqual(cleared["active_hotpatches_count"], 8)


if __name__ == "__main__":
    unittest.main()
