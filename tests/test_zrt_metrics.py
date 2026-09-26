"""Histogram quantiles from a ZRT / vLLM metrics scrape."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from core.zrt_metrics import histogram_quantile, parse_vllm_metrics  # noqa: E402


SAMPLE = """
# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="mistral"} 3.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.25",model_name="mistral"} 7.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="mistral"} 13.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="mistral"} 13.0
vllm:request_prompt_tokens_sum{engine="0",model_name="mistral"} 14030.0
vllm:request_generation_tokens_sum{engine="0",model_name="mistral"} 847.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="mistral"} 13.0
vllm:request_success_total{engine="0",finished_reason="error",model_name="mistral"} 1.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="5.0",model_name="mistral"} 3.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="10.0",model_name="mistral"} 13.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="mistral"} 13.0
"""


class TestZrtMetrics(unittest.TestCase):
    def test_quantile_interpolates_inside_the_bucket(self):
        buckets = [(0.1, 3.0), (0.25, 7.0), (0.5, 13.0)]
        self.assertAlmostEqual(histogram_quantile(buckets, 0.99), 0.4945, places=3)

    def test_parse_ignores_failed_requests(self):
        parsed = parse_vllm_metrics(SAMPLE)
        self.assertEqual(parsed["requests_succeeded"], 13)
        self.assertEqual(parsed["prompt_tokens_total"], 14030)
        self.assertEqual(parsed["generated_tokens_total"], 847)
        self.assertAlmostEqual(parsed["ttft_p99_ms"], 494.6, places=1)
        self.assertIsNotNone(parsed["e2e_latency_p50_s"])

    def test_clear_baseline_drops_earlier_requests(self):
        from core import zrt_metrics

        previous = zrt_metrics._BASELINE
        try:
            zrt_metrics._BASELINE = zrt_metrics._raw_from_text(SAMPLE)
            parsed = zrt_metrics._session_since_clear(SAMPLE)
            self.assertIsNone(parsed["ttft_p50_ms"])
            self.assertIsNone(parsed["prompt_tokens_total"])
            self.assertIsNone(parsed["requests_succeeded"])
            self.assertIsNone(parsed["e2e_latency_p50_s"])
        finally:
            zrt_metrics._BASELINE = previous

    def test_empty_scrape_stays_empty(self):
        parsed = parse_vllm_metrics("")
        self.assertIsNone(parsed["ttft_p50_ms"])
        self.assertIsNone(parsed["prompt_tokens_total"])


if __name__ == "__main__":
    unittest.main()
