"""Unit tests for the pure functions in ``load_test``.

These cover the request-building / response-parsing logic for every backend,
the ``LevelResult.summary`` aggregation (including the all-failed branch), and
the interpolating ``percentile`` helper.
"""

import statistics

import pytest

from load_test import (
    PROMPTS,
    LevelResult,
    RequestResult,
    build_request,
    parse_response,
    percentile,
)

SAMPLE_PROMPT = PROMPTS[0]


# --------------------------------------------------------------------------- #
# build_request
# --------------------------------------------------------------------------- #
class TestBuildRequest:
    def test_llama_default(self):
        path, body, headers = build_request(SAMPLE_PROMPT, 64, "llama", "")
        assert path == "/infill"
        assert body["input_prefix"] == SAMPLE_PROMPT["input_prefix"]
        assert body["input_suffix"] == SAMPLE_PROMPT["input_suffix"]
        assert body["n_predict"] == 64
        assert body["stream"] is False
        assert body["cache_prompt"] is True
        assert headers == {}

    def test_unknown_backend_falls_back_to_llama(self):
        # Anything that is not ollama/anthropic/openai uses the llama /infill path.
        path, body, headers = build_request(SAMPLE_PROMPT, 32, "something-else", "")
        assert path == "/infill"
        assert "n_predict" in body
        assert headers == {}

    def test_ollama(self):
        path, body, headers = build_request(SAMPLE_PROMPT, 128, "ollama", "qwen2.5-coder:1.5b")
        assert path == "/api/generate"
        assert body["model"] == "qwen2.5-coder:1.5b"
        assert body["stream"] is False
        assert body["options"] == {"num_predict": 128}
        # FIM template wraps prefix/suffix with the sentinel tokens.
        assert body["prompt"] == (
            f"<|fim_prefix|>{SAMPLE_PROMPT['input_prefix']}"
            f"<|fim_suffix|>{SAMPLE_PROMPT['input_suffix']}<|fim_middle|>"
        )
        assert headers == {}

    def test_anthropic(self):
        path, body, headers = build_request(SAMPLE_PROMPT, 256, "anthropic", "claude-3-5-sonnet")
        assert path == "/v1/messages"
        assert body["model"] == "claude-3-5-sonnet"
        assert body["max_tokens"] == 256
        assert body["messages"][0]["role"] == "user"
        # Both prefix and suffix are embedded in the single user message.
        content = body["messages"][0]["content"]
        assert SAMPLE_PROMPT["input_prefix"] in content
        assert SAMPLE_PROMPT["input_suffix"] in content
        # Anthropic requires the version header.
        assert headers["anthropic-version"] == "2023-06-01"

    def test_openai(self):
        path, body, headers = build_request(SAMPLE_PROMPT, 200, "openai", "gpt-4o-mini")
        assert path == "/v1/chat/completions"
        assert body["model"] == "gpt-4o-mini"
        assert body["max_tokens"] == 200
        assert body["messages"][0]["role"] == "user"
        content = body["messages"][0]["content"]
        assert SAMPLE_PROMPT["input_prefix"] in content
        assert SAMPLE_PROMPT["input_suffix"] in content
        # OpenAI uses the bearer header (added by the caller), no extra headers here.
        assert headers == {}


# --------------------------------------------------------------------------- #
# parse_response
# --------------------------------------------------------------------------- #
class TestParseResponse:
    def test_llama_default(self):
        assert parse_response({"tokens_predicted": 42}, "llama") == 42

    def test_llama_missing_field_defaults_zero(self):
        assert parse_response({}, "llama") == 0

    def test_unknown_backend_uses_llama_field(self):
        assert parse_response({"tokens_predicted": 7}, "mystery") == 7

    def test_ollama(self):
        assert parse_response({"eval_count": 17}, "ollama") == 17
        assert parse_response({}, "ollama") == 0

    def test_anthropic(self):
        assert parse_response({"usage": {"output_tokens": 99}}, "anthropic") == 99
        assert parse_response({}, "anthropic") == 0
        assert parse_response({"usage": {}}, "anthropic") == 0

    def test_openai(self):
        assert parse_response({"usage": {"completion_tokens": 55}}, "openai") == 55
        assert parse_response({}, "openai") == 0
        assert parse_response({"usage": {}}, "openai") == 0


# --------------------------------------------------------------------------- #
# percentile (the new interpolating helper)
# --------------------------------------------------------------------------- #
class TestPercentile:
    def test_matches_numpy_linear_method(self):
        # Reference values from numpy.percentile(..., method="linear").
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert percentile(values, 0.50) == pytest.approx(5.5)
        assert percentile(values, 0.90) == pytest.approx(9.1)
        assert percentile(values, 0.99) == pytest.approx(9.91)

    def test_p50_equals_median(self):
        for values in ([10, 20, 30, 40], [3.0, 1.0, 4.0, 1.0, 5.0]):
            assert percentile(values, 0.5) == pytest.approx(statistics.median(values))

    def test_boundaries_are_min_and_max(self):
        values = [4.0, 1.0, 9.0, 2.0]
        assert percentile(values, 0.0) == pytest.approx(1.0)
        assert percentile(values, 1.0) == pytest.approx(9.0)

    def test_single_element(self):
        assert percentile([4.2], 0.0) == pytest.approx(4.2)
        assert percentile([4.2], 0.99) == pytest.approx(4.2)
        assert percentile([4.2], 1.0) == pytest.approx(4.2)

    def test_unsorted_input(self):
        assert percentile([5, 1, 3, 2, 4], 0.5) == pytest.approx(3.0)

    def test_small_sample_does_not_collapse_to_max(self):
        # The crux of the fix: old code did sorted(v)[int(len(v) * q)].
        # For n=3, q=0.9 -> int(2.7) = 2 -> v[2] == max. Interpolation gives 2.8.
        values = [1.0, 2.0, 3.0]
        old_index_based = sorted(values)[int(len(values) * 0.9)]
        assert old_index_based == max(values)  # demonstrates the old collapse
        assert percentile(values, 0.9) == pytest.approx(2.8)
        assert percentile(values, 0.9) < max(values)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 0.5)

    @pytest.mark.parametrize("bad_q", [-0.1, 1.1, 2.0])
    def test_out_of_range_q_raises(self, bad_q):
        with pytest.raises(ValueError):
            percentile([1.0, 2.0, 3.0], bad_q)


# --------------------------------------------------------------------------- #
# LevelResult.summary
# --------------------------------------------------------------------------- #
def _ok(latency, tokens=10):
    return RequestResult(latency=latency, status=200, tokens_predicted=tokens)


def _fail(latency=0.0, status=0, error="boom"):
    return RequestResult(latency=latency, status=status, tokens_predicted=0, error=error)


class TestLevelResultSummary:
    def test_success_partition(self):
        level = LevelResult(
            concurrency=3,
            results=[_ok(1.0), _ok(2.0), _fail(), _fail(status=500, error=None)],
        )
        # status 500 with no error string still counts as a failure.
        assert len(level.successes) == 2
        assert len(level.failures) == 2

    def test_summary_success_path(self):
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        level = LevelResult(concurrency=5, results=[_ok(x, tokens=20) for x in latencies])
        s = level.summary()
        assert s["concurrency"] == 5
        assert s["total_requests"] == 5
        assert s["successes"] == 5
        assert s["failures"] == 0
        assert s["latency_min"] == 1.0
        assert s["latency_max"] == 5.0
        assert s["latency_mean"] == 3.0
        assert s["latency_p50"] == 3.0
        # Percentiles use the interpolating helper.
        assert s["latency_p90"] == round(percentile(latencies, 0.90), 2)
        assert s["latency_p99"] == round(percentile(latencies, 0.99), 2)
        assert s["avg_tokens"] == 20.0
        assert "error" not in s

    def test_summary_counts_failures_alongside_successes(self):
        level = LevelResult(
            concurrency=4,
            results=[_ok(1.0), _ok(2.0), _fail(), _fail(status=503, error=None)],
        )
        s = level.summary()
        assert s["total_requests"] == 4
        assert s["successes"] == 2
        assert s["failures"] == 2

    def test_summary_all_failed_branch(self):
        level = LevelResult(concurrency=2, results=[_fail(), _fail(status=500, error=None)])
        s = level.summary()
        assert s == {"concurrency": 2, "error": "all requests failed"}

    def test_summary_empty_results_is_all_failed(self):
        # No results at all -> no successes -> all-failed branch.
        s = LevelResult(concurrency=1).summary()
        assert s == {"concurrency": 1, "error": "all requests failed"}
