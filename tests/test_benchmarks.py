from __future__ import annotations

from w8_biayn.benchmarks import get_benchmark, list_benchmarks


def test_benchmark_scorecard_contains_browser_and_mobile_transfer():
    keys = {benchmark.key for benchmark in list_benchmarks()}

    assert "webvoyager-domdiff-heldout" in keys
    assert "harbor-domdiff-browser-swe" in keys
    assert "androidworld-transfer" in keys
    assert "osworld-smoke" in keys
    assert "osworld-tiny-local" in keys


def test_get_benchmark_rejects_unknown_key():
    try:
        get_benchmark("missing")
    except KeyError as exc:
        assert "Known benchmarks" in str(exc)
    else:
        raise AssertionError("expected KeyError")
