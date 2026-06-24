from __future__ import annotations

import pytest

from w8_biayn.benchmarks import get_benchmark, list_benchmarks


def test_benchmark_ladder_contains_cpp_performance_steps():
    keys = {benchmark.key for benchmark in list_benchmarks()}

    assert "pie-one-smoke" in keys
    assert "pie-10-task" in keys
    assert "skyrl-data-bundle" in keys
    assert "pie-heldout-effk" in keys
    assert "sft-cold-start" in keys
    assert "grpo-tiny" in keys


def test_get_benchmark_errors_with_known_keys():
    with pytest.raises(KeyError) as exc:
        get_benchmark("missing")
    assert "Known benchmarks" in str(exc)
