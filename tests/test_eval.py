from __future__ import annotations

from w8_biayn.cpp_perf.eval import aggregate_eval_records, compare_eval_summaries


def test_aggregate_eval_records_prefers_best_per_task():
    records = [
        {
            "task_id": "t1",
            "reward": -1.0,
            "reason": "invalid_format",
            "all_tests_pass": False,
            "compile_error": False,
            "sanitizer_error": False,
            "timeout": False,
            "runtime_cpu_ns": None,
            "reference_runtime_cpu_ns": 100,
        },
        {
            "task_id": "t1",
            "reward": 1.5,
            "reason": "correct",
            "all_tests_pass": True,
            "compile_error": False,
            "sanitizer_error": False,
            "timeout": False,
            "runtime_cpu_ns": 50,
            "reference_runtime_cpu_ns": 100,
        },
        {
            "task_id": "t2",
            "reward": 1.0,
            "reason": "correct",
            "all_tests_pass": True,
            "compile_error": False,
            "sanitizer_error": False,
            "timeout": False,
            "runtime_cpu_ns": None,
            "reference_runtime_cpu_ns": 100,
        },
    ]

    summary = aggregate_eval_records(records, label="grpo")

    assert summary["task_count"] == 2
    assert summary["sample_count"] == 3
    assert summary["pass_rate"] == 1.0
    assert summary["correct_and_faster_rate"] == 0.5
    assert summary["missing_runtime_rate"] == 0.5
    assert summary["invalid_format_rate"] == 1 / 3
    assert summary["mean_correct_faster_speedup"] == 2.0


def test_compare_eval_summaries_selects_primary_winners():
    comparison = compare_eval_summaries(
        [
            {"label": "base", "correct_and_faster_rate": 0.1, "mean_best_reward": 0.5},
            {"label": "grpo", "correct_and_faster_rate": 0.2, "mean_best_reward": 0.4},
            {"label": "sft", "correct_and_faster_rate": 0.15, "mean_best_reward": 0.7},
        ]
    )

    assert comparison["best_correct_and_faster"] == "grpo"
    assert comparison["best_mean_reward"] == "sft"
