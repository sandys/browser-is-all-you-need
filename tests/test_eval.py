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
    assert summary["missing_runtime_count"] == 1
    assert summary["missing_runtime_rate"] == 0.5
    assert summary["missing_runtime_task_ids"] == ["t2"]
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
    assert comparison["uplift_gate"]["passed"] is False
    assert comparison["uplift_gate"]["verdict"] == "no_uplift"


def test_compare_eval_summaries_reports_formal_uplift_pass():
    comparison = compare_eval_summaries(
        [
            {
                "label": "base",
                "correct_and_faster_rate": 0.0,
                "mean_best_reward": -0.9,
                "missing_runtime_count": 0,
                "missing_runtime_rate": 0.0,
            },
            {
                "label": "sft",
                "correct_and_faster_rate": 0.17,
                "mean_best_reward": -0.47,
                "missing_runtime_count": 0,
                "missing_runtime_rate": 0.0,
            },
            {
                "label": "grpo",
                "correct_and_faster_rate": 0.25,
                "mean_best_reward": 0.24,
                "missing_runtime_count": 0,
                "missing_runtime_rate": 0.0,
            },
        ]
    )

    gate = comparison["uplift_gate"]
    assert gate["passed"] is True
    assert gate["verdict"] == "formal_uplift_passed"
    assert gate["held_out_lift"] is True
    assert gate["missing_runtime_clean"] is True


def test_compare_eval_summaries_blocks_formal_uplift_on_missing_runtime():
    comparison = compare_eval_summaries(
        [
            {
                "label": "base",
                "correct_and_faster_rate": 0.0,
                "mean_best_reward": -0.9,
                "missing_runtime_count": 0,
                "missing_runtime_rate": 0.0,
                "missing_runtime_task_ids": [],
            },
            {
                "label": "sft",
                "correct_and_faster_rate": 0.17,
                "mean_best_reward": -0.47,
                "missing_runtime_count": 2,
                "missing_runtime_rate": 0.0019,
                "missing_runtime_task_ids": ["pie_cpp_validation_001331", "pie_cpp_validation_003055"],
            },
            {
                "label": "grpo",
                "correct_and_faster_rate": 0.25,
                "mean_best_reward": 0.24,
                "missing_runtime_count": 7,
                "missing_runtime_rate": 0.0068,
                "missing_runtime_task_ids": ["pie_cpp_test_002495", "pie_cpp_validation_001331"],
            },
        ]
    )

    gate = comparison["uplift_gate"]
    assert gate["passed"] is False
    assert gate["verdict"] == "held_out_lift_but_gate_failed"
    assert gate["held_out_lift"] is True
    assert gate["missing_runtime_clean"] is False
    assert gate["reasons"] == ["missing_runtime_nonzero:grpo,sft"]
    assert gate["label_status"]["grpo"]["missing_runtime_count"] == 7
    assert gate["label_status"]["grpo"]["missing_runtime_task_ids"] == [
        "pie_cpp_test_002495",
        "pie_cpp_validation_001331",
    ]


def test_compare_eval_summaries_requires_base_sft_and_grpo():
    comparison = compare_eval_summaries(
        [
            {
                "label": "grpo",
                "correct_and_faster_rate": 0.25,
                "mean_best_reward": 0.24,
                "missing_runtime_count": 0,
                "missing_runtime_rate": 0.0,
            }
        ]
    )

    gate = comparison["uplift_gate"]
    assert gate["passed"] is False
    assert gate["verdict"] == "insufficient_labels"
    assert gate["missing_required_labels"] == ["base", "sft"]
