"""Unit tests for the repo-owned W&B reporting layer (no wandb/network needed)."""

from __future__ import annotations

from w8_biayn import wandb_report
from w8_biayn.wandb_report import (
    RECORDS_TABLE_COLUMNS,
    RolloutHealth,
    abort_counts,
    alert,
    classify_failure,
    eval_metrics_from_summary,
    init_run,
    log_abort_distribution,
    log_eval_summary,
    log_uplift,
    records_table_rows,
    uplift_rows,
)


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.alerts: list[dict] = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def alert(self, **kwargs) -> None:
        self.alerts.append(kwargs)


def _summary() -> dict:
    return {
        "label": "base",
        "task_count": 8,
        "sample_count": 16,
        "samples_per_task_mean": 2.0,
        "pass_rate": 0.5,
        "correct_and_faster_rate": 0.25,
        "missing_runtime_rate": 0.0,
        "missing_runtime_task_ids": ["t1"],  # non-numeric: must be skipped
        "compile_error_rate": 0.125,
        "sanitizer_error_rate": 0.0,
        "timeout_rate": 0.0625,
        "invalid_format_rate": 0.1875,
        "mean_best_reward": 0.42,
        "mean_sample_reward": 0.21,
        "mean_correct_faster_speedup": 1.8,
        "best_records": [{"task_id": "t1"}],  # non-numeric: must be skipped
    }


def test_eval_metrics_map_numeric_summary_keys_only() -> None:
    metrics = eval_metrics_from_summary(_summary())

    assert metrics["eval/pass_rate"] == 0.5
    assert metrics["eval/mean_correct_faster_speedup"] == 1.8
    assert metrics["eval/task_count"] == 8.0
    assert "eval/best_records" not in metrics
    assert "eval/missing_runtime_task_ids" not in metrics
    assert "eval/label" not in metrics


def test_log_eval_summary_logs_and_mirrors_to_summary() -> None:
    run = FakeRun()

    metrics = log_eval_summary(run, _summary())

    assert run.logged == [metrics]
    assert run.summary["eval/pass_rate"] == 0.5
    # None run: still returns the mapping, never raises.
    assert log_eval_summary(None, _summary()) == metrics


def test_records_table_rows_shape_and_truncation() -> None:
    records = [
        {
            "task_id": "t1",
            "problem_id": "p1",
            "reason": "ok",
            "reward": 0.9,
            "all_tests_pass": True,
            "tests_passed": 4,
            "tests_total": 4,
            "compile_error": False,
            "timeout": False,
            "format_valid": True,
            "runtime_speedup": 2.5,
            "agent_steps": 7,
            "response": "x" * 1000,
        }
    ] * 3

    rows = records_table_rows(records, max_rows=2)

    assert len(rows) == 2
    assert len(rows[0]) == len(RECORDS_TABLE_COLUMNS)
    snippet = rows[0][RECORDS_TABLE_COLUMNS.index("response_snippet")]
    assert len(snippet) == 300


def test_classify_failure_categories() -> None:
    assert classify_failure({"abort_reason": "adapter_session_empty"}) == "adapter_session_empty"
    assert classify_failure({"reason": "invalid_format"}) == "invalid_format"
    assert classify_failure({"reason": "exception:TimeoutError"}) == "exception:TimeoutError"
    assert classify_failure({"reason": "reward_exception", "exception": "boom"}) == "reward_exception"
    # Test failures are quality outcomes, not infra aborts.
    assert classify_failure({"reason": "tests_failed", "all_tests_pass": False}) is None
    assert classify_failure({"reason": "ok", "all_tests_pass": True}) is None


def test_abort_distribution_counts_and_metrics() -> None:
    records = [
        {"abort_reason": "adapter_session_empty"},
        {"abort_reason": "adapter_session_empty"},
        {"reason": "invalid_format"},
        {"reason": "ok", "all_tests_pass": True},
    ]
    run = FakeRun()

    counts = log_abort_distribution(run, records)

    assert counts == {"adapter_session_empty": 2, "invalid_format": 1}
    assert abort_counts(records) == counts
    logged = run.logged[0]
    assert logged["eval/abort_rate"] == 0.75
    assert logged["eval/abort/adapter_session_empty"] == 2
    assert run.summary["eval/abort_rate"] == 0.75
    assert log_abort_distribution(None, records) == counts


def test_uplift_rows_and_summary_keys() -> None:
    comparison = {
        "summaries": [
            {"label": "base", "pass_rate": 0.4, "correct_and_faster_rate": 0.1, "mean_best_reward": 0.2},
            {"label": "grpo", "pass_rate": 0.6, "correct_and_faster_rate": 0.3, "mean_best_reward": 0.5},
        ],
        "best_correct_and_faster": "grpo",
        "best_mean_reward": "grpo",
        "uplift_gate": {"passed": True, "threshold": 0.05},
    }
    run = FakeRun()

    rows = log_uplift(run, comparison)

    assert [row[0] for row in rows] == ["base", "grpo"]
    assert rows == uplift_rows(comparison)
    assert run.summary["uplift/best_correct_and_faster"] == "grpo"
    assert run.summary["uplift/gate_passed"] is True
    assert run.summary["uplift/gate_threshold"] == 0.05


def test_rollout_health_windows_and_zero_variance_groups() -> None:
    flushed: list[dict] = []
    health = RolloutHealth(window=6, log_fn=flushed.append)

    rows = [
        # group A: identical rewards -> zero-variance group
        {"task_id": "a", "reward": 1.0, "format_valid": True, "all_tests_pass": True, "agent_steps": 4},
        {"task_id": "a", "reward": 1.0, "format_valid": True, "all_tests_pass": True, "agent_steps": 6},
        # group B: spread rewards -> healthy group
        {"task_id": "b", "reward": 0.0, "compile_error": True, "format_valid": False},
        {"task_id": "b", "reward": 0.5, "format_valid": True, "timeout": True},
        # singleton group: excluded from group-variance accounting
        {"task_id": "c", "reward": 0.25, "format_valid": True},
        # abort
        {"task_id": "d", "reward": 0.0, "abort_reason": "wall_clock_timeout", "wall_time_s": 1800.0},
    ]
    for row in rows[:-1]:
        assert health.add(row) is None
    metrics = health.add(rows[-1])

    assert metrics is not None
    assert flushed == [metrics]
    assert metrics["rollout_health/step"] == 0
    assert metrics["rollout_health/samples"] == 6
    assert metrics["rollout_health/zero_variance_group_fraction"] == 0.5
    assert metrics["rollout_health/groups_evaluated"] == 2
    assert metrics["rollout_health/abort_rate"] == 1 / 6
    assert metrics["rollout_health/abort/wall_clock_timeout"] == 1
    assert metrics["rollout_health/format_valid_rate"] == 4 / 6
    assert metrics["rollout_health/compile_error_rate"] == 1 / 6
    assert metrics["rollout_health/agent_steps_mean"] == 5.0
    assert metrics["rollout_health/wall_time_mean_s"] == 1800.0
    assert abs(metrics["rollout_health/reward_mean"] - (2.75 / 6)) < 1e-9

    # next window increments the step axis
    for row in rows[:-1]:
        health.add(row)
    metrics2 = health.add(rows[-1])
    assert metrics2 is not None and metrics2["rollout_health/step"] == 1

    # partial window flushes on demand, empty flush is a no-op
    health.add(rows[0])
    assert health.flush() is not None
    assert health.flush() is None


def test_alert_and_disabled_paths() -> None:
    run = FakeRun()

    assert alert(run, title="NaN kl", text="kl is nan", level="ERROR") is True
    assert run.alerts[0]["title"] == "NaN kl"
    assert alert(None, title="x", text="y") is False


def test_workspace_spec_pins_drill_path_sections() -> None:
    spec = wandb_report.build_workspace_spec(project="proj", entity="me")

    names = [section["name"] for section in spec["sections"]]
    assert names == ["Uplift & Eval Comparison", "Rollout Health (live)", "Training Dynamics", "Pipeline"]
    all_panels = [panel for section in spec["sections"] for panel in section["panels"]]
    all_keys = [key for panel in all_panels for key in panel["y"]]
    # the GRPO heartbeat and the NaN-watch panels must exist
    assert "rollout_health/zero_variance_group_fraction" in all_keys
    assert "train/ppo_kl" in all_keys
    assert "pipeline/elapsed_seconds" in all_keys
    # panels with a custom axis carry it; eval comparison panels use the default
    health = [p for p in all_panels if "rollout_health/abort_rate" in p["y"]][0]
    assert health["x"] == "rollout_health/step"
    evals = [p for p in all_panels if "eval/pass_rate" in p["y"]][0]
    assert evals["x"] is None
    # workspace name must survive wandb-workspaces' no-emoji validator (rejects "C++")
    assert "C++" not in spec["name"]


def test_init_run_returns_none_without_project_or_wandb(monkeypatch) -> None:
    monkeypatch.delenv("SLIME_WANDB_PROJECT", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    assert init_run(run_id="r1", stage="base-eval", job_type="eval") is None
    # With a project but no wandb package installed (dev venv) it still degrades.
    if wandb_report.wandb is None:
        monkeypatch.setenv("SLIME_WANDB_PROJECT", "proj")
        monkeypatch.setenv("WANDB_API_KEY", "k")
        assert init_run(run_id="r1", stage="base-eval", job_type="eval") is None
