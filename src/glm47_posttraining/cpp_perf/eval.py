"""Evaluation aggregation for C++ performance-RL runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

UPLIFT_REQUIRED_LABELS = ("base", "sft", "grpo")
UPLIFT_METRICS = ("correct_and_faster_rate", "mean_best_reward")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: object) -> Path:
    """Write formatted JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def aggregate_aider_eval_records(records: Iterable[dict[str, Any]], *, label: str) -> dict[str, Any]:
    """Aggregate Aider-style pass@1/pass@2 and stratified correctness metrics."""

    rows = list(records)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    for task_rows in by_task.values():
        task_rows.sort(key=lambda row: int(row.get("sample_index") or 0))

    def passes_within(task_rows: list[dict[str, Any]], tries: int) -> bool:
        return any(row.get("all_tests_pass") is True for row in task_rows[:tries])

    task_count = len(by_task)
    pass_at_1 = sum(passes_within(task_rows, 1) for task_rows in by_task.values())
    pass_at_2 = sum(passes_within(task_rows, 2) for task_rows in by_task.values())
    passed_assertions = sum(int(row.get("passed_assertions") or 0) for row in rows)
    total_assertions = sum(int(row.get("total_assertions") or 0) for row in rows)

    strata: dict[str, dict[str, list[bool]]] = {
        "difficulty": defaultdict(list),
        "topic_category": defaultdict(list),
    }
    for task_rows in by_task.values():
        representative = task_rows[0]
        passed = passes_within(task_rows, 2)
        for field in strata:
            strata[field][str(representative.get(field) or "unknown")].append(passed)

    stratified = {
        field: {
            name: sum(outcomes) / len(outcomes) if outcomes else 0.0
            for name, outcomes in sorted(groups.items())
        }
        for field, groups in strata.items()
    }
    difficulty_metrics = {
        f"pass_rate_{name.lower()}": rate
        for name, rate in stratified["difficulty"].items()
    }
    topic_names = {
        "State & concurrency": "state_concurrency",
        "Time & date": "time_date",
        "Numerical reasoning": "numerical",
        "Text & parsing": "text_parsing",
        "Logic, grids & games": "logic_grids_games",
        "Algorithms & data structures": "algorithms_data_structures",
    }
    topic_metrics = {
        f"pass_rate_{topic_names.get(name, name.lower().replace(' ', '_'))}": rate
        for name, rate in stratified["topic_category"].items()
    }
    sample_count = len(rows)
    rubric_names = (
        "correctness",
        "reasoning",
        "memory_safety",
        "thread_safety",
        "runtime",
        "cpp_quality",
    )
    rubric_means = {
        f"mean_rubric_{name}": (
            sum(float(row.get(f"rubric_{name}") or 0.0) for row in rows) / sample_count
            if sample_count
            else 0.0
        )
        for name in rubric_names
    }
    return {
        "label": label,
        "task_count": task_count,
        "sample_count": sample_count,
        "pass_at_1": pass_at_1 / task_count if task_count else 0.0,
        "pass_at_2": pass_at_2 / task_count if task_count else 0.0,
        "assertion_pass_ratio": passed_assertions / total_assertions if total_assertions else 0.0,
        "context_exhausted_rate": sum(
            row.get("reason") in {"context_exhausted", "rubric1_context_exhausted"}
            or row.get("outcome") == "context_exhausted"
            for row in rows
        )
        / sample_count
        if sample_count
        else 0.0,
        "valid_format_rate": sum(bool(row.get("format_valid")) for row in rows) / sample_count
        if sample_count
        else 0.0,
        "compile_success_rate": sum(
            bool(row.get("sandbox_ran", row.get("format_valid")))
            and not bool(row.get("compile_error"))
            for row in rows
        )
        / sample_count
        if sample_count
        else 0.0,
        "sanitizer_pass_rate": sum(
            bool(row.get("sandbox_ran", row.get("format_valid")))
            and not bool(row.get("compile_error"))
            and not bool(row.get("sanitizer_error"))
            for row in rows
        )
        / sample_count
        if sample_count
        else 0.0,
        "timeout_rate": sum(bool(row.get("timeout")) for row in rows) / sample_count
        if sample_count
        else 0.0,
        "thread_sanitizer_error_rate": sum(
            bool(row.get("thread_sanitizer_error")) for row in rows
        )
        / sample_count
        if sample_count
        else 0.0,
        "thread_sanitizer_timeout_rate": sum(
            bool(row.get("thread_sanitizer_timeout")) for row in rows
        )
        / sample_count
        if sample_count
        else 0.0,
        "stratified_pass_at_2": stratified,
        **rubric_means,
        **difficulty_metrics,
        **topic_metrics,
    }


def aggregate_eval_records(records: Iterable[dict[str, Any]], *, label: str) -> dict[str, Any]:
    """Aggregate per-sample reward records into task-level uplift metrics."""

    rows = list(records)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)

    best_rows = [_best_record(task_rows) for task_rows in by_task.values()]
    task_count = len(best_rows)
    sample_count = len(rows)
    correct = [row for row in best_rows if row.get("all_tests_pass") is True]
    correct_and_faster = [
        row
        for row in correct
        if _positive_number(row.get("runtime_cpu_ns")) and _positive_number(row.get("reference_runtime_cpu_ns"))
        and float(row["runtime_cpu_ns"]) < float(row["reference_runtime_cpu_ns"])
    ]
    missing_runtime = [
        row
        for row in correct
        if row.get("runtime_cpu_ns") is None or row.get("reference_runtime_cpu_ns") is None
    ]
    missing_runtime_task_ids = sorted(str(row.get("task_id")) for row in missing_runtime if row.get("task_id") is not None)
    compile_errors = [row for row in rows if row.get("compile_error") is True]
    sanitizer_errors = [row for row in rows if row.get("sanitizer_error") is True]
    invalid_format = [row for row in rows if row.get("reason") == "invalid_format"]
    timeouts = [row for row in rows if row.get("timeout") is True]
    speedups = [
        float(row["reference_runtime_cpu_ns"]) / float(row["runtime_cpu_ns"])
        for row in correct_and_faster
        if float(row["runtime_cpu_ns"]) > 0
    ]
    best_rewards = [float(row.get("reward", 0.0)) for row in best_rows]
    sample_rewards = [float(row.get("reward", 0.0)) for row in rows]

    return {
        "label": label,
        "task_count": task_count,
        "sample_count": sample_count,
        "samples_per_task_mean": sample_count / task_count if task_count else 0.0,
        "pass_rate": len(correct) / task_count if task_count else 0.0,
        "correct_and_faster_rate": len(correct_and_faster) / task_count if task_count else 0.0,
        "missing_runtime_count": len(missing_runtime),
        "missing_runtime_rate": len(missing_runtime) / task_count if task_count else 0.0,
        "missing_runtime_task_ids": missing_runtime_task_ids,
        "compile_error_rate": len(compile_errors) / sample_count if sample_count else 0.0,
        "sanitizer_error_rate": len(sanitizer_errors) / sample_count if sample_count else 0.0,
        "timeout_rate": len(timeouts) / sample_count if sample_count else 0.0,
        "invalid_format_rate": len(invalid_format) / sample_count if sample_count else 0.0,
        "mean_best_reward": _mean(best_rewards),
        "mean_sample_reward": _mean(sample_rewards),
        "mean_correct_faster_speedup": _mean(speedups),
        "best_records": best_rows,
    }


def compare_eval_summaries(summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare model summaries and identify the best label for primary metrics."""

    rows = list(summaries)
    if not rows:
        return {
            "summaries": [],
            "best_correct_and_faster": None,
            "best_mean_reward": None,
            "uplift_gate": _uplift_gate([]),
        }
    best_correct = max(rows, key=lambda item: float(item.get("correct_and_faster_rate", 0.0)))
    best_reward = max(rows, key=lambda item: float(item.get("mean_best_reward", 0.0)))
    return {
        "summaries": rows,
        "best_correct_and_faster": best_correct["label"],
        "best_mean_reward": best_reward["label"],
        "uplift_gate": _uplift_gate(rows),
    }


def _best_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: float(row.get("reward", 0.0)))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _positive_number(value: object) -> bool:
    try:
        return value is not None and float(value) > 0
    except (TypeError, ValueError):
        return False


def _uplift_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {str(row.get("label")): row for row in rows if row.get("label") is not None}
    missing_required = [label for label in UPLIFT_REQUIRED_LABELS if label not in by_label]
    metric_winners = {
        "correct_and_faster_rate": _winning_label(rows, "correct_and_faster_rate"),
        "mean_best_reward": _winning_label(rows, "mean_best_reward"),
    }
    label_status = {
        label: {
            "missing_runtime_count": _missing_runtime_count(row),
            "missing_runtime_rate": _float_metric(row, "missing_runtime_rate"),
            "missing_runtime_task_ids": list(row.get("missing_runtime_task_ids", []))
            if isinstance(row.get("missing_runtime_task_ids"), list)
            else [],
        }
        for label, row in sorted(by_label.items())
    }
    reasons: list[str] = []
    metric_gates = {
        "grpo_beats_base_and_sft_correct_and_faster_rate": False,
        "grpo_beats_base_and_sft_mean_best_reward": False,
        "all_missing_runtime_rate_zero": False,
    }
    held_out_lift = False
    missing_runtime_clean = False
    if missing_required:
        reasons.append(f"missing_required_labels:{','.join(missing_required)}")
    else:
        grpo = by_label["grpo"]
        base = by_label["base"]
        sft = by_label["sft"]
        metric_gates["grpo_beats_base_and_sft_correct_and_faster_rate"] = _beats_all(
            grpo,
            [base, sft],
            "correct_and_faster_rate",
        )
        metric_gates["grpo_beats_base_and_sft_mean_best_reward"] = _beats_all(
            grpo,
            [base, sft],
            "mean_best_reward",
        )
        held_out_lift = all(metric_gates[key] for key in metric_gates if key.startswith("grpo_beats_"))
        if not metric_gates["grpo_beats_base_and_sft_correct_and_faster_rate"]:
            reasons.append("grpo_does_not_beat_base_and_sft_correct_and_faster_rate")
        if not metric_gates["grpo_beats_base_and_sft_mean_best_reward"]:
            reasons.append("grpo_does_not_beat_base_and_sft_mean_best_reward")
        missing_runtime_clean = all(
            _missing_runtime_count(row) == 0 and _float_metric(row, "missing_runtime_rate") == 0.0
            for row in by_label.values()
        )
        metric_gates["all_missing_runtime_rate_zero"] = missing_runtime_clean
        if not missing_runtime_clean:
            blocked_labels = [
                label
                for label, row in sorted(by_label.items())
                if _missing_runtime_count(row) > 0 or _float_metric(row, "missing_runtime_rate") > 0.0
            ]
            reasons.append(f"missing_runtime_nonzero:{','.join(blocked_labels)}")
    passed = not missing_required and held_out_lift and missing_runtime_clean
    if passed:
        verdict = "formal_uplift_passed"
    elif missing_required:
        verdict = "insufficient_labels"
    elif held_out_lift:
        verdict = "held_out_lift_but_gate_failed"
    else:
        verdict = "no_uplift"
    return {
        "passed": passed,
        "verdict": verdict,
        "held_out_lift": held_out_lift,
        "missing_runtime_clean": missing_runtime_clean,
        "required_labels": list(UPLIFT_REQUIRED_LABELS),
        "missing_required_labels": missing_required,
        "metric_winners": metric_winners,
        "metric_gates": metric_gates,
        "label_status": label_status,
        "reasons": reasons,
    }


def _winning_label(rows: list[dict[str, Any]], metric: str) -> str | None:
    if not rows:
        return None
    label = max(rows, key=lambda item: _float_metric(item, metric)).get("label")
    return str(label) if label is not None else None


def _beats_all(candidate: dict[str, Any], baselines: list[dict[str, Any]], metric: str) -> bool:
    value = _float_metric(candidate, metric)
    return all(value > _float_metric(baseline, metric) for baseline in baselines)


def _float_metric(row: dict[str, Any], metric: str) -> float:
    try:
        return float(row.get(metric, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _missing_runtime_count(row: dict[str, Any]) -> int:
    value = row.get("missing_runtime_count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    task_ids = row.get("missing_runtime_task_ids")
    if isinstance(task_ids, list):
        return len(task_ids)
    return 0
