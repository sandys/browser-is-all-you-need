"""Evaluation aggregation for C++ performance-RL runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


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
        "missing_runtime_rate": len(missing_runtime) / task_count if task_count else 0.0,
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
        return {"summaries": [], "best_correct_and_faster": None, "best_mean_reward": None}
    best_correct = max(rows, key=lambda item: float(item.get("correct_and_faster_rate", 0.0)))
    best_reward = max(rows, key=lambda item: float(item.get("mean_best_reward", 0.0)))
    return {
        "summaries": rows,
        "best_correct_and_faster": best_correct["label"],
        "best_mean_reward": best_reward["label"],
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
