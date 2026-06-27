"""Offline report generation for completed C++ performance-RL runs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .cpp_perf.eval import aggregate_eval_records, compare_eval_summaries, read_jsonl, write_json

DEFAULT_REPORT_METRICS = (
    "loss/avg_final_rewards",
    "reward/avg_pass_at_8",
    "eval/all/avg_score",
    "eval/all/pass_at_1",
    "eval/all/mean_positive_reward",
    "policy/policy_entropy",
    "policy/grad_norm",
    "loss/avg_raw_advantages_abs",
    "w8/reward_group_variance_mean",
    "w8/zero_variance_group_fraction",
    "policy/policy_kl",
    "policy/response_length",
    "timing/step",
    "timing/generate",
    "timing/policy_train",
    "timing/fwd_logprobs_values_reward",
    "timing/sync_weights",
)

DEFAULT_EVAL_LABELS = ("base", "sft", "grpo")
DISPLAY_LABELS = {"base": "Base", "sft": "SFT", "grpo": "GRPO-300"}
CHART_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b")


@dataclass(frozen=True)
class RawReportResult:
    """Paths written by the raw run report generator."""

    run_id: str
    report_path: Path
    assets_dir: Path
    uplift_summary_path: Path
    asset_paths: list[Path]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "report_path": str(self.report_path),
            "assets_dir": str(self.assets_dir),
            "uplift_summary_path": str(self.uplift_summary_path),
            "asset_paths": [str(path) for path in self.asset_paths],
        }


def build_raw_run_report(
    *,
    run_id: str,
    run_root: str | Path | None = None,
    output: str | Path | None = None,
    assets_dir: str | Path | None = None,
    records: Mapping[str, str | Path] | None = None,
    metrics_json: str | Path | None = None,
    status_json: str | Path | None = None,
    companion_report: str | Path | None = None,
) -> RawReportResult:
    """Write a raw Markdown report, CSVs, JSON, and SVG curves for one run."""

    root = Path(run_root) if run_root is not None else Path(".w8-biayn") / "runs" / run_id
    out_path = Path(output) if output is not None else Path(f"RUN_REPORT_RAW_{run_id}.md")
    asset_root = Path(assets_dir) if assets_dir is not None else Path(f"RUN_REPORT_RAW_{run_id}_assets")
    asset_root.mkdir(parents=True, exist_ok=True)

    record_paths = _resolve_record_paths(root, records)
    metric_path = Path(metrics_json) if metrics_json is not None else root / "metrics.api.json"
    status_path = Path(status_json) if status_json is not None else root / "status.json"
    companion_path = Path(companion_report) if companion_report is not None else Path(f"RUN_REPORT_{run_id}.md")

    record_rows = {label: read_jsonl(path) for label, path in record_paths.items()}
    eval_rows, missing_rows, comparison = _build_eval_tables(record_rows)
    uplift_summary_path = asset_root / "uplift-summary-recomputed.json"
    write_json(uplift_summary_path, comparison)

    metrics_payload = _read_optional_json(metric_path)
    series_points = _selected_series(metrics_payload, DEFAULT_REPORT_METRICS)
    latest_rows = _latest_metric_rows(series_points, DEFAULT_REPORT_METRICS)

    _write_csv(asset_root / "eval_summary.csv", eval_rows)
    _write_csv(asset_root / "missing_runtime_tasks.csv", missing_rows)
    _write_csv(asset_root / "mlflow_latest_selected.csv", latest_rows)
    _write_csv(asset_root / "mlflow_selected_series.csv", _series_csv_rows(series_points, DEFAULT_REPORT_METRICS))

    _write_eval_charts(asset_root, eval_rows)
    _write_metric_charts(asset_root, series_points)

    artifact_paths = _artifact_paths(root, record_paths, metric_path, status_path, uplift_summary_path)
    artifact_rows = _artifact_hash_rows(artifact_paths)
    _write_csv(asset_root / "artifact_hashes.csv", artifact_rows)

    metadata = {
        "run_id": run_id,
        "task_set_equal": _task_sets_equal(record_rows),
        "eval_summary": eval_rows,
        "missing_runtime_tasks": missing_rows,
        "artifact_hashes": artifact_rows,
        "mlflow_tracking": _metrics_tracking_state(metrics_payload),
        "latest_selected_metrics": latest_rows,
        "uplift_gate": comparison.get("uplift_gate"),
    }
    _write_json_strict(asset_root / "raw_report_metadata.json", metadata)

    report_text = _render_raw_markdown(
        run_id=run_id,
        root=root,
        asset_root=asset_root,
        companion_report=companion_path if companion_path.exists() else None,
        eval_rows=eval_rows,
        missing_rows=missing_rows,
        comparison=comparison,
        artifact_rows=artifact_rows,
        latest_rows=latest_rows,
        metrics_state=_metrics_tracking_state(metrics_payload),
        status_payload=_read_optional_json(status_path),
        task_set_equal=_task_sets_equal(record_rows),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")

    asset_paths = sorted(path for path in asset_root.iterdir() if path.is_file())
    return RawReportResult(
        run_id=run_id,
        report_path=out_path,
        assets_dir=asset_root,
        uplift_summary_path=uplift_summary_path,
        asset_paths=asset_paths,
    )


def _resolve_record_paths(root: Path, records: Mapping[str, str | Path] | None) -> dict[str, Path]:
    if records:
        return {label: Path(path) for label, path in records.items()}
    resolved = {label: root / "eval" / f"{label}.records.jsonl" for label in DEFAULT_EVAL_LABELS}
    missing = [str(path) for path in resolved.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing eval record files: " + ", ".join(missing))
    return resolved


def _build_eval_tables(
    record_rows: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[OrderedDict[str, object]], list[OrderedDict[str, object]], dict[str, Any]]:
    eval_rows: list[OrderedDict[str, object]] = []
    missing_rows: list[OrderedDict[str, object]] = []
    summaries: list[dict[str, Any]] = []
    for label, rows in record_rows.items():
        summary = aggregate_eval_records(rows, label=label)
        best_records = list(summary.pop("best_records", []))
        summaries.append(summary)
        task_count = int(summary.get("task_count", len(best_records)))
        sample_count = int(summary.get("sample_count", len(rows)))
        pass_count = sum(1 for row in best_records if row.get("all_tests_pass") is True)
        correct_and_faster_count = sum(1 for row in best_records if _correct_and_faster(row))
        missing_runtime_records = [row for row in best_records if _missing_runtime(row)]
        invalid_format_count = sum(1 for row in rows if row.get("reason") == "invalid_format")
        compile_error_count = sum(1 for row in rows if row.get("compile_error") is True)
        sanitizer_error_count = sum(1 for row in rows if row.get("sanitizer_error") is True)
        timeout_count = sum(1 for row in rows if row.get("timeout") is True)
        rewards = [float(row.get("reward", 0.0)) for row in best_records]
        speedups = [
            float(row["reference_runtime_cpu_ns"]) / float(row["runtime_cpu_ns"])
            for row in best_records
            if _correct_and_faster(row) and float(row["runtime_cpu_ns"]) > 0
        ]
        eval_rows.append(
            OrderedDict(
                [
                    ("label", label),
                    ("display_label", DISPLAY_LABELS.get(label, label)),
                    ("sample_count", sample_count),
                    ("task_count", task_count),
                    ("pass_count", pass_count),
                    ("pass_rate", _rate(pass_count, task_count)),
                    ("correct_and_faster_count", correct_and_faster_count),
                    ("correct_and_faster_rate", _rate(correct_and_faster_count, task_count)),
                    ("missing_runtime_count", len(missing_runtime_records)),
                    ("missing_runtime_rate", _rate(len(missing_runtime_records), task_count)),
                    ("invalid_format_count", invalid_format_count),
                    ("invalid_format_rate", _rate(invalid_format_count, sample_count)),
                    ("compile_error_count", compile_error_count),
                    ("compile_error_rate", _rate(compile_error_count, sample_count)),
                    ("sanitizer_error_count", sanitizer_error_count),
                    ("sanitizer_error_rate", _rate(sanitizer_error_count, sample_count)),
                    ("timeout_count", timeout_count),
                    ("timeout_rate", _rate(timeout_count, sample_count)),
                    ("mean_reward_from_records", mean(rewards) if rewards else 0.0),
                    ("mean_best_reward_from_evaluator", float(summary.get("mean_best_reward", 0.0))),
                    ("mean_correct_faster_speedup", mean(speedups) if speedups else 0.0),
                ]
            )
        )
        for row in missing_runtime_records:
            missing_rows.append(
                OrderedDict(
                    [
                        ("label", label),
                        ("display_label", DISPLAY_LABELS.get(label, label)),
                        ("task_id", row.get("task_id", "")),
                        ("problem_id", row.get("problem_id", "")),
                        ("reason", row.get("reason", "")),
                        ("runtime_cpu_ns", _blank_none(row.get("runtime_cpu_ns"))),
                        ("reference_runtime_cpu_ns", _blank_none(row.get("reference_runtime_cpu_ns"))),
                        ("runtime_speedup", _blank_none(row.get("runtime_speedup"))),
                        ("reward", _blank_none(row.get("reward"))),
                        ("tests_passed", _blank_none(row.get("tests_passed"))),
                        ("tests_total", _blank_none(row.get("tests_total"))),
                    ]
                )
            )
    comparison = compare_eval_summaries(summaries)
    return eval_rows, missing_rows, comparison


def _correct_and_faster(row: Mapping[str, Any]) -> bool:
    runtime = row.get("runtime_cpu_ns")
    reference = row.get("reference_runtime_cpu_ns")
    try:
        return row.get("all_tests_pass") is True and runtime is not None and reference is not None and float(runtime) < float(reference)
    except (TypeError, ValueError):
        return False


def _missing_runtime(row: Mapping[str, Any]) -> bool:
    return row.get("all_tests_pass") is True and (
        row.get("runtime_cpu_ns") is None or row.get("reference_runtime_cpu_ns") is None
    )


def _rate(count: int | float, total: int | float) -> float:
    return 0.0 if total == 0 else float(count) / float(total)


def _blank_none(value: object) -> object:
    return "" if value is None else value


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _selected_series(
    metrics_payload: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> dict[str, list[tuple[float, float, int | None]]]:
    series = {}
    if metrics_payload:
        metrics = metrics_payload.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("series"), dict):
            series = metrics["series"]
        elif isinstance(metrics_payload.get("series"), dict):
            series = metrics_payload["series"]
    return {key: _parse_metric_points(series.get(key, [])) for key in keys}


def _parse_metric_points(raw_points: object) -> list[tuple[float, float, int | None]]:
    if not isinstance(raw_points, list):
        return []
    points: list[tuple[float, float, int | None]] = []
    for point in raw_points:
        if isinstance(point, dict):
            step = point.get("step")
            value = point.get("value")
            timestamp = point.get("timestamp_ms", point.get("timestamp"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            step = point[0]
            value = point[1]
            timestamp = point[2] if len(point) >= 3 else None
        else:
            continue
        try:
            step_value = float(step)
            metric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(metric_value) or math.isinf(metric_value):
            continue
        points.append((step_value, metric_value, int(timestamp) if timestamp is not None else None))
    return sorted(points, key=lambda item: item[0])


def _latest_metric_rows(
    series_points: Mapping[str, list[tuple[float, float, int | None]]],
    keys: tuple[str, ...],
) -> list[OrderedDict[str, object]]:
    rows: list[OrderedDict[str, object]] = []
    for key in keys:
        points = series_points.get(key, [])
        if points:
            step, value, timestamp = points[-1]
            rows.append(
                OrderedDict(
                    [
                        ("metric", key),
                        ("point_count", len(points)),
                        ("latest_step", int(step) if step.is_integer() else step),
                        ("latest_value", value),
                        ("timestamp_ms", timestamp if timestamp is not None else ""),
                    ]
                )
            )
        else:
            rows.append(
                OrderedDict(
                    [
                        ("metric", key),
                        ("point_count", 0),
                        ("latest_step", ""),
                        ("latest_value", ""),
                        ("timestamp_ms", ""),
                    ]
                )
            )
    return rows


def _series_csv_rows(
    series_points: Mapping[str, list[tuple[float, float, int | None]]],
    keys: tuple[str, ...],
) -> list[OrderedDict[str, object]]:
    rows: list[OrderedDict[str, object]] = []
    for key in keys:
        for step, value, timestamp in series_points.get(key, []):
            rows.append(
                OrderedDict(
                    [
                        ("metric", key),
                        ("step", int(step) if step.is_integer() else step),
                        ("value", value),
                        ("timestamp_ms", timestamp if timestamp is not None else ""),
                    ]
                )
            )
    return rows


def _metrics_tracking_state(metrics_payload: dict[str, Any] | None) -> dict[str, object]:
    metrics = metrics_payload.get("metrics") if metrics_payload else None
    if not isinstance(metrics, dict):
        return {
            "backend": "",
            "tracking_state": "unavailable",
            "latest_step": "",
            "metric_count": 0,
            "metric_row_count": 0,
        }
    return {
        "backend": metrics.get("backend", ""),
        "tracking_state": metrics.get("tracking_state", ""),
        "latest_step": metrics.get("latest_step", ""),
        "metric_count": metrics.get("metric_count", 0),
        "metric_row_count": metrics.get("metric_row_count", 0),
    }


def _write_csv(path: Path, rows: list[OrderedDict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json_strict(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_eval_charts(asset_root: Path, eval_rows: list[OrderedDict[str, object]]) -> None:
    labels = [str(row["label"]) for row in eval_rows]
    count_values = {}
    rate_values = {}
    for row in eval_rows:
        label = str(row["label"])
        for metric in (
            "pass_count",
            "correct_and_faster_count",
            "invalid_format_count",
            "compile_error_count",
            "missing_runtime_count",
        ):
            count_values[(label, metric)] = float(row[metric])
        for metric in (
            "pass_rate",
            "correct_and_faster_rate",
            "invalid_format_rate",
            "compile_error_rate",
            "missing_runtime_rate",
        ):
            rate_values[(label, metric)] = float(row[metric]) * 100.0
    _grouped_bar_chart(
        asset_root / "eval_outcome_counts.svg",
        "Held-out eval outcome counts",
        "count",
        labels,
        (
            ("pass_count", "pass"),
            ("correct_and_faster_count", "correct+faster"),
            ("invalid_format_count", "invalid format"),
            ("compile_error_count", "compile error"),
            ("missing_runtime_count", "missing runtime"),
        ),
        count_values,
    )
    _grouped_bar_chart(
        asset_root / "eval_outcome_rates.svg",
        "Held-out eval outcome rates",
        "percent",
        labels,
        (
            ("pass_rate", "pass"),
            ("correct_and_faster_rate", "correct+faster"),
            ("invalid_format_rate", "invalid format"),
            ("compile_error_rate", "compile error"),
            ("missing_runtime_rate", "missing runtime"),
        ),
        rate_values,
    )


def _write_metric_charts(
    asset_root: Path,
    series_points: Mapping[str, list[tuple[float, float, int | None]]],
) -> None:
    _line_chart(
        asset_root / "train_reward_pass.svg",
        "GRPO training reward and pass@8",
        "value",
        ("loss/avg_final_rewards", "reward/avg_pass_at_8"),
        series_points,
        {"loss/avg_final_rewards": "avg final reward", "reward/avg_pass_at_8": "pass@8"},
    )
    _line_chart(
        asset_root / "heldout_eval_curves.svg",
        "Held-out eval curves during GRPO",
        "value",
        ("eval/all/avg_score", "eval/all/pass_at_1", "eval/all/mean_positive_reward"),
        series_points,
        {
            "eval/all/avg_score": "avg score",
            "eval/all/pass_at_1": "pass@1",
            "eval/all/mean_positive_reward": "mean positive reward",
        },
    )
    _line_chart(
        asset_root / "policy_entropy.svg",
        "GRPO policy entropy",
        "entropy",
        ("policy/policy_entropy",),
        series_points,
        {"policy/policy_entropy": "policy entropy"},
    )
    _line_chart(
        asset_root / "policy_grad_norm.svg",
        "GRPO gradient norm",
        "grad norm",
        ("policy/grad_norm",),
        series_points,
        {"policy/grad_norm": "grad norm"},
    )
    _line_chart(
        asset_root / "advantage_variance.svg",
        "GRPO advantage and reward variance metrics",
        "value",
        ("loss/avg_raw_advantages_abs", "w8/reward_group_variance_mean", "w8/zero_variance_group_fraction"),
        series_points,
        {
            "loss/avg_raw_advantages_abs": "mean |raw advantage|",
            "w8/reward_group_variance_mean": "reward group variance",
            "w8/zero_variance_group_fraction": "zero-variance group fraction",
        },
    )
    _line_chart(
        asset_root / "response_length.svg",
        "GRPO response length",
        "tokens",
        ("policy/response_length",),
        series_points,
        {"policy/response_length": "response length"},
    )
    _line_chart(
        asset_root / "timing_breakdown.svg",
        "GRPO phase timing curves",
        "seconds",
        (
            "timing/step",
            "timing/generate",
            "timing/policy_train",
            "timing/fwd_logprobs_values_reward",
            "timing/sync_weights",
        ),
        series_points,
        {
            "timing/step": "step",
            "timing/generate": "generate",
            "timing/policy_train": "policy train",
            "timing/fwd_logprobs_values_reward": "fwd/logprobs/value/reward",
            "timing/sync_weights": "sync weights",
        },
    )


def _artifact_paths(
    root: Path,
    record_paths: Mapping[str, Path],
    metrics_path: Path,
    status_path: Path,
    uplift_summary_path: Path,
) -> list[Path]:
    paths: list[Path] = []
    paths.extend(record_paths.values())
    for label in record_paths:
        summary = root / "eval" / f"{label}.summary.json"
        if summary.exists():
            paths.append(summary)
    for path in (status_path, metrics_path, root / "uplift-summary.json", uplift_summary_path):
        if path.exists():
            paths.append(path)
    return paths


def _artifact_hash_rows(paths: list[Path]) -> list[OrderedDict[str, object]]:
    rows: list[OrderedDict[str, object]] = []
    for path in paths:
        rows.append(OrderedDict([("path", str(path)), ("size_bytes", path.stat().st_size), ("sha256", _sha256(path))]))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_sets_equal(record_rows: Mapping[str, list[dict[str, Any]]]) -> bool:
    sets = {tuple(sorted(str(row.get("task_id", "")) for row in rows)) for rows in record_rows.values()}
    return len(sets) <= 1


def _render_raw_markdown(
    *,
    run_id: str,
    root: Path,
    asset_root: Path,
    companion_report: Path | None,
    eval_rows: list[OrderedDict[str, object]],
    missing_rows: list[OrderedDict[str, object]],
    comparison: Mapping[str, Any],
    artifact_rows: list[OrderedDict[str, object]],
    latest_rows: list[OrderedDict[str, object]],
    metrics_state: Mapping[str, object],
    status_payload: Mapping[str, Any] | None,
    task_set_equal: bool,
) -> str:
    lines: list[str] = [
        f"# Raw Run Evidence: `{run_id}`",
        "",
        "This file is data-first. It lists raw artifacts, recomputed aggregate tables, and generated curves without narrative interpretation.",
        "",
    ]
    if companion_report:
        lines.extend([f"Companion interpretation report: [{companion_report.name}]({companion_report.as_posix()})", ""])
    lines.extend(
        [
            "## Source Artifacts",
            "",
            "Run ID:",
            "",
            "```text",
            run_id,
            "```",
            "",
            "Run root:",
            "",
            "```text",
            root.as_posix(),
            "```",
            "",
        ]
    )
    dataset = _status_dataset_prefix(status_payload)
    if dataset:
        lines.extend(["Dataset:", "", "```text", dataset, "```", ""])
    checkpoint = _status_grpo_checkpoint(status_payload)
    if checkpoint:
        lines.extend(["GRPO checkpoint evaluated:", "", "```text", checkpoint, "```", ""])
    lines.extend(
        [
            "Raw asset bundle:",
            "",
            "```text",
            asset_root.as_posix() + "/",
            "```",
            "",
            "## Recompute Command",
            "",
            "```bash",
            "uv run w8-biayn eval cpp \\",
            f"  --records base={root.as_posix()}/eval/base.records.jsonl \\",
            f"  --records sft={root.as_posix()}/eval/sft.records.jsonl \\",
            f"  --records grpo={root.as_posix()}/eval/grpo.records.jsonl \\",
            f"  --out {asset_root.as_posix()}/uplift-summary-recomputed.json",
            "```",
            "",
            "## Eval Summary",
            "",
            f"CSV source: [eval_summary.csv]({(asset_root / 'eval_summary.csv').as_posix()})",
            "",
            "| Label | Samples | Pass Count | Pass Rate | Correct+Faster Count | Correct+Faster Rate | Missing Runtime | Invalid Format | Compile Error | Sanitizer Error | Timeout | Mean Reward | Winner Speedup Mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in eval_rows:
        lines.append(
            "| {display_label} | {sample_count:,} | {pass_count:,} | {pass_rate} | {cf_count:,} | {cf_rate} | "
            "{missing} | {invalid} | {compile_error} | {sanitizer} | {timeout} | {reward} | {speedup}x |".format(
                display_label=row["display_label"],
                sample_count=int(row["sample_count"]),
                pass_count=int(row["pass_count"]),
                pass_rate=_percent(float(row["pass_rate"])),
                cf_count=int(row["correct_and_faster_count"]),
                cf_rate=_percent(float(row["correct_and_faster_rate"])),
                missing=_count_rate(row, "missing_runtime"),
                invalid=_count_rate(row, "invalid_format"),
                compile_error=_count_rate(row, "compile_error"),
                sanitizer=_count_rate(row, "sanitizer_error"),
                timeout=_count_rate(row, "timeout"),
                reward=_decimal(float(row["mean_best_reward_from_evaluator"]), 6),
                speedup=_decimal(float(row["mean_correct_faster_speedup"]), 6),
            )
        )
    lines.extend(
        [
            "",
            "Task-set equality:",
            "",
            "```text",
            f"base task IDs == SFT task IDs == GRPO task IDs: {str(task_set_equal).lower()}",
            "```",
            "",
            "## Eval Curves",
            "",
            f"![Held-out eval outcome rates]({(asset_root / 'eval_outcome_rates.svg').as_posix()})",
            "",
            f"![Held-out eval outcome counts]({(asset_root / 'eval_outcome_counts.svg').as_posix()})",
            "",
            "## Formal Gate Raw JSON",
            "",
            f"Source: [uplift-summary-recomputed.json]({(asset_root / 'uplift-summary-recomputed.json').as_posix()})",
            "",
            "```json",
            json.dumps({"uplift_gate": comparison.get("uplift_gate")}, indent=2, sort_keys=False),
            "```",
            "",
            "## Missing Runtime Rows",
            "",
            f"CSV source: [missing_runtime_tasks.csv]({(asset_root / 'missing_runtime_tasks.csv').as_posix()})",
            "",
        ]
    )
    if missing_rows:
        lines.extend(
            [
                "| Label | Task ID | Problem ID | Reason | Runtime CPU ns | Reference CPU ns | Reward | Tests |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in missing_rows:
            lines.append(
                f"| {row['display_label']} | `{row['task_id']}` | `{row['problem_id']}` | `{row['reason']}` | "
                f"{row['runtime_cpu_ns']} | {row['reference_runtime_cpu_ns']} | {row['reward']} | "
                f"{row['tests_passed']} / {row['tests_total']} |"
            )
    else:
        lines.append("No missing-runtime rows.")
    lines.extend(
        [
            "",
            "## MLflow Tracking Snapshot",
            "",
            f"Source: `{(root / 'metrics.api.json').as_posix()}`",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Backend | `{metrics_state.get('backend', '')}` |",
            f"| Tracking state | `{metrics_state.get('tracking_state', '')}` |",
            f"| Latest scalar step | {metrics_state.get('latest_step', '')} |",
            f"| Metric count | {metrics_state.get('metric_count', 0)} |",
            f"| Metric row count | {metrics_state.get('metric_row_count', 0)} |",
            "",
            "Selected latest metrics:",
            "",
            "| Metric | Points | Latest Step | Latest Value |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in latest_rows:
        lines.append(
            f"| `{row['metric']}` | {row['point_count']} | {row['latest_step']} | {_metric_value(row['latest_value'])} |"
        )
    lines.extend(
        [
            "",
            "## MLflow Curves",
            "",
            f"![GRPO training reward and pass@8]({(asset_root / 'train_reward_pass.svg').as_posix()})",
            "",
            f"![Held-out eval curves during GRPO]({(asset_root / 'heldout_eval_curves.svg').as_posix()})",
            "",
            f"![GRPO policy entropy]({(asset_root / 'policy_entropy.svg').as_posix()})",
            "",
            f"![GRPO gradient norm]({(asset_root / 'policy_grad_norm.svg').as_posix()})",
            "",
            f"![GRPO advantage and reward variance metrics]({(asset_root / 'advantage_variance.svg').as_posix()})",
            "",
            f"![GRPO response length]({(asset_root / 'response_length.svg').as_posix()})",
            "",
            f"![GRPO phase timing curves]({(asset_root / 'timing_breakdown.svg').as_posix()})",
            "",
            "## Artifact Hashes",
            "",
            f"CSV source: [artifact_hashes.csv]({(asset_root / 'artifact_hashes.csv').as_posix()})",
            "",
            "| Artifact | Size | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for row in artifact_rows:
        lines.append(f"| `{row['path']}` | {int(row['size_bytes']):,} | `{row['sha256']}` |")
    lines.extend(["", "## Raw Bundle Index", "", "```text"])
    for path in sorted(asset_root.iterdir()):
        if path.is_file():
            lines.append(path.as_posix())
    lines.extend(["```", ""])
    return "\n".join(lines)


def _status_dataset_prefix(status_payload: Mapping[str, Any] | None) -> str:
    if not status_payload:
        return ""
    dataset = status_payload.get("dataset")
    if isinstance(dataset, dict):
        value = dataset.get("gcs_prefix")
        if isinstance(value, str):
            return value
    return ""


def _status_grpo_checkpoint(status_payload: Mapping[str, Any] | None) -> str:
    if not status_payload:
        return ""
    pipelines = status_payload.get("pipelines")
    if not isinstance(pipelines, list):
        return ""
    for pipeline in pipelines:
        if not isinstance(pipeline, dict) or pipeline.get("pipeline") != "cpp-grpo":
            continue
        artifacts = pipeline.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        checkpoint = artifacts.get("checkpoint")
        if not isinstance(checkpoint, dict):
            continue
        latest = checkpoint.get("latest") or checkpoint.get("highest")
        if isinstance(latest, dict) and isinstance(latest.get("prefix"), str):
            return latest["prefix"]
    return ""


def _count_rate(row: Mapping[str, object], prefix: str) -> str:
    count = int(row[f"{prefix}_count"])
    total = int(row["sample_count"] if prefix in {"invalid_format", "compile_error", "sanitizer_error", "timeout"} else row["task_count"])
    rate_key = f"{prefix}_rate"
    return f"{count:,} / {total:,} ({_percent(float(row[rate_key]))})"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _decimal(value: float, places: int) -> str:
    return f"{value:.{places}f}"


def _metric_value(value: object) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.10f}"
    except (TypeError, ValueError):
        return str(value)


def _svg_escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _fmt_tick(value: float) -> str:
    abs_value = abs(value)
    if value == 0:
        return "0"
    if abs_value < 0.001 or abs_value >= 10000:
        return f"{value:.1e}"
    if abs_value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if abs_value < 100:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def _write_svg(path: Path, body: str, width: int, height: int) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        f"{body}\n</svg>\n",
        encoding="utf-8",
    )


def _line_chart(
    path: Path,
    title: str,
    y_label: str,
    metric_keys: tuple[str, ...],
    series_points: Mapping[str, list[tuple[float, float, int | None]]],
    display: Mapping[str, str],
) -> None:
    width, height = 960, 420
    left, right, top, bottom = 82, 26, 52, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    datasets = []
    for index, key in enumerate(metric_keys):
        points = series_points.get(key, [])
        if points:
            datasets.append((display.get(key, key), points, CHART_COLORS[index % len(CHART_COLORS)]))
    if not datasets:
        _write_svg(
            path,
            f'<text x="20" y="40" font-family="Arial" font-size="18">{_svg_escape(title)}: no data</text>',
            width,
            height,
        )
        return
    xs = [x for _, points, _ in datasets for x, _, _ in points]
    ys = [y for _, points, _ in datasets for _, y, _ in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        delta = abs(y_min) * 0.1 or 1.0
        y_min -= delta
        y_max += delta
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    def xmap(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def ymap(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = _chart_axes(width, height, left, top, plot_width, plot_height, title, y_label, x_min, x_max, y_min, y_max)
    legend_x = left + 8
    legend_y = top + 12
    for index, (name, points, color) in enumerate(datasets):
        coords = " ".join(f"{xmap(x):.1f},{ymap(y):.1f}" for x, y, _ in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for x, y, _ in (points[0], points[-1]):
            parts.append(f'<circle cx="{xmap(x):.1f}" cy="{ymap(y):.1f}" r="3" fill="{color}"/>')
        ly = legend_y + index * 19
        parts.append(f'<rect x="{legend_x}" y="{ly - 9}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 18}" y="{ly + 1}" font-family="Arial" font-size="12">{_svg_escape(name)}</text>')
    _write_svg(path, "\n".join(parts), width, height)


def _chart_axes(
    width: int,
    height: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    title: str,
    y_label: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[str]:
    parts = [
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_svg_escape(title)}</text>',
        f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13">step</text>',
        f'<text transform="translate(20 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{_svg_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
    ]
    for i in range(6):
        x = left + plot_width * i / 5
        step = x_min + (x_max - x_min) * i / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#eee"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="Arial" font-size="11">{_fmt_tick(step)}</text>'
        )
    for i in range(6):
        y = top + plot_height * i / 5
        value = y_max - (y_max - y_min) * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{_fmt_tick(value)}</text>'
        )
    return parts


def _grouped_bar_chart(
    path: Path,
    title: str,
    y_label: str,
    groups: list[str],
    metrics: tuple[tuple[str, str], ...],
    values: Mapping[tuple[str, str], float],
) -> None:
    width, height = 960, 440
    left, right, top, bottom = 82, 28, 52, 96
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max([0.0] + [values.get((group, metric), 0.0) for group in groups for metric, _ in metrics])
    y_max = max_value * 1.12 if max_value else 1.0
    parts = [
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{_svg_escape(title)}</text>',
        f'<text transform="translate(20 {top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13">{_svg_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#333"/>',
    ]
    for i in range(6):
        y = top + plot_height * i / 5
        value = y_max - y_max * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{_fmt_tick(value)}</text>'
        )
    group_width = plot_width / max(1, len(groups))
    bar_gap = 4
    bar_width = max(8.0, min(30.0, (group_width - 38) / len(metrics) - bar_gap))
    for group_index, group in enumerate(groups):
        center_x = left + group_width * group_index + group_width / 2
        total_width = len(metrics) * bar_width + (len(metrics) - 1) * bar_gap
        start_x = center_x - total_width / 2
        parts.append(
            f'<text x="{center_x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial" font-size="13">{_svg_escape(DISPLAY_LABELS.get(group, group))}</text>'
        )
        for metric_index, (metric, _) in enumerate(metrics):
            value = values.get((group, metric), 0.0)
            height_value = 0.0 if y_max == 0 else value / y_max * plot_height
            x = start_x + metric_index * (bar_width + bar_gap)
            y = top + plot_height - height_value
            color = CHART_COLORS[metric_index % len(CHART_COLORS)]
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height_value:.1f}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{max(top + 12, y - 4):.1f}" text-anchor="middle" font-family="Arial" font-size="10">{_fmt_tick(value)}</text>'
            )
    legend_x = left + 6
    legend_y = height - 45
    for metric_index, (_, metric_name) in enumerate(metrics):
        x = legend_x + metric_index * 170
        color = CHART_COLORS[metric_index % len(CHART_COLORS)]
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 18}" y="{legend_y}" font-family="Arial" font-size="12">{_svg_escape(metric_name)}</text>')
    _write_svg(path, "\n".join(parts), width, height)
