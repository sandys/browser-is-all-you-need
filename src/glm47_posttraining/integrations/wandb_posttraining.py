"""Canonical W&B observability for PIE C++ post-training runs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from glm47_posttraining.cpp_perf.eval import compare_eval_summaries

MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
ALLOWED_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
}
SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|token|secret|password|credential|"
    r"private[_-]?key|ssh[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
SENSITIVE_PATH_RE = re.compile(
    r"(?:^|[._-])(?:api[_-]?keys?|credentials?|netrc|passwords?|private[_-]?key|"
    r"secrets?|tokens?)(?:$|[._-])",
    re.IGNORECASE,
)
SENSITIVE_CONTENT_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|auth[_-]?token|token[_-]?secret|"
    r"secret|password|credential|private[_-]?key|ssh[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
URL_CREDENTIAL_RE = re.compile(r"(://)[^/@\s:]+:[^/@\s]+@")
ASSIGNMENT_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_.-]*)\s*([=:])\s*([^\s,;]+)")

EVAL_TABLE_COLUMNS = (
    "experiment_id",
    "timing_status",
    "label",
    "task_id",
    "problem_id",
    "split",
    "sample_index",
    "reason",
    "reward",
    "all_tests_pass",
    "tests_passed",
    "tests_total",
    "compile_error",
    "sanitizer_error",
    "thread_sanitizer_ran",
    "thread_sanitizer_error",
    "thread_sanitizer_timeout",
    "timeout",
    "reasoning_tokens",
    "rubric_category",
    "active_rubrics",
    "rubric_weights",
    "rubric_correctness",
    "rubric_reasoning",
    "rubric_memory_safety",
    "rubric_thread_safety",
    "rubric_runtime",
    "rubric_cpp_quality",
    "runtime_cpu_ns",
    "reference_runtime_cpu_ns",
    "runtime_speedup",
    "completion_tokens",
    "prompt_tokens",
    "truncated",
    "finish_reason",
    "response_preview",
)

FAILURE_TABLE_COLUMNS = (
    "experiment_id",
    "label",
    "split",
    "bucket",
    "count",
    "rate",
)

COMPARISON_TABLE_COLUMNS = (
    "experiment_id",
    "timing_status",
    "label",
    "task_count",
    "sample_count",
    "pass_rate",
    "valid_format_rate",
    "correct_and_faster_rate",
    "mean_best_reward",
    "mean_correct_faster_speedup",
    "missing_runtime_count",
    "missing_runtime_rate",
    "compile_error_rate",
    "sanitizer_error_rate",
    "timeout_rate",
    "truncated_ratio",
    "mean_completion_tokens",
)

PIPELINE_TABLE_COLUMNS = (
    "experiment_id",
    "event_time_unix",
    "stage",
    "event",
    "status",
    "wall_s",
    "repo_sha",
    "image",
    "receipt",
    "error",
)

MILES_SAMPLE_TABLE_COLUMNS = (
    "experiment_id",
    "timing_status",
    "stage",
    "rollout_id",
    "group_index",
    "sample_index",
    "task_id",
    "problem_id",
    "split",
    "label",
    "status",
    "reason",
    "reward",
    "all_tests_pass",
    "format_valid",
    "compile_error",
    "sanitizer_error",
    "thread_sanitizer_ran",
    "thread_sanitizer_error",
    "thread_sanitizer_timeout",
    "timeout",
    "reasoning_tokens",
    "rubric_category",
    "active_rubrics",
    "rubric_weights",
    "rubric_correctness",
    "rubric_reasoning",
    "rubric_memory_safety",
    "rubric_thread_safety",
    "rubric_runtime",
    "rubric_cpp_quality",
    "tests_passed",
    "tests_total",
    "runtime_cpu_ns",
    "reference_runtime_cpu_ns",
    "runtime_speedup",
    "response_length",
    "truncated",
    "cached_tokens",
    "prompt_tokens",
    "prompt_preview",
    "response_preview",
)

MILES_REWARD_OUTCOME_TABLE_COLUMNS = (
    "experiment_id",
    "stage",
    "reason",
    "count",
    "rate",
    "mean_reward",
    "all_tests_pass_rate",
    "compile_error_rate",
    "sanitizer_error_rate",
    "timeout_rate",
)

MILES_METRIC_TABLE_COLUMNS = (
    "experiment_id",
    "timing_status",
    "family",
    "source_step",
    "metric",
    "value",
)

MILES_SYNC_TABLE_COLUMNS = (
    "experiment_id",
    "sync",
    "rank",
    "n_tensors",
    "sha256",
    "total_sum_abs",
    "matches_sync_rank0",
    "changed_from_previous_sync",
)

MILES_CHECKPOINT_TABLE_COLUMNS = (
    "experiment_id",
    "checkpoint_root",
    "path",
    "size_bytes",
    "sha256",
)

MILES_METRIC_LINE_RE = re.compile(
    r"\s-\s(?P<family>rollout|passrate|eval|step|perf)\s+"
    r"(?P<step>\d+):\s+(?P<payload>\{.*\})\s*$"
)
NUMPY_SCALAR_RE = re.compile(r"np\.float(?:16|32|64)\(([^()]*)\)")

CURATED_STAGE_METRIC_TERMS = (
    "accuracy",
    "correct_and_faster",
    "entropy",
    "grad_norm",
    "kl",
    "learning_rate",
    "loss",
    "mfu",
    "pass_rate",
    "passrate",
    "response_length",
    "reward",
    "throughput",
    "tokens_per",
)

AIDER_METRIC_CATALOG = {
    "eval/pass_at_1": {"summary": "max", "goal": "maximize"},
    "eval/pass_at_2": {"summary": "max", "goal": "maximize"},
    "eval/assertion_pass_ratio": {"summary": "max", "goal": "maximize"},
    "eval/context_exhausted_rate": {"summary": "min", "goal": "minimize"},
    "eval/valid_format_rate": {"summary": "max", "goal": "maximize"},
    "eval/compile_success_rate": {"summary": "max", "goal": "maximize"},
    "eval/sanitizer_pass_rate": {"summary": "max", "goal": "maximize"},
    "eval/thread_sanitizer_error_rate": {"summary": "min", "goal": "minimize"},
    "eval/thread_sanitizer_timeout_rate": {"summary": "min", "goal": "minimize"},
    "eval/mean_rubric_correctness": {"summary": "max", "goal": "maximize"},
    "eval/mean_rubric_reasoning": {"summary": "max", "goal": "maximize"},
    "eval/mean_rubric_memory_safety": {"summary": "max", "goal": "maximize"},
    "eval/mean_rubric_thread_safety": {"summary": "max", "goal": "maximize"},
    "eval/mean_rubric_runtime": {"summary": "max", "goal": "maximize"},
    "eval/mean_rubric_cpp_quality": {"summary": "max", "goal": "maximize"},
    "eval/pass_rate_easy": {"summary": "max", "goal": "maximize"},
    "eval/pass_rate_medium": {"summary": "max", "goal": "maximize"},
    "eval/pass_rate_hard": {"summary": "max", "goal": "maximize"},
    "eval/pass_rate_state_concurrency": {"summary": "max", "goal": "maximize"},
    "eval/pass_rate_time_date": {"summary": "max", "goal": "maximize"},
    "eval/pass_rate_numerical": {"summary": "max", "goal": "maximize"},
}


def register_aider_wandb_metrics(run: Any) -> None:
    """Register summary direction for the Polyglot evaluation catalog."""

    for metric_name, config in AIDER_METRIC_CATALOG.items():
        _define_metric(run, metric_name, **config)


def safe_identifier(value: str, *, fallback: str = "run") -> str:
    """Return a W&B-safe stable identifier."""

    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.")
    return (cleaned or fallback)[:120]


def resolve_experiment_id(*, explicit: str = "", run_id: str = "", label: str = "") -> str:
    value = explicit or os.environ.get("GLM47_EXPERIMENT_ID", "") or run_id or label
    return safe_identifier(value, fallback="pie-cpp-posttraining")


def redact_sensitive(value: Any, *, key: str = "") -> Any:
    """Recursively redact secrets while preserving metric and config structure."""

    if key and SENSITIVE_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item_key): redact_sensitive(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        redacted = URL_CREDENTIAL_RE.sub(r"\1<redacted>@", value)
        return ASSIGNMENT_RE.sub(_redact_assignment, redacted)
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object row: {path}")
            rows.append(row)
    return rows


def read_key_value(path: str | Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            values[key] = _coerce_scalar(value)
    return values


def build_eval_table_rows(
    records: Iterable[dict[str, Any]],
    generations: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
    timing_status: str,
) -> list[list[Any]]:
    generation_by_key = {_sample_key(row): row for row in generations}
    rows: list[list[Any]] = []
    for record in records:
        generation = generation_by_key.get(_sample_key(record), {})
        rows.append(
            [
                experiment_id,
                timing_status,
                record.get("label") or generation.get("label"),
                record.get("task_id") or generation.get("task_id"),
                record.get("problem_id") or generation.get("problem_id"),
                record.get("split") or generation.get("split"),
                _int_or_none(record.get("sample_index", generation.get("sample_index"))),
                record.get("reason"),
                _number_or_none(record.get("reward")),
                record.get("all_tests_pass"),
                _int_or_none(record.get("tests_passed")),
                _int_or_none(record.get("tests_total")),
                bool(record.get("compile_error", False)),
                bool(record.get("sanitizer_error", False)),
                bool(record.get("thread_sanitizer_ran", False)),
                bool(record.get("thread_sanitizer_error", False)),
                bool(record.get("thread_sanitizer_timeout", False)),
                bool(record.get("timeout", False)),
                _int_or_none(record.get("reasoning_tokens")),
                record.get("rubric_category"),
                json.dumps(record.get("active_rubrics", []), sort_keys=True),
                json.dumps(record.get("rubric_weights", {}), sort_keys=True),
                _number_or_none(record.get("rubric_correctness")),
                _number_or_none(record.get("rubric_reasoning")),
                _number_or_none(record.get("rubric_memory_safety")),
                _number_or_none(record.get("rubric_thread_safety")),
                _number_or_none(record.get("rubric_runtime")),
                _number_or_none(record.get("rubric_cpp_quality")),
                _int_or_none(record.get("runtime_cpu_ns")),
                _int_or_none(record.get("reference_runtime_cpu_ns")),
                _number_or_none(record.get("runtime_speedup")),
                _int_or_none(generation.get("completion_tokens")),
                _int_or_none(generation.get("prompt_tokens")),
                generation.get("truncated"),
                generation.get("finish_reason"),
                _preview(generation.get("response")),
            ]
        )
    return rows


def build_failure_bucket_rows(
    records: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
) -> list[list[Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for record in records:
        label = str(record.get("label") or "unknown")
        split = str(record.get("split") or "unknown")
        bucket = _failure_bucket(record)
        counts[(label, split, bucket)] += 1
        totals[(label, split)] += 1
    return [
        [
            experiment_id,
            label,
            split,
            bucket,
            count,
            count / totals[(label, split)] if totals[(label, split)] else 0.0,
        ]
        for (label, split, bucket), count in sorted(counts.items())
    ]


def build_comparison_table_rows(
    summaries: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
    timing_status: str,
) -> list[list[Any]]:
    return [
        [
            experiment_id,
            timing_status,
            summary.get("label"),
            summary.get("task_count"),
            summary.get("sample_count"),
            summary.get("pass_rate"),
            summary.get("valid_format_rate"),
            summary.get("correct_and_faster_rate"),
            summary.get("mean_best_reward"),
            summary.get("mean_correct_faster_speedup"),
            summary.get("missing_runtime_count"),
            summary.get("missing_runtime_rate"),
            summary.get("compile_error_rate"),
            summary.get("sanitizer_error_rate"),
            summary.get("timeout_rate"),
            summary.get("truncated_ratio"),
            summary.get("mean_completion_tokens"),
        ]
        for summary in summaries
    ]


def parse_miles_metric_events(path: str | Path) -> list[dict[str, Any]]:
    """Recover numeric Miles metric events from a preserved console log."""

    source = Path(path)
    if not source.is_file():
        return []
    events: list[dict[str, Any]] = []
    seen: set[tuple[str, int, tuple[tuple[str, int | float], ...]]] = set()
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = MILES_METRIC_LINE_RE.search(line.rstrip())
            if not match:
                continue
            payload_text = NUMPY_SCALAR_RE.sub(r"\1", match.group("payload"))
            try:
                raw_payload = ast.literal_eval(payload_text)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(raw_payload, dict):
                continue
            metrics = {
                str(key): value
                for key, value in raw_payload.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if not metrics:
                continue
            family = match.group("family")
            step = int(match.group("step"))
            fingerprint = (family, step, tuple(sorted(metrics.items())))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            events.append({"family": family, "step": step, "metrics": metrics})
    return events


def build_miles_metric_table_rows(
    events: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
    timing_status: str,
) -> list[list[Any]]:
    return [
        [experiment_id, timing_status, event["family"], event["step"], metric, value]
        for event in events
        for metric, value in sorted(event["metrics"].items())
    ]


def load_miles_sample_evidence(
    dump_dir: str | Path,
    *,
    experiment_id: str,
    timing_status: str,
    receipt: dict[str, Any],
    max_rows_per_table: int,
) -> tuple[dict[str, list[list[Any]]], dict[str, int]]:
    """Load repo-generated Miles debug dumps into bounded public table rows."""

    root = Path(dump_dir)
    rows: dict[str, list[list[Any]]] = {"rollout": [], "eval": []}
    totals = {"rollout": 0, "eval": 0}
    if not root.is_dir():
        return rows, totals
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to read Miles rollout evidence") from exc

    for path in sorted(root.glob("*.pt")):
        stage = "eval" if "_eval_" in path.name else "rollout"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            continue
        samples = payload["samples"]
        totals[stage] += len(samples)
        remaining = max(0, max_rows_per_table - len(rows[stage]))
        if not remaining:
            continue
        response_limit_key = (
            "eval_max_response_len" if stage == "eval" else "rollout_max_response_len"
        )
        rows[stage].extend(
            build_miles_sample_table_rows(
                samples[:remaining],
                experiment_id=experiment_id,
                timing_status=timing_status,
                stage=stage,
                rollout_id=_int_or_none(payload.get("rollout_id")),
                response_limit=_int_or_none(receipt.get(response_limit_key)),
            )
        )
    return rows, totals


def build_miles_sample_table_rows(
    samples: Iterable[dict[str, Any]],
    *,
    experiment_id: str,
    timing_status: str,
    stage: str,
    rollout_id: int | None,
    response_limit: int | None,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for sample in samples:
        reward_value = sample.get("reward")
        reward = reward_value if isinstance(reward_value, dict) else {"reward": reward_value}
        metadata_value = sample.get("metadata")
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        prefix_value = sample.get("prefix_cache_info")
        prefix = prefix_value if isinstance(prefix_value, dict) else {}
        response_length = _int_or_none(sample.get("response_length"))
        rows.append(
            [
                experiment_id,
                timing_status,
                stage,
                rollout_id,
                _int_or_none(sample.get("group_index")),
                _int_or_none(sample.get("index")),
                reward.get("task_id") or metadata.get("task_id"),
                reward.get("problem_id") or metadata.get("problem_id"),
                reward.get("split") or metadata.get("split"),
                sample.get("label"),
                sample.get("status"),
                reward.get("reason"),
                _number_or_none(reward.get("reward", reward.get("score"))),
                reward.get("all_tests_pass"),
                reward.get("format_valid"),
                bool(reward.get("compile_error", False)),
                bool(reward.get("sanitizer_error", False)),
                bool(reward.get("thread_sanitizer_ran", False)),
                bool(reward.get("thread_sanitizer_error", False)),
                bool(reward.get("thread_sanitizer_timeout", False)),
                bool(reward.get("timeout", False)),
                _int_or_none(reward.get("reasoning_tokens")),
                reward.get("rubric_category"),
                json.dumps(reward.get("active_rubrics", []), sort_keys=True),
                json.dumps(reward.get("rubric_weights", {}), sort_keys=True),
                _number_or_none(reward.get("rubric_correctness")),
                _number_or_none(reward.get("rubric_reasoning")),
                _number_or_none(reward.get("rubric_memory_safety")),
                _number_or_none(reward.get("rubric_thread_safety")),
                _number_or_none(reward.get("rubric_runtime")),
                _number_or_none(reward.get("rubric_cpp_quality")),
                _int_or_none(reward.get("tests_passed")),
                _int_or_none(reward.get("tests_total")),
                _int_or_none(reward.get("runtime_cpu_ns")),
                _int_or_none(reward.get("reference_runtime_cpu_ns")),
                _number_or_none(reward.get("runtime_speedup")),
                response_length,
                response_length >= response_limit
                if response_length is not None and response_limit
                else None,
                _int_or_none(prefix.get("cached_tokens")),
                _int_or_none(prefix.get("total_prompt_tokens")),
                _preview(sample.get("prompt"), limit=1000),
                _preview(sample.get("response"), limit=2000),
            ]
        )
    return rows


def build_miles_reward_outcome_rows(
    sample_rows: Iterable[list[Any]],
    *,
    experiment_id: str,
) -> list[list[Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        record = dict(zip(MILES_SAMPLE_TABLE_COLUMNS, row))
        grouped[(str(record["stage"]), str(record["reason"] or "unknown"))].append(record)

    result: list[list[Any]] = []
    for (stage, reason), records in sorted(grouped.items()):
        stage_total = sum(
            len(items) for (item_stage, _), items in grouped.items() if item_stage == stage
        )
        rewards = [
            value for record in records if (value := _number_or_none(record["reward"])) is not None
        ]
        result.append(
            [
                experiment_id,
                stage,
                reason,
                len(records),
                len(records) / stage_total if stage_total else 0.0,
                sum(rewards) / len(rewards) if rewards else None,
                _boolean_rate(records, "all_tests_pass"),
                _boolean_rate(records, "compile_error"),
                _boolean_rate(records, "sanitizer_error"),
                _boolean_rate(records, "timeout"),
            ]
        )
    return result


def build_miles_sync_evidence_rows(
    sync_dir: str | Path,
    *,
    experiment_id: str,
) -> tuple[list[list[Any]], dict[str, Any]]:
    root = Path(sync_dir)
    payloads: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("sync*_rank*.json")):
            payload = read_json(path)
            if all(
                key in payload for key in ("sync", "rank", "n_tensors", "sha256", "total_sum_abs")
            ):
                payloads.append(payload)
    rank0_hashes = {
        int(payload["sync"]): str(payload["sha256"])
        for payload in payloads
        if int(payload["rank"]) == 0
    }
    sorted_syncs = sorted(rank0_hashes)
    previous_hash = {
        sync: rank0_hashes[sorted_syncs[index - 1]] if index else None
        for index, sync in enumerate(sorted_syncs)
    }
    rows = [
        [
            experiment_id,
            int(payload["sync"]),
            int(payload["rank"]),
            int(payload["n_tensors"]),
            str(payload["sha256"]),
            float(payload["total_sum_abs"]),
            str(payload["sha256"]) == rank0_hashes.get(int(payload["sync"])),
            (
                str(payload["sha256"]) != previous_hash[int(payload["sync"])]
                if previous_hash.get(int(payload["sync"])) is not None
                else None
            ),
        ]
        for payload in sorted(payloads, key=lambda item: (int(item["sync"]), int(item["rank"])))
    ]
    per_sync_hashes = {
        sync: {str(payload["sha256"]) for payload in payloads if int(payload["sync"]) == sync}
        for sync in sorted_syncs
    }
    summary: dict[str, Any] = {
        "sync/records": len(rows),
        "sync/all_ranks_match": bool(rows)
        and all(len(hashes) == 1 for hashes in per_sync_hashes.values()),
    }
    if sorted_syncs:
        summary["sync/hash_before"] = rank0_hashes[sorted_syncs[0]]
        summary["sync/hash_after"] = rank0_hashes[sorted_syncs[-1]]
        summary["sync/updated_after_train"] = (
            len(sorted_syncs) > 1
            and rank0_hashes[sorted_syncs[0]] != rank0_hashes[sorted_syncs[-1]]
        )
    return rows, summary


def write_miles_checkpoint_manifest(
    output_path: str | Path,
    checkpoint_dir: str | Path,
    *,
    experiment_id: str,
) -> tuple[Path, list[list[Any]]]:
    root = Path(checkpoint_dir)
    files = _latest_checkpoint_files(root)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_root": str(root),
                "latest_iteration": _latest_iteration_name(root),
                "files": entries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        [experiment_id, str(root), entry["path"], entry["size_bytes"], entry["sha256"]]
        for entry in entries
    ]
    return output, rows


def select_artifact_paths(paths: Iterable[str | Path]) -> tuple[list[Path], list[dict[str, str]]]:
    selected: list[Path] = []
    skipped: list[dict[str, str]] = []
    seen: set[Path] = set()
    for item in paths:
        path = Path(item)
        if path in seen:
            continue
        seen.add(path)
        reason = _artifact_skip_reason(path)
        if reason:
            skipped.append({"name": path.name, "reason": reason})
        else:
            selected.append(path)
    return selected, skipped


def write_artifact_manifest(
    output_path: str | Path,
    paths: Iterable[str | Path],
) -> tuple[Path, list[Path]]:
    selected, skipped = select_artifact_paths(paths)
    payload = {
        "schema_version": 1,
        "files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in selected
        ],
        "skipped": skipped,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, selected


def log_eval_run(
    wandb_module: Any,
    *,
    project: str,
    entity: str | None,
    experiment_id: str,
    run_id: str,
    name: str,
    group: str,
    job_type: str,
    mode: str,
    timing_status: str,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    config: dict[str, Any],
    artifact_paths: Iterable[str | Path],
    manifest_dir: str | Path,
    tags: Iterable[str] = (),
) -> dict[str, str]:
    safe_experiment = resolve_experiment_id(explicit=experiment_id, run_id=run_id)
    safe_run = safe_identifier(run_id, fallback=f"{safe_experiment}-eval")
    label = str(summary.get("label") or config.get("label") or "eval")
    run = wandb_module.init(
        project=project,
        entity=entity or None,
        id=safe_run,
        name=name or safe_run,
        group=group or safe_experiment,
        job_type=job_type or "eval",
        mode=mode,
        resume="allow",
        tags=sorted(
            set(
                [
                    "canonical",
                    "aider-polyglot-cpp" if "pass_at_1" in summary else "pie-cpp",
                    label,
                    timing_status,
                    *tags,
                ]
            )
        ),
        config=redact_sensitive(
            {
                **config,
                "experiment_id": safe_experiment,
                "timing_status": timing_status,
                "proof_surface_schema": 1,
            }
        ),
    )
    _define_metric(run, "eval/index")
    _define_metric(run, "eval/*", step_metric="eval/index")
    if "pass_at_1" in summary:
        register_aider_wandb_metrics(run)

    scalar_metrics = {
        key: value
        for key, value in summary.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    payload: dict[str, Any] = {
        "eval/index": 0,
        **{f"eval/{key}": value for key, value in scalar_metrics.items()},
        "tables/eval_samples": wandb_module.Table(
            columns=list(EVAL_TABLE_COLUMNS),
            data=build_eval_table_rows(
                records,
                generations,
                experiment_id=safe_experiment,
                timing_status=timing_status,
            ),
        ),
        "tables/failure_buckets": wandb_module.Table(
            columns=list(FAILURE_TABLE_COLUMNS),
            data=build_failure_bucket_rows(records, experiment_id=safe_experiment),
        ),
    }
    run.log(payload)
    _set_summary(
        run,
        {
            **summary,
            "observability/experiment_id": safe_experiment,
            "observability/schema_version": 1,
            "observability/timing_status": timing_status,
            "stage/job_type": job_type or "eval",
            "stage/status": summary.get("status") or "success",
        },
    )

    manifest_path, selected = write_artifact_manifest(
        Path(manifest_dir) / f"{safe_run}.artifact_manifest.json",
        artifact_paths,
    )
    artifact = wandb_module.Artifact(
        safe_identifier(f"{safe_experiment}-{label}-eval"),
        type="eval",
        metadata=redact_sensitive(
            {
                "experiment_id": safe_experiment,
                "label": label,
                "timing_status": timing_status,
            }
        ),
    )
    for path in [*selected, manifest_path]:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    result = {"run_id": str(getattr(run, "id", safe_run)), "url": str(getattr(run, "url", ""))}
    run.finish()
    return result


def log_comparison_run(
    wandb_module: Any,
    *,
    project: str,
    entity: str | None,
    experiment_id: str,
    run_id: str,
    mode: str,
    timing_status: str,
    summaries: list[dict[str, Any]],
    summary_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, str]:
    safe_experiment = resolve_experiment_id(explicit=experiment_id, run_id=run_id)
    safe_run = safe_identifier(run_id, fallback=f"{safe_experiment}-comparison")
    comparison = compare_eval_summaries(summaries)
    comparison["experiment_id"] = safe_experiment
    comparison["timing_status"] = timing_status
    output_path = Path(output_dir) / f"{safe_run}.comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run = wandb_module.init(
        project=project,
        entity=entity or None,
        id=safe_run,
        name=safe_run,
        group=safe_experiment,
        job_type="comparison",
        mode=mode,
        resume="allow",
        tags=["canonical", "pie-cpp", "comparison", timing_status],
        config={
            "experiment_id": safe_experiment,
            "timing_status": timing_status,
            "labels": [summary.get("label") for summary in summaries],
            "proof_surface_schema": 1,
        },
    )
    payload: dict[str, Any] = {
        "tables/checkpoint_comparison": wandb_module.Table(
            columns=list(COMPARISON_TABLE_COLUMNS),
            data=build_comparison_table_rows(
                summaries,
                experiment_id=safe_experiment,
                timing_status=timing_status,
            ),
        )
    }
    for summary in summaries:
        label = safe_identifier(str(summary.get("label") or "unknown"))
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[f"comparison/{label}/{key}"] = value
    run.log(payload)
    _set_summary(
        run,
        {
            "observability/experiment_id": safe_experiment,
            "observability/timing_status": timing_status,
            "comparison/uplift_gate": comparison.get("uplift_gate"),
            "comparison/best_correct_and_faster": comparison.get("best_correct_and_faster"),
            "comparison/best_mean_reward": comparison.get("best_mean_reward"),
        },
    )
    manifest_path, selected = write_artifact_manifest(
        Path(output_dir) / f"{safe_run}.artifact_manifest.json",
        [*summary_paths, output_path],
    )
    artifact = wandb_module.Artifact(
        safe_identifier(f"{safe_experiment}-comparison"),
        type="comparison",
        metadata={"experiment_id": safe_experiment, "timing_status": timing_status},
    )
    for path in [*selected, manifest_path]:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    result = {"run_id": str(getattr(run, "id", safe_run)), "url": str(getattr(run, "url", ""))}
    run.finish()
    return result


def log_stage_finalization(
    wandb_module: Any,
    *,
    project: str,
    entity: str | None,
    experiment_id: str,
    run_id: str,
    group: str,
    stage: str,
    status: str,
    mode: str,
    timing_status: str,
    receipt: str | Path,
    artifact_paths: Iterable[str | Path],
    manifest_dir: str | Path,
    run_log: str | Path | None = None,
    rollout_dump_dir: str | Path | None = None,
    sync_metrics_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    max_table_rows: int = 5000,
) -> dict[str, str]:
    safe_experiment = resolve_experiment_id(explicit=experiment_id, run_id=run_id)
    safe_run = safe_identifier(run_id, fallback=f"{safe_experiment}-{stage}")
    safe_stage = safe_identifier(stage, fallback="stage")
    receipt_path = Path(receipt)
    receipt_values = read_key_value(receipt_path)
    output_dir = Path(manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_table_rows = max(1, max_table_rows)

    log_path = Path(run_log) if run_log else Path(str(receipt_values.get("log_file") or ""))
    metric_events = parse_miles_metric_events(log_path) if str(log_path) else []
    latest_metrics: dict[str, int | float] = {}
    for event in metric_events:
        latest_metrics.update(event["metrics"])

    sample_rows = {"rollout": [], "eval": []}
    sample_totals = {"rollout": 0, "eval": 0}
    if rollout_dump_dir:
        sample_rows, sample_totals = load_miles_sample_evidence(
            rollout_dump_dir,
            experiment_id=safe_experiment,
            timing_status=timing_status,
            receipt=receipt_values,
            max_rows_per_table=max_table_rows,
        )
    reward_rows = build_miles_reward_outcome_rows(
        [*sample_rows["rollout"], *sample_rows["eval"]],
        experiment_id=safe_experiment,
    )
    sync_rows, sync_summary = (
        build_miles_sync_evidence_rows(sync_metrics_dir, experiment_id=safe_experiment)
        if sync_metrics_dir
        else ([], {})
    )

    generated_artifacts: list[Path] = []
    checkpoint_rows: list[list[Any]] = []
    resolved_checkpoint_dir = checkpoint_dir or receipt_values.get("save_dir")
    if resolved_checkpoint_dir and Path(str(resolved_checkpoint_dir)).is_dir():
        checkpoint_manifest, checkpoint_rows = write_miles_checkpoint_manifest(
            output_dir / f"{safe_run}.checkpoint_manifest.json",
            str(resolved_checkpoint_dir),
            experiment_id=safe_experiment,
        )
        generated_artifacts.append(checkpoint_manifest)

    evidence_summary_path = output_dir / f"{safe_run}.evidence_summary.json"
    evidence_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": safe_experiment,
                "timing_status": timing_status,
                "metric_event_count": len(metric_events),
                "metric_count": sum(len(event["metrics"]) for event in metric_events),
                "sample_rows_logged": {key: len(value) for key, value in sample_rows.items()},
                "sample_rows_total": sample_totals,
                "reward_outcome_count": len(reward_rows),
                "sync_summary": sync_summary,
                "sync_record_count": len(sync_rows),
                "checkpoint_file_count": len(checkpoint_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    generated_artifacts.append(evidence_summary_path)
    run = wandb_module.init(
        project=project,
        entity=entity or None,
        id=safe_run,
        name=safe_run,
        group=group or safe_experiment,
        job_type=safe_stage,
        mode=mode,
        resume="allow",
        tags=["canonical", "pie-cpp", safe_stage, status, timing_status],
        config=redact_sensitive(
            {
                "experiment_id": safe_experiment,
                "timing_status": timing_status,
                "proof_surface_schema": 2,
                "stage_receipt": receipt_values,
                "evidence": {
                    "metric_events": len(metric_events),
                    "rollout_rows_logged": len(sample_rows["rollout"]),
                    "rollout_rows_total": sample_totals["rollout"],
                    "eval_rows_logged": len(sample_rows["eval"]),
                    "eval_rows_total": sample_totals["eval"],
                    "sync_records": len(sync_rows),
                    "checkpoint_files": len(checkpoint_rows),
                },
            }
        ),
    )
    existing_summary = dict(getattr(run, "summary", {}))
    for event_index, event in enumerate(metric_events):
        run.log(
            {
                "evidence/event_index": event_index,
                "evidence/source_family": event["family"],
                "evidence/source_step": event["step"],
                **event["metrics"],
            }
        )
    table_payload: dict[str, Any] = {}
    if metric_events:
        table_payload["tables/stage_metrics"] = wandb_module.Table(
            columns=list(MILES_METRIC_TABLE_COLUMNS),
            data=build_miles_metric_table_rows(
                metric_events,
                experiment_id=safe_experiment,
                timing_status=timing_status,
            ),
        )
    for sample_stage in ("rollout", "eval"):
        if sample_rows[sample_stage]:
            table_payload[f"tables/{sample_stage}_samples"] = wandb_module.Table(
                columns=list(MILES_SAMPLE_TABLE_COLUMNS),
                data=sample_rows[sample_stage],
            )
    if reward_rows:
        table_payload["tables/reward_outcomes"] = wandb_module.Table(
            columns=list(MILES_REWARD_OUTCOME_TABLE_COLUMNS),
            data=reward_rows,
        )
    if sync_rows:
        table_payload["tables/weight_sync"] = wandb_module.Table(
            columns=list(MILES_SYNC_TABLE_COLUMNS),
            data=sync_rows,
        )
    if checkpoint_rows:
        table_payload["tables/checkpoint_manifest"] = wandb_module.Table(
            columns=list(MILES_CHECKPOINT_TABLE_COLUMNS),
            data=checkpoint_rows,
        )
    if table_payload:
        run.log(table_payload)

    curated_metrics = _curate_stage_metrics({**existing_summary, **latest_metrics})
    wall_s = _number_or_none(receipt_values.get("wall_s"))
    peak_vram = _number_or_none(receipt_values.get("max_memory_used_mib"))
    checkpoint = str(receipt_values.get("save_dir") or "")
    run.log(
        {
            "stage/finalized": 1,
            "stage/wall_s": wall_s or 0.0,
            "stage/max_memory_used_mib": peak_vram or 0.0,
            **curated_metrics,
        }
    )
    _set_summary(
        run,
        {
            "observability/experiment_id": safe_experiment,
            "observability/schema_version": 2,
            "observability/timing_status": timing_status,
            "stage/job_type": safe_stage,
            "stage/status": status,
            "stage/wall_s": wall_s,
            "stage/max_memory_used_mib": peak_vram,
            "stage/checkpoint_or_adapter": checkpoint,
            "stage/receipt": receipt_path.name,
            "evidence/metric_event_count": len(metric_events),
            "evidence/rollout_rows_logged": len(sample_rows["rollout"]),
            "evidence/rollout_rows_total": sample_totals["rollout"],
            "evidence/eval_rows_logged": len(sample_rows["eval"]),
            "evidence/eval_rows_total": sample_totals["eval"],
            "evidence/reward_outcome_count": len(reward_rows),
            "evidence/checkpoint_file_count": len(checkpoint_rows),
            **latest_metrics,
            **sync_summary,
            **curated_metrics,
        },
    )
    manifest_path, selected = write_artifact_manifest(
        output_dir / f"{safe_run}.artifact_manifest.json",
        [receipt_path, *artifact_paths, *generated_artifacts],
    )
    artifact = wandb_module.Artifact(
        safe_identifier(f"{safe_experiment}-{safe_stage}-run"),
        type="stage-run",
        metadata={
            "experiment_id": safe_experiment,
            "stage": safe_stage,
            "status": status,
            "timing_status": timing_status,
        },
    )
    for path in [*selected, manifest_path]:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    _set_summary(run, {"stage/artifact_file_count": len(selected)})
    result = {"run_id": str(getattr(run, "id", safe_run)), "url": str(getattr(run, "url", ""))}
    run.finish()
    return result


def log_pipeline_milestone(
    wandb_module: Any,
    *,
    project: str,
    entity: str | None,
    experiment_id: str,
    stage: str,
    event: str,
    status: str,
    mode: str,
    run_id: str = "",
    wall_s: float | None = None,
    repo_sha: str = "",
    image: str = "",
    receipt: str | Path | None = None,
    error: str = "",
    event_time: float | None = None,
) -> dict[str, str]:
    safe_experiment = resolve_experiment_id(explicit=experiment_id)
    safe_stage = safe_identifier(stage, fallback="stage")
    safe_event = safe_identifier(event, fallback="event")
    resolved_run_id = safe_identifier(run_id or f"{safe_experiment}-pipeline")
    now = event_time if event_time is not None else time.time()
    receipt_path = Path(receipt) if receipt else None
    receipt_name = receipt_path.name if receipt_path and receipt_path.exists() else ""
    safe_error = str(redact_sensitive(error))[:1000]
    run = wandb_module.init(
        project=project,
        entity=entity or None,
        id=resolved_run_id,
        name=resolved_run_id,
        group=safe_experiment,
        job_type="pipeline",
        mode=mode,
        resume="allow",
        tags=["canonical", "pie-cpp", "pipeline"],
        config={"experiment_id": safe_experiment, "proof_surface_schema": 1},
    )
    _define_metric(run, "pipeline/event_time_unix")
    status_code = {"started": 0, "success": 1, "failed": -1}.get(status, 0)
    run.log(
        {
            "pipeline/event_time_unix": now,
            "pipeline/wall_s": wall_s or 0.0,
            "pipeline/status_code": status_code,
            "pipeline/stage": safe_stage,
            "pipeline/event": safe_event,
            f"pipeline/events/{safe_stage}/{safe_event}": 1,
            f"tables/pipeline_{safe_stage}_{safe_event}": wandb_module.Table(
                columns=list(PIPELINE_TABLE_COLUMNS),
                data=[
                    [
                        safe_experiment,
                        now,
                        safe_stage,
                        safe_event,
                        status,
                        wall_s,
                        repo_sha,
                        image,
                        receipt_name,
                        safe_error,
                    ]
                ],
            ),
        }
    )
    _set_summary(
        run,
        {
            "observability/experiment_id": safe_experiment,
            "pipeline/latest_stage": safe_stage,
            "pipeline/latest_event": safe_event,
            "pipeline/latest_status": status,
            "pipeline/latest_event_unix": now,
            f"pipeline/{safe_stage}/status": status,
            f"pipeline/{safe_stage}/wall_s": wall_s,
            f"pipeline/{safe_stage}/repo_sha": repo_sha,
            f"pipeline/{safe_stage}/image": image,
        },
    )
    if receipt_path and receipt_path.exists():
        manifest_path, selected = write_artifact_manifest(
            receipt_path.parent / f"{safe_stage}-{safe_event}.artifact_manifest.json",
            [receipt_path],
        )
        artifact = wandb_module.Artifact(
            safe_identifier(f"{safe_experiment}-{safe_stage}-{safe_event}"),
            type="stage-receipt",
            metadata={
                "experiment_id": safe_experiment,
                "stage": safe_stage,
                "event": safe_event,
                "status": status,
            },
        )
        for path in [*selected, manifest_path]:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
    result = {
        "run_id": str(getattr(run, "id", resolved_run_id)),
        "url": str(getattr(run, "url", "")),
    }
    run.finish()
    return result


def _sample_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("task_id") or ""), int(row.get("sample_index") or 0)


def _failure_bucket(record: dict[str, Any]) -> str:
    reason = str(record.get("reason") or "").strip()
    if reason:
        return reason
    if record.get("compile_error") is True:
        return "compile_error"
    if record.get("sanitizer_error") is True:
        return "sanitizer_error"
    if record.get("timeout") is True:
        return "timeout"
    if record.get("all_tests_pass") is True:
        return "correct"
    return "tests_failed"


def _preview(value: Any, *, limit: int = 4000) -> str:
    return str(value or "").replace("\x00", "")[:limit]


def _number_or_none(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:e[+-]?\d+)?", stripped, re.IGNORECASE):
        return float(stripped)
    return stripped


def _curate_stage_metrics(summary: dict[str, Any]) -> dict[str, int | float]:
    curated: dict[str, int | float] = {}
    for key, value in sorted(summary.items()):
        if key.startswith("_") or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if any(term in key.lower() for term in CURATED_STAGE_METRIC_TERMS):
            curated[f"stage/final_metrics/{safe_identifier(key)}"] = value
        if len(curated) >= 32:
            break
    return curated


def _boolean_rate(records: Iterable[dict[str, Any]], key: str) -> float:
    values = list(records)
    return sum(value.get(key) is True for value in values) / len(values) if values else 0.0


def _latest_iteration_name(root: Path) -> str:
    iterations = sorted(path.name for path in root.glob("iter_*") if path.is_dir())
    return iterations[-1] if iterations else ""


def _latest_checkpoint_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    iterations = sorted(path for path in root.glob("iter_*") if path.is_dir())
    selected: set[Path] = set()
    if iterations:
        selected.update(
            path for path in iterations[-1].rglob("*") if path.is_file() and not path.is_symlink()
        )
        selected.update(path for path in root.iterdir() if path.is_file() and not path.is_symlink())
        rollout_dir = root / "rollout"
        if rollout_dir.is_dir():
            rollout_files = sorted(
                path for path in rollout_dir.rglob("*") if path.is_file() and not path.is_symlink()
            )
            if rollout_files:
                selected.add(rollout_files[-1])
    else:
        selected.update(
            path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
        )
    return sorted(selected, key=lambda path: path.relative_to(root).as_posix())


def _artifact_skip_reason(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "not_file"
    if path.is_symlink():
        return "symlink"
    if any(SENSITIVE_PATH_RE.search(part) for part in path.parts):
        return "sensitive_name"
    if path.suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES:
        return "unsupported_suffix"
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        return "too_large"
    if _contains_sensitive_content(path):
        return "sensitive_content"
    return ""


def _contains_sensitive_content(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            assignments = ASSIGNMENT_RE.finditer(line)
            if URL_CREDENTIAL_RE.search(line) or any(
                SENSITIVE_CONTENT_KEY_RE.search(match.group(1)) and match.group(3) != "<redacted>"
                for match in assignments
            ):
                return True
    return False


def _redact_assignment(match: re.Match[str]) -> str:
    if SENSITIVE_KEY_RE.search(match.group(1)):
        return f"{match.group(1)}{match.group(2)}<redacted>"
    return match.group(0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _define_metric(run: Any, name: str, **kwargs: Any) -> None:
    define = getattr(run, "define_metric", None)
    if callable(define):
        define(name, **kwargs)


def _set_summary(run: Any, values: dict[str, Any]) -> None:
    for key, value in redact_sensitive(values).items():
        if value is not None:
            run.summary[key] = value
