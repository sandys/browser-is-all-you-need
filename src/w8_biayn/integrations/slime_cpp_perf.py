"""SLIME data and reward bridge for the PIE C++ performance task."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from w8_biayn.cpp_perf.eval import aggregate_eval_records, compare_eval_summaries, read_jsonl, write_json
from w8_biayn.cpp_perf.reward import RewardBreakdown, compute_reward
from w8_biayn.cpp_perf.sandbox import DEFAULT_CPU, DEFAULT_DOCKER_IMAGE, run_in_sandbox
from w8_biayn.cpp_perf.schema import CppTask, HarnessResult
from w8_biayn.cpp_perf.skyrl_dataset import DATA_SOURCE, build_prompt, load_tasks, sft_output

SLIME_CPP_PERF_DATASET_VERSION = 1
DEFAULT_EVAL_SPLITS = ("validation", "test")
DEFAULT_DATA_ROOT_ENV = "W8_BIAYN_DATA_DIR"
SANDBOX_IMAGE_ENV = "W8_CPP_SANDBOX_IMAGE"
SANDBOX_CPU_ENV = "W8_CPP_SANDBOX_CPU"
INCLUDE_LOGS_ENV = "W8_SLIME_CPP_INCLUDE_LOGS"


def build_slime_cpp_perf_datasets(
    tasks_dir: str | Path,
    output_dir: str | Path,
    *,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    eval_splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    profile: str = "smoke",
    run_id: str | None = None,
    sort_by_size: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Write SLIME SFT, GRPO, and eval JSONL files from validated C++ task JSON."""

    source_root = Path(tasks_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"{output} already exists and is not empty; pass force=True to replace it")
        shutil.rmtree(output)

    loaded = load_tasks(source_root)
    if not loaded:
        raise ValueError(f"No task JSON files found under {source_root}")

    train = [(path, task) for path, task in loaded if task.split == "train"]
    eval_split_names = set(eval_splits)
    eval_rows = [(path, task) for path, task in loaded if task.split in eval_split_names]
    if sort_by_size:
        train.sort(key=lambda row: _sft_size_key(row[1]))
    if train_limit is not None:
        train = train[:train_limit]
    if eval_limit is not None:
        eval_rows = eval_rows[:eval_limit]
    if not train:
        raise ValueError("SLIME C++ dataset requires at least one train task")
    if not eval_rows:
        raise ValueError(f"SLIME C++ dataset requires at least one eval task from {sorted(eval_split_names)}")

    selected = _dedupe_task_paths([path for path, _task in train] + [path for path, _task in eval_rows])
    task_paths = _copy_selected_tasks(source_root, output / "tasks", selected)

    sft_rows = [_sft_row(task, task_paths[path], source_path=path) for path, task in train]
    grpo_rows = [_prompt_row(task, task_paths[path], source_path=path, subset="train") for path, task in train]
    eval_prompt_rows = [
        _prompt_row(task, task_paths[path], source_path=path, subset="eval") for path, task in eval_rows
    ]

    paths = {
        "sft_train": output / "sft" / "train.jsonl",
        "grpo_train": output / "grpo" / "train.jsonl",
        "eval": output / "eval" / "validation.jsonl",
        "manifest": output / "manifest.json",
    }
    _write_jsonl(paths["sft_train"], sft_rows)
    _write_jsonl(paths["grpo_train"], grpo_rows)
    _write_jsonl(paths["eval"], eval_prompt_rows)
    write_json(
        paths["manifest"],
        {
            "kind": "slime-cpp-perf-dataset",
            "schema_version": SLIME_CPP_PERF_DATASET_VERSION,
            "data_source": DATA_SOURCE,
            "profile": profile,
            "run_id": run_id,
            "source_tasks_dir": str(source_root),
            "output_dir": str(output),
            "eval_splits": list(eval_splits),
            "sort_by_size": sort_by_size,
            "counts": {
                "train": len(train),
                "eval": len(eval_rows),
                "copied_tasks": len(selected),
            },
            "files": {key: path.relative_to(output).as_posix() for key, path in paths.items() if key != "manifest"},
        },
    )
    return paths


def _dedupe_task_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    output: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(path)
    return output


def _sft_size_key(task: CppTask) -> tuple[int, str]:
    return len(build_prompt(task)) + len(sft_output(task)), task.task_id


def _copy_selected_tasks(source_root: Path, task_output_root: Path, source_paths: Iterable[Path]) -> dict[Path, str]:
    task_paths: dict[Path, str] = {}
    for source_path in source_paths:
        try:
            relative_source = source_path.relative_to(source_root)
        except ValueError:
            relative_source = Path(source_path.name)
        destination = task_output_root / relative_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        task_paths[source_path] = destination.relative_to(task_output_root.parent).as_posix()
    return task_paths


def _task_metadata(task: CppTask, task_path: str, *, source_path: Path, subset: str) -> dict[str, Any]:
    return {
        "data_source": DATA_SOURCE,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "split": task.split,
        "subset": subset,
        "task_path": task_path,
        "source_task_path": str(source_path),
        "reference_metric": task.reference.metric,
        "reference_value": task.reference.value,
        "gem5_cycles": task.reference.gem5_cycles,
        "visible_test_count": len(task.unit_tests),
        "hidden_test_count": len(task.hidden_tests),
    }


def _sft_row(task: CppTask, task_path: str, *, source_path: Path) -> dict[str, Any]:
    metadata = _task_metadata(task, task_path, source_path=source_path, subset="train")
    return {
        "messages": [
            {"role": "user", "content": build_prompt(task)},
            {"role": "assistant", "content": sft_output(task)},
        ],
        "label": task.task_id,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "split": task.split,
        "metadata": metadata,
    }


def _prompt_row(task: CppTask, task_path: str, *, source_path: Path, subset: str) -> dict[str, Any]:
    metadata = _task_metadata(task, task_path, source_path=source_path, subset=subset)
    return {
        "prompt": build_prompt(task),
        "label": task.task_id,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "split": task.split,
        "metadata": metadata,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


async def reward_func(args: Any, sample: Any, **_kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
    """SLIME custom reward hook for one sample or a batch of samples."""

    if isinstance(sample, list):
        return [await reward_func(args, item) for item in sample]
    return _score_sample(sample)


def _score_sample(sample: Any) -> dict[str, Any]:
    metadata = _sample_metadata(sample)
    task_path = metadata.get("task_path")
    if not task_path:
        return _exception_record(sample, metadata, reason="missing_task_path", exception="metadata.task_path is required")

    try:
        task = CppTask.read_json(_resolve_task_path(str(task_path), metadata=metadata))
        breakdown = compute_reward(task, _sample_response(sample), runner=_sandbox_runner)
        return reward_record_from_breakdown(sample, task, breakdown)
    except Exception as exc:  # pragma: no cover - guards real rollout workers
        return _exception_record(sample, metadata, reason="reward_exception", exception=str(exc))


def _sandbox_runner(task: CppTask, code: str) -> HarnessResult:
    return run_in_sandbox(
        task,
        code,
        image=os.environ.get(SANDBOX_IMAGE_ENV, DEFAULT_DOCKER_IMAGE),
        cpu=os.environ.get(SANDBOX_CPU_ENV, DEFAULT_CPU),
    )


def reward_record_from_breakdown(sample: Any, task: CppTask, breakdown: RewardBreakdown) -> dict[str, Any]:
    """Convert a reward breakdown into the dict carried by SLIME's ``Sample.reward``."""

    harness = breakdown.harness
    record = _base_record(sample, task, score=breakdown.reward, reason=breakdown.reason)
    record.update(
        {
            "compile_error": bool(harness.compile_error) if harness else False,
            "sanitizer_error": bool(harness.sanitizer_error) if harness else False,
            "timeout": bool(harness.timeout) if harness else False,
            "tests_passed": harness.tests_passed if harness else 0,
            "tests_total": harness.tests_total if harness else 0,
            "all_tests_pass": bool(harness.all_tests_pass) if harness else False,
            "runtime_cpu_ns": harness.runtime_cpu_ns if harness else None,
            "runtime_wall_ns": harness.runtime_wall_ns if harness else None,
            "reference_runtime_cpu_ns": harness.reference_runtime_cpu_ns if harness else None,
            "reference_runtime_wall_ns": harness.reference_runtime_wall_ns if harness else None,
            "runtime_speedup": harness.runtime_speedup if harness else None,
            "candidate_bytes": len((breakdown.code or "").encode("utf-8")),
            "format_valid": breakdown.format_valid,
        }
    )
    if harness and _include_logs():
        record["logs"] = harness.logs
    elif harness:
        record["log_keys"] = sorted(harness.logs)
    return record


def _exception_record(sample: Any, metadata: dict[str, Any], *, reason: str, exception: str) -> dict[str, Any]:
    task_id = str(metadata.get("task_id") or metadata.get("label") or "")
    return {
        "score": -1.0,
        "reward": -1.0,
        "reason": reason,
        "task_id": task_id,
        "problem_id": metadata.get("problem_id"),
        "split": metadata.get("split"),
        "sample_index": _sample_index(sample),
        "rollout_id": getattr(sample, "rollout_id", None),
        "compile_error": False,
        "sanitizer_error": False,
        "timeout": False,
        "tests_passed": 0,
        "tests_total": 0,
        "all_tests_pass": False,
        "runtime_cpu_ns": None,
        "runtime_wall_ns": None,
        "reference_runtime_cpu_ns": None,
        "reference_runtime_wall_ns": None,
        "runtime_speedup": None,
        "candidate_bytes": 0,
        "format_valid": False,
        "exception": exception,
        "response": _sample_response(sample),
    }


def _base_record(sample: Any, task: CppTask, *, score: float, reason: str) -> dict[str, Any]:
    return {
        "score": score,
        "reward": score,
        "reason": reason,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "split": task.split,
        "sample_index": _sample_index(sample),
        "rollout_id": getattr(sample, "rollout_id", None),
        "response": _sample_response(sample),
    }


def _include_logs() -> bool:
    return os.environ.get(INCLUDE_LOGS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    if isinstance(sample, dict) and isinstance(sample.get("metadata"), dict):
        return sample["metadata"]
    return {}


def _sample_response(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("response") or "")
    return str(getattr(sample, "response", "") or "")


def _sample_index(sample: Any) -> int | None:
    if isinstance(sample, dict):
        value = sample.get("index")
    else:
        value = getattr(sample, "index", None)
    return int(value) if isinstance(value, int) else None


def _resolve_task_path(task_path: str, *, metadata: dict[str, Any] | None = None) -> Path:
    path = Path(task_path)
    if path.is_absolute():
        return path

    metadata = metadata or {}
    roots = [
        metadata.get("task_root"),
        os.environ.get(DEFAULT_DATA_ROOT_ENV),
        Path.cwd(),
    ]
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def load_slime_debug_samples(path: str | Path) -> list[dict[str, Any]]:
    """Load scored samples from a SLIME debug dump or JSONL sample file."""

    input_path = Path(path)
    if input_path.suffix == ".pt":
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on runtime image
            raise ImportError("torch is required to read SLIME .pt debug rollout dumps") from exc
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        samples = payload.get("samples")
        if not isinstance(samples, list):
            raise ValueError(f"{input_path} does not contain a list payload under 'samples'")
        return [dict(sample) for sample in samples]
    if input_path.suffix == ".jsonl":
        return read_jsonl(input_path)
    if input_path.suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        samples = payload.get("samples", payload)
        if not isinstance(samples, list):
            raise ValueError(f"{input_path} must contain a list or a mapping with a samples list")
        return [dict(sample) for sample in samples]
    raise ValueError(f"unsupported debug sample format: {input_path}")


def record_from_debug_sample(sample: dict[str, Any], *, label: str | None = None) -> dict[str, Any]:
    """Convert one SLIME debug sample dict into a C++ eval record."""

    reward_payload = sample.get("reward")
    if isinstance(reward_payload, dict):
        record = dict(reward_payload)
    else:
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        record = {
            "score": float(reward_payload or 0.0),
            "reward": float(reward_payload or 0.0),
            "reason": metadata.get("reason", "unknown"),
            "task_id": metadata.get("task_id"),
            "problem_id": metadata.get("problem_id"),
            "split": metadata.get("split"),
        }
    record.setdefault("reward", record.get("score", 0.0))
    record.setdefault("score", record.get("reward", 0.0))
    record.setdefault("task_id", _sample_metadata(sample).get("task_id"))
    record.setdefault("problem_id", _sample_metadata(sample).get("problem_id"))
    record.setdefault("split", _sample_metadata(sample).get("split"))
    record.setdefault("sample_index", sample.get("index"))
    record.setdefault("rollout_id", sample.get("rollout_id"))
    record.setdefault("response", sample.get("response", ""))
    if label is not None:
        record["label"] = label
    if "all_tests_pass" not in record:
        record["all_tests_pass"] = bool(record.get("tests_total")) and record.get("tests_passed") == record.get(
            "tests_total"
        )
    return record


def write_eval_artifacts(
    *,
    label: str,
    debug_samples_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write records and an aggregate summary from a SLIME debug rollout dump."""

    samples = load_slime_debug_samples(debug_samples_path)
    records = [record_from_debug_sample(sample, label=label) for sample in samples]
    output = Path(output_dir)
    records_path = output / f"{label}.records.jsonl"
    summary_path = output / f"{label}.summary.json"
    _write_jsonl(records_path, records)
    write_json(summary_path, aggregate_eval_records(records, label=label))
    return {"records": records_path, "summary": summary_path}


def write_comparison_artifact(summary_paths: Iterable[str | Path], output_path: str | Path) -> Path:
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in summary_paths]
    return write_json(output_path, compare_eval_summaries(summaries))


def _build_data_command(args: argparse.Namespace) -> None:
    paths = build_slime_cpp_perf_datasets(
        args.tasks_dir,
        args.out,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        eval_splits=tuple(args.eval_splits.split(",")),
        profile=args.profile,
        run_id=args.run_id,
        sort_by_size=args.sort_by_size,
        force=args.force,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


def _aggregate_command(args: argparse.Namespace) -> None:
    paths = write_eval_artifacts(label=args.label, debug_samples_path=args.debug_rollout, output_dir=args.out)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


def _compare_command(args: argparse.Namespace) -> None:
    path = write_comparison_artifact(args.summary, args.out)
    print(str(path))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_data = subparsers.add_parser("build-data", help="Convert C++ task JSON into SLIME JSONL files")
    build_data.add_argument("--tasks-dir", required=True)
    build_data.add_argument("--out", required=True)
    build_data.add_argument("--train-limit", type=int, default=None)
    build_data.add_argument("--eval-limit", type=int, default=None)
    build_data.add_argument("--eval-splits", default="validation,test")
    build_data.add_argument("--profile", default="smoke")
    build_data.add_argument("--run-id", default=None)
    build_data.add_argument("--sort-by-size", action="store_true")
    build_data.add_argument("--force", action="store_true")
    build_data.set_defaults(func=_build_data_command)

    aggregate = subparsers.add_parser("aggregate-debug", help="Aggregate SLIME debug rollout samples")
    aggregate.add_argument("--label", required=True, choices=("base", "sft", "grpo"))
    aggregate.add_argument("--debug-rollout", required=True)
    aggregate.add_argument("--out", required=True)
    aggregate.set_defaults(func=_aggregate_command)

    compare = subparsers.add_parser("compare", help="Compare base/SFT/GRPO summary JSON files")
    compare.add_argument("--summary", action="append", required=True)
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=_compare_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
