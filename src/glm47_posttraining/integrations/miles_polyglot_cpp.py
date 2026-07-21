"""Miles dataset and reward bridge for Aider Polyglot C++."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from glm47_posttraining.cpp_perf.dataset import (
    AIDER_DATA_SOURCE,
    ComplexityWireSampler,
    aider_sft_output,
    build_aider_prompt,
    load_aider_tasks,
    priority_weighted_sample,
)
from glm47_posttraining.cpp_perf.reward import AiderRewardBreakdown, compute_aider_reward
from glm47_posttraining.cpp_perf.sandbox import DEFAULT_DOCKER_IMAGE, run_aider_in_sandbox
from glm47_posttraining.cpp_perf.schema import AiderPolyglotTask, Catch2HarnessResult


DATASET_VERSION = 2
DATA_ROOT_ENV = "GLM47_DATA_DIR"
SANDBOX_IMAGE_ENV = "GLM47_CPP_SANDBOX_IMAGE"
REWARD_WORKERS_ENV = "GLM47_CPP_REWARD_WORKERS"
RUBRIC_RISK_FILE_ENV = "MILES_CPP_RUBRIC_RISK_FILE"
TAXONOMY_FILE_ENV = "MILES_CPP_TAXONOMY_FILE"
DEFAULT_REWARD_WORKERS = 8


def read_taxonomy(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    taxonomy_path = Path(path)
    if not taxonomy_path.exists():
        return {}
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("taxonomy must be a JSON object keyed by task id")
    return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}


def task_with_taxonomy(
    task: AiderPolyglotTask,
    taxonomy: Mapping[str, Mapping[str, object]],
) -> AiderPolyglotTask:
    meta = taxonomy.get(task.task_id, {})
    return AiderPolyglotTask.model_validate(
        {
            **task.model_dump(),
            "cyclomatic_complexity": int(meta.get("cyclomatic_complexity", task.cyclomatic_complexity)),
            "prompt_token_length": int(meta.get("prompt_token_length", task.prompt_token_length)),
            "empirical_failure_rate": float(
                meta.get("empirical_failure_rate", task.empirical_failure_rate)
            ),
            "difficulty_index": float(meta.get("difficulty_index_D_i", task.difficulty_index)),
            "priority_weight": max(
                1e-12, float(meta.get("priority_weight_P_i", task.priority_weight))
            ),
            "rubric_category": str(meta.get("rubric_category", task.rubric_category)),
            "rubric_risks": dict(meta.get("rubric_risks", task.rubric_risks)),
        }
    )


@lru_cache(maxsize=8)
def _risk_snapshot(path: str, modified_ns: int) -> dict[str, dict[str, object]]:
    del modified_ns
    return read_taxonomy(path)


def task_with_live_risks(task: AiderPolyglotTask) -> AiderPolyglotTask:
    """Overlay the evaluation-epoch risk sidecar without rebuilding task JSON."""

    configured = os.environ.get(RUBRIC_RISK_FILE_ENV) or os.environ.get(TAXONOMY_FILE_ENV)
    if not configured:
        return task
    path = Path(configured)
    if not path.is_file():
        return task
    taxonomy = _risk_snapshot(str(path.resolve()), path.stat().st_mtime_ns)
    return task_with_taxonomy(task, taxonomy)


def build_miles_polyglot_datasets(
    tasks_dir: str | Path,
    output_dir: str | Path,
    *,
    taxonomy_file: str | Path | None = None,
    sampling_strategy: str = "wire",
    sample_count: int | None = None,
    wire_batch_size: int = 32,
    gamma: float = 1.5,
    seed: int = 0,
    force: bool = False,
) -> dict[str, Path]:
    source_root = Path(tasks_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"{output} is not empty; pass --force to replace it")
        shutil.rmtree(output)

    taxonomy = read_taxonomy(taxonomy_file)
    loaded = [
        (path, task_with_taxonomy(task, taxonomy))
        for path, task in load_aider_tasks(source_root)
    ]
    train = [(path, task) for path, task in loaded if task.split == "train"]
    evaluation = [(path, task) for path, task in loaded if task.split in {"validation", "test"}]
    if not train:
        raise ValueError("Polyglot dataset requires at least one train task")
    if not evaluation:
        raise ValueError("Polyglot dataset requires at least one validation/test task")

    task_path_by_id: dict[str, str] = {}
    for source_path, task in loaded:
        destination = output / "tasks" / task.split / f"{task.task_id}.json"
        task.write_json(destination)
        task_path_by_id[task.task_id] = destination.relative_to(output).as_posix()

    train_tasks = [task for _path, task in train]
    requested_count = sample_count if sample_count is not None else max(len(train_tasks), wire_batch_size)
    sampled = sample_training_tasks(
        train_tasks,
        taxonomy,
        strategy=sampling_strategy,
        count=requested_count,
        wire_batch_size=wire_batch_size,
        gamma=gamma,
        seed=seed,
    )

    sft_rows = [
        {
            "messages": [
                {"role": "user", "content": build_aider_prompt(task)},
                {"role": "assistant", "content": aider_sft_output(task)},
            ],
            "label": task.task_id,
            "task_id": task.task_id,
            "split": task.split,
            "metadata": task_metadata(task, task_path_by_id[task.task_id], subset="train"),
        }
        for task in train_tasks
    ]
    grpo_rows = [prompt_row(task, task_path_by_id[task.task_id], subset="train") for task in sampled]
    eval_rows = [
        prompt_row(task, task_path_by_id[task.task_id], subset="eval")
        for _path, task in evaluation
    ]

    paths = {
        "sft_train": output / "sft" / "train.jsonl",
        "grpo_train": output / "grpo" / "train.jsonl",
        "eval": output / "eval" / "validation.jsonl",
        "manifest": output / "manifest.json",
    }
    write_jsonl(paths["sft_train"], sft_rows)
    write_jsonl(paths["grpo_train"], grpo_rows)
    write_jsonl(paths["eval"], eval_rows)
    manifest = {
        "kind": "miles-aider-polyglot-cpp-dataset",
        "schema_version": DATASET_VERSION,
        "data_source": AIDER_DATA_SOURCE,
        "source_tasks_dir": str(source_root),
        "taxonomy_file": str(taxonomy_file or ""),
        "sampling": {
            "strategy": sampling_strategy,
            "sample_count": requested_count,
            "wire_batch_size": wire_batch_size,
            "gamma": gamma,
            "seed": seed,
        },
        "counts": {
            "train_unique": len(train),
            "grpo_rows": len(grpo_rows),
            "eval": len(eval_rows),
        },
        "files": {
            key: value.relative_to(output).as_posix()
            for key, value in paths.items()
            if key != "manifest"
        },
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def sample_training_tasks(
    tasks: Sequence[AiderPolyglotTask],
    taxonomy: Mapping[str, Mapping[str, object]],
    *,
    strategy: str,
    count: int,
    wire_batch_size: int,
    gamma: float,
    seed: int,
) -> list[AiderPolyglotTask]:
    if count <= 0:
        raise ValueError("sample_count must be positive")
    if strategy == "priority":
        return priority_weighted_sample(
            tasks, taxonomy, count=count, gamma=gamma, seed=seed
        )
    if strategy == "wire":
        sampler = ComplexityWireSampler(tasks, taxonomy, seed=seed)
        rows: list[AiderPolyglotTask] = []
        while len(rows) < count:
            rows.extend(sampler.sample_batch(min(wire_batch_size, count - len(rows))))
        return rows
    if strategy == "uniform":
        rng = random.Random(seed)
        rows = []
        while len(rows) < count:
            epoch = list(tasks)
            rng.shuffle(epoch)
            rows.extend(epoch)
        return rows[:count]
    raise ValueError(f"unknown sampling strategy: {strategy}")


def task_metadata(task: AiderPolyglotTask, task_path: str, *, subset: str) -> dict[str, object]:
    return {
        "data_source": AIDER_DATA_SOURCE,
        "task_id": task.task_id,
        "problem_id": task.task_id,
        "split": task.split,
        "subset": subset,
        "task_path": task_path,
        "topic_category": task.topic_category,
        "rubric_category": task.effective_rubric_category,
        "rubric_risks": task.rubric_risks,
        "difficulty": task.difficulty,
        "difficulty_index": task.difficulty_index,
        "priority_weight": task.priority_weight,
        "cyclomatic_complexity": task.cyclomatic_complexity,
        "prompt_token_length": task.prompt_token_length,
    }


def prompt_row(task: AiderPolyglotTask, task_path: str, *, subset: str) -> dict[str, object]:
    return {
        "prompt": build_aider_prompt(task),
        "label": task.task_id,
        "task_id": task.task_id,
        "problem_id": task.task_id,
        "split": task.split,
        "metadata": task_metadata(task, task_path, subset=subset),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


async def reward_func(_args: Any, sample: Any, **_kwargs: Any) -> dict[str, object] | list[dict[str, object]]:
    if isinstance(sample, list):
        workers = min(max(1, reward_workers()), len(sample))
        semaphore = asyncio.Semaphore(workers)

        async def score(item: Any) -> dict[str, object]:
            async with semaphore:
                return await asyncio.to_thread(score_sample, item)

        return list(await asyncio.gather(*(score(item) for item in sample)))
    return await asyncio.to_thread(score_sample, sample)


def reward_workers() -> int:
    try:
        return max(1, int(os.environ.get(REWARD_WORKERS_ENV, str(DEFAULT_REWARD_WORKERS))))
    except ValueError:
        return DEFAULT_REWARD_WORKERS


def score_sample(sample: Any) -> dict[str, object]:
    metadata = sample_metadata(sample)
    response = sample_response(sample)
    try:
        task_path = resolve_task_path(str(metadata["task_path"]), metadata)
        task = task_with_live_risks(AiderPolyglotTask.read_json(task_path))
        breakdown = compute_aider_reward(
            task,
            response,
            reasoning_token_count=sample_reasoning_token_count(sample),
            context_exhausted=sample_context_exhausted(sample),
            runner=lambda current, files: run_aider_in_sandbox(
                current,
                files,
                image=os.environ.get(SANDBOX_IMAGE_ENV, DEFAULT_DOCKER_IMAGE),
            ),
        )
        return reward_record(sample, task, breakdown)
    except Exception as exc:
        return {
            "score": -1.0,
            "reward": -1.0,
            "reason": "reward_exception",
            "task_id": metadata.get("task_id", ""),
            "split": metadata.get("split"),
            "exception": str(exc),
            "response": response,
        }


def reward_record(
    sample: Any,
    task: AiderPolyglotTask,
    breakdown: AiderRewardBreakdown,
) -> dict[str, object]:
    harness = breakdown.harness
    verified_all_tests_pass = bool(
        harness
        and harness.all_tests_pass
        and (
            task.effective_rubric_category != "state_concurrency"
            or harness.thread_sanitizer_ran
        )
    )
    return {
        "score": breakdown.reward,
        "reward": breakdown.reward,
        "reason": breakdown.reason,
        "task_id": task.task_id,
        "problem_id": task.task_id,
        "split": task.split,
        "topic_category": task.topic_category,
        "rubric_category": task.effective_rubric_category,
        "difficulty": task.difficulty,
        "difficulty_index": task.difficulty_index,
        "sample_index": sample_index(sample),
        "rollout_id": getattr(sample, "rollout_id", None),
        "format_valid": breakdown.format_valid,
        "sandbox_ran": harness is not None,
        "reasoning_tokens": breakdown.reasoning_tokens,
        "rubric_scores": breakdown.rubric_scores,
        "rubric_weights": breakdown.rubric_weights,
        "rubric_risks": breakdown.rubric_risks,
        "active_rubrics": list(breakdown.active_rubrics),
        "rubric_observations": list(breakdown.observations),
        **{
            f"rubric_{name}": score
            for name, score in breakdown.rubric_scores.items()
        },
        **{
            f"rubric_weight_{name}": weight
            for name, weight in breakdown.rubric_weights.items()
        },
        "compile_error": bool(harness.compile_error) if harness else False,
        "sanitizer_error": bool(harness.sanitizer_error) if harness else False,
        "thread_sanitizer_ran": bool(harness.thread_sanitizer_ran) if harness else False,
        "thread_sanitizer_error": bool(harness.thread_sanitizer_error) if harness else False,
        "thread_sanitizer_timeout": bool(harness.thread_sanitizer_timeout) if harness else False,
        "timeout": bool(harness.timeout) if harness else False,
        "tests_passed": harness.passed_test_cases if harness else 0,
        "tests_total": harness.total_test_cases if harness else 0,
        "passed_assertions": harness.passed_assertions if harness else 0,
        "total_assertions": harness.total_assertions if harness else 0,
        "runtime_cpu_ns": harness.runtime_cpu_ns if harness else None,
        "reference_runtime_cpu_ns": harness.reference_runtime_cpu_ns if harness else None,
        "runtime_speedup": (
            harness.reference_runtime_cpu_ns / harness.runtime_cpu_ns
            if harness
            and harness.runtime_cpu_ns
            and harness.reference_runtime_cpu_ns is not None
            else None
        ),
        "all_tests_pass": verified_all_tests_pass,
        "candidate_bytes": sum(
            len(content.encode("utf-8")) for content in (breakdown.files or {}).values()
        ),
        "response": sample_response(sample),
    }


def compute_calibrated_grpo_advantages(
    group_rewards: list[list[float]],
    group_difficulties: list[float],
    *,
    lambda_d: float = 0.25,
    eta: float = 0.2,
    eps: float = 1e-8,
) -> list[list[float]]:
    """Compute the proposed cross-group advantages without a NumPy dependency.

    This helper is intentionally explicit; it must be wired into the matching
    Miles release's advantage estimator before enabling it in a GPU run.
    """

    if len(group_rewards) != len(group_difficulties):
        raise ValueError("group_rewards and group_difficulties must have equal length")
    if not group_rewards or any(not group for group in group_rewards):
        raise ValueError("reward groups must be non-empty")
    scaled = [
        [reward * (1.0 + lambda_d * difficulty) for reward in rewards]
        for rewards, difficulty in zip(group_rewards, group_difficulties)
    ]
    flat = [reward for group in scaled for reward in group]
    global_mean = sum(flat) / len(flat)
    output = []
    for rewards in scaled:
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        stddev = math.sqrt(variance)
        intra = [0.0] * len(rewards) if stddev <= eps else [
            (reward - mean) / (stddev + eps) for reward in rewards
        ]
        output.append([
            local + eta * (reward - global_mean)
            for local, reward in zip(intra, rewards)
        ])
    return output


def sample_metadata(sample: Any) -> dict[str, object]:
    value = sample.get("metadata") if isinstance(sample, dict) else getattr(sample, "metadata", None)
    return dict(value) if isinstance(value, dict) else {}


def sample_response(sample: Any) -> str:
    return str(sample.get("response", "") if isinstance(sample, dict) else getattr(sample, "response", ""))


def sample_index(sample: Any) -> int | None:
    value = sample.get("index") if isinstance(sample, dict) else getattr(sample, "index", None)
    return value if isinstance(value, int) else None


def sample_context_exhausted(sample: Any) -> bool:
    if isinstance(sample, dict):
        values = sample
    else:
        values = {
            "truncated": getattr(sample, "truncated", None),
            "finish_reason": getattr(sample, "finish_reason", None),
            "outcome": getattr(sample, "outcome", None),
        }
    if values.get("truncated") is True or values.get("outcome") == "context_exhausted":
        return True
    finish_reason = str(values.get("finish_reason") or "").lower()
    return finish_reason in {"length", "max_tokens", "abort_length", "context_exhausted"}


def sample_reasoning_token_count(sample: Any) -> int | None:
    """Use an exact rollout-provided reasoning count when one is available."""

    values = sample if isinstance(sample, dict) else {
        "reasoning_tokens": getattr(sample, "reasoning_tokens", None),
        "num_reasoning_tokens": getattr(sample, "num_reasoning_tokens", None),
    }
    for key in ("reasoning_tokens", "num_reasoning_tokens"):
        value = values.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    metadata = values.get("metadata")
    if isinstance(metadata, dict):
        for key in ("reasoning_tokens", "num_reasoning_tokens"):
            value = metadata.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def resolve_task_path(task_path: str, metadata: Mapping[str, object]) -> Path:
    path = Path(task_path)
    if path.is_absolute():
        return path
    for root in (metadata.get("task_root"), os.environ.get(DATA_ROOT_ENV), Path.cwd()):
        if root:
            candidate = Path(root) / path
            if candidate.exists():
                return candidate
    return Path.cwd() / path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-data")
    build.add_argument("--tasks-dir", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--taxonomy-file", default=None)
    build.add_argument("--sampling-strategy", choices=("uniform", "priority", "wire"), default="wire")
    build.add_argument("--sample-count", type=int, default=None)
    build.add_argument("--wire-batch-size", type=int, default=32)
    build.add_argument("--gamma", type=float, default=1.5)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = build_miles_polyglot_datasets(
        args.tasks_dir,
        args.out,
        taxonomy_file=args.taxonomy_file,
        sampling_strategy=args.sampling_strategy,
        sample_count=args.sample_count,
        wire_batch_size=args.wire_batch_size,
        gamma=args.gamma,
        seed=args.seed,
        force=args.force,
    )
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
