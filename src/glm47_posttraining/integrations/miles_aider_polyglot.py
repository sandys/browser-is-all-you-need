"""Miles bridge for shadow-task GRPO and official Aider Polyglot C++ evaluation."""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from typing import Any, Sequence

from glm47_posttraining.aider_polyglot.bank_account_curriculum import (
    CURRICULUM_NAME as BANK_ACCOUNT_CURRICULUM,
)
from glm47_posttraining.aider_polyglot.bank_account_curriculum import (
    build_bank_account_curriculum,
)
from glm47_posttraining.aider_polyglot.bank_account_official_drill import (
    CURRICULUM_NAME as BANK_ACCOUNT_OFFICIAL_DRILL,
)
from glm47_posttraining.aider_polyglot.bank_account_official_drill import (
    build_bank_account_official_drill,
    imitation_targets,
)
from glm47_posttraining.aider_polyglot.dataset import build_aider_polyglot_datasets
from glm47_posttraining.aider_polyglot.harness import (
    DEFAULT_AIDER_DOCKER_IMAGE,
    build_aider_sandbox_image,
    run_aider_tests,
    run_sandbox_preflight,
    run_shadow_tests,
)
from glm47_posttraining.aider_polyglot.parser import parse_whole_file_response
from glm47_posttraining.aider_polyglot.reward import AiderRewardBreakdown, compute_aider_reward
from glm47_posttraining.aider_polyglot.schema import AiderPolyglotTask


DEFAULT_DATA_ROOT_ENV = "GLM47_DATA_DIR"
SANDBOX_IMAGE_ENV = "GLM47_CPP_SANDBOX_IMAGE"
REWARD_WORKERS_ENV = "GLM47_CPP_REWARD_WORKERS"
INCLUDE_LOGS_ENV = "MILES_CPP_INCLUDE_LOGS"
DEFAULT_REWARD_WORKERS = 8
_ACTIVE_REWARD_WORKERS = 0
_ACTIVE_REWARD_WORKERS_LOCK = threading.Lock()


def run_response_contract_preflight() -> None:
    """Prove that Miles-retained GLM stop tokens cannot hide the final file."""

    parsed = parse_whole_file_response(
        "preflight.cpp\n```cpp\nint answer() { return 42; }\n```<|user|>",
        ["preflight.cpp"],
    )
    if parsed.files != {"preflight.cpp": "int answer() { return 42; }\n"}:
        raise RuntimeError("Aider response parser failed the retained-stop-token contract")


async def reward_func(
    args: Any, sample: Any, **_kwargs: Any
) -> dict[str, Any] | list[dict[str, Any]]:
    """Miles custom reward hook for one sample or a batch."""

    if isinstance(sample, list):
        workers = max(1, min(len(sample), _reward_workers()))
        semaphore = asyncio.Semaphore(workers)

        async def score(item: Any) -> dict[str, Any]:
            async with semaphore:
                return await asyncio.to_thread(_score_sample_with_worker_load, item)

        records = list(await asyncio.gather(*(score(item) for item in sample)))
        return neutralize_infrastructure_scores(records)
    return await asyncio.to_thread(_score_sample_with_worker_load, sample)


def neutralize_infrastructure_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Zero the GRPO advantage of infrastructure-invalid samples in place.

    Miles trains on ``record["score"]`` (``--reward-key score``) and exposes no
    channel for a custom reward to drop a sample, so an infrastructure-invalid
    sample's 0.0 would enter its prompt group's advantage baseline as if the
    policy had earned it (issue #110 r3: 61/256 training samples were sandbox
    EAGAIN deaths scored this way). Setting the invalid sample's score to the
    mean score of its group's valid members makes its group-normalized
    advantage exactly zero while preserving every valid sample's ordering; a
    group with no valid members collapses to identical scores, which likewise
    yields zero advantage. The audited ``reward`` field keeps the original
    value and ``score_neutralized`` marks the substitution.
    """

    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record.get("rollout_id"), record.get("problem_id"))].append(record)
    for group in groups.values():
        invalid = [record for record in group if record.get("infrastructure_error")]
        if not invalid:
            continue
        valid_scores = [
            float(record.get("score") or 0.0)
            for record in group
            if not record.get("infrastructure_error")
        ]
        anchor = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        for record in invalid:
            record["score"] = anchor
            record["score_neutralized"] = True
    return records


@contextmanager
def _active_reward_worker():
    global _ACTIVE_REWARD_WORKERS
    with _ACTIVE_REWARD_WORKERS_LOCK:
        _ACTIVE_REWARD_WORKERS += 1
        worker_load = _ACTIVE_REWARD_WORKERS
    try:
        yield worker_load
    finally:
        with _ACTIVE_REWARD_WORKERS_LOCK:
            _ACTIVE_REWARD_WORKERS -= 1


def _score_sample_with_worker_load(sample: Any) -> dict[str, Any]:
    with _active_reward_worker() as worker_load:
        return _score_sample(sample, reward_worker_load=worker_load)


def _score_sample(sample: Any, *, reward_worker_load: int = 1) -> dict[str, Any]:
    metadata = _sample_metadata(sample)
    task_path_value = metadata.get("task_path")
    if not task_path_value:
        return _exception_record(
            sample,
            metadata,
            "missing_task_path",
            "metadata.task_path is required",
            reward_worker_load=reward_worker_load,
        )
    try:
        task_path = _resolve_task_path(str(task_path_value), metadata)
        task = AiderPolyglotTask.read_json(task_path)
        exercise_dir = _resolve_exercise_dir(task_path, task.exercise_dir)

        harness_runner = (
            run_shadow_tests if task.harness_kind == "shadow_cpp17" else run_aider_tests
        )

        def runner(path: Path, files: dict[str, str]):
            kwargs: dict[str, Any] = {
                "image": os.environ.get(SANDBOX_IMAGE_ENV, DEFAULT_AIDER_DOCKER_IMAGE)
            }
            if task.harness_kind == "shadow_cpp17":
                kwargs["expected_test_sha256"] = task.hidden_test_sha256
            return harness_runner(path, files, **kwargs)

        breakdown = compute_aider_reward(
            task,
            exercise_dir,
            _sample_response(sample),
            runner=runner,
            strict_binary="strict-binary-reward" in task.tags,
        )
        return reward_record(
            sample, task, breakdown, reward_worker_load=reward_worker_load
        )
    except Exception as exc:  # pragma: no cover - protects remote rollout workers
        return _exception_record(
            sample,
            metadata,
            "reward_exception",
            str(exc),
            reward_worker_load=reward_worker_load,
        )


def reward_record(
    sample: Any,
    task: AiderPolyglotTask,
    breakdown: AiderRewardBreakdown,
    *,
    reward_worker_load: int = 1,
) -> dict[str, Any]:
    harness = breakdown.harness
    parsed = breakdown.parsed
    record = {
        "score": breakdown.reward,
        "reward": breakdown.reward,
        "reason": breakdown.reason,
        "task_id": task.task_id,
        "problem_id": task.exercise,
        "split": task.split,
        "sample_index": _sample_index(sample),
        "rollout_id": getattr(sample, "rollout_id", None),
        "response": _sample_response(sample),
        "format_valid": bool(parsed.format_valid) if parsed else False,
        "modified_files": sorted(parsed.files) if parsed else [],
        "tests_passed": harness.tests_passed if harness else 0,
        "tests_total": harness.tests_total if harness else 0,
        "all_tests_pass": bool(harness.all_tests_pass) if harness else False,
        "compile_error": bool(harness and harness.status == "compile_failed"),
        "timeout": bool(harness and harness.status == "candidate_timeout"),
        "candidate_returncode": harness.candidate_returncode if harness else None,
        "infrastructure_error": breakdown.infrastructure_error,
        "reward_worker_load": reward_worker_load,
        "hidden_test_sha256": task.hidden_test_sha256,
        "verification_gate": task.verification_gate,
        "objective_group": task.objective_group,
        "failure_signature": task.failure_signature,
        "strict_binary_reward": "strict-binary-reward" in task.tags,
    }
    if harness and _include_logs():
        record["logs"] = harness.logs
    elif harness:
        record["log_keys"] = sorted(harness.logs)
    return record


def _exception_record(
    sample: Any,
    metadata: dict[str, Any],
    reason: str,
    exception: str,
    *,
    reward_worker_load: int = 1,
) -> dict[str, Any]:
    return {
        "score": 0.0,
        "reward": 0.0,
        "reason": reason,
        "task_id": metadata.get("task_id"),
        "problem_id": metadata.get("problem_id"),
        "split": metadata.get("split"),
        "sample_index": _sample_index(sample),
        "rollout_id": getattr(sample, "rollout_id", None),
        "response": _sample_response(sample),
        "format_valid": False,
        "modified_files": [],
        "tests_passed": 0,
        "tests_total": 0,
        "all_tests_pass": False,
        "compile_error": False,
        "timeout": False,
        "infrastructure_error": True,
        "reward_worker_load": reward_worker_load,
        "exception": exception,
    }


def _sample_metadata(sample: Any) -> dict[str, Any]:
    metadata = (
        sample.get("metadata") if isinstance(sample, dict) else getattr(sample, "metadata", None)
    )
    return metadata if isinstance(metadata, dict) else {}


def _sample_response(sample: Any) -> str:
    value = sample.get("response") if isinstance(sample, dict) else getattr(sample, "response", "")
    return str(value or "")


def _sample_index(sample: Any) -> int | None:
    value = sample.get("index") if isinstance(sample, dict) else getattr(sample, "index", None)
    return int(value) if isinstance(value, int) else None


def _resolve_task_path(task_path: str, metadata: dict[str, Any]) -> Path:
    path = Path(task_path)
    if path.is_absolute():
        return path
    for root in (metadata.get("task_root"), os.environ.get(DEFAULT_DATA_ROOT_ENV), Path.cwd()):
        if root:
            candidate = Path(root) / path
            if candidate.exists():
                return candidate
    return Path.cwd() / path


def _resolve_exercise_dir(task_path: Path, exercise_dir: str) -> Path:
    for root in task_path.parents:
        candidate = root / exercise_dir
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cannot resolve {exercise_dir} from descriptor {task_path}")


def _reward_workers() -> int:
    try:
        return max(1, int(os.environ.get(REWARD_WORKERS_ENV, DEFAULT_REWARD_WORKERS)))
    except ValueError:
        return DEFAULT_REWARD_WORKERS


def _include_logs() -> bool:
    return os.environ.get(INCLUDE_LOGS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-data")
    build.add_argument("--tasks-dir", required=True, help="checked-in Aider shadow rubric tree")
    build.add_argument("--out", required=True)
    build.add_argument(
        "--curriculum",
        choices=[BANK_ACCOUNT_CURRICULUM, BANK_ACCOUNT_OFFICIAL_DRILL],
    )
    build.add_argument("--allow-non-gcc-curriculum", action="store_true")
    build.add_argument("--train-limit", type=int)
    build.add_argument("--eval-limit", type=int, help="training-task monitor size")
    build.add_argument("--eval-splits", default="validation,test")
    build.add_argument("--profile", default="aider-polyglot-cpp")
    build.add_argument("--run-id")
    build.add_argument("--sort-by-size", action="store_true")
    build.add_argument("--filter-train-oracle-full-marks", action="store_true")
    build.add_argument("--oracle-filter-workers", type=int, default=8)
    build.add_argument("--force", action="store_true")
    image = subparsers.add_parser("build-image")
    image.add_argument("--image", default=DEFAULT_AIDER_DOCKER_IMAGE)
    subparsers.add_parser("preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "build-image":
        result = build_aider_sandbox_image(image=args.image)
        print(result.stdout, end="")
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.returncode)
        return
    if args.command == "preflight":
        run_response_contract_preflight()
        run_sandbox_preflight()
        print("AIDER_REWARD_SANDBOX_READY")
        return
    if args.filter_train_oracle_full_marks:
        raise ValueError("the packaged shadow corpus is already restricted to terminal oracle passes")
    tasks_dir = args.tasks_dir
    sft_targets: dict[str, str] | None = None
    temporary: TemporaryDirectory[str] | None = None
    if args.curriculum == BANK_ACCOUNT_CURRICULUM:
        temporary = TemporaryDirectory(prefix="glm47-bank-account-rubrics-")
        tasks_dir = temporary.name
        build_bank_account_curriculum(
            args.tasks_dir,
            tasks_dir,
            compiler=os.environ.get("CXX", "c++"),
            require_gcc=not args.allow_non_gcc_curriculum,
        )
    elif args.curriculum == BANK_ACCOUNT_OFFICIAL_DRILL:
        temporary = TemporaryDirectory(prefix="glm47-bank-account-official-rubrics-")
        tasks_dir = temporary.name
        build_bank_account_official_drill(
            tasks_dir,
            compiler=os.environ.get("CXX", "g++"),
            require_gcc=not args.allow_non_gcc_curriculum,
        )
        sft_targets = imitation_targets()
    try:
        paths = build_aider_polyglot_datasets(
            tasks_dir,
            args.out,
            train_limit=args.train_limit,
            monitor_limit=args.eval_limit or 32,
            profile=args.profile,
            run_id=args.run_id,
            sort_by_size=args.sort_by_size,
            force=args.force,
            imitation_targets=sft_targets,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
