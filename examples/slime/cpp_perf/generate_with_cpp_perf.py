"""SLIME reward hook for PIE-derived C++ performance tasks."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from w8_biayn.cpp_perf.reward import compute_reward  # noqa: E402
from w8_biayn.cpp_perf.sandbox import DEFAULT_CPU, DEFAULT_DOCKER_IMAGE, run_in_sandbox  # noqa: E402
from w8_biayn.cpp_perf.schema import CppTask  # noqa: E402


def _resolve_task_path(args: Any, sample: Any) -> Path:
    metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
    raw_path = metadata.get("task_path") or sample.label
    if not raw_path:
        raise ValueError("SLIME C++ sample is missing metadata.task_path")

    path = Path(str(raw_path)).expanduser()
    if path.is_absolute() and path.exists():
        return path

    candidates: list[Path] = []
    if task_root := os.environ.get("W8_BIAYN_SLIME_TASK_ROOT"):
        candidates.append(Path(task_root).expanduser())
    if prompt_data := getattr(args, "prompt_data", None):
        prompt_parent = Path(str(prompt_data)).expanduser().resolve(strict=False).parent
        candidates.extend([prompt_parent, prompt_parent.parent])
    candidates.extend([REPO_ROOT, Path.cwd()])

    for base in candidates:
        candidate = base / path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(base / path) for base in candidates)
    raise FileNotFoundError(f"could not resolve task path {raw_path!r}; searched: {searched}")


def _score_sample(args: Any, sample: Any) -> dict[str, Any]:
    task = CppTask.read_json(_resolve_task_path(args, sample))
    image = os.environ.get("SLIME_CPP_SANDBOX_IMAGE", DEFAULT_DOCKER_IMAGE)
    cpu = os.environ.get("SLIME_CPP_SANDBOX_CPU", DEFAULT_CPU)

    def runner(candidate_task: CppTask, code: str):
        return run_in_sandbox(candidate_task, code, image=image, cpu=cpu)

    breakdown = compute_reward(task, sample.response, runner=runner)
    payload: dict[str, Any] = {
        "score": breakdown.reward,
        "reason": breakdown.reason,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "format_valid": breakdown.format_valid,
    }
    if breakdown.harness is not None:
        payload.update(
            {
                "tests_passed": breakdown.harness.tests_passed,
                "tests_total": breakdown.harness.tests_total,
                "compile_error": breakdown.harness.compile_error,
                "sanitizer_error": breakdown.harness.sanitizer_error,
                "timeout": breakdown.harness.timeout,
                "runtime_cpu_ns": breakdown.harness.runtime_cpu_ns,
                "reference_runtime_cpu_ns": breakdown.harness.reference_runtime_cpu_ns,
                "runtime_speedup": breakdown.harness.runtime_speedup,
            }
        )
    return payload


async def reward_func(args: Any, sample: Any, **_kwargs: Any) -> dict[str, Any]:
    """Return a SLIME reward dict using the repo's C++ reward harness."""

    try:
        return await asyncio.to_thread(_score_sample, args, sample)
    except Exception as exc:
        metadata = sample.metadata if isinstance(getattr(sample, "metadata", None), dict) else {}
        return {
            "score": -1.0,
            "reason": "reward_exception",
            "task_id": metadata.get("task_id") or sample.label or "",
            "exception": str(exc),
        }
