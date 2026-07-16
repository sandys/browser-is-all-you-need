"""Score generated C++ optimization candidates."""

from __future__ import annotations

from typing import Any

from glm47_posttraining.cpp_perf.reward import compute_reward
from glm47_posttraining.cpp_perf.sandbox import run_in_sandbox
from glm47_posttraining.cpp_perf.schema import CppTask


def score_generation(
    task: CppTask,
    model_output: str,
    *,
    label: str,
    sample_index: int,
    image: str,
    cpu: str,
) -> dict[str, Any]:
    """Score one generated answer and flatten key harness fields."""

    def runner(candidate_task: CppTask, code: str):
        return run_in_sandbox(candidate_task, code, image=image, cpu=cpu)

    breakdown = compute_reward(task, model_output, runner=runner)
    harness = breakdown.harness
    record: dict[str, Any] = {
        "label": label,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "sample_index": sample_index,
        "reward": breakdown.reward,
        "reason": breakdown.reason,
        "reference_metric": task.reference.metric,
        "reference_value": task.reference.value,
        "runtime_cpu_ns": harness.runtime_cpu_ns if harness else None,
        "runtime_wall_ns": harness.runtime_wall_ns if harness else None,
        "reference_runtime_cpu_ns": harness.reference_runtime_cpu_ns if harness else None,
        "reference_runtime_wall_ns": harness.reference_runtime_wall_ns if harness else None,
        "runtime_speedup": harness.runtime_speedup if harness else None,
        "all_tests_pass": harness.all_tests_pass if harness else False,
        "compile_error": harness.compile_error if harness else False,
        "sanitizer_error": harness.sanitizer_error if harness else False,
        "timeout": harness.timeout if harness else False,
        "tests_passed": harness.tests_passed if harness else 0,
        "tests_total": harness.tests_total if harness else 0,
    }
    return record
