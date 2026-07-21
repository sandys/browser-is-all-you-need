"""Score generated C++ optimization candidates."""

from __future__ import annotations

from typing import Any

from glm47_posttraining.cpp_perf.reward import compute_aider_reward, compute_reward
from glm47_posttraining.cpp_perf.sandbox import run_aider_in_sandbox, run_in_sandbox
from glm47_posttraining.cpp_perf.schema import AiderPolyglotTask, CppTask


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


def score_aider_generation(
    task: AiderPolyglotTask,
    model_output: str,
    *,
    label: str,
    sample_index: int,
    image: str,
    context_exhausted: bool = False,
) -> dict[str, Any]:
    """Score one Polyglot multi-file edit and flatten Catch2 metrics."""

    breakdown = compute_aider_reward(
        task,
        model_output,
        context_exhausted=context_exhausted,
        runner=lambda current, files: run_aider_in_sandbox(current, files, image=image),
    )
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
        "label": label,
        "task_id": task.task_id,
        "problem_id": task.task_id,
        "split": task.split,
        "topic_category": task.topic_category,
        "rubric_category": task.effective_rubric_category,
        "difficulty": task.difficulty,
        "difficulty_index": task.difficulty_index,
        "sample_index": sample_index,
        "reward": breakdown.reward,
        "reason": breakdown.reason,
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
        "all_tests_pass": verified_all_tests_pass,
        "compile_error": harness.compile_error if harness else False,
        "sanitizer_error": harness.sanitizer_error if harness else False,
        "thread_sanitizer_ran": harness.thread_sanitizer_ran if harness else False,
        "thread_sanitizer_error": harness.thread_sanitizer_error if harness else False,
        "thread_sanitizer_timeout": harness.thread_sanitizer_timeout if harness else False,
        "timeout": harness.timeout if harness else False,
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
    }
