"""Pass@1-aligned reward for Aider-style shadow tasks and official evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from glm47_posttraining.cpp_perf.sandbox import SandboxInfrastructureError

from .harness import CandidatePolicyError, run_aider_tests
from .parser import AiderResponseError, ParsedAiderResponse, parse_whole_file_response
from .schema import AiderPolyglotTask, AiderTestResult


Runner = Callable[[Path, dict[str, str]], AiderTestResult]


@dataclass(frozen=True)
class AiderRewardBreakdown:
    reward: float
    reason: str
    parsed: ParsedAiderResponse | None = None
    harness: AiderTestResult | None = None
    infrastructure_error: bool = False


def compute_aider_reward(
    task: AiderPolyglotTask,
    exercise_dir: Path,
    model_output: str,
    *,
    runner: Runner | None = None,
    strict_binary: bool = False,
) -> AiderRewardBreakdown:
    """Score correctness first and use formatting only as a small tie-breaker."""

    try:
        parsed = parse_whole_file_response(model_output, task.editable_files)
    except AiderResponseError as exc:
        if strict_binary:
            return AiderRewardBreakdown(reward=0.0, reason=exc.reason)
        reward = -1.0 if exc.reason == "forbidden_file" else -0.8
        return AiderRewardBreakdown(reward=reward, reason=exc.reason)

    try:
        harness = (runner or run_aider_tests)(exercise_dir, parsed.files)
    except CandidatePolicyError:
        if strict_binary:
            return AiderRewardBreakdown(
                reward=0.0,
                reason="forbidden_runtime_primitive",
                parsed=parsed,
            )
        return AiderRewardBreakdown(
            reward=-1.0,
            reason="forbidden_runtime_primitive",
            parsed=parsed,
        )
    except SandboxInfrastructureError as exc:
        return AiderRewardBreakdown(
            reward=0.0,
            reason="infrastructure_error",
            parsed=parsed,
            harness=AiderTestResult(status="infrastructure_error", logs={"error": str(exc)}),
            infrastructure_error=True,
        )

    if strict_binary:
        passed = parsed.format_valid and harness.all_tests_pass
        reason = "passed" if passed else harness.status
        if not parsed.format_valid:
            reason = f"recoverable_format_{reason}"
        return AiderRewardBreakdown(
            reward=1.0 if passed else 0.0,
            reason=reason,
            parsed=parsed,
            harness=harness,
        )

    semantic = {
        "compile_failed": -0.5,
        "candidate_timeout": -0.5,
        "infrastructure_error": 0.0,
    }.get(harness.status)
    if semantic is None:
        semantic = 1.0 if harness.all_tests_pass else 0.6 * harness.fraction_tests_passed

    format_penalty = 0.0 if parsed.format_valid else 0.1
    reward = max(-1.0, min(1.0, semantic - format_penalty))
    reason = "passed" if harness.all_tests_pass else harness.status
    if not parsed.format_valid:
        reason = f"recoverable_format_{reason}"
    return AiderRewardBreakdown(reward=reward, reason=reason, parsed=parsed, harness=harness)
