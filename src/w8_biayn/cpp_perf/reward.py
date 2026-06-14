"""Correctness-gated efficiency reward for C++ performance RL."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

from .sandbox import run_in_sandbox
from .schema import CppTask, HarnessResult


EPS = 1e-9
GAIN_CAP = 1.0
SCALE = 2.0
BETA_C = 1.0
BETA_E = 1.0

Runner = Callable[[CppTask, str], HarnessResult]


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    reason: str
    harness: HarnessResult | None = None


def valid_model_output(model_output: str) -> bool:
    """Require a reasoning block followed by exactly one fenced code block."""

    try:
        extract_code_block(model_output)
    except ValueError:
        return False
    reasoning_matches = list(re.finditer(r"<reasoning>.*?</reasoning>", model_output, flags=re.DOTALL))
    if len(reasoning_matches) != 1:
        return False
    code_match = re.search(r"```(?:cpp|c\+\+|cc)?\s*\n.*?```", model_output, flags=re.DOTALL)
    return code_match is not None and reasoning_matches[0].end() <= code_match.start()


def extract_code_block(model_output: str) -> str:
    """Extract the only fenced code block from a model response."""

    matches = re.findall(r"```(?:cpp|c\+\+|cc)?\s*\n(.*?)```", model_output, flags=re.DOTALL)
    if len(matches) != 1:
        raise ValueError("model output must contain exactly one fenced C++ code block")
    return matches[0].strip() + "\n"


def compute_reward(
    task: CppTask,
    model_output: str,
    *,
    runner: Runner | None = None,
) -> RewardBreakdown:
    """Compute the Phase-1 reward from model output and sandbox result."""

    if not valid_model_output(model_output):
        return RewardBreakdown(reward=-1.0, reason="invalid_format")

    code = extract_code_block(model_output)
    harness = (runner or run_in_sandbox)(task, code)
    if harness.compile_error:
        return RewardBreakdown(reward=-0.5, reason="compile_error", harness=harness)
    if harness.sanitizer_error:
        return RewardBreakdown(reward=-0.5, reason="sanitizer_error", harness=harness)
    if not harness.all_tests_pass:
        shaped = -0.2 + 0.2 * harness.fraction_tests_passed
        return RewardBreakdown(reward=shaped, reason="tests_failed", harness=harness)
    if harness.instr_count is None:
        return RewardBreakdown(reward=-0.5, reason="missing_instruction_count", harness=harness)

    base = float(task.reference.value)
    candidate = float(harness.instr_count)
    rel_gain = (base - candidate) / (base + EPS)
    rel_gain = min(max(rel_gain, 0.0), GAIN_CAP)
    efficiency = math.tanh(SCALE * rel_gain)
    return RewardBreakdown(
        reward=BETA_C * 1.0 + BETA_E * efficiency,
        reason="correct",
        harness=harness,
    )
