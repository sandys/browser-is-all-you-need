"""Correctness-gated runtime-efficiency reward for C++ performance RL."""

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
MISSING_RUNTIME_REWARD = 0.5
RECOVERABLE_FORMAT_COMPILE_REWARD = -0.75
RECOVERABLE_FORMAT_MISSING_RUNTIME_REWARD = 0.2
RECOVERABLE_FORMAT_CORRECT_BASE = 0.1
RECOVERABLE_FORMAT_CORRECT_GAIN = 0.3
CODE_BLOCK_RE = re.compile(
    r"```(?:(?:cpp|c\+\+|cc)(?:[ \t]+|\s*\n)|\s*\n)(.*?)```",
    flags=re.DOTALL,
)
CPP_MAIN_RE = re.compile(r"\b(?:int|signed|auto)\s+main\s*\(")
CPP_MARKER_RE = re.compile(
    r"(^\s*#\s*include\b|^\s*#\s*define\b|\busing\s+namespace\s+std\b|\bstd::|\btemplate\s*<|\bvector\s*<|\blong\s+long\b)",
    flags=re.MULTILINE,
)
CPP_FIRST_LINE_RE = re.compile(
    r"\s*(#\s*(?:include|define)|//|/\*|using\b|typedef\b|template\b|namespace\b|class\b|struct\b|static\b|const\b|"
    r"int\b|long\b|signed\b|void\b|auto\b)"
)

Runner = Callable[[CppTask, str], HarnessResult]


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    reason: str
    harness: HarnessResult | None = None
    code: str | None = None
    format_valid: bool = False


def valid_model_output(model_output: str) -> bool:
    """Require a reasoning block followed by exactly one fenced code block."""

    try:
        extract_code_block(model_output)
    except ValueError:
        return False
    reasoning_matches = list(re.finditer(r"<reasoning>.*?</reasoning>", model_output, flags=re.DOTALL))
    if len(reasoning_matches) != 1:
        return False
    code_match = CODE_BLOCK_RE.search(model_output)
    return code_match is not None and reasoning_matches[0].end() <= code_match.start()


def extract_code_block(model_output: str) -> str:
    """Extract the only fenced code block from a model response."""

    matches = CODE_BLOCK_RE.findall(model_output)
    if len(matches) != 1:
        raise ValueError("model output must contain exactly one fenced C++ code block")
    return matches[0].strip() + "\n"


def extract_reward_code(model_output: str) -> tuple[str, bool]:
    """Extract C++ for scoring and report whether the strict model format was used."""

    if valid_model_output(model_output):
        return extract_code_block(model_output), True
    return extract_recoverable_code(model_output), False


def extract_recoverable_code(model_output: str) -> str:
    """Extract C++ from recoverable-but-invalid outputs for shaped GRPO rewards."""

    matches = CODE_BLOCK_RE.findall(model_output)
    if len(matches) == 1:
        return matches[0].strip() + "\n"
    if len(matches) > 1:
        raise ValueError("model output contains multiple fenced C++ code blocks")

    stripped = model_output.strip()
    if not _looks_like_cpp_submission(stripped):
        raise ValueError("model output does not contain recoverable C++")
    return stripped + "\n"


def compute_reward(
    task: CppTask,
    model_output: str,
    *,
    runner: Runner | None = None,
) -> RewardBreakdown:
    """Compute the Phase-1 reward from model output and sandbox result."""

    try:
        code, format_valid = extract_reward_code(model_output)
    except ValueError:
        return RewardBreakdown(reward=-1.0, reason="invalid_format")

    harness = (runner or run_in_sandbox)(task, code)
    if not format_valid:
        return _recoverable_format_reward(harness, code=code)
    return _strict_format_reward(harness, code=code)


def file_state_reward(harness: HarnessResult, *, code: str) -> RewardBreakdown:
    """Correctness-gated child-CPU-ns reward for a graded candidate FILE.

    The agentic lane grades the final ``candidate.cpp`` the model left in the
    sandbox, so there is no response-format axis to score: the reward is exactly
    the strict correctness ladder over the harness result. Reusing
    ``_strict_format_reward`` keeps one source of truth for the ladder shared
    with the single-turn lane; ``format_valid`` on the returned breakdown is a
    dead field here (always ``True``). ``code`` is the final file contents,
    carried through for logging/debugging only.
    """

    return _strict_format_reward(harness, code=code)


def _strict_format_reward(harness: HarnessResult, *, code: str) -> RewardBreakdown:
    if harness.compile_error:
        return RewardBreakdown(reward=-0.5, reason="compile_error", harness=harness, code=code, format_valid=True)
    if harness.sanitizer_error:
        return RewardBreakdown(reward=-0.5, reason="sanitizer_error", harness=harness, code=code, format_valid=True)
    if harness.timeout:
        return RewardBreakdown(reward=-0.5, reason="timeout", harness=harness, code=code, format_valid=True)
    if not harness.all_tests_pass:
        shaped = -0.2 + 0.2 * harness.fraction_tests_passed
        return RewardBreakdown(reward=shaped, reason="tests_failed", harness=harness, code=code, format_valid=True)
    if harness.runtime_cpu_ns is None or harness.reference_runtime_cpu_ns is None:
        return RewardBreakdown(
            reward=MISSING_RUNTIME_REWARD,
            reason="missing_runtime",
            harness=harness,
            code=code,
            format_valid=True,
        )

    efficiency = _runtime_efficiency(harness)
    return RewardBreakdown(
        reward=BETA_C * 1.0 + BETA_E * efficiency,
        reason="correct",
        harness=harness,
        code=code,
        format_valid=True,
    )


def _recoverable_format_reward(harness: HarnessResult, *, code: str) -> RewardBreakdown:
    if harness.compile_error:
        return RewardBreakdown(
            reward=RECOVERABLE_FORMAT_COMPILE_REWARD,
            reason="recoverable_format_compile_error",
            harness=harness,
            code=code,
            format_valid=False,
        )
    if harness.sanitizer_error:
        return RewardBreakdown(
            reward=RECOVERABLE_FORMAT_COMPILE_REWARD,
            reason="recoverable_format_sanitizer_error",
            harness=harness,
            code=code,
            format_valid=False,
        )
    if harness.timeout:
        return RewardBreakdown(
            reward=RECOVERABLE_FORMAT_COMPILE_REWARD,
            reason="recoverable_format_timeout",
            harness=harness,
            code=code,
            format_valid=False,
        )
    if not harness.all_tests_pass:
        shaped = -0.4 + 0.4 * harness.fraction_tests_passed
        return RewardBreakdown(
            reward=shaped,
            reason="recoverable_format_tests_failed",
            harness=harness,
            code=code,
            format_valid=False,
        )
    if harness.runtime_cpu_ns is None or harness.reference_runtime_cpu_ns is None:
        return RewardBreakdown(
            reward=RECOVERABLE_FORMAT_MISSING_RUNTIME_REWARD,
            reason="recoverable_format_missing_runtime",
            harness=harness,
            code=code,
            format_valid=False,
        )

    efficiency = _runtime_efficiency(harness)
    return RewardBreakdown(
        reward=RECOVERABLE_FORMAT_CORRECT_BASE + RECOVERABLE_FORMAT_CORRECT_GAIN * efficiency,
        reason="recoverable_format_correct",
        harness=harness,
        code=code,
        format_valid=False,
    )


def _runtime_efficiency(harness: HarnessResult) -> float:
    base = float(harness.reference_runtime_cpu_ns)
    candidate = float(harness.runtime_cpu_ns)
    rel_gain = (base - candidate) / (base + EPS)
    rel_gain = min(max(rel_gain, 0.0), GAIN_CAP)
    return math.tanh(SCALE * rel_gain)


def _looks_like_cpp_submission(text: str) -> bool:
    if not text:
        return False
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    return bool(CPP_FIRST_LINE_RE.match(first_line) and CPP_MAIN_RE.search(text) and CPP_MARKER_RE.search(text))
