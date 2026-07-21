"""Correctness-gated runtime-efficiency reward for C++ performance RL."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from .sandbox import run_aider_in_sandbox, run_in_sandbox
from .rubrics import ALL_RUBRICS, active_rubrics, normalize_risks, weighted_rubric_reward
from .schema import AiderPolyglotTask, Catch2HarnessResult, CppTask, HarnessResult


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
    """Compute the correctness-gated efficiency reward."""

    try:
        code, format_valid = extract_reward_code(model_output)
    except ValueError:
        return RewardBreakdown(reward=-1.0, reason="invalid_format")

    harness = (runner or run_in_sandbox)(task, code)
    if not format_valid:
        return _recoverable_format_reward(harness, code=code)
    return _strict_format_reward(harness, code=code)


def _strict_format_reward(harness: HarnessResult, *, code: str) -> RewardBreakdown:
    if harness.compile_error:
        return RewardBreakdown(reward=-0.5, reason="compile_error", harness=harness, code=code, format_valid=True)
    if harness.sanitizer_error:
        return RewardBreakdown(reward=-0.5, reason="sanitizer_error", harness=harness, code=code, format_valid=True)
    if harness.timeout:
        return RewardBreakdown(reward=-0.5, reason="timeout", harness=harness, code=code, format_valid=True)
    functional_tests_pass = (
        harness.total_test_cases > 0
        and harness.passed_test_cases == harness.total_test_cases
        and harness.total_assertions > 0
        and harness.passed_assertions == harness.total_assertions
    )
    if not functional_tests_pass:
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


AIDER_FILE_MARKER_RE = re.compile(r"^// ===== FILE: ([^\r\n]+?) =====[ \t]*$", re.MULTILINE)
REASONING_CONTENT_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
REASONING_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
HEAVY_BY_VALUE_RE = re.compile(
    r"(?:\(|,)\s*(?:const\s+)?std::(?:string|vector\s*<[^()>{};]+>|"
    r"map\s*<[^()>{};]+>|unordered_map\s*<[^()>{};]+>|set\s*<[^()>{};]+>|"
    r"unordered_set\s*<[^()>{};]+>)\s+\w+\s*(?=[,)=])",
    re.MULTILINE,
)
RAW_ALLOCATION_RE = re.compile(r"\bnew\s+(?!\()|\bdelete(?:\s*\[\s*\])?\s+")
CONST_REFERENCE_RE = re.compile(
    r"\bconst\s+(?:std::)?[\w:]+(?:\s*<[^;(){}]+>)?\s*&|\bconst\s+auto\s*&"
)
REASONING_CODE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:#\s*include\b|// ===== FILE:|(?:class|struct|namespace)\s+\w+\s*\{)",
    re.MULTILINE,
)
AiderRunner = Callable[[AiderPolyglotTask, dict[str, str]], Catch2HarnessResult]


@dataclass(frozen=True)
class AiderRewardBreakdown:
    reward: float
    reason: str
    harness: Catch2HarnessResult | None = None
    files: dict[str, str] | None = None
    code: str | None = None
    format_valid: bool = False
    rubric_scores: dict[str, float] = field(default_factory=dict)
    rubric_weights: dict[str, float] = field(default_factory=dict)
    rubric_risks: dict[str, float] = field(default_factory=dict)
    active_rubrics: tuple[str, ...] = ()
    reasoning_tokens: int | None = None
    observations: tuple[str, ...] = ()


def extract_aider_files(model_output: str, expected_files: set[str]) -> dict[str, str]:
    """Parse the complete multi-file edit encoded in one fenced C++ block."""

    code = extract_code_block(model_output)
    matches = list(AIDER_FILE_MARKER_RE.finditer(code))
    if not matches:
        raise ValueError("submission contains no // ===== FILE: ... ===== markers")
    files: dict[str, str] = {}
    for index, match in enumerate(matches):
        name = match.group(1).strip()
        if name in files:
            raise ValueError(f"duplicate candidate file: {name}")
        start = match.end()
        if code[start : start + 2] == "\r\n":
            start += 2
        elif code[start : start + 1] == "\n":
            start += 1
        end = matches[index + 1].start() if index + 1 < len(matches) else len(code)
        files[name] = code[start:end].rstrip() + "\n"
    if set(files) != expected_files:
        missing = sorted(expected_files - set(files))
        extra = sorted(set(files) - expected_files)
        raise ValueError(f"candidate file mismatch; missing={missing}, extra={extra}")
    return files


def valid_aider_model_output(model_output: str, task: AiderPolyglotTask) -> bool:
    if not valid_model_output(model_output):
        return False
    try:
        extract_aider_files(model_output, set(task.solution_files))
    except ValueError:
        return False
    return True


def compute_aider_reward(
    task: AiderPolyglotTask,
    model_output: str,
    *,
    runner: AiderRunner | None = None,
    reasoning_token_count: int | None = None,
    context_exhausted: bool = False,
) -> AiderRewardBreakdown:
    """Compatibility entry point for the rubric-scored Polyglot reward."""

    return compute_rubric_reward(
        task,
        model_output,
        runner=runner,
        reasoning_token_count=reasoning_token_count,
        context_exhausted=context_exhausted,
    )


def compute_rubric_reward(
    task: AiderPolyglotTask,
    model_output: str,
    *,
    runner: AiderRunner | None = None,
    reasoning_token_count: int | None = None,
    context_exhausted: bool = False,
) -> AiderRewardBreakdown:
    """Compute the five evidence-backed rubric components.

    Reasoning counts use a supplied tokenizer count when the rollout runtime
    provides one, otherwise a deterministic lexical approximation. Test,
    ASan/UBSan, and TSan components come only from executed sandbox evidence.
    """

    category = task.effective_rubric_category
    selected_rubrics = active_rubrics(category)
    risks = (
        {
            name: max(0.0, float(task.rubric_risks.get(name, 0.0)))
            for name in selected_rubrics
        }
        if task.rubric_risks
        else {name: 1.0 for name in selected_rubrics}
    )
    weights = normalize_risks(category, risks)
    empty_scores = {name: 0.0 for name in ALL_RUBRICS}

    def finish(
        *,
        scores: dict[str, float],
        reason: str,
        observations: tuple[str, ...],
        fixed_reward: float | None = None,
        **kwargs: object,
    ) -> AiderRewardBreakdown:
        return AiderRewardBreakdown(
            reward=(
                fixed_reward
                if fixed_reward is not None
                else weighted_rubric_reward(scores, weights)
            ),
            reason=reason,
            rubric_scores=scores,
            rubric_weights=weights,
            rubric_risks=risks,
            active_rubrics=selected_rubrics,
            observations=observations,
            **kwargs,
        )

    if context_exhausted:
        scores = {**empty_scores, "reasoning": -0.5}
        return finish(
            scores=scores,
            reason="rubric1_context_exhausted",
            reasoning_tokens=reasoning_token_count,
            observations=("context_exhausted",),
        )

    if not valid_model_output(model_output):
        scores = {**empty_scores, "reasoning": -1.0}
        return finish(
            scores=scores,
            reason="rubric1_invalid_format",
            fixed_reward=-1.0,
            observations=("invalid_format",),
        )
    try:
        code = extract_code_block(model_output)
        files = extract_aider_files(model_output, set(task.solution_files))
    except ValueError:
        scores = {**empty_scores, "reasoning": -1.0}
        return finish(
            scores=scores,
            reason="rubric1_invalid_format",
            fixed_reward=-1.0,
            observations=("invalid_multifile_format",),
        )

    reasoning_match = REASONING_CONTENT_RE.search(model_output)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    token_count = (
        reasoning_token_count
        if reasoning_token_count is not None
        else len(REASONING_TOKEN_RE.findall(reasoning))
    )
    observations: list[str] = []
    if REASONING_CODE_MARKER_RE.search(reasoning):
        s1_score = -0.5
        observations.append("code_repeated_inside_reasoning")
    elif token_count > 1000:
        s1_score = -0.5
        observations.append("reasoning_over_1000_tokens")
    elif token_count < 400:
        s1_score = 0.1
        observations.append("concise_reasoning")
    else:
        s1_score = 0.0
    scores = {**empty_scores, "reasoning": s1_score}

    combined_code = "\n".join(files.values())
    starter_code = "\n".join(task.solution_files.values())
    introduced_heavy_copies = max(
        0,
        len(HEAVY_BY_VALUE_RE.findall(combined_code))
        - len(HEAVY_BY_VALUE_RE.findall(starter_code)),
    )
    introduced_raw_allocations = max(
        0,
        len(RAW_ALLOCATION_RE.findall(combined_code))
        - len(RAW_ALLOCATION_RE.findall(starter_code)),
    )
    if introduced_heavy_copies or introduced_raw_allocations:
        scores["cpp_quality"] = -0.1
        if introduced_heavy_copies:
            observations.append("introduced_heavy_pass_by_value")
        if introduced_raw_allocations:
            observations.append("introduced_raw_allocation")
    else:
        scores["cpp_quality"] = 0.1
        observations.append(
            "const_reference_present"
            if CONST_REFERENCE_RE.search(combined_code)
            else "no_new_static_cpp_antipatterns"
        )

    harness = (runner or run_aider_in_sandbox)(task, files)
    common = {
        "harness": harness,
        "files": files,
        "code": code,
        "format_valid": True,
        "reasoning_tokens": token_count,
    }
    if harness.compile_error:
        scores["memory_safety"] = -0.5
        return finish(
            scores=scores,
            reason="rubric3_compile_error",
            observations=tuple([*observations, "compile_error"]),
            **common,
        )
    if harness.timeout:
        scores["correctness"] = -0.5
        return finish(
            scores=scores,
            reason="rubric2_timeout",
            observations=tuple([*observations, "execution_timeout"]),
            **common,
        )
    scores["memory_safety"] = -0.5 if harness.sanitizer_error else 0.1
    functional_tests_pass = (
        harness.total_test_cases > 0
        and harness.passed_test_cases == harness.total_test_cases
        and harness.total_assertions > 0
        and harness.passed_assertions == harness.total_assertions
    )
    if functional_tests_pass:
        scores["correctness"] = 1.0
    else:
        test_fraction = 0.5 * harness.test_case_pass_ratio + 0.5 * harness.assertion_pass_ratio
        scores["correctness"] = -0.2 + 0.7 * test_fraction

    if harness.sanitizer_error:
        return finish(
            scores=scores,
            reason="rubric3_sanitizer_error",
            observations=tuple([*observations, "asan_or_ubsan_error"]),
            **common,
        )
    if not functional_tests_pass:
        return finish(
            scores=scores,
            reason="rubric2_tests_failed",
            observations=tuple([*observations, "tests_failed"]),
            **common,
        )

    if "thread_safety" in selected_rubrics:
        if harness.thread_sanitizer_timeout:
            scores["thread_safety"] = -0.5
            observations.append("thread_sanitizer_timeout_or_deadlock")
        elif harness.thread_sanitizer_error:
            scores["thread_safety"] = -0.5
            observations.append("thread_sanitizer_error")
        elif harness.thread_sanitizer_ran:
            scores["thread_safety"] = 0.1
            observations.append("thread_sanitizer_clean")
        else:
            observations.append("thread_sanitizer_not_run")

    if "runtime" in selected_rubrics:
        candidate_runtime = harness.runtime_cpu_ns
        reference_runtime = harness.reference_runtime_cpu_ns
        if candidate_runtime is not None and reference_runtime is not None:
            scores["runtime"] = 0.1 if candidate_runtime < reference_runtime else 0.0
            observations.append("runtime_faster" if scores["runtime"] > 0.0 else "runtime_not_faster")
        else:
            observations.append("runtime_not_measured")

    if scores["thread_safety"] < 0.0:
        reason = "rubric5_thread_safety_failed"
    elif "thread_safety" in selected_rubrics and not harness.thread_sanitizer_ran:
        reason = "correct_thread_sanitizer_not_run"
    else:
        reason = "correct_all_rubrics_passed"
    return finish(
        scores=scores,
        reason=reason,
        observations=tuple(observations),
        **common,
    )


def compute_multi_try_aider_reward(
    task: AiderPolyglotTask,
    model_output: str,
    *,
    attempt_index: int = 1,
    runner: AiderRunner | None = None,
) -> AiderRewardBreakdown:
    """Apply the requested 1.0/0.8/0.5 credit schedule to recovered solutions."""

    if attempt_index < 1:
        raise ValueError("attempt_index must be >= 1")
    result = compute_aider_reward(task, model_output, runner=runner)
    if result.reason != "correct_all_rubrics_passed":
        return result
    discount = 1.0 if attempt_index == 1 else (0.8 if attempt_index == 2 else 0.5)
    return AiderRewardBreakdown(
        reward=result.reward * discount,
        reason=f"correct_try{attempt_index}",
        harness=result.harness,
        files=result.files,
        code=result.code,
        format_valid=result.format_valid,
        rubric_scores=result.rubric_scores,
        rubric_weights=result.rubric_weights,
        rubric_risks=result.rubric_risks,
        active_rubrics=result.active_rubrics,
        reasoning_tokens=result.reasoning_tokens,
        observations=result.observations,
    )


def build_aider_retry_prompt(task: AiderPolyglotTask, error_log: str) -> str:
    """Build the next user turn for pass@k recovery without unbounded logs."""

    return (
        f"Your previous edit for `{task.task_id}` failed. The sandbox reported:\n\n"
        f"```text\n{error_log[-4000:]}\n```\n\n"
        "Fix the implementation and return all required files again using the same "
        "single-block FILE marker format."
    )
