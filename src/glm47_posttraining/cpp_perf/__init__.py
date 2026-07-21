"""C++ performance-RL task, harness, and reward helpers."""

from .reward import (
    AiderRewardBreakdown,
    build_aider_retry_prompt,
    compute_aider_reward,
    compute_multi_try_aider_reward,
    compute_rubric_reward,
    compute_reward,
    extract_aider_files,
    extract_code_block,
    extract_recoverable_code,
    extract_reward_code,
    valid_aider_model_output,
    valid_model_output,
)
from .schema import (
    AiderPolyglotTask,
    BuildConfig,
    Catch2HarnessResult,
    CppTask,
    HarnessResult,
    ReferencePerformance,
    TestCase,
    TestCoverage,
)

__all__ = [
    "AiderPolyglotTask",
    "AiderRewardBreakdown",
    "BuildConfig",
    "Catch2HarnessResult",
    "CppTask",
    "HarnessResult",
    "ReferencePerformance",
    "TestCase",
    "TestCoverage",
    "build_aider_retry_prompt",
    "compute_aider_reward",
    "compute_multi_try_aider_reward",
    "compute_rubric_reward",
    "compute_reward",
    "extract_aider_files",
    "extract_code_block",
    "extract_recoverable_code",
    "extract_reward_code",
    "valid_aider_model_output",
    "valid_model_output",
]
