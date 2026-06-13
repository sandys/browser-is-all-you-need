"""C++ performance-RL task, harness, and reward helpers."""

from .reward import compute_reward, extract_code_block, valid_model_output
from .schema import (
    BuildConfig,
    CppTask,
    HarnessResult,
    ReferencePerformance,
    TestCase,
    TestCoverage,
)

__all__ = [
    "BuildConfig",
    "CppTask",
    "HarnessResult",
    "ReferencePerformance",
    "TestCase",
    "TestCoverage",
    "compute_reward",
    "extract_code_block",
    "valid_model_output",
]
