"""Data models for C++ optimization tasks and harness results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TestCase(BaseModel):
    """One stdin/stdout correctness check."""

    __test__ = False

    input: str
    expected: str


class TestCoverage(BaseModel):
    """Coverage metadata for the reference/oracle test suite."""

    __test__ = False

    line: float = Field(ge=0.0, le=1.0)
    branch: float = Field(ge=0.0, le=1.0)


class ReferencePerformance(BaseModel):
    """Reference performance for the best known correct solution."""

    metric: Literal["runtime_cpu_ns", "cpu_time_ns", "legacy_cpu_time"] = "runtime_cpu_ns"
    value: int = Field(gt=0)
    gem5_cycles: int | None = Field(default=None, gt=0)
    compiler_flags: str = "-O3 -std=c++20"


class BuildConfig(BaseModel):
    """Candidate compile command metadata."""

    cmd: str = "g++ -O3 -std=c++20 candidate.cpp -o candidate"
    timeout_s: int = Field(default=10, gt=0)


class CppTask(BaseModel):
    """A runnable PIE-derived C++ performance optimization task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    problem_id: str
    language: Literal["cpp"] = "cpp"
    prompt_code: str
    unit_tests: list[TestCase]
    hidden_tests: list[TestCase]
    oracle_solution: str
    test_coverage: TestCoverage
    reference: ReferencePerformance
    build: BuildConfig = Field(default_factory=BuildConfig)
    split: Literal["train", "test", "validation"]
    source: str = "PIE"

    @field_validator("unit_tests", "hidden_tests")
    @classmethod
    def _require_tests(cls, tests: list[TestCase]) -> list[TestCase]:
        if not tests:
            raise ValueError("task requires at least one visible and one hidden test")
        return tests

    @field_validator("test_coverage")
    @classmethod
    def _require_coverage_gate(cls, coverage: TestCoverage) -> TestCoverage:
        if coverage.line < 0.95 or coverage.branch < 0.85:
            raise ValueError("task coverage must be at least 95% line and 85% branch")
        return coverage

    @classmethod
    def read_json(cls, path: str | Path) -> "CppTask":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output


class HarnessResult(BaseModel):
    """Result of compiling, testing, and measuring one candidate."""

    compile_error: bool = False
    timeout: bool = False
    sanitizer_error: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    runtime_cpu_ns: int | None = Field(default=None, ge=0)
    runtime_wall_ns: int | None = Field(default=None, ge=0)
    reference_runtime_cpu_ns: int | None = Field(default=None, ge=0)
    reference_runtime_wall_ns: int | None = Field(default=None, ge=0)
    runtime_speedup: float | None = Field(default=None, ge=0.0)
    logs: dict[str, str] = Field(default_factory=dict)

    @property
    def all_tests_pass(self) -> bool:
        return self.tests_total > 0 and self.tests_passed == self.tests_total

    @property
    def fraction_tests_passed(self) -> float:
        if self.tests_total <= 0:
            return 0.0
        return self.tests_passed / self.tests_total

    @classmethod
    def read_json(cls, path: str | Path) -> "HarnessResult":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2, sort_keys=True)
