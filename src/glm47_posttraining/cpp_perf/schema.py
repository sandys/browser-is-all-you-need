"""Data models for C++ optimization tasks and harness results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .rubrics import ALL_RUBRICS, PERFORMANCE_INTENSIVE, STANDARD, STATE_CONCURRENCY


LINE_CONTINUATION_BLANK_GAP_RE = re.compile(
    r"(\\[^\S\r\n]*\r?\n)(?:[^\S\r\n]*\r?\n)+"
)


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

    @field_validator("prompt_code", "oracle_solution")
    @classmethod
    def _repair_preprocessor_line_continuations(cls, source: str) -> str:
        """Remove exporter-added blank lines that split a trailing backslash."""

        return LINE_CONTINUATION_BLANK_GAP_RE.sub(r"\1", source)

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


class AiderPolyglotTask(BaseModel):
    """One C++ exercise from Aider's Exercism-derived polyglot benchmark.

    The upstream benchmark is a multi-file editing task.  Keeping solution,
    oracle, test, and support files separate prevents tests from leaking into
    prompts while still making each serialized task self-contained for remote
    reward workers.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    language: Literal["cpp"] = "cpp"
    topic_category: str = "General"
    rubric_category: Literal["standard", "state_concurrency", "performance_intensive"] = "standard"
    rubric_risks: dict[str, float] = Field(default_factory=dict)
    difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    instructions_md: str
    solution_files: dict[str, str]
    oracle_files: dict[str, str] = Field(default_factory=dict)
    test_files: dict[str, str]
    support_files: dict[str, str] = Field(default_factory=dict)
    cmake_lists: str = ""
    link_flags: list[str] = Field(default_factory=list)
    split: Literal["train", "test", "validation"] = "train"
    cyclomatic_complexity: int = Field(default=1, ge=1)
    prompt_token_length: int = Field(default=0, ge=0)
    empirical_failure_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    difficulty_index: float = Field(default=0.5, ge=0.0, le=1.0)
    priority_weight: float = Field(default=1.0, gt=0.0)
    source: str = "Aider-AI/polyglot-benchmark"
    source_revision: str = ""

    @field_validator("solution_files", "test_files")
    @classmethod
    def _require_files(cls, files: dict[str, str]) -> dict[str, str]:
        if not files:
            raise ValueError("polyglot task requires solution and test files")
        return files

    @field_validator("solution_files", "oracle_files", "test_files", "support_files")
    @classmethod
    def _validate_relative_file_paths(cls, files: dict[str, str]) -> dict[str, str]:
        for raw_path in files:
            path = Path(raw_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError(f"unsafe task file path: {raw_path!r}")
        return files

    @field_validator("rubric_risks")
    @classmethod
    def _validate_rubric_risks(cls, risks: dict[str, float]) -> dict[str, float]:
        unknown = set(risks) - set(ALL_RUBRICS)
        if unknown:
            raise ValueError(f"unknown rubric risks: {sorted(unknown)}")
        if any(value < 0.0 for value in risks.values()):
            raise ValueError("rubric risks must be non-negative")
        return risks

    @property
    def problem_id(self) -> str:
        """Compatibility identifier used by the existing Miles/eval plumbing."""

        return self.task_id

    @property
    def effective_rubric_category(self) -> str:
        """Resolve legacy topic-only records into the static rubric category."""

        if self.rubric_category != STANDARD:
            return self.rubric_category
        if self.topic_category == "State & concurrency":
            return STATE_CONCURRENCY
        if self.topic_category == "Performance-intensive":
            return PERFORMANCE_INTENSIVE
        return STANDARD

    @property
    def header_filename(self) -> str:
        return next((name for name in self.solution_files if Path(name).suffix in {".h", ".hpp"}), "")

    @property
    def source_filename(self) -> str:
        return next((name for name in self.solution_files if Path(name).suffix in {".cc", ".cpp", ".cxx"}), "")

    @property
    def header_stub(self) -> str:
        return self.solution_files.get(self.header_filename, "")

    @property
    def source_stub(self) -> str:
        return self.solution_files.get(self.source_filename, "")

    @classmethod
    def read_json(cls, path: str | Path) -> "AiderPolyglotTask":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return output


class Catch2HarnessResult(BaseModel):
    """Compilation, sanitizer, and Catch2 assertion results for one edit."""

    compile_error: bool = False
    sanitizer_error: bool = False
    thread_sanitizer_ran: bool = False
    thread_sanitizer_error: bool = False
    thread_sanitizer_timeout: bool = False
    timeout: bool = False
    runtime_cpu_ns: int | None = Field(default=None, ge=0)
    reference_runtime_cpu_ns: int | None = Field(default=None, ge=0)
    passed_test_cases: int = Field(default=0, ge=0)
    total_test_cases: int = Field(default=0, ge=0)
    passed_assertions: int = Field(default=0, ge=0)
    total_assertions: int = Field(default=0, ge=0)
    logs: dict[str, str] = Field(default_factory=dict)

    @property
    def all_tests_pass(self) -> bool:
        return (
            not self.compile_error
            and not self.sanitizer_error
            and not self.thread_sanitizer_error
            and not self.thread_sanitizer_timeout
            and not self.timeout
            and self.total_test_cases > 0
            and self.passed_test_cases == self.total_test_cases
            and self.total_assertions > 0
            and self.passed_assertions == self.total_assertions
        )

    @property
    def tests_passed(self) -> int:
        return self.passed_test_cases

    @property
    def tests_total(self) -> int:
        return self.total_test_cases

    @property
    def test_case_pass_ratio(self) -> float:
        return self.passed_test_cases / self.total_test_cases if self.total_test_cases else 0.0

    @property
    def assertion_pass_ratio(self) -> float:
        return self.passed_assertions / self.total_assertions if self.total_assertions else 0.0
