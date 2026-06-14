"""PIE TSV ingestion for C++ performance-RL tasks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from .schema import BuildConfig, CppTask, ReferencePerformance, TestCase, TestCoverage


Split = Literal["train", "test", "validation"]


CODE_V0_ALIASES = ("v0", "code_v0", "src_v0", "program_v0", "submission_v0")
CODE_V1_ALIASES = ("v1", "code_v1", "src_v1", "program_v1", "submission_v1")
PROBLEM_ALIASES = ("problem_id", "problem", "problem_name", "p_id")
CPU_V1_ALIASES = ("cpu_time_v1", "gem5_time_v1", "runtime_v1")
CPU_V0_ALIASES = ("cpu_time_v0", "gem5_time_v0", "runtime_v0")
STATUS_V0_ALIASES = ("status_v0", "result_v0")


@dataclass(frozen=True)
class PiePair:
    problem_id: str
    prompt_code: str
    oracle_solution: str
    reference_value: int
    gem5_cycles: int | None
    status_v0: str


@dataclass
class TaskBuildReport:
    """Admission report for PIE rows converted into task JSON."""

    split: Split
    rows_seen: int = 0
    tasks_built: int = 0
    missing_tests: list[str] = field(default_factory=list)
    invalid_tasks: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "rows_seen": self.rows_seen,
            "tasks_built": self.tasks_built,
            "missing_tests": self.missing_tests,
            "invalid_tasks": self.invalid_tasks,
        }


def read_pie_pairs(
    path: str | Path,
    *,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> list[PiePair]:
    """Read PIE trajectory rows from TSV or JSONL files."""

    rows: list[PiePair] = []
    source = Path(path)
    if source.suffix.lower() in {".jsonl", ".json"}:
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rows.append(pair_from_row(json.loads(line)))
                except ValueError:
                    if not skip_invalid:
                        raise
                if limit is not None and len(rows) >= limit:
                    break
        return rows

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            try:
                rows.append(pair_from_row(raw))
            except ValueError:
                if not skip_invalid:
                    raise
            if limit is not None and len(rows) >= limit:
                break
    return rows


def pair_from_row(row: dict[str, str]) -> PiePair:
    """Normalize one PIE TSV row into the fields this project needs."""

    prompt_code = _required(row, CODE_V0_ALIASES + ("input", "code_v0_no_empty_lines"))
    oracle_solution = _required(row, CODE_V1_ALIASES + ("target", "code_v1_no_empty_lines"))
    problem_id = _required(row, PROBLEM_ALIASES)
    reference_value = _positive_int(_required(row, CPU_V1_ALIASES), "cpu_time_v1")
    gem5_cycles = _optional_positive_int(_first_present(row, CPU_V1_ALIASES))
    status_v0 = _first_present(row, STATUS_V0_ALIASES) or "Accepted"
    return PiePair(
        problem_id=problem_id,
        prompt_code=prompt_code,
        oracle_solution=oracle_solution,
        reference_value=reference_value,
        gem5_cycles=gem5_cycles,
        status_v0=status_v0,
    )


def load_tests_manifest(path: str | Path) -> dict[str, dict[str, object]]:
    """Load a problem-id keyed tests manifest.

    Shape:
    {
      "p03050": {
        "unit_tests": [{"input": "...", "expected": "..."}],
        "hidden_tests": [{"input": "...", "expected": "..."}],
        "coverage": {"line": 0.96, "branch": 0.87}
      }
    }
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tests manifest must be a JSON object keyed by problem_id")
    return data


def build_tasks(
    pairs: Iterable[PiePair],
    tests_manifest: dict[str, dict[str, object]],
    *,
    split: Split,
    task_id_prefix: str = "pie_cpp",
    compiler_flags: str = "-O3 -std=c++20",
) -> list[CppTask]:
    """Build validated tasks, skipping problems that lack tests or coverage."""

    tasks, _report = build_tasks_with_report(
        pairs,
        tests_manifest,
        split=split,
        task_id_prefix=task_id_prefix,
        compiler_flags=compiler_flags,
    )
    return tasks


def build_tasks_with_report(
    pairs: Iterable[PiePair],
    tests_manifest: dict[str, dict[str, object]],
    *,
    split: Split,
    task_id_prefix: str = "pie_cpp",
    compiler_flags: str = "-O3 -std=c++20",
) -> tuple[list[CppTask], TaskBuildReport]:
    """Build validated tasks and record why rows were rejected."""

    tasks: list[CppTask] = []
    report = TaskBuildReport(split=split)
    for index, pair in enumerate(pairs, start=1):
        report.rows_seen += 1
        tests = tests_manifest.get(pair.problem_id)
        if not tests:
            report.missing_tests.append(pair.problem_id)
            continue
        task_id = f"{task_id_prefix}_{index:06d}"
        try:
            task = task_from_pair(
                pair,
                tests,
                task_id=task_id,
                split=split,
                compiler_flags=compiler_flags,
            )
        except ValueError as exc:
            report.invalid_tasks.append({"problem_id": pair.problem_id, "reason": str(exc)})
            continue
        tasks.append(task)
    report.tasks_built = len(tasks)
    return tasks, report


def task_from_pair(
    pair: PiePair,
    tests: dict[str, object],
    *,
    task_id: str,
    split: Split,
    compiler_flags: str = "-O3 -std=c++20",
) -> CppTask:
    unit_tests = [TestCase.model_validate(item) for item in _list_field(tests, "unit_tests")]
    hidden_tests = [TestCase.model_validate(item) for item in _list_field(tests, "hidden_tests")]
    coverage = TestCoverage.model_validate(tests.get("coverage"))
    return CppTask(
        task_id=task_id,
        problem_id=pair.problem_id,
        prompt_code=pair.prompt_code,
        unit_tests=unit_tests,
        hidden_tests=hidden_tests,
        oracle_solution=pair.oracle_solution,
        test_coverage=coverage,
        reference=ReferencePerformance(
            value=pair.reference_value,
            gem5_cycles=pair.gem5_cycles,
            compiler_flags=compiler_flags,
        ),
        build=BuildConfig(cmd=f"g++ {compiler_flags} candidate.cpp -o candidate"),
        split=split,
    )


def _required(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    value = _first_present(row, aliases)
    if value is None or value == "":
        raise ValueError(f"PIE row missing required column; tried {aliases}")
    return value


def _first_present(row: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _positive_int(value: str, label: str) -> int:
    parsed_float = float(value)
    if parsed_float <= 0:
        raise ValueError(f"{label} must be positive")
    parsed = int(parsed_float)
    if parsed <= 0:
        parsed = int(round(parsed_float * 1_000_000))
    if parsed <= 0:
        parsed = 1
    return parsed


def _optional_positive_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    parsed_float = float(value)
    if parsed_float <= 0:
        return None
    parsed = int(parsed_float)
    if parsed <= 0:
        parsed = int(round(parsed_float * 1_000_000))
    return parsed if parsed > 0 else 1


def _list_field(data: dict[str, object], key: str) -> list[object]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ValueError(f"tests manifest field {key!r} must be a list")
    return value
