"""Coverage-gate helpers for PIE-derived C++ tasks."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from .schema import TestCase, TestCoverage


LINE_THRESHOLD = 0.95
BRANCH_THRESHOLD = 0.85


def coverage_passes(coverage: TestCoverage) -> bool:
    """Return True when the suite clears the Phase-1 coverage gate."""

    return coverage.line >= LINE_THRESHOLD and coverage.branch >= BRANCH_THRESHOLD


@dataclass(frozen=True)
class CoverageMeasurement:
    """Result of compiling a reference solution and running tests under gcov."""

    coverage: TestCoverage | None
    tests_total: int
    tests_passed: int
    ok: bool
    reason: str
    logs: str = ""

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "coverage": self.coverage.model_dump() if self.coverage else None,
            "tests_total": self.tests_total,
            "tests_passed": self.tests_passed,
            "ok": self.ok,
            "reason": self.reason,
        }
        if self.logs:
            payload["logs"] = self.logs
        return payload


def measure_cpp_coverage(
    code: str,
    tests: list[TestCase],
    *,
    timeout_s: int = 5,
) -> CoverageMeasurement:
    """Compile code with gcov instrumentation, run tests, and measure coverage."""

    if not tests:
        return CoverageMeasurement(
            coverage=None,
            tests_total=0,
            tests_passed=0,
            ok=False,
            reason="missing_tests",
        )
    with TemporaryDirectory(prefix="w8-cpp-coverage-") as temp:
        return _measure_cpp_coverage_in_dir(code, tests, Path(temp), timeout_s=timeout_s)


def _measure_cpp_coverage_in_dir(
    code: str,
    tests: list[TestCase],
    scratch: Path,
    *,
    timeout_s: int,
) -> CoverageMeasurement:
    source = scratch / "solution.cpp"
    binary = scratch / "solution"
    source.write_text(code, encoding="utf-8")
    compile_proc = subprocess.run(
        [
            "g++",
            "-O0",
            "-g",
            "--coverage",
            "-std=c++20",
            str(source),
            "-o",
            str(binary),
        ],
        cwd=scratch,
        check=False,
        capture_output=True,
        text=True,
    )
    if compile_proc.returncode != 0:
        return CoverageMeasurement(
            coverage=None,
            tests_total=len(tests),
            tests_passed=0,
            ok=False,
            reason="compile_failed",
            logs=compile_proc.stderr or compile_proc.stdout,
        )

    tests_passed = 0
    logs: list[str] = []
    for index, test in enumerate(tests):
        proc = subprocess.run(
            ["timeout", f"{timeout_s}s", str(binary)],
            cwd=scratch,
            input=test.input,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout == test.expected:
            tests_passed += 1
            continue
        logs.append(
            f"test_{index}: returncode={proc.returncode} "
            f"stdout={proc.stdout[:200]!r} stderr={proc.stderr[:200]!r}"
        )
    if tests_passed != len(tests):
        return CoverageMeasurement(
            coverage=None,
            tests_total=len(tests),
            tests_passed=tests_passed,
            ok=False,
            reason="tests_failed",
            logs="\n".join(logs),
        )

    gcov_proc = subprocess.run(
        ["gcov", "-b", "-c", "solution.cpp"],
        cwd=scratch,
        check=False,
        capture_output=True,
        text=True,
    )
    if gcov_proc.returncode != 0:
        return CoverageMeasurement(
            coverage=None,
            tests_total=len(tests),
            tests_passed=tests_passed,
            ok=False,
            reason="gcov_failed",
            logs=gcov_proc.stderr or gcov_proc.stdout,
        )
    try:
        coverage = parse_gcov_file(scratch / "solution.cpp.gcov")
    except ValueError as exc:
        return CoverageMeasurement(
            coverage=None,
            tests_total=len(tests),
            tests_passed=tests_passed,
            ok=False,
            reason="coverage_parse_failed",
            logs=str(exc),
        )
    return CoverageMeasurement(
        coverage=coverage,
        tests_total=len(tests),
        tests_passed=tests_passed,
        ok=coverage_passes(coverage),
        reason="ok" if coverage_passes(coverage) else "coverage_below_threshold",
    )


def parse_lcov_summary(text: str) -> TestCoverage:
    """Parse `lcov --summary` style output into normalized coverage values."""

    line = _percent_for_label(text, "lines")
    branch = _percent_for_label(text, "branches")
    return TestCoverage(line=line, branch=branch)


def parse_gcov_file(path: str | Path) -> TestCoverage:
    """Parse `gcov -b -c` output for line and branch coverage."""

    gcov_path = Path(path)
    if not gcov_path.exists():
        raise ValueError(f"missing gcov file: {gcov_path}")
    executable_lines = 0
    covered_lines = 0
    branches = 0
    covered_branches = 0
    for raw_line in gcov_path.read_text(encoding="utf-8", errors="replace").splitlines():
        prefix = raw_line.split(":", 2)[0].strip()
        if prefix and prefix not in {"-", "====="} and (prefix == "#####" or prefix.isdigit()):
            executable_lines += 1
            if prefix.isdigit() and int(prefix) > 0:
                covered_lines += 1
        stripped = raw_line.strip()
        if stripped.startswith("branch "):
            branches += 1
            if "never executed" in stripped:
                continue
            match = re.search(r"taken\s+([0-9]+)%", stripped)
            if match and int(match.group(1)) > 0:
                covered_branches += 1
            elif "taken 0" not in stripped:
                count_match = re.search(r"taken\s+([0-9]+)", stripped)
                if count_match and int(count_match.group(1)) > 0:
                    covered_branches += 1
    if executable_lines == 0:
        raise ValueError(f"no executable lines in gcov file: {gcov_path}")
    line = covered_lines / executable_lines
    branch = 1.0 if branches == 0 else covered_branches / branches
    return TestCoverage(line=line, branch=branch)


def _percent_for_label(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\.*:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not match:
        raise ValueError(f"Could not parse {label} coverage from summary")
    return float(match.group(1)) / 100.0
