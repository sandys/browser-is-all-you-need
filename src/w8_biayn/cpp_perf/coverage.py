"""Coverage-gate helpers for PIE-derived C++ tasks."""

from __future__ import annotations

import re

from .schema import TestCoverage


LINE_THRESHOLD = 0.95
BRANCH_THRESHOLD = 0.85


def coverage_passes(coverage: TestCoverage) -> bool:
    """Return True when the suite clears the Phase-1 coverage gate."""

    return coverage.line >= LINE_THRESHOLD and coverage.branch >= BRANCH_THRESHOLD


def parse_lcov_summary(text: str) -> TestCoverage:
    """Parse `lcov --summary` style output into normalized coverage values."""

    line = _percent_for_label(text, "lines")
    branch = _percent_for_label(text, "branches")
    return TestCoverage(line=line, branch=branch)


def _percent_for_label(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\.*:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not match:
        raise ValueError(f"Could not parse {label} coverage from summary")
    return float(match.group(1)) / 100.0
