"""Programming-contest stdout comparison helpers."""

from __future__ import annotations


def normalize_judge_output(text: str) -> str:
    """Normalize harmless stdout differences accepted by contest judges."""

    lines = [line.rstrip() for line in text.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def judge_output_matches(actual: str, expected: str) -> bool:
    """Return True when stdout differs only by trailing whitespace."""

    return normalize_judge_output(actual) == normalize_judge_output(expected)
