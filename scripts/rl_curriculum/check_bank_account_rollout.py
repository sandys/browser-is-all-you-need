#!/usr/bin/env python3
"""Apply issue #110 hard gates to a no-update bank-account reward JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SAMPLES = 8
EXPECTED_GROUPS = 40
MINIMUM_MIXED_FRACTION = 0.30
HIGH_LOAD_RATE_DELTA = 0.10
# Mirrors THREAD_EXHAUSTION_MARKERS in glm47_posttraining.aider_polyglot.harness;
# this script stays stdlib-only so it can audit pulled JSONLs anywhere.
THREAD_EXHAUSTION_MARKERS = (
    "resource temporarily unavailable",
    "thread constructor failed",
    "pthread_create failed",
)


def _is_thread_exhaustion(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in THREAD_EXHAUSTION_MARKERS)


def _reward_record(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("reward")
    if isinstance(value, dict):
        return value
    return row


def _log_text(record: dict[str, Any]) -> str:
    logs = record.get("logs")
    if isinstance(logs, dict):
        return "\n".join(str(value) for value in logs.values())
    return ""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(_reward_record(value))
    if not rows:
        raise ValueError("reward JSONL is empty")
    return rows


def evaluate_gate(
    rows: list[dict[str, Any]], *, expected_groups: int = EXPECTED_GROUPS
) -> dict[str, Any]:
    bank_rows = [
        row
        for row in rows
        if str(row.get("task_id", "")).startswith("aider-shadow-cpp/bank-account-")
    ]
    if not bank_rows:
        raise ValueError("no bank-account curriculum reward records found")

    infrastructure = [row for row in bank_rows if bool(row.get("infrastructure_error"))]
    valid = [row for row in bank_rows if not bool(row.get("infrastructure_error"))]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[str(row.get("task_id"))].append(row)

    malformed_groups = {
        task_id: len(group) for task_id, group in groups.items() if len(group) != EXPECTED_SAMPLES
    }
    mixed_groups = 0
    all_pass_groups = 0
    all_fail_groups = 0
    for group in groups.values():
        outcomes = [bool(row.get("all_tests_pass")) for row in group]
        if all(outcomes):
            all_pass_groups += 1
        elif any(outcomes):
            mixed_groups += 1
        else:
            all_fail_groups += 1
    mixed_fraction = mixed_groups / len(groups) if groups else 0.0

    concurrency_rows: list[tuple[int, bool]] = []
    for row in valid:
        load = row.get("reward_worker_load")
        if not isinstance(load, int) or load < 1:
            raise ValueError("reward_worker_load is missing from a reward record")
        text = _log_text(row)
        # A concurrency failure is either the named assertion or a pthread
        # EAGAIN abort, which kills the grader before any FAILED line can
        # print (issue #110 r3: the assertion-only grep saw 0 of 121 deaths).
        concurrency_failure = (
            "FAILED: concurrent_transactions" in text or _is_thread_exhaustion(text)
        )
        concurrency_rows.append((load, concurrency_failure))

    loads = [load for load, _ in concurrency_rows]
    load_threshold = statistics.median(loads)
    # Bucket with >= / < so a modal maximum load cannot empty the high bucket
    # (r3: median 32 was also the max, leaving 0 high-load records and making
    # the correlation check vacuous). The check is computable only when both
    # buckets are populated; all-equal loads are surfaced, not silently passed.
    low = [failed for load, failed in concurrency_rows if load < load_threshold]
    high = [failed for load, failed in concurrency_rows if load >= load_threshold]
    load_check_computable = bool(low) and bool(high)
    low_rate = sum(low) / len(low) if low else 0.0
    high_rate = sum(high) / len(high) if high else 0.0
    load_correlated = (
        load_check_computable
        and sum(high) >= 2
        and high_rate >= low_rate + HIGH_LOAD_RATE_DELTA
    )

    reasons: list[str] = []
    if infrastructure:
        reasons.append("infrastructure-invalid-records-present")
    if len(groups) != expected_groups:
        reasons.append("prompt-group-count-mismatch")
    if malformed_groups:
        reasons.append("prompt-groups-do-not-have-eight-valid-samples")
    if groups and all_pass_groups == len(groups):
        reasons.append("all-prompt-groups-pass")
    elif groups and all_fail_groups == len(groups):
        reasons.append("all-prompt-groups-fail")
    elif mixed_fraction < MINIMUM_MIXED_FRACTION:
        reasons.append("mixed-prompt-group-fraction-below-0.30")
    if load_correlated:
        reasons.append("concurrency-failures-correlate-with-worker-load")

    if infrastructure:
        action = "repair-infrastructure-and-repeat"
    elif malformed_groups:
        action = "repeat-no-update-rollout"
    elif load_correlated:
        action = "cap-reward-workers-and-repeat"
    elif groups and all_pass_groups == len(groups):
        action = "stop-environment-too-easy"
    elif groups and all_fail_groups == len(groups):
        action = "use-atomic-repair-tier-only"
    elif mixed_fraction < MINIMUM_MIXED_FRACTION:
        action = "stop-insufficient-correctness-variance"
    else:
        action = "continue-to-bounded-training"

    passed = not reasons
    return {
        "gate": "issue-110-bank-account-no-update-rollout-v1",
        "passed": passed,
        "action": action,
        "reasons": reasons,
        "records": len(bank_rows),
        "valid_records": len(valid),
        "infrastructure_invalid_records": len(infrastructure),
        "prompt_groups": len(groups),
        "expected_prompt_groups": expected_groups,
        "malformed_groups": malformed_groups,
        "mixed_groups": mixed_groups,
        "mixed_prompt_group_fraction": mixed_fraction,
        "minimum_mixed_prompt_group_fraction": MINIMUM_MIXED_FRACTION,
        "all_pass_groups": all_pass_groups,
        "all_fail_groups": all_fail_groups,
        "concurrency_load_check": {
            "observed_failures": sum(failed for _, failed in concurrency_rows),
            "load_split_median": load_threshold,
            "low_load_records": len(low),
            "low_load_failure_rate": low_rate,
            "high_load_records": len(high),
            "high_load_failure_rate": high_rate,
            "computable": load_check_computable,
            "correlated": load_correlated,
            "correlation_rule": (
                "failure is the named concurrency assertion or a thread-exhaustion "
                "abort; buckets split at load >= median vs < median; requires at "
                "least two high-load failures and high-load rate exceeding "
                "low-load rate by at least 0.10"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rewards", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-groups", type=int, default=EXPECTED_GROUPS)
    args = parser.parse_args()
    if args.expected_groups < 1:
        raise SystemExit("--expected-groups must be positive")
    result = evaluate_gate(_load_rows(args.rewards), expected_groups=args.expected_groups)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
