#!/usr/bin/env python3
"""Gate-2 checker tests (issue #110): EAGAIN-aware detector, robust load split.

Run: python3 tests/test_check_bank_account_rollout.py
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "check_bank_account_rollout", ROOT / "scripts/rl_curriculum/check_bank_account_rollout.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EAGAIN_LOG = (
    "terminate called after throwing an instance of 'std::system_error'\n"
    "  what():  Resource temporarily unavailable\n"
)
LEDGER = ROOT / "docs/worklogs/issue110-r3-rollout-forensics/gate_records.jsonl"


def _row(task, sample, *, passed=False, load=1, test_log="", infra=False):
    return {
        "task_id": f"aider-shadow-cpp/bank-account-{task}",
        "sample_index": sample,
        "all_tests_pass": passed,
        "infrastructure_error": infra,
        "reward_worker_load": load,
        "logs": {"compile": "", "test": test_log},
    }


def _groups(n_groups, per_group):
    rows = []
    for g in range(n_groups):
        for s in range(8):
            rows.append(per_group(g, s))
    return rows


def test_detector_counts_thread_exhaustion() -> None:
    rows = _groups(
        40,
        lambda g, s: _row(
            f"t{g}",
            s,
            passed=(s < 4),
            load=1 + (s % 2),
            test_log="" if s < 4 else EAGAIN_LOG,
        ),
    )
    result = MODULE.evaluate_gate(rows)
    assert result["concurrency_load_check"]["observed_failures"] == 40 * 4
    assert result["concurrency_load_check"]["computable"] is True


def test_named_assertion_still_detected() -> None:
    rows = _groups(
        40,
        lambda g, s: _row(
            f"t{g}",
            s,
            passed=(s != 0),
            load=1 + (s % 2),
            test_log="FAILED: concurrent_transactions\n" if s == 0 else "",
        ),
    )
    result = MODULE.evaluate_gate(rows)
    assert result["concurrency_load_check"]["observed_failures"] == 40


def test_modal_max_load_split_is_not_degenerate() -> None:
    # r3 shape: most records at the maximum load (32), the rest below. The old
    # (> median) split put zero records in the high bucket.
    rows = _groups(
        40,
        lambda g, s: _row(
            f"t{g}",
            s,
            passed=(s < 2),
            load=32 if s < 6 else 3,
            test_log="" if s < 2 else EAGAIN_LOG,
        ),
    )
    result = MODULE.evaluate_gate(rows)
    check = result["concurrency_load_check"]
    assert check["computable"] is True
    assert check["high_load_records"] > 0 and check["low_load_records"] > 0


def test_all_equal_loads_marked_not_computable() -> None:
    rows = _groups(40, lambda g, s: _row(f"t{g}", s, passed=(s < 4), load=7))
    result = MODULE.evaluate_gate(rows)
    check = result["concurrency_load_check"]
    assert check["computable"] is False
    assert check["correlated"] is False


def test_load_correlation_still_fires() -> None:
    # Failures concentrated at high load must trip the gate.
    rows = _groups(
        40,
        lambda g, s: _row(
            f"t{g}",
            s,
            passed=(s < 4) or (s % 2 == 0),
            load=30 if s >= 4 else 2,
            test_log=EAGAIN_LOG if (s >= 4 and s % 2 == 1) else "",
        ),
    )
    result = MODULE.evaluate_gate(rows)
    assert result["concurrency_load_check"]["correlated"] is True
    assert "concurrency-failures-correlate-with-worker-load" in result["reasons"]
    assert result["action"] == "cap-reward-workers-and-repeat"


def test_r3_ledger_now_detects_what_the_run_missed() -> None:
    rows = [json.loads(line) for line in LEDGER.open()]
    result = MODULE.evaluate_gate(rows)
    check = result["concurrency_load_check"]
    # The shipped r3 gate reported observed_failures=0 and high_load_records=0
    # on these exact records; both blind spots are gone. 121 EAGAIN deaths in
    # clean tests_failed records plus 14 in recoverable-format records whose
    # recovered parse still reached the test stage.
    assert check["observed_failures"] == 135, check["observed_failures"]
    assert check["computable"] is True
    assert check["high_load_records"] > 0 and check["low_load_records"] > 0


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
