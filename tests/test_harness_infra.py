#!/usr/bin/env python3
"""Infra classification tests for the aider sandbox harness (issue #110).

Run: PYTHONPATH=src python3 tests/test_harness_infra.py

Covers the EAGAIN thread-exhaustion classification and the parameterized
docker pids budget, and replays both against the committed r3 forensics
ledger so the classifier is proven on the exact records that motivated it.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from glm47_posttraining.aider_polyglot.harness import (  # noqa: E402
    SANDBOX_PIDS_LIMIT,
    _is_thread_exhaustion,
    is_test_stage_infrastructure_failure,
)
from glm47_posttraining.cpp_perf.sandbox import docker_base_args  # noqa: E402

EAGAIN_LOG = (
    "terminate called after throwing an instance of 'std::system_error'\n"
    "  what():  Resource temporarily unavailable\n"
    "timeout: the monitored command dumped core\n"
)
EAGAIN_AFTER_ASSERT_LOG = "FAILED: restart_resets_zero\n" + EAGAIN_LOG
SEMANTIC_LOG = "FAILED: restart_resets_zero\nVerification failed after 19 assertions\n"
PASS_LOG = "All tests passed (19 assertions in 19 test cases)\n"

LEDGER = (
    pathlib.Path(__file__).resolve().parent.parent
    / "docs/worklogs/issue110-r3-rollout-forensics/gate_records.jsonl"
)


def test_thread_exhaustion_detection() -> None:
    assert _is_thread_exhaustion(EAGAIN_LOG)
    assert _is_thread_exhaustion(EAGAIN_AFTER_ASSERT_LOG)
    assert not _is_thread_exhaustion(SEMANTIC_LOG)
    assert not _is_thread_exhaustion(PASS_LOG)
    assert not _is_thread_exhaustion("")


def test_test_stage_infrastructure_rule() -> None:
    # No failed assertion before the abort: the sandbox robbed a probable pass.
    assert is_test_stage_infrastructure_failure(EAGAIN_LOG)
    # An assertion already failed: the semantic verdict stands, record is valid.
    assert not is_test_stage_infrastructure_failure(EAGAIN_AFTER_ASSERT_LOG)
    assert not is_test_stage_infrastructure_failure(SEMANTIC_LOG)
    assert not is_test_stage_infrastructure_failure(PASS_LOG)


def test_docker_pids_limit_parameter() -> None:
    default_args = docker_base_args("/tmp/x")
    assert "--pids-limit" in default_args
    assert default_args[default_args.index("--pids-limit") + 1] == "128"
    oracle_args = docker_base_args("/tmp/x", pids_limit=SANDBOX_PIDS_LIMIT)
    assert oracle_args[oracle_args.index("--pids-limit") + 1] == str(SANDBOX_PIDS_LIMIT)
    # The oracle budget must clear the 1,000-thread suite with process headroom.
    assert SANDBOX_PIDS_LIMIT >= 1100


def test_classifier_against_r3_ledger() -> None:
    records = [json.loads(line) for line in LEDGER.open()]
    assert len(records) == 320
    robbed = semantic_after_eagain = 0
    for record in records:
        if record["format_valid"] and not record["compile_error"] and not record["all_tests_pass"]:
            test_log = record.get("logs", {}).get("test", "") or ""
            if is_test_stage_infrastructure_failure(test_log):
                robbed += 1
            elif _is_thread_exhaustion(test_log):
                semantic_after_eagain += 1
    # The r3 forensics ground truth: 80 rollouts died with zero failed
    # assertions (now infra-invalid) and 41 failed an assertion first (still
    # valid semantic failures).
    assert robbed == 80, robbed
    assert semantic_after_eagain == 41, semantic_after_eagain


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
