#!/usr/bin/env python3
"""Verify the bank-account-equivalent benchmark suite and write its receipt."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUITE_ROOT.parents[2]
MANIFEST_PATH = SUITE_ROOT / "manifest.json"
RECEIPT_PATH = SUITE_ROOT / "verification_receipt.json"
EXPECTED_PARENT_TEST_SHA256 = (
    "3696b9383f62ab639ad0a26610410fb662b6b927f2fe1dee956b849ff8dcf5c8"
)
EXPECTED_HISTORICAL_PROMPT_SHA256 = (
    "c1faf70cf9fdbd2e7e4493850787168e2d940b6291ee0d70c9b94e267d2d7e81"
)
EXPECTED_CORRECTED_CONTRACT_SHA256 = (
    "c6aa125fa8dff54144e72d55165f75689bc3919f3a4947fc98b97c2861471d95"
)
EXPECTED_PARENT_API = {
    "namespace": "Bankaccount",
    "class": "Bankaccount",
    "constructor": "Bankaccount",
    "start": "open",
    "credit": "deposit",
    "debit": "withdraw",
    "stop": "close",
    "value": "balance",
}
EXPECTED_PARENT_TESTS = (
    "newly_started_zero",
    "single_credit",
    "multiple_credits",
    "single_debit",
    "multiple_debits",
    "sequential_operations",
    "value_after_stop_throws",
    "credit_after_stop_throws",
    "credit_before_start_throws",
    "debit_after_stop_throws",
    "stop_before_start_throws",
    "start_twice_throws",
    "restart_resets_zero",
    "overdraft_throws",
    "negative_debit_throws",
    "negative_credit_throws",
    "concurrent_transactions",
)
EXPECTED_EXTRA_TESTS = (
    "zero_debit_throws",
    "zero_credit_throws",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command timed out after {timeout} seconds: {rendered}"
        ) from error
    if result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"command failed ({result.returncode}): {rendered}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def verify_hashes(manifest: dict[str, object]) -> None:
    generator = manifest["generator"]
    generator_path = REPO_ROOT / generator["path"]
    require(generator_path.is_file(), f"missing generator: {generator_path}")
    require(
        sha256(generator_path) == generator["sha256"],
        f"generator hash mismatch: {generator_path}",
    )

    for relative, record in manifest["root_files"].items():
        path = SUITE_ROOT / relative
        require(path.is_file(), f"missing root file: {path}")
        require(sha256(path) == record["sha256"], f"hash mismatch: {path}")
        require(path.stat().st_size == record["bytes"], f"size mismatch: {path}")

    for variant in manifest["variants"]:
        variant_root = SUITE_ROOT / variant["directory"]
        for relative, record in variant["files"].items():
            path = variant_root / relative
            require(path.is_file(), f"missing variant file: {path}")
            require(sha256(path) == record["sha256"], f"hash mismatch: {path}")
            require(path.stat().st_size == record["bytes"], f"size mismatch: {path}")


def verify_mapping(variant: dict[str, object]) -> None:
    api = variant["api"]
    expected = [
        {
            "role": role,
            "parent": parent_name,
            "variant": api[role],
        }
        for role, parent_name in EXPECTED_PARENT_API.items()
    ]
    require(
        variant["api_mapping"] == expected,
        f'{variant["id"]}: parent-to-variant API mapping drifted',
    )


def verify_prompt(variant: dict[str, object], prompt: str) -> None:
    api = variant["api"]
    domain = variant["domain"]
    normalized = " ".join(prompt.split())
    required_fragments = [
        f'default-constructed {domain["resource"]} is inactive',
        f'`{api["start"]}()` activates it and starts its value at zero',
        f'Calling `{api["start"]}()` while it is already active throws',
        "std::runtime_error",
        f'`{api["credit"]}(amount)` adds a positive amount',
        f'`{api["debit"]}(amount)` removes a positive amount',
        f'`{api["value"]}()` returns the current amount',
        "amount exceeds the current amount",
        "`amount` is zero or negative",
        f'Calling `{api["stop"]}()` deactivates',
        "Calling it before activation or after deactivation",
        f'`{api["value"]}`, `{api["credit"]}`, or `{api["debit"]}` operation '
        "on an inactive",
        "Reactivating after deactivation starts a fresh value of zero",
        "No amount from the previous active lifecycle is retained",
        "Many threads call",
        "validate and update shared state atomically under synchronization",
        f'namespace {api["namespace"]}',
        f'class {api["class"]}',
        f'{api["constructor"]}();',
        f'void {api["start"]}();',
        f'void {api["credit"]}(int amount);',
        f'void {api["debit"]}(int amount);',
        f'void {api["stop"]}();',
        f'int {api["value"]}();',
        "must remain default-constructible",
        "header must be self-contained",
        "correct namespace and class qualification",
        "must not declare a second replacement class",
        "-Wall -Wextra -Wpedantic -Werror -pthread",
    ]
    for fragment in required_fragments:
        require(
            " ".join(fragment.split()) in normalized,
            f'{variant["id"]}: prompt missing {fragment!r}',
        )


def verify_static_tests(variant: dict[str, object], tests: str) -> None:
    api = variant["api"]
    inventory = variant["tests"]
    require(
        tuple(inventory["parent_mapped"]) == EXPECTED_PARENT_TESTS,
        f'{variant["id"]}: parent test inventory drifted',
    )
    require(
        tuple(inventory["contract_completion"]) == EXPECTED_EXTRA_TESTS,
        f'{variant["id"]}: extra test inventory drifted',
    )
    require(
        inventory["parent_count"] == 17
        and inventory["additional_count"] == 2
        and inventory["total"] == 19
        and inventory["concurrent_threads"] == 1000,
        f'{variant["id"]}: declared test counts drifted',
    )
    assertion_calls = re.findall(r"checks\.expect(?:_runtime_error)?\(", tests)
    require(
        len(assertion_calls) == 19,
        f'{variant["id"]}: expected 19 checks, found {len(assertion_calls)}',
    )
    for test_id in EXPECTED_PARENT_TESTS + EXPECTED_EXTRA_TESTS:
        require(
            tests.count(f'"{test_id}"') == 1,
            f'{variant["id"]}: test ID {test_id!r} is missing or duplicated',
        )

    normalized = " ".join(tests.split())
    semantic_fragments = [
        f'subject.{api["start"]}(); checks.expect(subject.{api["value"]}() == 0, '
        '"newly_started_zero"',
        f'subject.{api["credit"]}(100); checks.expect(subject.{api["value"]}() == 100, '
        '"single_credit"',
        f'subject.{api["credit"]}(100); subject.{api["credit"]}(50); '
        f'checks.expect(subject.{api["value"]}() == 150, "multiple_credits"',
        f'subject.{api["debit"]}(75); checks.expect(subject.{api["value"]}() == 25, '
        '"single_debit"',
        f'subject.{api["debit"]}(80); subject.{api["debit"]}(20); '
        f'checks.expect(subject.{api["value"]}() == 0, "multiple_debits"',
        f'subject.{api["credit"]}(100); subject.{api["credit"]}(110); '
        f'subject.{api["debit"]}(200); subject.{api["credit"]}(60); '
        f'subject.{api["debit"]}(50); '
        f'checks.expect(subject.{api["value"]}() == 20, "sequential_operations"',
        f'[&]() {{ (void)subject.{api["value"]}(); }}, "value_after_stop_throws"',
        f'[&]() {{ subject.{api["credit"]}(50); }}, "credit_after_stop_throws"',
        f'[&]() {{ subject.{api["credit"]}(50); }}, "credit_before_start_throws"',
        f'[&]() {{ subject.{api["debit"]}(50); }}, "debit_after_stop_throws"',
        f'[&]() {{ subject.{api["stop"]}(); }}, "stop_before_start_throws"',
        f'[&]() {{ subject.{api["start"]}(); }}, "start_twice_throws"',
        f'subject.{api["stop"]}(); subject.{api["start"]}(); '
        f'checks.expect(subject.{api["value"]}() == 0, "restart_resets_zero"',
        f'subject.{api["credit"]}(25); checks.expect_runtime_error( '
        f'[&]() {{ subject.{api["debit"]}(50); }}, "overdraft_throws"',
        f'[&]() {{ subject.{api["debit"]}(-50); }}, "negative_debit_throws"',
        f'[&]() {{ subject.{api["credit"]}(-50); }}, "negative_credit_throws"',
        f'[&]() {{ subject.{api["debit"]}(0); }}, "zero_debit_throws"',
        f'[&]() {{ subject.{api["credit"]}(0); }}, "zero_credit_throws"',
        "threads.reserve(1000)",
        "index < 1000",
        f'subject.{api["credit"]}(1)',
        "std::this_thread::sleep_for(5ms)",
        f'subject.{api["debit"]}(1)',
        f'checks.expect(subject.{api["value"]}() == 0, "concurrent_transactions"',
    ]
    for fragment in semantic_fragments:
        require(
            " ".join(fragment.split()) in normalized,
            f'{variant["id"]}: static test matrix missing {fragment!r}',
        )


def verify_variant(
    variant: dict[str, object],
    compiler: str,
    temp_root: Path,
    timeout: int,
) -> dict[str, object]:
    variant_root = SUITE_ROOT / variant["directory"]
    api = variant["api"]
    verify_mapping(variant)
    verify_prompt(
        variant,
        (variant_root / "PROMPT.md").read_text(encoding="utf-8"),
    )

    source_name = next(
        name
        for name in variant["files"]
        if name.endswith(".cpp") and not name.endswith("_test.cpp")
    )
    test_name = next(name for name in variant["files"] if name.endswith("_test.cpp"))
    header_name = next(name for name in variant["files"] if name.endswith(".h"))
    tests = (variant_root / test_name).read_text(encoding="utf-8")
    verify_static_tests(variant, tests)

    consumer = temp_root / f'{variant["id"]}_header_consumer.cpp'
    consumer.write_text(
        "#include <type_traits>\n"
        f'#include "{header_name}"\n'
        f"static_assert(std::is_default_constructible_v<"
        f"{api['namespace']}::{api['class']}>);\n"
        f"int main() {{ {api['namespace']}::{api['class']} value{{}}; "
        f"(void)value; return 0; }}\n",
        encoding="utf-8",
    )
    run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-pthread",
            "-fsyntax-only",
            "-I",
            str(variant_root),
            str(consumer),
        ]
    )

    executable = temp_root / f'{variant["id"]}_tests'
    compile_result = run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-pthread",
            "-I",
            str(variant_root),
            str(variant_root / source_name),
            str(variant_root / test_name),
            "-o",
            str(executable),
        ]
    )
    test_result = run([str(executable)], timeout=timeout)
    expected_output = "All tests passed (19 assertions in 19 test cases)"
    require(
        test_result.stdout.strip() == expected_output,
        f'{variant["id"]}: unexpected test output: {test_result.stdout!r}',
    )
    return {
        "id": variant["id"],
        "status": "passed",
        "header_self_contained": True,
        "default_constructible": True,
        "strict_compile": True,
        "runtime_within_timeout": True,
        "parent_mapped_assertions": 17,
        "contract_completion_assertions": 2,
        "total_assertions": 19,
        "concurrent_threads": 1000,
        "test_output": test_result.stdout.strip(),
        "compiler_stderr": compile_result.stderr.strip(),
    }


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    lineage = manifest["lineage"]
    contract = manifest["contract"]
    require(manifest["schema_version"] == 1, "unsupported manifest schema")
    require(manifest["role"] == "benchmark_and_evaluation_only", "incorrect role")
    require(manifest["training_exclusion"]["trainable"] is False, "suite is trainable")
    require(
        lineage["parent_test_sha256"] == EXPECTED_PARENT_TEST_SHA256,
        "parent test hash mismatch",
    )
    require(
        lineage["historical_prompt_sha256"] == EXPECTED_HISTORICAL_PROMPT_SHA256,
        "historical prompt hash mismatch",
    )
    require(
        lineage["corrected_contract_sha256"] == EXPECTED_CORRECTED_CONTRACT_SHA256,
        "corrected contract hash mismatch",
    )
    require(lineage["parent_api"] == EXPECTED_PARENT_API, "parent API drifted")
    require(len(lineage["failure_taxonomy"]) == 8, "failure taxonomy is incomplete")
    require(len(manifest["variants"]) == 10, "expected exactly ten variants")
    require(contract["total_assertions"] == 190, "expected 190 assertions")
    require(contract["total_concurrent_threads"] == 10000, "thread coverage drifted")
    verify_hashes(manifest)

    compiler = os.environ.get("CXX", "c++")
    compiler_version = run([compiler, "--version"]).stdout.splitlines()[0]
    timeout = int(contract["runtime_timeout_seconds_per_variant"])
    with tempfile.TemporaryDirectory(prefix="bank-account-equivalent-verify-") as temp:
        temp_root = Path(temp)
        results = [
            verify_variant(variant, compiler, temp_root, timeout)
            for variant in manifest["variants"]
        ]

    receipt = {
        "schema_version": 1,
        "suite_id": manifest["id"],
        "status": "passed",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "parent_test_sha256": EXPECTED_PARENT_TEST_SHA256,
        "historical_prompt_sha256": EXPECTED_HISTORICAL_PROMPT_SHA256,
        "corrected_contract_sha256": EXPECTED_CORRECTED_CONTRACT_SHA256,
        "role": manifest["role"],
        "trainable": False,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "compiler_command": compiler,
            "compiler_version": compiler_version,
        },
        "summary": {
            "variants": 10,
            "headers_self_contained": 10,
            "default_constructible_types": 10,
            "strict_compiles": 10,
            "runtime_timeout_seconds_per_variant": timeout,
            "runtime_timeouts": 0,
            "parent_mapped_assertions": 170,
            "contract_completion_assertions": 20,
            "total_assertions": 190,
            "failed_assertions": 0,
            "concurrent_threads_exercised": 10000,
            "prompt_test_mismatches": 0,
        },
        "variants": results,
    }
    RECEIPT_PATH.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "Verified 10 benchmark variants: 10 strict compiles, "
        "10 self-contained/default-constructible headers, "
        "190/190 assertions passed, 10,000 concurrent threads exercised, "
        "0 timeouts, 0 prompt/test mismatches"
    )
    print(f"Wrote {RECEIPT_PATH}")


if __name__ == "__main__":
    main()
