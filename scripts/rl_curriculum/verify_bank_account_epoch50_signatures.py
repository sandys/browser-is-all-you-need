#!/usr/bin/env python3
"""Replay every accepted Synth-v1 epoch-50 bank-account signature on Linux/GCC.

The verifier reconstructs candidate files from the four tracked accepted trial
transcripts, adds the pinned healthcheck candidate, fetches the exact official
polyglot oracle at its pinned commit, and checks the complete pass/fail table.
It intentionally emits no repository artifacts; stdout is the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
RAW_ROOT = (
    "https://raw.githubusercontent.com/Aider-AI/polyglot-benchmark/"
    f"{POLYGLOT_COMMIT}/cpp/exercises/practice/bank-account"
)
OFFICIAL_TEST_SHA256 = "3696b9383f62ab639ad0a26610410fb662b6b927f2fe1dee956b849ff8dcf5c8"
CATCH_SHA256 = "681e7505a50887c9085539e5135794fc8f66d8e5de28eadf13a30978627b0f47"
TEST_MAIN_SHA256 = "5847fda35c1320d94f8d088aaf34229d689f66f1da235f885cbb28c8f17e4260"

CHECKPOINT_IDENTITY = "Synth v1 epoch 50"
CHECKPOINT_ITERATION = 649
ADAPTER_SHA256 = "4acb7f23c295f45380155c5d9ee6bc59422262f0cb51f0c02f7e550d405b575a"

FLAGS = [
    "-std=c++17",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-pthread",
    "-DEXERCISM_RUN_ALL_TESTS",
]

EXPECTED_TESTS = [
    "Newly opened account has zero balance",
    "Single deposit",
    "Multiple deposits",
    "Withdraw once",
    "Withdraw twice",
    "Can do multiple operations sequentially",
    "annot check balance of closed account",
    "Cannot deposit into closed account",
    "Cannot deposit into unopened account",
    "Cannot withdraw from closed account",
    "Cannot close an account that was not opened",
    "Cannot open an already opened account",
    "Reopened account does not retain balance",
    "Cannot withdraw more than deposited",
    "Cannot withdraw negative",
    "Cannot deposit negative",
    "Can handle concurrent transactions",
]

EVIDENCE = {
    "a1": {
        "chat_sha256": "118ee0c9ed5988200f5dcd236f4c2cc7ac87b97a0bc3a30f2f0f707411d5e1fd",
        "result_sha256": "dbdd0ea9860cc1d2aaf249b20de613a780fc4225bf05f17c8e320c1b789e38cf",
        "pairs": 3,
        "unique": [
            (
                "a98c5fe0e968b2e7f3e234617d6a686346a07662dd704a6aa985e4c73d15d5ed",
                "ffec413954787c9678f54d8e9426c08108ec85bbe365d7a1a19219da14e77631",
            )
        ],
        "outcomes": [True],
        "context_exhaustions": 0,
    },
    "a2": {
        "chat_sha256": "187cc497783e1359757c80542e4939efb6d53ae84268d4ab3506f0dcd424b3e8",
        "result_sha256": "39e3603432ef52e4ba5d5376532db1ad5a9929f5b66914768b963fd2b0918352",
        "pairs": 1,
        "unique": [
            (
                "c5e5fe1e814ad531dc3444de1e5abd9e7cc43460b9cf5d920c8fd8379f273c56",
                "edb48e70ae530ba347efad8b9eb93846534a33d86c1dc02caf4442441804b44f",
            )
        ],
        "outcomes": [True],
        "context_exhaustions": 1,
    },
    "a3": {
        "chat_sha256": "c2f7901905066f4592a5de73dfdf273b47aef5a09191c34b31a707b876042a75",
        "result_sha256": "0b3a93aa3e6f2e2ad9f54f618b0e3cf9d69a175c6c90502fa0d67d1c69ea5f6f",
        "pairs": 3,
        "unique": [
            (
                "c5e5fe1e814ad531dc3444de1e5abd9e7cc43460b9cf5d920c8fd8379f273c56",
                "2e6f37cee2a38fb2a21733aa6d8db0f29787698d0980f9827951235ff3009f6f",
            ),
            (
                "ee266b20f1782a8effd367dc7b7df5af4ca99be56cbb00a011cb3c129ae2f532",
                "0902f3ed47a1f140e1cffce15e502bc9cdf7a5122a6422c3a35d824ae7b63986",
            ),
        ],
        "outcomes": [False, False],
        "context_exhaustions": 1,
    },
    "a4": {
        "chat_sha256": "7d626d9d90c9a3fb0df455431cfd5b45fb43a08e961efc8af09e6b6957744ce2",
        "result_sha256": "717e278934fe3074cb1b17c52ffdd3c3ef69ae0242541ec1cf32b3ade6f2ae6f",
        "pairs": 1,
        "unique": [
            (
                "2bc6ef911132b8ae0c72a92f4110fcdca3b1122160e7122741af078839a44d9b",
                "5cf6e88aac572656e08cede2672d2fd257a4250ad5d00f34c94702010f791999",
            )
        ],
        "outcomes": [False, False],
        "context_exhaustions": 0,
        "applied_markers": 2,
    },
}

A1_HEADER = """#if !defined(BANK_ACCOUNT_H)
#define BANK_ACCOUNT_H

#include <mutex>

namespace Bankaccount {

class Bankaccount {
public:
    void open();
    void deposit(int amount);
    void withdraw(int amount);
    void close();
    int balance();

private:
    int balance_{0};
    bool open_{false};
    std::mutex mutex_{};
};

}  // namespace Bankaccount

#endif  // BANK_ACCOUNT_H"""

A1_SOURCE = """#include "bank_account.h"

#include <stdexcept>

namespace Bankaccount {

void Bankaccount::open() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ == true) {
        throw std::runtime_error("resource is already active");
    }
    open_ = true;
    balance_ = 0;
}

void Bankaccount::deposit(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    balance_ += amount;
}

void Bankaccount::withdraw(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("resource is not active");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > balance_) {
        throw std::runtime_error("amount exceeds available value");
    }
    balance_ -= amount;
}

void Bankaccount::close() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("resource is not active");
    }
    open_ = false;
}

int Bankaccount::balance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("resource is not active");
    }
    return balance_;
}

}  // namespace Bankaccount"""

A2_HEADER = """#pragma once

#include <mutex>

namespace Bankaccount {

class Bankaccount {
public:
    void open();
    void deposit(int amount);
    void withdraw(int amount);
    void close();
    int balance();

private:
    int balance_{0};
    bool is_open_{false};
    std::mutex mutex_{};
};

}  // namespace Bankaccount"""

A2_SOURCE = """#include "bank_account.h"

#include <stdexcept>

namespace Bankaccount {

void Bankaccount::open() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ == true) {
        throw std::runtime_error("account is already open");
    }
    balance_ = 0;
    is_open_ = true;
}

void Bankaccount::deposit(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ == false) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    balance_ += amount;
}

void Bankaccount::withdraw(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ == false) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > balance_) {
        throw std::runtime_error("amount exceeds available balance");
    }
    balance_ -= amount;
}

void Bankaccount::close() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ == false) {
        throw std::runtime_error("account is not open");
    }
    is_open_ = false;
}

int Bankaccount::balance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ == false) {
        throw std::runtime_error("account is not open");
    }
    return balance_;
}

}  // namespace Bankaccount"""

A3_TURN1_SOURCE = """#include "bank_account.h"

namespace Bankaccount {

void Bankaccount::open() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ == true) {
        throw std::runtime_error("account is already open");
    }
    is_open_ = true;
}

void Bankaccount::deposit(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ != true) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    balance_ += amount;
}

void Bankaccount::withdraw(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ != true) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > balance_) {
        throw std::runtime_error("amount exceeds available balance");
    }
    balance_ -= amount;
}

void Bankaccount::close() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ != true) {
        throw std::runtime_error("account is not open");
    }
    is_open_ = false;
}

int Bankaccount::balance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (is_open_ != true) {
        throw std::runtime_error("account is not open");
    }
    return balance_;
}

}  // namespace Bankaccount"""

A3_TURN2_HEADER = A2_HEADER.replace("#include <mutex>", "#include <mutex>\n#include <stdexcept>")
A3_TURN2_SOURCE = A3_TURN1_SOURCE.replace(
    '#include "bank_account.h"', '#include "bank_account.h"\n#include <stdexcept>', 1
)

A4_HEADER = """#if !defined(BANK_ACCOUNT_H)
#define BANK_ACCOUNT_H

#include <mutex>
#include <stdexcept>

namespace Bankaccount {

class Bankaccount {
public:
    void open();
    void deposit(int amount);
    void withdraw(int amount);
    void close();
    int balance();

private:
    int _balance{0};
    bool _open{false};
    std::mutex _mutex{};

    void _require_open() const {
        if (!_open) {
            throw std::runtime_error("account is not open");
        }
    }
};

}  // namespace Bankaccount

#endif  // BANK_ACCOUNT_H"""

A4_SOURCE = """#include "bank_account.h"

namespace Bankaccount {

void Bankaccount::open() {
    std::lock_guard<std::mutex> guard(_mutex);
    if (_open) {
        throw std::runtime_error("account is already open");
    }
    _open = true;
}

void Bankaccount::deposit(int amount) {
    std::lock_guard<std::mutex> guard(_mutex);
    _require_open();
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    _balance += amount;
}

void Bankaccount::withdraw(int amount) {
    std::lock_guard<std::mutex> guard(_mutex);
    _require_open();
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > _balance) {
        throw std::runtime_error("amount exceeds balance");
    }
    _balance -= amount;
}

void Bankaccount::close() {
    std::lock_guard<std::mutex> guard(_mutex);
    _require_open();
    _open = false;
}

int Bankaccount::balance() {
    std::lock_guard<std::mutex> guard(_mutex);
    _require_open();
    return _balance;
}

}  // namespace Bankaccount"""

HISTORICAL_STATES = {
    "a1": [{"bank_account.h": A1_HEADER, "bank_account.cpp": A1_SOURCE}],
    "a2": [{"bank_account.h": A2_HEADER, "bank_account.cpp": A2_SOURCE}],
    "a3": [
        {"bank_account.h": A2_HEADER, "bank_account.cpp": A3_TURN1_SOURCE},
        {"bank_account.h": A3_TURN2_HEADER, "bank_account.cpp": A3_TURN2_SOURCE},
    ],
    "a4": [{"bank_account.h": A4_HEADER, "bank_account.cpp": A4_SOURCE}],
}

HEALTH_FILES = {
    "bank_account.h": """#pragma once

#include <mutex>

namespace Bankaccount {

class Bankaccount {
public:
    void open();
    void deposit(int amount);
    void withdraw(int amount);
    void close();
    int balance();

private:
    int balance_{0};
    bool open_{false};
    std::mutex mutex_{};
};

}  // namespace Bankaccount""",
    "bank_account.cpp": """#include "bank_account.h"

namespace Bankaccount {

void Bankaccount::open() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ == true) {
        throw std::runtime_error("account is already open");
    }
    open_ = true;
}

void Bankaccount::deposit(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    balance_ += amount;
}

void Bankaccount::withdraw(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    if (amount > balance_) {
        throw std::runtime_error("amount exceeds available value");
    }
    balance_ -= amount;
}

void Bankaccount::close() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_ != true) {
        throw std::runtime_error("account is not open");
    }
    open_ = false;
}

int Bankaccount::balance() {
    std::lock_guard<std::mutex> guard(mutex_);
    return balance_;
}

}  // namespace Bankaccount""",
}

HEALTH_FILE_HASHES = {
    "bank_account.h": "4699cf791b5f13e7b5ddffc70e1d1fba3ec23fb2e82c17fe2c16180500a5c27a",
    "bank_account.cpp": "a024ffa863ef6b71063f1c784c40bf9551147a60d946662daed8ef7ccf512658",
}

EXPECTED_LEDGER = {
    "synth-v1-ep50-a1-bank-account": {
        "chat_sha256": EVIDENCE["a1"]["chat_sha256"],
        "result_sha256": EVIDENCE["a1"]["result_sha256"],
        "turns": [{"turn": 1, "outcome": "pass", "terminal_stage": "pass"}],
    },
    "synth-v1-ep50-a2-bank-account": {
        "chat_sha256": EVIDENCE["a2"]["chat_sha256"],
        "result_sha256": EVIDENCE["a2"]["result_sha256"],
        "turns": [{"turn": 1, "outcome": "pass", "terminal_stage": "pass"}],
    },
    "synth-v1-ep50-a3-bank-account": {
        "chat_sha256": EVIDENCE["a3"]["chat_sha256"],
        "result_sha256": EVIDENCE["a3"]["result_sha256"],
        "turns": [
            {"turn": 1, "outcome": "fail", "terminal_stage": "compile"},
            {"turn": 2, "outcome": "fail", "terminal_stage": "semantic-counterexample"},
        ],
    },
    "synth-v1-ep50-a4-bank-account": {
        "chat_sha256": EVIDENCE["a4"]["chat_sha256"],
        "result_sha256": EVIDENCE["a4"]["result_sha256"],
        "turns": [
            {"turn": 1, "outcome": "fail", "terminal_stage": "semantic-counterexample"},
            {"turn": 2, "outcome": "fail", "terminal_stage": "format-or-apply"},
        ],
    },
    "synth-v1-ep50-healthcheck-20260815-shard0-bank-account": {
        "chat_sha256": "27b464fe18a517e34c492fbba786ab4bc11314f4339bcb65b22e96d993857920",
        "result_sha256": "321836e5f6bae1cb0174a880e38740f82585aa13e0d400c9498fa0ccefea9fb9",
        "turns": [
            {"turn": 1, "outcome": "fail", "terminal_stage": "compile"},
            {"turn": 2, "outcome": "fail", "terminal_stage": "format-or-apply"},
        ],
    },
}

EXPECTED_LEDGER_CONTEXT = {
    "synth-v1-ep50-a1-bank-account": [False],
    "synth-v1-ep50-a2-bank-account": [True],
    "synth-v1-ep50-a3-bank-account": [True, False],
    "synth-v1-ep50-a4-bank-account": [False, False],
    "synth-v1-ep50-healthcheck-20260815-shard0-bank-account": [False, False],
}


@dataclass(frozen=True)
class HarnessFiles:
    test: str
    catch: str
    main: str


@dataclass(frozen=True)
class Evaluation:
    stage: str
    compile_diagnostic: str
    failed_tests: Tuple[str, ...]
    tests_passed: Optional[int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def fetch(url: str, attempts: int = 4) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "issue111-verifier/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - exercised only on network failure
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def load_harness() -> HarnessFiles:
    test = fetch(f"{RAW_ROOT}/bank_account_test.cpp")
    catch = fetch(f"{RAW_ROOT}/test/catch.hpp")
    main = fetch(f"{RAW_ROOT}/test/tests-main.cpp")
    require(sha256_bytes(test) == OFFICIAL_TEST_SHA256, "official test SHA drifted")
    require(sha256_bytes(catch) == CATCH_SHA256, "Catch2 header SHA drifted")
    require(sha256_bytes(main) == TEST_MAIN_SHA256, "test main SHA drifted")
    names = re.findall(r'TEST_CASE\("([^"]+)"', test.decode("utf-8"))
    require(names == EXPECTED_TESTS, f"official test inventory drifted: {names}")
    return HarnessFiles(test.decode("utf-8"), catch.decode("utf-8"), main.decode("utf-8"))


def pair_hash(files: Mapping[str, str]) -> Tuple[str, str]:
    return (sha256_text(files["bank_account.h"]), sha256_text(files["bank_account.cpp"]))


def add_stdexcept(files: Mapping[str, str]) -> Dict[str, str]:
    result = dict(files)
    combined = result["bank_account.h"] + result["bank_account.cpp"]
    require("<stdexcept>" not in combined, "include-only probe already has <stdexcept>")
    result["bank_account.cpp"] = result["bank_account.cpp"].replace(
        '#include "bank_account.h"', '#include "bank_account.h"\n#include <stdexcept>', 1
    )
    return result


def compile_identity(compiler: str, expected_major: int) -> str:
    require(sys.platform.startswith("linux"), f"verifier requires Linux, got {sys.platform}")
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True, timeout=20
    ).stdout.splitlines()[0]
    macros = subprocess.run(
        [compiler, "-dM", "-E", "-x", "c++", "-"],
        input="",
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    ).stdout
    require("#define __GNUC__ " in macros, f"not GNU C++: {version}")
    require("#define __clang__" not in macros, f"Clang is not accepted: {version}")
    major_match = re.search(r"#define __GNUC__ (\d+)", macros)
    require(bool(major_match), "cannot resolve GCC major")
    require(int(major_match.group(1)) == expected_major, f"expected GCC {expected_major}, got {version}")
    return version


def evaluate(
    compiler: str,
    harness: HarnessFiles,
    files: Mapping[str, str],
    root: Path,
    label: str,
) -> Evaluation:
    work = root / label
    (work / "test").mkdir(parents=True)
    for name, body in files.items():
        (work / name).write_text(body + "\n", encoding="utf-8")
    (work / "bank_account_test.cpp").write_text(harness.test, encoding="utf-8")
    (work / "test" / "catch.hpp").write_text(harness.catch, encoding="utf-8")
    (work / "test" / "tests-main.cpp").write_text(harness.main, encoding="utf-8")

    objects: List[str] = []
    compile_logs: List[str] = []
    for index, unit in enumerate(("bank_account_test.cpp", "bank_account.cpp", "test/tests-main.cpp")):
        obj = f"unit-{index}.o"
        command = [compiler, *FLAGS, "-I.", "-Itest", "-c", unit, "-o", obj]
        process = subprocess.run(
            command, cwd=work, check=False, capture_output=True, text=True, timeout=180
        )
        compile_logs.append(process.stdout + process.stderr)
        if process.returncode != 0:
            return Evaluation("compile", "".join(compile_logs), (), None)
        objects.append(obj)

    link = subprocess.run(
        [compiler, "-pthread", *objects, "-o", "candidate"],
        cwd=work,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if link.returncode != 0:
        return Evaluation("link", link.stdout + link.stderr, (), None)

    failed: List[str] = []
    for name in EXPECTED_TESTS:
        run = subprocess.run(
            [str(work / "candidate"), name],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if run.returncode != 0:
            diagnostic = run.stdout + run.stderr
            if "Resource temporarily unavailable" in diagnostic:
                raise RuntimeError(f"{label}: infrastructure-invalid thread failure: {diagnostic}")
            failed.append(name)
    return Evaluation(
        "pass" if not failed else "semantic",
        "",
        tuple(failed),
        len(EXPECTED_TESTS) - len(failed),
    )


def expect_evaluation(
    records: List[dict],
    compiler: str,
    harness: HarnessFiles,
    files: Mapping[str, str],
    root: Path,
    label: str,
    stage: str,
    failed_tests: Sequence[str] = (),
    compile_pattern: Optional[str] = None,
) -> None:
    result = evaluate(compiler, harness, files, root, label)
    require(result.stage == stage, f"{label}: expected {stage}, got {result.stage}")
    require(
        result.failed_tests == tuple(failed_tests),
        f"{label}: expected failed tests {failed_tests}, got {result.failed_tests}",
    )
    if compile_pattern:
        require(
            re.search(compile_pattern, result.compile_diagnostic, re.I | re.S) is not None,
            f"{label}: compiler signature missing: {result.compile_diagnostic}",
        )
    records.append(
        {
            "scenario": label,
            "stage": result.stage,
            "tests_passed": result.tests_passed,
            "failed_tests": list(result.failed_tests),
            "candidate_sha256": {
                name: sha256_text(body) for name, body in sorted(files.items())
            },
        }
    )


def verify_historical_fixtures() -> None:
    require(set(HISTORICAL_STATES) == set(EVIDENCE), "historical fixture trials drifted")
    for trial, states in HISTORICAL_STATES.items():
        expected = EVIDENCE[trial]
        require(
            [pair_hash(value) for value in states] == expected["unique"],
            f"{trial}: embedded candidate hashes drifted",
        )


def verify_ledger(repo: Path) -> None:
    path = repo / "docs" / "worklogs" / "synth-v1-failure-coverage" / "atomic-ledger.jsonl"
    records = {
        value["ledger_id"]: value
        for value in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
        )
        if value.get("ledger_id") in EXPECTED_LEDGER
    }
    require(set(records) == set(EXPECTED_LEDGER), "epoch-50 bank ledger rows are missing")
    for ledger_id, expected in EXPECTED_LEDGER.items():
        record = records[ledger_id]
        require(record["checkpoint_iteration"] == CHECKPOINT_ITERATION, f"{ledger_id}: iteration drifted")
        require(record["adapter_sha256"] == ADAPTER_SHA256, f"{ledger_id}: adapter drifted")
        require(record["chat_sha256"] == expected["chat_sha256"], f"{ledger_id}: chat SHA drifted")
        require(record["result_sha256"] == expected["result_sha256"], f"{ledger_id}: result SHA drifted")
        observed_turns = [
            {key: turn[key] for key in ("turn", "outcome", "terminal_stage")}
            for turn in record["turns"]
        ]
        require(observed_turns == expected["turns"], f"{ledger_id}: turn contract drifted")
        require(
            [turn["context_exhaustion"] for turn in record["turns"]]
            == EXPECTED_LEDGER_CONTEXT[ledger_id],
            f"{ledger_id}: context-exhaustion evidence drifted",
        )


def verify_health_fixture() -> None:
    require(
        {name: sha256_text(body) for name, body in HEALTH_FILES.items()} == HEALTH_FILE_HASHES,
        "healthcheck fixture SHA drifted",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--gcc-major", type=int, default=13)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    verify_ledger(repo)
    verify_historical_fixtures()
    verify_health_fixture()
    compiler = compile_identity(args.compiler, args.gcc_major)
    harness = load_harness()
    states = HISTORICAL_STATES

    require(
        pair_hash(states["a2"][0])[0] == pair_hash(states["a3"][0])[0],
        "a2-pass and a3-fail no longer share the exact header",
    )
    require("<stdexcept>" in states["a2"][0]["bank_account.cpp"], "a2 pass lost <stdexcept>")
    require("balance_ = 0" in states["a2"][0]["bank_account.cpp"], "a2 pass lost reopen reset")
    require("<stdexcept>" not in states["a3"][0]["bank_account.h"] + states["a3"][0]["bank_account.cpp"], "a3 first failure gained <stdexcept>")
    require("balance_ = 0" not in states["a3"][0]["bank_account.cpp"], "a3 first failure gained reset")
    require("<stdexcept>" in states["a3"][1]["bank_account.h"] + states["a3"][1]["bank_account.cpp"], "a3 repair did not add <stdexcept>")
    require("balance_ = 0" not in states["a3"][1]["bank_account.cpp"], "a3 repair unexpectedly fixed reset")
    require("_balance = 0" not in states["a4"][0]["bank_account.cpp"], "a4 unexpectedly fixed reset")

    records: List[dict] = []
    compile_pattern = r"runtime_error.*not.*member.*std|std.*has no member.*runtime_error"
    reopen = ["Reopened account does not retain balance"]
    health_semantic = [
        "annot check balance of closed account",
        "Reopened account does not retain balance",
    ]

    with tempfile.TemporaryDirectory(prefix="bank-account-epoch50-gcc-") as temporary:
        root = Path(temporary)
        expect_evaluation(records, args.compiler, harness, states["a1"][0], root, "a1-turn1-pass", "pass")
        expect_evaluation(records, args.compiler, harness, states["a2"][0], root, "a2-turn1-pass-context-exhausted", "pass")
        expect_evaluation(
            records,
            args.compiler,
            harness,
            states["a3"][0],
            root,
            "a3-turn1-missing-stdexcept",
            "compile",
            compile_pattern=compile_pattern,
        )
        expect_evaluation(
            records,
            args.compiler,
            harness,
            states["a3"][1],
            root,
            "a3-turn2-stale-reopen",
            "semantic",
            reopen,
        )
        expect_evaluation(
            records,
            args.compiler,
            harness,
            states["a4"][0],
            root,
            "a4-turn1-stale-reopen",
            "semantic",
            reopen,
        )
        expect_evaluation(
            records,
            args.compiler,
            harness,
            HEALTH_FILES,
            root,
            "health-turn1-missing-stdexcept",
            "compile",
            compile_pattern=compile_pattern,
        )
        expect_evaluation(
            records,
            args.compiler,
            harness,
            add_stdexcept(states["a3"][0]),
            root,
            "a3-include-only-probe",
            "semantic",
            reopen,
        )
        expect_evaluation(
            records,
            args.compiler,
            harness,
            add_stdexcept(HEALTH_FILES),
            root,
            "health-include-only-probe",
            "semantic",
            health_semantic,
        )

    no_edit_records = []
    for trial in ("a4",):
        expected = EVIDENCE[trial]
        require(expected["applied_markers"] == 2, f"{trial}: unexpected apply count")
        require(len(states[trial]) == 1, f"{trial}: feedback created a new candidate")
        require(expected["outcomes"] == [False, False], f"{trial}: expected two failures")
        no_edit_records.append(
            {
                "scenario": f"{trial}-turn2-no-edit",
                "new_candidate": False,
                "verification": "hash-bound atomic-ledger evidence plus embedded applied-state hash",
            }
        )
    require(len(HEALTH_FILES) == 2, "health fixture incomplete")
    no_edit_records.append(
        {
            "scenario": "health-turn2-no-edit",
            "new_candidate": False,
            "verification": "hash-bound atomic-ledger evidence; raw health transcript is not packaged",
        }
    )

    summary = {
        "status": "passed",
        "kind": "synth-v1-epoch50-bank-account-signature-verification",
        "checkpoint": {
            "identity": CHECKPOINT_IDENTITY,
            "iteration": CHECKPOINT_ITERATION,
            "adapter_sha256": ADAPTER_SHA256,
        },
        "compiler": compiler,
        "official_test_sha256": OFFICIAL_TEST_SHA256,
        "test_count": len(EXPECTED_TESTS),
        "scenario_count": len(records) + len(no_edit_records),
        "executed_scenarios": records,
        "no_edit_scenarios": no_edit_records,
        "differentiable_contract": {
            "strict_pass_reward": "1 for 17/17; 0 for every valid failure",
            "compile_blocked_tests": "unreachable, never counted as failed observations",
            "verified_pass_replay": ["a1-turn1-pass", "a2-turn1-pass-context-exhausted"],
            "context_exhaustion_is_not_terminal_failure": True,
        },
        "limitations": [
            "The a4 and healthcheck turn-2 no-edit labels are verified from the committed hash-bound ledger; raw transcripts are intentionally not copied into the SkyPilot workdir.",
            "Historical candidate fixtures are embedded and checked against hashes extracted from the accepted transcripts; the transcripts themselves remain the primary evidence.",
            "Compiler-blocked tests are unreachable; include-only probes are reported separately from historical observations.",
            "The official oracle does not test zero-valued deposit or withdrawal despite the prompt requiring non-positive amounts to fail.",
            "The concurrency case is executable but probabilistic; this verifier runs it once per compiled scenario.",
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("BANK_ACCOUNT_EPOCH50_SIGNATURE_VERIFIER_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
