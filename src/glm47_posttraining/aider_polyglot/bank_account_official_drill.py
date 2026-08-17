"""Build the exact official fixed26 bank-account self-imitation RL environment."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Mapping

from .bank_account_curriculum import compiler_identity, compiler_is_gcc
from .dataset import SOURCE_MANIFEST_KIND
from .schema import AiderShadowRubric


CURRICULUM_NAME = "bank-account-official-drill-v1"
CURRICULUM_ID = "fixed26-bank-account-self-imitation-rl-v1"
POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
OFFICIAL_TEST_SHA256 = "3696b9383f62ab639ad0a26610410fb662b6b927f2fe1dee956b849ff8dcf5c8"
VERIFICATION_GATE = "bank-account-official-drill-gcc13-v1"
FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-pthread", "-I."]
COMPILE_TIMEOUT_S = 120
TEST_TIMEOUT_S = 45

HEADER = "bank_account.h"
SOURCE = "bank_account.cpp"
TEST = "bank_account_test.cpp"

Stage = Literal["pass", "compile", "link-or-odr", "runtime", "semantic"]


@dataclass(frozen=True)
class Evaluation:
    stage: Stage
    diagnostic: str
    failed_ordinal: int | None = None


@dataclass(frozen=True)
class Episode:
    episode_kind: str
    failure_signature: str
    starter: Mapping[str, str]
    instructions: str
    expected_stage: Stage
    expected_ordinal: int | None = None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


OFFICIAL_PROMPT_SHA256 = "e72a3d80588754696bd1153d5b553fda47f26e465bc83b3a84ec512d69f76d8c"

OFFICIAL_INSTRUCTIONS = """# Introduction

After years of filling out forms and waiting, you've finally acquired your banking license.
This means you are now officially eligible to open your own bank, hurray!

Your first priority is to get the IT systems up and running.
After a day of hard work, you can already open and close accounts, as well as handle withdrawals and deposits.

Since you couldn't be bothered writing tests, you invite some friends to help test the system.
However, after just five minutes, one of your friends claims they've lost money!
While you're confident your code is bug-free, you start looking through the logs to investigate.

Ah yes, just as you suspected, your friend is at fault!
They shared their test credentials with another friend, and together they conspired to make deposits and withdrawals from the same account _in parallel_.
Who would do such a thing?

While you argue that it's physically _impossible_ for someone to access their account in parallel, your friend smugly notifies you that the banking rules _require_ you to support this.
Thus, no parallel banking support, no go-live signal.
Sighing, you create a mental note to work on this tomorrow.
This will set your launch date back at _least_ one more day, but well...
# Instructions

Your task is to implement bank accounts supporting opening/closing, withdrawals, and deposits of money.

As bank accounts can be accessed in many different ways (internet, mobile phones, automatic charges), your bank software must allow accounts to be safely accessed from multiple threads/processes (terminology depends on your programming language) in parallel.
For example, there may be many deposits and withdrawals occurring in parallel; you need to ensure there are no [race conditions][wikipedia] between when you read the account balance and set the new balance.

It should be possible to close an account; operations against a closed account must fail.

[wikipedia]: https://en.wikipedia.org/wiki/Race_condition#In_software


## C++ interface contract

The test file is not shown to you, so the interface it expects is stated here in
full. Implement exactly these names and signatures; the tests use nothing else.

```cpp
namespace Bankaccount {
class Bankaccount {
public:
    void open();
    void deposit(int amount);
    void withdraw(int amount);
    void close();
    int balance();
};
}
```

Note the namespace and class name are both `Bankaccount` (capital B, one word), not `bank_account`.

Every misuse throws `std::runtime_error`: opening an already-open account, closing or using an account that is not open, depositing or withdrawing a non-positive amount, and withdrawing more than the balance.

A newly opened account has a zero balance. After `close()`, a later `open()` starts a fresh account at zero; the previous balance is not retained.

The tests call `deposit` and `withdraw` concurrently from many `std::thread`s on one account, so the class must be internally thread-safe.

## Build environment

- The exercise is compiled as C++17 with `-Wall -Wextra -Wpedantic -Werror`, so
  any warning fails the build.
- Only `bank_account.h` and `bank_account.cpp` are editable. `CMakeLists.txt` and the test
  file are fixed and must not be modified.
- The test file includes only `bank_account.h`, so every name above must be visible
  from that header.
- You may either declare in `bank_account.h` and define in `bank_account.cpp`, or define
  everything `inline`/in-class in `bank_account.h` and leave `bank_account.cpp` unchanged.
  Both are accepted.

####

Use the above instructions to modify the supplied files: bank_account.cpp bank_account.h
Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc.
Only use standard libraries, don't suggest installing any packages.
"""

ORIGINAL_STARTER = {
    HEADER: """#if !defined(BANK_ACCOUNT_H)
#define BANK_ACCOUNT_H

namespace Bankaccount {
class Bankaccount {};  // class Bankaccount

}  // namespace Bankaccount

#endif  // BANK_ACCOUNT_H""",
    SOURCE: """#include "bank_account.h"

namespace Bankaccount {}""",
}

REFERENCE = {
    HEADER: """#pragma once

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
    SOURCE: """#include "bank_account.h"

#include <stdexcept>

namespace Bankaccount {

void Bankaccount::open() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (open_) {
        throw std::runtime_error("account is already open");
    }
    balance_ = 0;
    open_ = true;
}

void Bankaccount::deposit(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!open_) {
        throw std::runtime_error("account is not open");
    }
    if (amount <= 0) {
        throw std::runtime_error("amount must be positive");
    }
    balance_ += amount;
}

void Bankaccount::withdraw(int amount) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!open_) {
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
    if (!open_) {
        throw std::runtime_error("account is not open");
    }
    open_ = false;
}

int Bankaccount::balance() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!open_) {
        throw std::runtime_error("account is not open");
    }
    return balance_;
}

}  // namespace Bankaccount""",
}


def _replace_once(value: str, old: str, new: str, label: str) -> str:
    if value.count(old) != 1:
        raise ValueError(f"{label}: expected one mutation anchor, found {value.count(old)}")
    return value.replace(old, new, 1)


def missing_stdexcept() -> dict[str, str]:
    files = dict(REFERENCE)
    files[SOURCE] = _replace_once(files[SOURCE], "\n#include <stdexcept>\n", "\n", "stdexcept")
    return files


def stale_reopen() -> dict[str, str]:
    files = dict(REFERENCE)
    files[SOURCE] = _replace_once(files[SOURCE], "    balance_ = 0;\n", "", "reopen reset")
    return files


def closed_balance() -> dict[str, str]:
    files = dict(REFERENCE)
    old = """    if (!open_) {
        throw std::runtime_error("account is not open");
    }
    return balance_;"""
    files[SOURCE] = _replace_once(files[SOURCE], old, "    return balance_;", "closed balance")
    return files


def missing_definitions() -> dict[str, str]:
    return {
        HEADER: REFERENCE[HEADER],
        SOURCE: '#include "bank_account.h"\n\nnamespace Bankaccount {}',
    }


def balance_name_collision() -> dict[str, str]:
    files = dict(REFERENCE)
    files[HEADER] = _replace_once(files[HEADER], "int balance_{0};", "int balance{0};", "name collision")
    return files


def with_feedback(diagnostic: str) -> str:
    return (
        OFFICIAL_INSTRUCTIONS.rstrip()
        + "\n\n## Executed failure evidence\n\n"
        + "The supplied implementation produced this output under the declared GCC contract:\n\n"
        + "```text\n"
        + diagnostic.rstrip()
        + "\n```\n\nFix the implementation. The hidden tests and build contract are correct.\n"
    )


MISSING_HEADER_DIAGNOSTIC = """bank_account.cpp: error: ‘runtime_error’ is not a member of ‘std’
note: ‘std::runtime_error’ is defined in header ‘<stdexcept>’; did you forget to include it?"""

STALE_REOPEN_DIAGNOSTIC = """Reopened account does not retain balance
FAILED: REQUIRE(account.balance() == 0)
with expansion: 50 == 0
test cases: 17 | 16 passed | 1 failed"""


def episodes() -> list[Episode]:
    return [
        Episode("full-solve", "official-full-solve", ORIGINAL_STARTER, OFFICIAL_INSTRUCTIONS, "compile"),
        Episode("missing-stdexcept-repair", "missing-standard-header", missing_stdexcept(), OFFICIAL_INSTRUCTIONS, "compile"),
        Episode("stale-reopen-repair", "stale-state-on-reopen", stale_reopen(), OFFICIAL_INSTRUCTIONS, "semantic", 13),
        Episode("closed-balance-repair", "closed-account-balance-guard", closed_balance(), OFFICIAL_INSTRUCTIONS, "semantic", 7),
        Episode("missing-definitions-repair", "missing-method-definitions", missing_definitions(), OFFICIAL_INSTRUCTIONS, "link-or-odr"),
        Episode("name-collision-repair", "method-member-name-collision", balance_name_collision(), OFFICIAL_INSTRUCTIONS, "compile"),
        Episode(
            "compiler-feedback-repair",
            "feedback-missing-standard-header",
            missing_stdexcept(),
            with_feedback(MISSING_HEADER_DIAGNOSTIC),
            "compile",
        ),
        Episode(
            "test-feedback-repair",
            "feedback-stale-state-on-reopen",
            stale_reopen(),
            with_feedback(STALE_REOPEN_DIAGNOSTIC),
            "semantic",
            13,
        ),
    ]


def imitation_response(episode: Episode) -> str:
    """Return the smallest whole-file response that reaches the proved reference."""

    changed = [
        name
        for name in (HEADER, SOURCE)
        if episode.starter[name].rstrip() != REFERENCE[name].rstrip()
    ]
    if not changed:
        raise ValueError(f"{episode.episode_kind}: imitation target makes no change")
    explanation = (
        "I’ll replace the incomplete implementation with the thread-safe account contract."
        if len(changed) == 2
        else f"I’ll correct {changed[0]} so the implementation satisfies the account contract."
    )
    listings = "\n\n".join(
        f"{name}\n```cpp\n{REFERENCE[name]}\n```" for name in changed
    )
    return f"{explanation}\n\n{listings}"


def imitation_targets() -> dict[str, str]:
    """Map every authorized drill task to an executable passing SFT target."""

    return {
        f"bank-account-official--{episode.episode_kind}": imitation_response(episode)
        for episode in episodes()
    }


OFFICIAL_ORDINAL_ORACLE = r'''#include "bank_account.h"

#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

template <class Function>
bool throws_runtime(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int main() {
    {
        Bankaccount::Bankaccount account{};
        account.open();
        if (account.balance() != 0) return 1;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(100);
        if (account.balance() != 100) return 2;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(100);
        account.deposit(50);
        if (account.balance() != 150) return 3;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(100);
        account.withdraw(75);
        if (account.balance() != 25) return 4;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(100);
        account.withdraw(80);
        account.withdraw(20);
        if (account.balance() != 0) return 5;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(100);
        account.deposit(110);
        account.withdraw(200);
        account.deposit(60);
        account.withdraw(50);
        if (account.balance() != 20) return 6;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.close();
        if (!throws_runtime([&] { (void)account.balance(); })) return 7;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.close();
        if (!throws_runtime([&] { account.deposit(50); })) return 8;
    }
    {
        Bankaccount::Bankaccount account{};
        if (!throws_runtime([&] { account.deposit(50); })) return 9;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.close();
        if (!throws_runtime([&] { account.withdraw(50); })) return 10;
    }
    {
        Bankaccount::Bankaccount account{};
        if (!throws_runtime([&] { account.close(); })) return 11;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        if (!throws_runtime([&] { account.open(); })) return 12;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(50);
        account.close();
        account.open();
        if (account.balance() != 0) return 13;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(25);
        if (!throws_runtime([&] { account.withdraw(50); })) return 14;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        account.deposit(100);
        if (!throws_runtime([&] { account.withdraw(-50); })) return 15;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        if (!throws_runtime([&] { account.deposit(-50); })) return 16;
    }
    {
        Bankaccount::Bankaccount account{};
        account.open();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&] {
                using namespace std::chrono_literals;
                account.deposit(1);
                std::this_thread::sleep_for(5ms);
                account.withdraw(1);
            });
        }
        for (auto& thread : threads) thread.join();
        if (account.balance() != 0) return 17;
    }
    return 0;
}
'''


def _normalize(text: str, work: Path) -> str:
    replacement = "/aider/bank-account"
    for value in sorted({str(work), str(work.resolve())}, key=len, reverse=True):
        text = text.replace(value, replacement)
    return text.strip()


def evaluate_files(files: Mapping[str, str], *, compiler: str) -> Evaluation:
    with TemporaryDirectory(prefix="bank-account-official-drill-") as temporary:
        work = Path(temporary)
        for name, body in files.items():
            (work / name).write_text(body + "\n", encoding="utf-8")
        (work / TEST).write_text(OFFICIAL_ORDINAL_ORACLE, encoding="utf-8")

        objects: list[str] = []
        for index, unit in enumerate((TEST, SOURCE)):
            object_name = f"unit-{index}.o"
            result = subprocess.run(
                [compiler, *FLAGS, "-c", unit, "-o", object_name],
                cwd=work,
                check=False,
                capture_output=True,
                text=True,
                timeout=COMPILE_TIMEOUT_S,
            )
            diagnostic = _normalize(result.stdout + result.stderr, work)
            if result.returncode != 0:
                return Evaluation("compile", diagnostic)
            objects.append(object_name)

        link = subprocess.run(
            [compiler, *FLAGS, *objects, "-o", "candidate"],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_S,
        )
        diagnostic = _normalize(link.stdout + link.stderr, work)
        if link.returncode != 0:
            return Evaluation("link-or-odr", diagnostic)

        run = subprocess.run(
            [str(work / "candidate")],
            cwd=work,
            check=False,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT_S,
        )
        diagnostic = _normalize(run.stdout + run.stderr, work)
        if run.returncode == 0:
            return Evaluation("pass", diagnostic)
        if run.returncode < 0 or run.returncode > 128:
            return Evaluation("runtime", diagnostic)
        return Evaluation("semantic", diagnostic, run.returncode)


def _write_task(root: Path, episode: Episode) -> AiderShadowRubric:
    task_id = f"bank-account-official--{episode.episode_kind}"
    task_root = root / task_id
    docs = task_root / ".docs"
    docs.mkdir(parents=True)
    (docs / "instructions.md").write_text(episode.instructions.rstrip() + "\n", encoding="utf-8")
    for name, body in episode.starter.items():
        (task_root / name).write_text(body + "\n", encoding="utf-8")
    (task_root / TEST).write_text(OFFICIAL_ORDINAL_ORACLE, encoding="utf-8")
    (task_root / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(bank_account_official LANGUAGES CXX)\n",
        encoding="utf-8",
    )
    rubric = AiderShadowRubric(
        task_id=task_id,
        split="train",
        editable_files=[HEADER, SOURCE],
        hidden_test_file=TEST,
        hidden_test_sha256=sha256_text(OFFICIAL_ORDINAL_ORACLE),
        source_prompt_sha256=sha256_text(episode.instructions.rstrip() + "\n"),
        reference_answer_packaged=False,
        verification_stage="passed",
        verification_gate=VERIFICATION_GATE,
        family="bank-account",
        category="official-bank-account-drill",
        lineage_id="fixed26/bank-account",
        episode_kind=episode.episode_kind,
        objective_group="bank-account-drill",
        failure_signature=episode.failure_signature,
        tags=[
            "official-task-training-authorized",
            "strict-binary-reward",
            "whole-file-action",
            episode.episode_kind,
            episode.failure_signature,
        ],
    )
    (task_root / ".rubric.json").write_text(
        rubric.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return rubric


def build_bank_account_official_drill(
    output_root: str | Path,
    *,
    compiler: str = "g++",
    require_gcc: bool = True,
) -> dict[str, object]:
    output = Path(output_root).resolve()
    identity = compiler_identity(compiler)
    gcc = compiler_is_gcc(compiler)
    if sha256_text(OFFICIAL_INSTRUCTIONS) != OFFICIAL_PROMPT_SHA256:
        raise ValueError("official fixed26 prompt bytes drifted")
    if require_gcc and not gcc:
        raise ValueError(f"official drill requires GCC, got: {identity}")
    if output.exists():
        if output.is_symlink():
            raise ValueError(f"refusing to replace symlink: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    reference_result = evaluate_files(REFERENCE, compiler=compiler)
    if reference_result.stage != "pass":
        raise ValueError(f"reference failed: {reference_result}")

    receipts = []
    for episode in episodes():
        result = evaluate_files(episode.starter, compiler=compiler)
        if result.stage != episode.expected_stage:
            raise ValueError(
                f"{episode.episode_kind}: expected {episode.expected_stage}, got {result.stage}"
            )
        if episode.expected_ordinal is not None and result.failed_ordinal != episode.expected_ordinal:
            raise ValueError(
                f"{episode.episode_kind}: expected ordinal {episode.expected_ordinal}, "
                f"got {result.failed_ordinal}"
            )
        rubric = _write_task(output, episode)
        receipts.append(
            {
                "task_id": rubric.task_id,
                "episode_kind": episode.episode_kind,
                "failure_signature": episode.failure_signature,
                "starter_rejected_as": result.stage,
                "starter_failed_ordinal": result.failed_ordinal,
                "starter_sha256": {
                    name: sha256_text(body) for name, body in sorted(episode.starter.items())
                },
                "prompt_sha256": rubric.source_prompt_sha256,
                "reference_sha256": {
                    name: sha256_text(body) for name, body in sorted(REFERENCE.items())
                },
            }
        )

    verification = output / "verification.jsonl"
    verification.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in receipts),
        encoding="utf-8",
    )
    manifest = {
        "kind": SOURCE_MANIFEST_KIND,
        "schema_version": 2,
        "curriculum_id": CURRICULUM_ID,
        "source_locator": f"official:Aider-AI/polyglot-benchmark@{POLYGLOT_COMMIT}/bank-account",
        "counts": {"tasks": len(receipts), "train": len(receipts), "validation": 0},
        "compiler": identity,
        "compiler_is_gcc": gcc,
        "compile_flags": FLAGS,
        "contract": {
            "official_task_id_overlap": ["bank-account"],
            "official_training_authorized": True,
            "reference_answers_packaged": False,
            "shared_hidden_tests_within_lineage": True,
            "strict_binary_reward": True,
            "zero_held_out": True,
            "verification_gate": VERIFICATION_GATE,
            "official_test_sha256": OFFICIAL_TEST_SHA256,
            "official_prompt_sha256": OFFICIAL_PROMPT_SHA256,
            "ordinal_oracle_sha256": sha256_text(OFFICIAL_ORDINAL_ORACLE),
            "authorization_issue": "https://github.com/tokenbender/browser-is-all-you-need/issues/111",
        },
        "verification_receipts": "verification.jsonl",
        "verification_receipts_sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
