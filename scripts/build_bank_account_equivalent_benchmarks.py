#!/usr/bin/env python3
"""Build ten lineage-declared benchmark variants of the C++ bank-account task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = REPO_ROOT / "benchmarks" / "cpp" / "bank-account-equivalent-v1"
PARENT_TEST_SHA256 = (
    "3696b9383f62ab639ad0a26610410fb662b6b927f2fe1dee956b849ff8dcf5c8"
)
HISTORICAL_PROMPT_SHA256 = (
    "c1faf70cf9fdbd2e7e4493850787168e2d940b6291ee0d70c9b94e267d2d7e81"
)
CORRECTED_CONTRACT_SHA256 = (
    "c6aa125fa8dff54144e72d55165f75689bc3919f3a4947fc98b97c2861471d95"
)

PARENT_API = {
    "namespace": "Bankaccount",
    "class": "Bankaccount",
    "constructor": "Bankaccount",
    "start": "open",
    "credit": "deposit",
    "debit": "withdraw",
    "stop": "close",
    "value": "balance",
}

PARENT_TESTS = (
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

EXTRA_TESTS = (
    "zero_debit_throws",
    "zero_credit_throws",
)

FAILURE_TAXONOMY = (
    {
        "attempt": 1,
        "category": "execution_deadlock",
        "cause": "pending counter decremented without increment or notification",
    },
    {
        "attempt": 2,
        "category": "historical_prompt_omission",
        "cause": "restart did not reset value because the July 27 prompt omitted that rule",
    },
    {
        "attempt": 3,
        "category": "missing_definitions",
        "cause": "header declared the API while the source file was empty",
    },
    {
        "attempt": 4,
        "category": "undefined_constructor",
        "cause": "default constructor was declared but never defined",
    },
    {
        "attempt": 5,
        "category": "class_redefinition",
        "cause": "source file defined a second class instead of member definitions",
    },
    {
        "attempt": 6,
        "category": "qualification_error",
        "cause": "namespace was used where namespace plus class qualification was required",
    },
    {
        "attempt": 7,
        "category": "default_constructor_suppressed",
        "cause": "a deleted copy constructor removed implicit default construction",
    },
    {
        "attempt": 8,
        "category": "class_redefinition",
        "cause": "source file defined a second class instead of member definitions",
    },
)

VARIANTS = (
    {
        "id": "secure-wallet",
        "title": "Secure wallet",
        "stem": "secure_wallet",
        "namespace": "secure_wallet",
        "class": "wallet_account",
        "start": "activate",
        "credit": "deposit",
        "debit": "withdraw",
        "stop": "deactivate",
        "value": "balance",
        "resource": "wallet",
        "unit": "credits",
        "style": "defaulted_out_of_line",
    },
    {
        "id": "energy-reserve",
        "title": "Energy reserve",
        "stem": "energy_reserve",
        "namespace": "energy_reserve",
        "class": "reserve_meter",
        "start": "enable",
        "credit": "add_units",
        "debit": "consume_units",
        "stop": "disable",
        "value": "remaining_units",
        "resource": "reserve meter",
        "unit": "energy units",
        "style": "explicit_constructor",
    },
    {
        "id": "arcade-card",
        "title": "Arcade card",
        "stem": "arcade_card",
        "namespace": "arcade_card",
        "class": "player_card",
        "start": "issue",
        "credit": "load",
        "debit": "spend",
        "stop": "revoke",
        "value": "credits",
        "resource": "player card",
        "unit": "arcade credits",
        "style": "validation_helpers",
    },
    {
        "id": "inventory-ledger",
        "title": "Inventory ledger",
        "stem": "inventory_ledger",
        "namespace": "inventory_ledger",
        "class": "stock_ledger",
        "start": "begin",
        "credit": "receive",
        "debit": "dispatch",
        "stop": "end",
        "value": "quantity",
        "resource": "stock ledger",
        "unit": "items",
        "style": "inline_header",
    },
    {
        "id": "transit-pass",
        "title": "Transit pass",
        "stem": "transit_pass",
        "namespace": "transit_pass",
        "class": "fare_pass",
        "start": "activate",
        "credit": "top_up",
        "debit": "charge",
        "stop": "suspend",
        "value": "funds",
        "resource": "fare pass",
        "unit": "fare credits",
        "style": "enum_state",
    },
    {
        "id": "cloud-quota",
        "title": "Cloud quota",
        "stem": "cloud_quota",
        "namespace": "cloud_quota",
        "class": "quota_bucket",
        "start": "provision",
        "credit": "grant",
        "debit": "consume",
        "stop": "retire",
        "value": "available",
        "resource": "quota bucket",
        "unit": "quota units",
        "style": "unique_lock",
    },
    {
        "id": "library-credit",
        "title": "Library credit",
        "stem": "library_credit",
        "namespace": "library_credit",
        "class": "patron_account",
        "start": "enroll",
        "credit": "add_credit",
        "debit": "use_credit",
        "stop": "close",
        "value": "credit",
        "resource": "patron account",
        "unit": "borrowing credits",
        "style": "scoped_lock",
    },
    {
        "id": "reward-points",
        "title": "Reward points",
        "stem": "reward_points",
        "namespace": "reward_points",
        "class": "reward_account",
        "start": "open",
        "credit": "earn",
        "debit": "redeem",
        "stop": "close",
        "value": "points",
        "resource": "reward account",
        "unit": "points",
        "style": "nested_state",
    },
    {
        "id": "prepaid-data",
        "title": "Prepaid data",
        "stem": "prepaid_data",
        "namespace": "prepaid_data",
        "class": "data_wallet",
        "start": "connect",
        "credit": "add_megabytes",
        "debit": "use_megabytes",
        "stop": "disconnect",
        "value": "remaining_megabytes",
        "resource": "data wallet",
        "unit": "megabytes",
        "style": "defaulted_in_header",
    },
    {
        "id": "workshop-tokens",
        "title": "Workshop tokens",
        "stem": "workshop_tokens",
        "namespace": "workshop_tokens",
        "class": "token_box",
        "start": "unlock",
        "credit": "add_tokens",
        "debit": "take_tokens",
        "stop": "lock",
        "value": "token_count",
        "resource": "token box",
        "unit": "tokens",
        "style": "plain_out_of_line",
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def public_api(variant: dict[str, str]) -> str:
    return dedent(
        f"""\
        namespace {variant["namespace"]} {{
        class {variant["class"]} {{
        public:
            {variant["class"]}();
            void {variant["start"]}();
            void {variant["credit"]}(int amount);
            void {variant["debit"]}(int amount);
            void {variant["stop"]}();
            int {variant["value"]}();
        }};
        }}  // namespace {variant["namespace"]}
        """
    ).rstrip()


def prompt_for(variant: dict[str, str]) -> str:
    api_block = public_api(variant).replace("\n", "\n        ")
    return dedent(
        f"""\
        # {variant["title"]}

        Implement a thread-safe {variant["resource"]} that manages an integer
        quantity of {variant["unit"]}.

        ## Lifecycle and operations

        A default-constructed {variant["resource"]} is inactive. Calling
        `{variant["start"]}()` activates it and starts its value at zero. Calling
        `{variant["start"]}()` while it is already active throws
        `std::runtime_error`.

        While active:

        - `{variant["credit"]}(amount)` adds a positive amount.
        - `{variant["debit"]}(amount)` removes a positive amount.
        - `{variant["value"]}()` returns the current amount.
        - `{variant["debit"]}` throws `std::runtime_error` when the requested
          amount exceeds the current amount.
        - Both amount-changing operations throw `std::runtime_error` when
          `amount` is zero or negative.

        Calling `{variant["stop"]}()` deactivates an active
        {variant["resource"]}. Calling it before activation or after deactivation
        throws `std::runtime_error`. Every `{variant["value"]}`,
        `{variant["credit"]}`, or `{variant["debit"]}` operation on an inactive
        {variant["resource"]} also throws `std::runtime_error`.

        Reactivating after deactivation starts a fresh value of zero. No amount
        from the previous active lifecycle is retained.

        ## Concurrency

        Many threads call `{variant["credit"]}` and `{variant["debit"]}` on the
        same active object. Each public operation must be internally thread-safe:
        validate and update shared state atomically under synchronization so no
        transaction is lost and no data race occurs.

        ## C++ interface contract

        Preserve this exact public API:

        ```cpp
        {api_block}
        ```

        The object must remain default-constructible exactly as shown. The test
        suite constructs it with
        `{variant["namespace"]}::{variant["class"]} value{{}};`.

        ## File and build contract

        Use only `{variant["stem"]}.h` and `{variant["stem"]}.cpp` for the
        implementation. The header must be self-contained and contain the class
        declaration and all private state required by the implementation. The
        source must define those declared members using the correct namespace and
        class qualification; it must not declare a second replacement class.

        Only those two implementation files are editable during evaluation. The
        test and CMake files are fixed. The implementation must compile as C++17
        with `-Wall -Wextra -Wpedantic -Werror -pthread`.
        """
    )


def standard_header(
    variant: dict[str, str],
    *,
    private_block: str = "    int value_{0};\n    bool active_{false};\n    std::mutex mutex_{};",
    constructor: str | None = None,
    extra_include: str = "",
    extra_private: str = "",
) -> str:
    constructor_line = constructor or f'    {variant["class"]}();'
    includes = "#include <mutex>\n"
    if extra_include:
        includes += extra_include.rstrip() + "\n"
    private_members = private_block.rstrip() + "\n"
    if extra_private:
        private_members += "\n" + extra_private.rstrip() + "\n"
    return (
        "#pragma once\n\n"
        f"{includes}\n"
        f'namespace {variant["namespace"]} {{\n\n'
        f'class {variant["class"]} {{\n'
        "public:\n"
        f"{constructor_line}\n"
        f'    void {variant["start"]}();\n'
        f'    void {variant["credit"]}(int amount);\n'
        f'    void {variant["debit"]}(int amount);\n'
        f'    void {variant["stop"]}();\n'
        f'    int {variant["value"]}();\n\n'
        "private:\n"
        f"{private_members}"
        "};\n\n"
        f'}}  // namespace {variant["namespace"]}\n'
    )


def method_definitions(
    variant: dict[str, str],
    *,
    lock_type: str = "std::lock_guard<std::mutex>",
    value_expr: str = "value_",
    active_expr: str = "active_",
    mutex_expr: str = "mutex_",
    active_true: str = "true",
    active_false: str = "false",
    check_active: str | None = None,
    check_positive: str | None = None,
) -> str:
    cls = variant["class"]
    ns = variant["namespace"]
    active_check = check_active or (
        f'    if ({active_expr} != {active_true}) {{\n'
        '        throw std::runtime_error("resource is not active");\n'
        "    }"
    )
    positive_check = check_positive or (
        '    if (amount <= 0) {\n'
        '        throw std::runtime_error("amount must be positive");\n'
        "    }"
    )
    return (
        f'void {cls}::{variant["start"]}() {{\n'
        f"    {lock_type} guard({mutex_expr});\n"
        f"    if ({active_expr} == {active_true}) {{\n"
        '        throw std::runtime_error("resource is already active");\n'
        "    }\n"
        f"    {value_expr} = 0;\n"
        f"    {active_expr} = {active_true};\n"
        "}\n\n"
        f'void {cls}::{variant["credit"]}(int amount) {{\n'
        f"    {lock_type} guard({mutex_expr});\n"
        f"{active_check}\n"
        f"{positive_check}\n"
        f"    {value_expr} += amount;\n"
        "}\n\n"
        f'void {cls}::{variant["debit"]}(int amount) {{\n'
        f"    {lock_type} guard({mutex_expr});\n"
        f"{active_check}\n"
        f"{positive_check}\n"
        f"    if (amount > {value_expr}) {{\n"
        '        throw std::runtime_error("amount exceeds available value");\n'
        "    }\n"
        f"    {value_expr} -= amount;\n"
        "}\n\n"
        f'void {cls}::{variant["stop"]}() {{\n'
        f"    {lock_type} guard({mutex_expr});\n"
        f"{active_check}\n"
        f"    {active_expr} = {active_false};\n"
        "}\n\n"
        f'int {cls}::{variant["value"]}() {{\n'
        f"    {lock_type} guard({mutex_expr});\n"
        f"{active_check}\n"
        f"    return {value_expr};\n"
        "}\n\n"
        f"}}  // namespace {ns}\n"
    )


def source_with_standard_methods(
    variant: dict[str, str],
    constructor_definition: str,
    **method_kwargs: str,
) -> str:
    constructor_block = (
        f"{constructor_definition}\n\n" if constructor_definition else ""
    )
    return (
        f'#include "{variant["stem"]}.h"\n\n'
        "#include <stdexcept>\n\n"
        f'namespace {variant["namespace"]} {{\n\n'
        f"{constructor_block}"
        f"{method_definitions(variant, **method_kwargs)}"
    )


def inline_header(variant: dict[str, str]) -> str:
    cls = variant["class"]
    return dedent(
        f"""\
        #pragma once

        #include <mutex>
        #include <stdexcept>

        namespace {variant["namespace"]} {{

        class {cls} {{
        public:
            {cls}() = default;

            void {variant["start"]}() {{
                std::lock_guard<std::mutex> guard(mutex_);
                if (active_) {{
                    throw std::runtime_error("resource is already active");
                }}
                value_ = 0;
                active_ = true;
            }}

            void {variant["credit"]}(int amount) {{
                std::lock_guard<std::mutex> guard(mutex_);
                require_active();
                require_positive(amount);
                value_ += amount;
            }}

            void {variant["debit"]}(int amount) {{
                std::lock_guard<std::mutex> guard(mutex_);
                require_active();
                require_positive(amount);
                if (amount > value_) {{
                    throw std::runtime_error("amount exceeds available value");
                }}
                value_ -= amount;
            }}

            void {variant["stop"]}() {{
                std::lock_guard<std::mutex> guard(mutex_);
                require_active();
                active_ = false;
            }}

            int {variant["value"]}() {{
                std::lock_guard<std::mutex> guard(mutex_);
                require_active();
                return value_;
            }}

        private:
            void require_active() const {{
                if (!active_) {{
                    throw std::runtime_error("resource is not active");
                }}
            }}

            static void require_positive(int amount) {{
                if (amount <= 0) {{
                    throw std::runtime_error("amount must be positive");
                }}
            }}

            int value_{{0}};
            bool active_{{false}};
            std::mutex mutex_{{}};
        }};

        }}  // namespace {variant["namespace"]}
        """
    )


def header_and_source(variant: dict[str, str]) -> tuple[str, str]:
    style = variant["style"]
    cls = variant["class"]

    if style == "defaulted_out_of_line":
        header = standard_header(variant)
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() = default;",
        )
    elif style == "explicit_constructor":
        header = standard_header(
            variant,
            private_block="    int amount_;\n    bool enabled_;\n    std::mutex mutex_;",
        )
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() : amount_(0), enabled_(false), mutex_() {{}}",
            value_expr="amount_",
            active_expr="enabled_",
        )
    elif style == "validation_helpers":
        header = standard_header(
            variant,
            extra_private=(
                "    void require_active() const;\n"
                "    static void require_positive(int amount);"
            ),
        )
        active_check = "    require_active();"
        positive_check = "    require_positive(amount);"
        helpers = dedent(
            f"""\
            void {cls}::require_active() const {{
                if (!active_) {{
                    throw std::runtime_error("resource is not active");
                }}
            }}

            void {cls}::require_positive(int amount) {{
                if (amount <= 0) {{
                    throw std::runtime_error("amount must be positive");
                }}
            }}
            """
        )
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() = default;",
            check_active=active_check,
            check_positive=positive_check,
        ).replace(
            f"\n}}  // namespace {variant['namespace']}\n",
            f"\n{helpers}\n}}  // namespace {variant['namespace']}\n",
        )
    elif style == "inline_header":
        header = inline_header(variant)
        source = f'#include "{variant["stem"]}.h"\n'
    elif style == "enum_state":
        header = standard_header(
            variant,
            private_block=(
                "    enum class status { inactive, active };\n\n"
                "    int amount_{0};\n"
                "    status status_{status::inactive};\n"
                "    std::mutex mutex_{};"
            ),
        )
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() = default;",
            value_expr="amount_",
            active_expr="status_",
            active_true="status::active",
            active_false="status::inactive",
        )
    elif style == "unique_lock":
        header = standard_header(variant)
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() = default;",
            lock_type="std::unique_lock<std::mutex>",
        )
    elif style == "scoped_lock":
        header = standard_header(variant)
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() = default;",
            lock_type="std::scoped_lock<std::mutex>",
        )
    elif style == "nested_state":
        header = standard_header(
            variant,
            private_block=(
                "    struct state {\n"
                "        int value{0};\n"
                "        bool active{false};\n"
                "    };\n\n"
                "    state state_{};\n"
                "    std::mutex mutex_{};"
            ),
        )
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() = default;",
            value_expr="state_.value",
            active_expr="state_.active",
        )
    elif style == "defaulted_in_header":
        header = standard_header(
            variant,
            constructor=f'    {cls}() = default;',
        )
        source = source_with_standard_methods(variant, "")
    elif style == "plain_out_of_line":
        header = standard_header(
            variant,
            private_block=(
                "    int count_;\n"
                "    bool unlocked_;\n"
                "    std::mutex mutex_;"
            ),
        )
        source = source_with_standard_methods(
            variant,
            f"{cls}::{cls}() : count_(0), unlocked_(false) {{}}",
            value_expr="count_",
            active_expr="unlocked_",
        )
    else:
        raise ValueError(f"unknown style: {style}")

    return header, source


def tests_for(variant: dict[str, str]) -> str:
    qualified = f'{variant["namespace"]}::{variant["class"]}'
    start = variant["start"]
    credit = variant["credit"]
    debit = variant["debit"]
    stop = variant["stop"]
    value = variant["value"]
    return dedent(
        f"""\
        #include "{variant["stem"]}.h"

        #include <chrono>
        #include <cstddef>
        #include <exception>
        #include <iostream>
        #include <stdexcept>
        #include <string_view>
        #include <thread>
        #include <utility>
        #include <vector>

        namespace {{

        class check_suite {{
        public:
            void expect(bool condition, std::string_view description) {{
                ++count_;
                if (!condition) {{
                    failed_ = true;
                    std::cerr << "FAILED: " << description << '\\n';
                }}
            }}

            template <typename Function>
            void expect_runtime_error(Function&& function, std::string_view description) {{
                ++count_;
                try {{
                    std::forward<Function>(function)();
                }} catch (std::runtime_error const&) {{
                    return;
                }} catch (std::exception const& error) {{
                    failed_ = true;
                    std::cerr << "FAILED: " << description
                              << " threw a different exception: " << error.what() << '\\n';
                    return;
                }}
                failed_ = true;
                std::cerr << "FAILED: " << description
                          << " did not throw std::runtime_error\\n";
            }}

            std::size_t count() const {{ return count_; }}
            bool failed() const {{ return failed_; }}

        private:
            std::size_t count_{{0}};
            bool failed_{{false}};
        }};

        }}  // namespace

        int main() {{
            check_suite checks;

            {{
                {qualified} subject{{}};
                subject.{start}();
                checks.expect(subject.{value}() == 0, "newly_started_zero");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(100);
                checks.expect(subject.{value}() == 100, "single_credit");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(100);
                subject.{credit}(50);
                checks.expect(subject.{value}() == 150, "multiple_credits");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(100);
                subject.{debit}(75);
                checks.expect(subject.{value}() == 25, "single_debit");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(100);
                subject.{debit}(80);
                subject.{debit}(20);
                checks.expect(subject.{value}() == 0, "multiple_debits");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(100);
                subject.{credit}(110);
                subject.{debit}(200);
                subject.{credit}(60);
                subject.{debit}(50);
                checks.expect(subject.{value}() == 20, "sequential_operations");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{stop}();
                checks.expect_runtime_error(
                    [&]() {{ (void)subject.{value}(); }},
                    "value_after_stop_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{stop}();
                checks.expect_runtime_error(
                    [&]() {{ subject.{credit}(50); }},
                    "credit_after_stop_throws");
            }}
            {{
                {qualified} subject{{}};
                checks.expect_runtime_error(
                    [&]() {{ subject.{credit}(50); }},
                    "credit_before_start_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{stop}();
                checks.expect_runtime_error(
                    [&]() {{ subject.{debit}(50); }},
                    "debit_after_stop_throws");
            }}
            {{
                {qualified} subject{{}};
                checks.expect_runtime_error(
                    [&]() {{ subject.{stop}(); }},
                    "stop_before_start_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                checks.expect_runtime_error(
                    [&]() {{ subject.{start}(); }},
                    "start_twice_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(50);
                subject.{stop}();
                subject.{start}();
                checks.expect(subject.{value}() == 0, "restart_resets_zero");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(25);
                checks.expect_runtime_error(
                    [&]() {{ subject.{debit}(50); }},
                    "overdraft_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                subject.{credit}(100);
                checks.expect_runtime_error(
                    [&]() {{ subject.{debit}(-50); }},
                    "negative_debit_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                checks.expect_runtime_error(
                    [&]() {{ subject.{credit}(-50); }},
                    "negative_credit_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                std::vector<std::thread> threads;
                threads.reserve(1000);
                for (int index = 0; index < 1000; ++index) {{
                    threads.emplace_back([&]() {{
                        using namespace std::chrono_literals;
                        subject.{credit}(1);
                        std::this_thread::sleep_for(5ms);
                        subject.{debit}(1);
                    }});
                }}
                for (auto& thread : threads) {{
                    thread.join();
                }}
                checks.expect(subject.{value}() == 0, "concurrent_transactions");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                checks.expect_runtime_error(
                    [&]() {{ subject.{debit}(0); }},
                    "zero_debit_throws");
            }}
            {{
                {qualified} subject{{}};
                subject.{start}();
                checks.expect_runtime_error(
                    [&]() {{ subject.{credit}(0); }},
                    "zero_credit_throws");
            }}

            if (checks.failed() || checks.count() != 19U) {{
                std::cerr << "Verification failed after " << checks.count()
                          << " assertions\\n";
                return 1;
            }}

            std::cout << "All tests passed (19 assertions in 19 test cases)\\n";
            return 0;
        }}
        """
    )


def cmake_for(variant: dict[str, str]) -> str:
    return dedent(
        f"""\
        cmake_minimum_required(VERSION 3.16)
        project({variant["stem"]} LANGUAGES CXX)

        find_package(Threads REQUIRED)

        add_executable(
            {variant["stem"]}_tests
            {variant["stem"]}.cpp
            {variant["stem"]}_test.cpp
        )

        target_compile_features({variant["stem"]}_tests PRIVATE cxx_std_17)
        target_compile_options(
            {variant["stem"]}_tests
            PRIVATE -Wall -Wextra -Wpedantic -Werror
        )
        target_link_libraries({variant["stem"]}_tests PRIVATE Threads::Threads)

        enable_testing()
        add_test(NAME {variant["stem"]} COMMAND {variant["stem"]}_tests)
        """
    )


def suite_readme() -> str:
    rows = "\n        ".join(
        f'| [{variant["title"]}](variants/{variant["id"]}/PROMPT.md) | '
        f'`{variant["namespace"]}::{variant["class"]}` | '
        f'`{variant["start"]}` / `{variant["credit"]}` / '
        f'`{variant["debit"]}` / `{variant["stop"]}` / `{variant["value"]}` |'
        for variant in VARIANTS
    )
    return dedent(
        f"""\
        # Bank-account-equivalent C++ benchmark variants

        This directory contains ten independently named benchmark fixtures for
        the complete fixed26 `bank-account` state-machine, exception, file-layout,
        and concurrency contract. They are intended only for new benchmark
        construction and robustness evaluation. They are explicitly excluded
        from SFT, RL, distillation, and every other training corpus.

        ## Coverage and lineage

        The parent static test SHA-256 is
        `{PARENT_TEST_SHA256}`.
        Each variant maps all 17 parent cases one-for-one and adds two tests for
        zero-valued credit and debit operations. Those two checks close the
        historical gap where the prompt said “non-positive” but the parent suite
        exercised only negative values.

        Every variant therefore contains 19 assertions:

        - six ordinary sequential state/value cases;
        - seven lifecycle and inactive-state exception cases;
        - five amount-boundary cases, including negative, zero, and overdraft;
        - one 1,000-thread transaction test.

        The historical July 27 prompt omitted reset-on-reopen even though the
        parent test required it. These prompts include the corrected lifecycle
        rule. They also state default construction and header/source ownership
        explicitly because six of the eight observed failures were C++ delivery
        failures rather than incorrect account arithmetic.

        | Variant | Public class | Operations |
        | --- | --- | --- |
        {rows}

        ## Verify

        From the repository root:

        ```bash
        python3 benchmarks/cpp/bank-account-equivalent-v1/verify.py
        ```

        Verification checks lineage and file hashes, exact API and prompt
        coverage, the 17+2 test inventory, standalone header compilation, strict
        C++17 compilation, a deadlock timeout, and all 190 runtime assertions.
        The committed
        [`verification_receipt.json`](verification_receipt.json) records the
        successful environment and per-variant results.

        The deterministic source generator is
        [`scripts/build_bank_account_equivalent_benchmarks.py`](../../../scripts/build_bank_account_equivalent_benchmarks.py).
        """
    )


def main() -> None:
    suite_readme_path = SUITE_ROOT / "README.md"
    write(suite_readme_path, suite_readme())
    generated_files = [suite_readme_path]
    manifest_variants = []

    for variant in VARIANTS:
        variant_root = SUITE_ROOT / "variants" / variant["id"]
        header, source = header_and_source(variant)
        contents = {
            "PROMPT.md": prompt_for(variant),
            f'{variant["stem"]}.h': header,
            f'{variant["stem"]}.cpp': source,
            f'{variant["stem"]}_test.cpp': tests_for(variant),
            "CMakeLists.txt": cmake_for(variant),
        }
        files = {}
        for name, content in contents.items():
            path = variant_root / name
            write(path, content)
            generated_files.append(path)
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        api = {
            "namespace": variant["namespace"],
            "class": variant["class"],
            "constructor": variant["class"],
            "start": variant["start"],
            "credit": variant["credit"],
            "debit": variant["debit"],
            "stop": variant["stop"],
            "value": variant["value"],
        }
        manifest_variants.append(
            {
                "id": variant["id"],
                "title": variant["title"],
                "directory": str(variant_root.relative_to(SUITE_ROOT)),
                "domain": {
                    "resource": variant["resource"],
                    "unit": variant["unit"],
                },
                "api": api,
                "api_mapping": [
                    {
                        "role": role,
                        "parent": PARENT_API[role],
                        "variant": api[role],
                    }
                    for role in PARENT_API
                ],
                "reference_style": variant["style"],
                "tests": {
                    "parent_mapped": list(PARENT_TESTS),
                    "contract_completion": list(EXTRA_TESTS),
                    "parent_count": 17,
                    "additional_count": 2,
                    "total": 19,
                    "concurrent_threads": 1000,
                },
                "files": files,
            }
        )

    manifest = {
        "schema_version": 1,
        "kind": "cpp_benchmark_variant_suite",
        "id": "bank-account-equivalent-v1",
        "created_date": "2026-07-30",
        "role": "benchmark_and_evaluation_only",
        "training_exclusion": {
            "trainable": False,
            "excluded_uses": [
                "supervised_fine_tuning",
                "reinforcement_learning",
                "distillation",
                "training_data_augmentation",
            ],
        },
        "lineage": {
            "parent_task": "Aider Polyglot C++ fixed26/bank-account",
            "parent_test_sha256": PARENT_TEST_SHA256,
            "historical_prompt_sha256": HISTORICAL_PROMPT_SHA256,
            "corrected_contract_sha256": CORRECTED_CONTRACT_SHA256,
            "parent_contract": "fixed26-contract-v2",
            "contract_audit_issue": (
                "https://github.com/tokenbender/browser-is-all-you-need/issues/89"
            ),
            "construction_issue": (
                "https://github.com/tokenbender/browser-is-all-you-need/issues/92"
            ),
            "parent_api": PARENT_API,
            "historical_single_turn_failures": 8,
            "failure_taxonomy": list(FAILURE_TAXONOMY),
            "mapping": "complete_behavioral_api_and_test_case_mapping",
        },
        "contract": {
            "language": "C++17",
            "compiler_flags": [
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-pthread",
            ],
            "variant_count": 10,
            "parent_assertions_per_variant": 17,
            "additional_assertions_per_variant": 2,
            "assertions_per_variant": 19,
            "total_assertions": 190,
            "total_concurrent_threads": 10000,
            "runtime_timeout_seconds_per_variant": 30,
        },
        "verification": {
            "status": "verified_by_committed_receipt",
            "command": "python3 benchmarks/cpp/bank-account-equivalent-v1/verify.py",
            "receipt": "verification_receipt.json",
        },
        "generator": {
            "path": "scripts/build_bank_account_equivalent_benchmarks.py",
            "sha256": sha256(Path(__file__)),
        },
        "root_files": {
            "README.md": {
                "bytes": suite_readme_path.stat().st_size,
                "sha256": sha256(suite_readme_path),
            }
        },
        "variants": manifest_variants,
    }
    write(SUITE_ROOT / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(
        f"Wrote {len(generated_files)} generated files for "
        f"{len(VARIANTS)} variants to {SUITE_ROOT}"
    )


if __name__ == "__main__":
    main()
