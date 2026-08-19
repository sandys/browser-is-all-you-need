#!/usr/bin/env python3
"""Fairness overlay for the Aider fixed-26 C++ evaluation set.

The fixed-26 prompt is built from ``.docs/instructions.md`` plus the two editable
starter files. The test file is never shown to the model, and every starter
header is an empty stub, so the required namespace/class/method names have to be
guessed. Several are unguessable (``bank-account`` lives in namespace
``Bankaccount``; ``clock`` lives in ``date_independent``). The harness also
appends "Only use standard libraries, don't suggest installing any packages",
which is false for the two Boost exercises.

This module appends a stated interface contract and build-environment section to
each exercise's ``instructions.md``. It states *what the tests call* -- names,
signatures, exception types, and the build constraints -- and never states how to
implement anything.

Every edit is pinned to the sha256 of the original ``instructions.md``. If
upstream polyglot-benchmark text ever changes, application fails loudly instead
of silently producing a different benchmark.

Usage:
  aider_fixed26_contract_overlay.py apply <root> [--check]
  aider_fixed26_contract_overlay.py manifest <root>
  aider_fixed26_contract_overlay.py audit <root>

``<root>`` is a directory containing the 26 exercise directories (i.e. a
``cpp/exercises/practice`` tree, or the self-eval ``pristine`` tree).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1
KIND = "aider-fixed26-contract-overlay"
OVERLAY_VERSION = "fixed26-contract-v2"
MARKER = "## C++ interface contract"

# sha256 of each pristine .docs/instructions.md as shipped by
# Aider-AI/polyglot-benchmark @ 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f.
ORIGINAL_SHA256: dict[str, str] = {}  # populated from originals.json beside this file

_COMMON = """
## Build environment

- The exercise is compiled as C++17 with `-Wall -Wextra -Wpedantic -Werror`, so
  any warning fails the build.
- Only `{stem}.h` and `{stem}.cpp` are editable. `CMakeLists.txt` and the test
  file are fixed and must not be modified.
- The test file includes only `{stem}.h`, so every name above must be visible
  from that header.{extra}
"""

_HEADER_ONLY_NOTE = """
- The declarations above are templates, so their definitions must live in
  `{stem}.h`. Leaving `{stem}.cpp` unchanged is fine."""

_SPLIT_NOTE = """
- You may either declare in `{stem}.h` and define in `{stem}.cpp`, or define
  everything `inline`/in-class in `{stem}.h` and leave `{stem}.cpp` unchanged.
  Both are accepted."""

_BOOST_NOTE = """
- Boost (1.58 or newer) is installed and `CMakeLists.txt` already links
  `Boost::date_time`. Use it. The generic "only use standard libraries, don't
  suggest installing any packages" instruction does not apply to this exercise:
  Boost is part of the provided environment and needs no installation.
- Include it with `#include {boost_header}`."""

_INTRO = """
{marker}

The test file is not shown to you, so the interface it expects is stated here in
full. Implement exactly these names and signatures; the tests use nothing else.

```cpp
{decl}
```
"""


def _section(stem: str, decl: str, *, notes: str = "", header_only: bool = False,
             boost_header: str | None = None) -> str:
    extra = ""
    if boost_header:
        extra += _BOOST_NOTE.format(boost_header=boost_header)
    extra += (_HEADER_ONLY_NOTE if header_only else _SPLIT_NOTE).format(stem=stem)
    out = _INTRO.format(marker=MARKER, decl=decl.strip())
    if notes:
        out += "\n" + notes.strip() + "\n"
    out += _COMMON.format(stem=stem, extra=extra)
    return out.rstrip() + "\n"


# --------------------------------------------------------------------------
# Per-task contracts. Declarations state the public API the tests exercise;
# private members and any implementation detail are deliberately omitted.
# --------------------------------------------------------------------------

CONTRACTS: dict[str, str] = {
    "all-your-base": _section(
        "all_your_base",
        """
namespace all_your_base {
std::vector<unsigned int> convert(unsigned int input_base,
                                  const std::vector<unsigned int>& input_digits,
                                  unsigned int output_base);
}
""",
        notes="A base below 2 or a digit outside the input base must raise "
              "`std::invalid_argument`. An empty digit sequence and any sequence "
              "representing zero must return an empty vector.",
    ),
    "allergies": _section(
        "allergies",
        """
namespace allergies {
class allergy_test {
public:
    allergy_test(unsigned int test_result);
    bool is_allergic_to(std::string const& allergen) const;
    std::unordered_set<std::string> get_allergies() const;
};
}
""",
    ),
    "bank-account": _section(
        "bank_account",
        """
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
""",
        notes="Note the namespace and class name are both `Bankaccount` (capital B, one "
              "word), not `bank_account`.\n\n"
              "Every misuse throws `std::runtime_error`: opening an already-open account, "
              "closing or using an account that is not open, depositing or withdrawing a "
              "non-positive amount, and withdrawing more than the balance.\n\n"
              "A newly opened account has a zero balance. After `close()`, a later "
              "`open()` starts a fresh account at zero; the previous balance is not "
              "retained.\n\n"
              "The tests call `deposit` and `withdraw` concurrently from many "
              "`std::thread`s on one account, so the class must be internally "
              "thread-safe.",
    ),
    "binary-search-tree": _section(
        "binary_search_tree",
        """
namespace binary_search_tree {
template <typename T>
class binary_tree {
public:
    explicit binary_tree(T data);
    void insert(T data);
    const T& data() const;
    /* left() and right() return an owning pointer-like value (e.g.
       std::unique_ptr<binary_tree>) that is contextually convertible to bool
       and supports ->data(), ->left(), ->right(). */
    const <pointer-like>& left() const;
    const <pointer-like>& right() const;
    <iterator> begin() const;
    <iterator> end() const;
};
}
""",
        notes="Values less than or equal to a node go left, greater go right.\n\n"
              "`begin()`/`end()` must yield the values in sorted (in-order) order and "
              "support `*it`, `++it`, and `!=`, so that a range-for over the tree "
              "produces ascending values.",
        header_only=True,
    ),
    "circular-buffer": _section(
        "circular_buffer",
        """
namespace circular_buffer {
template <typename ValueType>
class circular_buffer {
public:
    circular_buffer(std::size_t capacity);
    ValueType read();
    void write(ValueType item);
    void overwrite(ValueType item);
    void clear();
};
}
""",
        notes="`read()` on an empty buffer and `write()` on a full buffer both throw "
              "`std::domain_error`. `overwrite()` on a full buffer replaces the oldest "
              "element instead of throwing. The tests instantiate the template with both "
              "`int` and `std::string`.",
        header_only=True,
    ),
    "clock": _section(
        "clock",
        """
namespace date_independent {
class clock {
public:
    static clock at(int hour, int minute = 0);
    clock& plus(int minutes);
    clock& minus(int minutes);
    operator std::string() const;
    bool operator==(const clock& rhs) const;
};
bool operator!=(const clock& lhs, const clock& rhs);
}
""",
        notes="Note the namespace is `date_independent`, not `clock`.\n\n"
              "The conversion to `std::string` must be implicit (the tests write "
              "`string(clock::at(8, 0))`) and must produce zero-padded `HH:MM`, e.g. "
              "`\"08:00\"`. Hours and minutes wrap in both directions, including negative "
              "values and multiples of a day.",
    ),
    "complex-numbers": _section(
        "complex_numbers",
        """
namespace complex_numbers {
class Complex {
public:
    Complex(double real, double imaginary);
    Complex operator+(const Complex& other) const;
    Complex operator-(const Complex& other) const;
    Complex operator*(const Complex& other) const;
    Complex operator/(const Complex& other) const;
    double abs() const;
    Complex conj() const;
    double real() const;
    double imag() const;
    Complex exp() const;
};
bool operator==(const Complex& lhs, const Complex& rhs);
std::ostream& operator<<(std::ostream& os, Complex const& value);
Complex operator+(const Complex& complex, double scalar);
Complex operator+(double scalar, const Complex& complex);
Complex operator-(const Complex& complex, double scalar);
Complex operator-(double scalar, const Complex& complex);
Complex operator*(const Complex& complex, double scalar);
Complex operator*(double scalar, const Complex& complex);
Complex operator/(const Complex& complex, double scalar);
Complex operator/(double scalar, const Complex& complex);
}
""",
        notes="Both scalar orderings are required for each of the four arithmetic "
              "operators. The tests compare with a floating-point tolerance, so exact "
              "bit equality is not required.",
    ),
    "crypto-square": _section(
        "crypto_square",
        """
namespace crypto_square {
class cipher {
public:
    cipher(std::string const& text);
    std::string normalize_plain_text() const;
    std::size_t size() const;
    std::vector<std::string> plain_text_segments() const;
    std::string cipher_text() const;
    std::string normalized_cipher_text() const;
};
}
""",
    ),
    "diamond": _section(
        "diamond",
        """
namespace diamond {
std::vector<std::string> rows(char middle_letter);
}
""",
        notes="Each returned row is a full square line: leading and trailing padding "
              "included, so every string has the same length.",
    ),
    "dnd-character": _section(
        "dnd_character",
        """
namespace dnd_character {
int modifier(int score);
int ability();
struct Character {
    Character();
    int strength;
    int dexterity;
    int constitution;
    int intelligence;
    int wisdom;
    int charisma;
    int hitpoints;
};
}
""",
        notes="`ability()` rolls four six-sided dice and sums the largest three, so it "
              "returns a value in [3, 18]. The default-constructed `Character` must roll "
              "all six abilities and set `hitpoints` to 10 + `modifier(constitution)`.",
    ),
    "gigasecond": _section(
        "gigasecond",
        """
namespace gigasecond {
boost::posix_time::ptime advance(const boost::posix_time::ptime& start);
}
""",
        notes="Taking the parameter by value is equally acceptable; the return type must "
              "be `boost::posix_time::ptime`.",
        boost_header='"boost/date_time/posix_time/posix_time.hpp"',
    ),
    "grade-school": _section(
        "grade_school",
        """
namespace grade_school {
class school {
public:
    const std::map<int, std::vector<std::string>>& roster() const;
    void add(std::string const& name, int grade);
    std::vector<std::string> grade(int grade) const;
};
}
""",
        notes="Names within a grade are sorted alphabetically, and `roster()` is keyed by "
              "grade in ascending order. `grade()` on an unknown grade returns an empty "
              "vector.",
    ),
    "kindergarten-garden": _section(
        "kindergarten_garden",
        """
namespace kindergarten_garden {
enum class Plants : char {
    grass = 'G', clover = 'C', radishes = 'R', violets = 'V'
};
std::array<Plants, 4> plants(std::string_view diagram, std::string_view student);
}
""",
        notes="The enum name, its enumerator names, and its `char` underlying values are "
              "all required exactly as written.",
    ),
    "knapsack": _section(
        "knapsack",
        """
namespace knapsack {
struct Item {
    int weight;
    int value;
};
int maximum_value(int maximum_weight, const std::vector<Item>& items);
}
""",
        notes="The tests build `Item` values with aggregate initialisation in "
              "`{weight, value}` order, so the member order matters.",
    ),
    "linked-list": _section(
        "linked_list",
        """
namespace linked_list {
template <typename T>
class List {
public:
    List();
    void push(T entry);
    void unshift(T entry);
    T pop();
    T shift();
    bool erase(T entry);
    std::size_t count();
};
}
""",
        notes="`push`/`pop` operate on the back, `unshift`/`shift` on the front. `erase` "
              "removes the first matching element and returns whether it removed one. "
              "Removing from an empty list throws `std::runtime_error`.",
        header_only=True,
    ),
    "meetup": _section(
        "meetup",
        """
namespace meetup {
class scheduler {
public:
    scheduler(boost::gregorian::date::month_type month,
              boost::gregorian::date::year_type year);

    // "teenth" = the weekday falling on days 13-19 of the month.
    boost::gregorian::date monteenth() const;
    boost::gregorian::date tuesteenth() const;
    boost::gregorian::date wednesteenth() const;
    boost::gregorian::date thursteenth() const;
    boost::gregorian::date friteenth() const;
    boost::gregorian::date saturteenth() const;
    boost::gregorian::date sunteenth() const;

    boost::gregorian::date first_monday() const;
    boost::gregorian::date first_tuesday() const;
    boost::gregorian::date first_wednesday() const;
    boost::gregorian::date first_thursday() const;
    boost::gregorian::date first_friday() const;
    boost::gregorian::date first_saturday() const;
    boost::gregorian::date first_sunday() const;

    // ... and the same seven weekday methods with the prefixes
    // second_, third_, fourth_, and last_ (35 methods in total).
};
}
""",
        notes="The constructor takes the month first and the year second. All 49 methods "
              "(7 teenth + 7 each for first/second/third/fourth/last) are exercised.",
        boost_header='"boost/date_time/gregorian/gregorian.hpp"',
    ),
    "parallel-letter-frequency": _section(
        "parallel_letter_frequency",
        """
namespace parallel_letter_frequency {
std::unordered_map<char, size_t> frequency(
    std::vector<std::string_view> const& texts);
}
""",
        notes="Count ASCII letters case-insensitively, keyed by the lowercase letter. "
              "Non-letters are ignored. Despite the exercise name, a correct "
              "single-threaded implementation passes the tests.",
    ),
    "perfect-numbers": _section(
        "perfect_numbers",
        """
namespace perfect_numbers {
enum class classification { deficient, perfect, abundant };
classification classify(int n);
}
""",
        notes="`classify` throws `std::domain_error` for zero and for negative input.",
    ),
    "phone-number": _section(
        "phone_number",
        """
namespace phone_number {
class phone_number {
public:
    phone_number(const std::string& text);
    std::string area_code() const;
    std::string number() const;
    explicit operator std::string() const;
};
}
""",
        notes="The constructor throws `std::domain_error` for any invalid NANP input "
              "(wrong digit count, a leading country code other than 1, or an area or "
              "exchange code starting with 0 or 1). `number()` returns the bare ten "
              "digits. The conversion to `std::string` is `explicit`.",
    ),
    "queen-attack": _section(
        "queen_attack",
        """
namespace queen_attack {
class chess_board {
public:
    chess_board(const std::pair<int, int>& white, const std::pair<int, int>& black);
    chess_board();
    std::pair<int, int> white() const;
    std::pair<int, int> black() const;
    operator std::string() const;
    bool can_attack() const;
};
}
""",
        notes="Positions are `{row, column}`, each in [0, 7]. The constructor throws "
              "`std::domain_error` for an off-board position or for two queens on the "
              "same square. The default constructor places white at `{0, 3}` and black at "
              "`{7, 3}`. The tests construct with brace initialisation, e.g. "
              "`chess_board{white, black}`.",
    ),
    "robot-name": _section(
        "robot_name",
        """
namespace robot_name {
class robot {
public:
    robot();
    std::string const& name() const;
    void reset();
};
}
""",
        notes="A name is two uppercase letters followed by three digits, e.g. `\"RX837\"`. "
              "`reset()` assigns a new name. The tests create many robots and require all "
              "names to be distinct, so issued names must never repeat.",
    ),
    "space-age": _section(
        "space_age",
        """
namespace space_age {
class space_age {
public:
    explicit space_age(unsigned long long seconds);
    unsigned long long seconds() const;
    double on_earth() const;
    double on_mercury() const;
    double on_venus() const;
    double on_mars() const;
    double on_jupiter() const;
    double on_saturn() const;
    double on_uranus() const;
    double on_neptune() const;
};
}
""",
        notes="The class and its namespace are both named `space_age`. The constructor is "
              "`explicit`. Results are compared with a floating-point tolerance.",
    ),
    "spiral-matrix": _section(
        "spiral_matrix",
        """
namespace spiral_matrix {
std::vector<std::vector<uint32_t>> spiral_matrix(uint32_t size);
}
""",
        notes="The function has the same name as its namespace; the tests call it as "
              "`spiral_matrix::spiral_matrix(n)`. Size 0 returns an empty vector.",
    ),
    "sublist": _section(
        "sublist",
        """
namespace sublist {
enum class List_comparison { equal, sublist, superlist, unequal };
List_comparison sublist(const std::vector<int>& list_one,
                        const std::vector<int>& list_two);
}
""",
        notes="Note the capital L and underscore in `List_comparison`. The result is "
              "stated from the perspective of `list_one` relative to `list_two`: "
              "`sublist` means `list_one` is contained in `list_two`.",
    ),
    "yacht": _section(
        "yacht",
        """
namespace yacht {
int score(std::array<int, 5> dice, const std::string& category);
}
""",
        notes="The category strings the tests pass are exactly: `\"ones\"`, `\"twos\"`, "
              "`\"threes\"`, `\"fours\"`, `\"fives\"`, `\"sixes\"`, `\"full house\"`, "
              "`\"four of a kind\"`, `\"little straight\"`, `\"big straight\"`, "
              "`\"choice\"`, and `\"yacht\"`.",
    ),
    "zebra-puzzle": _section(
        "zebra_puzzle",
        """
namespace zebra_puzzle {
struct Solution {
    std::string drinksWater;
    std::string ownsZebra;
};
Solution solve();
}
""",
        notes="Member names are camelCase exactly as written. Each member holds a "
              "nationality, e.g. `\"Englishman\"`.",
    ),
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _originals_path() -> Path:
    return Path(__file__).with_name("aider_fixed26_originals.json")


def load_originals() -> dict[str, str]:
    path = _originals_path()
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned original hashes: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("kind") != KIND or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected originals manifest: {path}")
    return data["instructions_sha256"]


def load_test_hashes() -> dict[str, dict[str, str]]:
    path = _originals_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    tests = data.get("tests")
    if not isinstance(tests, dict):
        raise ValueError(f"missing pinned test ledger: {path}")
    return tests


def instructions_path(root: Path, task: str) -> Path:
    return root / task / ".docs" / "instructions.md"


def apply(root: Path, *, check_only: bool = False) -> dict[str, object]:
    originals = load_originals()
    present = sorted(p.name for p in root.iterdir() if p.is_dir())
    unknown = sorted(set(present) - set(CONTRACTS))
    if unknown:
        raise ValueError(f"root {root} has exercises with no contract: {unknown}")
    if not present:
        raise ValueError(f"root {root} contains no exercises")

    records = []
    for task in present:
        path = instructions_path(root, task)
        if not path.is_file():
            raise FileNotFoundError(f"missing instructions: {path}")
        text = path.read_text(encoding="utf-8")

        if MARKER in text:
            base, _, _ = text.partition("\n" + MARKER)
            base = base.rstrip("\n") + "\n"
            already = True
        else:
            base = text.rstrip("\n") + "\n"
            already = False

        base_sha = sha256_text(base)
        expected = originals.get(task)
        if expected is None:
            raise ValueError(f"no pinned original hash for {task}")
        if base_sha != expected:
            raise ValueError(
                f"original instructions for {task} do not match the pinned hash "
                f"(got {base_sha}, want {expected}); upstream text changed"
            )

        patched = base + "\n" + CONTRACTS[task]
        patched_sha = sha256_text(patched)
        if not check_only and (already or text != patched):
            path.write_text(patched, encoding="utf-8")
        records.append({
            "task": task,
            "original_sha256": base_sha,
            "patched_sha256": patched_sha,
            "was_already_patched": already,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "overlay_version": OVERLAY_VERSION,
        "root": str(root),
        "tasks": len(records),
        "records": records,
        "overlay_sha256": sha256_text(
            "".join(r["patched_sha256"] for r in records)  # type: ignore[misc]
        ),
    }


_REGRESSION_REQUIREMENTS: dict[str, dict[str, list[str]]] = {
    "all-your-base": {
        "prompt": [
            "must raise `std::invalid_argument`",
            "representing zero must return an empty vector",
        ],
        "test": [
            'TEST_CASE("empty_list")',
            'TEST_CASE("single_zero")',
            'TEST_CASE("multiple_zeros")',
            'TEST_CASE("first_base_is_one")',
            'TEST_CASE("first_base_is_zero")',
            'TEST_CASE("invalid_positive_digit")',
            'TEST_CASE("second_base_is_one")',
            'TEST_CASE("second_base_is_zero")',
        ],
    },
    "bank-account": {
        "prompt": [
            "A newly opened account has a zero balance.",
            "the previous balance is not retained.",
        ],
        "test": ['TEST_CASE("Reopened account does not retain balance"'],
    },
    "gigasecond": {
        "prompt": [
            "Boost (1.58 or newer) is installed",
            'Include it with `#include "boost/date_time/posix_time/posix_time.hpp"`',
        ],
        "test": ['#include "boost/date_time/posix_time/posix_time.hpp"'],
    },
    "meetup": {
        "prompt": [
            "Boost (1.58 or newer) is installed",
            'Include it with `#include "boost/date_time/gregorian/gregorian.hpp"`',
        ],
        "test": ["#include <boost/date_time/gregorian/gregorian.hpp>"],
    },
}


def audit(root: Path, *, require_all: bool = False) -> dict[str, object]:
    """Fail closed unless all 26 reviewed prompt/test pairs match the pinned contract."""

    tests = load_test_hashes()
    present = sorted(path.name for path in root.iterdir() if path.is_dir())
    if set(tests) != set(CONTRACTS):
        raise ValueError("review ledger must cover exactly the pinned fixed-26 tasks")
    unknown = sorted(set(present) - set(CONTRACTS))
    if not present or unknown:
        raise ValueError(f"contract audit has no tasks or unknown tasks: {unknown}")
    if require_all and set(present) != set(CONTRACTS):
        raise ValueError("full contract audit must cover exactly the pinned fixed-26 tasks")

    records: list[dict[str, object]] = []
    for task in present:
        prompt_path = instructions_path(root, task)
        prompt = prompt_path.read_text(encoding="utf-8")
        expected_prompt = (
            prompt.partition("\n" + MARKER)[0].rstrip("\n") + "\n\n" + CONTRACTS[task]
        )
        if prompt != expected_prompt:
            raise ValueError(f"patched prompt does not match reviewed contract: {task}")

        test_record = tests[task]
        test_path = root / task / str(test_record["file"])
        if not test_path.is_file():
            raise FileNotFoundError(f"missing pinned test: {test_path}")
        test_text = test_path.read_text(encoding="utf-8")
        test_sha = sha256_text(test_text)
        if test_sha != test_record["sha256"]:
            raise ValueError(
                f"test for {task} does not match the reviewed hash "
                f"(got {test_sha}, want {test_record['sha256']})"
            )

        requirements = _REGRESSION_REQUIREMENTS.get(task, {})
        missing_prompt = [
            fragment for fragment in requirements.get("prompt", []) if fragment not in prompt
        ]
        missing_test = [
            fragment for fragment in requirements.get("test", []) if fragment not in test_text
        ]
        if missing_prompt or missing_test:
            raise ValueError(
                f"unexplained deterministic requirements for {task}: "
                f"prompt={missing_prompt}, test={missing_test}"
            )
        records.append({
            "task": task,
            "classification": "prompt_test_aligned",
            "review_disposition": "reviewed_allowlist",
            "prompt_sha256": sha256_text(prompt),
            "test_file": test_record["file"],
            "test_sha256": test_sha,
            "regression_requirements": requirements,
            "unexplained_deterministic_requirements": 0,
        })

    audit_sha = sha256_text(
        "".join(
            f"{record['task']}:{record['prompt_sha256']}:{record['test_sha256']}\n"
            for record in records
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "aider-fixed26-prompt-test-audit",
        "overlay_version": OVERLAY_VERSION,
        "root": str(root),
        "tasks": len(records),
        "reviewed_prompt_test_pairs": len(records),
        "unexplained_deterministic_requirements": 0,
        "audit_sha256": audit_sha,
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["apply", "manifest", "audit"])
    ap.add_argument("root", type=Path)
    ap.add_argument("--check", action="store_true", help="verify without writing")
    args = ap.parse_args()
    root = args.root.resolve()
    if args.command == "audit":
        result = audit(root, require_all=True)
    else:
        result = apply(root, check_only=args.check or args.command == "manifest")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
