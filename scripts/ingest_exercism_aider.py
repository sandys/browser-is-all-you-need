#!/usr/bin/env python3
"""Ingest Aider Polyglot C++ exercises into self-contained task JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm47_posttraining.cpp_perf.schema import AiderPolyglotTask


CATEGORY_MAPPING = {
    "allergies": "Numerical reasoning",
    "all-your-base": "Numerical reasoning",
    "bank-account": "State & concurrency",
    "binary-search-tree": "Algorithms & data structures",
    "circular-buffer": "Algorithms & data structures",
    "clock": "Time & date",
    "complex-numbers": "Numerical reasoning",
    "crypto-square": "Text & parsing",
    "diamond": "Text & parsing",
    "dnd-character": "Numerical reasoning",
    "gigasecond": "Time & date",
    "grade-school": "Algorithms & data structures",
    "kindergarten-garden": "Text & parsing",
    "knapsack": "Algorithms & data structures",
    "linked-list": "Algorithms & data structures",
    "meetup": "Time & date",
    "parallel-letter-frequency": "State & concurrency",
    "perfect-numbers": "Numerical reasoning",
    "phone-number": "Text & parsing",
    "queen-attack": "Logic, grids & games",
    "robot-name": "State & concurrency",
    "space-age": "Numerical reasoning",
    "spiral-matrix": "Logic, grids & games",
    "sublist": "Algorithms & data structures",
    "yacht": "Logic, grids & games",
    "zebra-puzzle": "Logic, grids & games",
}

EASY = {
    "allergies",
    "clock",
    "complex-numbers",
    "diamond",
    "gigasecond",
    "perfect-numbers",
    "queen-attack",
    "space-age",
}
HARD = {
    "bank-account",
    "binary-search-tree",
    "circular-buffer",
    "knapsack",
    "linked-list",
    "parallel-letter-frequency",
    "robot-name",
    "spiral-matrix",
    "zebra-puzzle",
}
VALIDATION_TASKS = {"clock", "circular-buffer", "zebra-puzzle"}
TEST_TASKS = {"space-age", "bank-account", "parallel-letter-frequency"}
BOOST_DATE_TIME_TASKS = {"gigasecond", "meetup"}
STATE_CONCURRENCY_TASKS = {"bank-account", "parallel-letter-frequency", "robot-name"}
PERFORMANCE_INTENSIVE_TASKS = {"knapsack"}


def rubric_category_for(task_id: str) -> str:
    if task_id in STATE_CONCURRENCY_TASKS:
        return "state_concurrency"
    if task_id in PERFORMANCE_INTENSIVE_TASKS:
        return "performance_intensive"
    return "standard"


def difficulty_for(task_id: str) -> str:
    if task_id in EASY:
        return "Easy"
    if task_id in HARD:
        return "Hard"
    return "Medium"


def split_for(task_id: str, mode: str) -> str:
    if mode == "all-train":
        return "train"
    if mode == "all-test":
        return "test"
    if task_id in VALIDATION_TASKS:
        return "validation"
    if task_id in TEST_TASKS:
        return "test"
    return "train"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ingest_exercise_dir(
    exercise_dir: Path,
    output_dir: Path,
    *,
    split_mode: str = "curriculum",
    source_revision: str = "",
) -> Path:
    config_path = exercise_dir / ".meta" / "config.json"
    instructions_path = exercise_dir / ".docs" / "instructions.md"
    if not config_path.is_file() or not instructions_path.is_file():
        raise ValueError(f"not a complete Exercism exercise: {exercise_dir}")

    config = json.loads(read_text(config_path))
    configured_files = config.get("files", {})
    solution_names = [str(name) for name in configured_files.get("solution", [])]
    test_names = [str(name) for name in configured_files.get("test", [])]
    if not solution_names or not test_names:
        raise ValueError(f"missing solution/test metadata: {config_path}")

    instruction_parts = []
    for relative in (
        ".docs/introduction.md",
        ".docs/instructions.md",
        ".docs/instructions.append.md",
    ):
        path = exercise_dir / relative
        if path.is_file():
            instruction_parts.append(read_text(path).strip())

    solution_files = {name: read_text(exercise_dir / name) for name in solution_names}
    test_files = {name: read_text(exercise_dir / name) for name in test_names}
    support_files = {
        path.relative_to(exercise_dir).as_posix(): read_text(path)
        for path in sorted((exercise_dir / "test").rglob("*"))
        if path.is_file()
    }

    oracle_files: dict[str, str] = {}
    for name, stub in solution_files.items():
        example_path = exercise_dir / ".meta" / f"example{Path(name).suffix}"
        oracle_files[name] = read_text(example_path) if example_path.is_file() else stub

    task_id = exercise_dir.name
    task = AiderPolyglotTask(
        task_id=task_id,
        topic_category=CATEGORY_MAPPING.get(task_id, "General"),
        rubric_category=rubric_category_for(task_id),
        difficulty=difficulty_for(task_id),
        instructions_md="\n\n".join(part for part in instruction_parts if part),
        solution_files=solution_files,
        oracle_files=oracle_files,
        test_files=test_files,
        support_files=support_files,
        cmake_lists=read_text(exercise_dir / "CMakeLists.txt"),
        link_flags=["-lboost_date_time"] if task_id in BOOST_DATE_TIME_TASKS else [],
        split=split_for(task_id, split_mode),
        source_revision=source_revision,
    )
    return task.write_json(output_dir / task.split / f"{task_id}.json")


def git_revision(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def ingest_all(practice_dir: Path, output_dir: Path, *, split_mode: str) -> list[Path]:
    if not practice_dir.is_dir():
        raise FileNotFoundError(practice_dir)
    revision = git_revision(practice_dir)
    outputs = []
    for exercise_dir in sorted(path for path in practice_dir.iterdir() if path.is_dir()):
        outputs.append(
            ingest_exercise_dir(
                exercise_dir,
                output_dir,
                split_mode=split_mode,
                source_revision=revision,
            )
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("practice_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--split-mode",
        choices=("curriculum", "all-train", "all-test"),
        default="curriculum",
        help="curriculum creates 20 train, 3 validation, and 3 test tasks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = ingest_all(args.practice_dir, args.output_dir, split_mode=args.split_mode)
    counts: dict[str, int] = {}
    for path in outputs:
        counts[path.parent.name] = counts.get(path.parent.name, 0) + 1
    print(json.dumps({"tasks": len(outputs), "splits": counts, "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
