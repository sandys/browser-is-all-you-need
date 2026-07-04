"""Prompt and task-loading helpers for C++ performance-RL datasets."""

from __future__ import annotations

from pathlib import Path

from .schema import CppTask


CPP_PERF_ENV_ID = "cpp-perf"
DATA_SOURCE = "pie-cpp-perf"


def build_prompt(task: CppTask) -> str:
    """Return the C++ optimization instruction shown to the model."""

    visible_tests = "\n\n".join(
        f"Input:\n{case.input}\nExpected output:\n{case.expected}" for case in task.unit_tests
    )
    return (
        "Optimize the following C++20 program while preserving its exact behavior.\n"
        "Return exactly one <reasoning>...</reasoning> block followed by exactly one fenced cpp code block.\n"
        "Do not mention or rely on hidden tests. The program must read stdin and write stdout.\n\n"
        "Visible tests:\n"
        f"{visible_tests}\n\n"
        "Program:\n"
        f"```cpp\n{task.prompt_code.rstrip()}\n```"
    )


def sft_output(task: CppTask) -> str:
    """Return the supervised target using the PIE fast solution."""

    return (
        "<reasoning>The optimized program preserves the tested behavior while reducing work.</reasoning>\n"
        f"```cpp\n{task.oracle_solution.rstrip()}\n```"
    )


def load_tasks(tasks_dir: str | Path) -> list[tuple[Path, CppTask]]:
    """Load all task JSON files from a directory tree."""

    root = Path(tasks_dir)
    tasks: list[tuple[Path, CppTask]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_w8_"):
            continue
        tasks.append((path, CppTask.read_json(path)))
    return tasks
