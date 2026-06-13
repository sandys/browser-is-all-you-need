"""Build SkyRL-ready datasets from validated C++ task JSON."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .data import write_data_manifest
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


def task_to_grpo_row(task: CppTask, *, task_path: str) -> dict[str, Any]:
    """Convert one task into SkyRL's PPO/GRPO prompt row format."""

    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": build_prompt(task)}],
        "env_class": CPP_PERF_ENV_ID,
        "reward_spec": {"method": "rule", "ground_truth": task.task_id},
        "extra_info": {
            "task_id": task.task_id,
            "problem_id": task.problem_id,
            "split": task.split,
            "task_path": task_path,
            "reference_metric": task.reference.metric,
            "reference_value": task.reference.value,
            "gem5_cycles": task.reference.gem5_cycles,
            "visible_tests": [case.model_dump() for case in task.unit_tests],
            "hidden_test_count": len(task.hidden_tests),
            "source": task.source,
        },
    }


def task_to_sft_row(task: CppTask) -> dict[str, Any]:
    """Convert one task into SkyRL SFT's Alpaca-style row format."""

    return {
        "instruction": build_prompt(task),
        "output": sft_output(task),
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "split": task.split,
        "data_source": DATA_SOURCE,
    }


def load_tasks(tasks_dir: str | Path) -> list[tuple[Path, CppTask]]:
    """Load all task JSON files from a directory tree."""

    root = Path(tasks_dir)
    tasks: list[tuple[Path, CppTask]] = []
    for path in sorted(root.rglob("*.json")):
        tasks.append((path, CppTask.read_json(path)))
    return tasks


def build_skyrl_datasets(
    tasks_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_splits: Iterable[str] = ("validation", "test"),
) -> dict[str, Path]:
    """Write GRPO parquet and SFT JSONL datasets from validated tasks."""

    loaded = load_tasks(tasks_dir)
    if not loaded:
        raise ValueError(f"No task JSON files found under {tasks_dir}")

    output = Path(output_dir)
    tasks_out = output / "tasks"
    grpo_out = output / "grpo"
    sft_out = output / "sft"
    for directory in (tasks_out, grpo_out, sft_out):
        directory.mkdir(parents=True, exist_ok=True)

    copied: dict[Path, str] = {}
    for source_path, task in loaded:
        rel_source = source_path.relative_to(Path(tasks_dir))
        destination = tasks_out / rel_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied[source_path] = destination.relative_to(output).as_posix()

    validation_names = set(validation_splits)
    train_tasks = [task for _path, task in loaded if task.split == "train"]
    validation_tasks = [task for _path, task in loaded if task.split in validation_names]
    if not train_tasks:
        raise ValueError("SkyRL dataset build requires at least one train task")
    if not validation_tasks:
        raise ValueError(f"SkyRL dataset build requires at least one validation task in {sorted(validation_names)}")

    train_grpo = [
        task_to_grpo_row(task, task_path=copied[path])
        for path, task in loaded
        if task.split == "train"
    ]
    val_grpo = [
        task_to_grpo_row(task, task_path=copied[path])
        for path, task in loaded
        if task.split in validation_names
    ]
    train_sft = [task_to_sft_row(task) for task in train_tasks]
    val_sft = [task_to_sft_row(task) for task in validation_tasks]

    train_grpo_path = grpo_out / "train.parquet"
    val_grpo_path = grpo_out / "validation.parquet"
    _write_parquet(train_grpo, train_grpo_path)
    _write_parquet(val_grpo, val_grpo_path)

    train_sft_path = sft_out / "train.jsonl"
    val_sft_path = sft_out / "validation.jsonl"
    _write_jsonl(train_sft, train_sft_path)
    _write_jsonl(val_sft, val_sft_path)

    manifest = write_data_manifest(
        output,
        kind="skyrl-dataset",
        sources=[str(tasks_dir)],
        options={"validation_splits": sorted(validation_names)},
    )
    return {
        "grpo_train": train_grpo_path,
        "grpo_validation": val_grpo_path,
        "sft_train": train_sft_path,
        "sft_validation": val_sft_path,
        "manifest": manifest,
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
