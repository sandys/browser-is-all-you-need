"""Build SLIME-ready JSONL datasets from validated C++ task JSON."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from .data import write_data_manifest
from .schema import CppTask
from .skyrl_dataset import DATA_SOURCE, build_prompt


def task_to_slime_row(task: CppTask, *, task_path: str) -> dict[str, Any]:
    """Convert one task into SLIME's prompt/label/metadata JSONL format."""

    return {
        "prompt": build_prompt(task),
        "label": task.task_id,
        "metadata": {
            "task_id": task.task_id,
            "problem_id": task.problem_id,
            "split": task.split,
            "task_path": task_path,
            "data_source": DATA_SOURCE,
            "reference_metric": task.reference.metric,
            "reference_value": task.reference.value,
            "gem5_cycles": task.reference.gem5_cycles,
            "hidden_test_count": len(task.hidden_tests),
            "source": task.source,
        },
    }


def load_tasks(tasks_dir: str | Path) -> list[tuple[Path, CppTask]]:
    """Load all task JSON files from a directory tree."""

    root = Path(tasks_dir)
    tasks: list[tuple[Path, CppTask]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_w8_"):
            continue
        tasks.append((path, CppTask.read_json(path)))
    return tasks


def build_slime_datasets(
    tasks_dir: str | Path,
    output_dir: str | Path,
    *,
    validation_splits: Iterable[str] = ("validation", "test"),
    profile: str = "smoke",
    run_id: str | None = None,
    min_train_tasks: int = 1,
    min_validation_tasks: int = 1,
    absolute_task_paths: bool = False,
) -> dict[str, Path]:
    """Write SLIME JSONL prompt datasets from validated tasks."""

    source_root = Path(tasks_dir)
    loaded = load_tasks(source_root)
    if not loaded:
        raise ValueError(f"No task JSON files found under {tasks_dir}")

    output = Path(output_dir)
    tasks_out = output / "tasks"
    tasks_out.mkdir(parents=True, exist_ok=True)

    copied: dict[Path, str] = {}
    for source_path, _task in loaded:
        rel_source = source_path.relative_to(source_root)
        destination = tasks_out / rel_source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        if absolute_task_paths:
            copied[source_path] = str(destination.resolve())
        else:
            copied[source_path] = destination.relative_to(output).as_posix()

    validation_names = set(validation_splits)
    train_rows = [
        task_to_slime_row(task, task_path=copied[path])
        for path, task in loaded
        if task.split == "train"
    ]
    validation_rows = [
        task_to_slime_row(task, task_path=copied[path])
        for path, task in loaded
        if task.split in validation_names
    ]
    test_rows = [
        task_to_slime_row(task, task_path=copied[path])
        for path, task in loaded
        if task.split == "test"
    ]

    if len(train_rows) < min_train_tasks:
        raise ValueError(
            f"SLIME dataset build requires at least {min_train_tasks} train tasks; "
            f"found {len(train_rows)}"
        )
    if len(validation_rows) < min_validation_tasks:
        raise ValueError(
            f"SLIME dataset build requires at least {min_validation_tasks} validation tasks "
            f"in {sorted(validation_names)}; found {len(validation_rows)}"
        )

    train_path = output / "train.jsonl"
    validation_path = output / "validation.jsonl"
    test_path = output / "test.jsonl"
    _write_jsonl(train_rows, train_path)
    _write_jsonl(validation_rows, validation_path)
    _write_jsonl(test_rows, test_path)

    counts = {
        "train": len(train_rows),
        "validation": len([task for _path, task in loaded if task.split == "validation"]),
        "test": len(test_rows),
        "validation_effective": len(validation_rows),
    }
    manifest = write_data_manifest(
        output,
        kind="slime-dataset",
        sources=[str(tasks_dir)],
        options={
            "validation_splits": sorted(validation_names),
            "profile": profile,
            "run_id": run_id,
            "absolute_task_paths": absolute_task_paths,
            "counts": counts,
        },
    )
    return {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
        "manifest": manifest,
    }


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
