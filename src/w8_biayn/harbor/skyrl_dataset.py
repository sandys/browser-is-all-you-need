"""Prepare packaged Harbor DOMDiff tasks for SkyRL-Gym."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tasks import HarborTask, selected_task_ids, validate_tasks

HARBOR_SYSTEM_PROMPT = """You are fixing a UI bug in a checked-out frontend repository.

Return exactly one shell script that can run from /testbed inside the task container.
Wrap the script in <solution>...</solution>.
Do not include explanations outside the solution block.
"""


def make_harbor_record(task: HarborTask, split: str, index: int, *, oracle: bool = False) -> dict[str, Any]:
    return {
        "prompt": [
            {"role": "system", "content": HARBOR_SYSTEM_PROMPT},
            {"role": "user", "content": task.instruction},
        ],
        "env_class": "harbor-domdiff",
        "task_id": task.task_id,
        "task_path": str(task.path),
        "preview_path": task.preview_path,
        "split": split,
        "index": index,
        "oracle": bool(oracle),
    }


def prepare_harbor_skyrl_dataset(
    output_dir: str | Path,
    *,
    task_root: str | Path | None = None,
    task_ids: tuple[str, ...] | list[str] | None = None,
    oracle: bool = False,
) -> tuple[Path, Path]:
    """Write train/validation parquet files for SkyRL's PromptDataset."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional runtime extras.
        raise RuntimeError("Harbor SkyRL dataset preparation requires pandas and pyarrow") from exc

    ids = selected_task_ids(task_root, task_ids)
    tasks = validate_tasks(task_root=task_root, task_ids=list(ids))
    if not tasks:
        raise RuntimeError("No Harbor tasks selected")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "train.parquet"
    val_path = output_path / "validation.parquet"
    metadata_path = output_path / "metadata.json"

    train_rows = [make_harbor_record(task, "train", index, oracle=oracle) for index, task in enumerate(tasks)]
    val_rows = [make_harbor_record(task, "validation", index, oracle=oracle) for index, task in enumerate(tasks)]
    pd.DataFrame(train_rows).to_parquet(train_path)
    pd.DataFrame(val_rows).to_parquet(val_path)
    metadata_path.write_text(
        json.dumps(
            {
                "benchmark": "harbor-domdiff-browser-swe",
                "train_count": len(train_rows),
                "validation_count": len(val_rows),
                "task_ids": [task.task_id for task in tasks],
                "oracle": bool(oracle),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return train_path, val_path
