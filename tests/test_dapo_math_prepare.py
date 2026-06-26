from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from datasets import Dataset

MODULE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prepare_dapo_math_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_dapo_math_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_slice_dataset_limits_rows() -> None:
    dataset = Dataset.from_dict({"prompt": ["a", "b", "c"], "label": ["1", "2", "3"]})

    limited = MODULE.slice_dataset(dataset, 2)
    full = MODULE.slice_dataset(dataset, None)

    assert len(limited) == 2
    assert len(full) == 3


def test_write_jsonl_and_summary(tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"prompt": ["a", "b"], "label": ["1", "2"]})
    data_path = tmp_path / "dapo-math-17k.jsonl"

    rows = MODULE.write_jsonl(dataset, data_path)
    summary = MODULE.build_summary(
        dataset_name="zhuzilin/dapo-math-17k",
        split="train",
        limit=32,
        rows=rows,
        path=data_path,
    )

    assert rows == 2
    assert len(data_path.read_text(encoding="utf-8").splitlines()) == 2
    assert summary["limits_applied"] is True
    assert summary["limit"] == 32

    parsed = json.loads(data_path.read_text(encoding="utf-8").splitlines()[0])
    assert parsed["prompt"] == "a"
