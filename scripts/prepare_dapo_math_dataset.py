from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


DEFAULT_DATASET = "zhuzilin/dapo-math-17k"
DEFAULT_OUT = Path(".w8-biayn/slime/dapo-math-17k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and materialize DAPO-Math-17k as JSONL for SLIME.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset name.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory.")
    parser.add_argument("--split", default="train", help="Dataset split name.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to keep.")
    return parser.parse_args()


def slice_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return dataset.select(range(min(limit, len(dataset))))


def prepare_dataset(dataset_name: str, split_name: str, limit: int | None) -> Dataset:
    dataset = load_dataset(dataset_name, split=split_name)
    return slice_dataset(dataset, limit)


def write_jsonl(dataset: Dataset, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(path), orient="records", lines=True, force_ascii=False)
    return len(dataset)


def build_summary(
    *,
    dataset_name: str,
    split: str,
    limit: int | None,
    rows: int,
    path: Path,
) -> dict[str, Any]:
    return {
        "dataset": dataset_name,
        "split": split,
        "limit": limit,
        "rows": rows,
        "path": str(path),
        "limits_applied": limit is not None,
    }


def main() -> int:
    args = parse_args()
    out_dir = args.out
    data_path = out_dir / "dapo-math-17k.jsonl"
    summary_path = out_dir / "summary.json"

    dataset = prepare_dataset(args.dataset, args.split, args.limit)
    rows = write_jsonl(dataset, data_path)
    summary = build_summary(
        dataset_name=args.dataset,
        split=args.split,
        limit=args.limit,
        rows=rows,
        path=data_path,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
