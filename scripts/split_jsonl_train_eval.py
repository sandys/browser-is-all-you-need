#!/usr/bin/env python3
"""Deterministically split a JSONL file into train and held-out eval files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def split_jsonl(input_path: Path, train_out: Path, eval_out: Path, eval_size: int, seed: int) -> tuple[int, int]:
    rows = [line for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if eval_size <= 0:
        raise ValueError("eval_size must be positive")
    if eval_size >= len(rows):
        raise ValueError(f"eval_size={eval_size} must be smaller than row count={len(rows)}")
    rng = random.Random(seed)
    rng.shuffle(rows)
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]
    train_out.parent.mkdir(parents=True, exist_ok=True)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    train_out.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    eval_out.write_text("\n".join(eval_rows) + "\n", encoding="utf-8")
    return len(train_rows), len(eval_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path.")
    parser.add_argument("--train-out", type=Path, required=True, help="Output train JSONL path.")
    parser.add_argument("--eval-out", type=Path, required=True, help="Output held-out eval JSONL path.")
    parser.add_argument("--eval-size", type=int, default=500, help="Number of rows to reserve for eval.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count, eval_count = split_jsonl(args.input, args.train_out, args.eval_out, args.eval_size, args.seed)
    print(f"train_rows={train_count}")
    print(f"eval_rows={eval_count}")


if __name__ == "__main__":
    main()
