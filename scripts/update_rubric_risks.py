#!/usr/bin/env python3
"""Update per-task rubric risks once from one evaluation epoch's records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm47_posttraining.cpp_perf.rubrics import (
    normalize_risks,
    rubric_pass_rates,
    update_risks_ema,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def update_taxonomy_risks(
    taxonomy_file: Path,
    records_file: Path,
    output_file: Path,
    *,
    alpha: float = 0.8,
) -> dict[str, dict[str, object]]:
    taxonomy = json.loads(taxonomy_file.read_text(encoding="utf-8"))
    if not isinstance(taxonomy, dict):
        raise ValueError("taxonomy must be an object keyed by task id")
    records = read_jsonl(records_file)
    by_task: dict[str, list[dict[str, object]]] = {}
    for record in records:
        task_id = str(record.get("task_id") or "")
        if task_id:
            by_task.setdefault(task_id, []).append(record)

    updated: dict[str, dict[str, object]] = {}
    for task_id, raw_record in taxonomy.items():
        record = dict(raw_record)
        category = str(record.get("rubric_category") or "standard")
        task_records = by_task.get(str(task_id), [])
        if task_records:
            pass_rates = rubric_pass_rates(task_records)
            risks = update_risks_ema(
                category,
                record.get("rubric_risks", {}),
                pass_rates,
                alpha=alpha,
            )
            record["rubric_pass_rates"] = pass_rates
            record["rubric_risks"] = {name: round(value, 6) for name, value in risks.items()}
            record["rubric_weights"] = normalize_risks(category, risks)
            record["rubric_risk_alpha"] = alpha
        updated[str(task_id)] = record

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(f".{output_file.name}.tmp")
    temporary.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_file)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("taxonomy_file", type=Path)
    parser.add_argument("records_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument("--alpha", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updated = update_taxonomy_risks(
        args.taxonomy_file,
        args.records_file,
        args.output_file,
        alpha=args.alpha,
    )
    print(f"Updated rubric risks for {len(updated)} tasks in {args.output_file}")


if __name__ == "__main__":
    main()
