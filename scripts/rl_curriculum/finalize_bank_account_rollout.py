#!/usr/bin/env python3
"""Convert one Miles no-update dump set and apply the issue #110 gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from glm47_posttraining.integrations.miles_cpp_perf import (  # noqa: E402
    load_miles_debug_samples,
    record_from_debug_sample,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    dumps = sorted((run_root / "rollout_dumps").glob("*.pt"))
    if not dumps:
        raise SystemExit(f"no Miles rollout dumps found under {run_root / 'rollout_dumps'}")

    records: list[dict[str, object]] = []
    for dump in dumps:
        records.extend(
            record_from_debug_sample(sample)
            for sample in load_miles_debug_samples(dump)
        )
    bank_records = [
        record
        for record in records
        if str(record.get("task_id", "")).startswith("aider-shadow-cpp/bank-account-")
    ]
    if not bank_records:
        raise SystemExit("rollout dumps contain no bank-account reward records")

    output = run_root / "bank_account_rollout"
    output.mkdir(parents=True, exist_ok=False)
    records_path = output / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in bank_records),
        encoding="utf-8",
    )
    gate_path = output / "gate.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "rl_curriculum" / "check_bank_account_rollout.py"),
            str(records_path),
            "--out",
            str(gate_path),
            "--expected-groups",
            "40",
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "dumps": [str(path) for path in dumps],
                "records": len(bank_records),
                "records_path": str(records_path),
                "gate_path": str(gate_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
