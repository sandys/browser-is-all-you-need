#!/usr/bin/env python3
"""Generate and verify the issue #110 bank-account execution curriculum."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from glm47_posttraining.aider_polyglot.bank_account_curriculum import (  # noqa: E402
    build_bank_account_curriculum,
)
from glm47_posttraining.aider_polyglot.dataset import (  # noqa: E402
    build_aider_polyglot_datasets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO / "benchmarks" / "cpp" / "bank-account-equivalent-v1",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "rl-curriculum" / "bank-account-v1",
    )
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--allow-non-gcc", action="store_true")
    args = parser.parse_args()

    output = args.out.resolve()
    rubric_root = output / "rubrics"
    dataset_root = output / "dataset"
    if output.exists():
        if output.is_symlink():
            raise SystemExit(f"refusing to replace symlink output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_manifest = build_bank_account_curriculum(
        args.source,
        rubric_root,
        compiler=args.compiler,
        require_gcc=not args.allow_non_gcc,
    )
    paths = build_aider_polyglot_datasets(
        rubric_root,
        dataset_root,
        monitor_limit=8,
        profile="issue-110-bank-account-v1",
        run_id="cpu-verification",
        sort_by_size=True,
    )
    data_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if data_manifest["counts"] != {
        "available_shadow": 40,
        "monitor": 8,
        "train": 32,
        "validation": 8,
    }:
        raise SystemExit(f"unexpected dataset counts: {data_manifest['counts']}")

    train_rows = sum(1 for line in paths["grpo_train"].read_text().splitlines() if line)
    validation_rows = sum(1 for line in paths["eval"].read_text().splitlines() if line)
    if (train_rows, validation_rows) != (32, 8):
        raise SystemExit(
            f"serialized split count drifted: train={train_rows} validation={validation_rows}"
        )

    serialized = paths["grpo_train"].read_text() + paths["eval"].read_text()
    pinned_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    source_tests: list[str] = []
    source_answers: list[str] = []
    for variant in pinned_manifest["variants"]:
        variant_root = args.source / variant["directory"]
        for name in variant["files"]:
            path = variant_root / name
            if name.endswith("_test.cpp"):
                source_tests.append(path.read_text(encoding="utf-8"))
            elif name.endswith((".h", ".cpp")):
                source_answers.append(path.read_text(encoding="utf-8"))
    leaked_tests = sum(test in serialized for test in source_tests)
    leaked_answers = sum(answer.strip() and answer in serialized for answer in source_answers)
    if leaked_tests or leaked_answers:
        raise SystemExit(
            f"answer-blind packaging failed: tests={leaked_tests} answers={leaked_answers}"
        )

    summary = {
        "status": "passed",
        "compiler": source_manifest["compiler"],
        "compiler_is_gcc": source_manifest["compiler_is_gcc"],
        "tasks": 40,
        "train": 32,
        "validation": 8,
        "reference_answers_in_serialized_prompts": leaked_answers,
        "hidden_tests_in_serialized_prompts": leaked_tests,
        "source_manifest": str(rubric_root / "manifest.json"),
        "data_manifest": str(paths["manifest"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
