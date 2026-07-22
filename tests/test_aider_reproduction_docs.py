from __future__ import annotations

import json
import re
from pathlib import Path


LEDGER = Path("docs/aider_posttraining_runs.json")
FINAL_RECEIPT = Path("docs/receipts/glm47-aider-rl-v2-fixed26-run-receipt.json")
README = Path("README.md")


def test_aider_progress_ledger_matches_final_rl_receipt() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    receipt = json.loads(FINAL_RECEIPT.read_text(encoding="utf-8"))
    rl = next(stage for stage in ledger["stages"] if stage["id"] == "rl-v2")

    assert receipt["status"] == "complete"
    assert rl["adapter_sha256"] == receipt["adapter_sha256"]
    assert rl["training"]["dataset_manifest_sha256"] == receipt[
        "training_data_manifest_sha256"
    ]
    assert rl["pass_at_1"] == receipt["validation"]["pass_at_1"] == 1
    assert rl["pass_at_2"] == receipt["validation"]["pass_at_k"] == 6
    assert rl["well_formed_tasks"] == receipt["validation"]["well_formed_tasks"] == 26


def test_aider_progress_ledger_has_the_promoted_lineage_only() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert [stage["id"] for stage in ledger["stages"]] == [
        "base",
        "sft-v1",
        "sft-v2",
        "sft-v3",
        "rl-v2",
    ]


def test_readme_local_links_exist() -> None:
    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" in target:
            continue
        path = Path(target.split("#", 1)[0])
        assert path.exists(), target
