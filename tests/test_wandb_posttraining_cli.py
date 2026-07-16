from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path("scripts/publish_results.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("wandb_posttraining_cli", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_checkpoint_label_handles_numbered_source_labels() -> None:
    module = _load_module()

    assert module._canonical_checkpoint_label("base_h100_spec") == "base"
    assert module._canonical_checkpoint_label("sft1_h100_spec") == "sft"
    assert module._canonical_checkpoint_label("grpo-iter20") == "grpo"
    assert module._canonical_checkpoint_label("baseline") == "baseline"
