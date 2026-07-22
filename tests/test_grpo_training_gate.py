from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.create_grpo_training_gate import create_gate


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_model.bin").write_bytes(b"adapter")
    (path / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    run_root = tmp_path / "runs" / "run"
    source = tmp_path / "runs" / "sft" / "adapter"
    _write_adapter(source)
    data_manifest = run_root / "data" / "manifest.json"
    data_manifest.parent.mkdir(parents=True)
    data_manifest.write_text(
        json.dumps(
            {
                "kind": "aider-polyglot-cpp-shadow-grpo",
                "counts": {"train": 253},
                "source_tree_sha256": "a" * 64,
                "split_contract": {"official_26": "external fixed evaluation only"},
            }
        ),
        encoding="utf-8",
    )
    hybrid = run_root / "adapter_hybrid"
    hybrid.mkdir(parents=True)
    hybrid_manifest = hybrid / "mtp_strip_manifest.json"
    hybrid_manifest.write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "source_adapter_model_sha256": _sha256(source / "adapter_model.bin"),
                "native_files": {f"adapter_megatron_tp{i}_pp0.pt": "x" for i in range(4)},
                "training_state_files": {},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = run_root / "checkpoints" / "grpo_lora_r16" / "iter_0000000" / "adapter"
    _write_adapter(checkpoint)
    for index in range(4):
        (checkpoint / f"adapter_megatron_tp{index}_pp0.pt").write_bytes(b"native")
    for index in range(8):
        (checkpoint / f"training_state_rank{index}.pt").write_bytes(b"state")
    return {
        "run_root": run_root,
        "source": source,
        "source_sha": _sha256(source / "adapter_model.bin"),
        "data_manifest": data_manifest,
        "hybrid_manifest": hybrid_manifest,
        "save_dir": run_root / "checkpoints" / "grpo_lora_r16",
        "checkpoint": checkpoint,
    }


def _fake_torch(monkeypatch) -> None:
    state = {
        **{f"model.layers.47.tensor_{index}": object() for index in range(207)},
        **{f"model.layers.0.tensor_{index}": object() for index in range(9534)},
    }
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: state))


def test_create_gate_binds_data_warm_start_and_checkpoint(tmp_path: Path, monkeypatch) -> None:
    _fake_torch(monkeypatch)
    fixture = _fixture(tmp_path)
    output = fixture["run_root"] / "grpo_lora_r16" / "grpo_training_gate.json"
    gate = create_gate(
        run_id="run",
        run_root=fixture["run_root"],
        save_dir=fixture["save_dir"],
        data_manifest_path=fixture["data_manifest"],
        source_adapter_path=fixture["source"],
        expected_source_adapter_sha256=fixture["source_sha"],
        hybrid_manifest_path=fixture["hybrid_manifest"],
        source_commit="deadbeef",
        phase="profile",
        num_rollout=1,
        gpus_per_node=8,
        expected_native_shards=4,
        output=output,
    )
    assert gate["status"] == "passed"
    assert gate["training_task_count"] == 253
    assert gate["latest_checkpoint"]["adapter_path"] == str(fixture["checkpoint"])
    assert gate["latest_checkpoint"]["tensor_count"] == 9741
    assert output.is_file()


def test_create_gate_rejects_incomplete_training_state(tmp_path: Path, monkeypatch) -> None:
    _fake_torch(monkeypatch)
    fixture = _fixture(tmp_path)
    (fixture["checkpoint"] / "training_state_rank7.pt").unlink()
    with pytest.raises(RuntimeError, match="training-state"):
        create_gate(
            run_id="run",
            run_root=fixture["run_root"],
            save_dir=fixture["save_dir"],
            data_manifest_path=fixture["data_manifest"],
            source_adapter_path=fixture["source"],
            expected_source_adapter_sha256=fixture["source_sha"],
            hybrid_manifest_path=fixture["hybrid_manifest"],
            source_commit="deadbeef",
            phase="profile",
            num_rollout=1,
            gpus_per_node=8,
            expected_native_shards=4,
            output=fixture["run_root"] / "gate.json",
        )
