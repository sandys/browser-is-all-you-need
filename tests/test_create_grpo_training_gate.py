#!/usr/bin/env python3
"""Training-receipt schema tests (issue #110 T3).

Run: python3 tests/test_create_grpo_training_gate.py

Reproduces the r3 failure shape — warm start from an EP8 hybrid adapter
(8 native files) with a TP4 output checkpoint (4 shards) under the
``one-update`` phase — and proves the fixed gate accepts it while the old
single-shard-expectation behavior still fails it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "create_grpo_training_gate", ROOT / "scripts/create_grpo_training_gate.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

OUTPUT_SHARDS = 4  # TP4 trained checkpoint
SOURCE_SHARDS = 8  # EP8 warm-start hybrid adapter
GPUS = 8


def _write_adapter_model(path: pathlib.Path) -> None:
    state = {}
    for index in range(MODULE.EXPECTED_SOURCE_TENSORS - MODULE.EXPECTED_LAYER_47_TENSORS):
        state[f"decoder.layers.0.weight_{index}"] = torch.zeros(1)
    for index in range(MODULE.EXPECTED_LAYER_47_TENSORS):
        state[f"decoder.layers.47.weight_{index}"] = torch.zeros(1)
    torch.save(state, path)


def _build_fixture(root: pathlib.Path) -> dict[str, pathlib.Path]:
    source_adapter = root / "source" / "adapter"
    source_adapter.mkdir(parents=True)
    _write_adapter_model(source_adapter / "adapter_model.bin")
    (source_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    hybrid_manifest = root / "hybrid" / "mtp_strip_manifest.json"
    hybrid_manifest.parent.mkdir(parents=True)
    hybrid_manifest.write_text(
        json.dumps(
            {
                "source": str(source_adapter.resolve()),
                "source_adapter_model_sha256": MODULE.sha256_path(
                    source_adapter / "adapter_model.bin"
                ),
                "native_files": {
                    f"adapter_megatron_native_{index}.pt": "0" * 64
                    for index in range(SOURCE_SHARDS)
                },
                "training_state_files": {},
            }
        ),
        encoding="utf-8",
    )

    save_adapter = root / "save" / "iter_0000000" / "adapter"
    save_adapter.mkdir(parents=True)
    _write_adapter_model(save_adapter / "adapter_model.bin")
    (save_adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    for index in range(OUTPUT_SHARDS):
        torch.save({}, save_adapter / f"adapter_megatron_tp{index}_pp0.pt")
    for index in range(GPUS):
        torch.save({}, save_adapter / f"training_state_rank{index}.pt")

    data_manifest = root / "data" / "manifest.json"
    data_manifest.parent.mkdir(parents=True)
    data_manifest.write_text(
        json.dumps(
            {
                "kind": "aider-polyglot-cpp-shadow-grpo",
                "counts": {"train": 32},
                "source_tree_sha256": "f" * 64,
                "split_contract": {"official_26": "benchmark_and_evaluation_only"},
            }
        ),
        encoding="utf-8",
    )
    return {
        "source_adapter": source_adapter,
        "hybrid_manifest": hybrid_manifest,
        "save_dir": root / "save",
        "data_manifest": data_manifest,
        "output": root / "gate.json",
    }


def _create_gate(paths: dict[str, pathlib.Path], **overrides):
    kwargs = dict(
        run_id="test-run",
        run_root=paths["save_dir"].parent,
        save_dir=paths["save_dir"],
        data_manifest_path=paths["data_manifest"],
        source_adapter_path=paths["source_adapter"],
        expected_source_adapter_sha256=MODULE.sha256_path(
            paths["source_adapter"] / "adapter_model.bin"
        ),
        hybrid_manifest_path=paths["hybrid_manifest"],
        source_commit="deadbeef",
        phase="one-update",
        num_rollout=1,
        gpus_per_node=GPUS,
        expected_native_shards=OUTPUT_SHARDS,
        expected_source_native_shards=SOURCE_SHARDS,
        expected_train_count=32,
        output=paths["output"],
    )
    kwargs.update(overrides)
    return MODULE.create_gate(**kwargs)


def test_one_update_phase_with_mismatched_shards_passes() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        paths = _build_fixture(pathlib.Path(scratch))
        gate = _create_gate(paths)
        assert gate["status"] == "passed"
        assert gate["phase"] == "one-update"
        assert gate["expected_native_shards"] == OUTPUT_SHARDS
        assert gate["expected_source_native_shards"] == SOURCE_SHARDS
        assert len(gate["latest_checkpoint"]["native_shards"]) == OUTPUT_SHARDS
        assert paths["output"].is_file()


def test_shared_shard_expectation_still_fails_r3_shape() -> None:
    # Without the split expectation the r3 shape must fail exactly as it did.
    with tempfile.TemporaryDirectory() as scratch:
        paths = _build_fixture(pathlib.Path(scratch))
        try:
            _create_gate(paths, expected_source_native_shards=None)
        except RuntimeError as error:
            assert "hybrid adapter" in str(error)
        else:
            raise AssertionError("EP8 hybrid passed against a TP4-only expectation")


def test_cli_accepts_one_update_phase() -> None:
    parser_error = None
    argv_backup = sys.argv
    try:
        sys.argv = [
            "create_grpo_training_gate.py",
            "--run-id", "x", "--run-root", "/tmp", "--save-dir", "/tmp",
            "--data-manifest", "/tmp/none.json", "--source-adapter", "/tmp",
            "--expected-source-adapter-sha256", "0" * 64,
            "--hybrid-manifest", "/tmp/none.json", "--source-commit", "x",
            "--phase", "one-update", "--num-rollout", "1",
            "--gpus-per-node", "8", "--expected-native-shards", "4",
            "--expected-source-native-shards", "8", "--output", "/tmp/out.json",
        ]
        try:
            MODULE.main()
        except SystemExit as exc:  # argparse rejection would exit 2
            parser_error = exc.code
        except FileNotFoundError:
            parser_error = None  # parsed fine; fixture paths do not exist
    finally:
        sys.argv = argv_backup
    assert parser_error != 2, "argparse rejected --phase one-update"


def test_sha256_helper_stable() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        target = pathlib.Path(scratch) / "blob"
        target.write_bytes(b"receipt")
        assert MODULE.sha256_path(target) == hashlib.sha256(b"receipt").hexdigest()


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok {test.__name__}")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    main()
