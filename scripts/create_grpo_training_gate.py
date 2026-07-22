"""Create a provenance gate for a completed Aider GRPO checkpoint set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ITERATION_RE = re.compile(r"^iter_(\d+)$")
EXPECTED_ADAPTER_FILES = ("adapter_model.bin", "adapter_config.json")
EXPECTED_SOURCE_TENSORS = 9_741
EXPECTED_LAYER_47_TENSORS = 207


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_record(
    iteration_dir: Path,
    *,
    expected_native_shards: int,
    expected_training_states: int,
) -> dict[str, Any]:
    import torch

    match = ITERATION_RE.fullmatch(iteration_dir.name)
    if match is None:
        raise ValueError(f"invalid checkpoint iteration name: {iteration_dir.name}")
    adapter = iteration_dir / "adapter"
    for name in EXPECTED_ADAPTER_FILES:
        if not (adapter / name).is_file():
            raise FileNotFoundError(f"incomplete GRPO adapter: {adapter / name}")
    native = sorted(adapter.glob("adapter_megatron_tp*_pp*.pt"))
    expected_names = {
        f"adapter_megatron_tp{index}_pp0.pt" for index in range(expected_native_shards)
    }
    if {path.name for path in native} != expected_names:
        raise RuntimeError(f"GRPO native shard set is incomplete: {adapter}")
    training_state = sorted(adapter.glob("training_state_rank*.pt"))
    expected_training_names = {
        f"training_state_rank{index}.pt" for index in range(expected_training_states)
    }
    if {path.name for path in training_state} != expected_training_names:
        raise RuntimeError(f"GRPO training-state set is incomplete: {adapter}")
    state = torch.load(
        adapter / "adapter_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    layer_47 = [key for key in state if ".layers.47." in key]
    if len(state) != EXPECTED_SOURCE_TENSORS or len(layer_47) != EXPECTED_LAYER_47_TENSORS:
        raise RuntimeError(f"GRPO adapter tensor domain is invalid: {adapter}")
    return {
        "iteration": int(match.group(1)),
        "adapter_path": str(adapter),
        "adapter_model_sha256": sha256_path(adapter / "adapter_model.bin"),
        "adapter_config_sha256": sha256_path(adapter / "adapter_config.json"),
        "tensor_count": len(state),
        "layer_47_tensor_count": len(layer_47),
        "native_shards": {path.name: sha256_path(path) for path in native},
        "training_state_files": sorted(path.name for path in training_state),
    }


def create_gate(
    *,
    run_id: str,
    run_root: Path,
    save_dir: Path,
    data_manifest_path: Path,
    source_adapter_path: Path,
    expected_source_adapter_sha256: str,
    hybrid_manifest_path: Path,
    source_commit: str,
    phase: str,
    num_rollout: int,
    gpus_per_node: int,
    expected_native_shards: int,
    expected_train_count: int = 253,
    output: Path,
) -> dict[str, Any]:
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if data_manifest.get("kind") != "aider-polyglot-cpp-shadow-grpo":
        raise RuntimeError("training data is not the Aider shadow GRPO corpus")
    if data_manifest.get("counts", {}).get("train") != expected_train_count:
        raise RuntimeError(
            f"Aider GRPO gate requires {expected_train_count} train tasks, "
            f"manifest has {data_manifest.get('counts', {}).get('train')}"
        )
    source_sha256 = sha256_path(source_adapter_path / "adapter_model.bin")
    if source_sha256 != expected_source_adapter_sha256:
        raise RuntimeError("warm-start adapter bytes do not match the bound SHA-256")
    hybrid_manifest = json.loads(hybrid_manifest_path.read_text(encoding="utf-8"))
    if (
        hybrid_manifest.get("source_adapter_model_sha256") != source_sha256
        or hybrid_manifest.get("source") != str(source_adapter_path.resolve())
        or len(hybrid_manifest.get("native_files", {})) != expected_native_shards
        or hybrid_manifest.get("training_state_files") != {}
    ):
        raise RuntimeError("hybrid adapter is not bound to the selected SFT warm start")

    iteration_dirs = sorted(
        (path for path in save_dir.glob("iter_*") if path.is_dir()),
        key=lambda path: int(ITERATION_RE.fullmatch(path.name).group(1))
        if ITERATION_RE.fullmatch(path.name)
        else -1,
    )
    checkpoints = [
        _checkpoint_record(
            path,
            expected_native_shards=expected_native_shards,
            expected_training_states=gpus_per_node,
        )
        for path in iteration_dirs
    ]
    if not checkpoints:
        raise RuntimeError(f"no complete GRPO checkpoints found beneath {save_dir}")
    if checkpoints[-1]["iteration"] + 1 < num_rollout:
        raise RuntimeError("checkpoint iterations do not cover the configured rollout count")

    gate = {
        "schema_version": 1,
        "kind": "glm47-aider-grpo-training-gate",
        "status": "passed",
        "phase": phase,
        "run_id": run_id,
        "run_root": str(run_root),
        "source_commit": source_commit,
        "gpus_per_node": gpus_per_node,
        "num_rollout": num_rollout,
        "data_manifest_path": str(data_manifest_path),
        "data_manifest_sha256": sha256_path(data_manifest_path),
        "data_source_tree_sha256": data_manifest["source_tree_sha256"],
        "training_task_count": data_manifest["counts"]["train"],
        "official_26_role": data_manifest["split_contract"]["official_26"],
        "source_adapter_path": str(source_adapter_path),
        "source_adapter_sha256": source_sha256,
        "hybrid_manifest_path": str(hybrid_manifest_path),
        "hybrid_manifest_sha256": sha256_path(hybrid_manifest_path),
        "checkpoints": checkpoints,
        "latest_checkpoint": checkpoints[-1],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--expected-source-adapter-sha256", required=True)
    parser.add_argument("--hybrid-manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=("profile", "full"), required=True)
    parser.add_argument("--num-rollout", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--expected-native-shards", type=int, required=True)
    parser.add_argument("--expected-train-count", type=int, default=253)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = create_gate(
        run_id=args.run_id,
        run_root=args.run_root,
        save_dir=args.save_dir,
        data_manifest_path=args.data_manifest,
        source_adapter_path=args.source_adapter,
        expected_source_adapter_sha256=args.expected_source_adapter_sha256,
        hybrid_manifest_path=args.hybrid_manifest,
        source_commit=args.source_commit,
        phase=args.phase,
        num_rollout=args.num_rollout,
        gpus_per_node=args.gpus_per_node,
        expected_native_shards=args.expected_native_shards,
        expected_train_count=args.expected_train_count,
        output=args.output,
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
