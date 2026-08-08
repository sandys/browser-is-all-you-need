#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import torch

EXPECTED_SOURCE_SHA256 = "4acb7f23c295f45380155c5d9ee6bc59422262f0cb51f0c02f7e550d405b575a"
EXPECTED_DATA_MANIFEST_SHA256 = "afec0d05d5c1f9460ac9b2ed4e65aea69775b49639c7ae727f3660308ae1b8b8"
EXPECTED_SOURCE_TENSORS = 9741
EXPECTED_LAYER_47_TENSORS = 207
EXPECTED_SERVING_TENSORS = 9534


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    source_model = args.source / "adapter_model.bin"
    source_config = args.source / "adapter_config.json"
    if not source_model.is_file() or not source_config.is_file():
        raise SystemExit("source adapter is incomplete")
    source_sha = sha256(source_model)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"source adapter SHA-256 mismatch: {source_sha}")
    config = json.loads(source_config.read_text(encoding="utf-8"))
    if config.get("r") != 16:
        raise SystemExit(f"unexpected LoRA rank: {config.get('r')!r}")
    if args.destination.exists():
        raise SystemExit(f"refusing to overwrite existing destination: {args.destination}")

    state = torch.load(source_model, map_location="cpu", weights_only=True, mmap=True)
    layer_47 = [name for name in state if ".layers.47." in name]
    if len(state) != EXPECTED_SOURCE_TENSORS or len(layer_47) != EXPECTED_LAYER_47_TENSORS:
        raise SystemExit(
            f"unexpected adapter tensor domain: total={len(state)} layer47={len(layer_47)}"
        )
    filtered = {name: tensor for name, tensor in state.items() if ".layers.47." not in name}
    if len(filtered) != EXPECTED_SERVING_TENSORS:
        raise SystemExit(f"unexpected serving tensor count: {len(filtered)}")

    temporary = args.destination.with_name(
        f"{args.destination.name}-preparing-{uuid.uuid4().hex[:8]}"
    )
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        torch.save(filtered, temporary / "adapter_model.bin")
        shutil.copy2(source_config, temporary / "adapter_config.json")
        receipt = {
            "schema_version": 1,
            "kind": "glm47-serving-adapter-conversion",
            "checkpoint": "Synth v1 epoch 50",
            "iteration": 649,
            "lora_rank": 16,
            "adapter_revision": "a8d386d6b5bedf761cbe4effe2fdc36a97ad62bc",
            "source_adapter_sha256": source_sha,
            "source_adapter_config_sha256": sha256(source_config),
            "source_tensor_count": len(state),
            "removed_layer_47_tensor_count": len(layer_47),
            "serving_tensor_count": len(filtered),
            "serving_adapter_sha256": sha256(temporary / "adapter_model.bin"),
            "serving_adapter_config_sha256": sha256(temporary / "adapter_config.json"),
            "training_data_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        }
        (temporary / "conversion_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
