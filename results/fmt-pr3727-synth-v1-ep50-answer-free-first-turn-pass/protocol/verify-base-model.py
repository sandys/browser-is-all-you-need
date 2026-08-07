#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPOSITORY = "zai-org/GLM-4.7-Flash"
REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    config_path = args.model / "config.json"
    index_path = args.model / "model.safetensors.index.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted({str(name) for name in index["weight_map"].values()})
    if config.get("model_type") != "glm4_moe_lite":
        raise SystemExit(f"unexpected model_type: {config.get('model_type')!r}")
    if len(shards) != 48:
        raise SystemExit(f"unexpected model shard count: {len(shards)}")
    missing = [name for name in shards if not (args.model / name).is_file()]
    if missing:
        raise SystemExit(f"missing model shards: {missing}")
    receipt = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "revision": REVISION,
        "model_type": config["model_type"],
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
        "shard_count": len(shards),
        "shard_bytes": {name: (args.model / name).stat().st_size for name in shards},
        "total_shard_bytes": sum((args.model / name).stat().st_size for name in shards),
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "shard_bytes"}, indent=2))


if __name__ == "__main__":
    main()
