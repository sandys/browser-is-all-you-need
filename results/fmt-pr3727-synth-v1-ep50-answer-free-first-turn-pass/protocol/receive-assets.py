#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

BASE_REPOSITORY = "zai-org/GLM-4.7-Flash"
BASE_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
ADAPTER_REPOSITORY = "TokenBender/glm47-synth-v1-100ep"
ADAPTER_REVISION = "a8d386d6b5bedf761cbe4effe2fdc36a97ad62bc"
ADAPTER_PATTERN = "checkpoints/iter_0000649/adapter/*"


def main() -> None:
    request = json.load(sys.stdin)
    token = request["token"]
    base_destination = Path(request["base_destination"])
    adapter_destination = Path(request["adapter_destination"])
    base_destination.parent.mkdir(parents=True, exist_ok=True)
    adapter_destination.parent.mkdir(parents=True, exist_ok=True)

    base_path = snapshot_download(
        repo_id=BASE_REPOSITORY,
        revision=BASE_REVISION,
        token=token,
        local_dir=base_destination,
    )
    adapter_root = snapshot_download(
        repo_id=ADAPTER_REPOSITORY,
        revision=ADAPTER_REVISION,
        token=token,
        local_dir=adapter_destination,
        allow_patterns=[ADAPTER_PATTERN],
    )
    adapter_path = Path(adapter_root) / "checkpoints/iter_0000649/adapter"
    if not (Path(base_path) / "model.safetensors.index.json").is_file():
        raise RuntimeError("base model download is incomplete")
    if not (adapter_path / "adapter_model.bin").is_file():
        raise RuntimeError("adapter download is incomplete")
    print(
        json.dumps(
            {
                "base_path": str(base_path),
                "adapter_path": str(adapter_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
