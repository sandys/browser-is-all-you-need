#!/usr/bin/env python3
"""Download the canonical dataset and adapters from Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


ASSETS = {
    "model": {
        "repo_id": "zai-org/GLM-4.7-Flash",
        "repo_type": "model",
        "revision_env": "GLM47_MODEL_REVISION",
        "default_revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        "destination": "GLM-4.7-Flash",
        "verify_checksums": False,
    },
    "data": {
        "repo_id": "TokenBender/glm47-pie-cpp-posttraining-data",
        "repo_type": "dataset",
        "revision_env": "GLM47_DATA_REVISION",
        "default_revision": "09bc0276a0ff8ab84a8db81880ca7f739057e654",
        "destination": "data",
        "verify_checksums": True,
    },
    "sft": {
        "repo_id": "TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100",
        "repo_type": "model",
        "revision_env": "GLM47_SFT_REVISION",
        "default_revision": "f1ac8df367080cc040f7cf769db219ee58f20f63",
        "destination": "adapters/sft",
        "verify_checksums": True,
    },
    "grpo": {
        "repo_id": "TokenBender/glm47-flash-pie-cpp-lora-r16-grpo-h100",
        "repo_type": "model",
        "revision_env": "GLM47_GRPO_REVISION",
        "default_revision": "1fbac6f6fd59829a64776937102351c6318a7fd4",
        "destination": "adapters/grpo",
        "verify_checksums": True,
    },
}

DEFAULT_ASSETS = ("data", "sft", "grpo")


def _verify_checksums(root: Path) -> None:
    checksum_file = root / "SHA256SUMS"
    if not checksum_file.is_file():
        raise FileNotFoundError(f"Missing checksum manifest: {checksum_file}")
    for line in checksum_file.read_text().splitlines():
        expected, relative_path = line.split(maxsplit=1)
        relative_path = relative_path.removeprefix("*").removeprefix("./")
        path = root / relative_path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")


def _extract_task_archive(root: Path) -> Path:
    archive = root / "tasks.tar.gz"
    destination = root / "tasks"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing task archive: {archive}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    destination_root = destination.resolve()

    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"Task archive contains a link: {member.name}")
            target = (destination / member.name).resolve()
            if target != destination_root and destination_root not in target.parents:
                raise RuntimeError(f"Task archive escapes destination: {member.name}")
        handle.extractall(destination)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = int(manifest["counts"]["copied_tasks"])
    actual = sum(1 for path in destination.rglob("*.json") if path.is_file())
    if actual != expected:
        raise RuntimeError(f"Extracted task count mismatch: {actual} != {expected}")
    return destination


def _download(name: str, output_root: Path, verify: bool) -> Path:
    asset = ASSETS[name]
    destination = output_root / asset["destination"]
    revision = os.environ.get(asset["revision_env"], asset["default_revision"])
    snapshot_download(
        repo_id=asset["repo_id"],
        repo_type=asset["repo_type"],
        revision=revision,
        local_dir=destination,
    )
    if verify and asset["verify_checksums"]:
        _verify_checksums(destination)
    if name == "data":
        _extract_task_archive(destination)
    print(f"{name}: {destination}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", choices=[*ASSETS, "all"], nargs="?", default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".glm47-posttraining/assets"),
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    names = list(DEFAULT_ASSETS) if args.asset == "all" else [args.asset]
    for name in names:
        _download(name, args.output_root, verify=not args.no_verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
