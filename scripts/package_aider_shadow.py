#!/usr/bin/env python3
"""Build a deterministic, checksum-bound archive of the Aider shadow corpus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

from glm47_posttraining.aider_polyglot.dataset import (
    EXPECTED_SHADOW_TASKS,
    SOURCE_MANIFEST_KIND,
    _load_verified_rubric,
    _source_tree_sha256,
    _validate_source_manifest,
    discover_shadow_exercises,
)


ARCHIVE_NAME = "aider-shadow-rubrics.tar.gz"
ARCHIVE_ROOT = "aider_polyglot_cpp_shadow"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
    else:
        info.size = path.stat().st_size
        info.mode = 0o644
    return info


def create_deterministic_archive(source: Path, archive: Path) -> None:
    source = source.resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"source must be a real directory: {source}")
    paths = [source, *sorted(source.rglob("*"), key=lambda path: path.as_posix())]
    for path in paths:
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError(f"unsupported corpus entry: {path}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in paths:
                    relative = path.relative_to(source)
                    arcname = ARCHIVE_ROOT if not relative.parts else (
                        f"{ARCHIVE_ROOT}/{relative.as_posix()}"
                    )
                    info = _tar_info(path, arcname)
                    if path.is_file():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    else:
                        tar.addfile(info)


def package(source: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    manifest_path, source_manifest = _validate_source_manifest(source)
    exercises = discover_shadow_exercises(source)
    for exercise in exercises:
        _load_verified_rubric(exercise)

    output.mkdir(parents=True, exist_ok=True)
    archive = output / ARCHIVE_NAME
    create_deterministic_archive(source, archive)
    source_manifest_target = output / "source_manifest.json"
    source_manifest_target.write_bytes(manifest_path.read_bytes())

    artifact_manifest = {
        "kind": "glm47-aider-shadow-rubrics-archive",
        "schema_version": 1,
        "archive": ARCHIVE_NAME,
        "archive_root": ARCHIVE_ROOT,
        "archive_sha256": sha256_path(archive),
        "source_manifest_kind": SOURCE_MANIFEST_KIND,
        "source_manifest_sha256": sha256_path(source_manifest_target),
        "source_tree_sha256": _source_tree_sha256(exercises),
        "counts": {
            "tasks": len(exercises),
            "files": sum(1 for path in source.rglob("*") if path.is_file()),
        },
        "contract": source_manifest["contract"],
    }
    if artifact_manifest["counts"]["tasks"] != EXPECTED_SHADOW_TASKS:
        raise RuntimeError("archive does not contain exactly 253 tasks")
    artifact_manifest_path = output / "artifact_manifest.json"
    artifact_manifest_path.write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksums = [
        (sha256_path(archive), archive.name),
        (sha256_path(artifact_manifest_path), artifact_manifest_path.name),
        (sha256_path(source_manifest_target), source_manifest_target.name),
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in checksums),
        encoding="utf-8",
    )
    return artifact_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("rubrics/aider_polyglot_cpp_shadow"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.source, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
