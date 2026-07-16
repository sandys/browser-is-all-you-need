#!/usr/bin/env python3
"""Fail fast when the Miles/SGLang FlashInfer packages are incompatible."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from importlib import metadata


MINIMUM_VERSIONS = {
    "flashinfer-python": "0.6.12",
    "flashinfer-cubin": "0.6.12",
    "flashinfer-jit-cache": "0.6.12",
    "sglang-kernel": "0.4.4",
    "torch-memory-saver": "0.0.9.post1",
}
FLASHINFER_PACKAGES = (
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
)
REQUIRED_PACKAGES = (*FLASHINFER_PACKAGES, "sglang-kernel", "torch-memory-saver")


def version_key(raw_version: str) -> tuple[int, int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?", raw_version)
    if match is None:
        raise ValueError(f"cannot parse package version {raw_version!r}")
    major, minor, patch, post = match.groups()
    return int(major), int(minor), int(patch), int(post or 0)


def validate_miles_h100_runtime(
    version_lookup: Callable[[str], str] = metadata.version,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = version_lookup(package)
        except metadata.PackageNotFoundError:
            missing.append(package)

    if missing:
        raise RuntimeError(f"missing Miles H100 runtime packages: {', '.join(missing)}")

    releases = {package: version_key(raw) for package, raw in versions.items()}
    for package, minimum in MINIMUM_VERSIONS.items():
        if releases[package] < version_key(minimum):
            raise RuntimeError(
                f"{package} {versions[package]} is below the required minimum {minimum}"
            )

    python_release = releases["flashinfer-python"]
    if any(releases[package] != python_release for package in FLASHINFER_PACKAGES):
        rendered = ", ".join(f"{name}={versions[name]}" for name in FLASHINFER_PACKAGES)
        raise RuntimeError(f"FlashInfer package versions are not aligned: {rendered}")
    return versions


def main() -> int:
    try:
        versions = validate_miles_h100_runtime()
    except (RuntimeError, ValueError) as exc:
        print(f"Miles H100 runtime preflight failed: {exc}", file=sys.stderr)
        print(
            "Build examples/miles/Dockerfile.h100-runtime and relaunch with that image.",
            file=sys.stderr,
        )
        return 2

    rendered = ", ".join(f"{name}={versions[name]}" for name in REQUIRED_PACKAGES)
    print(f"Miles H100 runtime preflight passed: {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
