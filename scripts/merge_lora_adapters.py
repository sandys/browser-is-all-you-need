"""Merge two compatible LoRA adapters without introducing factor cross-terms.

For input rank ``r`` and scale ``alpha / r``, the output uses rank ``2r`` and
the same alpha.  A tensors are concatenated on their rank dimension; B tensors
are concatenated after applying the coefficient required to represent the
requested weighted sum of the two adapter deltas exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


ADAPTER_MODEL = "adapter_model.bin"
ADAPTER_CONFIG = "adapter_config.json"
NATIVE_GLOB = "adapter_megatron_tp*_pp*.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_dim(tensor: Any, rank: int, key: str) -> int:
    # Megatron TP shards partition the global LoRA rank, so a rank-16 adapter
    # has local rank 4 at TP=4. Its module layout fixes the factor axes even
    # though the local dimension no longer equals the global config rank.
    if ".adapter.linear_in.weight" in key:
        if tensor.ndim < 2:
            raise ValueError(f"{key}: expected a matrix-like A factor, got {tensor.shape}")
        return tensor.ndim - 2
    if ".adapter.linear_out.weight" in key:
        if tensor.ndim < 2:
            raise ValueError(f"{key}: expected a matrix-like B factor, got {tensor.shape}")
        return tensor.ndim - 1
    dimensions = [index for index, size in enumerate(tensor.shape) if size == rank]
    if len(dimensions) != 1:
        raise ValueError(
            f"{key}: expected exactly one rank-{rank} dimension, got {tensor.shape}"
        )
    return dimensions[0]


def merge_state(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    rank: int,
    input_scale: float,
    output_scale: float,
    left_weight: float,
    right_weight: float,
    prefix: str = "",
) -> dict[str, Any]:
    """Return a state tree whose LoRA delta is the weighted input-delta sum."""
    import torch

    if set(left) != set(right):
        missing_left = sorted(set(right) - set(left))[:5]
        missing_right = sorted(set(left) - set(right))[:5]
        raise ValueError(
            f"state keys differ; missing_left={missing_left}, missing_right={missing_right}"
        )

    merged: dict[str, Any] = {}
    for name in sorted(left):
        key = f"{prefix}.{name}" if prefix else name
        lhs = left[name]
        rhs = right[name]
        if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
            merged[name] = merge_state(
                lhs,
                rhs,
                rank=rank,
                input_scale=input_scale,
                output_scale=output_scale,
                left_weight=left_weight,
                right_weight=right_weight,
                prefix=key,
            )
            continue
        if torch.is_tensor(lhs) and torch.is_tensor(rhs):
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
                raise ValueError(
                    f"{key}: tensor mismatch {lhs.shape}/{lhs.dtype} != "
                    f"{rhs.shape}/{rhs.dtype}"
                )
            is_a_factor = "lora_A" in key or ".adapter.linear_in.weight" in key
            is_b_factor = "lora_B" in key or ".adapter.linear_out.weight" in key
            if is_a_factor:
                dim = _rank_dim(lhs, rank, key)
                merged[name] = torch.cat((lhs, rhs), dim=dim).contiguous()
            elif is_b_factor:
                dim = _rank_dim(lhs, rank, key)
                lhs_factor = left_weight * input_scale / output_scale
                rhs_factor = right_weight * input_scale / output_scale
                merged[name] = torch.cat(
                    (lhs * lhs_factor, rhs * rhs_factor), dim=dim
                ).contiguous()
            elif torch.equal(lhs, rhs):
                merged[name] = lhs.clone()
            else:
                raise ValueError(f"{key}: non-LoRA tensors differ")
            continue
        if lhs != rhs:
            raise ValueError(f"{key}: non-tensor values differ")
        merged[name] = lhs
    return merged


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads((path / ADAPTER_CONFIG).read_text(encoding="utf-8"))


def _compatible_configs(left: dict[str, Any], right: dict[str, Any]) -> None:
    if left != right:
        differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise ValueError(f"adapter configs differ: {differing}")
    if left.get("peft_type") != "LORA":
        raise ValueError(f"expected LORA adapter, got {left.get('peft_type')!r}")


def merge_adapters(
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    *,
    left_weight: float = 0.5,
    right_weight: float = 0.5,
) -> dict[str, Any]:
    import torch

    if left_dir == right_dir or output_dir in {left_dir, right_dir}:
        raise ValueError("left, right, and output adapter directories must be distinct")
    if left_weight < 0 or right_weight < 0 or not math.isclose(
        left_weight + right_weight, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("merge weights must be nonnegative and sum to 1")
    left_config = _load_config(left_dir)
    right_config = _load_config(right_dir)
    _compatible_configs(left_config, right_config)
    rank = int(left_config["r"])
    alpha = float(left_config["lora_alpha"])
    output_rank = 2 * rank
    output_alpha = alpha
    input_scale = alpha / rank
    output_scale = output_alpha / output_rank

    left_native = {path.name: path for path in left_dir.glob(NATIVE_GLOB)}
    right_native = {path.name: path for path in right_dir.glob(NATIVE_GLOB)}
    if not left_native or set(left_native) != set(right_native):
        raise ValueError("native Megatron shard sets are missing or differ")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to replace nonempty destination: {output_dir}")

    inputs = [(ADAPTER_MODEL, left_dir / ADAPTER_MODEL, right_dir / ADAPTER_MODEL)]
    inputs.extend((name, left_native[name], right_native[name]) for name in sorted(left_native))
    for _, lhs, rhs in inputs:
        if not lhs.is_file() or not rhs.is_file():
            raise FileNotFoundError(f"missing merge input: {lhs} or {rhs}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-merging-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        file_receipts: dict[str, Any] = {}
        for name, lhs, rhs in inputs:
            left_state = torch.load(lhs, map_location="cpu", weights_only=True)
            right_state = torch.load(rhs, map_location="cpu", weights_only=True)
            if not isinstance(left_state, Mapping) or not isinstance(right_state, Mapping):
                raise TypeError(f"{name}: expected mapping state dictionaries")
            merged = merge_state(
                left_state,
                right_state,
                rank=rank,
                input_scale=input_scale,
                output_scale=output_scale,
                left_weight=left_weight,
                right_weight=right_weight,
            )
            destination = staging / name
            torch.save(merged, destination)
            file_receipts[name] = {
                "left_sha256": _sha256(lhs),
                "right_sha256": _sha256(rhs),
                "output_sha256": _sha256(destination),
                "top_level_keys": len(merged),
            }

        output_config = dict(left_config)
        output_config["r"] = output_rank
        output_config["lora_alpha"] = output_alpha
        (staging / ADAPTER_CONFIG).write_text(
            json.dumps(output_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "kind": "exact-weighted-lora-delta-merge",
            "left": str(left_dir),
            "right": str(right_dir),
            "left_weight": left_weight,
            "right_weight": right_weight,
            "input_rank": rank,
            "input_alpha": alpha,
            "output_rank": output_rank,
            "output_alpha": output_alpha,
            "files": file_receipts,
            "adapter_config_sha256": _sha256(staging / ADAPTER_CONFIG),
        }
        (staging / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("output")
    parser.add_argument("--left-weight", type=float, default=0.5)
    parser.add_argument("--right-weight", type=float, default=0.5)
    args = parser.parse_args(argv)
    manifest = merge_adapters(
        Path(args.left).resolve(),
        Path(args.right).resolve(),
        Path(args.output).resolve(),
        left_weight=args.left_weight,
        right_weight=args.right_weight,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("MERGED_ADAPTER_READY", Path(args.output).resolve())


if __name__ == "__main__":
    main()
