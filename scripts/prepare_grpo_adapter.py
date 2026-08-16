"""Create the GLM-4.7 adapter layout used by SGLang and Miles.

The Miles trainer saves LoRA adapters that include the MTP head's tensors
(GLM-4.7-Flash layer 47). The serving adapter contains decoder layers 0-46.
Usage:

    python scripts/prepare_grpo_adapter.py <trainer_adapter_dir> <serve_dir>
    python scripts/prepare_grpo_adapter.py --include-native <trainer_adapter_dir> <hybrid_dir>
    python scripts/prepare_grpo_adapter.py --include-native --include-training-state \
        <same_stage_checkpoint_dir> <resume_dir>

``--include-native`` adds Megatron adapter shards for an SFT-to-GRPO transfer.
``--include-training-state`` additionally includes same-stage optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Any


_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")
_GENERATED_PATTERNS = (
    "adapter_model.bin",
    "adapter_config.json",
    "mtp_strip_manifest.json",
    "adapter_megatron_tp*_pp*.pt",
    "training_state_rank*.pt",
)


def filter_served_layers(state_dict: dict[str, Any], *, num_layers: int) -> tuple[dict[str, Any], list[str]]:
    dropped = []
    kept = {}
    for key, value in state_dict.items():
        match = _LAYER_PATTERN.search(key)
        if match and int(match.group(1)) >= num_layers:
            dropped.append(key)
        else:
            kept[key] = value
    return kept, dropped


def clear_generated_outputs(dst: Path) -> list[Path]:
    removed = []
    for pattern in _GENERATED_PATTERNS:
        for target in sorted(dst.glob(pattern)):
            target.unlink()
            removed.append(target)
    return removed


def copy_native_state(
    src: Path,
    dst: Path,
    *,
    include_training_state: bool = False,
) -> tuple[list[Path], list[Path]]:
    native_files = []
    for source in sorted(src.glob("adapter_megatron_tp*_pp*.pt")):
        target = dst / source.name
        shutil.copy2(source, target)
        native_files.append(target)

    training_state_files = []
    if include_training_state:
        for source in sorted(src.glob("training_state_rank*.pt")):
            target = dst / source.name
            shutil.copy2(source, target)
            training_state_files.append(target)
    return native_files, training_state_files


_TP_ONLY_SHARD_RE = re.compile(r"^adapter_megatron_tp(\d+)_pp0\.pt$")
_EP_SHARDED_SHARD_RE = re.compile(r"^adapter_megatron_tp(\d+)_pp0_ep(\d+)\.pt$")


def classify_native_layout(paths: list[Path]) -> str:
    """Classify shard naming and reject anything the warm-start loader may skip.

    The Miles warm-start loader has only ever been proven to resolve the
    ``tp{r}_pp0.pt`` naming. The synth-v1 SFT saved per-rank
    ``tp{r}_pp0_ep{e}.pt`` shards instead; the r3 GRPO run staged those, the
    loader matched nothing, and training silently proceeded from a fresh LoRA
    init. Every layout must therefore be identified here and acknowledged
    explicitly before it can ship in a hybrid adapter.
    """

    names = sorted(path.name for path in paths)
    tp_only = [_TP_ONLY_SHARD_RE.fullmatch(name) for name in names]
    if all(tp_only):
        ranks = sorted(int(match.group(1)) for match in tp_only)
        if ranks != list(range(len(ranks))):
            raise ValueError(f"tp-only native shard set has rank gaps: {names}")
        return "tp-only"
    ep_sharded = [_EP_SHARDED_SHARD_RE.fullmatch(name) for name in names]
    if all(ep_sharded):
        pairs = [(int(match.group(1)), int(match.group(2))) for match in ep_sharded]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"ep-sharded native shard set has duplicate ranks: {names}")
        return "ep-sharded"
    raise ValueError(f"unrecognized native shard naming: {names}")


def validate_native_shards(paths: list[Path], expected_count: int | None) -> str:
    if not paths:
        raise ValueError("--include-native requires Megatron-native adapter shards")
    layout = classify_native_layout(paths)
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(
            f"native shard count mismatch: {len(paths)} files != expected {expected_count}"
        )
    return layout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--num-layers", type=int, default=47)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--include-training-state", action="store_true")
    parser.add_argument(
        "--allow-ep-sharded-native",
        action="store_true",
        help=(
            "Permit ep-sharded native shard naming (tp{r}_pp0_ep{e}.pt) in the hybrid. "
            "Only after proving in-image that the warm-start loader restores it."
        ),
    )
    parser.add_argument("--expected-native-shards", type=int)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-source-tensors", type=int)
    parser.add_argument("--expected-stripped-tensors", type=int)
    args = parser.parse_args(argv)
    if args.include_training_state and not args.include_native:
        parser.error("--include-training-state requires --include-native")

    import torch

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if src == dst:
        raise SystemExit("source and destination adapter directories must differ")
    source_model = src / "adapter_model.bin"
    source_config = src / "adapter_config.json"
    if not source_model.is_file() or not source_config.is_file():
        raise FileNotFoundError(f"source adapter is incomplete: {src}")
    source_sha256 = _sha256(source_model)
    if args.expected_source_sha256 and source_sha256 != args.expected_source_sha256.lower():
        raise ValueError(
            f"source adapter SHA-256 mismatch: {source_sha256} != "
            f"{args.expected_source_sha256.lower()}"
        )
    state_dict = torch.load(source_model, map_location="cpu", weights_only=True)
    kept, dropped = filter_served_layers(state_dict, num_layers=args.num_layers)
    if args.expected_source_tensors is not None and len(state_dict) != args.expected_source_tensors:
        raise ValueError(
            f"source tensor count mismatch: {len(state_dict)} != {args.expected_source_tensors}"
        )
    if (
        args.expected_stripped_tensors is not None
        and len(dropped) != args.expected_stripped_tensors
    ):
        raise ValueError(
            f"stripped tensor count mismatch: {len(dropped)} != "
            f"{args.expected_stripped_tensors}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"refusing to replace nonempty adapter destination: {dst}")
    with TemporaryDirectory(prefix=f".{dst.name}-preparing-", dir=dst.parent) as temporary:
        staging = Path(temporary)
        output_model = staging / "adapter_model.bin"
        torch.save(kept, output_model)
        shutil.copy2(source_config, staging / "adapter_config.json")
        native_layout = None
        if args.include_native:
            native_files, training_state_files = copy_native_state(
                src,
                staging,
                include_training_state=args.include_training_state,
            )
            native_layout = validate_native_shards(native_files, args.expected_native_shards)
            if native_layout != "tp-only" and not args.allow_ep_sharded_native:
                raise SystemExit(
                    f"source adapter native shards use the '{native_layout}' naming, which "
                    "the Miles warm-start loader has never been proven to resolve — the r3 "
                    "GRPO run silently trained from a fresh LoRA init on exactly this "
                    "layout. Re-shard the source to tp-only naming, or pass "
                    "--allow-ep-sharded-native after proving in-image that the loader "
                    "restores every adapter tensor."
                )
        else:
            native_files, training_state_files = [], []

        manifest = {
            "source": str(src),
            "native_layout": native_layout,
            "num_layers": args.num_layers,
            "source_tensor_count": len(state_dict),
            "kept_tensor_count": len(kept),
            "stripped_tensor_count": len(dropped),
            "first_stripped_tensor": dropped[0] if dropped else None,
            "source_adapter_model_sha256": source_sha256,
            "source_adapter_config_sha256": _sha256(source_config),
            "output_adapter_model_sha256": _sha256(output_model),
            "native_files": {path.name: _sha256(path) for path in native_files},
            "training_state_files": {
                path.name: _sha256(path) for path in training_state_files
            },
            "replaced_files": [],
        }
        (staging / "mtp_strip_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if dst.exists():
            dst.rmdir()
        os.replace(staging, dst)
    print(f"kept {len(kept)}/{len(state_dict)} tensors (stripped {len(dropped)} MTP tensors)")
    print("HYBRID_ADAPTER_READY" if args.include_native else "SERVE_COPY_READY", dst)


if __name__ == "__main__":
    main()
