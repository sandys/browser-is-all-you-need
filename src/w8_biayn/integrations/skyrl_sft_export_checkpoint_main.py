"""Export a SkyRL SFT FSDP checkpoint to HuggingFace format."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

try:
    import ray
    from skyrl.train.config.sft_config import SFTConfig, build_skyrl_config_for_sft, validate_sft_cfg
    from skyrl.train.sft_trainer import SFTTrainer
    from skyrl.train.utils.utils import initialize_ray
except ImportError:  # pragma: no cover - local tests can import helpers without SkyRL
    ray = None
    SFTConfig = None
    build_skyrl_config_for_sft = None
    validate_sft_cfg = None
    SFTTrainer = None
    initialize_ray = None


STEP_RE = re.compile(r"global_step_(\d+)")
WEIGHT_SUFFIXES = (".safetensors", ".bin")


def main() -> None:
    if (
        ray is None
        or SFTConfig is None
        or build_skyrl_config_for_sft is None
        or validate_sft_cfg is None
        or SFTTrainer is None
        or initialize_ray is None
    ):
        raise ImportError("SkyRL is required to export SFT checkpoints")

    parser = argparse.ArgumentParser(description="Export a SkyRL SFT checkpoint to HF format.")
    parser.add_argument("--model", required=True, help="Base HF model path used to initialize the FSDP workers.")
    parser.add_argument("--checkpoint", required=True, help="SkyRL global_step_N checkpoint directory.")
    parser.add_argument("--export-path", required=True, help="Local HF export root.")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-train-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--logger", default="console")
    args = parser.parse_args()

    step = _extract_step(args.checkpoint)
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.model.path = args.model
    cfg.placement.num_nodes = args.num_nodes
    cfg.placement.num_gpus_per_node = args.num_gpus_per_node
    cfg.batch_size = args.batch_size
    cfg.micro_train_batch_size_per_gpu = args.micro_train_batch_size_per_gpu
    cfg.max_length = args.max_length
    cfg.num_steps = 1
    cfg.num_epochs = None
    cfg.logger = args.logger
    cfg.project_name = "w8_biayn_cpp_sft_export"
    cfg.run_name = "cpp-sft-export"
    cfg.ckpt_path = ""
    cfg.ckpt_interval = 0
    cfg.hf_save_interval = 1
    cfg.export_path = args.export_path

    validate_sft_cfg(cfg)
    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    initialize_ray(skyrl_cfg)
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)
    try:
        trainer.setup()
        world_size = args.num_nodes * args.num_gpus_per_node
        with _policy_checkpoint_source(args.checkpoint, world_size) as policy_checkpoint:
            trainer.dispatch.load_checkpoint(
                "policy",
                policy_checkpoint,
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
            )
        trainer.global_step = step
        trainer.save_hf_model()
        summary = _validate_export(Path(args.export_path) / f"global_step_{step}" / "policy")
        print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        trainer.shutdown()
        ray.shutdown()


def _extract_step(checkpoint: str) -> int:
    match = STEP_RE.search(checkpoint)
    if not match:
        raise ValueError(f"checkpoint path must contain global_step_N: {checkpoint}")
    return int(match.group(1))


@contextmanager
def _policy_checkpoint_source(checkpoint: str, world_size: int) -> Iterator[str]:
    policy_checkpoint = checkpoint.rstrip("/") + "/policy"
    if not checkpoint.startswith(("gs://", "gcs://")):
        local_policy = Path(policy_checkpoint)
        _validate_policy_checkpoint(local_policy, world_size)
        yield str(local_policy)
        return

    with tempfile.TemporaryDirectory(prefix="w8-sft-policy-") as temp_dir:
        local_policy = Path(temp_dir) / "policy"
        local_policy.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "gcloud",
                "storage",
                "rsync",
                "--recursive",
                policy_checkpoint,
                str(local_policy),
            ],
            check=True,
        )
        _validate_policy_checkpoint(local_policy, world_size)
        yield str(local_policy)


def _validate_policy_checkpoint(policy_dir: Path, world_size: int) -> None:
    missing = []
    for rank in range(world_size):
        for prefix in ("model", "extra_state"):
            path = policy_dir / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            if not path.exists():
                missing.append(path.name)
    if missing:
        raise FileNotFoundError(
            f"Policy checkpoint is missing {len(missing)} expected rank file(s) in {policy_dir}: {missing}"
        )


def _validate_export(export_dir: Path) -> dict[str, object]:
    if not export_dir.exists():
        raise FileNotFoundError(f"HF export directory was not created: {export_dir}")
    objects = [path for path in export_dir.iterdir() if path.is_file()]
    weight_files = [path for path in objects if path.name.endswith(WEIGHT_SUFFIXES)]
    if not weight_files:
        names = sorted(path.name for path in objects)
        raise RuntimeError(f"HF export has no model weight files in {export_dir}: {names}")
    return {
        "export_dir": str(export_dir),
        "object_count": len(objects),
        "total_bytes": sum(path.stat().st_size for path in objects),
        "weight_files": [path.name for path in weight_files],
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
    }


if __name__ == "__main__":
    main()
