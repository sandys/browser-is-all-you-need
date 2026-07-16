"""Canonical Modal reproduction path for GLM-4.7-Flash post-training."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal


APP_NAME = "glm47-pie-cpp"
MODEL_ID = "zai-org/GLM-4.7-Flash"
MODEL_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
MILES_IMAGE = (
    "radixark/miles:latest-cu12@"
    "sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37"
)

LOCAL_REPO = Path(__file__).resolve().parents[2]
REMOTE_REPO = "/workspace/glm47-h100-posttraining"
MODELS_DIR = "/root/models"
ASSETS_DIR = "/workspace/assets"
RUNS_DIR = "/workspace/runs"

app = modal.App(APP_NAME)
models = modal.Volume.from_name("glm47-models", create_if_missing=True)
assets = modal.Volume.from_name("glm47-assets", create_if_missing=True)
runs = modal.Volume.from_name("glm47-runs", create_if_missing=True)

source_ignore = [".git", ".glm47-posttraining", ".pytest_cache", ".ruff_cache", "wandb"]

prepare_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub[hf-transfer]==1.23.0")
    .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, copy=True, ignore=source_ignore)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# This is the same runtime definition as the repository Dockerfile, extended
# with the GCC 13 toolchain used by the Modal C++ reward scorer.
training_image = (
    modal.Image.from_dockerfile(
        LOCAL_REPO / "Dockerfile",
        context_dir=LOCAL_REPO,
        add_python=None,
        build_args={"MILES_BASE_IMAGE": MILES_IMAGE},
    )
    .apt_install("software-properties-common", "rsync", "gawk", "util-linux", "git")
    .run_commands(
        "add-apt-repository -y ppa:ubuntu-toolchain-r/test "
        "&& apt-get update "
        "&& DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-13 g++-13 "
        "&& update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 "
        "&& update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100"
    )
    .add_local_dir(LOCAL_REPO, remote_path=REMOTE_REPO, copy=True, ignore=source_ignore)
)

gpu_config = {
    "image": training_image,
    "gpu": "H100!:8",
    "cpu": 48.0,
    "memory": (262_144, 1_048_576),
    "timeout": 86_400,
    "volumes": {
        MODELS_DIR: models,
        ASSETS_DIR: assets,
        RUNS_DIR: runs,
    },
    "secrets": [modal.Secret.from_name("wandb-glm47")],
}


def _run(command: str, *, env: dict[str, str] | None = None) -> None:
    merged = {**os.environ, **(env or {})}
    subprocess.run(
        ["bash", "-lc", command],
        cwd=REMOTE_REPO,
        env=merged,
        check=True,
    )


@app.function(
    image=prepare_image,
    cpu=16.0,
    memory=32_768,
    timeout=14_400,
    volumes={MODELS_DIR: models, ASSETS_DIR: assets},
)
def prepare_assets() -> None:
    """Download the exact base model, dataset, and validated SFT adapter."""
    from huggingface_hub import HfApi, snapshot_download

    resolved = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION).sha
    if resolved != MODEL_REVISION:
        raise RuntimeError(f"model revision mismatch: {resolved} != {MODEL_REVISION}")

    target = Path(MODELS_DIR, "GLM-4.7-Flash")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=target,
    )
    target.joinpath("MODEL_REVISION").write_text(f"{MODEL_REVISION}\n", encoding="utf-8")

    _run(f"python3 scripts/download_assets.py data --output-root {ASSETS_DIR}")
    _run(f"python3 scripts/download_assets.py sft --output-root {ASSETS_DIR}")
    models.commit()
    assets.commit()


def _stage_env(stage: str, run_id: str, adapter_path: str) -> dict[str, str]:
    env = {
        "MILES_RUN_ID": run_id,
        "MILES_RUN_ROOT": f"{RUNS_DIR}/{run_id}",
        "MILES_HF_CHECKPOINT": f"{MODELS_DIR}/GLM-4.7-Flash",
        "MILES_REF_LOAD_DIR": f"{MODELS_DIR}/GLM-4.7-Flash_torch_dist_tp4_pp1_ep8",
        "MILES_CPP_TASKS_DIR": f"{ASSETS_DIR}/data/tasks",
        "MILES_CPP_DATA_DIR": f"{ASSETS_DIR}/prepared",
        "GLM47_CPP_SANDBOX_BACKEND": "local",
        "GLM47_MODEL_REVISION": MODEL_REVISION,
        "GLM47_TRAINING_IMAGE": MILES_IMAGE,
        "GLM47_EXPERIMENT_ID": run_id,
        "MILES_WANDB_PROJECT": "glm47-pie-cpp-posttraining",
        "MILES_WANDB_GROUP": run_id,
        "MILES_WANDB_RUN_ID": run_id,
        "MILES_WANDB_JOB_TYPE": stage,
        "WANDB_RUN_GROUP": run_id,
        "WANDB_JOB_TYPE": stage,
        "WANDB_TAGS": f"canonical,modal,8xh100,pie-cpp,{stage}",
    }
    if stage == "grpo":
        env["MILES_LORA_ADAPTER_PATH"] = adapter_path
    return env


@app.function(**gpu_config)
def run_stage(
    stage: str,
    run_id: str = "",
    adapter_path: str = f"{ASSETS_DIR}/adapters/sft",
) -> str:
    """Run conversion, SFT, or GRPO on one Modal 8x H100 container."""
    if stage not in {"convert", "sft", "grpo"}:
        raise ValueError(f"unsupported stage: {stage}")

    resolved_run_id = run_id or f"glm47-modal-{stage}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    env = _stage_env(stage, resolved_run_id, adapter_path)
    _run("python3 -m pip install --no-deps -e .")
    _run("python3 scripts/check_runtime.py")
    _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

    try:
        if stage == "convert":
            _run("bash scripts/convert_checkpoint.sh", env=env)
        else:
            _run(f"bash examples/{stage}.sh", env=env)
    finally:
        models.commit()
        assets.commit()
        runs.commit()
    return resolved_run_id


@app.local_entrypoint()
def prepare() -> None:
    prepare_assets.remote()


@app.local_entrypoint()
def convert(run_id: str = "") -> None:
    print(run_stage.remote("convert", run_id=run_id))


@app.local_entrypoint()
def sft(run_id: str = "") -> None:
    print(run_stage.remote("sft", run_id=run_id))


@app.local_entrypoint()
def grpo(run_id: str = "", adapter_path: str = f"{ASSETS_DIR}/adapters/sft") -> None:
    print(run_stage.remote("grpo", run_id=run_id, adapter_path=adapter_path))
