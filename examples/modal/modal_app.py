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

# Only meaningful locally (image build/mount). Modal re-imports this module inside
# the container as /root/modal_app.py, where parents[2] does not exist, so fall back
# to the module dir there rather than crashing the import before the function loads.
_MODULE_PATH = Path(__file__).resolve()
LOCAL_REPO = _MODULE_PATH.parents[2] if len(_MODULE_PATH.parents) >= 3 else _MODULE_PATH.parent
REMOTE_REPO = "/workspace/glm47-h100-posttraining"
MODELS_DIR = "/root/models"
ASSETS_DIR = "/workspace/assets"
RUNS_DIR = "/workspace/runs"
AIDER_TASKS_DIR = f"{ASSETS_DIR}/aider-shadow/tasks/aider_polyglot_cpp_shadow"
AIDER_DATASET_KIND = "aider-polyglot-cpp-shadow-grpo"
AIDER_SFT_ADAPTER = (
    f"{RUNS_DIR}/glm47-aider-complement-530-sft-20260721/checkpoints/"
    "sft_lora_r16/iter_0000025/adapter"
)
AIDER_SFT_ADAPTER_SHA256 = "f1ea45bc327dc6e28d0287aea75c6b691e99d2ec2f7fdb7f07bbbf5ccd6cf36a"
AIDER_1211_ADAPTER = (
    f"{RUNS_DIR}/glm47-aider-1211-sft-20260718T192250Z/checkpoints/"
    "sft_lora_r16/iter_0000036/adapter"
)
AIDER_MERGED_ADAPTER = f"{RUNS_DIR}/glm47-aider-1211-530-equal-delta-merge-r32"

app = modal.App(APP_NAME)
models = modal.Volume.from_name("glm47-models", create_if_missing=True)
assets = modal.Volume.from_name("glm47-assets", create_if_missing=True)
runs = modal.Volume.from_name("glm47-runs", create_if_missing=True)

source_ignore = [
    ".git",
    ".glm47-posttraining",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".w8-biayn",
    "artifacts",
    "rubrics",
    "wandb",
]

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
    .apt_install(
        "software-properties-common", "rsync", "gawk", "util-linux", "git", "bubblewrap"
    )
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
    _run(f"python3 scripts/download_assets.py aider-shadow --output-root {ASSETS_DIR}")
    models.commit()
    assets.commit()


@app.function(
    image=prepare_image,
    cpu=4.0,
    memory=8_192,
    timeout=3_600,
    volumes={ASSETS_DIR: assets},
)
def prepare_aider_shadow_asset() -> dict[str, object]:
    """Download and verify only the externally versioned Aider corpus."""
    import json

    _run(f"python3 scripts/download_assets.py aider-shadow --output-root {ASSETS_DIR}")
    manifest_path = Path(ASSETS_DIR, "aider-shadow", "artifact_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets.commit()
    return manifest


@app.function(
    image=training_image,
    cpu=16.0,
    memory=65_536,
    timeout=3_600,
    volumes={RUNS_DIR: runs},
)
def merge_aider_adapter_files(
    left_path: str = AIDER_1211_ADAPTER,
    right_path: str = AIDER_SFT_ADAPTER,
    output_path: str = AIDER_MERGED_ADAPTER,
) -> dict[str, object]:
    """Create the exact equal-weight rank-32 delta merge used for GRPO."""
    import json

    _run(
        "python3 scripts/merge_lora_adapters.py "
        f"--left-weight 0.5 --right-weight 0.5 {left_path} {right_path} {output_path}"
    )
    runs.commit()
    return json.loads(Path(output_path, "merge_manifest.json").read_text(encoding="utf-8"))


@app.function(
    image=training_image,
    cpu=16.0,
    memory=32_768,
    timeout=3_600,
    volumes={ASSETS_DIR: assets, RUNS_DIR: runs},
)
def validate_aider_path(adapter_path: str = AIDER_SFT_ADAPTER) -> dict[str, object]:
    """CPU-only proof that data, sandbox, and exact warm-start bytes are usable."""

    import json
    import shutil

    data_dir = Path("/tmp/aider-shadow-preflight")
    hybrid_dir = Path("/tmp/aider-hybrid-preflight")
    shutil.rmtree(data_dir, ignore_errors=True)
    shutil.rmtree(hybrid_dir, ignore_errors=True)
    env = {
        "GLM47_CPP_SANDBOX_BACKEND": "local",
        "GLM47_CPP_SANDBOX_UNSHARE_NET": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    _run("python3 -m pip install --no-deps -e .", env=env)
    _run(
        "python3 -m glm47_posttraining.integrations.miles_aider_polyglot "
        f"build-data --tasks-dir {AIDER_TASKS_DIR} --out {data_dir} --force",
        env=env,
    )
    _run(
        "python3 -m glm47_posttraining.integrations.miles_aider_polyglot preflight",
        env=env,
    )
    _run(
        "python3 scripts/prepare_grpo_adapter.py --include-native "
        "--expected-native-shards 4 --expected-source-tensors 9741 "
        "--expected-stripped-tensors 207 "
        f"--expected-source-sha256 {AIDER_SFT_ADAPTER_SHA256} "
        f"{adapter_path} {hybrid_dir}",
        env=env,
    )
    data_manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    adapter_manifest = json.loads(
        (hybrid_dir / "mtp_strip_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "status": "passed",
        "dataset_kind": data_manifest["kind"],
        "train_tasks": data_manifest["counts"]["train"],
        "monitor_tasks": data_manifest["counts"]["monitor"],
        "source_tree_sha256": data_manifest["source_tree_sha256"],
        "source_adapter_sha256": adapter_manifest["source_adapter_model_sha256"],
        "source_tensors": adapter_manifest["source_tensor_count"],
        "serving_tensors": adapter_manifest["kept_tensor_count"],
        "stripped_tensors": adapter_manifest["stripped_tensor_count"],
        "native_shards": sorted(adapter_manifest["native_files"]),
    }


def _stage_env(
    stage: str,
    run_id: str,
    adapter_path: str,
    source_commit: str = "",
    sft_num_epoch: str = "",
    sft_save_interval: str = "",
    sft_data_dir: str = "",
    aider_data_dir: str = "",
    adapter_sha256: str = "",
    lora_rank: str = "",
    lora_alpha: str = "",
    num_rollout: str = "",
) -> dict[str, str]:
    env = {
        "MILES_RUN_ID": run_id,
        "MILES_RUN_ROOT": f"{RUNS_DIR}/{run_id}",
        "MILES_HF_CHECKPOINT": f"{MODELS_DIR}/GLM-4.7-Flash",
        "MILES_REF_LOAD_DIR": f"{MODELS_DIR}/GLM-4.7-Flash_torch_dist_tp4_pp1_ep8",
        "MILES_CPP_TASKS_DIR": f"{ASSETS_DIR}/data/tasks",
        "MILES_CPP_DATA_DIR": f"{ASSETS_DIR}/prepared",
        "GLM47_CPP_SANDBOX_BACKEND": "local",
        # Modal's runtime rejects bwrap's netns loopback setup (RTM_NEWADDR).
        "GLM47_CPP_SANDBOX_UNSHARE_NET": "0",
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
        "GLM47_SOURCE_COMMIT": source_commit or "unbound",
    }
    if stage == "sft":
        # Data max is 2270 tokens (600-set token audit), so 3072 removes ~25% of
        # the wasted seq buffer vs 4096 without truncating any row (3072 % TP4 == 0).
        env["MILES_SEQ_LENGTH"] = "3072"
        # Batch 20 divides 600 evenly (0 rows dropped) and maximizes optimizer steps
        # (30/epoch). A larger batch would cut steps and recreate the undertraining.
        env["MILES_GLOBAL_BATCH_SIZE"] = "20"
        env["MILES_ROLLOUT_BATCH_SIZE"] = "20"
        if sft_num_epoch:
            env["MILES_SFT_NUM_EPOCH"] = sft_num_epoch
        if sft_save_interval:
            env["MILES_SAVE_INTERVAL"] = sft_save_interval
        if sft_data_dir:
            env["MILES_CPP_DATA_DIR"] = sft_data_dir
    elif stage == "grpo":
        env["MILES_LORA_ADAPTER_PATH"] = adapter_path
    elif stage in {"aider_profile", "aider_grpo"}:
        env.update(
            {
                "MILES_CPP_TASKS_DIR": AIDER_TASKS_DIR,
                "MILES_CPP_DATA_DIR": f"{RUNS_DIR}/{run_id}/data",
                "MILES_DATA_BUILD_MODULE": (
                    "glm47_posttraining.integrations.miles_aider_polyglot"
                ),
                "MILES_CUSTOM_RM_PATH": (
                    "glm47_posttraining.integrations.miles_aider_polyglot.reward_func"
                ),
                "MILES_REWARD_PREFLIGHT_MODULE": (
                    "glm47_posttraining.integrations.miles_aider_polyglot"
                ),
                "MILES_EXPECTED_DATASET_KIND": AIDER_DATASET_KIND,
                "MILES_EVAL_NAME": "aider_shadow_train_monitor",
                "MILES_EVAL_PROMPT_DATA": (
                    f"{RUNS_DIR}/{run_id}/data/eval/train_monitor.jsonl"
                ),
                "MILES_LORA_ADAPTER_PATH": adapter_path,
                "MILES_EXPECTED_SOURCE_ADAPTER_SHA256": (
                    adapter_sha256 or AIDER_SFT_ADAPTER_SHA256
                ),
                "MILES_EXPECTED_SOURCE_TENSORS": "9741",
                "MILES_EXPECTED_STRIPPED_TENSORS": "207",
                "MILES_EXPECTED_NATIVE_SHARDS": "4",
                "GLM47_SYNC_METRICS_DIR": f"{RUNS_DIR}/{run_id}/sync_metrics",
                "MILES_SEQ_LENGTH": "6144",
                "MILES_ROLLOUT_MAX_RESPONSE_LEN": "4096",
                # The pinned GLM generation config declares endoftext, user,
                # and observation as terminal ids. Miles otherwise preserves
                # the decoded stop marker and can glue it to Aider's final
                # closing fence, hiding the last edited file from the scorer.
                "MILES_ROLLOUT_STOP_TOKEN_IDS": "154820 154827 154829",
                "MILES_ROLLOUT_SKIP_SPECIAL_TOKENS": "1",
                # Match the frozen fixed-26 eval contract (temp 0.7); the failed
                # 20260721 run rolled at 1.0 and 49% of samples died on format.
                "MILES_ROLLOUT_TEMPERATURE": "0.7",
                "MILES_EVAL_MAX_RESPONSE_LEN": "4096",
                "MILES_MAX_TOKENS_PER_GPU": "12288",
                "MILES_RECOMPUTE_GRANULARITY": "full",
                "MILES_LR": "5e-7",
                "MILES_NO_REF": "0",
                "MILES_KL_LOSS_COEF": "0.02",
                "MILES_USE_KL_LOSS": "1",
                "MILES_SAVE_INTERVAL": "1",
                "MILES_EVAL_INTERVAL": "1",
                # ~5 epochs over the 169-task difficulty-filtered set:
                # 26 updates x 32 prompts = 832 slots = 4.92 passes.
                "MILES_NUM_ROLLOUT": num_rollout or (
                    "1" if stage == "aider_profile" else "26"
                ),
                "MILES_WANDB_PROJECT": "glm47-aider-polyglot-cpp-grpo",
                "MILES_WANDB_GROUP": run_id,
                "MILES_WANDB_RUN_ID": run_id,
                "MILES_WANDB_JOB_TYPE": "grpo-profile" if stage == "aider_profile" else "grpo",
                "WANDB_RUN_GROUP": run_id,
                "WANDB_JOB_TYPE": "grpo-profile" if stage == "aider_profile" else "grpo",
                "WANDB_TAGS": (
                    "aider-shadow,modal,8xh100,grpo,profile"
                    if stage == "aider_profile"
                    else "aider-shadow,modal,8xh100,grpo,full"
                ),
                "GLM47_TIMING_STATUS": "profile" if stage == "aider_profile" else "full",
            }
        )
        if aider_data_dir:
            # Pre-built (and difficulty-filtered) dataset staged on the assets
            # volume; train_grpo.sh skips its in-container 253-task build when
            # grpo/train.jsonl already exists. The staged manifest declares the
            # filtered count, and the training gate verifies against it.
            env["MILES_CPP_DATA_DIR"] = aider_data_dir
            env["MILES_EVAL_PROMPT_DATA"] = f"{aider_data_dir}/eval/train_monitor.jsonl"
            env["MILES_EXPECTED_TRAIN_COUNT"] = "169"
        if lora_rank:
            env["MILES_LORA_RANK"] = lora_rank
        if lora_alpha:
            env["MILES_LORA_ALPHA"] = lora_alpha
    return env


@app.function(**gpu_config)
def run_stage(
    stage: str,
    run_id: str = "",
    adapter_path: str = f"{ASSETS_DIR}/adapters/sft",
    source_commit: str = "",
    sft_num_epoch: str = "",
    sft_save_interval: str = "",
    sft_data_dir: str = "",
    aider_data_dir: str = "",
    adapter_sha256: str = "",
    lora_rank: str = "",
    lora_alpha: str = "",
    num_rollout: str = "",
) -> str:
    """Run conversion, SFT, or GRPO on one Modal 8x H100 container."""
    if stage not in {"convert", "sft", "grpo", "aider_profile", "aider_grpo"}:
        raise ValueError(f"unsupported stage: {stage}")

    resolved_run_id = run_id or f"glm47-modal-{stage}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    env = _stage_env(
        stage,
        resolved_run_id,
        adapter_path,
        source_commit,
        sft_num_epoch=sft_num_epoch,
        sft_save_interval=sft_save_interval,
        sft_data_dir=sft_data_dir,
        aider_data_dir=aider_data_dir,
        adapter_sha256=adapter_sha256,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        num_rollout=num_rollout,
    )
    if stage in {"aider_profile", "aider_grpo"} and Path(RUNS_DIR, resolved_run_id).exists():
        raise FileExistsError(f"refusing to reuse Aider run ID: {resolved_run_id}")
    _run("python3 -m pip install --no-deps -e .")
    _run("python3 scripts/check_runtime.py")
    _run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

    try:
        if stage == "convert":
            _run("bash scripts/convert_checkpoint.sh", env=env)
        else:
            script_stage = "grpo" if stage in {"aider_profile", "aider_grpo"} else stage
            _run(f"bash examples/{script_stage}.sh", env=env)
    finally:
        models.commit()
        assets.commit()
        runs.commit()
    return resolved_run_id


@app.local_entrypoint()
def prepare() -> None:
    prepare_assets.remote()


@app.local_entrypoint()
def merge_aider(
    left_path: str = AIDER_1211_ADAPTER,
    right_path: str = AIDER_SFT_ADAPTER,
    output_path: str = AIDER_MERGED_ADAPTER,
) -> None:
    import json

    print(
        json.dumps(
            merge_aider_adapter_files.remote(left_path, right_path, output_path),
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint()
def aider_preflight(adapter_path: str = AIDER_SFT_ADAPTER) -> None:
    import json

    print(json.dumps(validate_aider_path.remote(adapter_path), indent=2, sort_keys=True))


@app.local_entrypoint()
def convert(run_id: str = "") -> None:
    print(run_stage.remote("convert", run_id=run_id))


@app.local_entrypoint()
def sft(
    run_id: str = "",
    num_epoch: str = "",
    save_interval: str = "",
    data_dir: str = "",
) -> None:
    print(
        run_stage.remote(
            "sft",
            run_id=run_id,
            sft_num_epoch=num_epoch,
            sft_save_interval=save_interval,
            sft_data_dir=data_dir,
        )
    )


@app.local_entrypoint()
def grpo(run_id: str = "", adapter_path: str = f"{ASSETS_DIR}/adapters/sft") -> None:
    print(run_stage.remote("grpo", run_id=run_id, adapter_path=adapter_path))


def _local_source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=LOCAL_REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@app.local_entrypoint()
def aider_profile(
    run_id: str = "", adapter_path: str = AIDER_SFT_ADAPTER, data_dir: str = ""
) -> None:
    print(
        run_stage.remote(
            "aider_profile",
            run_id=run_id,
            adapter_path=adapter_path,
            source_commit=_local_source_commit(),
            aider_data_dir=data_dir,
        )
    )


@app.local_entrypoint()
def aider_grpo(
    run_id: str = "",
    adapter_path: str = AIDER_SFT_ADAPTER,
    data_dir: str = "",
    adapter_sha256: str = "",
    lora_rank: str = "",
    lora_alpha: str = "",
    num_rollout: str = "",
) -> None:
    print(
        run_stage.remote(
            "aider_grpo",
            run_id=run_id,
            adapter_path=adapter_path,
            source_commit=_local_source_commit(),
            aider_data_dir=data_dir,
            adapter_sha256=adapter_sha256,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            num_rollout=num_rollout,
        )
    )
