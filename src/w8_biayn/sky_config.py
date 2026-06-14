"""Render SkyPilot YAML for C++ performance-RL bridge runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .cpp_perf.sandbox import DEFAULT_CPU, DEFAULT_DOCKER_IMAGE
from .constants import (
    CPP_DATA_SCHEMA_VERSION,
    DEFAULT_CPP_CONTAINER_IMAGE,
    DEFAULT_CPP_MODEL,
    DEFAULT_CPP_SMOKE_ACCELERATORS,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_RENDER_DIR,
    RLLM_PIN,
    RLLM_REPO,
    SKYRL_PIN,
    SKYRL_REPO,
)
from .secrets import default_bucket_for_project

Pipeline = Literal["cpp-smoke", "cpp-sft", "cpp-grpo", "cpp-eval"]
LABEL_VALUE_MAX = 63


class LiteralStr(str):
    """Marker for YAML block strings."""


def _literal_representer(dumper: yaml.SafeDumper, data: LiteralStr) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.SafeDumper.add_representer(LiteralStr, _literal_representer)


@dataclass(frozen=True)
class RenderOptions:
    pipeline: Pipeline
    project_id: str
    bucket: str | None = None
    credentials_path: str = DEFAULT_CREDENTIALS_PATH
    accelerators: str = DEFAULT_CPP_SMOKE_ACCELERATORS
    num_nodes: int = 1
    cluster_name: str | None = None
    model: str = DEFAULT_CPP_MODEL
    gpu_container_image: str = DEFAULT_CPP_CONTAINER_IMAGE
    dataset_gcs_prefix: str | None = None
    remote_data_dir: str = "$HOME/.w8-biayn/data/skyrl"
    train_batch_size: int = 16
    n_samples_per_prompt: int = 4
    train_epochs: int = 1
    eval_interval: int = 50
    ckpt_interval: int = -1
    hf_save_interval: int = -1
    ckpt_path: str = "~/ckpts/"
    export_path: str = "~/exports/"
    max_ckpts_to_keep: int = -1
    sandbox_image: str = DEFAULT_DOCKER_IMAGE
    sandbox_cpu: str = DEFAULT_CPU
    logger: str = "console"
    run_id: str | None = None
    owner: str = "sss"
    eval_label: str = "model"
    eval_max_tasks: int | None = None

    @property
    def artifact_bucket(self) -> str:
        return self.bucket or default_bucket_for_project(self.project_id)

    @property
    def name(self) -> str:
        if self.run_id:
            return self.cluster_name or f"w8-biayn-{self.pipeline}-{self.run_id}"
        return self.cluster_name or f"w8-biayn-{self.pipeline}"

    @property
    def data_gcs_prefix(self) -> str:
        return self.dataset_gcs_prefix or f"{self.artifact_bucket}/datasets/cpp-perf/{CPP_DATA_SCHEMA_VERSION}/skyrl"

    @property
    def gpu_count(self) -> int:
        return gpu_count_from_accelerators(self.accelerators)

    @property
    def eval_gcs_prefix(self) -> str:
        run_id = self.run_id or self.pipeline
        return f"{self.artifact_bucket}/runs/cpp-perf/{run_id}/{self.pipeline}"

    @property
    def run_gcs_prefix(self) -> str:
        return self.eval_gcs_prefix

    @property
    def labels(self) -> dict[str, str]:
        if not self.run_id:
            return {}
        return {
            "project": "w8-biayn",
            "phase": "cpp-perf-rl",
            "pipeline": _label_value(self.pipeline),
            "run_id": _label_value(self.run_id),
            "owner": _label_value(self.owner),
            "ttl": "training",
        }


def gpu_count_from_accelerators(accelerators: str) -> int:
    """Parse SkyPilot accelerator strings like A100:8 into a GPU count."""

    if ":" not in accelerators:
        return 1
    try:
        return int(accelerators.rsplit(":", 1)[1])
    except ValueError:
        return 1


def _label_value(value: str) -> str:
    normalized = value.lower().replace("/", "-").replace("_", "-").replace(":", "-")
    normalized = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in normalized)
    normalized = normalized.strip("-")[:LABEL_VALUE_MAX]
    return normalized or "unset"


def setup_script(options: RenderOptions) -> LiteralStr:
    return LiteralStr(
        f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
if ! command -v docker >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl git ca-certificates linux-tools-common linux-tools-generic
  fi
fi
if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl enable --now docker || true
fi
sudo chmod 666 /var/run/docker.sock || true
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
W8_CACHE="$HOME/.cache/w8-biayn/upstreams"
mkdir -p "$W8_CACHE"
if [ ! -d "$W8_CACHE/SkyRL/.git" ]; then
  git clone {SKYRL_REPO} "$W8_CACHE/SkyRL"
fi
git -C "$W8_CACHE/SkyRL" fetch origin {SKYRL_PIN} --depth 1 || git -C "$W8_CACHE/SkyRL" fetch --all --tags
git -C "$W8_CACHE/SkyRL" checkout {SKYRL_PIN}
if [ ! -d "$W8_CACHE/rllm/.git" ]; then
  git clone {RLLM_REPO} "$W8_CACHE/rllm"
fi
git -C "$W8_CACHE/rllm" fetch origin {RLLM_PIN} --depth 1 || git -C "$W8_CACHE/rllm" fetch --all --tags
git -C "$W8_CACHE/rllm" checkout {RLLM_PIN}
"""
    )


def run_script(options: RenderOptions) -> LiteralStr:
    if options.pipeline == "cpp-smoke":
        return smoke_run_script(options)
    if options.pipeline == "cpp-sft":
        return sft_run_script(options)
    if options.pipeline == "cpp-grpo":
        return grpo_run_script(options)
    if options.pipeline == "cpp-eval":
        return eval_run_script(options)
    raise ValueError(f"Unknown pipeline: {options.pipeline}")


def smoke_run_script(options: RenderOptions) -> LiteralStr:
    return LiteralStr(
        f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
export ARTIFACT_BUCKET="{options.artifact_bucket}"
export W8_BIAYN_PIPELINE="cpp-smoke"
export W8_BIAYN_MODEL="{options.model}"
export W8_GPU_CONTAINER_IMAGE="{options.gpu_container_image}"
docker pull "$W8_GPU_CONTAINER_IMAGE"
docker run --rm --gpus all --network host --shm-size=32g \\
  -v "$PWD":/workspace \\
  -v "$HOME/.cache/w8-biayn":/root/.cache/w8-biayn \\
  -v /tmp/w8-gcp-service-account.json:/tmp/w8-gcp-service-account.json:ro \\
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json \\
  -e HF_HOME=/root/.cache/huggingface \\
  -w /workspace \\
  "$W8_GPU_CONTAINER_IMAGE" bash -lc '
set -euxo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 --seed /tmp/w8-cpp-smoke
source /tmp/w8-cpp-smoke/bin/activate
uv pip install vllm transformers accelerate
python - <<PY
from vllm import LLM, SamplingParams
model = "{options.model}"
prompt = "Optimize this C++ program while preserving behavior.\\n```cpp\\n#include <bits/stdc++.h>\\nint main(){{long long n,s=0; std::cin>>n; for(long long i=1;i<=n;i++) s+=i; std::cout<<s<<\"\\\\n\";}}\\n```"
llm = LLM(model=model, max_model_len=4096, trust_remote_code=True)
outputs = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.2))
print(outputs[0].outputs[0].text[:1000])
PY
'
"""
    )


def training_prelude(options: RenderOptions) -> str:
    return f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
export ARTIFACT_BUCKET="{options.artifact_bucket}"
export W8_BIAYN_PIPELINE="{options.pipeline}"
export W8_BIAYN_MODEL="{options.model}"
export W8_BIAYN_MODEL_PATH="{options.model}"
export W8_GPU_CONTAINER_IMAGE="{options.gpu_container_image}"
export W8_DATA_GCS_PREFIX="{options.data_gcs_prefix}"
export W8_DATA_DIR="{options.remote_data_dir}"
export W8_GPUS_PER_NODE="{options.gpu_count}"
export W8_RUN_GCS_PREFIX="{options.run_gcs_prefix}"
export W8_ARTIFACT_DIR="$HOME/.w8-biayn/runs/{options.run_id or options.pipeline}/{options.pipeline}"
export W8_CKPT_PATH="{options.ckpt_path}"
export W8_EXPORT_PATH="{options.export_path}"
mkdir -p "$W8_DATA_DIR"
rm -rf "$W8_ARTIFACT_DIR"
mkdir -p "$W8_ARTIFACT_DIR/ckpts" "$W8_ARTIFACT_DIR/exports"
if [ "$W8_CKPT_PATH" = "~/ckpts/" ]; then
  export W8_CKPT_PATH="/artifacts/ckpts"
fi
if [ "$W8_EXPORT_PATH" = "~/exports/" ]; then
  export W8_EXPORT_PATH="/artifacts/exports"
fi
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required on the SkyPilot host to restore dataset cache from GCS" >&2
  exit 2
fi
gcloud storage cp --recursive "$W8_DATA_GCS_PREFIX/*" "$W8_DATA_DIR/"
test -f "$W8_DATA_DIR/_w8_data_manifest.json"
test -d "$W8_DATA_DIR/tasks"
if [[ "$W8_BIAYN_MODEL_PATH" == gs://* ]]; then
  export W8_LOCAL_MODEL_DIR="$HOME/.w8-biayn/models/{options.pipeline}"
  rm -rf "$W8_LOCAL_MODEL_DIR"
  mkdir -p "$W8_LOCAL_MODEL_DIR"
  gcloud storage cp --recursive "$W8_BIAYN_MODEL_PATH/*" "$W8_LOCAL_MODEL_DIR/"
  export W8_BIAYN_MODEL_PATH="$W8_LOCAL_MODEL_DIR"
fi
docker pull "$W8_GPU_CONTAINER_IMAGE"
if [ "{options.sandbox_image}" != "{DEFAULT_DOCKER_IMAGE}" ]; then
  docker pull "{options.sandbox_image}"
fi
"""


def training_container_prefix(options: RenderOptions) -> str:
    return """docker run --rm --gpus all --network host --shm-size=32g \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -v /tmp:/tmp \\
  -v "$PWD":/workspace \\
  -v "$HOME/.cache/w8-biayn":/root/.cache/w8-biayn \\
  -v "$W8_DATA_DIR":/data \\
  -v "$W8_ARTIFACT_DIR":/artifacts \\
  -v /tmp/w8-gcp-service-account.json:/tmp/w8-gcp-service-account.json:ro \\
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json \\
  -e W8_BIAYN_DATA_DIR=/data \\
  -e W8_BIAYN_MODEL_PATH="$W8_BIAYN_MODEL_PATH" \\
  -e W8_CKPT_PATH="$W8_CKPT_PATH" \\
  -e W8_EXPORT_PATH="$W8_EXPORT_PATH" \\
  -e HF_HOME=/root/.cache/huggingface \\
  -w /workspace \\
  "$W8_GPU_CONTAINER_IMAGE" bash -lc '
set -euxo pipefail
if ! command -v docker >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
  else
    echo "docker CLI is required inside the training container for cpp-perf rewards" >&2
    exit 2
  fi
fi
docker version
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 --seed /tmp/w8-train
source /tmp/w8-train/bin/activate
uv pip install -e /workspace
uv pip install -e /root/.cache/w8-biayn/upstreams/rllm
cd /root/.cache/w8-biayn/upstreams/SkyRL
uv sync --active --extra fsdp --extra gcp
cd /workspace
uv pip install --no-deps -e /workspace
"""


def training_artifact_upload_script() -> str:
    return """
if find "$W8_ARTIFACT_DIR/exports" -mindepth 1 -print -quit | grep -q .; then
  gcloud storage cp --recursive "$W8_ARTIFACT_DIR/exports" "$W8_RUN_GCS_PREFIX/"
else
  echo "no exports found under $W8_ARTIFACT_DIR/exports" >&2
fi
if find "$W8_ARTIFACT_DIR/ckpts" -mindepth 1 -print -quit | grep -q .; then
  gcloud storage cp --recursive "$W8_ARTIFACT_DIR/ckpts" "$W8_RUN_GCS_PREFIX/"
else
  echo "no checkpoints found under $W8_ARTIFACT_DIR/ckpts" >&2
fi
"""


def sft_run_script(options: RenderOptions) -> LiteralStr:
    return LiteralStr(
        training_prelude(options)
        + training_container_prefix(options)
        + f"""
test -f /data/sft/train.jsonl
test -f /data/sft/validation.jsonl
python -m skyrl.train.main_sft \\
  strategy=fsdp \\
  model.path="$W8_BIAYN_MODEL_PATH" \\
  dataset_name=/data/sft \\
  dataset_split=train \\
  eval_dataset_name=/data/sft \\
  eval_dataset_split=validation \\
  eval_interval={options.eval_interval} \\
  placement.num_nodes={options.num_nodes} \\
  placement.num_gpus_per_node={options.gpu_count} \\
  batch_size={options.train_batch_size} \\
  micro_train_batch_size_per_gpu=1 \\
  num_epochs={options.train_epochs} \\
  max_length=8192 \\
  ckpt_path="$W8_CKPT_PATH" \\
  ckpt_interval={max(options.ckpt_interval, 0)} \\
  hf_save_interval={max(options.hf_save_interval, 0)} \\
  export_path="$W8_EXPORT_PATH" \\
  max_ckpts_to_keep={options.max_ckpts_to_keep} \\
  logger={options.logger} \\
  project_name=w8_biayn_cpp_sft \\
  run_name={options.pipeline}
'
"""
        + training_artifact_upload_script()
    )


def grpo_run_script(options: RenderOptions) -> LiteralStr:
    return LiteralStr(
        training_prelude(options)
        + training_container_prefix(options)
        + f"""
test -f /data/grpo/train.parquet
test -f /data/grpo/validation.parquet
w8-biayn cpp harness preflight --image "{options.sandbox_image}" --cpu "{options.sandbox_cpu}"
python -m w8_biayn.integrations.skyrl_cpp_perf_main \\
  'data.train_data=[/data/grpo/train.parquet]' \\
  'data.val_data=[/data/grpo/validation.parquet]' \\
  environment.env_class=cpp-perf \\
  trainer.algorithm.advantage_estimator=grpo \\
  trainer.algorithm.use_kl_loss=false \\
  trainer.policy.model.path="$W8_BIAYN_MODEL_PATH" \\
  trainer.strategy=fsdp \\
  trainer.placement.policy_num_nodes={options.num_nodes} \\
  trainer.placement.policy_num_gpus_per_node={options.gpu_count} \\
  trainer.logger={options.logger} \\
  trainer.epochs={options.train_epochs} \\
  trainer.eval_interval={options.eval_interval} \\
  trainer.ckpt_interval={options.ckpt_interval} \\
  trainer.hf_save_interval={options.hf_save_interval} \\
  trainer.ckpt_path="$W8_CKPT_PATH" \\
  trainer.export_path="$W8_EXPORT_PATH" \\
  trainer.max_ckpts_to_keep={options.max_ckpts_to_keep} \\
  trainer.train_batch_size={options.train_batch_size} \\
  trainer.policy_mini_batch_size={options.train_batch_size} \\
  trainer.micro_train_batch_size_per_gpu=1 \\
  trainer.max_prompt_length=8192 \\
  generator.max_turns=1 \\
  generator.n_samples_per_prompt={options.n_samples_per_prompt} \\
  generator.inference_engine.num_engines={options.gpu_count} \\
  generator.inference_engine.tensor_parallel_size=1 \\
  generator.inference_engine.pipeline_parallel_size=1 \\
  generator.inference_engine.data_parallel_size=1 \\
  generator.sampling_params.temperature=0.8
'
"""
        + training_artifact_upload_script()
    )


def eval_run_script(options: RenderOptions) -> LiteralStr:
    max_tasks_arg = f"--max-tasks {options.eval_max_tasks}" if options.eval_max_tasks else ""
    return LiteralStr(
        training_prelude(options)
        + f"""export W8_EVAL_OUTPUT_DIR="/tmp/w8-cpp-eval-{options.run_id or 'manual'}"
export W8_EVAL_MODEL="{options.model}"
if [[ "$W8_EVAL_MODEL" == gs://* ]]; then
  export W8_EVAL_LOCAL_MODEL="$HOME/.w8-biayn/models/{options.eval_label}"
  rm -rf "$W8_EVAL_LOCAL_MODEL"
  mkdir -p "$W8_EVAL_LOCAL_MODEL"
  gcloud storage cp --recursive "$W8_EVAL_MODEL/*" "$W8_EVAL_LOCAL_MODEL/"
  export W8_EVAL_MODEL="$W8_EVAL_LOCAL_MODEL"
fi
rm -rf "$W8_EVAL_OUTPUT_DIR"
mkdir -p "$W8_EVAL_OUTPUT_DIR"
docker run --rm --gpus all --network host --shm-size=32g \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -v /tmp:/tmp \\
  -v "$PWD":/workspace \\
  -v "$HOME/.cache/w8-biayn":/root/.cache/w8-biayn \\
  -v "$W8_DATA_DIR":/data \\
  -v /tmp/w8-gcp-service-account.json:/tmp/w8-gcp-service-account.json:ro \\
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json \\
  -e W8_BIAYN_DATA_DIR=/data \\
  -e W8_EVAL_OUTPUT_DIR="$W8_EVAL_OUTPUT_DIR" \\
  -e W8_EVAL_MODEL="$W8_EVAL_MODEL" \\
  -e HF_HOME=/root/.cache/huggingface \\
  -w /workspace \\
  "$W8_GPU_CONTAINER_IMAGE" bash -lc '
set -euxo pipefail
if ! command -v docker >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
  else
    echo "docker CLI is required inside the eval container for cpp-perf rewards" >&2
    exit 2
  fi
fi
docker version
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 --seed /tmp/w8-eval
source /tmp/w8-eval/bin/activate
uv pip install -e /workspace
uv pip install vllm
w8-biayn cpp harness preflight --image "{options.sandbox_image}" --cpu "{options.sandbox_cpu}"
python -m w8_biayn.integrations.cpp_eval_main \\
  --data-dir /data \\
  --model "$W8_EVAL_MODEL" \\
  --label "{options.eval_label}" \\
  --output-dir "$W8_EVAL_OUTPUT_DIR" \\
  --samples-per-task {options.n_samples_per_prompt} \\
  --sandbox-image "{options.sandbox_image}" \\
  --sandbox-cpu "{options.sandbox_cpu}" \\
  {max_tasks_arg}
'
gcloud storage cp --recursive "$W8_EVAL_OUTPUT_DIR" "{options.eval_gcs_prefix}/"
"""
    )


def render_sky_yaml(options: RenderOptions) -> str:
    config: dict[str, Any] = {
        "name": options.name,
        "resources": {
            "infra": "gcp",
            "accelerators": options.accelerators,
            "memory": "128+",
        },
        "num_nodes": options.num_nodes,
        "workdir": ".",
        "file_mounts": {
            "/tmp/w8-gcp-service-account.json": options.credentials_path,
        },
        "envs": {
            "W8_BIAYN_PIPELINE": options.pipeline,
            "W8_BIAYN_ARTIFACT_BUCKET": options.artifact_bucket,
            "W8_BIAYN_MODEL": options.model,
            "W8_BIAYN_DATA_GCS_PREFIX": options.data_gcs_prefix,
            "W8_BIAYN_RUN_ID": options.run_id or "",
        },
        "setup": setup_script(options),
        "run": run_script(options),
    }
    if options.labels:
        config["resources"]["labels"] = options.labels
    return yaml.safe_dump(config, sort_keys=False)


def write_sky_yaml(options: RenderOptions, output: str | Path | None = None) -> Path:
    output_path = Path(output or f"{DEFAULT_RENDER_DIR}/{options.pipeline}.sky.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_sky_yaml(options), encoding="utf-8")
    return output_path
