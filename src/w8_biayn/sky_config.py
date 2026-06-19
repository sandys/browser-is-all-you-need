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
    DEFAULT_CPP_EVAL_TRANSFORMERS_VERSION,
    DEFAULT_CPP_EVAL_VLLM_VERSION,
    DEFAULT_CPP_MODEL,
    DEFAULT_CPP_SMOKE_ACCELERATORS,
    DEFAULT_CPP_TRAIN_MODEL,
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
GRPO_MIN_SAMPLES_PER_GPU_STEP = 16
GRPO_MULTINODE_RESUME_MIN_DISK_GB = 2048
SKYRL_RAY_PG_TIMEOUT_S = 1800
SKYRL_WORKER_NCCL_TIMEOUT_S = 3600


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
    disk_size: int | None = None
    cluster_name: str | None = None
    model: str | None = None
    gpu_container_image: str = DEFAULT_CPP_CONTAINER_IMAGE
    dataset_gcs_prefix: str | None = None
    remote_data_dir: str = "$HOME/.w8-biayn/data/skyrl"
    train_batch_size: int = 16
    micro_train_batch_size_per_gpu: int = 1
    grpo_use_kl_loss: bool = True
    grpo_kl_loss_coef: float = 0.001
    grpo_use_entropy_loss: bool = True
    grpo_entropy_loss_coef: float = 0.001
    grpo_vllm_gpu_memory_utilization: float = 0.7
    n_samples_per_prompt: int = 4
    train_epochs: int = 1
    eval_before_train: bool = True
    eval_interval: int = 50
    max_env_workers: int = 32
    ckpt_interval: int = -1
    hf_save_interval: int = -1
    ckpt_path: str = "~/ckpts/"
    export_path: str = "~/exports/"
    max_ckpts_to_keep: int = -1
    resume_from: str = ""
    sft_export_checkpoint: str = ""
    sandbox_image: str = DEFAULT_DOCKER_IMAGE
    sandbox_cpu: str = DEFAULT_CPU
    logger: str = "console"
    run_id: str | None = None
    owner: str = "sss"
    eval_label: str = "model"
    eval_max_tasks: int | None = None
    allow_low_multinode_utilization: bool = False

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
    def effective_model(self) -> str:
        if self.model:
            return self.model
        if self.pipeline == "cpp-smoke":
            return DEFAULT_CPP_MODEL
        return DEFAULT_CPP_TRAIN_MODEL

    @property
    def gpu_count(self) -> int:
        return gpu_count_from_accelerators(self.accelerators)

    @property
    def total_gpu_count(self) -> int:
        return self.gpu_count * self.num_nodes

    @property
    def effective_samples_per_step(self) -> int:
        return self.train_batch_size * self.n_samples_per_prompt

    @property
    def samples_per_gpu_per_step(self) -> float | None:
        if self.total_gpu_count <= 0:
            return None
        return self.effective_samples_per_step / self.total_gpu_count

    @property
    def min_multinode_effective_samples_per_step(self) -> int:
        return GRPO_MIN_SAMPLES_PER_GPU_STEP * self.total_gpu_count

    @property
    def grpo_multinode_utilization_ok(self) -> bool:
        if self.pipeline != "cpp-grpo" or self.num_nodes <= 1:
            return True
        return (
            self.effective_samples_per_step >= self.min_multinode_effective_samples_per_step
            and self.max_env_workers >= self.effective_samples_per_step
        )

    @property
    def effective_disk_size(self) -> int:
        if self.disk_size is not None:
            return self.disk_size
        if self.pipeline == "cpp-grpo" and self.num_nodes > 1 and self.resume_from.strip():
            return GRPO_MULTINODE_RESUME_MIN_DISK_GB
        if self.pipeline in {"cpp-sft", "cpp-grpo", "cpp-eval"}:
            return 1024
        return 256

    @property
    def effective_memory(self) -> str:
        if self.pipeline == "cpp-eval":
            return "80+"
        return "128+"

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


def _skyrl_resume_mode(resume_from: str) -> str:
    value = resume_from.strip()
    if not value:
        return "null"
    if value == "latest":
        return "latest"
    return f'"{value}"'


def gcp_env_exports(options: RenderOptions) -> str:
    """Return scoped GCP auth exports used by host and container scripts."""

    return f"""export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/tmp/w8-gcp-service-account.json
export CLOUDSDK_CORE_PROJECT="{options.project_id}"
export GOOGLE_CLOUD_PROJECT="{options.project_id}"
export GCLOUD_PROJECT="{options.project_id}"
export CLOUDSDK_CORE_DISABLE_PROMPTS=1"""


def setup_script(options: RenderOptions) -> LiteralStr:
    return LiteralStr(
        f"""set -euxo pipefail
{gcp_env_exports(options)}
if ! command -v docker >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl git ca-certificates
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
{gcp_env_exports(options)}
export ARTIFACT_BUCKET="{options.artifact_bucket}"
export W8_BIAYN_PIPELINE="cpp-smoke"
export W8_BIAYN_MODEL="{options.effective_model}"
export W8_GPU_CONTAINER_IMAGE="{options.gpu_container_image}"
docker pull "$W8_GPU_CONTAINER_IMAGE"
docker run --rm --gpus all --network host --shm-size=32g \\
  -v "$PWD":/workspace \\
  -v "$HOME/.cache/w8-biayn":/root/.cache/w8-biayn \\
  -v /tmp/w8-gcp-service-account.json:/tmp/w8-gcp-service-account.json:ro \\
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json \\
  -e CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/tmp/w8-gcp-service-account.json \\
  -e CLOUDSDK_CORE_PROJECT="$CLOUDSDK_CORE_PROJECT" \\
  -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \\
  -e GCLOUD_PROJECT="$GCLOUD_PROJECT" \\
  -e CLOUDSDK_CORE_DISABLE_PROMPTS="$CLOUDSDK_CORE_DISABLE_PROMPTS" \\
  -e HF_HOME=/root/.cache/huggingface \\
  -w /workspace \\
  "$W8_GPU_CONTAINER_IMAGE" bash -lc '
set -euxo pipefail
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear --python 3.12 --seed /tmp/w8-cpp-smoke
source /tmp/w8-cpp-smoke/bin/activate
uv pip install vllm transformers accelerate
python - <<PY
from vllm import LLM, SamplingParams
model = "{options.effective_model}"
prompt = "Optimize this C++ program while preserving behavior.\\n```cpp\\n#include <bits/stdc++.h>\\nint main(){{long long n,s=0; std::cin>>n; for(long long i=1;i<=n;i++) s+=i; std::cout<<s<<\"\\\\n\";}}\\n```"
llm = LLM(model=model, max_model_len=4096, trust_remote_code=True)
outputs = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.2))
print(outputs[0].outputs[0].text[:1000])
PY
'
"""
    )


def training_prelude(options: RenderOptions) -> str:
    runtime_prelude = ""
    if options.pipeline in {"cpp-grpo", "cpp-eval"}:
        runtime_prelude = f"""uv run w8-biayn cpp harness preflight --image "{options.sandbox_image}" --cpu "{options.sandbox_cpu}"
"""
    prelude = f"""set -euxo pipefail
{gcp_env_exports(options)}
export ARTIFACT_BUCKET="{options.artifact_bucket}"
export W8_BIAYN_PIPELINE="{options.pipeline}"
export W8_BIAYN_MODEL="{options.effective_model}"
export W8_BIAYN_MODEL_PATH="{options.effective_model}"
export W8_GPU_CONTAINER_IMAGE="{options.gpu_container_image}"
export W8_DATA_GCS_PREFIX="{options.data_gcs_prefix}"
export W8_DATA_DIR="{options.remote_data_dir}"
export W8_GPUS_PER_NODE="{options.gpu_count}"
export W8_RUN_GCS_PREFIX="{options.run_gcs_prefix}"
export W8_ARTIFACT_DIR="$HOME/.w8-biayn/runs/{options.run_id or options.pipeline}/{options.pipeline}"
export W8_CKPT_PATH="{options.ckpt_path}"
export W8_EXPORT_PATH="{options.export_path}"
export W8_SFT_EXPORT_CHECKPOINT="{options.sft_export_checkpoint}"
export SKYRL_RAY_PG_TIMEOUT_IN_S="${{SKYRL_RAY_PG_TIMEOUT_IN_S:-{SKYRL_RAY_PG_TIMEOUT_S}}}"
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${{SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-{SKYRL_WORKER_NCCL_TIMEOUT_S}}}"
mkdir -p "$W8_DATA_DIR"
rm -rf "$W8_ARTIFACT_DIR"
mkdir -p "$W8_ARTIFACT_DIR/ckpts" "$W8_ARTIFACT_DIR/exports"
if [ "$W8_CKPT_PATH" = "~/ckpts/" ]; then
  export W8_CKPT_PATH="/artifacts/ckpts"
fi
if [ "$W8_EXPORT_PATH" = "~/exports/" ]; then
  export W8_EXPORT_PATH="/artifacts/exports"
fi
{runtime_prelude.rstrip()}
if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is required on the SkyPilot host to restore dataset cache from GCS" >&2
  exit 2
fi
resolve_gcs_model_export() {{
  local source="${{1%/}}"
  has_gcs_policy_weights() {{
    local candidate="${{1%/}}"
    gcloud storage ls "$candidate/*.safetensors" >/dev/null 2>&1 || gcloud storage ls "$candidate/pytorch_model*.bin" >/dev/null 2>&1 || gcloud storage ls "$candidate/model*.bin" >/dev/null 2>&1
  }}
  if [[ "$source" =~ /global_step_[0-9]+/policy$ ]]; then
    if has_gcs_policy_weights "$source"; then
      echo "$source"
      return 0
    fi
    echo "no model weight files found under concrete HF policy export $source" >&2
    return 2
  fi
  local steps=()
  mapfile -t steps < <(gcloud storage ls "$source/" 2>/dev/null | sed -n 's#.*/global_step_\\([0-9][0-9]*\\)/$#\\1#p' | sort -nr)
  if [ "${{#steps[@]}}" -eq 0 ]; then
    echo "$source"
    return 0
  fi
  local step
  for step in "${{steps[@]}}"; do
    local candidate="$source/global_step_${{step}}/policy"
    if has_gcs_policy_weights "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  echo "no complete HF policy export with weights found under $source" >&2
  return 2
}}
assert_local_hf_model() {{
  local model_dir=$1
  test -f "$model_dir/config.json"
  if ! find "$model_dir" -maxdepth 1 \\( -name "*.safetensors" -o -name "pytorch_model*.bin" -o -name "model*.bin" \\) -print -quit | grep -q .; then
    echo "no model weight files found under $model_dir" >&2
    return 2
  fi
}}
normalize_local_hf_tokenizer_config() {{
  local model_dir=$1
  local tokenizer_config="$model_dir/tokenizer_config.json"
  if [ ! -f "$tokenizer_config" ]; then
    return 0
  fi
  python3 - "$tokenizer_config" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
extra = data.get("extra_special_tokens")
if not isinstance(extra, list):
    raise SystemExit(0)
additional = data.get("additional_special_tokens", [])
if not isinstance(additional, list):
    additional = []
merged = list(additional)
for token in extra:
    if token not in merged:
        merged.append(token)
data["additional_special_tokens"] = merged
data.pop("extra_special_tokens", None)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n")
print(f"normalized tokenizer_config extra_special_tokens list into additional_special_tokens for {{path.parent}}")
PY
}}
"""
    if options.sft_export_checkpoint:
        data_restore = f"""echo "{options.pipeline} export-only: skipping dataset restore"
"""
    else:
        data_restore = """if [ -f "$W8_DATA_DIR/.w8_data_gcs_prefix" ] \\
  && [ "$(cat "$W8_DATA_DIR/.w8_data_gcs_prefix")" = "$W8_DATA_GCS_PREFIX" ] \\
  && [ -f "$W8_DATA_DIR/_w8_data_manifest.json" ] \\
  && [ -d "$W8_DATA_DIR/tasks" ]; then
  echo "dataset cache already restored for $W8_DATA_GCS_PREFIX"
else
  rm -rf "$W8_DATA_DIR"
  mkdir -p "$W8_DATA_DIR"
  gcloud storage cp --recursive "$W8_DATA_GCS_PREFIX/*" "$W8_DATA_DIR/"
  printf '%s\\n' "$W8_DATA_GCS_PREFIX" > "$W8_DATA_DIR/.w8_data_gcs_prefix"
fi
test -f "$W8_DATA_DIR/_w8_data_manifest.json"
test -d "$W8_DATA_DIR/tasks"
"""
    return (
        prelude
        + data_restore
        + f"""
if [[ "$W8_BIAYN_MODEL_PATH" == gs://* ]]; then
  export W8_BIAYN_MODEL_SOURCE="$W8_BIAYN_MODEL_PATH"
  export W8_BIAYN_MODEL_PATH="$(resolve_gcs_model_export "$W8_BIAYN_MODEL_PATH")"
  echo "resolved GCS model export: $W8_BIAYN_MODEL_SOURCE -> $W8_BIAYN_MODEL_PATH"
  export W8_LOCAL_MODEL_DIR="$W8_ARTIFACT_DIR/model"
  rm -rf "$W8_LOCAL_MODEL_DIR"
  mkdir -p "$W8_LOCAL_MODEL_DIR"
  gcloud storage cp --recursive "$W8_BIAYN_MODEL_PATH/*" "$W8_LOCAL_MODEL_DIR/"
  assert_local_hf_model "$W8_LOCAL_MODEL_DIR"
  normalize_local_hf_tokenizer_config "$W8_LOCAL_MODEL_DIR"
  export W8_BIAYN_MODEL_PATH="/artifacts/model"
fi
docker pull "$W8_GPU_CONTAINER_IMAGE"
if [ "{options.sandbox_image}" != "{DEFAULT_DOCKER_IMAGE}" ]; then
  docker pull "{options.sandbox_image}"
fi
"""
    )


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
  -e CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/tmp/w8-gcp-service-account.json \\
  -e CLOUDSDK_CORE_PROJECT="$CLOUDSDK_CORE_PROJECT" \\
  -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \\
  -e GCLOUD_PROJECT="$GCLOUD_PROJECT" \\
  -e CLOUDSDK_CORE_DISABLE_PROMPTS="$CLOUDSDK_CORE_DISABLE_PROMPTS" \\
  -e W8_BIAYN_DATA_DIR=/data \\
  -e W8_BIAYN_MODEL_PATH="$W8_BIAYN_MODEL_PATH" \\
  -e W8_CKPT_PATH="$W8_CKPT_PATH" \\
  -e W8_EXPORT_PATH="$W8_EXPORT_PATH" \\
  -e W8_SFT_EXPORT_CHECKPOINT="$W8_SFT_EXPORT_CHECKPOINT" \\
  -e SKYRL_RAY_PG_TIMEOUT_IN_S="$SKYRL_RAY_PG_TIMEOUT_IN_S" \\
  -e SKYRL_WORKER_NCCL_TIMEOUT_IN_S="$SKYRL_WORKER_NCCL_TIMEOUT_IN_S" \\
  -e SKYPILOT_NODE_RANK="${SKYPILOT_NODE_RANK:-0}" \\
  -e SKYPILOT_NODE_IPS="${SKYPILOT_NODE_IPS:-}" \\
  -e SKYPILOT_NUM_NODES="${SKYPILOT_NUM_NODES:-1}" \\
  -e SKYPILOT_NUM_GPUS_PER_NODE="${SKYPILOT_NUM_GPUS_PER_NODE:-$W8_GPUS_PER_NODE}" \\
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
uv venv --clear --python 3.12 --seed /tmp/w8-train
source /tmp/w8-train/bin/activate
uv pip install -e /workspace
uv pip install -e /root/.cache/w8-biayn/upstreams/rllm
cd /root/.cache/w8-biayn/upstreams/SkyRL
python -m w8_biayn.integrations.skyrl_io_patch
uv sync --active --extra fsdp --extra gcp
cd /workspace
uv pip install gcsfs
uv pip install --no-deps -e /workspace
"""


def training_artifact_upload_script() -> str:
    return """
if [ "${SKYPILOT_NODE_RANK:-0}" != "0" ]; then
  echo "skipping artifact upload on worker rank ${SKYPILOT_NODE_RANK:-unknown}"
  exit 0
fi
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


def grpo_training_command(options: RenderOptions) -> str:
    # fsdp_size = GPUs-per-node keeps policy/ref FSDP sharding node-local (HSDP) on
    # multi-node runs: the model reduce-scatters over NVLink and only does one gradient
    # all-reduce across the slow inter-node link per step, instead of all-gathering the
    # whole model across nodes on every micro-step. On a single node it equals the world
    # size, so SkyRL renders a no-op flat mesh and behavior is unchanged.
    return f"""python -m w8_biayn.integrations.skyrl_cpp_perf_main \\
  'data.train_data=[/data/grpo/train.parquet]' \\
  'data.val_data=[/data/grpo/validation.parquet]' \\
  environment.env_class=cpp-perf \\
  environment.skyrl_gym.max_env_workers={options.max_env_workers} \\
  trainer.algorithm.advantage_estimator=grpo \\
  trainer.algorithm.use_kl_loss={str(options.grpo_use_kl_loss).lower()} \\
  trainer.algorithm.kl_loss_coef={options.grpo_kl_loss_coef:g} \\
  trainer.algorithm.use_entropy_loss={str(options.grpo_use_entropy_loss).lower()} \\
  trainer.algorithm.entropy_loss_coef={options.grpo_entropy_loss_coef:g} \\
  trainer.policy.model.path="$W8_BIAYN_MODEL_PATH" \\
  trainer.strategy=fsdp \\
  trainer.placement.colocate_all=true \\
  trainer.placement.policy_num_nodes={options.num_nodes} \\
  trainer.placement.policy_num_gpus_per_node={options.gpu_count} \\
  trainer.placement.ref_num_nodes={options.num_nodes} \\
  trainer.placement.ref_num_gpus_per_node={options.gpu_count} \\
  trainer.policy.fsdp_config.fsdp_size={options.gpu_count} \\
  trainer.ref.fsdp_config.fsdp_size={options.gpu_count} \\
  trainer.logger={options.logger} \\
  trainer.epochs={options.train_epochs} \\
  trainer.eval_before_train={str(options.eval_before_train).lower()} \\
  trainer.eval_interval={options.eval_interval} \\
  trainer.ckpt_interval={options.ckpt_interval} \\
  trainer.hf_save_interval={options.hf_save_interval} \\
  trainer.ckpt_path="$W8_CKPT_PATH" \\
  trainer.export_path="$W8_EXPORT_PATH" \\
  trainer.max_ckpts_to_keep={options.max_ckpts_to_keep} \\
  trainer.resume_mode={_skyrl_resume_mode(options.resume_from)} \\
  trainer.train_batch_size={options.train_batch_size} \\
  trainer.policy_mini_batch_size={options.train_batch_size} \\
  trainer.micro_train_batch_size_per_gpu={options.micro_train_batch_size_per_gpu} \\
  trainer.max_prompt_length=8192 \\
  generator.max_turns=1 \\
  generator.n_samples_per_prompt={options.n_samples_per_prompt} \\
  generator.inference_engine.num_engines={options.total_gpu_count} \\
  generator.inference_engine.tensor_parallel_size=1 \\
  generator.inference_engine.pipeline_parallel_size=1 \\
  generator.inference_engine.data_parallel_size=1 \\
  generator.inference_engine.backend=vllm \\
  generator.inference_engine.run_engines_locally=true \\
  generator.inference_engine.weight_sync_backend=nccl \\
  generator.inference_engine.async_engine=true \\
  generator.inference_engine.gpu_memory_utilization={options.grpo_vllm_gpu_memory_utilization:g} \\
  generator.batched=false \\
  generator.sampling_params.max_generate_length=1024 \\
  generator.sampling_params.temperature=0.8"""


def grpo_multinode_ray_script(options: RenderOptions, train_command: str) -> str:
    if options.num_nodes <= 1:
        return train_command
    indented_train = "\n".join(f"  {line}" if line else "" for line in train_command.splitlines())
    return f"""wait_for_ray() {{
  local address=$1
  for _ in $(seq 1 60); do
    if ray status --address "$address" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "Ray cluster at $address failed to become ready" >&2
  return 1
}}

start_ray_worker() {{
  local address=$1
  for _ in $(seq 1 60); do
    if ray start --address "$address" --disable-usage-stats --num-gpus {options.gpu_count}; then
      return 0
    fi
    sleep 5
  done
  echo "Worker rank ${{SKYPILOT_NODE_RANK:-unknown}} failed to join Ray at $address" >&2
  return 1
}}

read -r W8_RAY_HEAD_IP _ <<< "${{SKYPILOT_NODE_IPS:-127.0.0.1}}"
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
if [ "${{SKYPILOT_NODE_RANK:-0}}" = "0" ]; then
  if ! ray status --address 127.0.0.1:6479 >/dev/null 2>&1; then
    ray start --head --disable-usage-stats --port 6479 --num-gpus {options.gpu_count}
  fi
  wait_for_ray 127.0.0.1:6479
  export RAY_ADDRESS=127.0.0.1:6479
  cleanup_ray() {{ ray stop --force || true; }}
  trap cleanup_ray EXIT
{indented_train}
else
  start_ray_worker "$W8_RAY_HEAD_IP:6479"
  wait_for_ray "$W8_RAY_HEAD_IP:6479"
  echo "worker rank ${{SKYPILOT_NODE_RANK:-unknown}} joined Ray at $W8_RAY_HEAD_IP:6479"
  while ray status --address "$W8_RAY_HEAD_IP:6479" >/dev/null 2>&1; do
    sleep 30
  done
  ray stop --force || true
fi"""


def export_checkpoint_run_body(options: RenderOptions) -> str:
    return f"""
test -n "$W8_SFT_EXPORT_CHECKPOINT"
python -m w8_biayn.integrations.skyrl_sft_export_checkpoint_main \\
  --model "$W8_BIAYN_MODEL_PATH" \\
  --checkpoint "$W8_SFT_EXPORT_CHECKPOINT" \\
  --export-path "$W8_EXPORT_PATH" \\
  --num-nodes {options.num_nodes} \\
  --num-gpus-per-node {options.gpu_count} \\
  --batch-size {options.train_batch_size} \\
  --micro-train-batch-size-per-gpu {options.micro_train_batch_size_per_gpu} \\
  --max-length 8192 \\
  --logger {options.logger}
"""


def sft_run_script(options: RenderOptions) -> LiteralStr:
    if options.sft_export_checkpoint:
        run_body = export_checkpoint_run_body(options)
    else:
        run_body = f"""
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
  micro_train_batch_size_per_gpu={options.micro_train_batch_size_per_gpu} \\
  num_epochs={options.train_epochs} \\
  max_length=8192 \\
  ckpt_path="$W8_CKPT_PATH" \\
  ckpt_interval={max(options.ckpt_interval, 0)} \\
  hf_save_interval={max(options.hf_save_interval, 0)} \\
  export_path="$W8_EXPORT_PATH" \\
  max_ckpts_to_keep={options.max_ckpts_to_keep} \\
  resume_from="{options.resume_from}" \\
  logger={options.logger} \\
  project_name=w8_biayn_cpp_sft \\
  run_name={options.pipeline}
"""
    return LiteralStr(
        training_prelude(options)
        + training_container_prefix(options)
        + run_body
        + "\n'\n"
        + training_artifact_upload_script()
    )


def grpo_run_script(options: RenderOptions) -> LiteralStr:
    if options.sft_export_checkpoint:
        return LiteralStr(
            training_prelude(options)
            + training_container_prefix(options)
            + export_checkpoint_run_body(options)
            + "\n'\n"
            + training_artifact_upload_script()
        )
    train_command = grpo_multinode_ray_script(options, grpo_training_command(options))
    return LiteralStr(
        training_prelude(options)
        + training_container_prefix(options)
        + f"""
test -f /data/grpo/train.parquet
test -f /data/grpo/validation.parquet
w8-biayn cpp harness preflight --image "{options.sandbox_image}" --cpu "{options.sandbox_cpu}"
{train_command}
'
"""
        + training_artifact_upload_script()
    )


def eval_run_script(options: RenderOptions) -> LiteralStr:
    max_tasks_arg = f"--max-tasks {options.eval_max_tasks}" if options.eval_max_tasks else ""
    return LiteralStr(
        training_prelude(options)
        + f"""export W8_EVAL_OUTPUT_DIR="/tmp/w8-cpp-eval-{options.run_id or 'manual'}"
export W8_EVAL_MODEL="$W8_BIAYN_MODEL_PATH"
rm -rf "$W8_EVAL_OUTPUT_DIR"
mkdir -p "$W8_EVAL_OUTPUT_DIR"
if [[ "$W8_EVAL_MODEL" == gs://* ]]; then
  echo "eval GCS model export was not staged by training_prelude: $W8_EVAL_MODEL" >&2
  exit 2
fi
docker run --rm --gpus all --network host --shm-size=32g \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -v /tmp:/tmp \\
  -v "$PWD":/workspace \\
  -v "$HOME/.cache/w8-biayn":/root/.cache/w8-biayn \\
  -v "$W8_DATA_DIR":/data \\
  -v "$W8_ARTIFACT_DIR":/artifacts \\
  -v /tmp/w8-gcp-service-account.json:/tmp/w8-gcp-service-account.json:ro \\
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json \\
  -e CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=/tmp/w8-gcp-service-account.json \\
  -e CLOUDSDK_CORE_PROJECT="$CLOUDSDK_CORE_PROJECT" \\
  -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \\
  -e GCLOUD_PROJECT="$GCLOUD_PROJECT" \\
  -e CLOUDSDK_CORE_DISABLE_PROMPTS="$CLOUDSDK_CORE_DISABLE_PROMPTS" \\
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
uv venv --clear --python 3.10 --seed /tmp/w8-eval
source /tmp/w8-eval/bin/activate
uv pip install -e /workspace
uv pip install "vllm=={DEFAULT_CPP_EVAL_VLLM_VERSION}" "transformers=={DEFAULT_CPP_EVAL_TRANSFORMERS_VERSION}" --extra-index-url https://download.pytorch.org/whl/cu124
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
            "memory": options.effective_memory,
            "disk_size": options.effective_disk_size,
        },
        "num_nodes": options.num_nodes,
        "workdir": ".",
        "file_mounts": {
            "/tmp/w8-gcp-service-account.json": options.credentials_path,
        },
        "envs": {
            "W8_BIAYN_PIPELINE": options.pipeline,
            "W8_BIAYN_ARTIFACT_BUCKET": options.artifact_bucket,
            "W8_BIAYN_MODEL": options.effective_model,
            "W8_BIAYN_DATA_GCS_PREFIX": options.data_gcs_prefix,
            "W8_BIAYN_RUN_ID": options.run_id or "",
            "W8_BIAYN_TOTAL_GPU_COUNT": str(options.total_gpu_count),
            "W8_BIAYN_EFFECTIVE_SAMPLES_PER_STEP": str(options.effective_samples_per_step),
            "W8_BIAYN_SAMPLES_PER_GPU_PER_STEP": (
                "" if options.samples_per_gpu_per_step is None else f"{options.samples_per_gpu_per_step:.6g}"
            ),
            "W8_BIAYN_ALLOW_LOW_MULTINODE_UTILIZATION": str(options.allow_low_multinode_utilization).lower(),
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/w8-gcp-service-account.json",
            "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": "/tmp/w8-gcp-service-account.json",
            "CLOUDSDK_CORE_PROJECT": options.project_id,
            "GOOGLE_CLOUD_PROJECT": options.project_id,
            "GCLOUD_PROJECT": options.project_id,
            "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
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
