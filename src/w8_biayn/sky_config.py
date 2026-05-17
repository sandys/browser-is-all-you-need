"""Render SkyPilot YAML for BrowserGym/SkyRL runs."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

from .constants import (
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_DOMDIFF_IMAGE,
    DEFAULT_GPU_CONTAINER_IMAGE,
    RLLM_PIN,
    RLLM_REPO,
    SKYRL_PIN,
    SKYRL_REPO,
)
from .harbor.tasks import DEFAULT_HARBOR_TASK_IDS, DEFAULT_HARBOR_TASK_ROOT
from .secrets import default_bucket_for_project

Pipeline = Literal["miniwob", "webarena", "r3"]


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
    accelerators: str = "A100:4"
    num_nodes: int = 1
    cluster_name: str | None = None
    logger: str = "console"
    wandb_secret: bool = False
    webarena_archives_gcs: str | None = None
    chromiumrl_url: str | None = None
    cdp_url: str | None = None
    domdiff_reward_image: str = DEFAULT_DOMDIFF_IMAGE
    benchmark: str | None = None
    harbor_task_ids: tuple[str, ...] = DEFAULT_HARBOR_TASK_IDS
    harbor_oracle: bool = True
    gpu_container_image: str = DEFAULT_GPU_CONTAINER_IMAGE
    tinker_secret: bool = False

    @property
    def artifact_bucket(self) -> str:
        return self.bucket or default_bucket_for_project(self.project_id)

    @property
    def name(self) -> str:
        return self.cluster_name or f"w8-biayn-{self.pipeline}"


def remote_data_dir(pipeline: Pipeline) -> str:
    benchmark = "miniwob" if pipeline in ("miniwob", "r3") else "webarena"
    return f"$HOME/data/w8-biayn/{benchmark}"


def model_for_pipeline(pipeline: Pipeline) -> str:
    if pipeline == "r3":
        return "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    if pipeline == "webarena":
        return "Qwen/Qwen3-8B"
    return "Qwen/Qwen2.5-1.5B-Instruct"


def benchmark_for_pipeline(pipeline: Pipeline) -> str:
    return "webarena" if pipeline == "webarena" else "miniwob"


def is_harbor_domdiff_benchmark(options: RenderOptions) -> bool:
    return options.pipeline == "r3" and options.benchmark == "harbor-domdiff-browser-swe"


def is_private_runtime_url(url: str | None) -> bool:
    """Return True when a URL cannot be reached from a remote SkyPilot VM."""

    if not url:
        return False
    host = (urlparse(url).hostname or "").strip().lower()
    if not host:
        return False
    if host in {"localhost", "host.docker.internal", "docker.for.mac.localhost"}:
        return True
    if host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
        or address.is_reserved
    )


def validate_remote_runtime_urls(options: RenderOptions) -> None:
    """Reject DOMDiff URLs that would only work on the operator workstation."""

    for label, url in (
        ("CHROMIUMRL_URL", options.chromiumrl_url),
        ("CDP_URL", options.cdp_url),
    ):
        if is_private_runtime_url(url):
            raise ValueError(
                f"{label} must be reachable from the GCP/SkyPilot trainer, not a local/private URL: {url}. "
                "Use `w8-biayn domdiff local up` and pass the Cloudflare tunnel URL."
            )


def setup_script(options: RenderOptions) -> LiteralStr:
    if is_harbor_domdiff_benchmark(options):
        return harbor_setup_script(options)
    benchmark = benchmark_for_pipeline(options.pipeline)
    extra_browser_pkg = "browsergym-webarena browsergym" if benchmark == "webarena" else "browsergym-miniwob"
    maybe_playwright = "python -m playwright install chromium || true" if benchmark == "webarena" else "true"
    webarena_setup = webarena_provision_script(options) if benchmark == "webarena" else "true"
    return LiteralStr(
        f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
export W8_WEBARENA_ARCHIVES_GCS="{options.webarena_archives_gcs or ""}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
W8_WORKDIR="$PWD"
SKYRL_DIR="$HOME/.cache/w8-biayn/upstreams/SkyRL"
mkdir -p "$(dirname "$SKYRL_DIR")"
if [ ! -d "$SKYRL_DIR/.git" ]; then
  git clone {SKYRL_REPO} "$SKYRL_DIR"
fi
git -C "$SKYRL_DIR" fetch origin {SKYRL_PIN} --depth 1 || git -C "$SKYRL_DIR" fetch --all --tags
git -C "$SKYRL_DIR" checkout {SKYRL_PIN}
cd "$SKYRL_DIR"
uv venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra fsdp --extra gcp
uv pip install -e "$W8_WORKDIR"
uv pip install pandas pyarrow gymnasium {extra_browser_pkg}
{maybe_playwright}
{webarena_setup}
w8-biayn data prepare {benchmark} --out "{remote_data_dir(options.pipeline)}"
"""
    )


def harbor_setup_script(_options: RenderOptions) -> LiteralStr:
    return LiteralStr(
        f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
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
docker version
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  sudo install -m 0755 /tmp/cloudflared /usr/local/bin/cloudflared
fi
W8_CACHE="$HOME/.cache/w8-biayn/upstreams"
mkdir -p "$W8_CACHE"
if [ ! -d "$W8_CACHE/rllm/.git" ]; then
  git clone {RLLM_REPO} "$W8_CACHE/rllm"
fi
git -C "$W8_CACHE/rllm" fetch origin {RLLM_PIN} --depth 1 || git -C "$W8_CACHE/rllm" fetch --all --tags
git -C "$W8_CACHE/rllm" checkout {RLLM_PIN}
if [ ! -d "$W8_CACHE/SkyRL/.git" ]; then
  git clone {SKYRL_REPO} "$W8_CACHE/SkyRL"
fi
git -C "$W8_CACHE/SkyRL" fetch origin {SKYRL_PIN} --depth 1 || git -C "$W8_CACHE/SkyRL" fetch --all --tags
git -C "$W8_CACHE/SkyRL" checkout {SKYRL_PIN}
"""
    )


def webarena_env_exports() -> str:
    return """BASE_URL="${W8_WEBARENA_BASE_URL:-http://127.0.0.1}"
export WA_SHOPPING="${WA_SHOPPING:-$BASE_URL:8082/}"
export WA_SHOPPING_ADMIN="${WA_SHOPPING_ADMIN:-$BASE_URL:8083/admin}"
export WA_REDDIT="${WA_REDDIT:-$BASE_URL:8080}"
export WA_GITLAB="${WA_GITLAB:-$BASE_URL:9001}"
export WA_WIKIPEDIA="${WA_WIKIPEDIA:-$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing}"
export WA_MAP="${WA_MAP:-$BASE_URL:443}"
export WA_HOMEPAGE="${WA_HOMEPAGE:-$BASE_URL:80}"
export WA_FULL_RESET="${WA_FULL_RESET:-$BASE_URL:7565}"
"""


def webarena_provision_script(_options: RenderOptions) -> str:
    return f"""{webarena_env_exports()}
if [ -n "$W8_WEBARENA_ARCHIVES_GCS" ]; then
  WEBARENA_ASSETS="$HOME/webarena-assets"
  WEBARENA_SETUP="$HOME/.cache/w8-biayn/webarena-setup"
  mkdir -p "$WEBARENA_ASSETS" "$(dirname "$WEBARENA_SETUP")"
  gcloud storage cp -r "$W8_WEBARENA_ARCHIVES_GCS/*" "$WEBARENA_ASSETS/" || gsutil -m cp -r "$W8_WEBARENA_ARCHIVES_GCS/*" "$WEBARENA_ASSETS/"
  if [ ! -d "$WEBARENA_SETUP/.git" ]; then
    git clone https://github.com/gasse/webarena-setup.git "$WEBARENA_SETUP"
  fi
  cd "$WEBARENA_SETUP/webarena"
  cp -R "$WEBARENA_ASSETS"/. .
  python - <<'PY'
from pathlib import Path
path = Path("00_vars.sh")
text = path.read_text()
text = text.replace('PUBLIC_HOSTNAME="YOUR_HOSTNAME_HERE"', 'PUBLIC_HOSTNAME="127.0.0.1"')
text = text.replace('ARCHIVES_LOCATION="./"', f'ARCHIVES_LOCATION="{Path.home() / "webarena-assets"}"')
path.write_text(text)
PY
  sudo bash 01_docker_load_images.sh
  sudo bash 02_docker_remove_containers.sh || true
  sudo bash 03_docker_create_containers.sh
  sudo bash 04_docker_start_containers.sh
  sudo bash 05_docker_patch_containers.sh
  nohup sudo bash 06_serve_homepage.sh > "$HOME/webarena-homepage.log" 2>&1 &
  nohup sudo bash 07_serve_reset.sh > "$HOME/webarena-reset.log" 2>&1 &
else
  echo "W8_WEBARENA_ARCHIVES_GCS is not set; WebArena services will not be provisioned."
  echo "MiniWoB works without this. WebArena training requires WebArena services or WA_* URLs."
fi"""


def skyrl_overrides(options: RenderOptions) -> list[str]:
    pipeline = options.pipeline
    data_dir = remote_data_dir(pipeline)
    num_gpus = "$SKYPILOT_NUM_GPUS_PER_NODE"
    model = model_for_pipeline(pipeline)
    max_turns = "6" if pipeline == "miniwob" else "12"
    train_batch_size = "16" if pipeline == "miniwob" else "8"
    mini_batch = train_batch_size
    micro_batch = "4" if pipeline == "miniwob" else "1"
    max_prompt = "4096" if pipeline == "miniwob" else "8192"
    max_gen = "1024" if pipeline == "miniwob" else "2048"

    overrides = [
        f"data.train_data=\"['{data_dir}/train.parquet']\"",
        f"data.val_data=\"['{data_dir}/validation.parquet']\"",
        'trainer.algorithm.advantage_estimator="grpo"',
        f'trainer.policy.model.path="{model}"',
        "trainer.placement.colocate_all=true",
        "trainer.strategy=fsdp",
        f"trainer.placement.policy_num_gpus_per_node={num_gpus}",
        f"trainer.placement.ref_num_gpus_per_node={num_gpus}",
        f"generator.inference_engine.num_engines={num_gpus}",
        "generator.inference_engine.tensor_parallel_size=1",
        "trainer.epochs=3",
        "trainer.eval_batch_size=16",
        "trainer.eval_before_train=true",
        "trainer.eval_interval=5",
        "trainer.update_epochs_per_batch=1",
        f"trainer.train_batch_size={train_batch_size}",
        f"trainer.policy_mini_batch_size={mini_batch}",
        f"trainer.micro_forward_batch_size_per_gpu={micro_batch}",
        f"trainer.micro_train_batch_size_per_gpu={micro_batch}",
        "trainer.ckpt_interval=5",
        f"trainer.max_prompt_length={max_prompt}",
        f"generator.sampling_params.max_generate_length={max_gen}",
        "generator.sampling_params.temperature=0.7",
        "generator.sampling_params.top_p=0.95",
        "generator.sampling_params.stop='[\"</action>\"]'",
        "generator.eval_sampling_params.stop='[\"</action>\"]'",
        f"generator.eval_sampling_params.max_generate_length={max_gen}",
        "trainer.policy.optimizer_config.lr=1.0e-6",
        "trainer.algorithm.use_kl_loss=true",
        f"generator.max_turns={max_turns}",
        "generator.inference_engine.backend=vllm",
        "generator.inference_engine.run_engines_locally=true",
        "generator.inference_engine.weight_sync_backend=nccl",
        "generator.inference_engine.async_engine=true",
        "generator.batched=false",
        "environment.env_class=browsergym",
        "generator.use_conversation_multi_turn=true",
        "generator.n_samples_per_prompt=4",
        "generator.inference_engine.gpu_memory_utilization=0.8",
        f'trainer.logger="{options.logger}"',
        f'trainer.project_name="w8-biayn-{pipeline}"',
        f'trainer.run_name="{pipeline}"',
        "trainer.resume_mode=latest",
        f'trainer.ckpt_path="$HOME/ckpts/w8-biayn/{pipeline}"',
        f'trainer.export_path="$HOME/exports/w8-biayn/{pipeline}"',
        "trainer.dump_data_batch=true",
    ]
    if pipeline == "r3":
        overrides.extend(
            [
                "generator.inference_engine.enable_return_routed_experts=true",
                "generator.inference_engine.num_engines=1",
                f"generator.inference_engine.tensor_parallel_size={num_gpus}",
                f"generator.inference_engine.expert_parallel_size={num_gpus}",
                "generator.inference_engine.data_parallel_size=1",
                "trainer.policy.megatron_config.moe_enable_routing_replay=true",
                "trainer.ref.megatron_config.moe_enable_routing_replay=true",
                'trainer.project_name="w8-biayn-r3"',
                'trainer.run_name="miniwob-qwen15-moe-routing-replay"',
            ]
        )
    return overrides


def run_script(options: RenderOptions) -> LiteralStr:
    if is_harbor_domdiff_benchmark(options):
        return harbor_run_script(options)
    overrides = " \\\n  ".join(skyrl_overrides(options))
    webarena_exports = webarena_env_exports() if options.pipeline == "webarena" else ""
    domdiff_enabled = "1" if options.chromiumrl_url or options.cdp_url else "0"
    return LiteralStr(
        f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
export ARTIFACT_BUCKET="{options.artifact_bucket}"
export W8_WEBARENA_ARCHIVES_GCS="{options.webarena_archives_gcs or ""}"
export W8_BIAYN_DOMDIFF_ENABLED="{domdiff_enabled}"
export W8_BIAYN_DOMDIFF_REWARD_IMAGE="{options.domdiff_reward_image}"
export W8_BIAYN_BENCHMARK="{options.benchmark or ""}"
export CHROMIUMRL_URL="{options.chromiumrl_url or ""}"
export CHROMIUMRL_API_URL="{options.chromiumrl_url or ""}"
export CDP_URL="{options.cdp_url or ""}"
{webarena_exports}
cd "$HOME/.cache/w8-biayn/upstreams/SkyRL"
source .venv/bin/activate
python -m w8_biayn.integrations.skyrl_browsergym_main \\
  {overrides}
"""
    )


def harbor_run_script(options: RenderOptions) -> LiteralStr:
    task_ids = ",".join(options.harbor_task_ids or DEFAULT_HARBOR_TASK_IDS)
    task_ids_hydra = "[" + ",".join(options.harbor_task_ids or DEFAULT_HARBOR_TASK_IDS) + "]"
    oracle_value = "true" if options.harbor_oracle else "false"
    return LiteralStr(
        f"""set -euxo pipefail
export GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json
export ARTIFACT_BUCKET="{options.artifact_bucket}"
export W8_BIAYN_BENCHMARK="{options.benchmark or ""}"
export W8_HARBOR_TASK_ROOT="{DEFAULT_HARBOR_TASK_ROOT}"
export W8_HARBOR_TASK_IDS="{task_ids}"
export CHROMIUMRL_URL="{options.chromiumrl_url or ""}"
export CHROMIUMRL_API_URL="{options.chromiumrl_url or ""}"
export W8_GPU_CONTAINER_IMAGE="{options.gpu_container_image}"
if [ -z "${{TINKER_API_KEY:-}}" ]; then
  echo "TINKER_API_KEY is required for the Harbor rLLM/Tinker R3 smoke." >&2
  exit 2
fi
docker pull "$W8_GPU_CONTAINER_IMAGE"
docker run --rm --gpus all --network host \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -v "$PWD":/workspace \\
  -v "$HOME/.cache/w8-biayn":/root/.cache/w8-biayn \\
  -v /tmp/w8-gcp-service-account.json:/tmp/w8-gcp-service-account.json:ro \\
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/w8-gcp-service-account.json \\
  -e TINKER_API_KEY \\
  -e CHROMIUMRL_URL \\
  -e CHROMIUMRL_API_URL \\
  -e W8_REPO_ROOT=/workspace \\
  -e W8_HARBOR_TASK_ROOT=/workspace/{DEFAULT_HARBOR_TASK_ROOT} \\
  -e W8_HARBOR_TASK_IDS \\
  -w /workspace \\
  "$W8_GPU_CONTAINER_IMAGE" bash -lc '
set -euxo pipefail
if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi
if ! command -v curl >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y curl git ca-certificates
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"
RLLM_DIR="$HOME/.cache/w8-biayn/upstreams/rllm"
SKYRL_DIR="$HOME/.cache/w8-biayn/upstreams/SkyRL"
uv venv --python 3.12 --seed "$HOME/.cache/w8-biayn/venvs/harbor-r3"
source "$HOME/.cache/w8-biayn/venvs/harbor-r3/bin/activate"
uv pip install -e /workspace
uv pip install -e "$RLLM_DIR[harbor,tinker]"
uv pip install docker openai hydra-core
python - <<PY
import subprocess
subprocess.run(["nvidia-smi"], check=False)
PY
python -m w8_biayn.integrations.harbor_r3_main \\
  rllm/backend=tinker \\
  model.name=Qwen/Qwen3-4B-Instruct-2507 \\
  model.lora_rank=16 \\
  training.group_size=2 \\
  validation.group_size=1 \\
  data.train_batch_size=1 \\
  data.val_batch_size=1 \\
  data.max_prompt_length=32768 \\
  data.max_response_length=8192 \\
  rllm.workflow.n_parallel_tasks=1 \\
  rllm.workflow.retry_limit=0 \\
  rllm.trainer.total_batches=1 \\
  rllm.trainer.total_epochs=1 \\
  rllm.trainer.val_before_train=false \\
  rllm.trainer.test_freq=0 \\
  rllm.trainer.save_freq=-1 \\
  rllm.trainer.logger="[console]" \\
  rllm.trainer.project_name=w8-biayn-r3 \\
  rllm.trainer.experiment_name=harbor-domdiff-smoke \\
  w8.harbor.dataset_name=w8_harbor_domdiff_smoke \\
  w8.harbor.task_root=/workspace/{DEFAULT_HARBOR_TASK_ROOT} \\
  w8.harbor.task_ids="{task_ids_hydra}" \\
  w8.harbor.max_samples=2 \\
  w8.harbor.chromiumrl_url="$CHROMIUMRL_URL" \\
  w8.harbor.oracle={oracle_value} \\
  w8.harbor.keep_containers=false
'
"""
    )


def render_sky_yaml(options: RenderOptions) -> str:
    validate_remote_runtime_urls(options)
    secrets: dict[str, Any] = {}
    if options.wandb_secret:
        secrets["WANDB_API_KEY"] = None
    if options.tinker_secret:
        secrets["TINKER_API_KEY"] = None

    config: dict[str, Any] = {
        "name": options.name,
        "resources": {
            "infra": "gcp",
            "accelerators": options.accelerators,
            "memory": "128+",
            "ports": 6479,
        },
        "num_nodes": options.num_nodes,
        "workdir": ".",
        "file_mounts": {
            "/tmp/w8-gcp-service-account.json": options.credentials_path,
        },
        "envs": {
            "W8_BIAYN_PIPELINE": options.pipeline,
            "W8_BIAYN_ARTIFACT_BUCKET": options.artifact_bucket,
        },
        "setup": setup_script(options),
        "run": run_script(options),
    }
    if secrets:
        config["secrets"] = secrets
    if options.webarena_archives_gcs:
        config["envs"]["W8_WEBARENA_ARCHIVES_GCS"] = options.webarena_archives_gcs
    if options.benchmark:
        config["envs"]["W8_BIAYN_BENCHMARK"] = options.benchmark
    if options.chromiumrl_url or options.cdp_url:
        config["envs"]["W8_BIAYN_DOMDIFF_ENABLED"] = "1"
        config["envs"]["W8_BIAYN_DOMDIFF_REWARD_IMAGE"] = options.domdiff_reward_image
        if options.chromiumrl_url:
            config["envs"]["CHROMIUMRL_URL"] = options.chromiumrl_url
            config["envs"]["CHROMIUMRL_API_URL"] = options.chromiumrl_url
        if options.cdp_url:
            config["envs"]["CDP_URL"] = options.cdp_url
    return yaml.safe_dump(config, sort_keys=False)


def write_sky_yaml(options: RenderOptions, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_sky_yaml(options), encoding="utf-8")
    return output_path
