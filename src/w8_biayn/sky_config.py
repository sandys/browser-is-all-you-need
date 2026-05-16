"""Render SkyPilot YAML for BrowserGym/SkyRL runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .constants import DEFAULT_CREDENTIALS_PATH, DEFAULT_DOMDIFF_IMAGE, SKYRL_PIN, SKYRL_REPO
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


def setup_script(options: RenderOptions) -> LiteralStr:
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


def render_sky_yaml(options: RenderOptions) -> str:
    secrets: dict[str, Any] = {}
    if options.wandb_secret:
        secrets["WANDB_API_KEY"] = None

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
