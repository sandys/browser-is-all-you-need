from __future__ import annotations

import pytest
import yaml

from w8_biayn.constants import DEFAULT_GPU_CONTAINER_IMAGE, DEFAULT_HARBOR_R3_ACCELERATORS
from w8_biayn.harbor.tasks import DEFAULT_HARBOR_TASK_IDS
from w8_biayn.sky_config import (
    RenderOptions,
    accelerator_count,
    harbor_run_script,
    is_private_runtime_url,
    render_sky_yaml,
    run_script,
    skyrl_overrides,
)


def test_render_miniwob_sky_yaml_contains_gcp_and_mount():
    text = render_sky_yaml(
        RenderOptions(
            pipeline="miniwob",
            project_id="proj",
            credentials_path=".gcp-service-account.json",
        )
    )
    config = yaml.safe_load(text)

    assert config["resources"]["infra"] == "gcp"
    assert config["resources"]["accelerators"] == "A100:4"
    assert config["file_mounts"]["/tmp/w8-gcp-service-account.json"] == ".gcp-service-account.json"
    assert "w8-biayn data prepare miniwob" in config["setup"]
    assert "w8_biayn.integrations.skyrl_browsergym_main" in config["run"]


def test_r3_overrides_enable_routing_replay():
    overrides = skyrl_overrides(RenderOptions(pipeline="r3", project_id="proj"))

    assert "generator.inference_engine.enable_return_routed_experts=true" in overrides
    assert "trainer.policy.megatron_config.moe_enable_routing_replay=true" in overrides
    assert "trainer.placement.policy_num_gpus_per_node=4" in overrides
    assert "generator.inference_engine.tensor_parallel_size=4" in overrides
    assert "generator.inference_engine.distributed_executor_backend=mp" in overrides
    assert "trainer.strategy=megatron" in overrides
    assert "trainer.algorithm.use_kl_loss=false" in overrides
    assert "trainer.policy.megatron_config.tensor_model_parallel_size=2" in overrides
    assert "trainer.policy.megatron_config.expert_model_parallel_size=4" in overrides
    assert "trainer.use_sample_packing=true" in overrides
    assert "trainer.flash_attn=false" in overrides
    assert not any("SKYPILOT_NUM_GPUS_PER_NODE" in item for item in overrides)
    assert any("moonshotai/Moonlight-16B-A3B-Instruct" in item for item in overrides)


def test_accelerator_count_parses_skypilot_accelerator_request():
    assert accelerator_count("A100:4") == 4
    assert accelerator_count(DEFAULT_HARBOR_R3_ACCELERATORS) == 8
    assert accelerator_count("L4") == 1
    assert accelerator_count("A100:abc") == 1
    assert accelerator_count("") == 1


def test_render_webarena_includes_service_provision_hook():
    text = render_sky_yaml(
        RenderOptions(
            pipeline="webarena",
            project_id="proj",
            webarena_archives_gcs="gs://proj-w8-biayn/webarena",
        )
    )
    config = yaml.safe_load(text)

    assert config["envs"]["W8_WEBARENA_ARCHIVES_GCS"] == "gs://proj-w8-biayn/webarena"
    assert "webarena-setup" in config["setup"]
    assert "WA_SHOPPING" in config["run"]


def test_render_r3_includes_domdiff_reward_envs():
    text = render_sky_yaml(
        RenderOptions(
            pipeline="r3",
            project_id="proj",
            chromiumrl_url="https://reward.trycloudflare.com",
            cdp_url="wss://cdp.trycloudflare.com",
            benchmark="webvoyager-domdiff-heldout",
        )
    )
    config = yaml.safe_load(text)

    assert config["envs"]["W8_BIAYN_DOMDIFF_ENABLED"] == "1"
    assert config["envs"]["CHROMIUMRL_URL"] == "https://reward.trycloudflare.com"
    assert config["envs"]["CDP_URL"] == "wss://cdp.trycloudflare.com"
    assert config["envs"]["W8_BIAYN_BENCHMARK"] == "webvoyager-domdiff-heldout"
    assert "uv sync --extra megatron --extra gcp" in config["setup"]
    assert "trainer.strategy=megatron" in config["run"]
    assert 'export CHROMIUMRL_API_URL="https://reward.trycloudflare.com"' in config["run"]


def test_private_domdiff_urls_are_rejected_for_remote_runtime():
    assert is_private_runtime_url("http://127.0.0.1:8080")
    assert is_private_runtime_url("ws://192.168.1.10:9224")
    assert not is_private_runtime_url("https://reward.trycloudflare.com")

    with pytest.raises(ValueError, match="local/private URL"):
        render_sky_yaml(
            RenderOptions(
                pipeline="r3",
                project_id="proj",
                chromiumrl_url="http://localhost:8080",
            )
        )


def test_render_harbor_r3_uses_gpu_container_and_skyrl_entrypoint():
    text = render_sky_yaml(
        RenderOptions(
            pipeline="r3",
            project_id="proj",
            benchmark="harbor-domdiff-browser-swe",
            chromiumrl_url="https://reward.trycloudflare.com",
            harbor_task_ids=(DEFAULT_HARBOR_TASK_IDS[0],),
        )
    )
    config = yaml.safe_load(text)

    assert "secrets" not in config
    assert config["envs"]["CHROMIUMRL_URL"] == "https://reward.trycloudflare.com"
    assert "CDP_URL" not in config["envs"]
    assert "rllm-org/rllm" not in config["setup"]
    assert "git clone https://github.com/NovaSky-AI/SkyRL.git" in config["setup"]
    assert DEFAULT_GPU_CONTAINER_IMAGE in config["run"]
    assert "docker run --rm --gpus all --network host --shm-size=32g" in config["run"]
    assert 'if [ ! -x "$HARBOR_VENV/bin/python" ]; then' in config["run"]
    assert 'uv venv --python 3.12 --seed --clear "$HARBOR_VENV"' in config["run"]
    assert "uv sync --active --extra megatron --extra gcp" in config["run"]
    assert "cd /workspace" in config["run"]
    assert "w8-biayn harbor prepare-data" in config["run"]
    assert "w8_biayn.integrations.skyrl_harbor_main" in config["run"]
    assert "TINKER_API_KEY" not in config["run"]
    assert "environment.env_class=harbor-domdiff" in config["run"]
    assert "generator.inference_engine.enable_return_routed_experts=true" in config["run"]
    assert "SKYPILOT_NUM_GPUS_PER_NODE" not in config["run"]
    assert config["resources"]["accelerators"] == "A100:4"
    assert "trainer.placement.policy_num_gpus_per_node=4" in config["run"]
    assert "generator.inference_engine.tensor_parallel_size=4" in config["run"]
    assert "generator.inference_engine.expert_parallel_size=4" in config["run"]
    assert "generator.inference_engine.distributed_executor_backend=mp" in config["run"]
    assert "trainer.strategy=megatron" in config["run"]
    assert "trainer.algorithm.use_kl_loss=false" in config["run"]
    assert "trainer.policy.megatron_config.tensor_model_parallel_size=2" in config["run"]
    assert "trainer.policy.megatron_config.expert_model_parallel_size=4" in config["run"]
    assert "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true" in config["run"]
    assert "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0" in config["run"]
    assert "trainer.use_sample_packing=true" in config["run"]
    assert DEFAULT_HARBOR_TASK_IDS[0] in config["run"]
    assert DEFAULT_HARBOR_TASK_IDS[1] not in config["run"]


def test_render_harbor_r3_h100_eight_gpu_topology():
    text = render_sky_yaml(
        RenderOptions(
            pipeline="r3",
            project_id="proj",
            benchmark="harbor-domdiff-browser-swe",
            accelerators=DEFAULT_HARBOR_R3_ACCELERATORS,
            chromiumrl_url="https://reward.trycloudflare.com",
        )
    )
    config = yaml.safe_load(text)

    assert config["resources"]["accelerators"] == DEFAULT_HARBOR_R3_ACCELERATORS
    assert "trainer.placement.policy_num_gpus_per_node=8" in config["run"]
    assert "generator.inference_engine.tensor_parallel_size=8" in config["run"]
    assert "generator.inference_engine.expert_parallel_size=8" in config["run"]
    assert "trainer.policy.megatron_config.tensor_model_parallel_size=4" in config["run"]
    assert "trainer.policy.megatron_config.expert_model_parallel_size=8" in config["run"]
    assert "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=false" in config["run"]
    assert "trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=0.0" in config["run"]


def _harbor_opts(profile: str) -> RenderOptions:
    return RenderOptions(
        pipeline="r3",
        project_id="proj",
        benchmark="harbor-domdiff-browser-swe",
        chromiumrl_url="https://reward.trycloudflare.com",
        optimization_profile=profile,
    )


def test_baseline_profile_is_the_default_and_injects_no_kernels():
    default_render = harbor_run_script(_harbor_opts("baseline"))
    # The default (no profile arg) must equal the explicit baseline render.
    omitted = harbor_run_script(
        RenderOptions(
            pipeline="r3",
            project_id="proj",
            benchmark="harbor-domdiff-browser-swe",
            chromiumrl_url="https://reward.trycloudflare.com",
        )
    )
    assert default_render == omitted
    assert "W8_BIAYN_KERNELS" not in default_render


def test_kernel_lab_profile_is_exactly_baseline_plus_injected_fragments():
    base = harbor_run_script(_harbor_opts("baseline"))
    lab = harbor_run_script(_harbor_opts("a100-kernel-lab"))

    assert 'export W8_BIAYN_KERNELS="logprob,entropy,ppo"' in lab
    assert "-e W8_BIAYN_KERNELS \\" in lab
    # Additive-only guarantee: remove the two injected fragments and the rest must be
    # byte-for-byte identical to the baseline render.
    reconstructed = lab.replace(
        'export W8_BIAYN_KERNELS="logprob,entropy,ppo"\n', ""
    ).replace("-e W8_BIAYN_KERNELS \\\n  ", "")
    assert reconstructed == base


def test_kernel_lab_profile_injects_export_in_generic_r3_run_script():
    base = run_script(RenderOptions(pipeline="r3", project_id="proj"))
    lab = run_script(RenderOptions(pipeline="r3", project_id="proj", optimization_profile="a100-kernel-lab"))

    assert "W8_BIAYN_KERNELS" not in base
    assert 'export W8_BIAYN_KERNELS="logprob,entropy,ppo"' in lab
    reconstructed = lab.replace('export W8_BIAYN_KERNELS="logprob,entropy,ppo"\n', "")
    assert reconstructed == base


def test_a100_safe_profile_leaves_numerics_unchanged():
    safe = harbor_run_script(_harbor_opts("a100-safe"))
    assert "W8_BIAYN_KERNELS" not in safe
    assert safe == harbor_run_script(_harbor_opts("baseline"))


def test_render_kernel_lab_yaml_provisions_single_a100():
    from w8_biayn.sky_config import render_kernel_lab_yaml

    text = render_kernel_lab_yaml(project_id="proj", kernel="mla")
    config = yaml.safe_load(text)

    assert config["name"] == "w8-biayn-kernel-lab-mla"
    assert config["resources"]["accelerators"] == "A100:1"
    assert config["resources"]["infra"] == "gcp"
    assert config["resources"]["labels"]["w8-biayn"] == "kernel-lab"
    assert config["resources"]["labels"]["w8-biayn-kernel"] == "mla"
    assert "git clone https://github.com/NovaSky-AI/SkyRL.git" in config["setup"]
    assert "uv sync --extra megatron --extra gcp" in config["setup"]
    assert "nvidia-smi" in config["run"]
    assert "w8-biayn kernels lab" in config["run"]
    assert "--kernel mla" in config["run"]
