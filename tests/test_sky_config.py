from __future__ import annotations

import pytest
import yaml

from w8_biayn.constants import DEFAULT_GPU_CONTAINER_IMAGE
from w8_biayn.harbor.tasks import DEFAULT_HARBOR_TASK_IDS
from w8_biayn.sky_config import (
    RenderOptions,
    accelerator_count,
    is_private_runtime_url,
    render_sky_yaml,
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
    assert not any("SKYPILOT_NUM_GPUS_PER_NODE" in item for item in overrides)
    assert any("Qwen/Qwen1.5-MoE-A2.7B-Chat" in item for item in overrides)


def test_accelerator_count_parses_skypilot_accelerator_request():
    assert accelerator_count("A100:4") == 4
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
    assert "uv sync --active --extra megatron --extra gcp" in config["run"]
    assert "cd /workspace" in config["run"]
    assert "w8-biayn harbor prepare-data" in config["run"]
    assert "w8_biayn.integrations.skyrl_harbor_main" in config["run"]
    assert "TINKER_API_KEY" not in config["run"]
    assert "environment.env_class=harbor-domdiff" in config["run"]
    assert "generator.inference_engine.enable_return_routed_experts=true" in config["run"]
    assert "SKYPILOT_NUM_GPUS_PER_NODE" not in config["run"]
    assert "trainer.placement.policy_num_gpus_per_node=4" in config["run"]
    assert "generator.inference_engine.tensor_parallel_size=4" in config["run"]
    assert "generator.inference_engine.expert_parallel_size=4" in config["run"]
    assert "generator.inference_engine.distributed_executor_backend=mp" in config["run"]
    assert "trainer.strategy=megatron" in config["run"]
    assert "trainer.algorithm.use_kl_loss=false" in config["run"]
    assert "trainer.policy.megatron_config.tensor_model_parallel_size=2" in config["run"]
    assert "trainer.policy.megatron_config.expert_model_parallel_size=4" in config["run"]
    assert "trainer.use_sample_packing=true" in config["run"]
    assert DEFAULT_HARBOR_TASK_IDS[0] in config["run"]
    assert DEFAULT_HARBOR_TASK_IDS[1] not in config["run"]
