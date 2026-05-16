from __future__ import annotations

import pytest
import yaml

from w8_biayn.sky_config import RenderOptions, render_sky_yaml, is_private_runtime_url, skyrl_overrides


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
    assert any("Qwen/Qwen1.5-MoE-A2.7B-Chat" in item for item in overrides)


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
