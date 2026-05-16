from __future__ import annotations

import yaml

from w8_biayn.sky_config import RenderOptions, render_sky_yaml, skyrl_overrides


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
