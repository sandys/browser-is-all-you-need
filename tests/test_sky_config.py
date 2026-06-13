from __future__ import annotations

import yaml

from w8_biayn.sky_config import RenderOptions, render_sky_yaml


def test_render_cpp_smoke_yaml_contains_gcp_model_and_upstreams():
    rendered = render_sky_yaml(RenderOptions(pipeline="cpp-smoke", project_id="proj"))
    config = yaml.safe_load(rendered)

    assert config["name"] == "w8-biayn-cpp-smoke"
    assert config["resources"]["infra"] == "gcp"
    assert config["resources"]["accelerators"] == "H100:8"
    assert config["file_mounts"]["/tmp/w8-gcp-service-account.json"] == ".gcp-service-account.json"
    assert config["envs"]["W8_BIAYN_MODEL"] == "zai-org/GLM-5.1"
    assert "NovaSky-AI/SkyRL" in config["setup"]
    assert "rllm-org/rllm" in config["setup"]
    assert "vllm" in config["run"]
    assert "domdiff" not in rendered.lower()
    assert "harbor" not in rendered.lower()


def test_render_cpp_training_yaml_uses_skyrl_entrypoints_and_a100_defaults():
    sft = yaml.safe_load(
        render_sky_yaml(RenderOptions(pipeline="cpp-sft", project_id="proj", accelerators="A100:8"))
    )
    grpo_text = render_sky_yaml(RenderOptions(pipeline="cpp-grpo", project_id="proj", accelerators="A100:8"))
    grpo = yaml.safe_load(grpo_text)

    assert sft["resources"]["accelerators"] == "A100:8"
    assert "python -m skyrl.train.main_sft" in sft["run"]
    assert "dataset_name=/data/sft" in sft["run"]
    assert "python -m w8_biayn.integrations.skyrl_cpp_perf_main" in grpo["run"]
    assert grpo["run"].index("uv sync --active --extra fsdp --extra gcp") < grpo["run"].index(
        "uv pip install --no-deps -e /workspace"
    )
    assert grpo["run"].index("uv pip install --no-deps -e /workspace") < grpo["run"].index(
        "python -m w8_biayn.integrations.skyrl_cpp_perf_main"
    )
    assert "environment.env_class=cpp-perf" in grpo["run"]
    assert 'docker pull "gcc:13"' in grpo["run"]
    assert "-v /var/run/docker.sock:/var/run/docker.sock" in grpo["run"]
    assert "-v /tmp:/tmp" in grpo["run"]
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io" in grpo["run"]
    assert "docker version" in grpo["run"]
    assert "trainer.logger=console" in grpo["run"]
    assert "trainer.ckpt_interval=-1" in grpo["run"]
    assert "trainer.hf_save_interval=-1" in grpo["run"]
    assert "trainer.placement.policy_num_gpus_per_node=8" in grpo["run"]
    assert "generator.inference_engine.num_engines=8" in grpo["run"]
    assert "generator.inference_engine.tensor_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.pipeline_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.data_parallel_size=1" in grpo["run"]
    assert "BasePPOExp" not in grpo_text
    assert "domdiff" not in grpo_text.lower()
    assert "harbor" not in grpo_text.lower()
    assert grpo["envs"]["W8_BIAYN_DATA_GCS_PREFIX"].endswith("/datasets/cpp-perf/cpp-perf-v1/skyrl")
