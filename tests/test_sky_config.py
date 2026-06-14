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
    assert "W8_BIAYN_MODEL_PATH" in grpo["run"]
    assert 'trainer.policy.model.path="$W8_BIAYN_MODEL_PATH"' in grpo["run"]
    assert "environment.env_class=cpp-perf" in grpo["run"]
    assert "-v /var/run/docker.sock:/var/run/docker.sock" in grpo["run"]
    assert "-v /tmp:/tmp" in grpo["run"]
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io" in grpo["run"]
    assert "docker version" in grpo["run"]
    assert "w8-biayn cpp harness preflight --image \"w8-biayn-cpp-perf:latest\" --cpu \"3\"" in grpo["run"]
    assert "trainer.logger=console" in grpo["run"]
    assert "trainer.ckpt_interval=-1" in grpo["run"]
    assert "trainer.hf_save_interval=-1" in grpo["run"]
    assert '-v "$W8_ARTIFACT_DIR":/artifacts' in grpo["run"]
    assert 'trainer.ckpt_path="$W8_CKPT_PATH"' in grpo["run"]
    assert 'trainer.export_path="$W8_EXPORT_PATH"' in grpo["run"]
    assert "gcloud storage cp --recursive \"$W8_ARTIFACT_DIR/exports\"" in grpo["run"]
    assert "trainer.epochs=1" in grpo["run"]
    assert "trainer.eval_interval=50" in grpo["run"]
    assert "trainer.placement.policy_num_gpus_per_node=8" in grpo["run"]
    assert "generator.inference_engine.num_engines=8" in grpo["run"]
    assert "generator.inference_engine.tensor_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.pipeline_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.data_parallel_size=1" in grpo["run"]
    assert "BasePPOExp" not in grpo_text
    assert "domdiff" not in grpo_text.lower()
    assert "harbor" not in grpo_text.lower()
    assert grpo["envs"]["W8_BIAYN_DATA_GCS_PREFIX"].endswith("/datasets/cpp-perf/cpp-perf-v1/skyrl")


def test_render_cpp_training_yaml_exposes_full_run_knobs():
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            train_epochs=3,
            eval_interval=25,
            ckpt_interval=50,
            hf_save_interval=100,
            max_ckpts_to_keep=2,
            ckpt_path="gs://bucket/ckpts",
            export_path="gs://bucket/exports",
            sandbox_image="w8-cpp:perf",
            sandbox_cpu="7",
        )
    )

    assert "w8-biayn cpp harness preflight --image \"w8-cpp:perf\" --cpu \"7\"" in rendered
    assert "trainer.epochs=3" in rendered
    assert "trainer.eval_interval=25" in rendered
    assert "trainer.ckpt_interval=50" in rendered
    assert "trainer.hf_save_interval=100" in rendered
    assert "trainer.max_ckpts_to_keep=2" in rendered
    assert 'export W8_CKPT_PATH="gs://bucket/ckpts"' in rendered
    assert 'export W8_EXPORT_PATH="gs://bucket/exports"' in rendered
    assert 'trainer.ckpt_path="$W8_CKPT_PATH"' in rendered
    assert 'trainer.export_path="$W8_EXPORT_PATH"' in rendered


def test_render_run_id_adds_cluster_suffix_labels_and_env():
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            run_id="R2026_06_14_Test",
            owner="SSS",
        )
    )
    config = yaml.safe_load(rendered)

    assert config["name"] == "w8-biayn-cpp-grpo-R2026_06_14_Test"
    assert config["envs"]["W8_BIAYN_RUN_ID"] == "R2026_06_14_Test"
    assert config["resources"]["labels"] == {
        "project": "w8-biayn",
        "phase": "cpp-perf-rl",
        "pipeline": "cpp-grpo",
        "run_id": "r2026-06-14-test",
        "owner": "sss",
        "ttl": "training",
    }


def test_render_cpp_eval_yaml_uses_eval_entrypoint_and_a100_one():
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-eval",
            project_id="proj",
            accelerators="A100:1",
            run_id="rtest",
            eval_label="base",
            eval_max_tasks=8,
        )
    )
    config = yaml.safe_load(rendered)

    assert config["resources"]["accelerators"] == "A100:1"
    assert config["resources"]["labels"]["pipeline"] == "cpp-eval"
    assert "python -m w8_biayn.integrations.cpp_eval_main" in config["run"]
    assert "--label \"base\"" in config["run"]
    assert "--max-tasks 8" in config["run"]
    assert "W8_EVAL_MODEL" in config["run"]
    assert "gcloud storage cp --recursive" in config["run"]
    assert "runs/cpp-perf/rtest/cpp-eval" in config["run"]


def test_render_training_stages_gcs_model_exports():
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            model="gs://bucket/runs/cpp-sft/exports",
        )
    )

    assert 'export W8_BIAYN_MODEL_PATH="gs://bucket/runs/cpp-sft/exports"' in rendered
    assert "gcloud storage cp --recursive \"$W8_BIAYN_MODEL_PATH/*\" \"$W8_LOCAL_MODEL_DIR/\"" in rendered
    assert 'trainer.policy.model.path="$W8_BIAYN_MODEL_PATH"' in rendered
