from __future__ import annotations

import subprocess

import yaml

from w8_biayn.sky_config import RenderOptions, render_sky_yaml


def test_render_cpp_smoke_yaml_contains_gcp_model_and_upstreams():
    rendered = render_sky_yaml(RenderOptions(pipeline="cpp-smoke", project_id="proj"))
    config = yaml.safe_load(rendered)

    assert config["name"] == "w8-biayn-cpp-smoke"
    assert config["resources"]["infra"] == "gcp"
    assert config["resources"]["accelerators"] == "H100:8"
    assert config["resources"]["disk_size"] == 256
    assert config["file_mounts"]["/tmp/w8-gcp-service-account.json"] == ".gcp-service-account.json"
    assert config["envs"]["W8_BIAYN_MODEL"] == "zai-org/GLM-5.1"
    assert "NovaSky-AI/SkyRL" in config["setup"]
    assert "rllm-org/rllm" in config["setup"]
    assert "vllm" in config["run"]
    assert "uv venv --clear --python 3.12 --seed /tmp/w8-cpp-smoke" in config["run"]
    assert "domdiff" not in rendered.lower()
    assert "harbor" not in rendered.lower()


def test_render_training_and_eval_default_to_qwen_model():
    for pipeline in ("cpp-sft", "cpp-grpo", "cpp-eval"):
        rendered = render_sky_yaml(RenderOptions(pipeline=pipeline, project_id="proj"))
        config = yaml.safe_load(rendered)

        assert config["envs"]["W8_BIAYN_MODEL"] == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert 'Qwen/Qwen2.5-Coder-7B-Instruct' in config["run"]


def test_render_cpp_training_yaml_uses_skyrl_entrypoints_and_a100_defaults():
    sft = yaml.safe_load(
        render_sky_yaml(RenderOptions(pipeline="cpp-sft", project_id="proj", accelerators="A100:8"))
    )
    grpo_text = render_sky_yaml(RenderOptions(pipeline="cpp-grpo", project_id="proj", accelerators="A100:8"))
    grpo = yaml.safe_load(grpo_text)

    assert sft["resources"]["accelerators"] == "A100:8"
    assert sft["resources"]["disk_size"] == 1024
    assert grpo["resources"]["disk_size"] == 1024
    assert "python -m skyrl.train.main_sft" in sft["run"]
    assert "dataset_name=/data/sft" in sft["run"]
    assert "python -m w8_biayn.integrations.skyrl_io_patch" in sft["run"]
    assert "python -m w8_biayn.integrations.skyrl_cpp_perf_main" in grpo["run"]
    assert "uv venv --clear --python 3.12 --seed /tmp/w8-train" in grpo["run"]
    assert grpo["run"].index("uv sync --active --extra fsdp --extra gcp") < grpo["run"].index(
        "uv pip install gcsfs"
    )
    assert grpo["run"].index("uv pip install gcsfs") < grpo["run"].index(
        "uv pip install --no-deps -e /workspace"
    )
    assert grpo["run"].index("uv pip install --no-deps -e /workspace") < grpo["run"].index(
        "python -m w8_biayn.integrations.skyrl_cpp_perf_main"
    )
    assert "W8_BIAYN_MODEL_PATH" in grpo["run"]
    assert 'trainer.policy.model.path="$W8_BIAYN_MODEL_PATH"' in grpo["run"]
    assert 'export SKYRL_RAY_PG_TIMEOUT_IN_S="${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}"' in grpo["run"]
    assert 'export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}"' in grpo["run"]
    assert '-e SKYRL_RAY_PG_TIMEOUT_IN_S="$SKYRL_RAY_PG_TIMEOUT_IN_S"' in grpo["run"]
    assert '-e SKYRL_WORKER_NCCL_TIMEOUT_IN_S="$SKYRL_WORKER_NCCL_TIMEOUT_IN_S"' in grpo["run"]
    assert "environment.env_class=cpp-perf" in grpo["run"]
    assert "-v /var/run/docker.sock:/var/run/docker.sock" in grpo["run"]
    assert "-v /tmp:/tmp" in grpo["run"]
    assert "DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io" in grpo["run"]
    assert "docker version" in grpo["run"]
    assert "kernel.perf_event_paranoid" not in grpo["run"]
    assert 'uv run w8-biayn cpp harness preflight --image "w8-biayn-cpp-perf:latest" --cpu "3"' in grpo["run"]
    assert grpo["run"].index("uv run w8-biayn cpp harness preflight") < grpo["run"].index(
        "gcloud storage cp --recursive"
    )
    assert grpo["run"].index("uv run w8-biayn cpp harness preflight") < grpo["run"].index(
        "docker pull \"$W8_GPU_CONTAINER_IMAGE\""
    )
    assert "w8-biayn cpp harness preflight --image \"w8-biayn-cpp-perf:latest\" --cpu \"3\"" in grpo["run"]
    assert grpo["run"].count("w8-biayn cpp harness preflight") == 2
    assert "environment.skyrl_gym.max_env_workers=32" in grpo["run"]
    assert "trainer.logger=console" in grpo["run"]
    assert "trainer.ckpt_interval=-1" in grpo["run"]
    assert "trainer.hf_save_interval=-1" in grpo["run"]
    assert "trainer.micro_train_batch_size_per_gpu=1" in grpo["run"]
    assert "trainer.algorithm.use_kl_loss=true" in grpo["run"]
    assert "trainer.algorithm.kl_loss_coef=0.001" in grpo["run"]
    assert "trainer.algorithm.use_entropy_loss=true" in grpo["run"]
    assert "trainer.algorithm.entropy_loss_coef=0.001" in grpo["run"]
    assert "micro_train_batch_size_per_gpu=1" in sft["run"]
    assert '-v "$W8_ARTIFACT_DIR":/artifacts' in grpo["run"]
    assert 'trainer.ckpt_path="$W8_CKPT_PATH"' in grpo["run"]
    assert 'trainer.export_path="$W8_EXPORT_PATH"' in grpo["run"]
    assert "gcloud storage cp --recursive \"$W8_ARTIFACT_DIR/exports\"" in grpo["run"]
    assert "trainer.epochs=1" in grpo["run"]
    assert "trainer.eval_before_train=true" in grpo["run"]
    assert "trainer.eval_interval=50" in grpo["run"]
    assert "trainer.placement.colocate_all=true" in grpo["run"]
    assert "trainer.placement.policy_num_nodes=1" in grpo["run"]
    assert "trainer.placement.policy_num_gpus_per_node=8" in grpo["run"]
    assert "trainer.placement.ref_num_nodes=1" in grpo["run"]
    assert "trainer.placement.ref_num_gpus_per_node=8" in grpo["run"]
    assert "trainer.policy.fsdp_config.fsdp_size=8" in grpo["run"]
    assert "trainer.ref.fsdp_config.fsdp_size=8" in grpo["run"]
    assert "generator.inference_engine.num_engines=8" in grpo["run"]
    assert "generator.inference_engine.tensor_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.pipeline_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.data_parallel_size=1" in grpo["run"]
    assert "generator.inference_engine.backend=vllm" in grpo["run"]
    assert "generator.inference_engine.run_engines_locally=true" in grpo["run"]
    assert "generator.inference_engine.weight_sync_backend=nccl" in grpo["run"]
    assert "generator.inference_engine.async_engine=true" in grpo["run"]
    assert "generator.inference_engine.gpu_memory_utilization=0.7" in grpo["run"]
    assert "generator.batched=false" in grpo["run"]
    assert "ray start --head" not in grpo["run"]
    assert "BasePPOExp" not in grpo_text
    assert "domdiff" not in grpo_text.lower()
    assert "harbor" not in grpo_text.lower()
    assert grpo["envs"]["W8_BIAYN_DATA_GCS_PREFIX"].endswith("/datasets/cpp-perf/cpp-perf-v1/skyrl")
    assert grpo["envs"]["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] == "/tmp/w8-gcp-service-account.json"
    assert grpo["envs"]["CLOUDSDK_CORE_PROJECT"] == "proj"


def test_render_cpp_training_yaml_exposes_full_run_knobs():
    grpo_rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            disk_size=2048,
            train_epochs=3,
            eval_before_train=False,
            eval_interval=25,
            max_env_workers=128,
            ckpt_interval=50,
            hf_save_interval=100,
            max_ckpts_to_keep=2,
            ckpt_path="gs://bucket/ckpts",
            export_path="gs://bucket/exports",
            micro_train_batch_size_per_gpu=2,
            grpo_use_kl_loss=True,
            grpo_kl_loss_coef=0.002,
            grpo_use_entropy_loss=True,
            grpo_entropy_loss_coef=0.003,
            grpo_vllm_gpu_memory_utilization=0.65,
            sandbox_image="w8-cpp:perf",
            sandbox_cpu="7",
        )
    )
    sft_rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-sft",
            project_id="proj",
            accelerators="A100:8",
            resume_from="latest",
            micro_train_batch_size_per_gpu=2,
        )
    )

    assert "w8-biayn cpp harness preflight --image \"w8-cpp:perf\" --cpu \"7\"" in grpo_rendered
    assert yaml.safe_load(grpo_rendered)["resources"]["disk_size"] == 2048
    assert "uv run w8-biayn cpp harness preflight --image \"w8-cpp:perf\" --cpu \"7\"" in grpo_rendered
    assert "uv venv --clear --python 3.12 --seed /tmp/w8-train" in grpo_rendered
    assert "python -m w8_biayn.integrations.skyrl_io_patch" in grpo_rendered
    assert 'export SKYRL_RAY_PG_TIMEOUT_IN_S="${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}"' in sft_rendered
    assert 'export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}"' in sft_rendered
    assert "trainer.epochs=3" in grpo_rendered
    assert "trainer.eval_before_train=false" in grpo_rendered
    assert "trainer.eval_interval=25" in grpo_rendered
    assert "environment.skyrl_gym.max_env_workers=128" in grpo_rendered
    assert "trainer.ckpt_interval=50" in grpo_rendered
    assert "trainer.hf_save_interval=100" in grpo_rendered
    assert "trainer.micro_train_batch_size_per_gpu=2" in grpo_rendered
    assert "trainer.algorithm.use_kl_loss=true" in grpo_rendered
    assert "trainer.algorithm.kl_loss_coef=0.002" in grpo_rendered
    assert "trainer.algorithm.use_entropy_loss=true" in grpo_rendered
    assert "trainer.algorithm.entropy_loss_coef=0.003" in grpo_rendered
    assert "generator.inference_engine.gpu_memory_utilization=0.65" in grpo_rendered
    assert "micro_train_batch_size_per_gpu=2" in sft_rendered
    assert "trainer.max_ckpts_to_keep=2" in grpo_rendered
    assert 'export W8_CKPT_PATH="gs://bucket/ckpts"' in grpo_rendered
    assert 'export W8_EXPORT_PATH="gs://bucket/exports"' in grpo_rendered
    assert 'trainer.ckpt_path="$W8_CKPT_PATH"' in grpo_rendered
    assert 'trainer.export_path="$W8_EXPORT_PATH"' in grpo_rendered
    assert 'resume_from="latest"' in sft_rendered


def test_render_cpp_training_run_scripts_parse_nested_skyrl_patch():
    for pipeline in ("cpp-sft", "cpp-grpo"):
        rendered = render_sky_yaml(RenderOptions(pipeline=pipeline, project_id="proj", accelerators="A100:8"))
        run = yaml.safe_load(rendered)["run"]

        shell_result = subprocess.run(["bash", "-n"], input=run, text=True, capture_output=True)
        assert shell_result.returncode == 0, shell_result.stderr
        assert "python -m w8_biayn.integrations.skyrl_io_patch" in run


def test_render_cpp_grpo_multinode_uses_total_rollout_engines_and_resume_mode():
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            num_nodes=2,
            train_batch_size=32,
            n_samples_per_prompt=8,
            max_env_workers=256,
            resume_from="latest",
        )
    )
    config = yaml.safe_load(rendered)
    run = config["run"]

    assert config["num_nodes"] == 2
    assert config["resources"]["disk_size"] == 2048
    assert "trainer.placement.policy_num_nodes=2" in run
    assert "trainer.placement.policy_num_gpus_per_node=8" in run
    assert "trainer.placement.ref_num_nodes=2" in run
    assert "trainer.placement.ref_num_gpus_per_node=8" in run
    assert "trainer.policy.fsdp_config.fsdp_size=8" in run
    assert "trainer.ref.fsdp_config.fsdp_size=8" in run
    assert "generator.inference_engine.num_engines=16" in run
    assert "trainer.train_batch_size=32" in run
    assert "generator.n_samples_per_prompt=8" in run
    assert "environment.skyrl_gym.max_env_workers=256" in run
    assert "generator.inference_engine.tensor_parallel_size=1" in run
    assert "generator.inference_engine.data_parallel_size=1" in run
    assert "trainer.resume_mode=latest" in run
    assert '-e SKYPILOT_NODE_RANK="${SKYPILOT_NODE_RANK:-0}"' in run
    assert '-e SKYPILOT_NODE_IPS="${SKYPILOT_NODE_IPS:-}"' in run
    assert "if [ \"${SKYPILOT_NODE_RANK:-0}\" = \"0\" ]" in run
    assert "ray start --head --disable-usage-stats --port 6479 --num-gpus 8" in run
    assert 'start_ray_worker "$W8_RAY_HEAD_IP:6479"' in run
    assert "export RAY_ADDRESS=127.0.0.1:6479" in run
    assert "worker rank ${SKYPILOT_NODE_RANK:-unknown} joined Ray" in run
    assert "skipping artifact upload on worker rank" in run
    assert run.count("python -m w8_biayn.integrations.skyrl_cpp_perf_main") == 1
    assert config["envs"]["W8_BIAYN_TOTAL_GPU_COUNT"] == "16"
    assert config["envs"]["W8_BIAYN_EFFECTIVE_SAMPLES_PER_STEP"] == "256"
    assert config["envs"]["W8_BIAYN_SAMPLES_PER_GPU_PER_STEP"] == "16"


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
    assert config["resources"]["disk_size"] == 1024
    assert config["resources"]["labels"]["pipeline"] == "cpp-eval"
    assert "python -m w8_biayn.integrations.cpp_eval_main" in config["run"]
    assert "--label \"base\"" in config["run"]
    assert "--max-tasks 8" in config["run"]
    assert "W8_EVAL_MODEL" in config["run"]
    assert "kernel.perf_event_paranoid" not in config["run"]
    assert "uv run w8-biayn cpp harness preflight" in config["run"]
    assert config["run"].index("uv run w8-biayn cpp harness preflight") < config["run"].index(
        "gcloud storage cp --recursive"
    )
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
    assert "resolve_gcs_model_export()" in rendered
    assert 'export W8_BIAYN_MODEL_SOURCE="$W8_BIAYN_MODEL_PATH"' in rendered
    assert 'export W8_BIAYN_MODEL_PATH="$(resolve_gcs_model_export "$W8_BIAYN_MODEL_PATH")"' in rendered
    assert 'echo "resolved GCS model export: $W8_BIAYN_MODEL_SOURCE -> $W8_BIAYN_MODEL_PATH"' in rendered
    assert 'export W8_LOCAL_MODEL_DIR="$W8_ARTIFACT_DIR/model"' in rendered
    assert "gcloud storage cp --recursive \"$W8_BIAYN_MODEL_PATH/*\" \"$W8_LOCAL_MODEL_DIR/\"" in rendered
    assert 'assert_local_hf_model "$W8_LOCAL_MODEL_DIR"' in rendered
    assert 'export W8_BIAYN_MODEL_PATH="/artifacts/model"' in rendered
    assert '-v "$W8_ARTIFACT_DIR":/artifacts' in rendered
    assert 'trainer.policy.model.path="$W8_BIAYN_MODEL_PATH"' in rendered
    assert "global_step_${step}/policy" in rendered


def test_render_eval_stages_gcs_model_under_mounted_output_dir():
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-eval",
            project_id="proj",
            accelerators="A100:1",
            model="gs://bucket/runs/cpp-grpo/exports/global_step_10/policy",
            run_id="rtest",
            eval_label="grpo",
        )
    )
    run = yaml.safe_load(rendered)["run"]

    assert "resolve_gcs_model_export()" in run
    assert 'export W8_EVAL_MODEL_SOURCE="$W8_EVAL_MODEL"' in run
    assert 'export W8_EVAL_MODEL="$(resolve_gcs_model_export "$W8_EVAL_MODEL")"' in run
    assert 'echo "resolved GCS eval model export: $W8_EVAL_MODEL_SOURCE -> $W8_EVAL_MODEL"' in run
    assert 'export W8_EVAL_LOCAL_MODEL="$W8_EVAL_OUTPUT_DIR/model"' in run
    assert "gcloud storage cp --recursive \"$W8_EVAL_MODEL/*\" \"$W8_EVAL_LOCAL_MODEL/\"" in run
    assert 'assert_local_hf_model "$W8_EVAL_LOCAL_MODEL"' in run
    assert 'export W8_EVAL_MODEL="$W8_EVAL_LOCAL_MODEL"' in run
    assert "-v /tmp:/tmp" in run
    assert '--model "$W8_EVAL_MODEL"' in run
