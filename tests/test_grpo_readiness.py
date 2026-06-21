from __future__ import annotations

import json
from pathlib import Path

import yaml

from w8_biayn.grpo_readiness import build_grpo_readiness, readiness_blocks_launch
from w8_biayn.sky_config import RenderOptions, render_sky_yaml


def _write_rendered(tmp_path: Path, *, text: str | None = None, **overrides) -> Path:
    option_values = {
        "pipeline": "cpp-grpo",
        "project_id": "proj",
        "accelerators": "A100:8",
        "num_nodes": 2,
        "train_batch_size": 32,
        "n_samples_per_prompt": 8,
        "max_env_workers": 256,
        "run_id": "rtest",
        "max_ckpts_to_keep": 8,
        "hf_save_interval": 10000,
    }
    option_values.update(overrides)
    options = RenderOptions(**option_values)
    path = tmp_path / "cpp-grpo.sky.yaml"
    path.write_text(text if text is not None else render_sky_yaml(options), encoding="utf-8")
    return path


def _check(payload: dict, check_id: str) -> dict:
    return next(check for check in payload["checks"] if check["id"] == check_id)


def test_grpo_readiness_passes_valid_multinode_render(tmp_path: Path) -> None:
    path = _write_rendered(tmp_path)

    payload = build_grpo_readiness(path)

    assert payload["schema_version"] == "w8-grpo-readiness-v1"
    assert payload["overall"] == "pass"
    assert readiness_blocks_launch(payload) is False
    assert _check(payload, "network.nccl_env")["ok"] is True
    assert _check(payload, "network.gloo_concrete_interface")["ok"] is True
    assert _check(payload, "skyrl.patch.io")["ok"] is True
    assert _check(payload, "skyrl.patch.vllm_logprob")["ok"] is True
    assert _check(payload, "skyrl.patch.grpo_health")["ok"] is True
    assert _check(payload, "skyrl.patch.startup")["ok"] is True
    assert _check(payload, "tracking.console_logger")["ok"] is True
    assert _check(payload, "tracking.mlflow_server")["ok"] is True
    assert _check(payload, "tracking.mlflow_persistence")["ok"] is True
    assert _check(payload, "multinode.hsdp_mesh")["ok"] is True
    assert _check(payload, "multinode.rollout_engines")["ok"] is True
    assert _check(payload, "multinode.utilization_gate")["ok"] is True
    assert _check(payload, "training.checkpoint_retention")["ok"] is True


def test_grpo_readiness_fails_bad_gloo_exclusion_syntax(tmp_path: Path) -> None:
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            num_nodes=2,
            train_batch_size=32,
            n_samples_per_prompt=8,
            max_env_workers=256,
            max_ckpts_to_keep=8,
            hf_save_interval=10000,
        )
    )
    rendered = rendered.replace(
        '-e GLOO_SOCKET_IFNAME="$W8_GLOO_SOCKET_IFNAME"',
        '-e GLOO_SOCKET_IFNAME="^lo,docker,veth"',
    )
    path = _write_rendered(tmp_path, text=rendered)

    payload = build_grpo_readiness(path)

    assert payload["overall"] == "fail"
    assert readiness_blocks_launch(payload) is True
    assert _check(payload, "network.gloo_concrete_interface")["ok"] is False


def test_grpo_readiness_fails_missing_vllm_patch(tmp_path: Path) -> None:
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            num_nodes=2,
            train_batch_size=32,
            n_samples_per_prompt=8,
            max_env_workers=256,
            max_ckpts_to_keep=8,
            hf_save_interval=10000,
        )
    ).replace("python -m w8_biayn.integrations.skyrl_vllm_logprob_patch\n", "")
    path = _write_rendered(tmp_path, text=rendered)

    payload = build_grpo_readiness(path)

    assert payload["overall"] == "fail"
    assert _check(payload, "skyrl.patch.vllm_logprob")["ok"] is False


def test_grpo_readiness_fails_missing_grpo_health_patch(tmp_path: Path) -> None:
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            num_nodes=2,
            train_batch_size=32,
            n_samples_per_prompt=8,
            max_env_workers=256,
            max_ckpts_to_keep=8,
            hf_save_interval=10000,
        )
    ).replace("python -m w8_biayn.integrations.skyrl_grpo_health_patch\n", "")
    path = _write_rendered(tmp_path, text=rendered)

    payload = build_grpo_readiness(path)

    assert payload["overall"] == "fail"
    assert _check(payload, "skyrl.patch.grpo_health")["ok"] is False


def test_grpo_readiness_fails_missing_startup_patch(tmp_path: Path) -> None:
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            num_nodes=2,
            train_batch_size=32,
            n_samples_per_prompt=8,
            max_env_workers=256,
            max_ckpts_to_keep=8,
            hf_save_interval=10000,
        )
    ).replace("python -m w8_biayn.integrations.skyrl_startup_patch\n", "")
    path = _write_rendered(tmp_path, text=rendered)

    payload = build_grpo_readiness(path)

    assert payload["overall"] == "fail"
    assert _check(payload, "skyrl.patch.startup")["ok"] is False


def test_grpo_readiness_fails_wrong_hsdp_mesh(tmp_path: Path) -> None:
    rendered = render_sky_yaml(
        RenderOptions(
            pipeline="cpp-grpo",
            project_id="proj",
            accelerators="A100:8",
            num_nodes=2,
            train_batch_size=32,
            n_samples_per_prompt=8,
            max_env_workers=256,
            max_ckpts_to_keep=8,
            hf_save_interval=10000,
        )
    ).replace("trainer.policy.fsdp_config.fsdp_size=8", "trainer.policy.fsdp_config.fsdp_size=16")
    path = _write_rendered(tmp_path, text=rendered)

    payload = build_grpo_readiness(path)

    assert payload["overall"] == "fail"
    assert _check(payload, "multinode.hsdp_mesh")["ok"] is False


def test_grpo_readiness_fails_low_multinode_utilization_without_override(tmp_path: Path) -> None:
    path = _write_rendered(
        tmp_path,
        train_batch_size=32,
        n_samples_per_prompt=4,
        max_env_workers=128,
        max_ckpts_to_keep=8,
        hf_save_interval=10000,
    )

    payload = build_grpo_readiness(path)

    assert payload["overall"] == "fail"
    check = _check(payload, "multinode.utilization_gate")
    assert check["ok"] is False
    assert check["evidence"]["effective_samples_per_step"] == 128


def test_grpo_readiness_status_requires_all_active_node_health(tmp_path: Path) -> None:
    path = _write_rendered(tmp_path)
    status = {
        "schema_version": "w8-run-status-v1",
        "run_id": "rtest",
        "pipelines": [
            {
                "pipeline": "cpp-grpo",
                "cluster": "w8-biayn-cpp-grpo-rtest",
                "state": "running",
                "active_job": {"job_id": 2, "status": "RUNNING"},
                "progress": {"grpo_config": {"policy_num_nodes": 2, "total_gpu_count": 16}},
                "resources": {
                    "total_instance_count": 2,
                    "active_instance_count": 2,
                    "gpu_count": 16,
                    "sampled_node_count": 1,
                    "sampled_gpu_count": 8,
                    "failed_node_count": 0,
                },
                "node_health": {
                    "sample_scope": "head",
                    "expected_node_count": 2,
                    "sampled_node_count": 1,
                    "failed_node_count": 0,
                    "gpus": [{} for _ in range(8)],
                },
                "commands": {
                    "cancel": ["uv", "run", "w8-biayn", "ops", "cancel", "w8-biayn-cpp-grpo-rtest", "2"],
                },
                "recovery": {"available": False},
            }
        ],
    }

    payload = build_grpo_readiness(path, status_payload=status)

    assert payload["overall"] == "action_required"
    assert _check(payload, "status.multinode_node_health")["ok"] is False
    assert payload["recommended_commands"][0][:5] == ["uv", "run", "w8-biayn", "ops", "run-status"]


def test_grpo_readiness_status_flags_should_stop_training_health(tmp_path: Path) -> None:
    path = _write_rendered(tmp_path)
    status = {
        "schema_version": "w8-run-status-v1",
        "run_id": "rtest",
        "pipelines": [
            {
                "pipeline": "cpp-grpo",
                "state": "running",
                "active_job": {"job_id": 2, "status": "RUNNING"},
                "progress": {
                    "grpo_config": {"policy_num_nodes": 2, "total_gpu_count": 16},
                    "training_health": {
                        "should_stop": True,
                        "verdict": "deterministic_low_gradient",
                        "recommended_action": "stop_and_evaluate_checkpoint",
                        "checkpoint_step": 150,
                    },
                },
                "resources": {
                    "total_instance_count": 2,
                    "active_instance_count": 2,
                    "gpu_count": 16,
                    "sampled_node_count": 2,
                    "sampled_gpu_count": 16,
                    "failed_node_count": 0,
                },
                "node_health": {
                    "sample_scope": "all_active",
                    "expected_node_count": 2,
                    "sampled_node_count": 2,
                    "failed_node_count": 0,
                    "gpus": [{} for _ in range(16)],
                },
                "commands": {},
                "recovery": {"available": False},
            }
        ],
    }

    payload = build_grpo_readiness(path, status_payload=status)

    assert payload["overall"] == "action_required"
    check = _check(payload, "status.training_health_should_continue")
    assert check["ok"] is False
    assert check["evidence"]["recommended_action"] == "stop_and_evaluate_checkpoint"


def test_grpo_readiness_status_flags_learning_signal_eval_recommendation(tmp_path: Path) -> None:
    path = _write_rendered(tmp_path)
    status = {
        "schema_version": "w8-run-status-v1",
        "run_id": "rtest",
        "pipelines": [
            {
                "pipeline": "cpp-grpo",
                "state": "running",
                "active_job": {"job_id": 2, "status": "RUNNING"},
                "progress": {
                    "grpo_config": {"policy_num_nodes": 2, "total_gpu_count": 16},
                    "training_health": {"should_stop": False, "recommended_action": "evaluate_checkpoint"},
                    "learning_signal": {
                        "available": True,
                        "verdict": "deterministic_convergence_risk",
                        "severity": "warning",
                        "recommended_action": "evaluate_checkpoint",
                        "reasons": ["policy_entropy_near_zero"],
                    },
                },
                "resources": {
                    "total_instance_count": 2,
                    "active_instance_count": 2,
                    "gpu_count": 16,
                    "sampled_node_count": 2,
                    "sampled_gpu_count": 16,
                    "failed_node_count": 0,
                },
                "node_health": {
                    "sample_scope": "all_active",
                    "expected_node_count": 2,
                    "sampled_node_count": 2,
                    "failed_node_count": 0,
                    "gpus": [{} for _ in range(16)],
                },
                "commands": {},
                "recovery": {"available": False},
            }
        ],
    }

    payload = build_grpo_readiness(path, status_payload=status)

    assert payload["overall"] == "action_required"
    assert _check(payload, "status.learning_signal_available")["ok"] is True
    check = _check(payload, "status.learning_signal_recommendation")
    assert check["ok"] is False
    assert check["evidence"]["recommended_action"] == "evaluate_checkpoint"


def test_grpo_readiness_json_is_serializable(tmp_path: Path) -> None:
    path = _write_rendered(tmp_path)
    payload = build_grpo_readiness(path)

    assert json.loads(json.dumps(payload))["overall"] == "pass"
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["num_nodes"] == 2
