from __future__ import annotations

import subprocess
from typing import Any

from w8_biayn import run_status
from w8_biayn.run_status import _extract_log_signals


def test_extract_log_signals_tracks_steps_checkpoints_and_errors():
    signals = _extract_log_signals(
        [
            "2026-06-14 | INFO | Step 295: loss=0.2099, grad_norm=1.10",
            "2026-06-14 | INFO | Step 300: eval_loss=0.2987 over 130 batches",
            "2026-06-14 | INFO | Saving checkpoint at step 300 to gs://bucket/run/ckpts/global_step_300",
            "2026-06-14 | INFO | Checkpoint saved for global_step_300",
            "2026-06-14 | INFO | Finished: 'save_checkpoint', time cost: 643.99s",
            "Traceback (most recent call last):",
        ]
    )

    assert signals["last_step"] == 300
    assert signals["last_loss"] == {"step": 295, "loss": 0.2099}
    assert signals["last_eval_loss"] == {"step": 300, "eval_loss": 0.2987}
    assert signals["stage"] == "failed"
    assert signals["checkpoint_events"][0]["event"] == "save_started"
    assert signals["checkpoint_events"][0]["step"] == 300
    assert signals["checkpoint_events"][1] == {
        "event": "save_completed",
        "step": 300,
        "line": "2026-06-14 | INFO | Checkpoint saved for global_step_300",
    }
    assert signals["checkpoint_events"][2]["duration_s"] == 643.99
    assert signals["errors"] == ["Traceback (most recent call last):"]


def test_extract_log_signals_marks_checkpoint_save_stage():
    signals = _extract_log_signals(
        [
            "2026-06-16 18:47:28.827 | INFO | Started: 'save_checkpoint'",
            "2026-06-16 18:47:28.827 | INFO | Saving checkpoint at step 800 to gs://bucket/ckpts/global_step_800",
        ]
    )

    assert signals["stage"] == "checkpoint_stage"
    assert [event["stage"] for event in signals["stage_events"]] == ["checkpoint_stage", "checkpoint_stage"]


def test_extract_log_signals_ignores_benign_setup_warnings():
    signals = _extract_log_signals(
        [
            "warning: Failed to hardlink files; falling back to full copy.",
            "Downloading nvidia-nccl-cu12 (283.0MiB)",
            "Downloaded nvidia-nccl-cu12",
            "+ nvidia-nccl-cu12==2.27.3",
            "- nvidia-nccl-cu13==2.29.7",
            "weight_sync_backend: nccl",
            "2026-06-14 | ERROR | aiohttp.client:__del__:465 - Unclosed client session",
            "2026-06-14 | ERROR | aiohttp.connector:__del__:388 - Unclosed connector",
            "FutureWarning: Tip: In future versions of Ray, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 to turn off this error message",
            "Job started. Streaming logs... (Ctrl-C to exit log streaming; job will not be killed)",
            "(cleanup_old_checkpoints pid=9143) 2026-06-17 15:57:01.698 | WARNING  | "
            "skyrl.train.utils.trainer_utils:cleanup_old_checkpoints:164 - Failed to remove old checkpoint "
            "gs://bucket/run/ckpts/global_step_150: {\"code\": 404, \"message\": \"The specified key does not exist.\", "
            "\"errors\": [{\"reason\": \"notFound\"}]}",
        ]
    )

    assert signals["stage"] is None
    assert signals["errors"] == []


def test_extract_log_signals_keeps_nccl_timeout_and_ignores_cleanup_idle_kills():
    signals = _extract_log_signals(
        [
            "[2026-06-21 E] Process group watchdog thread terminated with exception: "
            "Watchdog caught collective operation timeout: WorkNCCL(OpType=BROADCAST)",
            "(worker1) 2026-06-21 VINFO scripts.py:1254 -- Killed `ray::IDLE \"\" \"\"` (via SIGKILL)",
        ]
    )

    assert signals["stage"] == "failed"
    assert signals["errors"] == [
        "[2026-06-21 E] Process group watchdog thread terminated with exception: "
        "Watchdog caught collective operation timeout: WorkNCCL(OpType=BROADCAST)"
    ]


def test_extract_log_signals_ignores_nccl_timeout_env_knob():
    signals = _extract_log_signals(
        [
            "docker run -e SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600 "
            "-e SKYRL_RAY_PG_TIMEOUT_IN_S=1800 image bash -lc '...'",
        ]
    )

    assert signals["errors"] == []
    assert signals["config"]["skyrl.worker_nccl_timeout_s"] == 3600


def test_extract_log_signals_ignores_multinode_ray_startup_noise():
    signals = _extract_log_signals(
        [
            "(worker1)   echo \"Ray cluster at $address failed to become ready\" >&2",
            "(worker1) [2026-06-15 09:50:57,864 W 2200 2200] rpc_client.h:153: Failed to connect to GCS at address 10.178.0.15:6479 within 5 seconds.",
            "(worker1) [2026-06-15 09:51:27,868 W 2200 2200] gcs_client.cc:182: Failed to get cluster ID from GCS server: TimedOut: Timed out while waiting for GCS to become available.",
            "(head) (raylet) Failed to establish connection to the metrics exporter agent. Metrics will not be exported. Exporter agent status: RpcError: Running out of retries to initialize the metrics agent. rpc_code: 14",
        ]
    )

    assert signals["errors"] == []


def test_extract_log_signals_ignores_printed_exception_handler_source():
    signals = _extract_log_signals(
        [
            "(head, rank=0, pid=7594)     except Exception:",
            "(head, rank=0, pid=7594) uv pip install \"mlflow>=2.12,<3\"",
        ]
    )

    assert signals["stage"] == "dependency_setup"
    assert signals["errors"] == []


def test_extract_log_signals_does_not_treat_error_metric_names_as_failures():
    signals = _extract_log_signals(
        [
            "'eval/all/environment/compile_error': '0.1143',",
            "'eval/all/environment/sanitizer_error': '0.0000',",
            "#define error(args...) {vector <string> _v = split(#args, ',');}",
        ]
    )

    assert signals["stage"] is None
    assert signals["errors"] == []


def test_extract_log_signals_reports_only_model_export_events():
    signals = _extract_log_signals(
        [
            "+ export W8_EXPORT_PATH=/artifacts/exports",
            "cpp-sft export-only: skipping dataset restore",
            "-e HF_HOME=/root/.cache/huggingface",
            "+ huggingface-hub==1.19.0",
            "Saving final HF model at step 1074",
            "Saved HF model weights at step 1074 to /artifacts/exports/global_step_1074/policy",
        ]
    )

    assert signals["export_events"] == [
        "Saving final HF model at step 1074",
        "Saved HF model weights at step 1074 to /artifacts/exports/global_step_1074/policy",
    ]
    assert signals["stage"] == "hf_export_done"
    assert [event["stage"] for event in signals["stage_events"]] == [
        "dependency_setup",
        "hf_export",
        "hf_export_done",
    ]


def test_extract_log_signals_reports_worker_init_stage():
    signals = _extract_log_signals(
        [
            "2026-06-14 INFO worker.py -- Started a local Ray instance.",
            "INFO worker:_initiate_actors:606 - Initializing process group for RayActorGroup",
            "INFO worker:_initiate_actors:610 - Mesh Ranks: [MeshRank(dp=0)]",
        ]
    )

    assert signals["stage"] == "worker_init"


def test_extract_log_signals_reports_w8_setup_stage_json():
    signals = run_status._extract_log_signals(
        [
            'W8_SETUP_STAGE {"schema_version":"w8-setup-stage-v1","stage":"ref_model_init","event":"start","rank":0}',
        ]
    )

    assert signals["stage"] == "ref_model_init"
    assert signals["setup_events"] == [
        {"schema_version": "w8-setup-stage-v1", "stage": "ref_model_init", "event": "start", "rank": 0}
    ]
    assert signals["stage_events"][0]["stage"] == "ref_model_init"


def test_extract_log_signals_ignores_xtrace_setup_stage_template():
    signals = run_status._extract_log_signals(
        [
            '+ printf W8_SETUP_STAGE {"schema_version":"w8-setup-stage-v1","stage":"%s","event":"%s","ts_utc":"%s"}',
            'W8_SETUP_STAGE {"schema_version":"w8-setup-stage-v1","stage":"dependency_setup","event":"start"}',
        ]
    )

    assert signals["stage"] == "dependency_setup"
    assert signals["setup_events"] == [
        {"schema_version": "w8-setup-stage-v1", "stage": "dependency_setup", "event": "start"}
    ]


def test_extract_log_signals_uses_priority_for_combined_stage_lines():
    signals = _extract_log_signals(
        [
            "uv sync completed; Saving final HF model at step 10",
        ]
    )

    assert signals["stage"] == "hf_export"
    assert signals["stage_events"][0]["matched_stages"] == ["dependency_setup", "hf_export"]


def test_log_signals_reports_tail_truncation(monkeypatch):
    def fake_run(args: list[str], **_: Any) -> run_status.CommandResult:
        return run_status.CommandResult(
            command=args,
            returncode=0,
            stdout="line1\nline2\nline3\n",
            stderr="",
        )

    monkeypatch.setattr(run_status, "_run", fake_run)

    signals, check = run_status._log_signals(
        "w8-biayn-cpp-grpo-run",
        job_id=None,
        tail=3,
        env={},
        timeout_s=1,
        retries=0,
        dry_run=False,
    )

    assert signals["tail_lines_scanned"] == 3
    assert signals["tail_lines_requested"] == 3
    assert signals["tail_may_be_truncated"] is True
    assert check["attempt_count"] == 1


def test_run_retries_timed_out_command(monkeypatch):
    calls = []

    def fake_subprocess_run(args: list[str], **kwargs: Any):
        calls.append(args)
        if len(calls) == 1:
            raise run_status.subprocess.TimeoutExpired(args, kwargs["timeout"])
        return run_status.subprocess.CompletedProcess(args, 0, "ok", "")

    monkeypatch.setattr(run_status.subprocess, "run", fake_subprocess_run)

    result = run_status._run(["gcloud", "storage", "ls"], env={}, timeout_s=1, retries=1, dry_run=False)

    assert len(calls) == 2
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.attempt_count == 2
    assert result.check("retry")["attempt_count"] == 2


def test_extract_log_signals_distinguishes_restore_stage_uploads():
    signals = _extract_log_signals(
        [
            "+ gcloud storage cp --recursive 'gs://bucket/datasets/cpp-perf/cpp-perf-v1/full-official/r1/skyrl/*' /home/gcpuser/.w8-biayn/data/skyrl/",
            "+ gcloud storage cp --recursive 'gs://bucket/runs/cpp-perf/r1/cpp-sft/exports/global_step_10/policy/*' /home/gcpuser/.w8-biayn/runs/r1/cpp-grpo/model/",
            "+ gcloud storage cp --recursive /home/gcpuser/.w8-biayn/runs/r1/cpp-grpo/exports gs://bucket/runs/cpp-perf/r1/cpp-grpo/",
        ]
    )

    assert signals["stage"] == "artifact_upload"
    assert [event["stage"] for event in signals["stage_events"]] == [
        "data_restore",
        "model_stage",
        "artifact_upload",
    ]


def test_extract_log_signals_reports_grpo_progress_bars():
    signals = _extract_log_signals(
        [
            "Generating Trajectories:  97%|█████████▋| 995/1024 [05:26<00:00, 44.49it/s]",
            "Evaluation Progress: 100%|██████████| 2/2 [06:35<00:00, 224.93s/it]",
            "Generating Trajectories:   0%|          | 0/11 [00:00<?, ?it/s]",
        ]
    )

    assert signals["stage"] == "trajectory_generation"
    assert signals["trajectory_progress"] == {
        "completed": 0,
        "total": 11,
        "percent": 0,
        "line": "Generating Trajectories:   0%|          | 0/11 [00:00<?, ?it/s]",
    }
    assert signals["evaluation_progress"]["completed"] == 2
    assert signals["evaluation_progress"]["total"] == 2
    assert signals["evaluation_progress"]["percent"] == 100


def test_extract_log_signals_reports_cpp_eval_scoring_progress():
    signals = _extract_log_signals(
        [
            "W8 eval generation start: label=sft total=1035",
            "W8 eval generation complete: label=sft total=1035",
            "W8 eval scoring start: label=sft total=1035",
            "W8 eval scoring progress: Evaluation Progress:  48%| 500/1035",
        ]
    )

    assert signals["stage"] == "eval_scoring"
    assert signals["evaluation_progress"] == {
        "completed": 500,
        "total": 1035,
        "percent": 48,
        "line": "W8 eval scoring progress: Evaluation Progress:  48%| 500/1035",
    }
    assert [event["stage"] for event in signals["stage_events"]] == [
        "eval_generation",
        "eval_generation",
        "eval_scoring",
        "eval_scoring",
    ]


def test_extract_log_signals_reports_training_and_generation_stages():
    signals = _extract_log_signals(
        [
            "INFO skyrl.train.entrypoints.main_base:_get_new_inference_client:341 - HTTP Inference: Colocated mode",
            "INFO skyrl.train.trainer:build_models:596 - init policy/ref/critic models done",
            "INFO skyrl.train.trainer:train:192 - Started: 'init_weight_sync_state'",
            "INFO skyrl.train.trainer:train:201 - Finished: 'sync_weights', time cost: 17.69s",
            "Training Batches Processed:   0%|          | 0/804 [00:00<?, ?it/s]",
            "INFO skyrl.train.trainer:train:220 - Started: 'step'",
            "INFO skyrl.train.trainer:train:237 - Started: 'generate'",
        ]
    )

    assert signals["stage"] == "trajectory_generation"
    assert signals["training_progress"] == {
        "completed": 0,
        "total": 804,
        "percent": 0,
        "line": "Training Batches Processed:   0%|          | 0/804 [00:00<?, ?it/s]",
    }
    assert [event["stage"] for event in signals["stage_events"]] == [
        "model_init",
        "weight_sync",
        "weight_sync",
        "training",
        "training",
        "trajectory_generation",
    ]


def test_progress_summary_normalizes_training_throughput_and_metrics():
    signals = _extract_log_signals(
        [
            "trainer.ckpt_interval=50 trainer.max_ckpts_to_keep=8 trainer.train_batch_size=16 generator.n_samples_per_prompt=8 "
            "trainer.micro_train_batch_size_per_gpu=1 "
            "trainer.algorithm.use_kl_loss=true trainer.algorithm.kl_loss_coef=0.001 "
            "trainer.algorithm.use_entropy_loss=true trainer.algorithm.entropy_loss_coef=0.001 "
            "trainer.placement.policy_num_nodes=1 trainer.placement.policy_num_gpus_per_node=8 "
            "trainer.policy.fsdp_config.fsdp_size=8 trainer.ref.fsdp_config.fsdp_size=8 "
            "generator.inference_engine.num_engines=8 environment.skyrl_gym.max_env_workers=128 "
            "generator.inference_engine.gpu_memory_utilization=0.7 "
            "SKYRL_RAY_PG_TIMEOUT_IN_S=1800 SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600",
            "Training Batches Processed:   1%|▏         | 12/804 [58:25<62:28:36, 283.99s/it]2026-06-15 13:24:42.955 | INFO     | skyrl.train.trainer:train:220 - Finished: 'step', time cost: 288.78s",
            "Generating Trajectories: 100%|██████████| 128/128 [00:45<00:00,  2.84it/s]",
            "2026-06-15 13:25:28.074 | INFO | Finished: 'generate', time cost: 45.11s",
            "'timing/step': '286.05', 'timing/generate': '27.65', 'timing/policy_train': '191.74', 'timing/fwd_logprobs_values_reward': '47.40'",
            "reward/avg_pass_at_4: 0.6875, reward/avg_raw_reward: -0.06094937803135128, reward/mean_positive_reward: 0.46639437196864875",
            "'batch_num_seq': '128', 'batch_padded_seq_len': '2435'",
        ]
    )

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=signals, artifacts={})

    assert progress["primary"] == "training_batches"
    assert progress["training"]["completed"] == 12
    assert progress["training"]["remaining"] == 792
    assert progress["training"]["timing"]["elapsed_s"] == 3505
    assert progress["training"]["timing"]["eta_s"] == 224916
    assert progress["training"]["timing"]["last_duration_s"] == 288.78
    assert 12.6 < progress["throughput"]["training_steps_per_hour"] < 12.8
    assert progress["trajectory"]["throughput"]["items_per_second"] == 2.84
    assert progress["checkpoint"]["next_checkpoint_step"] == 50
    assert progress["checkpoint"]["steps_until_next_checkpoint"] == 38
    assert progress["metrics"]["reward"]["avg_pass_at_4"] == 0.6875
    assert progress["metrics"]["batch_num_seq"] == 128
    assert progress["grpo_config"]["effective_samples_per_step"] == 128
    assert progress["grpo_config"]["micro_train_batch_size_per_gpu"] == 1
    assert progress["grpo_config"]["use_kl_loss"] is True
    assert progress["grpo_config"]["kl_loss_coef"] == 0.001
    assert progress["grpo_config"]["use_entropy_loss"] is True
    assert progress["grpo_config"]["entropy_loss_coef"] == 0.001
    assert progress["grpo_config"]["vllm_gpu_memory_utilization"] == 0.7
    assert progress["grpo_config"]["total_gpu_count"] == 8
    assert progress["grpo_config"]["policy_fsdp_size"] == 8
    assert progress["grpo_config"]["ref_fsdp_size"] == 8
    assert progress["grpo_config"]["fsdp_mesh_shape"] == {"ddp": 1, "fsdp": 8}
    assert progress["grpo_config"]["hsdp_active"] is False
    assert progress["grpo_config"]["samples_per_gpu_per_step"] == 16
    assert progress["grpo_config"]["skyrl_ray_pg_timeout_s"] == 1800
    assert progress["grpo_config"]["skyrl_worker_nccl_timeout_s"] == 3600
    assert progress["grpo_config"]["max_ckpts_to_keep"] == 8
    assert progress["throughput"]["training_samples_per_gpu_hour"] > 200
    assert progress["bottleneck"]["verdict"] == "policy_update_dominant"
    assert progress["bottleneck"]["dominant_stage"]["stage"] == "policy_train"


def test_progress_summary_prefers_mlflow_params_over_stale_rendered_config():
    signals = {
        "config": {
            "trainer.train_batch_size": 32,
            "trainer.placement.policy_num_nodes": 2,
            "trainer.placement.policy_num_gpus_per_node": 8,
            "trainer.policy.fsdp_config.fsdp_size": 8,
            "trainer.ref.fsdp_config.fsdp_size": 8,
            "generator.n_samples_per_prompt": 8,
            "generator.inference_engine.num_engines": 16,
            "environment.skyrl_gym.max_env_workers": 256,
        },
        "config_sources": ["rendered_yaml"],
        "metrics": {},
        "timings": {},
    }
    tracking = {
        "backend": "mlflow_tracking_server",
        "mlflow": {
            "available": True,
            "tracking_state": "metrics_available",
            "params": {
                "count": 313,
                "selected": {
                    "trainer/train_batch_size": "16",
                    "trainer/placement/policy_num_nodes": "1",
                    "trainer/placement/policy_num_gpus_per_node": "8",
                    "trainer/policy/fsdp_config/fsdp_size": "8",
                    "trainer/ref/fsdp_config/fsdp_size": "8",
                    "generator/n_samples_per_prompt": "8",
                    "generator/inference_engine/num_engines": "8",
                    "environment/skyrl_gym/max_env_workers": "128",
                },
            },
            "latest": {},
            "series": {},
        },
    }

    merged = run_status._with_tracking_metrics(log_signals=signals, tracking=tracking)
    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=merged, artifacts={})

    assert merged["config_sources"] == ["rendered_yaml", "mlflow_params"]
    assert progress["grpo_config"]["train_batch_size"] == 16
    assert progress["grpo_config"]["effective_samples_per_step"] == 128
    assert progress["grpo_config"]["policy_num_nodes"] == 1
    assert progress["grpo_config"]["total_gpu_count"] == 8
    assert progress["grpo_config"]["rollout_engine_count"] == 8
    assert progress["grpo_config"]["max_env_workers"] == 128
    assert progress["grpo_config"]["fsdp_mesh_shape"] == {"ddp": 1, "fsdp": 8}
    assert progress["grpo_config"]["hsdp_active"] is False


def test_progress_summary_flags_grpo_entropy_collapse():
    signals = _extract_log_signals(
        [
            "trainer.train_batch_size=16 generator.n_samples_per_prompt=8 "
            "trainer.algorithm.use_kl_loss=true trainer.algorithm.kl_loss_coef=0.001 "
            "trainer.algorithm.use_entropy_loss=true trainer.algorithm.entropy_loss_coef=0.001",
            "'loss/avg_final_rewards': '-1.0000', 'policy/policy_entropy': '0.00068', "
            "'policy/grad_norm': '0.0', 'policy/policy_loss': '0.0', 'reward/avg_pass_at_8': '0.0'",
        ]
    )

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=signals, artifacts={})

    assert progress["training_health"]["verdict"] == "collapsed"
    assert progress["training_health"]["should_stop"] is True
    assert progress["training_health"]["reason"] == "entropy_collapse_zero_advantage_trap"
    assert progress["training_health"]["recommended_action"] == "stop_and_relaunch_from_sft"
    assert progress["training_health"]["action_reason"] == "terminal_invalid_format_zero_advantage_trap"
    assert progress["training_health"]["signals"]["policy_entropy"] == 0.00068


def test_progress_summary_prefers_high_precision_policy_health_metrics():
    signals = _extract_log_signals(
        [
            "2026-06-18 | INFO | {'policy_entropy': 8.228687511291355e-05, "
            "'response_length': 1024.0, 'policy_loss': -0.0012181581696495414, "
            "'grad_norm': 1.1297262972220778e-05}",
            "2026-06-18 | INFO | Step 250: 'policy/policy_entropy': '0.0001', "
            "'policy/grad_norm': '0.0000', 'policy/policy_loss': '-0.0012'",
        ]
    )

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=signals, artifacts={})

    assert progress["metrics"]["policy/policy_entropy"] == 8.228687511291355e-05
    assert progress["metrics"]["policy/grad_norm"] == 1.1297262972220778e-05
    assert progress["training_health"]["signals"]["policy_entropy"] == 8.228687511291355e-05
    assert progress["training_health"]["signals"]["grad_norm"] == 1.1297262972220778e-05


def test_progress_summary_flags_deterministic_low_gradient_with_checkpoint():
    signals = _extract_log_signals(
        [
            "2026-06-18 | INFO | {'policy_entropy': 9.5e-05, 'policy_loss': -0.0009, 'grad_norm': 1.5e-05}",
            "2026-06-18 | INFO | {'policy_entropy': 7.3e-05, 'policy_loss': 0.0, 'grad_norm': 8.4e-06}",
            "2026-06-18 | INFO | {'policy_entropy': 8.3e-05, 'policy_loss': -0.0, 'grad_norm': 5.3e-05}",
            "'loss/avg_final_rewards': '0.9550', 'reward/avg_pass_at_8': '0.8750'",
        ]
    )
    artifacts = {"checkpoint": {"latest": {"step": 250, "resumable": True}}}

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=signals, artifacts=artifacts)

    assert progress["training_health"]["verdict"] == "deterministic_low_gradient"
    assert progress["training_health"]["severity"] == "warning"
    assert progress["training_health"]["should_stop"] is True
    assert progress["training_health"]["recommended_action"] == "stop_and_evaluate_checkpoint"
    assert progress["training_health"]["checkpoint_step"] == 250
    assert progress["training_health"]["reason"] == "low_entropy_low_gradient_plateau"
    assert progress["training_health"]["signals"]["recent_low_gradient_events"] == 3
    assert progress["training_health"]["signals"]["checkpoint_resumable"] is True


def test_progress_summary_uses_mlflow_tracking_metrics_for_learning_signal():
    signals = {"metrics": {}, "timings": {}, "config": {}, "policy_health_events": [], "grpo_health_events": []}
    tracking = {
        "backend": "mlflow_tracking_server",
        "mlflow": {
            "available": True,
            "latest_step": 3,
            "metric_count": 5,
            "selected_keys": [],
            "source": {"gcs_uri": "gs://bucket/run/tracking/mlflow/mlflow.db"},
            "latest": {
                "loss/avg_final_rewards": {"step": 3, "value": 0.8, "timestamp_ms": 30},
                "reward/avg_pass_at_8": {"step": 3, "value": 0.75, "timestamp_ms": 30},
                "policy/policy_entropy": {"step": 3, "value": 0.0001, "timestamp_ms": 30},
                "policy/grad_norm": {"step": 3, "value": 0.0, "timestamp_ms": 30},
                "loss/avg_raw_advantages_abs": {"step": 3, "value": 0.0, "timestamp_ms": 30},
            },
            "series": {
                "policy/policy_entropy": [
                    {"step": 1, "value": 0.0001, "timestamp_ms": 10},
                    {"step": 2, "value": 0.0001, "timestamp_ms": 20},
                    {"step": 3, "value": 0.0001, "timestamp_ms": 30},
                ],
                "policy/grad_norm": [
                    {"step": 1, "value": 0.0, "timestamp_ms": 10},
                    {"step": 2, "value": 0.0, "timestamp_ms": 20},
                    {"step": 3, "value": 0.0, "timestamp_ms": 30},
                ],
            },
        },
    }
    merged = run_status._with_tracking_metrics(log_signals=signals, tracking=tracking)
    artifacts = {"checkpoint": {"latest": {"step": 3, "resumable": True}}}

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=merged, artifacts=artifacts)

    assert progress["tracking"]["available"] is True
    assert progress["metrics"]["policy/policy_entropy"] == 0.0001
    assert progress["metrics"]["reward"]["avg_pass_at_8"] == 0.75
    assert progress["training_health"]["verdict"] == "deterministic_low_gradient"
    assert progress["learning_signal"]["verdict"] == "deterministic_convergence_risk"
    assert progress["learning_signal"]["trends"]["policy_entropy"]["available"] is True


def test_tracking_status_prefers_live_mlflow_api_for_running_job(monkeypatch):
    def fake_tunnel(**kwargs):
        return (
            {
                "available": True,
                "reason": None,
                "backend": "mlflow_api",
                "tracking_state": "metrics_available",
                "source": {"base_url": "http://127.0.0.1:12345"},
                "latest_step": 1,
                "metric_count": 1,
                "metric_row_count": 1,
                "available_keys": ["loss/avg_final_rewards"],
                "selected_keys": ["loss/avg_final_rewards"],
                "latest": {"loss/avg_final_rewards": {"step": 1, "value": 0.25}},
                "series": {},
            },
            [{"name": "ssh_tunnel:mlflow_api", "ok": True}],
        )

    def fail_run(*args, **kwargs):
        raise AssertionError("GCS SQLite fallback should not be used when live API is available")

    monkeypatch.setattr(run_status, "_read_tracking_via_tunnel", fake_tunnel)
    monkeypatch.setattr(run_status, "_run", fail_run)

    tracking = run_status._tracking_status(
        cluster="w8-biayn-cpp-grpo-rtest",
        pipeline="cpp-grpo",
        active_job={"status": "RUNNING"},
        run_gcs_prefix="gs://bucket/runs/cpp-perf/rtest/cpp-grpo",
        env={},
        timeout_s=30,
        retries=0,
        dry_run=False,
    )

    assert tracking["available"] is True
    assert tracking["mlflow"]["backend"] == "mlflow_api"
    assert tracking["mlflow"]["latest_step"] == 1
    assert tracking["checks"] == [{"name": "ssh_tunnel:mlflow_api", "ok": True}]


def test_progress_summary_does_not_flag_single_low_gradient_event():
    signals = _extract_log_signals(
        [
            "2026-06-18 | INFO | {'policy_entropy': 9.5e-05, 'policy_loss': -0.0009, 'grad_norm': 1.5e-05}",
            "2026-06-18 | INFO | {'policy_entropy': 7.3e-05, 'policy_loss': 0.0, 'grad_norm': 0.022}",
            "2026-06-18 | INFO | {'policy_entropy': 8.3e-05, 'policy_loss': -0.0, 'grad_norm': 0.051}",
            "'loss/avg_final_rewards': '0.9550', 'reward/avg_pass_at_8': '0.8750'",
        ]
    )
    artifacts = {"checkpoint": {"latest": {"step": 250, "resumable": True}}}

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=signals, artifacts=artifacts)

    assert progress["training_health"]["verdict"] == "healthy_or_insufficient_collapse_signal"
    assert progress["training_health"]["should_stop"] is False


def test_progress_summary_uses_w8_grpo_health_learning_signal():
    signals = _extract_log_signals(
        [
            'W8_GRPO_HEALTH {"schema_version": "w8-grpo-health-v1", "step": 200, "metrics": {'
            '"loss/avg_final_rewards": 0.795, "reward/avg_pass_at_8": 0.875, '
            '"policy/policy_entropy": 0.000248, "policy/grad_norm": 0.011, '
            '"policy/policy_loss": -0.000035, "policy/policy_kl": 0.042, '
            '"loss/avg_raw_advantages_abs": 0.015, '
            '"w8/reward_group_variance_mean": 0.0001, '
            '"w8/zero_variance_group_fraction": 0.82, '
            '"w8/zero_advantage_token_fraction": 0.79, '
            '"timing/step": 265.5, "timing/generate": 124.1, "timing/policy_train": 137.0}}',
            'W8_GRPO_HEALTH {"schema_version": "w8-grpo-health-v1", "step": 207, "metrics": {'
            '"loss/avg_final_rewards": 0.8295, "reward/avg_pass_at_8": 0.9375, '
            '"policy/policy_entropy": 0.000251, "policy/grad_norm": 0.009, '
            '"policy/policy_loss": -0.000031, "policy/policy_kl": 0.045, '
            '"loss/avg_raw_advantages_abs": 0.014, '
            '"w8/reward_group_variance_mean": 0.00008, '
            '"w8/zero_variance_group_fraction": 0.86, '
            '"w8/zero_advantage_token_fraction": 0.81, '
            '"timing/step": 260.0, "timing/generate": 120.0, "timing/policy_train": 134.0}}',
        ]
    )
    artifacts = {"checkpoint": {"latest": {"step": 200, "resumable": True}}}

    progress = run_status._progress_summary(pipeline="cpp-grpo", log_signals=signals, artifacts=artifacts)

    assert signals["grpo_health_events"][-1]["step"] == 207
    assert progress["metrics"]["policy/policy_kl"] == 0.045
    assert progress["learning_signal"]["available"] is True
    assert progress["learning_signal"]["verdict"] == "deterministic_convergence_risk"
    assert progress["learning_signal"]["recommended_action"] == "evaluate_checkpoint"
    assert progress["learning_signal"]["kl"]["policy_kl"] == 0.045
    assert progress["learning_signal"]["variance"]["zero_variance_group_fraction"] == 0.86
    assert progress["learning_signal"]["trends"]["avg_final_reward"]["delta"] == 0.0345
    assert progress["training_health"]["verdict"] == "deterministic_convergence_risk"
    assert progress["training_health"]["should_stop"] is False
    assert progress["training_health"]["recommended_action"] == "evaluate_checkpoint"
    assert progress["phase_timing"]["groups"][0]["group"] in {"policy_update", "rollout"}


def test_sft_progress_summary_uses_last_step_metrics_and_expected_final_step():
    signals = _extract_log_signals(
        [
            "python -m skyrl.train.main_sft strategy=fsdp model.path=Qwen/Qwen2.5-Coder-7B-Instruct "
            "dataset_name=/data/sft batch_size=16 num_epochs=2 placement.num_nodes=1 "
            "placement.num_gpus_per_node=8 micro_train_batch_size_per_gpu=1 "
            "ckpt_interval=100 hf_save_interval=100 max_ckpts_to_keep=2 "
            "SKYRL_RAY_PG_TIMEOUT_IN_S=1800 SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600",
            "2026-06-16 | INFO | Step 8: loss=0.2847, grad_norm=2.33",
            "'timing/step': '2.1009', 'timing/forward_backward': '2.0071', "
            "'train/tokens_per_second': '7492.4949'",
        ]
    )

    progress = run_status._progress_summary(
        pipeline="cpp-sft",
        log_signals=signals,
        artifacts={},
        expected_final_step=1074,
    )

    assert progress["primary"] == "training_batches"
    assert progress["training"]["completed"] == 8
    assert progress["training"]["total"] == 1074
    assert progress["training"]["remaining"] == 1066
    assert progress["training"]["timing"]["last_duration_s"] == 2.1009
    assert progress["training"]["timing"]["eta_s"] == 2239.559
    assert progress["training"]["throughput"]["samples_per_hour"] > 27000
    assert progress["checkpoint"]["next_checkpoint_step"] == 100
    assert progress["checkpoint"]["steps_until_next_checkpoint"] == 92
    assert progress["sft_config"]["batch_size"] == 16
    assert progress["sft_config"]["micro_train_batch_size_per_gpu"] == 1
    assert progress["sft_config"]["num_epochs"] == 2
    assert progress["sft_config"]["total_gpu_count"] == 8
    assert progress["sft_config"]["skyrl_ray_pg_timeout_s"] == 1800
    assert progress["sft_config"]["skyrl_worker_nccl_timeout_s"] == 3600
    assert progress["sft_config"]["max_ckpts_to_keep"] == 2
    assert progress["throughput"]["training_samples_per_gpu_hour"] > 3000


def test_progress_summary_uses_rendered_config_after_launch_command_leaves_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "RENDERED_CONFIG_DIR", tmp_path)
    (tmp_path / "cpp-sft.sky.yaml").write_text(
        """
run: |
  export SKYRL_RAY_PG_TIMEOUT_IN_S="${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}"
  export SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-3600}"
  python -m skyrl.train.main_sft \\
    batch_size=16 \\
    micro_train_batch_size_per_gpu=1 \\
    num_epochs=2 \\
    placement.num_nodes=1 \\
    placement.num_gpus_per_node=8 \\
    ckpt_interval=100 \\
    hf_save_interval=100
"""
    )
    signals = _extract_log_signals(
        [
            "2026-06-16 | INFO | Step 699: loss=0.1104, grad_norm=1.07",
            "'timing/step': '1.6074', 'train/tokens_per_second': '9289.4219'",
        ]
    )
    signals = run_status._with_rendered_config_fallback(pipeline="cpp-sft", log_signals=signals)

    progress = run_status._progress_summary(
        pipeline="cpp-sft",
        log_signals=signals,
        artifacts={},
        expected_final_step=1074,
    )

    assert signals["config_sources"] == ["rendered_yaml"]
    assert progress["training"]["samples_per_step"] == 16
    assert progress["training"]["throughput"]["samples_per_hour"] > 35000
    assert progress["checkpoint"]["interval"] == 100
    assert progress["checkpoint"]["next_checkpoint_step"] == 700
    assert progress["checkpoint"]["steps_until_next_checkpoint"] == 1
    assert progress["sft_config"]["batch_size"] == 16
    assert progress["sft_config"]["micro_train_batch_size_per_gpu"] == 1
    assert progress["sft_config"]["total_gpu_count"] == 8
    assert progress["sft_config"]["skyrl_ray_pg_timeout_s"] == 1800
    assert progress["sft_config"]["skyrl_worker_nccl_timeout_s"] == 3600
    assert progress["throughput"]["training_samples_per_gpu_hour"] > 4000


def test_speed_comparison_reports_baseline_rollout_factor():
    current_signals = _extract_log_signals(
        [
            "trainer.train_batch_size=32 generator.n_samples_per_prompt=8 "
            "trainer.placement.policy_num_nodes=2 trainer.placement.policy_num_gpus_per_node=8 "
            "trainer.policy.fsdp_config.fsdp_size=8 trainer.ref.fsdp_config.fsdp_size=8 "
            "generator.inference_engine.num_engines=16",
            "Generating Trajectories: 100%|██████████| 256/256 [01:30<00:00,  2.84it/s]",
        ]
    )
    current = run_status._progress_summary(pipeline="cpp-grpo", log_signals=current_signals, artifacts={})
    assert current["grpo_config"]["effective_samples_per_step"] == 256
    assert current["grpo_config"]["samples_per_gpu_per_step"] == 16
    assert current["grpo_config"]["policy_fsdp_size"] == 8
    assert current["grpo_config"]["ref_fsdp_size"] == 8
    assert current["grpo_config"]["fsdp_mesh_shape"] == {"ddp": 2, "fsdp": 8}
    assert current["grpo_config"]["hsdp_active"] is True
    baseline = {
        "_status_source": "single-node.json",
        "run_id": "rsingle",
        "pipelines": [
            {
                "pipeline": "cpp-grpo",
                "cluster": "w8-biayn-cpp-grpo-rsingle",
                "state": "running",
                "backend": {"queue": {"jobs": [{"resources": "1x[A100:8]"}]}},
                "logs": {
                    "trajectory_progress": {
                        "completed": 128,
                        "total": 128,
                        "percent": 100,
                        "line": "Generating Trajectories: 100%|██████████| 128/128 [00:19<00:00,  6.59it/s]",
                    }
                },
                "artifacts": {},
            }
        ],
    }

    comparison = run_status._speed_comparison(
        pipeline="cpp-grpo",
        current_progress=current,
        baseline_statuses=[baseline],
    )

    assert comparison["available"] is True
    primary = comparison["primary"]
    assert primary["source"] == "single-node.json"
    assert primary["resources"] == "1x[A100:8]"
    assert primary["rollout_speedup_factor"] == 0.431
    assert primary["rollout_samples_per_gpu_second_speedup_factor"] == 0.2155
    assert primary["gpu_scale_factor"] == 2
    assert primary["gpu_speedup_efficiency"] == 0.2155
    assert primary["cost_verdict"] == "cost_inefficient"
    assert primary["verdict"] == "slower"
    assert primary["message"] == (
        "Current run achieved 0.431x baseline throughput on the primary comparable metric; "
        "lower than 1.0 means slower."
    )
    assert primary["baseline"]["gpu_count"] == 8
    assert primary["baseline"]["rollout_samples_per_second"] == 6.59
    assert primary["baseline"]["rollout_samples_per_gpu_second"] == 0.82375
    assert primary["current"]["gpu_count"] == 16.0
    assert primary["current"]["rollout_samples_per_second"] == 2.84
    assert primary["current"]["rollout_samples_per_gpu_second"] == 0.1775


def test_artifact_status_reports_in_progress_checkpoint(monkeypatch):
    def fake_cat(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        assert uri.endswith("latest_ckpt_global_step.txt")
        return "600\n", {"name": f"cat:{uri}", "ok": True}

    def fake_ls(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        if uri.endswith("/ckpts/"):
            return (
                "\n".join(
                    [
                        "gs://bucket/runs/cpp-perf/run/cpp-sft/ckpts/global_step_600/",
                        "gs://bucket/runs/cpp-perf/run/cpp-sft/ckpts/global_step_700/",
                    ]
                ),
                {"name": f"ls:{uri}", "ok": True},
            )
        return "", {"name": f"ls:{uri}", "ok": False}

    def fake_checkpoint_detail(checkpoint_prefix: str, **_: Any) -> dict[str, Any]:
        step = int(checkpoint_prefix.rsplit("_", 1)[-1])
        return {
            "prefix": checkpoint_prefix,
            "trainer_state_present": step == 600,
            "resumable": step == 600,
            "checks": [{"name": f"detail:{step}", "ok": True}],
        }

    monkeypatch.setattr(run_status, "_storage_cat", fake_cat)
    monkeypatch.setattr(run_status, "_storage_ls", fake_ls)
    monkeypatch.setattr(run_status, "_checkpoint_detail", fake_checkpoint_detail)

    status = run_status._artifact_status(
        pipeline="cpp-sft",
        run_gcs_prefix="gs://bucket/runs/cpp-perf/run/cpp-sft",
        env={},
        expected_world_size=8,
        expected_final_step=1074,
        timeout_s=1,
        retries=0,
        dry_run=False,
    )

    checkpoint = status["checkpoint"]
    assert checkpoint["latest"]["step"] == 600
    assert checkpoint["latest"]["promoted"] is True
    assert checkpoint["highest"]["step"] == 700
    assert checkpoint["highest"]["promoted"] is False
    assert checkpoint["in_progress"]["step"] == 700
    assert checkpoint["in_progress"]["resumable"] is False


def test_artifact_status_uses_highest_export_when_final_step_unknown(monkeypatch):
    def fake_cat(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        assert uri.endswith("latest_ckpt_global_step.txt")
        return "1074\n", {"name": f"cat:{uri}", "ok": True}

    def fake_ls(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        if uri.endswith("/ckpts/"):
            return (
                "\n".join(
                    [
                        "gs://bucket/runs/cpp-perf/run/cpp-sft/ckpts/global_step_1000/",
                        "gs://bucket/runs/cpp-perf/run/cpp-sft/ckpts/global_step_1074/",
                    ]
                ),
                {"name": f"ls:{uri}", "ok": True},
            )
        if uri.endswith("/exports/"):
            return (
                "\n".join(
                    [
                        "gs://bucket/runs/cpp-perf/run/cpp-sft/exports/global_step_1000/",
                        "gs://bucket/runs/cpp-perf/run/cpp-sft/exports/global_step_1074/",
                    ]
                ),
                {"name": f"ls:{uri}", "ok": True},
            )
        return "", {"name": f"ls:{uri}", "ok": False}

    def fake_checkpoint_detail(checkpoint_prefix: str, **_: Any) -> dict[str, Any]:
        return {"prefix": checkpoint_prefix, "resumable": True, "checks": []}

    def fake_export_detail(export_prefix: str, **_: Any) -> dict[str, Any]:
        return {
            "prefix": export_prefix,
            "has_config": True,
            "weight_object_count": 1,
            "complete": True,
            "checks": [],
        }

    monkeypatch.setattr(run_status, "_storage_cat", fake_cat)
    monkeypatch.setattr(run_status, "_storage_ls", fake_ls)
    monkeypatch.setattr(run_status, "_checkpoint_detail", fake_checkpoint_detail)
    monkeypatch.setattr(run_status, "_export_detail", fake_export_detail)

    status = run_status._artifact_status(
        pipeline="cpp-sft",
        run_gcs_prefix="gs://bucket/runs/cpp-perf/run/cpp-sft",
        env={},
        expected_world_size=8,
        expected_final_step=None,
        timeout_s=1,
        retries=0,
        dry_run=False,
    )

    export = status["export"]
    assert export["expected_final_step"] is None
    assert export["final_export_step"] == 1074
    assert export["final_export_prefix"] == "gs://bucket/runs/cpp-perf/run/cpp-sft/exports/global_step_1074/policy"
    assert export["final_export_exists"] is True
    assert export["final_export"]["complete"] is True


def test_artifact_status_reports_eval_output_files(monkeypatch):
    def fake_ls(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        assert uri == "gs://bucket/runs/cpp-perf/run/cpp-eval/**"
        return (
            "\n".join(
                [
                    "gs://bucket/runs/cpp-perf/run/cpp-eval/w8-cpp-eval-run/base.records.jsonl",
                    "gs://bucket/runs/cpp-perf/run/cpp-eval/w8-cpp-eval-run/base.summary.json",
                    "gs://bucket/runs/cpp-perf/run/cpp-eval/w8-cpp-eval-run/sft.records.jsonl",
                ]
            ),
            {"name": f"ls:{uri}", "ok": True},
        )

    monkeypatch.setattr(run_status, "_storage_ls", fake_ls)

    status = run_status._artifact_status(
        pipeline="cpp-eval",
        run_gcs_prefix="gs://bucket/runs/cpp-perf/run/cpp-eval",
        env={},
        expected_world_size=8,
        expected_final_step=None,
        timeout_s=1,
        retries=0,
        dry_run=False,
    )

    outputs = status["eval_outputs"]
    assert outputs["labels"] == ["base", "sft"]
    assert outputs["complete_labels"] == ["base"]
    assert outputs["records"] == [
        {
            "label": "base",
            "uri": "gs://bucket/runs/cpp-perf/run/cpp-eval/w8-cpp-eval-run/base.records.jsonl",
        },
        {
            "label": "sft",
            "uri": "gs://bucket/runs/cpp-perf/run/cpp-eval/w8-cpp-eval-run/sft.records.jsonl",
        },
    ]
    assert outputs["summaries"] == [
        {
            "label": "base",
            "uri": "gs://bucket/runs/cpp-perf/run/cpp-eval/w8-cpp-eval-run/base.summary.json",
        }
    ]


def test_pipeline_state_reports_provisioning_before_checkpointed():
    state = run_status._derive_pipeline_state(
        queue={"jobs": []},
        artifacts={"checkpoint": {"latest_marker": 1074}},
        log_signals={"errors": []},
        instances=[{"status": "STAGING"}],
    )

    assert state == "provisioning"


def test_cleanup_is_not_safe_while_instance_is_provisioning(monkeypatch):
    monkeypatch.setattr(
        run_status,
        "_gcp_instances",
        lambda **_: (
            [{"status": "STAGING", "labels": {"pipeline": "cpp-sft"}}],
            {"name": "instances", "ok": True},
        ),
    )
    monkeypatch.setattr(
        run_status,
        "_dataset_stage",
        lambda *_, **__: {"state": "not_configured", "checks": []},
    )
    monkeypatch.setattr(
        run_status,
        "_pipeline_status",
        lambda **_: {
            "pipeline": "cpp-sft",
            "state": "provisioning",
            "backend": {"queue": {"jobs": []}},
        },
    )

    status = run_status.build_run_status(
        run_id="run",
        project_id="project",
        artifact_bucket="gs://bucket",
        env={},
        pipelines=["cpp-sft"],
    )

    assert status["cleanup"]["safe_to_cleanup"] is False
    assert status["cleanup"]["provisioning_instance_count"] == 1
    assert status["cleanup"]["working_pipeline_count"] == 1


def test_pipeline_state_prefers_active_queue_over_log_errors():
    state = run_status._derive_pipeline_state(
        queue={"jobs": [{"status": "RUNNING"}]},
        artifacts={},
        log_signals={"errors": ["Traceback (most recent call last):"]},
        instances=[{"status": "RUNNING"}],
    )

    assert state == "running"


def test_pipeline_state_prefers_active_retry_over_old_failures():
    state = run_status._derive_pipeline_state(
        queue={"jobs": [{"status": "RUNNING"}, {"status": "FAILED"}, {"status": "CANCELLED"}]},
        artifacts={},
        log_signals={"errors": []},
        instances=[{"status": "RUNNING"}],
    )

    assert state == "running"


def test_pipeline_state_ignores_old_failed_jobs_after_successful_retry():
    state = run_status._derive_pipeline_state(
        queue={"jobs": [{"status": "SUCCEEDED"}, {"status": "FAILED"}, {"status": "CANCELLED"}]},
        artifacts={"export": {"final_export_exists": True}},
        log_signals={"errors": []},
        instances=[{"status": "RUNNING"}],
    )

    assert state == "complete"


def test_pipeline_state_marks_pre_metric_tracking_without_live_backend_failed():
    state = run_status._derive_pipeline_state(
        queue={"jobs": []},
        artifacts={"checkpoint": {"latest_marker": None}, "export": {"final_export_exists": False}},
        log_signals={"errors": []},
        instances=[],
        tracking={
            "mlflow": {
                "tracking_state": "run_active_no_metrics",
                "metric_count": 0,
                "metric_row_count": 0,
            }
        },
    )

    assert state == "failed"


def test_pipeline_state_keeps_live_pre_metric_tracking_running():
    state = run_status._derive_pipeline_state(
        queue={"jobs": [{"status": "RUNNING"}]},
        artifacts={"checkpoint": {"latest_marker": None}, "export": {"final_export_exists": False}},
        log_signals={"errors": []},
        instances=[{"status": "RUNNING"}],
        tracking={
            "mlflow": {
                "tracking_state": "run_active_no_metrics",
                "metric_count": 0,
                "metric_row_count": 0,
            }
        },
    )

    assert state == "running"


def test_run_retries_nonzero_read_only_checks(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(args=args[0], returncode=255, stdout="", stderr="ssh failed")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(run_status.subprocess, "run", fake_run)

    result = run_status._run(["sky", "logs"], env={}, timeout_s=10, retries=1, dry_run=False)

    assert result.ok is True
    assert result.stdout == "ok"
    assert result.attempt_count == 2
    assert len(calls) == 2


def test_run_summary_prefers_failed_pipeline_when_no_active_job():
    summary = run_status._run_summary(
        pipelines=[
            {
                "pipeline": "cpp-sft",
                "cluster": "sft",
                "state": "not_started",
                "progress": {"primary": None},
            },
            {
                "pipeline": "cpp-grpo",
                "cluster": "grpo",
                "state": "failed",
                "active_job": {"status": "FAILED"},
                "phase": {"current": "failed"},
                "progress": {"primary": {"completed": 0}},
            },
        ],
        dataset={"state": "available"},
        instances=[],
        cleanup={"safe_to_cleanup": True},
    )

    assert summary["state"] == "failed"
    assert summary["current_pipeline"] == "cpp-grpo"
    assert summary["current_phase"] == {"current": "failed"}


def test_phase_summary_marks_terminal_queue_failure_without_logs():
    phase = run_status._phase_summary(
        log_signals={"stage": None, "errors": []},
        node_health=None,
        artifacts={},
        active_job={"status": "FAILED"},
    )

    assert phase["current"] == "failed"
    assert phase["source"] == "queue"
    assert phase["failed"] is True


def test_phase_summary_marks_pre_metric_tracking_failure_without_logs():
    phase = run_status._phase_summary(
        log_signals={"stage": None, "errors": []},
        node_health=None,
        artifacts={},
        active_job=None,
        pipeline_state="failed",
        startup={"recommended_action": "inspect_failed_startup_or_relaunch"},
    )

    assert phase["current"] == "failed"
    assert phase["source"] == "tracking"
    assert phase["failed"] is True
    assert phase["message"] == (
        "MLflow registered the run but no scalar training metrics were recorded before backend resources disappeared."
    )


def test_pipeline_status_reads_logs_for_selected_active_retry(monkeypatch):
    observed: dict[str, str | None] = {}

    monkeypatch.setattr(
        run_status,
        "_sky_queue",
        lambda *_, **__: (
            {
                "jobs": [
                    {"job_id": 1, "status": "FAILED", "resources": "1x[A100:8]"},
                    {"job_id": 2, "status": "RUNNING", "resources": "1x[A100:8]"},
                ]
            },
            {"name": "sky_queue", "ok": True},
        ),
    )

    def fake_log_signals(_cluster: str, *, job_id: str | None, **_kwargs):
        observed["job_id"] = job_id
        return (
            {
                "stage": "data_restore",
                "errors": [],
                "config": {},
                "metrics": {},
                "timings": {},
                "stage_events": [],
                "checkpoint_events": [],
                "export_events": [],
            },
            {"name": "sky_logs", "ok": True},
        )

    monkeypatch.setattr(run_status, "_log_signals", fake_log_signals)
    monkeypatch.setattr(
        run_status,
        "_artifact_status",
        lambda **_: {"checkpoint": {}, "export": {}, "checks": []},
    )

    status = run_status._pipeline_status(
        pipeline="cpp-sft",
        run_id="rretry",
        artifact_bucket="gs://bucket",
        env={},
        instances=[{"status": "RUNNING"}],
        log_tail=800,
        job_id=None,
        expected_world_size=8,
        expected_final_step=None,
        include_node_health=False,
        baseline_statuses=[],
        timeout_s=180,
        retries=0,
        dry_run=False,
    )

    assert observed["job_id"] == "2"
    assert status["active_job"]["job_id"] == 2
    assert status["state"] == "running"


def test_pipeline_phase_uses_active_in_progress_checkpoint_when_log_tail_is_stale(monkeypatch):
    monkeypatch.setattr(
        run_status,
        "_sky_queue",
        lambda *_, **__: (
            {"jobs": [{"job_id": 1, "status": "RUNNING", "resources": "1x[A100:8]"}]},
            {"name": "sky_queue", "ok": True},
        ),
    )
    monkeypatch.setattr(
        run_status,
        "_log_signals",
        lambda *_, **__: (
            {
                "stage": "container_pull",
                "errors": [],
                "config": {},
                "metrics": {},
                "timings": {},
                "stage_events": [],
                "checkpoint_events": [],
                "export_events": [],
                "tail_may_be_truncated": True,
            },
            {"name": "sky_logs", "ok": True},
        ),
    )
    monkeypatch.setattr(
        run_status,
        "_artifact_status",
        lambda **_: {
            "checkpoint": {
                "latest_marker": 700,
                "latest": {"step": 700, "resumable": True},
                "highest": {"step": 800, "resumable": False},
                "in_progress": {"step": 800, "resumable": False},
            },
            "export": {},
            "checks": [],
        },
    )

    status = run_status._pipeline_status(
        pipeline="cpp-sft",
        run_id="rcheckpoint",
        artifact_bucket="gs://bucket",
        env={},
        instances=[{"status": "RUNNING"}],
        log_tail=2500,
        job_id=None,
        expected_world_size=8,
        expected_final_step=1074,
        include_node_health=False,
        baseline_statuses=[],
        timeout_s=180,
        retries=0,
        dry_run=False,
    )

    assert status["phase"]["current"] == "checkpoint_stage"
    assert status["phase"]["source"] == "artifacts"
    assert status["phase"]["log_stage"] == "container_pull"
    assert status["phase"]["artifact_activity"] == "checkpoint_stage"


def test_pipeline_recovery_recommends_fresh_cluster_after_nccl_failure(monkeypatch):
    monkeypatch.setattr(
        run_status,
        "_sky_queue",
        lambda *_, **__: (
            {"jobs": [{"job_id": 7, "status": "FAILED", "resources": "1x[A100:8]"}]},
            {"name": "sky_queue", "ok": True},
        ),
    )
    monkeypatch.setattr(
        run_status,
        "_log_signals",
        lambda *_, **__: (
            {
                "stage": "failed",
                "errors": [
                    "RuntimeError: NCCL Error 1: unhandled cuda error",
                    "skyrl.train.fsdp_utils:fsdp2_load_full_state_dict distribute_tensor scatter failed",
                ],
                "config": {},
                "metrics": {},
                "timings": {},
                "stage_events": [{"stage": "checkpoint_load", "line": "Loading checkpoint"}],
                "checkpoint_events": [],
                "export_events": [],
            },
            {"name": "sky_logs", "ok": True},
        ),
    )
    monkeypatch.setattr(
        run_status,
        "_artifact_status",
        lambda **_: {
            "checkpoint": {
                "latest_marker": 700,
                "latest": {"step": 700, "resumable": True},
                "highest": {"step": 700, "resumable": True},
                "in_progress": None,
            },
            "export": {},
            "checks": [],
        },
    )

    status = run_status._pipeline_status(
        pipeline="cpp-sft",
        run_id="rfresh",
        artifact_bucket="gs://bucket",
        env={},
        instances=[{"status": "RUNNING"}],
        log_tail=2500,
        job_id=None,
        expected_world_size=8,
        expected_final_step=1074,
        include_node_health=False,
        baseline_statuses=[],
        timeout_s=180,
        retries=0,
        dry_run=False,
    )

    assert status["state"] == "failed"
    assert status["recovery"]["available"] is True
    assert status["recovery"]["recommended_action"] == "fresh_cluster_resume"
    assert status["recovery"]["fresh_cluster_recommended"] is True
    assert status["recovery"]["requires_down_before_resume"] is True
    assert status["recovery"]["resume_from"] == "latest"
    assert status["recovery"]["resume_checkpoint_step"] == 700
    assert status["recovery"]["signals"] == ["distributed_state_failure"]
    assert status["recovery"]["commands"]["down"] == ["uv", "run", "w8-biayn", "ops", "down", status["cluster"]]


def test_export_detail_requires_model_weights(monkeypatch):
    listing = "\n".join(
        [
            "      1376  2026-06-14T17:24:04Z  gs://bucket/export/policy/config.json",
            "  11421991  2026-06-14T17:24:06Z  gs://bucket/export/policy/tokenizer.json",
            "TOTAL: 2 objects, 11423367 bytes (10.9 MiB)",
        ]
    )

    def fake_ls(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        assert uri == "gs://bucket/export/policy/"
        return listing, {"name": f"ls:{uri}", "ok": True}

    monkeypatch.setattr(run_status, "_storage_ls", fake_ls)

    detail = run_status._export_detail(
        "gs://bucket/export/policy",
        env={},
        timeout_s=1,
        retries=0,
        dry_run=False,
    )

    assert detail["has_config"] is True
    assert detail["has_tokenizer"] is True
    assert detail["weight_object_count"] == 0
    assert detail["complete"] is False


def test_export_detail_accepts_weight_shards(monkeypatch):
    listing = "\n".join(
        [
            "      1376  2026-06-14T17:24:04Z  gs://bucket/export/policy/config.json",
            "  11421991  2026-06-14T17:24:06Z  gs://bucket/export/policy/tokenizer.json",
            "1073741824  2026-06-14T17:24:10Z  gs://bucket/export/policy/model-00001-of-00002.safetensors",
            "1073741824  2026-06-14T17:24:11Z  gs://bucket/export/policy/model-00002-of-00002.safetensors",
            "TOTAL: 4 objects, 2158905639 bytes (2.0 GiB)",
        ]
    )

    def fake_ls(uri: str, **_: Any) -> tuple[str, dict[str, Any]]:
        assert uri == "gs://bucket/export/policy/"
        return listing, {"name": f"ls:{uri}", "ok": True}

    monkeypatch.setattr(run_status, "_storage_ls", fake_ls)

    detail = run_status._export_detail(
        "gs://bucket/export/policy",
        env={},
        timeout_s=1,
        retries=0,
        dry_run=False,
    )

    assert detail["weight_object_count"] == 2
    assert detail["weight_bytes"] == 2147483648
    assert detail["complete"] is True


def test_parse_node_health_structures_gpu_disk_and_processes():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 42, 16070, 40960",
                "1, 0, 11302, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "25443 712 37.7 1.4 ray::FSDPPolicyWorkerBase.load_checkpoint",
            ]
        )
    )

    assert payload["gpus"][0] == {
        "index": 0,
        "utilization_gpu_percent": 42,
        "memory_used_mib": 16070,
        "memory_total_mib": 40960,
    }
    assert payload["filesystems"][0]["available_kb"] == 577765376
    assert payload["processes"][0]["cmd"] == "ray::FSDPPolicyWorkerBase.load_checkpoint"
    assert payload["processes"][0]["cpu_percent"] == 37.7
    assert payload["activity"] == "checkpoint_load"


def test_node_health_reports_all_active_pipeline_nodes(monkeypatch):
    calls = []

    def fake_run(args: list[str], **_: Any) -> run_status.CommandResult:
        calls.append(args)
        name = args[3]
        gpu_line = "0, 99, 8192, 40960" if "head" in name else "0, 100, 32768, 40960"
        return run_status.CommandResult(
            command=args,
            returncode=0,
            stdout="\n".join(
                [
                    "__W8_GPU__",
                    gpu_line,
                    "__W8_DF__",
                    "Filesystem 1024-blocks Used Available Capacity Mounted on",
                    "/dev/root 1041235968 463470592 577765376 45% /",
                    "__W8_PS__",
                    "PID ELAPSED %CPU %MEM CMD",
                    "25443 712 37.7 1.4 ray::FSDPPolicyWorkerBase.forward",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(run_status, "_run", fake_run)

    payload = run_status._node_health(
        "w8-biayn-cpp-grpo-run",
        project_id="project",
        instances=[
            {
                "name": "w8-biayn-cpp-grpo-run-worker-1",
                "zone": "asia-northeast3-b",
                "status": "RUNNING",
                "labels": {"ray-node-type": "worker"},
            },
            {
                "name": "w8-biayn-cpp-grpo-run-head",
                "zone": "asia-northeast3-b",
                "status": "RUNNING",
                "labels": {"ray-node-type": "head"},
            },
        ],
        env={},
        timeout_s=1,
        retries=0,
        dry_run=False,
        enabled=True,
    )

    assert [call[:4] for call in calls] == [
        ["gcloud", "compute", "ssh", "w8-biayn-cpp-grpo-run-head"],
        ["gcloud", "compute", "ssh", "w8-biayn-cpp-grpo-run-worker-1"],
    ]
    assert payload["sample_scope"] == "all_active"
    assert payload["sampled_node_count"] == 2
    assert payload["failed_node_count"] == 0
    assert len(payload["gpus"]) == 2
    assert payload["gpus"][0]["node_name"] == "w8-biayn-cpp-grpo-run-head"
    assert payload["gpus"][1]["role"] == "worker"
    assert payload["gpus"][1]["global_index"] == 1
    assert payload["nodes"][0]["role"] == "head"
    assert payload["nodes"][0]["gpus"][0]["utilization_gpu_percent"] == 99
    assert payload["nodes"][1]["role"] == "worker"
    assert payload["nodes"][1]["gpus"][0]["utilization_gpu_percent"] == 100
    assert payload["activity"] == "policy_forward"
    resources = run_status._resource_summary(
        instances=[
            {"status": "RUNNING", "accelerators": [{"type": "nvidia-tesla-a100", "count": 1}]},
            {"status": "RUNNING", "accelerators": [{"type": "nvidia-tesla-a100", "count": 1}]},
        ],
        node_health=payload,
    )
    assert resources["sampled_node_count"] == 2
    assert resources["failed_node_count"] == 0
    assert resources["sampled_gpu_count"] == 2
    assert resources["avg_gpu_utilization_percent"] == 99.5


def test_node_health_reports_partial_worker_probe_failure(monkeypatch):
    def fake_run(args: list[str], **_: Any) -> run_status.CommandResult:
        name = args[3]
        if "worker" in name:
            return run_status.CommandResult(
                command=args,
                returncode=255,
                stdout="",
                stderr="ssh: connect failed",
            )
        return run_status.CommandResult(
            command=args,
            returncode=0,
            stdout="\n".join(
                [
                    "__W8_GPU__",
                    "0, 77, 12000, 40960",
                    "__W8_DF__",
                    "Filesystem 1024-blocks Used Available Capacity Mounted on",
                    "/dev/root 1041235968 463470592 577765376 45% /",
                    "__W8_PS__",
                    "PID ELAPSED %CPU %MEM CMD",
                    "25443 712 37.7 1.4 ray::FSDPPolicyWorkerBase.forward_backward",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(run_status, "_run", fake_run)

    payload = run_status._node_health(
        "w8-biayn-cpp-grpo-run",
        project_id="project",
        instances=[
            {
                "name": "w8-biayn-cpp-grpo-run-head",
                "zone": "asia-northeast3-b",
                "status": "RUNNING",
                "labels": {"ray-node-type": "head"},
            },
            {
                "name": "w8-biayn-cpp-grpo-run-worker-1",
                "zone": "asia-northeast3-b",
                "status": "RUNNING",
                "labels": {"ray-node-type": "worker"},
            },
        ],
        env={},
        timeout_s=1,
        retries=0,
        dry_run=False,
        enabled=True,
    )

    assert payload["sample_scope"] == "partial"
    assert payload["sampled_node_count"] == 1
    assert payload["failed_node_count"] == 1
    assert payload["nodes"][1]["sampled"] is False
    assert payload["nodes"][1]["error"] == "ssh: connect failed"
    assert payload["error"].startswith("w8-biayn-cpp-grpo-run-worker-1:")
    assert payload["activity"] == "policy_update"


def test_parse_node_health_marks_container_pull_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 0, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "25443 22 45.4 0.1 /usr/bin/unpigz -d -c",
                "25444 22 37.0 0.4 /usr/bin/dockerd -H fd://",
            ]
        )
    )

    assert payload["activity"] == "container_pull"


def test_parse_node_health_marks_docker_runtime_without_layer_extract():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 0, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "25444 22 37.0 0.4 /usr/bin/dockerd -H fd://",
            ]
        )
    )

    assert payload["activity"] == "docker_runtime"


def test_parse_node_health_prefers_checkpoint_worker_over_background_docker():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 32103, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 188860928 851570080 19% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "26744 2558 27.4 1.6 ray::FSDPPolicyWorkerBase.save_checkpoint",
                "1636 3475 10.2 0.0 /usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock",
            ]
        )
    )

    assert payload["activity"] == "checkpoint_stage"


def test_parse_node_health_prefers_dataset_restore_over_background_docker():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 0, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                (
                    "25443 22 60.6 0.1 /snap/google-cloud-cli/273/lib/gcloud.py "
                    "storage cp --recursive gs://bucket/datasets/cpp-perf/skyrl/* "
                    "/home/gcpuser/.w8-biayn/data/skyrl/"
                ),
                "25444 22 31.6 0.4 /usr/bin/dockerd -H fd://",
            ]
        )
    )

    assert payload["activity"] == "data_restore"


def test_parse_node_health_marks_mounted_model_stage_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 0, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                (
                    "25443 22 60.6 0.1 /snap/google-cloud-cli/273/lib/gcloud.py "
                    "storage cp --recursive gs://bucket/runs/cpp-sft/exports/global_step_10/policy/* "
                    "/home/gcpuser/.w8-biayn/runs/r1/cpp-grpo/model/"
                ),
            ]
        )
    )

    assert payload["activity"] == "model_stage"


def test_parse_node_health_reports_ref_model_init_startup():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 100, 8850, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "15010 3330 199.0 2.3 ray::FSDPRefWorkerBase.init_model",
                "14193 3349 1.1 0.1 ray::FSDPPolicyWorkerBase",
            ]
        )
    )

    assert payload["activity"] == "model_init"
    assert payload["startup"]["available"] is True
    assert payload["startup"]["active_stage"] == "ref_model_init"
    assert payload["startup"]["long_running"] is True
    ref_group = payload["startup"]["groups"][0]
    assert ref_group["stage"] == "ref_model_init"
    assert ref_group["process_count"] == 1
    assert ref_group["max_elapsed_s"] == 3330


def test_parse_node_health_marks_ref_model_init_long_running_at_15_minutes():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 100, 8850, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "15010 901 199.0 2.3 ray::FSDPRefWorkerBase.init_model",
            ]
        )
    )

    assert payload["startup"]["active_stage"] == "ref_model_init"
    assert payload["startup"]["long_running"] is True
    assert payload["startup"]["long_running_threshold_s"] == 900


def test_startup_progress_warns_when_mlflow_has_no_scalars():
    node_health = {
        "startup": {
            "available": True,
            "active_stage": "ref_model_init",
            "long_running": True,
            "max_elapsed_s": 3330,
            "groups": [],
        }
    }
    tracking = {
        "mlflow": {
            "tracking_state": "run_active_no_metrics",
            "metric_count": 0,
            "metric_row_count": 0,
        }
    }

    progress = run_status._startup_progress_summary(node_health=node_health, tracking=tracking, pipeline_state="running")

    assert progress["active_stage"] == "ref_model_init"
    assert progress["scalar_metrics_available"] is False
    assert progress["severity"] == "warning"
    assert progress["recommended_action"] == "inspect_startup_stage"


def test_startup_progress_marks_pre_metric_after_teardown_as_failed_startup():
    tracking = {
        "mlflow": {
            "tracking_state": "run_active_no_metrics",
            "metric_count": 0,
            "metric_row_count": 0,
        }
    }

    progress = run_status._startup_progress_summary(
        node_health=None,
        tracking=tracking,
        pipeline_state="failed",
        active_job=None,
        instances=[],
    )

    assert progress["scalar_metrics_available"] is False
    assert progress["severity"] == "error"
    assert progress["recommended_action"] == "inspect_failed_startup_or_relaunch"


def test_startup_progress_keeps_live_pre_metric_action_as_node_health_poll():
    tracking = {
        "mlflow": {
            "tracking_state": "run_active_no_metrics",
            "metric_count": 0,
            "metric_row_count": 0,
        }
    }

    progress = run_status._startup_progress_summary(
        node_health=None,
        tracking=tracking,
        pipeline_state="running",
        active_job={"status": "RUNNING"},
        instances=[{"status": "RUNNING"}],
    )

    assert progress["severity"] == "warning"
    assert progress["recommended_action"] == "poll_with_node_health"


def test_parse_node_health_marks_reward_compile_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 34072, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                (
                    "25443 22 196.0 0.1 /usr/local/libexec/gcc/x86_64-linux-gnu/13.4.0/cc1plus "
                    "-quiet candidate.cpp -O3 -std=c++20 -o /tmp/out.s"
                ),
            ]
        )
    )

    assert payload["activity"] == "reward_compile"


def test_parse_node_health_marks_candidate_execution_as_reward_benchmark():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 34072, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "197756 2 65.5 0.0 ./candidate",
                "1637 4162 10.3 0.0 /usr/bin/dockerd -H fd://",
            ]
        )
    )

    assert payload["activity"] == "reward_benchmark"


def test_parse_node_health_marks_skyrl_startup_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 3, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                (
                    "245380 89 40.4 0.1 python -m w8_biayn.integrations.skyrl_cpp_perf_main "
                    "environment.skyrl_gym.max_env_workers=128 generator.batched=false"
                ),
                "251466 64 69.1 0.1 ray::skyrl_entrypoint",
            ]
        )
    )

    assert payload["activity"] == "skyrl_startup"


def test_parse_node_health_marks_cpp_eval_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 0, 37247, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 81766548 958664460 8% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                (
                    "28796 670 30.2 10.5 python -m w8_biayn.integrations.cpp_eval_main "
                    "--data-dir /data --model /artifacts/model --label sft"
                ),
            ]
        )
    )

    assert payload["activity"] == "evaluation"


def test_parse_node_health_marks_policy_update_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 100, 18322, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "257055 327 116.0 0.8 ray::FSDPPolicyWorkerBase.forward_backward",
                "253989 411 14.9 0.1 VLLM::EngineCore",
            ]
        )
    )

    assert payload["activity"] == "policy_update"


def test_parse_node_health_marks_rollout_inference_activity():
    payload = run_status._parse_node_health(
        "\n".join(
            [
                "__W8_GPU__",
                "0, 100, 8728, 40960",
                "__W8_DF__",
                "Filesystem 1024-blocks Used Available Capacity Mounted on",
                "/dev/root 1041235968 463470592 577765376 45% /",
                "__W8_PS__",
                "PID ELAPSED %CPU %MEM CMD",
                "253989 411 14.9 0.1 VLLM::EngineCore",
            ]
        )
    )

    assert payload["activity"] == "rollout_inference"
