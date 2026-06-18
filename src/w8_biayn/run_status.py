"""Machine-readable run status snapshots for C++ performance-RL operations."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIPELINES = ("cpp-sft", "cpp-grpo", "cpp-eval")
CHECKPOINT_PIPELINES = {"cpp-sft", "cpp-grpo"}
DEFAULT_COMMAND_TIMEOUT_S = 180
RENDERED_CONFIG_DIR = Path(".w8-biayn/rendered")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"Step\s+(\d+):")
LOSS_RE = re.compile(r"Step\s+(\d+):\s+loss=([0-9.eE+-]+)")
EVAL_LOSS_RE = re.compile(r"Step\s+(\d+):\s+eval_loss=([0-9.eE+-]+)")
PROGRESS_RE = re.compile(r"(Generating Trajectories|Evaluation Progress):\s+(\d+)%.*?\|\s*(\d+)/(\d+)")
TRAIN_BATCH_PROGRESS_RE = re.compile(r"Training Batches Processed:\s+(\d+)%.*?\|\s*(\d+)/(\d+)")
TQDM_TIMING_RE = re.compile(r"\[([^<\]]+)<([^,\]]+)(?:,\s*([^\]]+))?\]")
STEP_FINISHED_RE = re.compile(r"Finished: 'step', time cost:\s+([0-9.]+)s")
GENERATE_FINISHED_RE = re.compile(r"Finished: 'generate', time cost:\s+([0-9.]+)s")
SYNC_FINISHED_RE = re.compile(r"Finished: 'sync_weights', time cost:\s+([0-9.]+)s")
SAVE_START_RE = re.compile(r"Saving checkpoint at step\s+(\d+)\s+to\s+(\S+)")
SAVE_DONE_RE = re.compile(r"Checkpoint saved for global_step_(\d+)")
SAVE_FINISHED_RE = re.compile(r"Finished: 'save_checkpoint'.*time cost:\s+([0-9.]+)s")
HF_EXPORT_RE = re.compile(
    r"(Saving final HF model|Saved HF model weights|save_hf_model|HF export)",
    re.IGNORECASE,
)
STAGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("container_pull", re.compile(r"(docker pull|Pulling from|Pull complete|Download complete)", re.IGNORECASE)),
    ("container_start", re.compile(r"\bdocker run\b", re.IGNORECASE)),
    ("data_restore", re.compile(r"gcloud storage cp --recursive .*datasets/.*/skyrl", re.IGNORECASE)),
    ("model_stage", re.compile(r"gcloud storage cp --recursive .*runs/.*/exports/.*/policy", re.IGNORECASE)),
    ("dependency_setup", re.compile(r"(uv sync|pip install|Resolved \d+ packages|Installed \d+ packages|huggingface-hub==)", re.IGNORECASE)),
    ("worker_init", re.compile(r"(Started a local Ray instance|Initializing process group|Initialized process group|Mesh Ranks)", re.IGNORECASE)),
    ("model_init", re.compile(r"(Initializing new inference client|init policy/ref/critic models done|create_inference_servers)", re.IGNORECASE)),
    ("weight_sync", re.compile(r"(init_weight_sync_state|sync_weights|Initialized weight sync state)", re.IGNORECASE)),
    ("checkpoint_load", re.compile(r"(load_checkpoint|Loading checkpoint|Loaded checkpoint|restore checkpoint)", re.IGNORECASE)),
    ("checkpoint_stage", re.compile(r"(Started: 'save_checkpoint'|Saving checkpoint at step|Checkpoint saved for global_step_|Finished: 'save_checkpoint')", re.IGNORECASE)),
    ("trajectory_generation", re.compile(r"(Generating Trajectories:|Started: 'generate'|Finished: 'generate'|Capping concurrency for generation)", re.IGNORECASE)),
    ("training", re.compile(r"(Training Batches Processed:|Started: 'step'|Finished: 'step')", re.IGNORECASE)),
    ("evaluation", re.compile(r"Evaluation Progress:", re.IGNORECASE)),
    ("hf_export", re.compile(r"(Saving final HF model|save_hf_model|HF export)", re.IGNORECASE)),
    ("hf_export_done", re.compile(r"Saved HF model weights", re.IGNORECASE)),
    ("artifact_upload", re.compile(r"gcloud storage cp --recursive (\"?\$W8_ARTIFACT_DIR/exports|/[^ ]+/exports)(\s|\"|$)", re.IGNORECASE)),
    ("job_complete", re.compile(r"Job finished|SUCCEEDED", re.IGNORECASE)),
)
ERROR_RE = re.compile(
    r"(Traceback \(most recent call last\)|\|\s*ERROR\b|\bERROR:|\b[A-Za-z]*(?:Error|Exception):|\bfailed\b|No space|out of memory|\bkilled\b|CUDA out|ClusterNotUpError|nccl.*(error|failed|timeout))",
    re.IGNORECASE,
)
BENIGN_ERROR_RE = re.compile(
    r"(warning: Failed to hardlink files|Downloading nvidia-nccl|Downloaded nvidia-nccl|[+-]\s+nvidia-nccl|weight_sync_backend:\s*nccl|SKYRL_WORKER_NCCL_TIMEOUT_IN_S|Unclosed client session|Unclosed connector|FutureWarning: Tip:.*error message|job will not be killed|echo\s+['\"].*\bfailed\b.*['\"]|Failed to establish connection to the (event\+metrics exporter|metrics exporter) agent|Failed to connect to GCS at address .* within \d+ seconds|Failed to get cluster ID from GCS server: TimedOut|Running out of retries to initialize the metrics agent|cleanup_old_checkpoints.*Failed to remove old checkpoint.*(404|notFound|specified key does not exist))",
    re.IGNORECASE,
)
DISTRIBUTED_STATE_FAILURE_RE = re.compile(
    r"(NCCL|CUDA|CUBLAS|process group|collective|FSDP|fsdp2_load_full_state_dict|distribute_tensor|scatter|unhandled cuda error)",
    re.IGNORECASE,
)
EXPORT_RECOVERY_STAGES = {"hf_export", "artifact_upload", "export_entrypoint"}
CHECKPOINT_DIR_RE = re.compile(r"/global_step_(\d+)/?$")
RANK_RE = re.compile(r"/(model|optim|extra_state)_world_size_(\d+)_rank_(\d+)\.pt$")
GSUTIL_OBJECT_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(gs://\S+)$")
GSUTIL_TOTAL_RE = re.compile(r"^TOTAL:\s+(\d+)\s+objects?,\s+(\d+)\s+bytes")
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin")
ACTIVE_JOB_STATUSES = {"RUNNING", "PENDING", "SETTING_UP", "STARTING"}
TERMINAL_UNHEALTHY_JOB_STATUSES = {"FAILED", "CANCELLED"}
ACTIVE_INSTANCE_STATUSES = {"RUNNING", "PROVISIONING", "STAGING", "REPAIRING"}
PROVISIONING_INSTANCE_STATUSES = {"PROVISIONING", "STAGING", "REPAIRING"}
WORKING_PIPELINE_STATES = {"provisioning", "running", "checkpointing"}
STAGE_GROUPS = {
    "container_pull": "setup",
    "container_start": "setup",
    "data_restore": "setup",
    "model_stage": "setup",
    "dependency_setup": "setup",
    "worker_init": "startup",
    "model_init": "startup",
    "checkpoint_load": "startup",
    "weight_sync": "synchronization",
    "trajectory_generation": "rollout",
    "rollout_inference": "rollout",
    "reward_compile": "reward",
    "reward_benchmark": "reward",
    "docker_runtime": "reward",
    "training": "training",
    "policy_update": "training",
    "policy_forward": "training",
    "evaluation": "evaluation",
    "checkpoint_stage": "checkpoint",
    "hf_export": "export",
    "hf_export_done": "export",
    "export_entrypoint": "export",
    "artifact_upload": "artifact",
    "job_complete": "complete",
    "failed": "failed",
}
STAGE_PRIORITY = {
    "job_complete": 100,
    "failed": 95,
    "hf_export_done": 90,
    "hf_export": 85,
    "artifact_upload": 80,
    "checkpoint_stage": 75,
    "checkpoint_load": 70,
    "training": 65,
    "policy_update": 65,
    "trajectory_generation": 60,
    "rollout_inference": 60,
    "evaluation": 55,
    "weight_sync": 50,
    "model_init": 45,
    "worker_init": 40,
    "dependency_setup": 35,
    "model_stage": 30,
    "data_restore": 25,
    "container_start": 20,
    "container_pull": 15,
}
CONFIG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("trainer.ckpt_interval", re.compile(r"trainer\.ckpt_interval=([0-9-]+)")),
    ("trainer.hf_save_interval", re.compile(r"trainer\.hf_save_interval=([0-9-]+)")),
    ("trainer.max_ckpts_to_keep", re.compile(r"trainer\.max_ckpts_to_keep=([0-9-]+)")),
    ("trainer.train_batch_size", re.compile(r"trainer\.train_batch_size=([0-9]+)")),
    ("trainer.micro_train_batch_size_per_gpu", re.compile(r"trainer\.micro_train_batch_size_per_gpu=([0-9]+)")),
    ("trainer.algorithm.use_kl_loss", re.compile(r"trainer\.algorithm\.use_kl_loss=(true|false)", re.IGNORECASE)),
    ("trainer.algorithm.kl_loss_coef", re.compile(r"trainer\.algorithm\.kl_loss_coef=([0-9.eE+-]+)")),
    (
        "trainer.algorithm.use_entropy_loss",
        re.compile(r"trainer\.algorithm\.use_entropy_loss=(true|false)", re.IGNORECASE),
    ),
    ("trainer.algorithm.entropy_loss_coef", re.compile(r"trainer\.algorithm\.entropy_loss_coef=([0-9.eE+-]+)")),
    ("trainer.epochs", re.compile(r"trainer\.epochs=([0-9]+)")),
    ("trainer.placement.policy_num_nodes", re.compile(r"trainer\.placement\.policy_num_nodes=([0-9]+)")),
    ("trainer.placement.policy_num_gpus_per_node", re.compile(r"trainer\.placement\.policy_num_gpus_per_node=([0-9]+)")),
    ("trainer.policy.fsdp_config.fsdp_size", re.compile(r"trainer\.policy\.fsdp_config\.fsdp_size=([0-9-]+)")),
    ("trainer.ref.fsdp_config.fsdp_size", re.compile(r"trainer\.ref\.fsdp_config\.fsdp_size=([0-9-]+)")),
    ("sft.ckpt_interval", re.compile(r"(?<![\w.])ckpt_interval=([0-9-]+)")),
    ("sft.hf_save_interval", re.compile(r"(?<![\w.])hf_save_interval=([0-9-]+)")),
    ("sft.max_ckpts_to_keep", re.compile(r"(?<![\w.])max_ckpts_to_keep=([0-9-]+)")),
    ("sft.batch_size", re.compile(r"(?<![\w.])batch_size=([0-9]+)")),
    ("sft.micro_train_batch_size_per_gpu", re.compile(r"(?<![\w.])micro_train_batch_size_per_gpu=([0-9]+)")),
    ("sft.num_epochs", re.compile(r"(?<![\w.])num_epochs=([0-9]+)")),
    ("sft.placement.num_nodes", re.compile(r"placement\.num_nodes=([0-9]+)")),
    ("sft.placement.num_gpus_per_node", re.compile(r"placement\.num_gpus_per_node=([0-9]+)")),
    ("skyrl.ray_pg_timeout_s", re.compile(r"SKYRL_RAY_PG_TIMEOUT_IN_S(?:=|:-)([0-9]+)")),
    ("skyrl.worker_nccl_timeout_s", re.compile(r"SKYRL_WORKER_NCCL_TIMEOUT_IN_S(?:=|:-)([0-9]+)")),
    ("environment.skyrl_gym.max_env_workers", re.compile(r"environment\.skyrl_gym\.max_env_workers=([0-9]+)")),
    ("generator.n_samples_per_prompt", re.compile(r"generator\.n_samples_per_prompt=([0-9]+)")),
    ("generator.inference_engine.num_engines", re.compile(r"generator\.inference_engine\.num_engines=([0-9]+)")),
    (
        "generator.inference_engine.gpu_memory_utilization",
        re.compile(r"generator\.inference_engine\.gpu_memory_utilization=([0-9.eE+-]+)"),
    ),
)
METRIC_RE = re.compile(
    r"['\"]?([A-Za-z0-9_./-]+/[A-Za-z0-9_./-]+|batch_num_seq|batch_padded_seq_len)['\"]?\s*:\s*['\"]?([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
NODE_HEALTH_CMD = (
    "set +e; "
    "echo __W8_GPU__; "
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
    "--format=csv,noheader,nounits; "
    "echo __W8_DF__; "
    "df -P -k / /tmp /home/gcpuser; "
    "echo __W8_PS__; "
    "ps -eo pid,etimes,pcpu,pmem,cmd --sort=-pcpu | head -20"
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    skipped: bool = False
    timed_out: bool = False
    attempt_count: int = 1

    @property
    def ok(self) -> bool | None:
        if self.skipped:
            return None
        return self.returncode == 0

    def check(self, name: str, *, include_stdout: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": name,
            "command": self.command,
            "ok": self.ok,
            "returncode": self.returncode,
            "skipped": self.skipped,
            "timed_out": self.timed_out,
            "attempt_count": self.attempt_count,
        }
        if include_stdout and self.stdout:
            payload["stdout_tail"] = _tail(_strip_ansi(self.stdout), 4000)
        if self.stderr:
            payload["stderr_tail"] = _tail(_strip_ansi(self.stderr), 2000)
        return payload


def build_run_status(
    *,
    run_id: str,
    project_id: str,
    artifact_bucket: str,
    env: dict[str, str],
    credentials_path: str = ".gcp-service-account.json",
    dataset_gcs_prefix: str | None = None,
    pipelines: list[str] | None = None,
    owner: str = "sss",
    log_tail: int = 800,
    job_id: str | None = None,
    expected_world_size: int = 8,
    expected_sft_final_step: int | None = None,
    expected_grpo_final_step: int | None = None,
    include_node_health: bool = False,
    baseline_statuses: list[dict[str, Any]] | None = None,
    command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
    command_retries: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build one JSON-serializable status snapshot for a run."""

    selected_pipelines = pipelines or list(PIPELINES)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    instances, instance_check = _gcp_instances(
        project_id=project_id,
        run_id=run_id,
        owner=owner,
        env=env,
        timeout_s=command_timeout_s,
        retries=command_retries,
        dry_run=dry_run,
    )
    dataset = _dataset_stage(
        dataset_gcs_prefix,
        env=env,
        timeout_s=command_timeout_s,
        retries=command_retries,
        dry_run=dry_run,
    )
    pipeline_payloads = []
    for pipeline in selected_pipelines:
        pipeline_payloads.append(
            _pipeline_status(
                pipeline=pipeline,
                run_id=run_id,
                artifact_bucket=artifact_bucket,
                env=env,
                instances=[item for item in instances if item.get("labels", {}).get("pipeline") == pipeline],
                log_tail=log_tail,
                job_id=job_id,
                expected_world_size=expected_world_size,
                expected_final_step=_expected_final_step(
                    pipeline,
                    sft=expected_sft_final_step,
                    grpo=expected_grpo_final_step,
                ),
                include_node_health=include_node_health,
                baseline_statuses=baseline_statuses or [],
                timeout_s=command_timeout_s,
                retries=command_retries,
                dry_run=dry_run,
            )
        )
    active_instances = [item for item in instances if item.get("status") in ACTIVE_INSTANCE_STATUSES]
    provisioning_instances = [item for item in instances if item.get("status") in PROVISIONING_INSTANCE_STATUSES]
    active_jobs = [
        job
        for pipeline in pipeline_payloads
        for job in pipeline.get("backend", {}).get("queue", {}).get("jobs", [])
        if str(job.get("status")).upper() in ACTIVE_JOB_STATUSES
    ]
    working_pipelines = [
        pipeline
        for pipeline in pipeline_payloads
        if pipeline.get("state") in WORKING_PIPELINE_STATES
    ]
    cleanup = {
        "safe_to_cleanup": not active_jobs and not provisioning_instances and not working_pipelines,
        "active_job_count": len(active_jobs),
        "active_instance_count": len(active_instances),
        "provisioning_instance_count": len(provisioning_instances),
        "working_pipeline_count": len(working_pipelines),
        "commands": {
            "dry_run": [
                "uv",
                "run",
                "w8-biayn",
                "gcp",
                "cleanup",
                "--run-id",
                run_id,
                "--credentials",
                credentials_path,
                "--dry-run",
            ],
            "execute": [
                "uv",
                "run",
                "w8-biayn",
                "gcp",
                "cleanup",
                "--run-id",
                run_id,
                "--credentials",
                credentials_path,
                "--execute",
            ],
        },
    }
    return {
        "schema_version": "w8-run-status-v1",
        "generated_at_utc": generated_at,
        "run_id": run_id,
        "project_id": project_id,
        "artifact_bucket": artifact_bucket,
        "summary": _run_summary(
            pipelines=pipeline_payloads,
            dataset=dataset,
            instances=instances,
            cleanup=cleanup,
        ),
        "dataset": dataset,
        "pipelines": pipeline_payloads,
        "gcp": {
            "instances": instances,
            "active_instance_count": len(active_instances),
            "checks": [instance_check],
        },
        "cleanup": cleanup,
    }


def _pipeline_status(
    *,
    pipeline: str,
    run_id: str,
    artifact_bucket: str,
    env: dict[str, str],
    instances: list[dict[str, Any]],
    log_tail: int,
    job_id: str | None,
    expected_world_size: int,
    expected_final_step: int | None,
    include_node_health: bool,
    baseline_statuses: list[dict[str, Any]],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    cluster = f"w8-biayn-{pipeline}-{run_id}"
    run_gcs_prefix = f"{artifact_bucket}/runs/cpp-perf/{run_id}/{pipeline}"
    queue, queue_check = _sky_queue(cluster, env=env, timeout_s=timeout_s, retries=retries, dry_run=dry_run)
    active_job = _select_job(queue, job_id=job_id)
    selected_log_job_id = job_id
    if selected_log_job_id is None and isinstance(active_job, dict) and active_job.get("job_id") is not None:
        selected_log_job_id = str(active_job["job_id"])
    log_signals, log_check = _log_signals(
        cluster,
        job_id=selected_log_job_id,
        tail=log_tail,
        env=env,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    log_signals = _with_rendered_config_fallback(pipeline=pipeline, log_signals=log_signals)
    artifacts = _artifact_status(
        pipeline=pipeline,
        run_gcs_prefix=run_gcs_prefix,
        env=env,
        expected_world_size=expected_world_size,
        expected_final_step=expected_final_step,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    node_health = None
    progress = _progress_summary(
        pipeline=pipeline,
        log_signals=log_signals,
        artifacts=artifacts,
        expected_final_step=expected_final_step,
    )
    pipeline_state = _derive_pipeline_state(
        queue=queue,
        artifacts=artifacts,
        log_signals=log_signals,
        instances=instances,
    )
    commands = _pipeline_commands(cluster=cluster, job=active_job)
    payload = {
        "pipeline": pipeline,
        "cluster": cluster,
        "state": pipeline_state,
        "run_gcs_prefix": run_gcs_prefix,
        "active_job": active_job,
        "backend": {
            "instances": instances,
            "queue": queue,
        },
        "artifacts": artifacts,
        "logs": log_signals,
        "phase": _phase_summary(log_signals=log_signals, node_health=None, artifacts=artifacts, active_job=active_job),
        "progress": progress,
        "speed_comparison": _speed_comparison(
            pipeline=pipeline,
            current_progress=progress,
            current_resources=(active_job or {}).get("resources") if isinstance(active_job, dict) else None,
            baseline_statuses=baseline_statuses,
        ),
        "resources": _resource_summary(instances=instances, node_health=None),
        "commands": commands,
        "recovery": _recovery_summary(
            pipeline=pipeline,
            pipeline_state=pipeline_state,
            cluster=cluster,
            active_job=active_job,
            artifacts=artifacts,
            log_signals=log_signals,
            commands=commands,
        ),
        "checks": [queue_check, log_check],
    }
    if include_node_health:
        node_health = _node_health(
            cluster,
            env=env,
            timeout_s=timeout_s,
            retries=retries,
            dry_run=dry_run,
            enabled=bool(instances),
        )
        payload["node_health"] = node_health
        payload["checks"].extend(node_health.pop("checks", []))
        payload["phase"] = _phase_summary(
            log_signals=log_signals,
            node_health=node_health,
            artifacts=artifacts,
            active_job=active_job,
        )
        payload["resources"] = _resource_summary(instances=instances, node_health=node_health)
    return payload


def _dataset_stage(
    prefix: str | None,
    *,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    if not prefix:
        return {
            "state": "not_configured",
            "gcs_prefix": None,
            "manifest": None,
            "checks": [],
        }
    manifest_uri = f"{prefix.rstrip('/')}/_w8_data_manifest.json"
    manifest_text, manifest_check = _storage_cat(
        manifest_uri,
        env=env,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    manifest = _parse_json_payload(manifest_text) if manifest_text else None
    files = manifest.get("files", []) if isinstance(manifest, dict) else []
    split_files = {
        "sft_train": _file_entry(files, "sft/train.jsonl"),
        "sft_validation": _file_entry(files, "sft/validation.jsonl"),
        "grpo_train": _file_entry(files, "grpo/train.parquet"),
        "grpo_validation": _file_entry(files, "grpo/validation.parquet"),
    }
    task_counts = {
        "train": sum(1 for item in files if _manifest_path(item).startswith("tasks/train/")),
        "validation": sum(1 for item in files if _manifest_path(item).startswith("tasks/validation/")),
        "test": sum(1 for item in files if _manifest_path(item).startswith("tasks/test/")),
    }
    required_present = bool(manifest) and all(split_files.values()) and task_counts["train"] > 0
    return {
        "state": "available" if required_present else "missing" if not dry_run else "unknown",
        "gcs_prefix": prefix.rstrip("/"),
        "manifest_uri": manifest_uri,
        "schema_version": manifest.get("schema_version") if isinstance(manifest, dict) else None,
        "file_count": len(files),
        "task_counts": task_counts,
        "split_files": split_files,
        "checks": [manifest_check],
    }


def _artifact_status(
    *,
    pipeline: str,
    run_gcs_prefix: str,
    env: dict[str, str],
    expected_world_size: int,
    expected_final_step: int | None,
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"run_gcs_prefix": run_gcs_prefix, "checks": []}
    if pipeline in CHECKPOINT_PIPELINES:
        ckpt_prefix = f"{run_gcs_prefix}/ckpts"
        marker_text, marker_check = _storage_cat(
            f"{ckpt_prefix}/latest_ckpt_global_step.txt",
            env=env,
            timeout_s=timeout_s,
            retries=retries,
            dry_run=dry_run,
        )
        checkpoint_listing, checkpoint_check = _storage_ls(
            f"{ckpt_prefix}/",
            env=env,
            long=False,
            timeout_s=timeout_s,
            retries=retries,
            dry_run=dry_run,
        )
        marker = _parse_marker(marker_text)
        checkpoint_steps = sorted(
            {int(match.group(1)) for line in checkpoint_listing.splitlines() if (match := CHECKPOINT_DIR_RE.search(line))}
        )
        highest_step = checkpoint_steps[-1] if checkpoint_steps else None
        latest_checkpoint = None
        highest_checkpoint = None
        in_progress_checkpoint = None
        if marker is not None:
            latest_checkpoint = _checkpoint_detail(
                f"{ckpt_prefix}/global_step_{marker}",
                env=env,
                expected_world_size=expected_world_size,
                timeout_s=timeout_s,
                retries=retries,
                dry_run=dry_run,
            )
            payload["checks"].extend(latest_checkpoint.pop("checks", []))
            latest_checkpoint["step"] = marker
            latest_checkpoint["promoted"] = True
        if highest_step is not None:
            if highest_step == marker and latest_checkpoint:
                highest_checkpoint = dict(latest_checkpoint)
            else:
                highest_checkpoint = _checkpoint_detail(
                    f"{ckpt_prefix}/global_step_{highest_step}",
                    env=env,
                    expected_world_size=expected_world_size,
                    timeout_s=timeout_s,
                    retries=retries,
                    dry_run=dry_run,
                )
                payload["checks"].extend(highest_checkpoint.pop("checks", []))
                highest_checkpoint["step"] = highest_step
                highest_checkpoint["promoted"] = highest_step == marker
            if highest_step != marker:
                in_progress_checkpoint = highest_checkpoint
        payload["checkpoint"] = {
            "prefix": ckpt_prefix,
            "latest_marker": marker,
            "steps": checkpoint_steps,
            "latest": latest_checkpoint,
            "highest": highest_checkpoint,
            "in_progress": in_progress_checkpoint,
        }
        payload["checks"].extend([marker_check, checkpoint_check])
    if pipeline in CHECKPOINT_PIPELINES:
        export_prefix = f"{run_gcs_prefix}/exports"
        export_listing, export_check = _storage_ls(
            f"{export_prefix}/",
            env=env,
            long=False,
            timeout_s=timeout_s,
            retries=retries,
            dry_run=dry_run,
        )
        export_steps = sorted(
            {int(match.group(1)) for line in export_listing.splitlines() if (match := CHECKPOINT_DIR_RE.search(line))}
        )
        final_export_prefix = f"{export_prefix}/global_step_{expected_final_step}/policy" if expected_final_step else None
        final_export_exists = False
        final_export_detail = None
        if final_export_prefix:
            final_export_detail = _export_detail(
                final_export_prefix,
                env=env,
                timeout_s=timeout_s,
                retries=retries,
                dry_run=dry_run,
            )
            final_export_exists = final_export_detail["complete"]
        payload["export"] = {
            "prefix": export_prefix,
            "steps": export_steps,
            "expected_final_step": expected_final_step,
            "final_export_prefix": final_export_prefix,
            "final_export_exists": final_export_exists,
            "final_export": final_export_detail,
        }
        payload["checks"].append(export_check)
        if final_export_detail:
            payload["checks"].extend(final_export_detail.pop("checks", []))
    if pipeline == "cpp-eval":
        eval_listing, eval_check = _storage_ls(
            f"{run_gcs_prefix}/",
            env=env,
            long=False,
            timeout_s=timeout_s,
            retries=retries,
            dry_run=dry_run,
        )
        payload["eval_outputs"] = {
            "prefix": run_gcs_prefix,
            "objects": _gs_uris(eval_listing),
        }
        payload["checks"].append(eval_check)
    return payload


def _export_detail(
    export_prefix: str,
    *,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    listing, listing_check = _storage_ls(
        f"{export_prefix}/",
        env=env,
        long=True,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    objects = _gsutil_objects(listing)
    names = {_basename(item["uri"]) for item in objects}
    weight_objects = [
        item
        for item in objects
        if isinstance(item.get("uri"), str) and item["uri"].endswith(MODEL_WEIGHT_SUFFIXES)
    ]
    total_match = next((GSUTIL_TOTAL_RE.search(line) for line in listing.splitlines() if GSUTIL_TOTAL_RE.search(line)), None)
    total_objects = int(total_match.group(1)) if total_match else len(objects)
    total_bytes = int(total_match.group(2)) if total_match else sum(int(item["bytes"]) for item in objects)
    has_config = "config.json" in names
    has_tokenizer = "tokenizer.json" in names or "tokenizer.model" in names
    has_weights = bool(weight_objects)
    return {
        "prefix": export_prefix,
        "object_count": total_objects,
        "total_bytes": total_bytes,
        "has_config": has_config,
        "has_tokenizer": has_tokenizer,
        "weight_object_count": len(weight_objects),
        "weight_bytes": sum(int(item["bytes"]) for item in weight_objects),
        "sample_objects": objects[:20],
        "complete": listing_check.get("ok") is True and has_config and has_weights,
        "checks": [listing_check],
    }


def _checkpoint_detail(
    checkpoint_prefix: str,
    *,
    env: dict[str, str],
    expected_world_size: int,
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    trainer_listing, trainer_check = _storage_ls(
        f"{checkpoint_prefix}/trainer_state.pt",
        env=env,
        long=True,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    policy_listing, policy_check = _storage_ls(
        f"{checkpoint_prefix}/policy/",
        env=env,
        long=True,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    rank_counts = {"model": 0, "optim": 0, "extra_state": 0}
    world_sizes: dict[str, set[int]] = {"model": set(), "optim": set(), "extra_state": set()}
    ranks: dict[str, set[int]] = {"model": set(), "optim": set(), "extra_state": set()}
    total_objects = 0
    total_bytes = 0
    for line in policy_listing.splitlines():
        if total_match := GSUTIL_TOTAL_RE.search(line):
            total_objects = int(total_match.group(1))
            total_bytes = int(total_match.group(2))
        if match := RANK_RE.search(line):
            kind = match.group(1)
            world_sizes[kind].add(int(match.group(2)))
            ranks[kind].add(int(match.group(3)))
            rank_counts[kind] += 1
    trainer_present = "trainer_state.pt" in trainer_listing and trainer_check.get("ok") is True
    model_complete = len(ranks["model"]) >= expected_world_size
    optim_complete = len(ranks["optim"]) >= expected_world_size
    return {
        "prefix": checkpoint_prefix,
        "trainer_state_present": trainer_present,
        "expected_world_size": expected_world_size,
        "rank_counts": rank_counts,
        "world_sizes": {key: sorted(value) for key, value in world_sizes.items()},
        "ranks": {key: sorted(value) for key, value in ranks.items()},
        "total_objects": total_objects,
        "total_bytes": total_bytes,
        "resumable": trainer_present and model_complete and optim_complete,
        "checks": [trainer_check, policy_check],
    }


def _sky_queue(
    cluster: str,
    *,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _run(
        ["sky", "queue", "--output", "json", cluster],
        env=env,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    parsed = _parse_json_payload(result.stdout)
    jobs = []
    if isinstance(parsed, dict):
        jobs = parsed.get(cluster) or []
    return (
        {
            "cluster": cluster,
            "jobs": jobs if isinstance(jobs, list) else [],
            "available": result.ok is True,
            "error": None if result.ok is True else _tail(result.stderr or result.stdout, 2000),
        },
        result.check("sky_queue", include_stdout=False),
    )


def _log_signals(
    cluster: str,
    *,
    job_id: str | None,
    tail: int,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    args = ["sky", "logs", "--tail", str(tail), "--no-follow", cluster]
    if job_id is not None:
        args.append(job_id)
    result = _run(args, env=env, timeout_s=timeout_s, retries=retries, dry_run=dry_run)
    lines = [_strip_ansi(line) for line in result.stdout.splitlines()]
    signals = _extract_log_signals(lines)
    signals["tail_lines_scanned"] = len(lines)
    signals["tail_lines_requested"] = tail
    signals["tail_may_be_truncated"] = result.ok is True and tail > 0 and len(lines) >= tail
    signals["available"] = result.ok is True
    signals["error"] = None if result.ok is True else _tail(result.stderr or result.stdout, 2000)
    return signals, result.check("sky_logs", include_stdout=False)


def _extract_log_signals(lines: list[str]) -> dict[str, Any]:
    last_step = None
    last_loss = None
    last_eval_loss = None
    checkpoint_events: list[dict[str, Any]] = []
    export_events: list[str] = []
    errors: list[str] = []
    stage = None
    stage_events: list[dict[str, str]] = []
    trajectory_progress = None
    evaluation_progress = None
    training_progress = None
    timings: dict[str, Any] = {
        "step_durations_s": [],
        "generate_durations_s": [],
        "sync_weights_durations_s": [],
        "checkpoint_durations_s": [],
    }
    config: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for line in lines:
        stage_matches = _stage_matches(line)
        if stage_matches:
            stage = _select_stage(stage_matches)
            stage_events.append({"stage": stage, "line": line, "matched_stages": stage_matches})
        _record_config_matches(config, line)
        for metric_match in METRIC_RE.finditer(line):
            _record_metric(metrics, metric_match.group(1), float(metric_match.group(2)))
        if step_match := STEP_RE.search(line):
            last_step = int(step_match.group(1))
        if loss_match := LOSS_RE.search(line):
            last_loss = {"step": int(loss_match.group(1)), "loss": float(loss_match.group(2))}
        if eval_match := EVAL_LOSS_RE.search(line):
            last_eval_loss = {"step": int(eval_match.group(1)), "eval_loss": float(eval_match.group(2))}
        if progress_match := PROGRESS_RE.search(line):
            progress = {
                "completed": int(progress_match.group(3)),
                "total": int(progress_match.group(4)),
                "percent": int(progress_match.group(2)),
                "line": line,
            }
            if progress_match.group(1).lower().startswith("generating"):
                trajectory_progress = progress
            else:
                evaluation_progress = progress
        if train_progress_match := TRAIN_BATCH_PROGRESS_RE.search(line):
            training_progress = {
                "completed": int(train_progress_match.group(2)),
                "total": int(train_progress_match.group(3)),
                "percent": int(train_progress_match.group(1)),
                "line": line,
            }
        if save_start := SAVE_START_RE.search(line):
            checkpoint_events.append(
                {
                    "event": "save_started",
                    "step": int(save_start.group(1)),
                    "target": save_start.group(2),
                    "line": line,
                }
            )
        elif save_done := SAVE_DONE_RE.search(line):
            checkpoint_events.append({"event": "save_completed", "step": int(save_done.group(1)), "line": line})
        elif save_finished := SAVE_FINISHED_RE.search(line):
            duration_s = float(save_finished.group(1))
            timings["last_checkpoint_duration_s"] = duration_s
            timings["checkpoint_durations_s"].append(duration_s)
            checkpoint_events.append({"event": "save_finished", "duration_s": duration_s, "line": line})
        if step_finished := STEP_FINISHED_RE.search(line):
            duration_s = float(step_finished.group(1))
            timings["last_step_duration_s"] = duration_s
            timings["step_durations_s"].append(duration_s)
        if generate_finished := GENERATE_FINISHED_RE.search(line):
            duration_s = float(generate_finished.group(1))
            timings["last_generate_duration_s"] = duration_s
            timings["generate_durations_s"].append(duration_s)
        if sync_finished := SYNC_FINISHED_RE.search(line):
            duration_s = float(sync_finished.group(1))
            timings["last_sync_weights_duration_s"] = duration_s
            timings["sync_weights_durations_s"].append(duration_s)
        if HF_EXPORT_RE.search(line):
            export_events.append(line)
        if ERROR_RE.search(line) and not BENIGN_ERROR_RE.search(line):
            errors.append(line)
    for key, value in list(timings.items()):
        if isinstance(value, list):
            timings[key] = value[-20:]
    return {
        "stage": "failed" if errors else stage,
        "stage_events": stage_events[-20:],
        "last_step": last_step,
        "last_loss": last_loss,
        "last_eval_loss": last_eval_loss,
        "trajectory_progress": trajectory_progress,
        "evaluation_progress": evaluation_progress,
        "training_progress": training_progress,
        "checkpoint_events": checkpoint_events[-20:],
        "export_events": export_events[-20:],
        "timings": timings,
        "config": config,
        "metrics": metrics,
        "errors": errors[-20:],
    }


def _with_rendered_config_fallback(*, pipeline: str, log_signals: dict[str, Any]) -> dict[str, Any]:
    rendered_config, rendered_path = _rendered_config(pipeline)
    log_config = log_signals.get("config") if isinstance(log_signals.get("config"), dict) else {}
    sources = []
    if rendered_config:
        sources.append("rendered_yaml")
    if log_config:
        sources.append("logs")
    if not rendered_config:
        if sources:
            log_signals["config_sources"] = sources
        return log_signals
    merged = {**rendered_config, **log_config}
    payload = dict(log_signals)
    payload["config"] = merged
    payload["config_sources"] = sources
    payload["rendered_config_path"] = str(rendered_path)
    return payload


def _rendered_config(pipeline: str) -> tuple[dict[str, Any], Path | None]:
    path = RENDERED_CONFIG_DIR / f"{pipeline}.sky.yaml"
    if not path.exists():
        return {}, None
    config: dict[str, Any] = {}
    try:
        text = path.read_text()
    except OSError:
        return {}, path
    for line in text.splitlines():
        _record_config_matches(config, line)
    return config, path


def _record_config_matches(config: dict[str, Any], line: str) -> None:
    for key, pattern in CONFIG_PATTERNS:
        if config_match := pattern.search(line):
            config[key] = _parse_config_value(config_match.group(1))


def _parse_config_value(value: str) -> int | float | bool | str:
    normalized = value.strip()
    if normalized.lower() == "true":
        return True
    if normalized.lower() == "false":
        return False
    try:
        parsed_float = float(normalized)
    except ValueError:
        return normalized
    if parsed_float.is_integer() and not re.search(r"[.eE]", normalized):
        return int(parsed_float)
    return parsed_float


def _stage_matches(line: str) -> list[str]:
    return [stage_name for stage_name, pattern in STAGE_PATTERNS if pattern.search(line)]


def _select_stage(matches: list[str]) -> str:
    return max(matches, key=lambda stage_name: STAGE_PRIORITY.get(stage_name, 0))


def _node_health(
    cluster: str,
    *,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        check = CommandResult(
            command=["ssh", cluster, NODE_HEALTH_CMD],
            returncode=None,
            stdout="",
            stderr="",
            skipped=True,
        ).check("node_health")
        return {
            "available": False,
            "skipped": True,
            "reason": "no running instance",
            "sample_scope": "none",
            "nodes": [],
            "checks": [check],
        }
    result = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            cluster,
            NODE_HEALTH_CMD,
        ],
        env=env,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    health = _parse_node_health(result.stdout)
    health.update(
        {
            "available": result.ok is True,
            "skipped": result.skipped,
            "error": None if result.ok is True else _tail(result.stderr or result.stdout, 2000),
            "sample_scope": "head",
            "sample_note": "Health is sampled from the cluster SSH entrypoint; worker nodes are not individually probed.",
            "checks": [result.check("node_health", include_stdout=False)],
        }
    )
    health["nodes"] = [
        {
            "role": "head",
            "sampled": result.ok is True,
            "activity": health.get("activity"),
            "gpus": health.get("gpus", []),
            "filesystems": health.get("filesystems", []),
            "processes": health.get("processes", []),
        }
    ]
    return health


def _parse_node_health(text: str) -> dict[str, Any]:
    sections = _split_health_sections(text)
    processes = _parse_process_health(sections.get("__W8_PS__", ""))
    return {
        "gpus": _parse_gpu_health(sections.get("__W8_GPU__", "")),
        "filesystems": _parse_df_health(sections.get("__W8_DF__", "")),
        "processes": processes,
        "activity": _derive_node_activity(processes),
    }


def _split_health_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        stripped = _strip_ansi(line).strip()
        if stripped in {"__W8_GPU__", "__W8_DF__", "__W8_PS__"}:
            current = stripped
            sections[current] = []
        elif current:
            sections[current].append(stripped)
    return {key: "\n".join(value) for key, value in sections.items()}


def _parse_gpu_health(text: str) -> list[dict[str, int | None]]:
    gpus = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        gpus.append(
            {
                "index": _to_int(parts[0]),
                "utilization_gpu_percent": _to_int(parts[1]),
                "memory_used_mib": _to_int(parts[2]),
                "memory_total_mib": _to_int(parts[3]),
            }
        )
    return gpus


def _parse_df_health(text: str) -> list[dict[str, Any]]:
    filesystems = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        filesystems.append(
            {
                "filesystem": parts[0],
                "blocks_kb": _to_int(parts[1]),
                "used_kb": _to_int(parts[2]),
                "available_kb": _to_int(parts[3]),
                "capacity": parts[4],
                "mount": parts[5],
            }
        )
    return filesystems


def _parse_process_health(text: str) -> list[dict[str, Any]]:
    processes = []
    for line in text.splitlines()[1:]:
        parts = line.split(maxsplit=4)
        if len(parts) < 5:
            continue
        processes.append(
            {
                "pid": _to_int(parts[0]),
                "elapsed_s": _to_int(parts[1]),
                "cpu_percent": _to_float(parts[2]),
                "mem_percent": _to_float(parts[3]),
                "cmd": parts[4],
            }
        )
    return processes


def _derive_node_activity(processes: list[dict[str, Any]]) -> str | None:
    commands = "\n".join(str(process.get("cmd", "")) for process in processes).lower()
    if "gcloud.py storage cp" in commands and "data/skyrl" in commands:
        return "data_restore"
    if "gcloud.py storage cp" in commands and (
        "models" in commands or "/model" in commands or "/exports" in commands
    ):
        return "model_stage"
    if "gcloud storage rsync" in commands:
        return "checkpoint_stage"
    if "save_checkpoint" in commands:
        return "checkpoint_stage"
    if "init_model" in commands:
        return "model_init"
    if "load_checkpoint" in commands:
        return "checkpoint_load"
    if "save_hf_model" in commands:
        return "hf_export"
    if "skyrl_sft_export_checkpoint_main" in commands:
        return "export_entrypoint"
    if "cc1plus" in commands and ("candidate.cpp" in commands or "reference.cpp" in commands):
        return "reward_compile"
    if "./candidate" in commands:
        return "reward_benchmark"
    if "w8_runtime_bench.py" in commands:
        return "reward_benchmark"
    if "fsdppolicyworkerbase.forward_backward" in commands:
        return "policy_update"
    if "fsdppolicyworkerbase.forward" in commands:
        return "policy_forward"
    if "vllm::enginecore" in commands:
        return "rollout_inference"
    if "skyrl_cpp_perf_main" in commands or "skyrl_entrypoint" in commands:
        return "skyrl_startup"
    if "unpigz" in commands:
        return "container_pull"
    if any(
        "dockerd" in str(process.get("cmd", "")).lower()
        and float(process.get("cpu_percent") or 0) >= 10.0
        for process in processes
    ):
        return "docker_runtime"
    return None


def _gcp_instances(
    *,
    project_id: str,
    run_id: str,
    owner: str,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_filter = (
        "labels.project=w8-biayn AND labels.phase=cpp-perf-rl "
        f"AND labels.run_id={run_id.lower()} AND labels.owner={owner.lower()}"
    )
    result = _run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project_id,
            "--filter",
            label_filter,
            "--format",
            "json(name,zone,status,machineType,creationTimestamp,disks[].diskSizeGb,guestAccelerators,labels)",
        ],
        env=env,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    parsed = _parse_json_payload(result.stdout)
    instances = [_normalize_instance(item) for item in parsed] if isinstance(parsed, list) else []
    return instances, result.check("gcloud_compute_instances", include_stdout=False)


def _normalize_instance(item: dict[str, Any]) -> dict[str, Any]:
    disks = item.get("disks") if isinstance(item.get("disks"), list) else []
    boot_disk = disks[0] if disks else {}
    accelerators = item.get("guestAccelerators") if isinstance(item.get("guestAccelerators"), list) else []
    return {
        "name": item.get("name"),
        "zone": _basename(item.get("zone")),
        "status": item.get("status"),
        "machine_type": _basename(item.get("machineType")),
        "disk_size_gb": _to_int(boot_disk.get("diskSizeGb")),
        "accelerators": [
            {
                "type": _basename(accelerator.get("acceleratorType")),
                "count": _to_int(accelerator.get("acceleratorCount")),
            }
            for accelerator in accelerators
            if isinstance(accelerator, dict)
        ],
        "labels": item.get("labels") if isinstance(item.get("labels"), dict) else {},
        "creation_timestamp": item.get("creationTimestamp"),
    }


def _storage_cat(
    uri: str,
    *,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    result = _run(
        ["gcloud", "storage", "cat", uri],
        env=env,
        timeout_s=timeout_s,
        retries=retries,
        dry_run=dry_run,
    )
    return result.stdout if result.ok else "", result.check(f"gcloud_storage_cat:{uri}")


def _storage_ls(
    uri: str,
    *,
    env: dict[str, str],
    long: bool,
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> tuple[str, dict[str, Any]]:
    args = ["gcloud", "storage", "ls", "--format=gsutil"]
    if long:
        args.append("-l")
    args.append(uri)
    result = _run(args, env=env, timeout_s=timeout_s, retries=retries, dry_run=dry_run)
    return result.stdout if result.ok else "", result.check(f"gcloud_storage_ls:{uri}", include_stdout=False)


def _run(
    args: list[str],
    *,
    env: dict[str, str],
    timeout_s: int,
    retries: int,
    dry_run: bool,
) -> CommandResult:
    if dry_run:
        return CommandResult(command=args, returncode=None, stdout="", stderr="", skipped=True)
    attempts = max(1, retries + 1)
    last_timeout: CommandResult | None = None
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(args, env=env, check=False, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            stderr = stderr + f"\ncommand timed out after {timeout_s}s"
            last_timeout = CommandResult(
                command=args,
                returncode=-1,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                attempt_count=attempt,
            )
            continue
        return CommandResult(
            command=args,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            attempt_count=attempt,
        )
    return last_timeout or CommandResult(
        command=args,
        returncode=-1,
        stdout="",
        stderr=f"command timed out after {timeout_s}s",
        timed_out=True,
        attempt_count=attempts,
    )


def _derive_pipeline_state(
    *,
    queue: dict[str, Any],
    artifacts: dict[str, Any],
    log_signals: dict[str, Any],
    instances: list[dict[str, Any]],
) -> str:
    jobs = queue.get("jobs") if isinstance(queue.get("jobs"), list) else []
    statuses = {str(job.get("status")).upper() for job in jobs if isinstance(job, dict)}
    instance_statuses = {str(item.get("status")).upper() for item in instances}
    if instance_statuses & PROVISIONING_INSTANCE_STATUSES:
        return "provisioning"
    if statuses & ACTIVE_JOB_STATUSES:
        return "running"
    if statuses & {"FAILED", "CANCELLED"}:
        return "failed"
    if log_signals.get("errors"):
        return "failed"
    export = artifacts.get("export") if isinstance(artifacts.get("export"), dict) else {}
    if export.get("final_export_exists"):
        return "complete"
    checkpoint = artifacts.get("checkpoint") if isinstance(artifacts.get("checkpoint"), dict) else {}
    latest_marker = checkpoint.get("latest_marker")
    checkpoint_events = log_signals.get("checkpoint_events") if isinstance(log_signals.get("checkpoint_events"), list) else []
    last_checkpoint_event = checkpoint_events[-1] if checkpoint_events else {}
    if last_checkpoint_event.get("event") == "save_started" and last_checkpoint_event.get("step") != latest_marker:
        return "checkpointing"
    if any(item.get("status") == "RUNNING" for item in instances):
        return "cluster_up"
    if latest_marker:
        return "checkpointed"
    if statuses & {"SUCCEEDED"}:
        return "succeeded"
    return "not_started"


def _run_summary(
    *,
    pipelines: list[dict[str, Any]],
    dataset: dict[str, Any],
    instances: list[dict[str, Any]],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    active = [
        pipeline
        for pipeline in pipelines
        if pipeline.get("state") in WORKING_PIPELINE_STATES
        or str((pipeline.get("active_job") or {}).get("status", "")).upper() in ACTIVE_JOB_STATUSES
    ]
    current = active[0] if active else pipelines[0] if pipelines else {}
    pipeline_states = {str(pipeline.get("state")) for pipeline in pipelines if pipeline.get("state")}
    if any(state == "failed" for state in pipeline_states):
        state = "failed"
    elif active:
        state = "running"
    elif pipelines and all(state in {"complete", "succeeded", "checkpointed"} for state in pipeline_states):
        state = "complete"
    elif not pipelines:
        state = "unknown"
    else:
        state = "mixed"
    progress = current.get("progress") if isinstance(current.get("progress"), dict) else {}
    return {
        "state": state,
        "dataset_state": dataset.get("state"),
        "current_pipeline": current.get("pipeline"),
        "current_cluster": current.get("cluster"),
        "current_phase": current.get("phase"),
        "current_progress": {
            "primary": progress.get("primary"),
            "training": progress.get("training"),
            "trajectory": progress.get("trajectory"),
            "throughput": progress.get("throughput"),
            "training_health": progress.get("training_health"),
        },
        "training_health": progress.get("training_health"),
        "speed_comparison": current.get("speed_comparison"),
        "recovery": current.get("recovery"),
        "resources": _resource_summary(instances=instances, node_health=None),
        "cleanup_safe": cleanup.get("safe_to_cleanup"),
    }


def _select_job(queue: dict[str, Any], *, job_id: str | None) -> dict[str, Any] | None:
    jobs = queue.get("jobs") if isinstance(queue.get("jobs"), list) else []
    selected = None
    if job_id is not None:
        selected = next((job for job in jobs if str(job.get("job_id")) == str(job_id)), None)
    if selected is None:
        selected = next((job for job in jobs if str(job.get("status")).upper() in ACTIVE_JOB_STATUSES), None)
    if selected is None and jobs:
        selected = jobs[0]
    if not isinstance(selected, dict):
        return None
    start_at = _to_float(selected.get("start_at"))
    end_at = _to_float(selected.get("end_at"))
    duration_s = None
    if start_at is not None:
        duration_s = max(0.0, (end_at if end_at is not None else datetime.now(timezone.utc).timestamp()) - start_at)
    return {
        "job_id": selected.get("job_id"),
        "status": selected.get("status"),
        "resources": selected.get("resources"),
        "log_path": selected.get("log_path"),
        "start_at": selected.get("start_at"),
        "end_at": selected.get("end_at"),
        "duration_s": round(duration_s, 3) if duration_s is not None else None,
        "job_name": selected.get("job_name"),
    }


def _phase_summary(
    *,
    log_signals: dict[str, Any],
    node_health: dict[str, Any] | None,
    artifacts: dict[str, Any] | None = None,
    active_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    log_stage = log_signals.get("stage")
    node_activity = node_health.get("activity") if isinstance(node_health, dict) else None
    artifact_activity = _artifact_activity(artifacts=artifacts, active_job=active_job)
    errors = log_signals.get("errors") if isinstance(log_signals.get("errors"), list) else []
    current = "failed" if errors else node_activity or artifact_activity or log_stage
    group = STAGE_GROUPS.get(str(current), "unknown") if current else "unknown"
    source = "logs"
    if node_activity and not errors:
        source = "node_health"
    elif artifact_activity and not errors:
        source = "artifacts"
    elif not current:
        source = "none"
    return {
        "current": current,
        "group": group,
        "source": source,
        "log_stage": log_stage,
        "node_activity": node_activity,
        "artifact_activity": artifact_activity,
        "failed": bool(errors),
        "message": _phase_message(current, group),
    }


def _artifact_activity(*, artifacts: dict[str, Any] | None, active_job: dict[str, Any] | None) -> str | None:
    if not isinstance(active_job, dict) or str(active_job.get("status", "")).upper() not in ACTIVE_JOB_STATUSES:
        return None
    if not isinstance(artifacts, dict):
        return None
    checkpoint = artifacts.get("checkpoint") if isinstance(artifacts.get("checkpoint"), dict) else {}
    in_progress = checkpoint.get("in_progress") if isinstance(checkpoint.get("in_progress"), dict) else None
    if in_progress:
        return "checkpoint_stage"
    return None


def _phase_message(current: Any, group: str) -> str:
    if current == "failed":
        return "A failure signature was found in the scanned logs."
    if current:
        return f"Current observable activity is {current} ({group})."
    return "No active stage was found in the scanned logs."


def _recovery_summary(
    *,
    pipeline: str,
    pipeline_state: str,
    cluster: str,
    active_job: dict[str, Any] | None,
    artifacts: dict[str, Any],
    log_signals: dict[str, Any],
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    if pipeline not in CHECKPOINT_PIPELINES:
        return {"available": False, "reason": "pipeline_has_no_training_checkpoints"}

    checkpoint = artifacts.get("checkpoint") if isinstance(artifacts.get("checkpoint"), dict) else {}
    latest = checkpoint.get("latest") if isinstance(checkpoint.get("latest"), dict) else {}
    latest_step = latest.get("step") or checkpoint.get("latest_marker")
    latest_resumable = bool(latest.get("resumable"))
    job_status = str((active_job or {}).get("status") or "").upper()
    unhealthy_terminal = pipeline_state == "failed" or job_status in TERMINAL_UNHEALTHY_JOB_STATUSES
    has_errors = bool(log_signals.get("errors"))

    if not unhealthy_terminal and not has_errors:
        return {
            "available": False,
            "reason": "pipeline_not_failed",
            "checkpoint_resumable": latest_resumable,
            "resume_checkpoint_step": latest_step,
        }

    signals = _recovery_signals(log_signals=log_signals)
    command_hints = {
        "down": commands.get("down"),
        "logs": commands.get("logs"),
    }
    if latest_resumable and ("distributed_state_failure" in signals or "hf_export_interrupted" in signals):
        return {
            "available": True,
            "recommended_action": "fresh_cluster_resume",
            "fresh_cluster_recommended": True,
            "requires_down_before_resume": True,
            "reason": "resumable_checkpoint_after_distributed_or_export_failure",
            "signals": signals,
            "checkpoint_resumable": True,
            "resume_from": "latest",
            "resume_checkpoint_step": latest_step,
            "message": (
                "Latest checkpoint is resumable, but the failed job shows distributed-state or HF-export "
                "interruption signals. Tear down the cluster, then relaunch the same w8-biayn launch command "
                "with --resume-from latest and the same checkpoint/export paths."
            ),
            "commands": command_hints,
        }
    if latest_resumable:
        return {
            "available": True,
            "recommended_action": "resume_latest",
            "fresh_cluster_recommended": False,
            "requires_down_before_resume": False,
            "reason": "resumable_checkpoint_after_failed_job",
            "signals": signals,
            "checkpoint_resumable": True,
            "resume_from": "latest",
            "resume_checkpoint_step": latest_step,
            "message": "Latest checkpoint is resumable. Relaunch with --resume-from latest and the same checkpoint/export paths.",
            "commands": command_hints,
        }
    return {
        "available": True,
        "recommended_action": "inspect_artifacts",
        "fresh_cluster_recommended": False,
        "requires_down_before_resume": False,
        "reason": "failed_job_without_resumable_checkpoint",
        "signals": signals,
        "checkpoint_resumable": False,
        "resume_from": None,
        "resume_checkpoint_step": latest_step,
        "message": "No promoted resumable checkpoint is available; inspect logs and artifacts before relaunching.",
        "commands": command_hints,
    }


def _recovery_signals(*, log_signals: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    errors = [str(item) for item in log_signals.get("errors", []) if item]
    if DISTRIBUTED_STATE_FAILURE_RE.search("\n".join(errors)):
        signals.append("distributed_state_failure")
    stage = log_signals.get("stage")
    stage_events = log_signals.get("stage_events") if isinstance(log_signals.get("stage_events"), list) else []
    saw_export_stage = stage in EXPORT_RECOVERY_STAGES or any(
        isinstance(item, dict) and item.get("stage") in EXPORT_RECOVERY_STAGES for item in stage_events
    )
    export_events = log_signals.get("export_events") if isinstance(log_signals.get("export_events"), list) else []
    if saw_export_stage or export_events:
        signals.append("hf_export_interrupted")
    return signals


def _progress_summary(
    *,
    pipeline: str,
    log_signals: dict[str, Any],
    artifacts: dict[str, Any],
    expected_final_step: int | None = None,
) -> dict[str, Any]:
    timings = log_signals.get("timings") if isinstance(log_signals.get("timings"), dict) else {}
    config = log_signals.get("config") if isinstance(log_signals.get("config"), dict) else {}
    metrics = log_signals.get("metrics") if isinstance(log_signals.get("metrics"), dict) else {}
    grpo_config = _grpo_config_summary(config)
    sft_config = _sft_config_summary(config)
    training = _augment_progress(
        log_signals.get("training_progress"),
        unit="step",
        last_duration_s=timings.get("last_step_duration_s"),
        recent_durations_s=timings.get("step_durations_s"),
    )
    if training is None and pipeline == "cpp-sft":
        training = _sft_step_progress(
            last_step=log_signals.get("last_step"),
            metrics=metrics,
            sft_config=sft_config,
            expected_final_step=expected_final_step,
        )
    trajectory = _augment_progress(
        log_signals.get("trajectory_progress"),
        unit="sample",
        last_duration_s=timings.get("last_generate_duration_s"),
        recent_durations_s=timings.get("generate_durations_s"),
    )
    evaluation = _augment_progress(log_signals.get("evaluation_progress"), unit="eval_batch")
    batch_num_seq = _to_float(metrics.get("batch_num_seq"))
    if training and batch_num_seq:
        training.setdefault("throughput", {})
        training["samples_per_step"] = int(batch_num_seq)
        steps_per_hour = _nested_float(training, "throughput", "items_per_hour")
        if steps_per_hour is not None:
            training["throughput"]["samples_per_hour"] = round(steps_per_hour * batch_num_seq, 3)
    checkpoint = _checkpoint_progress(
        artifacts=artifacts,
        completed_step=(training or {}).get("completed") if training else log_signals.get("last_step"),
        config=config,
    )
    throughput = {
        "training_steps_per_hour": _nested_float(training, "throughput", "items_per_hour") if training else None,
        "seconds_per_training_step": _nested_float(training, "throughput", "seconds_per_item") if training else None,
        "rollout_samples_per_second": _nested_float(trajectory, "throughput", "items_per_second") if trajectory else None,
        "rollout_seconds_per_sample": _nested_float(trajectory, "throughput", "seconds_per_item") if trajectory else None,
        "rollout_samples_per_hour": _nested_float(trajectory, "throughput", "items_per_hour") if trajectory else None,
    }
    if training and training.get("samples_per_step") and throughput["training_steps_per_hour"] is not None:
        throughput["training_samples_per_hour"] = round(
            float(training["samples_per_step"]) * float(throughput["training_steps_per_hour"]),
            3,
        )
    gpu_count = _to_float(grpo_config.get("total_gpu_count")) or _to_float(sft_config.get("total_gpu_count"))
    if gpu_count:
        if throughput.get("training_samples_per_hour") is not None:
            throughput["training_samples_per_gpu_hour"] = round(
                float(throughput["training_samples_per_hour"]) / gpu_count,
                3,
            )
        if throughput.get("rollout_samples_per_second") is not None:
            throughput["rollout_samples_per_gpu_second"] = round(
                float(throughput["rollout_samples_per_second"]) / gpu_count,
                6,
            )
    primary = "training_batches" if training else "trajectories" if trajectory else "evaluation" if evaluation else None
    return {
        "pipeline": pipeline,
        "primary": primary,
        "training": training,
        "trajectory": trajectory,
        "evaluation": evaluation,
        "checkpoint": checkpoint,
        "timings": timings,
        "config": config,
        "grpo_config": grpo_config,
        "sft_config": sft_config,
        "metrics": metrics,
        "training_health": _training_health_summary(pipeline=pipeline, metrics=metrics),
        "bottleneck": _bottleneck_summary(metrics=metrics, timings=timings),
        "throughput": throughput,
    }


def _sft_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    num_nodes = _to_int(config.get("sft.placement.num_nodes"))
    gpus_per_node = _to_int(config.get("sft.placement.num_gpus_per_node"))
    total_gpu_count = None
    if num_nodes is not None and gpus_per_node is not None:
        total_gpu_count = num_nodes * gpus_per_node
    return {
        "batch_size": _to_int(config.get("sft.batch_size")),
        "micro_train_batch_size_per_gpu": _to_int(config.get("sft.micro_train_batch_size_per_gpu")),
        "num_epochs": _to_int(config.get("sft.num_epochs")),
        "policy_num_nodes": num_nodes,
        "policy_num_gpus_per_node": gpus_per_node,
        "total_gpu_count": total_gpu_count,
        "ckpt_interval": _to_int(config.get("sft.ckpt_interval")),
        "hf_save_interval": _to_int(config.get("sft.hf_save_interval")),
        "max_ckpts_to_keep": _to_int(config.get("sft.max_ckpts_to_keep")),
        "skyrl_ray_pg_timeout_s": _to_int(config.get("skyrl.ray_pg_timeout_s")),
        "skyrl_worker_nccl_timeout_s": _to_int(config.get("skyrl.worker_nccl_timeout_s")),
    }


def _sft_step_progress(
    *,
    last_step: Any,
    metrics: dict[str, Any],
    sft_config: dict[str, Any],
    expected_final_step: int | None,
) -> dict[str, Any] | None:
    completed = _to_int(last_step)
    if completed is None:
        return None
    total = _to_int(expected_final_step)
    payload: dict[str, Any] = {
        "completed": completed,
        "total": total,
        "percent": round(100.0 * completed / total) if total else None,
        "source": "last_step",
    }
    if total:
        payload["remaining"] = max(total - completed, 0)
        payload["percent_exact"] = round(100.0 * completed / total, 4)
    timing: dict[str, Any] = {}
    step_s = _to_float(metrics.get("timing/step"))
    if step_s is not None:
        timing["last_duration_s"] = round(step_s, 6)
        if total:
            timing["eta_s"] = round(max(total - completed, 0) * step_s, 3)
    payload["timing"] = timing
    payload["throughput"] = _throughput_payload("step", seconds_per_item=step_s, items_per_second=None)
    batch_size = _to_int(sft_config.get("batch_size"))
    if batch_size:
        payload["samples_per_step"] = batch_size
        steps_per_hour = _nested_float(payload, "throughput", "items_per_hour")
        if steps_per_hour is not None:
            payload["throughput"]["samples_per_hour"] = round(steps_per_hour * batch_size, 3)
    return payload


def _grpo_config_summary(config: dict[str, Any]) -> dict[str, Any]:
    train_batch_size = _to_int(config.get("trainer.train_batch_size"))
    micro_train_batch_size_per_gpu = _to_int(config.get("trainer.micro_train_batch_size_per_gpu"))
    n_samples_per_prompt = _to_int(config.get("generator.n_samples_per_prompt"))
    configured_engines = _to_int(config.get("generator.inference_engine.num_engines"))
    policy_nodes = _to_int(config.get("trainer.placement.policy_num_nodes"))
    policy_gpus_per_node = _to_int(config.get("trainer.placement.policy_num_gpus_per_node"))
    policy_fsdp_size = _to_int(config.get("trainer.policy.fsdp_config.fsdp_size"))
    ref_fsdp_size = _to_int(config.get("trainer.ref.fsdp_config.fsdp_size"))
    total_gpu_count = configured_engines
    if policy_nodes is not None and policy_gpus_per_node is not None:
        total_gpu_count = policy_nodes * policy_gpus_per_node
    effective_samples = None
    if train_batch_size is not None and n_samples_per_prompt is not None:
        effective_samples = train_batch_size * n_samples_per_prompt
    samples_per_gpu = None
    if effective_samples is not None and total_gpu_count:
        samples_per_gpu = round(effective_samples / total_gpu_count, 6)
    fsdp_mesh = _fsdp_mesh_shape(
        total_gpu_count=total_gpu_count,
        fsdp_size=policy_fsdp_size,
    )
    hsdp_active = _hsdp_active(
        policy_nodes=policy_nodes,
        policy_gpus_per_node=policy_gpus_per_node,
        total_gpu_count=total_gpu_count,
        policy_fsdp_size=policy_fsdp_size,
        ref_fsdp_size=ref_fsdp_size,
    )
    return {
        "train_batch_size": train_batch_size,
        "micro_train_batch_size_per_gpu": micro_train_batch_size_per_gpu,
        "use_kl_loss": _to_bool(config.get("trainer.algorithm.use_kl_loss")),
        "kl_loss_coef": _to_float(config.get("trainer.algorithm.kl_loss_coef")),
        "use_entropy_loss": _to_bool(config.get("trainer.algorithm.use_entropy_loss")),
        "entropy_loss_coef": _to_float(config.get("trainer.algorithm.entropy_loss_coef")),
        "n_samples_per_prompt": n_samples_per_prompt,
        "effective_samples_per_step": effective_samples,
        "policy_num_nodes": policy_nodes,
        "policy_num_gpus_per_node": policy_gpus_per_node,
        "total_gpu_count": total_gpu_count,
        "rollout_engine_count": configured_engines,
        "vllm_gpu_memory_utilization": _to_float(config.get("generator.inference_engine.gpu_memory_utilization")),
        "policy_fsdp_size": policy_fsdp_size,
        "ref_fsdp_size": ref_fsdp_size,
        "fsdp_mesh_shape": fsdp_mesh,
        "hsdp_active": hsdp_active,
        "samples_per_gpu_per_step": samples_per_gpu,
        "max_env_workers": _to_int(config.get("environment.skyrl_gym.max_env_workers")),
        "skyrl_ray_pg_timeout_s": _to_int(config.get("skyrl.ray_pg_timeout_s")),
        "skyrl_worker_nccl_timeout_s": _to_int(config.get("skyrl.worker_nccl_timeout_s")),
        "max_ckpts_to_keep": _to_int(config.get("trainer.max_ckpts_to_keep")),
    }


def _training_health_summary(*, pipeline: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if pipeline != "cpp-grpo":
        return {"available": False, "reason": "pipeline_has_no_grpo_policy_health"}
    reward_metrics = metrics.get("reward") if isinstance(metrics.get("reward"), dict) else {}
    entropy = _to_float(metrics.get("policy/policy_entropy"))
    avg_final_reward = _to_float(metrics.get("loss/avg_final_rewards"))
    pass_rates = [
        _to_float(value)
        for key, value in reward_metrics.items()
        if isinstance(key, str) and key.startswith("avg_pass_at_")
    ]
    pass_rate = max([value for value in pass_rates if value is not None], default=None)
    grad_norm = _to_float(metrics.get("policy/grad_norm"))
    policy_loss = _to_float(metrics.get("policy/policy_loss"))
    signals = {
        "policy_entropy": entropy,
        "avg_final_reward": avg_final_reward,
        "max_pass_rate": pass_rate,
        "grad_norm": grad_norm,
        "policy_loss": policy_loss,
    }
    if not any(value is not None for value in signals.values()):
        return {"available": False, "reason": "no_grpo_policy_health_metrics", "signals": signals}

    low_entropy = entropy is not None and entropy <= 0.01
    reward_floor = avg_final_reward is not None and avg_final_reward <= -0.99
    zero_pass = pass_rate is not None and pass_rate <= 0.0
    zero_grad = grad_norm is not None and abs(grad_norm) <= 1e-9
    zero_policy_loss = policy_loss is not None and abs(policy_loss) <= 1e-9
    if low_entropy and reward_floor and zero_pass and (zero_grad or zero_policy_loss):
        return {
            "available": True,
            "verdict": "collapsed",
            "severity": "critical",
            "should_stop": True,
            "reason": "entropy_collapse_zero_advantage_trap",
            "signals": signals,
            "message": (
                "GRPO appears terminally collapsed: entropy is near zero, rewards are pinned at the "
                "format floor, pass rate is zero, and policy gradients/loss are zero."
            ),
        }
    if low_entropy and reward_floor:
        return {
            "available": True,
            "verdict": "collapse_risk",
            "severity": "warning",
            "should_stop": False,
            "reason": "low_entropy_reward_floor",
            "signals": signals,
            "message": "GRPO entropy is very low while rewards are pinned at the floor; inspect samples and gradients.",
        }
    return {
        "available": True,
        "verdict": "healthy_or_insufficient_collapse_signal",
        "severity": "ok",
        "should_stop": False,
        "reason": "collapse_guard_not_triggered",
        "signals": signals,
    }


def _fsdp_mesh_shape(*, total_gpu_count: int | None, fsdp_size: int | None) -> dict[str, int] | None:
    if total_gpu_count is None or fsdp_size is None or fsdp_size <= 0:
        return None
    if fsdp_size >= total_gpu_count:
        return {"ddp": 1, "fsdp": total_gpu_count}
    if total_gpu_count % fsdp_size != 0:
        return {"fsdp": fsdp_size}
    return {"ddp": total_gpu_count // fsdp_size, "fsdp": fsdp_size}


def _hsdp_active(
    *,
    policy_nodes: int | None,
    policy_gpus_per_node: int | None,
    total_gpu_count: int | None,
    policy_fsdp_size: int | None,
    ref_fsdp_size: int | None,
) -> bool:
    if (
        policy_nodes is None
        or policy_gpus_per_node is None
        or total_gpu_count is None
        or policy_fsdp_size is None
        or ref_fsdp_size is None
    ):
        return False
    return (
        policy_nodes > 1
        and policy_fsdp_size == policy_gpus_per_node
        and ref_fsdp_size == policy_gpus_per_node
        and 0 < policy_fsdp_size < total_gpu_count
    )


def _bottleneck_summary(*, metrics: dict[str, Any], timings: dict[str, Any]) -> dict[str, Any]:
    step_s = _to_float(metrics.get("timing/step")) or _to_float(timings.get("last_step_duration_s"))
    stages: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.startswith("timing/") or key == "timing/step":
            continue
        duration = _to_float(value)
        if duration is not None:
            stages[key.removeprefix("timing/")] = duration
    fallback = {
        "generate": timings.get("last_generate_duration_s"),
        "sync_weights": timings.get("last_sync_weights_duration_s"),
    }
    for key, value in fallback.items():
        if key not in stages and (duration := _to_float(value)) is not None:
            stages[key] = duration
    breakdown = [
        {
            "stage": stage,
            "seconds": round(seconds, 6),
            "step_percent": round(100.0 * seconds / step_s, 3) if step_s else None,
        }
        for stage, seconds in sorted(stages.items(), key=lambda item: item[1], reverse=True)
    ]
    dominant = breakdown[0] if breakdown else None
    if dominant is None:
        verdict = "unknown"
        message = "No timing breakdown is available in the scanned logs."
    elif dominant["stage"] in {"policy_train", "train_critic_and_policy"}:
        verdict = "policy_update_dominant"
        message = "Policy training is the largest observed step cost."
    elif dominant["stage"] == "generate":
        verdict = "rollout_dominant"
        message = "Trajectory generation is the largest observed step cost."
    elif dominant["stage"] == "fwd_logprobs_values_reward":
        verdict = "reward_or_forward_dominant"
        message = "Reward/logprob/value forward work is the largest observed step cost."
    elif dominant["stage"] == "sync_weights":
        verdict = "sync_dominant"
        message = "Weight synchronization is the largest observed step cost."
    else:
        verdict = "mixed"
        message = f"{dominant['stage']} is the largest observed step cost."
    return {
        "available": bool(breakdown),
        "step_seconds": round(step_s, 6) if step_s is not None else None,
        "dominant_stage": dominant,
        "stages": breakdown,
        "verdict": verdict,
        "message": message,
    }


def _augment_progress(
    progress: Any,
    *,
    unit: str,
    last_duration_s: Any = None,
    recent_durations_s: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(progress, dict):
        return None
    completed = _to_int(progress.get("completed"))
    total = _to_int(progress.get("total"))
    payload = dict(progress)
    if completed is not None and total:
        payload["remaining"] = max(total - completed, 0)
        payload["percent_exact"] = round(100.0 * completed / total, 4)
    timing = _parse_tqdm_timing(str(progress.get("line", "")))
    if last_duration_s is not None:
        timing["last_duration_s"] = _to_float(last_duration_s)
    recent = [float(item) for item in recent_durations_s or [] if _to_float(item) is not None]
    if recent:
        timing["recent_duration_avg_s"] = round(sum(recent) / len(recent), 3)
    payload["timing"] = timing
    seconds_per_item = timing.get("seconds_per_item")
    items_per_second = timing.get("items_per_second")
    if seconds_per_item is None and timing.get("recent_duration_avg_s") is not None:
        seconds_per_item = timing["recent_duration_avg_s"]
        items_per_second = 1.0 / seconds_per_item if seconds_per_item else None
    payload["throughput"] = _throughput_payload(unit, seconds_per_item=seconds_per_item, items_per_second=items_per_second)
    return payload


def _throughput_payload(
    unit: str,
    *,
    seconds_per_item: Any,
    items_per_second: Any,
) -> dict[str, Any]:
    seconds = _to_float(seconds_per_item)
    rate = _to_float(items_per_second)
    if seconds is None and rate:
        seconds = 1.0 / rate
    if rate is None and seconds:
        rate = 1.0 / seconds
    return {
        "unit": unit,
        "seconds_per_item": round(seconds, 6) if seconds is not None else None,
        "items_per_second": round(rate, 6) if rate is not None else None,
        "items_per_minute": round(rate * 60.0, 3) if rate is not None else None,
        "items_per_hour": round(rate * 3600.0, 3) if rate is not None else None,
    }


def _checkpoint_progress(
    *,
    artifacts: dict[str, Any],
    completed_step: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = artifacts.get("checkpoint") if isinstance(artifacts.get("checkpoint"), dict) else {}
    interval = _to_int(config.get("trainer.ckpt_interval")) or _to_int(config.get("sft.ckpt_interval"))
    step = _to_int(completed_step)
    next_checkpoint_step = None
    steps_until_next = None
    if interval and interval > 0 and step is not None:
        next_checkpoint_step = ((step // interval) + 1) * interval
        steps_until_next = max(next_checkpoint_step - step, 0)
    return {
        "latest_marker": checkpoint.get("latest_marker"),
        "highest_step": (checkpoint.get("highest") or {}).get("step") if isinstance(checkpoint.get("highest"), dict) else None,
        "in_progress_step": (checkpoint.get("in_progress") or {}).get("step")
        if isinstance(checkpoint.get("in_progress"), dict)
        else None,
        "next_checkpoint_step": next_checkpoint_step,
        "steps_until_next_checkpoint": steps_until_next,
        "interval": interval,
    }


def _resource_summary(
    *,
    instances: list[dict[str, Any]],
    node_health: dict[str, Any] | None,
) -> dict[str, Any]:
    accelerator_counts: dict[str, int] = {}
    for instance in instances:
        for accelerator in instance.get("accelerators", []) if isinstance(instance.get("accelerators"), list) else []:
            accelerator_type = str(accelerator.get("type") or "unknown")
            accelerator_counts[accelerator_type] = accelerator_counts.get(accelerator_type, 0) + int(
                accelerator.get("count") or 0
            )
    gpus = node_health.get("gpus") if isinstance(node_health, dict) and isinstance(node_health.get("gpus"), list) else []
    gpu_utils = [gpu.get("utilization_gpu_percent") for gpu in gpus if gpu.get("utilization_gpu_percent") is not None]
    gpu_mem_used = [gpu.get("memory_used_mib") for gpu in gpus if gpu.get("memory_used_mib") is not None]
    gpu_mem_total = [gpu.get("memory_total_mib") for gpu in gpus if gpu.get("memory_total_mib") is not None]
    return {
        "total_instance_count": len(instances),
        "active_instance_count": sum(1 for item in instances if item.get("status") in ACTIVE_INSTANCE_STATUSES),
        "zones": sorted({str(item.get("zone")) for item in instances if item.get("zone")}),
        "machine_types": sorted({str(item.get("machine_type")) for item in instances if item.get("machine_type")}),
        "accelerators": accelerator_counts,
        "gpu_count": sum(accelerator_counts.values()),
        "node_health_sample_scope": node_health.get("sample_scope") if isinstance(node_health, dict) else None,
        "sampled_node_count": len(node_health.get("nodes", []))
        if isinstance(node_health, dict) and isinstance(node_health.get("nodes"), list)
        else 0,
        "node_health_activity": node_health.get("activity") if isinstance(node_health, dict) else None,
        "sampled_gpu_count": len(gpus),
        "avg_gpu_utilization_percent": round(sum(gpu_utils) / len(gpu_utils), 3) if gpu_utils else None,
        "max_gpu_utilization_percent": max(gpu_utils) if gpu_utils else None,
        "avg_gpu_memory_used_mib": round(sum(gpu_mem_used) / len(gpu_mem_used), 3) if gpu_mem_used else None,
        "avg_gpu_memory_total_mib": round(sum(gpu_mem_total) / len(gpu_mem_total), 3) if gpu_mem_total else None,
    }


def _pipeline_commands(cluster: str, job: dict[str, Any] | None) -> dict[str, list[str]]:
    commands = {
        "queue": ["uv", "run", "w8-biayn", "ops", "queue", cluster],
        "logs": ["uv", "run", "w8-biayn", "ops", "logs", cluster, "--tail", "200"],
        "down": ["uv", "run", "w8-biayn", "ops", "down", cluster],
    }
    if job and job.get("job_id") is not None:
        job_id = str(job["job_id"])
        commands["logs"] = ["uv", "run", "w8-biayn", "ops", "logs", cluster, job_id, "--tail", "200"]
        commands["follow_logs"] = ["uv", "run", "w8-biayn", "ops", "logs", cluster, job_id, "--follow"]
        commands["cancel"] = ["uv", "run", "w8-biayn", "ops", "cancel", cluster, job_id]
    return commands


def _speed_comparison(
    *,
    pipeline: str,
    current_progress: dict[str, Any],
    baseline_statuses: list[dict[str, Any]],
    current_resources: Any = None,
) -> dict[str, Any]:
    if not baseline_statuses:
        return {"available": False, "reason": "no_baseline_status"}
    comparisons = []
    current_metrics = _comparison_metrics(current_progress)
    if current_metrics.get("gpu_count") is None:
        current_metrics["gpu_count"] = _gpu_count_from_resource_text(current_resources)
        _add_gpu_normalized_metrics(current_metrics)
    for status in baseline_statuses:
        baseline = _find_pipeline_status(status, pipeline)
        if not baseline:
            continue
        baseline_progress = baseline.get("progress")
        if not isinstance(baseline_progress, dict):
            baseline_progress = _progress_summary(
                pipeline=pipeline,
                log_signals=baseline.get("logs") if isinstance(baseline.get("logs"), dict) else {},
                artifacts=baseline.get("artifacts") if isinstance(baseline.get("artifacts"), dict) else {},
            )
        baseline_metrics = _comparison_metrics(baseline_progress)
        baseline_resources = _baseline_resources(baseline)
        if baseline_metrics.get("gpu_count") is None:
            baseline_metrics["gpu_count"] = _gpu_count_from_resource_text(baseline_resources)
            _add_gpu_normalized_metrics(baseline_metrics)
        _add_gpu_normalized_metrics(current_metrics)
        comparison = {
            "source": status.get("_status_source"),
            "run_id": status.get("run_id"),
            "pipeline": pipeline,
            "cluster": baseline.get("cluster"),
            "state": baseline.get("state"),
            "resources": baseline_resources,
            "baseline": baseline_metrics,
            "current": current_metrics,
            "training_step_speedup_factor": _time_speedup(
                baseline_metrics.get("seconds_per_training_step"),
                current_metrics.get("seconds_per_training_step"),
            ),
            "training_steps_per_hour_speedup_factor": _rate_speedup(
                baseline_metrics.get("training_steps_per_hour"),
                current_metrics.get("training_steps_per_hour"),
            ),
            "training_samples_per_hour_speedup_factor": _rate_speedup(
                baseline_metrics.get("training_samples_per_hour"),
                current_metrics.get("training_samples_per_hour"),
            ),
            "training_samples_per_gpu_hour_speedup_factor": _rate_speedup(
                baseline_metrics.get("training_samples_per_gpu_hour"),
                current_metrics.get("training_samples_per_gpu_hour"),
            ),
            "rollout_speedup_factor": _rate_speedup(
                baseline_metrics.get("rollout_samples_per_second"),
                current_metrics.get("rollout_samples_per_second"),
            ),
            "rollout_samples_per_gpu_second_speedup_factor": _rate_speedup(
                baseline_metrics.get("rollout_samples_per_gpu_second"),
                current_metrics.get("rollout_samples_per_gpu_second"),
            ),
        }
        comparison.update(_comparison_verdict(comparison))
        if any(
            comparison.get(key) is not None
            for key in (
                "training_step_speedup_factor",
                "training_steps_per_hour_speedup_factor",
                "training_samples_per_hour_speedup_factor",
                "rollout_speedup_factor",
            )
        ):
            comparisons.append(comparison)
    return {
        "available": bool(comparisons),
        "reason": None if comparisons else "no_comparable_throughput_in_baseline",
        "primary": comparisons[0] if comparisons else None,
        "baselines": comparisons,
    }


def _baseline_resources(baseline: dict[str, Any]) -> Any:
    active_job = baseline.get("active_job")
    if isinstance(active_job, dict) and active_job.get("resources"):
        return active_job.get("resources")
    queue = baseline.get("backend", {}).get("queue") if isinstance(baseline.get("backend"), dict) else None
    jobs = queue.get("jobs") if isinstance(queue, dict) and isinstance(queue.get("jobs"), list) else []
    return jobs[0].get("resources") if jobs and isinstance(jobs[0], dict) else None


def _comparison_metrics(progress: dict[str, Any]) -> dict[str, Any]:
    throughput = progress.get("throughput") if isinstance(progress.get("throughput"), dict) else {}
    training = progress.get("training") if isinstance(progress.get("training"), dict) else {}
    trajectory = progress.get("trajectory") if isinstance(progress.get("trajectory"), dict) else {}
    metrics = {
        "training_completed": training.get("completed"),
        "training_total": training.get("total"),
        "seconds_per_training_step": throughput.get("seconds_per_training_step"),
        "training_steps_per_hour": throughput.get("training_steps_per_hour"),
        "training_samples_per_hour": throughput.get("training_samples_per_hour"),
        "training_samples_per_gpu_hour": throughput.get("training_samples_per_gpu_hour"),
        "rollout_completed": trajectory.get("completed"),
        "rollout_total": trajectory.get("total"),
        "rollout_samples_per_second": throughput.get("rollout_samples_per_second"),
        "rollout_samples_per_gpu_second": throughput.get("rollout_samples_per_gpu_second"),
        "rollout_samples_per_hour": throughput.get("rollout_samples_per_hour"),
        "gpu_count": _nested_float(progress.get("grpo_config") if isinstance(progress.get("grpo_config"), dict) else {}, "total_gpu_count"),
        "effective_samples_per_step": _nested_float(
            progress.get("grpo_config") if isinstance(progress.get("grpo_config"), dict) else {},
            "effective_samples_per_step",
        ),
    }
    _add_gpu_normalized_metrics(metrics)
    return metrics


def _add_gpu_normalized_metrics(metrics: dict[str, Any]) -> None:
    gpu_count = _to_float(metrics.get("gpu_count"))
    if not gpu_count:
        return
    training_samples = _to_float(metrics.get("training_samples_per_hour"))
    rollout_samples = _to_float(metrics.get("rollout_samples_per_second"))
    if training_samples is not None and metrics.get("training_samples_per_gpu_hour") is None:
        metrics["training_samples_per_gpu_hour"] = round(training_samples / gpu_count, 3)
    if rollout_samples is not None and metrics.get("rollout_samples_per_gpu_second") is None:
        metrics["rollout_samples_per_gpu_second"] = round(rollout_samples / gpu_count, 6)


def _comparison_verdict(comparison: dict[str, Any]) -> dict[str, Any]:
    preferred = (
        comparison.get("training_samples_per_hour_speedup_factor")
        or comparison.get("training_steps_per_hour_speedup_factor")
        or comparison.get("training_step_speedup_factor")
        or comparison.get("rollout_speedup_factor")
    )
    factor = _to_float(preferred)
    if factor is None:
        verdict = "inconclusive"
        message = "No comparable throughput metric was available in the baseline."
    elif factor > 1.05:
        verdict = "faster"
        message = f"Current run achieved {factor:.3g}x baseline throughput on the primary comparable metric."
    elif factor < 0.95:
        verdict = "slower"
        message = (
            f"Current run achieved {factor:.3g}x baseline throughput on the primary comparable metric; "
            "lower than 1.0 means slower."
        )
    else:
        verdict = "roughly_equal"
        message = f"Current run is roughly equal to baseline throughput ({factor:.3g}x)."
    baseline_gpu_count = _nested_float(comparison.get("baseline"), "gpu_count")
    current_gpu_count = _nested_float(comparison.get("current"), "gpu_count")
    gpu_scale = None
    gpu_speedup_efficiency = None
    cost_verdict = "unknown"
    if baseline_gpu_count and current_gpu_count:
        gpu_scale = round(current_gpu_count / baseline_gpu_count, 4)
        if factor is not None:
            gpu_speedup_efficiency = round(factor / gpu_scale, 4) if gpu_scale else None
            if current_gpu_count > baseline_gpu_count and factor <= 1.0:
                cost_verdict = "cost_inefficient"
            elif gpu_speedup_efficiency is not None and gpu_speedup_efficiency < 0.8:
                cost_verdict = "under_scaling"
            else:
                cost_verdict = "acceptable"
    return {
        "verdict": verdict,
        "message": message,
        "gpu_scale_factor": gpu_scale,
        "gpu_speedup_efficiency": gpu_speedup_efficiency,
        "cost_verdict": cost_verdict,
    }


def _gpu_count_from_resource_text(resources: Any) -> int | None:
    if not isinstance(resources, str):
        return None
    if match := re.search(r"(\d+)x\[[A-Za-z0-9_.-]+:(\d+)\]", resources):
        return int(match.group(1)) * int(match.group(2))
    if match := re.search(r"[A-Za-z0-9_.-]+:(\d+)", resources):
        return int(match.group(1))
    return None


def _find_pipeline_status(status: dict[str, Any], pipeline: str) -> dict[str, Any] | None:
    for item in status.get("pipelines", []) if isinstance(status.get("pipelines"), list) else []:
        if isinstance(item, dict) and item.get("pipeline") == pipeline:
            return item
    return None


def _time_speedup(baseline_seconds: Any, current_seconds: Any) -> float | None:
    baseline = _to_float(baseline_seconds)
    current = _to_float(current_seconds)
    if not baseline or not current:
        return None
    return round(baseline / current, 4)


def _rate_speedup(baseline_rate: Any, current_rate: Any) -> float | None:
    baseline = _to_float(baseline_rate)
    current = _to_float(current_rate)
    if not baseline or not current:
        return None
    return round(current / baseline, 4)


def _parse_tqdm_timing(line: str) -> dict[str, Any]:
    match = TQDM_TIMING_RE.search(line)
    if not match:
        return {}
    elapsed_s = _parse_duration_s(match.group(1))
    eta_s = _parse_duration_s(match.group(2))
    rate_text = (match.group(3) or "").strip()
    items_per_second, seconds_per_item = _parse_tqdm_rate(rate_text)
    return {
        "elapsed_s": elapsed_s,
        "eta_s": eta_s,
        "rate_text": rate_text or None,
        "items_per_second": items_per_second,
        "seconds_per_item": seconds_per_item,
    }


def _parse_duration_s(text: str | None) -> int | None:
    if not text or "?" in text:
        return None
    parts = text.strip().split(":")
    if not all(part.isdigit() for part in parts):
        return None
    values = [int(part) for part in parts]
    if len(values) == 2:
        minutes, seconds = values
        return minutes * 60 + seconds
    if len(values) == 3:
        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds
    if len(values) == 1:
        return values[0]
    return None


def _parse_tqdm_rate(text: str) -> tuple[float | None, float | None]:
    if not text or "?" in text:
        return None, None
    if match := re.search(r"([0-9.]+)\s*it/s", text):
        rate = float(match.group(1))
        return rate, 1.0 / rate if rate else None
    if match := re.search(r"([0-9.]+)s/it", text):
        seconds = float(match.group(1))
        return 1.0 / seconds if seconds else None, seconds
    return None, None


def _record_metric(metrics: dict[str, Any], name: str, value: float) -> None:
    if name.startswith("reward/"):
        reward = metrics.setdefault("reward", {})
        if isinstance(reward, dict):
            reward[name.split("/", 1)[1]] = value
        return
    if value.is_integer():
        metrics[name] = int(value)
    else:
        metrics[name] = value


def _nested_float(payload: dict[str, Any] | None, *keys: str) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _to_float(value)


def _expected_final_step(pipeline: str, *, sft: int | None, grpo: int | None) -> int | None:
    if pipeline == "cpp-sft":
        return sft
    if pipeline == "cpp-grpo":
        return grpo
    return None


def _parse_json_payload(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    for opener in ("{", "["):
        index = stripped.find(opener)
        if index >= 0:
            try:
                return json.loads(stripped[index:])
            except json.JSONDecodeError:
                continue
    return None


def _parse_marker(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


def _file_entry(files: list[Any], path: str) -> dict[str, Any] | None:
    for item in files:
        if isinstance(item, dict) and item.get("path") == path:
            return {"path": path, "bytes": item.get("bytes"), "sha256": item.get("sha256")}
    return None


def _manifest_path(item: Any) -> str:
    return str(item.get("path", "")) if isinstance(item, dict) else ""


def _gs_uris(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("gs://")]


def _gsutil_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for line in text.splitlines():
        if match := GSUTIL_OBJECT_RE.search(line):
            objects.append(
                {
                    "bytes": int(match.group(1)),
                    "timestamp": match.group(2),
                    "uri": match.group(3),
                }
            )
    return objects


def _strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _basename(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.rstrip("/").rsplit("/", 1)[-1]


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None
