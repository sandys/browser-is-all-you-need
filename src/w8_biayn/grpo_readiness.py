"""Machine-checkable readiness gates for GRPO launches and status snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import run_status
from .sky_config import GRPO_MIN_SAMPLES_PER_GPU_STEP, GRPO_MULTINODE_RESUME_MIN_DISK_GB
from .sky_config import gpu_count_from_accelerators

SCHEMA_VERSION = "w8-grpo-readiness-v1"
BLOCKING_SEVERITIES = {"critical", "action_required"}
ACTIVE_STATUS_VALUES = {"RUNNING", "PENDING", "SETTING_UP", "STARTING"}


def build_grpo_readiness(
    rendered_config_path: str | Path,
    *,
    status_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a dashboard-friendly readiness report for one rendered GRPO config."""

    path = Path(rendered_config_path)
    checks: list[dict[str, Any]] = []
    if not path.exists():
        _add_check(
            checks,
            "rendered_config.exists",
            False,
            "critical",
            f"Rendered config does not exist: {path}",
            remediation="Render cpp-grpo before launch.",
        )
        return _payload(path=path, config={}, checks=checks, status_payload=status_payload)

    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        _add_check(
            checks,
            "rendered_config.parse",
            False,
            "critical",
            f"Rendered config could not be parsed: {exc}",
            remediation="Re-render cpp-grpo and rerun ops grpo-readiness.",
        )
        return _payload(path=path, config={}, checks=checks, status_payload=status_payload)

    envs = config.get("envs") if isinstance(config.get("envs"), dict) else {}
    pipeline = str(envs.get("W8_BIAYN_PIPELINE") or "")
    run = str(config.get("run") or "")
    _add_check(
        checks,
        "rendered_config.pipeline",
        pipeline == "cpp-grpo",
        "critical",
        "Rendered config is for cpp-grpo.",
        evidence={"pipeline": pipeline or None},
        remediation="Use a cpp-grpo rendered YAML.",
    )
    normal_training = "python -m w8_biayn.integrations.skyrl_cpp_perf_main" in run
    export_only = "skyrl_sft_export_checkpoint_main" in run and not normal_training
    if pipeline == "cpp-grpo" and export_only:
        _add_check(
            checks,
            "rendered_config.training_mode",
            True,
            "info",
            "Rendered GRPO config is export-only; training readiness gates are not applicable.",
        )
        return _payload(path=path, config=config, checks=checks, status_payload=status_payload, not_applicable=True)

    _add_check(
        checks,
        "rendered_config.training_entrypoint",
        normal_training,
        "critical",
        "Rendered GRPO config invokes the SkyRL C++ performance entrypoint.",
        remediation="Re-render cpp-grpo; export-only configs are not training launches.",
    )
    _add_static_runtime_checks(checks, config=config, run=run)
    _add_static_multinode_checks(checks, config=config, run=run)
    if status_payload is not None:
        _add_status_checks(checks, status_payload=status_payload)
    return _payload(path=path, config=config, checks=checks, status_payload=status_payload)


def readiness_blocks_launch(payload: dict[str, Any]) -> bool:
    """Return True when a readiness payload should block a paid launch or operator flow."""

    return any(
        check.get("ok") is False and check.get("severity") in BLOCKING_SEVERITIES
        for check in payload.get("checks", [])
        if isinstance(check, dict)
    )


def _payload(
    *,
    path: Path,
    config: dict[str, Any],
    checks: list[dict[str, Any]],
    status_payload: dict[str, Any] | None,
    not_applicable: bool = False,
) -> dict[str, Any]:
    failed = [check for check in checks if check.get("ok") is False]
    critical_failed = [check for check in failed if check.get("severity") == "critical"]
    action_required = [check for check in failed if check.get("severity") == "action_required"]
    if not_applicable:
        overall = "not_applicable"
    elif critical_failed:
        overall = "fail"
    elif action_required:
        overall = "action_required"
    else:
        overall = "pass"
    run_id = None
    envs = config.get("envs") if isinstance(config.get("envs"), dict) else {}
    if envs:
        run_id = envs.get("W8_BIAYN_RUN_ID") or None
    if status_payload:
        run_id = status_payload.get("run_id") or run_id
    return {
        "schema_version": SCHEMA_VERSION,
        "overall": overall,
        "run_id": run_id,
        "rendered_config_path": str(path),
        "status_source": "provided" if status_payload is not None else None,
        "check_counts": {
            "total": len(checks),
            "failed": len(failed),
            "critical_failed": len(critical_failed),
            "action_required": len(action_required),
            "warnings": len([check for check in failed if check.get("severity") == "warning"]),
        },
        "checks": checks,
        "recommended_commands": _recommended_commands(checks, status_payload=status_payload),
    }


def _add_static_runtime_checks(checks: list[dict[str, Any]], *, config: dict[str, Any], run: str) -> None:
    _add_check(
        checks,
        "runtime.docker_mounts",
        "-v /var/run/docker.sock:/var/run/docker.sock" in run and "-v /tmp:/tmp" in run,
        "critical",
        "GRPO container mounts Docker socket and host /tmp for Docker-outside-Docker rewards.",
        remediation="Keep Docker socket and /tmp mounts in training_container_prefix.",
    )
    _add_check(
        checks,
        "runtime.preflight",
        "uv run w8-biayn cpp harness preflight" in run and "w8-biayn cpp harness preflight" in run,
        "critical",
        "Host and in-container C++ runtime preflights are present before GRPO training.",
        remediation="Rendered GRPO must preflight before GCS/model staging and again inside the container.",
    )
    _add_check(
        checks,
        "skyrl.patch.io",
        "python -m w8_biayn.integrations.skyrl_io_patch" in run,
        "critical",
        "SkyRL checkpoint download compatibility patch is invoked.",
        remediation="Invoke skyrl_io_patch before uv sync in the training container.",
    )
    _add_check(
        checks,
        "skyrl.patch.vllm_logprob",
        "python -m w8_biayn.integrations.skyrl_vllm_logprob_patch" in run,
        "critical",
        "SkyRL vLLM token/logprob alignment patch is invoked.",
        remediation="Invoke skyrl_vllm_logprob_patch before uv sync in the training container.",
    )
    _add_check(
        checks,
        "skyrl.patch.grpo_health",
        "python -m w8_biayn.integrations.skyrl_grpo_health_patch" in run,
        "critical",
        "SkyRL GRPO health logging patch is invoked.",
        remediation="Invoke skyrl_grpo_health_patch before uv sync in the training container.",
    )
    _add_check(
        checks,
        "skyrl.patch.startup",
        "python -m w8_biayn.integrations.skyrl_startup_patch" in run,
        "critical",
        "SkyRL startup stage logging patch is invoked.",
        remediation="Invoke skyrl_startup_patch before uv sync in the training container.",
    )
    _add_tracking_checks(checks, run=run)
    _add_check(
        checks,
        "network.nccl_env",
        "-e NCCL_IB_DISABLE=1" in run
        and '-e NCCL_SOCKET_IFNAME="^lo,docker,veth"' in run
        and "-e NCCL_DEBUG=WARN" in run,
        "critical",
        "Container passes required NCCL network environment.",
        remediation="Propagate NCCL_IB_DISABLE, NCCL_SOCKET_IFNAME, and NCCL_DEBUG into docker run.",
    )
    gloo_uses_default_route = (
        'export W8_GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$(ip route show default' in run
        and '-e GLOO_SOCKET_IFNAME="$W8_GLOO_SOCKET_IFNAME"' in run
    )
    gloo_bad_exclusion = 'GLOO_SOCKET_IFNAME="^lo,docker,veth"' in run or "GLOO_SOCKET_IFNAME=^lo,docker,veth" in run
    _add_check(
        checks,
        "network.gloo_concrete_interface",
        gloo_uses_default_route and not gloo_bad_exclusion,
        "critical",
        "Container passes a concrete default-route Gloo interface, not NCCL exclusion syntax.",
        remediation="Derive W8_GLOO_SOCKET_IFNAME from ip route and pass that value to Docker.",
    )
    extracted = _extract_config(run)
    _add_check(
        checks,
        "skyrl.timeouts",
        extracted.get("skyrl.ray_pg_timeout_s") == 1800
        and extracted.get("skyrl.worker_nccl_timeout_s") == 3600,
        "critical",
        "SkyRL/Ray distributed timeouts are long enough for checkpoint restore and HF export barriers.",
        evidence={
            "skyrl_ray_pg_timeout_s": extracted.get("skyrl.ray_pg_timeout_s"),
            "skyrl_worker_nccl_timeout_s": extracted.get("skyrl.worker_nccl_timeout_s"),
        },
        remediation="Keep SKYRL_RAY_PG_TIMEOUT_IN_S=1800 and SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600.",
    )
    _add_check(
        checks,
        "training.kl_entropy_enabled",
        extracted.get("trainer.algorithm.use_kl_loss") is True
        and _to_float(extracted.get("trainer.algorithm.kl_loss_coef"), default=0.0) > 0
        and extracted.get("trainer.algorithm.use_entropy_loss") is True
        and _to_float(extracted.get("trainer.algorithm.entropy_loss_coef"), default=0.0) > 0,
        "critical",
        "GRPO KL anchor and entropy bonus are enabled with positive coefficients.",
        evidence={
            "use_kl_loss": extracted.get("trainer.algorithm.use_kl_loss"),
            "kl_loss_coef": extracted.get("trainer.algorithm.kl_loss_coef"),
            "use_entropy_loss": extracted.get("trainer.algorithm.use_entropy_loss"),
            "entropy_loss_coef": extracted.get("trainer.algorithm.entropy_loss_coef"),
        },
        remediation="Do not launch full GRPO with KL or entropy disabled unless the reward plan changes.",
    )
    _add_checkpoint_retention_check(checks, extracted=extracted)


def _add_static_multinode_checks(checks: list[dict[str, Any]], *, config: dict[str, Any], run: str) -> None:
    num_nodes = _to_int(config.get("num_nodes"), default=1)
    resources = config.get("resources") if isinstance(config.get("resources"), dict) else {}
    envs = config.get("envs") if isinstance(config.get("envs"), dict) else {}
    accelerators = str(resources.get("accelerators") or "")
    gpus_per_node = gpu_count_from_accelerators(accelerators)
    total_gpu_count = gpus_per_node * num_nodes
    extracted = _extract_config(run)
    if num_nodes <= 1:
        _add_check(
            checks,
            "multinode.not_requested",
            True,
            "info",
            "Rendered GRPO config is single-node; multi-node-only checks are skipped.",
        )
        return
    policy_nodes = _to_int(extracted.get("trainer.placement.policy_num_nodes"))
    policy_gpus = _to_int(extracted.get("trainer.placement.policy_num_gpus_per_node"))
    policy_fsdp = _to_int(extracted.get("trainer.policy.fsdp_config.fsdp_size"))
    ref_fsdp = _to_int(extracted.get("trainer.ref.fsdp_config.fsdp_size"))
    rollout_engines = _to_int(extracted.get("generator.inference_engine.num_engines"))
    _add_check(
        checks,
        "multinode.hsdp_mesh",
        policy_nodes == num_nodes
        and policy_gpus == gpus_per_node
        and policy_fsdp == gpus_per_node
        and ref_fsdp == gpus_per_node,
        "critical",
        "Multi-node GRPO uses node-local FSDP sharding with HSDP replication across nodes.",
        evidence={
            "num_nodes": num_nodes,
            "gpus_per_node": gpus_per_node,
            "policy_num_nodes": policy_nodes,
            "policy_num_gpus_per_node": policy_gpus,
            "policy_fsdp_size": policy_fsdp,
            "ref_fsdp_size": ref_fsdp,
        },
        remediation="Set policy/ref fsdp_size to GPUs per node and policy placement to the rendered node shape.",
    )
    _add_check(
        checks,
        "multinode.rollout_engines",
        rollout_engines == total_gpu_count,
        "critical",
        "Rollout engine count matches total GPUs for colocated TP=1/DP=1 GRPO.",
        evidence={"rollout_engines": rollout_engines, "total_gpu_count": total_gpu_count},
        remediation="Set generator.inference_engine.num_engines to num_nodes * GPUs_PER_NODE.",
    )
    effective_samples = _to_int(envs.get("W8_BIAYN_EFFECTIVE_SAMPLES_PER_STEP"))
    if effective_samples is None:
        train_batch_size = _to_int(extracted.get("trainer.train_batch_size"))
        n_samples = _to_int(extracted.get("generator.n_samples_per_prompt"))
        effective_samples = train_batch_size * n_samples if train_batch_size and n_samples else None
    max_workers = _to_int(extracted.get("environment.skyrl_gym.max_env_workers"))
    allow_low = str(envs.get("W8_BIAYN_ALLOW_LOW_MULTINODE_UTILIZATION", "false")).lower() == "true"
    min_effective_samples = GRPO_MIN_SAMPLES_PER_GPU_STEP * total_gpu_count
    utilization_ok = (
        effective_samples is not None
        and effective_samples >= min_effective_samples
        and max_workers is not None
        and max_workers >= effective_samples
    )
    _add_check(
        checks,
        "multinode.utilization_gate",
        utilization_ok or allow_low,
        "critical",
        "Multi-node GRPO has enough samples/GPU/step and reward workers.",
        evidence={
            "effective_samples_per_step": effective_samples,
            "minimum_effective_samples_per_step": min_effective_samples,
            "max_env_workers": max_workers,
            "allow_low_multinode_utilization": allow_low,
        },
        remediation="For 2x[A100:8], use --train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256.",
    )
    if allow_low and not utilization_ok:
        _add_check(
            checks,
            "multinode.low_utilization_override",
            False,
            "warning",
            "Low multi-node utilization was explicitly allowed.",
            evidence={"effective_samples_per_step": effective_samples, "total_gpu_count": total_gpu_count},
            remediation="Treat throughput comparisons from this run as a deliberate low-utilization experiment.",
        )
    resume_mode = _resume_mode(run)
    disk_size = _to_int(resources.get("disk_size"))
    _add_check(
        checks,
        "multinode.resume_disk",
        not resume_mode or (disk_size is not None and disk_size >= GRPO_MULTINODE_RESUME_MIN_DISK_GB),
        "critical",
        "Multi-node GRPO resume has enough boot disk for FSDP checkpoint restore.",
        evidence={"resume_mode": resume_mode, "disk_size": disk_size},
        remediation=f"Use --disk-size {GRPO_MULTINODE_RESUME_MIN_DISK_GB} or larger for multi-node GRPO resume.",
    )


def _add_checkpoint_retention_check(checks: list[dict[str, Any]], *, extracted: dict[str, Any]) -> None:
    max_ckpts = _to_int(extracted.get("trainer.max_ckpts_to_keep"))
    ok = max_ckpts is None or max_ckpts == -1 or max_ckpts >= 8
    _add_check(
        checks,
        "training.checkpoint_retention",
        ok,
        "critical",
        "GRPO retains enough checkpoints to avoid losing the last healthy policy.",
        evidence={"max_ckpts_to_keep": max_ckpts},
        remediation="Use --max-ckpts-to-keep 8 or -1 for full GRPO training.",
    )
    hf_interval = _to_int(extracted.get("trainer.hf_save_interval"))
    _add_check(
        checks,
        "training.hf_save_interval",
        hf_interval is None or hf_interval == -1 or hf_interval >= 1000,
        "warning",
        "GRPO HF export interval is large enough to avoid frequent GPU-idling exports.",
        evidence={"hf_save_interval": hf_interval},
        remediation="Use a large interval such as --hf-save-interval 10000 unless intermediate exports are required.",
    )


def _add_tracking_checks(checks: list[dict[str, Any]], *, run: str) -> None:
    console_enabled = _trainer_logger_has(run, "console")
    mlflow_enabled = _mlflow_tracking_enabled(run)
    _add_check(
        checks,
        "tracking.console_logger",
        console_enabled,
        "critical",
        "GRPO keeps console logging enabled for log-tail fallback status.",
        evidence={"console_enabled": console_enabled},
        remediation="Keep console in trainer.logger even when MLflow tracking is enabled.",
    )
    _add_check(
        checks,
        "tracking.mlflow_server",
        (not mlflow_enabled) or ("mlflow server" in run and "MLFLOW_TRACKING_URI" in run),
        "critical",
        (
            "MLflow tracking is disabled for this render; no tracking server is required."
            if not mlflow_enabled
            else "GRPO starts an MLflow Tracking Server and points SkyRL at it through MLFLOW_TRACKING_URI."
        ),
        evidence={"mlflow_enabled": mlflow_enabled},
        remediation="Use the rendered MLflow tracking-server setup instead of direct SQLite writes.",
    )
    _add_check(
        checks,
        "tracking.mlflow_persistence",
        (not mlflow_enabled)
        or (
            "sync_mlflow_tracking_once" in run
            and "$W8_RUN_GCS_PREFIX/tracking/mlflow/mlflow.db" in run
            and "gcloud storage cp \"$snapshot\"" in run
        ),
        "critical",
        (
            "MLflow tracking is disabled for this render; no MLflow GCS persistence is required."
            if not mlflow_enabled
            else "MLflow SQLite backend snapshots are synced to the run GCS tracking path."
        ),
        evidence={"mlflow_enabled": mlflow_enabled},
        remediation="Keep the host-side MLflow tracking sync loop enabled for headless ops metrics.",
    )


def _mlflow_tracking_enabled(run: str) -> bool:
    if 'W8_ENABLE_MLFLOW_TRACKING="1"' in run:
        return True
    if 'W8_ENABLE_MLFLOW_TRACKING="0"' in run:
        return False
    return _trainer_logger_has(run, "mlflow")


def _trainer_logger_has(run: str, backend: str) -> bool:
    prefix = "trainer.logger="
    for line in run.splitlines():
        value = line.strip()
        if not value.startswith(prefix):
            continue
        value = value.removeprefix(prefix).strip().rstrip("\\").strip()
        value = value.strip("\"'")
        if value.startswith("[") and value.endswith("]"):
            return backend in {item.strip().lower() for item in value[1:-1].split(",")}
        return value.lower() == backend
    return False


def _add_status_checks(checks: list[dict[str, Any]], *, status_payload: dict[str, Any]) -> None:
    _add_check(
        checks,
        "status.schema",
        status_payload.get("schema_version") == "w8-run-status-v1",
        "critical",
        "Status JSON uses w8-run-status-v1.",
        evidence={"schema_version": status_payload.get("schema_version")},
        remediation="Use ops run-status to produce the status JSON.",
    )
    pipeline = _find_pipeline(status_payload, "cpp-grpo")
    _add_check(
        checks,
        "status.cpp_grpo_pipeline",
        pipeline is not None,
        "critical",
        "Status JSON includes the cpp-grpo pipeline.",
        remediation="Run ops run-status with --pipeline cpp-grpo or without a pipeline filter.",
    )
    if pipeline is None:
        return
    _add_status_node_health_check(checks, pipeline=pipeline)
    _add_status_startup_check(checks, pipeline=pipeline)
    _add_status_training_health_check(checks, pipeline=pipeline)
    _add_status_learning_signal_check(checks, pipeline=pipeline)
    _add_status_recovery_check(checks, pipeline=pipeline)


def _add_status_node_health_check(checks: list[dict[str, Any]], *, pipeline: dict[str, Any]) -> None:
    progress = pipeline.get("progress") if isinstance(pipeline.get("progress"), dict) else {}
    grpo_config = progress.get("grpo_config") if isinstance(progress.get("grpo_config"), dict) else {}
    resources = pipeline.get("resources") if isinstance(pipeline.get("resources"), dict) else {}
    active_job = pipeline.get("active_job") if isinstance(pipeline.get("active_job"), dict) else {}
    job_status = str(active_job.get("status") or "").upper()
    active = job_status in ACTIVE_STATUS_VALUES or pipeline.get("state") in {"running", "checkpointing", "provisioning"}
    policy_nodes = _to_int(grpo_config.get("policy_num_nodes"))
    total_instances = _to_int(resources.get("total_instance_count"), default=0)
    multinode = (policy_nodes is not None and policy_nodes > 1) or total_instances > 1
    if not active or not multinode:
        _add_check(
            checks,
            "status.multinode_node_health",
            True,
            "info",
            "Live all-node GPU telemetry is not required for this status snapshot.",
            evidence={"active": active, "multinode": multinode},
        )
        return
    node_health = pipeline.get("node_health") if isinstance(pipeline.get("node_health"), dict) else {}
    expected_nodes = _to_int(node_health.get("expected_node_count")) or _to_int(resources.get("active_instance_count"))
    sampled_nodes = _to_int(node_health.get("sampled_node_count")) or _to_int(resources.get("sampled_node_count"))
    failed_nodes = _to_int(node_health.get("failed_node_count"), default=0)
    expected_gpus = _to_int(grpo_config.get("total_gpu_count")) or _to_int(resources.get("gpu_count"))
    sampled_gpus = _to_int(resources.get("sampled_gpu_count"))
    if sampled_gpus is None and isinstance(node_health.get("gpus"), list):
        sampled_gpus = len(node_health["gpus"])
    ok = (
        node_health.get("sample_scope") == "all_active"
        and expected_nodes is not None
        and sampled_nodes == expected_nodes
        and failed_nodes == 0
        and (expected_gpus is None or sampled_gpus == expected_gpus)
    )
    _add_check(
        checks,
        "status.multinode_node_health",
        ok,
        "action_required",
        "Active multi-node GRPO status proves all active nodes and GPUs were sampled.",
        evidence={
            "sample_scope": node_health.get("sample_scope"),
            "expected_node_count": expected_nodes,
            "sampled_node_count": sampled_nodes,
            "failed_node_count": failed_nodes,
            "expected_gpu_count": expected_gpus,
            "sampled_gpu_count": sampled_gpus,
        },
        remediation="Poll with ops run-status --node-health and do not claim multi-node is active until sample_scope=all_active.",
    )


def _add_status_startup_check(checks: list[dict[str, Any]], *, pipeline: dict[str, Any]) -> None:
    progress = pipeline.get("progress") if isinstance(pipeline.get("progress"), dict) else {}
    startup = progress.get("startup") if isinstance(progress.get("startup"), dict) else {}
    action = startup.get("recommended_action")
    action_required = action in {"inspect_startup_stage", "inspect_failed_startup_or_relaunch"}
    _add_check(
        checks,
        "status.startup_progress",
        not action_required,
        "action_required",
        "GRPO startup is not stuck before the first scalar metric.",
        evidence={
            "available": startup.get("available"),
            "active_stage": startup.get("active_stage"),
            "max_elapsed_s": startup.get("max_elapsed_s"),
            "long_running": startup.get("long_running"),
            "tracking_state": startup.get("tracking_state"),
            "scalar_metrics_available": startup.get("scalar_metrics_available"),
            "recommended_action": action,
        },
        remediation=(
            "If startup is long-running with run_active_no_metrics, stop the run, inspect the startup stage, "
            "and do not relaunch paid multi-node GRPO until that stage has a concrete fix."
        ),
    )


def _add_status_training_health_check(checks: list[dict[str, Any]], *, pipeline: dict[str, Any]) -> None:
    progress = pipeline.get("progress") if isinstance(pipeline.get("progress"), dict) else {}
    health = progress.get("training_health") if isinstance(progress.get("training_health"), dict) else {}
    if not health:
        health = {}
    _add_check(
        checks,
        "status.training_health_should_continue",
        health.get("should_stop") is not True,
        "action_required",
        "GRPO training health does not require stopping.",
        evidence={
            "verdict": health.get("verdict"),
            "should_stop": health.get("should_stop"),
            "recommended_action": health.get("recommended_action"),
            "checkpoint_step": health.get("checkpoint_step"),
        },
        remediation="If should_stop=true, stop the run and follow training_health.recommended_action.",
    )


def _add_status_learning_signal_check(checks: list[dict[str, Any]], *, pipeline: dict[str, Any]) -> None:
    progress = pipeline.get("progress") if isinstance(pipeline.get("progress"), dict) else {}
    learning_signal = progress.get("learning_signal") if isinstance(progress.get("learning_signal"), dict) else {}
    available = learning_signal.get("available") is True
    _add_check(
        checks,
        "status.learning_signal_available",
        available,
        "warning",
        "Status JSON includes GRPO learning-signal metrics.",
        evidence={
            "available": learning_signal.get("available"),
            "verdict": learning_signal.get("verdict"),
            "recommended_action": learning_signal.get("recommended_action"),
        },
        remediation=(
            "Poll a run rendered with skyrl_grpo_health_patch and enough log tail to include W8_GRPO_HEALTH lines."
        ),
    )
    if not available:
        return
    recommended_action = learning_signal.get("recommended_action")
    action_required = recommended_action in {"evaluate_checkpoint", "stop_and_evaluate_checkpoint"}
    _add_check(
        checks,
        "status.learning_signal_recommendation",
        not action_required,
        "action_required",
        "GRPO learning signal does not currently recommend checkpoint evaluation.",
        evidence={
            "verdict": learning_signal.get("verdict"),
            "severity": learning_signal.get("severity"),
            "recommended_action": recommended_action,
            "reasons": learning_signal.get("reasons"),
        },
        remediation="If learning_signal recommends evaluation, export/evaluate the checkpoint before spending more GPU time.",
    )


def _add_status_recovery_check(checks: list[dict[str, Any]], *, pipeline: dict[str, Any]) -> None:
    recovery = pipeline.get("recovery") if isinstance(pipeline.get("recovery"), dict) else {}
    _add_check(
        checks,
        "status.recovery_action",
        recovery.get("available") is not True,
        "action_required",
        "Status JSON has no pending failed-run recovery action.",
        evidence={
            "available": recovery.get("available"),
            "recommended_action": recovery.get("recommended_action"),
            "resume_checkpoint_step": recovery.get("resume_checkpoint_step"),
        },
        remediation="Follow pipelines[].recovery before launching or reusing a cluster.",
    )


def _recommended_commands(checks: list[dict[str, Any]], *, status_payload: dict[str, Any] | None) -> list[list[str]]:
    commands: list[list[str]] = []
    run_id = status_payload.get("run_id") if status_payload else None
    if any(check.get("id") == "status.multinode_node_health" and check.get("ok") is False for check in checks) and run_id:
        commands.append(
            [
                "uv",
                "run",
                "w8-biayn",
                "ops",
                "run-status",
                "--run-id",
                str(run_id),
                "--credentials",
                ".gcp-service-account.json",
                "--pipeline",
                "cpp-grpo",
                "--node-health",
                "--out",
                f".w8-biayn/runs/{run_id}/status.json",
            ]
        )
    pipeline = _find_pipeline(status_payload or {}, "cpp-grpo")
    if pipeline:
        pipeline_commands = pipeline.get("commands") if isinstance(pipeline.get("commands"), dict) else {}
        for key in ("cancel", "down", "logs"):
            command = pipeline_commands.get(key)
            if isinstance(command, list) and all(isinstance(part, str) for part in command):
                commands.append(command)
    return commands


def _add_check(
    checks: list[dict[str, Any]],
    check_id: str,
    ok: bool,
    severity: str,
    message: str,
    *,
    evidence: dict[str, Any] | None = None,
    remediation: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "id": check_id,
        "severity": severity,
        "ok": ok,
        "message": message,
    }
    if evidence:
        payload["evidence"] = evidence
    if remediation:
        payload["remediation"] = remediation
    checks.append(payload)


def _extract_config(run: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for line in run.splitlines():
        run_status._record_config_matches(config, line)
    return config


def _resume_mode(run: str) -> str | None:
    for line in run.splitlines():
        if "trainer.resume_mode=" not in line:
            continue
        value = line.split("trainer.resume_mode=", 1)[1].split()[0].strip().strip("\\").strip("'\"")
        if value and value != "null":
            return value
    return None


def _find_pipeline(status_payload: dict[str, Any], pipeline: str) -> dict[str, Any] | None:
    pipelines = status_payload.get("pipelines") if isinstance(status_payload.get("pipelines"), list) else []
    for item in pipelines:
        if isinstance(item, dict) and item.get("pipeline") == pipeline:
            return item
    return None


def _to_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
