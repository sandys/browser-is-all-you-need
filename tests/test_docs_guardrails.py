from __future__ import annotations

from pathlib import Path


DOCS = (
    Path("README.md"),
    Path(".agents/REPO_GUIDE.md"),
    Path(".agents/skills/w8-biayn-framework/SKILL.md"),
)

REQUIRED_GRPO_GUARDRAILS = (
    "ops grpo-readiness",
    "w8-grpo-readiness-v1",
    "GLOO_SOCKET_IFNAME",
    "concrete default-route",
    "skyrl_io_patch.py",
    "skyrl_vllm_logprob_patch.py",
    "skyrl_grpo_health_patch.py",
    "W8_GRPO_HEALTH",
    "learning_signal",
    "phase_timing",
    "MLflow Tracking Server",
    "MLFLOW_TRACKING_URI",
    "w8-mlflow-metrics-v1",
    "ops metrics",
    "SSH tunnel",
    "GCS-synced SQLite",
    "MLflow runtime params",
    "mlflow_params",
    "config_sources",
    "skyrl_startup_patch.py",
    "run_active_no_metrics",
    "metrics_available",
    "node_health.sample_scope=all_active",
    "training_health.should_stop=true",
    "deterministic_low_gradient",
    "stop_and_evaluate_checkpoint",
    "uplift_gate",
    "uplift_gate.passed=true",
    "held_out_lift_but_gate_failed",
    "missing_runtime_task_ids",
    "eval raw-report",
    "src/w8_biayn/reporting.py",
)


def test_docs_keep_multinode_grpo_guardrails_documented() -> None:
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in REQUIRED_GRPO_GUARDRAILS if phrase not in text]
        assert not missing, f"{path} is missing guardrails: {missing}"
