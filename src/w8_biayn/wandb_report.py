"""Repo-owned W&B instrumentation: metrics, tables, histograms, artifacts, alerts.

This module is the single place that decides WHERE each kind of run data goes,
following the drill path curve -> distribution -> sample -> artifact:

- Training dynamics (`train/*`, `rollout/*`, `perf/*`) come from SLIME itself
  and are not touched here.
- Live rollout health during training (`rollout_health/*`) is logged by
  :class:`RolloutHealth` from the custom generate hook. The SLIME rollout actor
  is already attached to the stage's W&B run in shared mode
  (``init_wandb_secondary``), so ``wandb.log`` lands in the same run as
  ``train/*`` with no extra ``wandb.init``.
- Eval outcomes (`eval/*`) are logged by the offline scorer onto the stage's own
  resumed run. The keys are identical across stages, so the W&B workspace
  overlays base/sft/grpo in one panel per metric.
- Per-sample forensics go to ``wandb.Table``; categorical failures additionally
  become ``eval/abort/<reason>`` counts that drive alerts.
- The uplift verdict goes to the pipeline run as a table plus ``uplift/*``
  summary keys.
- Files (records, reports) become artifacts; GCS checkpoints become reference
  artifacts (no bytes uploaded).

Everything degrades to a no-op when wandb is not installed, not configured
(``WANDB_API_KEY`` unset and not offline), or explicitly disabled -- local runs
and unit tests never need network. The table/metric shaping lives in pure
helpers so it stays unit-testable without wandb.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - exercised only where wandb is installed
    import wandb
except Exception:  # pragma: no cover - control-plane venv has no wandb
    wandb = None  # type: ignore[assignment]

WANDB_PROJECT_ENV = "SLIME_WANDB_PROJECT"
WANDB_GROUP_ENV = "SLIME_WANDB_GROUP"
WANDB_ENTITY_ENV = "WANDB_ENTITY"

EVAL_PREFIX = "eval"
HEALTH_PREFIX = "rollout_health"
UPLIFT_PREFIX = "uplift"

#: numeric keys of ``aggregate_eval_records`` output logged as ``eval/<key>``.
EVAL_SUMMARY_METRIC_KEYS = (
    "task_count",
    "sample_count",
    "samples_per_task_mean",
    "pass_rate",
    "correct_and_faster_rate",
    "missing_runtime_rate",
    "compile_error_rate",
    "sanitizer_error_rate",
    "timeout_rate",
    "invalid_format_rate",
    "mean_best_reward",
    "mean_sample_reward",
    "mean_correct_faster_speedup",
)

RECORDS_TABLE_COLUMNS = (
    "task_id",
    "problem_id",
    "reason",
    "abort_reason",
    "reward",
    "all_tests_pass",
    "tests_passed",
    "tests_total",
    "compile_error",
    "timeout",
    "format_valid",
    "runtime_speedup",
    "agent_steps",
    "response_snippet",
)

RESPONSE_SNIPPET_CHARS = 300


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)[:120]


def wandb_enabled() -> bool:
    """True when logging can work: wandb importable, not disabled, creds or offline."""

    if wandb is None:
        return False
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    if mode == "disabled":
        return False
    if mode == "offline":
        return True
    return bool(os.environ.get("WANDB_API_KEY"))


def init_run(
    *,
    run_id: str,
    stage: str,
    job_type: str,
    project: str | None = None,
    group: str | None = None,
    entity: str | None = None,
    config: dict[str, Any] | None = None,
):
    """Open (or resume) the deterministic per-stage run ``<run_id>-<stage>``.

    Resuming the same id the lane exports via ``WANDB_RUN_ID`` means the offline
    scorer decorates the stage's existing run instead of creating a husk. Returns
    None whenever W&B is unavailable -- callers treat the run as optional.
    """

    project = project or os.environ.get(WANDB_PROJECT_ENV) or ""
    if not project or not wandb_enabled():
        return None
    try:
        run = wandb.init(
            project=project,
            entity=entity or os.environ.get(WANDB_ENTITY_ENV) or None,
            id=_safe_id(f"{run_id}-{stage}"),
            name=f"{run_id}-{stage}",
            group=group or os.environ.get(WANDB_GROUP_ENV) or run_id,
            job_type=job_type,
            resume="allow",
            reinit=True,
            settings=wandb.Settings(silent=True),
        )
    except Exception as exc:  # pragma: no cover - network/auth failures must not kill scoring
        print(f"wandb_report: init_run failed ({exc!r}); continuing without W&B", flush=True)
        return None
    if config:
        try:
            run.config.update(config, allow_val_change=True)
        except Exception:  # pragma: no cover
            pass
    return run


def attach_shared_run():
    """The already-attached shared-mode run (SLIME rollout actor), or None."""

    if wandb is None:
        return None
    return getattr(wandb, "run", None)


def finish_run(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- eval


def eval_metrics_from_summary(summary: dict[str, Any], *, prefix: str = EVAL_PREFIX) -> dict[str, float]:
    """Pure mapping: aggregate_eval_records output -> flat numeric metrics."""

    metrics: dict[str, float] = {}
    for key in EVAL_SUMMARY_METRIC_KEYS:
        value = summary.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[f"{prefix}/{key}"] = float(value)
    return metrics


def log_eval_summary(run, summary: dict[str, Any], *, prefix: str = EVAL_PREFIX) -> dict[str, float]:
    """Log eval outcome rates as metrics + summary mirror; returns what was logged."""

    metrics = eval_metrics_from_summary(summary, prefix=prefix)
    if run is None or not metrics:
        return metrics
    run.log(metrics)
    for key, value in metrics.items():
        run.summary[key] = value
    return metrics


def records_table_rows(records: list[dict[str, Any]], *, max_rows: int = 512) -> list[list[Any]]:
    """Pure per-sample forensics rows matching :data:`RECORDS_TABLE_COLUMNS`."""

    rows: list[list[Any]] = []
    for record in records[:max_rows]:
        response = str(record.get("response") or "")
        rows.append(
            [
                record.get("task_id"),
                record.get("problem_id"),
                record.get("reason"),
                record.get("abort_reason"),
                _as_float(record.get("reward")),
                bool(record.get("all_tests_pass")),
                record.get("tests_passed"),
                record.get("tests_total"),
                bool(record.get("compile_error")),
                bool(record.get("timeout")),
                bool(record.get("format_valid")),
                _as_float(record.get("runtime_speedup")),
                record.get("agent_steps"),
                response[-RESPONSE_SNIPPET_CHARS:],
            ]
        )
    return rows


def log_records_table(run, records: list[dict[str, Any]], *, prefix: str = EVAL_PREFIX, max_rows: int = 512) -> int:
    """Per-sample table + reward/speedup histograms. Returns row count."""

    rows = records_table_rows(records, max_rows=max_rows)
    if run is None or wandb is None:
        return len(rows)
    payload: dict[str, Any] = {f"{prefix}/records": wandb.Table(columns=list(RECORDS_TABLE_COLUMNS), data=rows)}
    rewards = [_as_float(record.get("reward")) for record in records]
    rewards = [value for value in rewards if value is not None]
    if rewards:
        payload[f"{prefix}/reward_hist"] = wandb.Histogram(rewards)
    speedups = [_as_float(record.get("runtime_speedup")) for record in records]
    speedups = [value for value in speedups if value is not None and value > 0]
    if speedups:
        payload[f"{prefix}/speedup_hist"] = wandb.Histogram(speedups)
    run.log(payload)
    return len(rows)


def classify_failure(record: dict[str, Any]) -> str | None:
    """Categorical infra-failure reason, or None for a scoreable sample.

    Test failures are quality outcomes, not aborts; they stay None here and are
    covered by the rate metrics instead.
    """

    abort_reason = record.get("abort_reason")
    if abort_reason:
        return str(abort_reason)
    reason = str(record.get("reason") or "")
    if reason.startswith("exception:") or record.get("exception"):
        return reason or "exception"
    if reason in {"invalid_format", "missing_task_path", "reward_exception", "task_load_error"}:
        return reason
    return None


def abort_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        failure = classify_failure(record)
        if failure:
            counts[failure] += 1
    return dict(counts)


def log_abort_distribution(run, records: list[dict[str, Any]], *, prefix: str = EVAL_PREFIX) -> dict[str, int]:
    """Failure-reason counts as metrics + one table; returns the counts."""

    counts = abort_counts(records)
    total = len(records)
    if run is None:
        return counts
    metrics: dict[str, Any] = {f"{prefix}/abort_rate": (sum(counts.values()) / total) if total else 0.0}
    for reason, count in sorted(counts.items()):
        metrics[f"{prefix}/abort/{_safe_id(reason)}"] = count
    if wandb is not None:
        table_rows = [[reason, count, count / total if total else 0.0] for reason, count in sorted(counts.items())]
        metrics[f"{prefix}/abort_reasons"] = wandb.Table(columns=["reason", "count", "fraction"], data=table_rows)
    run.log(metrics)
    run.summary[f"{prefix}/abort_rate"] = metrics[f"{prefix}/abort_rate"]
    return counts


# -------------------------------------------------------------------------- uplift


def uplift_rows(comparison: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for summary in comparison.get("summaries", []):
        rows.append(
            [
                summary.get("label"),
                _as_float(summary.get("pass_rate")),
                _as_float(summary.get("correct_and_faster_rate")),
                _as_float(summary.get("mean_best_reward")),
                _as_float(summary.get("mean_correct_faster_speedup")),
            ]
        )
    return rows


def log_uplift(run, comparison: dict[str, Any], *, prefix: str = UPLIFT_PREFIX) -> list[list[Any]]:
    """Cross-stage comparison table + verdict summary keys on the given run."""

    rows = uplift_rows(comparison)
    if run is None:
        return rows
    if wandb is not None and rows:
        run.log(
            {
                f"{prefix}/comparison": wandb.Table(
                    columns=["label", "pass_rate", "correct_and_faster_rate", "mean_best_reward", "mean_speedup"],
                    data=rows,
                )
            }
        )
    run.summary[f"{prefix}/best_correct_and_faster"] = comparison.get("best_correct_and_faster")
    run.summary[f"{prefix}/best_mean_reward"] = comparison.get("best_mean_reward")
    gate = comparison.get("uplift_gate")
    if isinstance(gate, dict):
        for key, value in gate.items():
            if isinstance(value, (bool, int, float, str)) or value is None:
                run.summary[f"{prefix}/gate_{key}"] = value
    return rows


# ----------------------------------------------------------------- artifacts/alerts


def log_artifact(run, path: str | Path, *, name: str, artifact_type: str) -> bool:
    if run is None or wandb is None:
        return False
    file_path = Path(path)
    if not file_path.exists():
        return False
    try:
        artifact = wandb.Artifact(_safe_id(name), type=artifact_type)
        artifact.add_file(str(file_path))
        run.log_artifact(artifact)
        return True
    except Exception as exc:  # pragma: no cover - artifact upload is best-effort
        print(f"wandb_report: log_artifact({name}) failed ({exc!r})", flush=True)
        return False


def log_reference_artifact(run, uri: str, *, name: str, artifact_type: str) -> bool:
    """Track a remote location (e.g. gs://... checkpoints) by reference, no bytes."""

    if run is None or wandb is None:
        return False
    try:
        artifact = wandb.Artifact(_safe_id(name), type=artifact_type)
        artifact.add_reference(uri, checksum=False)
        run.log_artifact(artifact)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"wandb_report: log_reference_artifact({name}) failed ({exc!r})", flush=True)
        return False


def alert(run, *, title: str, text: str, level: str = "WARN") -> bool:
    """Fire a W&B alert (email/Slack per user settings); best-effort."""

    if run is None:
        return False
    try:
        alert_level: Any = level
        if wandb is not None:
            try:
                alert_level = getattr(wandb.AlertLevel, level.upper())
            except Exception:
                alert_level = level
        run.alert(title=title, text=text, level=alert_level)
        return True
    except Exception:  # pragma: no cover - alerts must never break the run
        return False


# --------------------------------------------------------------------- live health


class RolloutHealth:
    """Windowed rollout-health aggregator fed by the agentic generate hook.

    ``add`` collects one dict per finished sample; every ``window`` samples the
    aggregate flushes as ``rollout_health/*`` metrics through ``log_fn`` (default:
    ``wandb.log`` on the shared-mode run the rollout actor already holds -- see
    module docstring). The window is sized to one GRPO rollout step
    (``rollout_batch_size * n_samples_per_prompt``) so the panels line up with
    ``rollout/*``. All computation is pure; without an attached run it is inert.

    ``zero_variance_group_fraction`` is the GRPO heartbeat: the fraction of
    prompt-groups (>= 2 samples, keyed by task id) whose rewards are identical --
    those groups contribute zero advantage and therefore zero learning signal.
    """

    def __init__(self, *, window: int, log_fn: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.window = max(1, int(window))
        self._log_fn = log_fn
        self._rows: list[dict[str, Any]] = []
        self._step = 0

    def add(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Record one sample outcome; returns the flushed metrics dict if any."""

        self._rows.append(dict(row))
        if len(self._rows) < self.window:
            return None
        return self.flush()

    def flush(self) -> dict[str, Any] | None:
        if not self._rows:
            return None
        rows, self._rows = self._rows, []
        metrics = self._metrics_for(rows)
        self._emit(metrics, rows)
        return metrics

    def _metrics_for(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        rewards = [_as_float(row.get("reward")) or 0.0 for row in rows]
        aborted = [row for row in rows if row.get("abort_reason")]
        steps = [_as_float(row.get("agent_steps")) for row in rows]
        steps = [value for value in steps if value is not None]
        wall_times = [_as_float(row.get("wall_time_s")) for row in rows]
        wall_times = [value for value in wall_times if value is not None]

        metrics: dict[str, Any] = {
            f"{HEALTH_PREFIX}/step": self._step,
            f"{HEALTH_PREFIX}/samples": total,
            f"{HEALTH_PREFIX}/reward_mean": _mean(rewards),
            f"{HEALTH_PREFIX}/reward_std": _std(rewards),
            f"{HEALTH_PREFIX}/abort_rate": len(aborted) / total,
            f"{HEALTH_PREFIX}/format_valid_rate": _rate(rows, "format_valid"),
            f"{HEALTH_PREFIX}/all_tests_pass_rate": _rate(rows, "all_tests_pass"),
            f"{HEALTH_PREFIX}/compile_error_rate": _rate(rows, "compile_error"),
            f"{HEALTH_PREFIX}/timeout_rate": _rate(rows, "timeout"),
        }
        if steps:
            metrics[f"{HEALTH_PREFIX}/agent_steps_mean"] = _mean(steps)
        if wall_times:
            metrics[f"{HEALTH_PREFIX}/wall_time_mean_s"] = _mean(wall_times)
        for reason, count in sorted(Counter(str(row.get("abort_reason")) for row in aborted).items()):
            metrics[f"{HEALTH_PREFIX}/abort/{_safe_id(reason)}"] = count

        groups: dict[str, list[float]] = defaultdict(list)
        for row, reward in zip(rows, rewards):
            task_id = row.get("task_id")
            if task_id:
                groups[str(task_id)].append(reward)
        scored_groups = [values for values in groups.values() if len(values) >= 2]
        if scored_groups:
            zero_variance = sum(1 for values in scored_groups if max(values) - min(values) < 1e-9)
            metrics[f"{HEALTH_PREFIX}/zero_variance_group_fraction"] = zero_variance / len(scored_groups)
            metrics[f"{HEALTH_PREFIX}/groups_evaluated"] = len(scored_groups)
        self._step += 1
        return metrics

    def _emit(self, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        log_fn = self._log_fn
        if log_fn is None:
            run = attach_shared_run()
            if run is None or wandb is None:
                return
            payload = dict(metrics)
            rewards = [_as_float(row.get("reward")) or 0.0 for row in rows]
            payload[f"{HEALTH_PREFIX}/reward_hist"] = wandb.Histogram(rewards)
            steps = [_as_float(row.get("agent_steps")) for row in rows]
            steps = [value for value in steps if value is not None]
            if steps:
                payload[f"{HEALTH_PREFIX}/agent_steps_hist"] = wandb.Histogram(steps)
            try:
                wandb.log(payload)
            except Exception:  # pragma: no cover - health logging is best-effort
                pass
            return
        log_fn(metrics)


def define_health_metrics() -> None:
    """Give rollout_health/* its own step axis on the attached shared run."""

    if wandb is None or attach_shared_run() is None:
        return
    try:
        wandb.define_metric(f"{HEALTH_PREFIX}/step")
        wandb.define_metric(f"{HEALTH_PREFIX}/*", step_metric=f"{HEALTH_PREFIX}/step")
    except Exception:  # pragma: no cover
        pass


# ----------------------------------------------------------------------- workspace

# NOTE: wandb-workspaces' name validator ("no emoji") rejects "C++".
WORKSPACE_NAME = "w8-biayn cpp RL observability"

#: Curated saved-view layout: the default W&B workspace is an unordered dump of
#: every metric key; this template puts the drill-path panels where people look.
#: Panels referencing keys a run has not logged simply render empty.
WORKSPACE_SECTIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "Uplift & Eval Comparison",
        "panels": [
            {"title": "Held-out pass rate (base vs sft vs grpo)", "x": None, "y": ["eval/pass_rate"]},
            {"title": "Correct+faster rate", "x": None, "y": ["eval/correct_and_faster_rate"]},
            {"title": "Mean winner speedup", "x": None, "y": ["eval/mean_correct_faster_speedup"]},
            {"title": "Eval abort rate", "x": None, "y": ["eval/abort_rate"]},
        ],
    },
    {
        "name": "Rollout Health (live)",
        "panels": [
            {
                "title": "Zero-variance group fraction (GRPO heartbeat)",
                "x": f"{HEALTH_PREFIX}/step",
                "y": [f"{HEALTH_PREFIX}/zero_variance_group_fraction"],
            },
            {"title": "Abort rate", "x": f"{HEALTH_PREFIX}/step", "y": [f"{HEALTH_PREFIX}/abort_rate"]},
            {
                "title": "Reward mean / std",
                "x": f"{HEALTH_PREFIX}/step",
                "y": [f"{HEALTH_PREFIX}/reward_mean", f"{HEALTH_PREFIX}/reward_std"],
            },
            {
                "title": "Correctness gates",
                "x": f"{HEALTH_PREFIX}/step",
                "y": [
                    f"{HEALTH_PREFIX}/format_valid_rate",
                    f"{HEALTH_PREFIX}/all_tests_pass_rate",
                    f"{HEALTH_PREFIX}/compile_error_rate",
                    f"{HEALTH_PREFIX}/timeout_rate",
                ],
            },
            {
                "title": "Agent steps / episode wall time",
                "x": f"{HEALTH_PREFIX}/step",
                "y": [f"{HEALTH_PREFIX}/agent_steps_mean", f"{HEALTH_PREFIX}/wall_time_mean_s"],
            },
        ],
    },
    {
        "name": "Training Dynamics",
        "panels": [
            {"title": "KL (NaN watch)", "x": "train/step", "y": ["train/ppo_kl", "train/kl_loss"]},
            {"title": "Clip fraction", "x": "train/step", "y": ["train/pg_clipfrac"]},
            {"title": "Grad norm", "x": "train/step", "y": ["train/grad_norm"]},
            {
                "title": "Rollout vs train logprob drift (engine mismatch)",
                "x": "train/step",
                "y": ["train/train_rollout_logprob_abs_diff"],
            },
            {"title": "Rollout reward", "x": "rollout/step", "y": ["rollout/rewards"]},
            {"title": "Response length", "x": "rollout/step", "y": ["rollout/response_length"]},
            {
                "title": "Pass@k buckets",
                "x": "rollout/step",
                "y": ["passrate/pass@1", "passrate/pass@2", "passrate/pass@4", "passrate/pass@8"],
            },
        ],
    },
    {
        "name": "Pipeline",
        "panels": [
            {"title": "Launch progress (elapsed seconds)", "x": "pipeline/step", "y": ["pipeline/elapsed_seconds"]},
        ],
    },
)


def build_workspace_spec(*, project: str, entity: str = "") -> dict[str, Any]:
    """Pure workspace layout spec (unit-testable without wandb-workspaces)."""

    return {
        "project": project,
        "entity": entity,
        "name": WORKSPACE_NAME,
        "sections": [
            {"name": section["name"], "panels": [dict(panel) for panel in section["panels"]]}
            for section in WORKSPACE_SECTIONS
        ],
    }


def push_workspace(*, project: str, entity: str = "") -> str:
    """Create/update the curated saved view on W&B; returns its URL."""

    spec = build_workspace_spec(project=project, entity=entity)
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:  # pragma: no cover - depends on the cloud extra
        raise RuntimeError(
            "wandb-workspaces is required for `w8-biayn wandb workspace`; "
            "run through the cloud extra: uv run --extra cloud w8-biayn wandb workspace"
        ) from exc
    if not entity:
        import wandb

        entity = wandb.Api().default_entity
    sections = []
    for section in spec["sections"]:
        panels = []
        for panel in section["panels"]:
            kwargs: dict[str, Any] = {"title": panel["title"], "y": list(panel["y"])}
            if panel.get("x"):
                kwargs["x"] = panel["x"]
            panels.append(wr.LinePlot(**kwargs))
        sections.append(ws.Section(name=section["name"], panels=panels, is_open=True))
    workspace = ws.Workspace(entity=entity, project=spec["project"], name=spec["name"], sections=sections)
    workspace.save()
    return str(workspace.url)


# ------------------------------------------------------------------------ helpers


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return (sum((value - mu) ** 2 for value in values) / len(values)) ** 0.5


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(row.get(key))) / len(rows)
