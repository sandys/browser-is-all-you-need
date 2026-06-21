"""Read headless MLflow tracking-server metrics for ops status."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_METRIC_KEYS = (
    "eval/all/avg_score",
    "eval/all/pass_at_1",
    "eval/all/pass_at_4",
    "eval/all/pass_at_8",
    "eval/all/mean_positive_reward",
    "loss/avg_final_rewards",
    "reward/avg_pass_at_1",
    "reward/avg_pass_at_4",
    "reward/avg_pass_at_8",
    "reward/avg_pass_at_16",
    "reward/avg_pass_at_32",
    "reward/avg_raw_reward",
    "reward/mean_positive_reward",
    "policy/policy_kl",
    "loss/avg_kl",
    "loss/avg_kl_max",
    "policy/policy_entropy",
    "policy/entropy_loss:sum",
    "policy/response_length",
    "policy/num_tokens:sum",
    "generate/avg_assistant_tokens",
    "generate/std_assistant_tokens",
    "generate/min_assistant_tokens",
    "generate/max_assistant_tokens",
    "policy/grad_norm",
    "policy/policy_loss",
    "policy/pg_loss:sum",
    "policy/total_loss:sum",
    "loss/avg_raw_advantages_abs",
    "policy/loss_metrics/clip_ratio",
    "policy/policy_lr",
    "policy/policy_lr:last",
    "policy/rollout_train_logprobs_abs_diff_mean",
    "policy/rollout_train_logprobs_abs_diff_std",
    "generate/avg_tokens_zero_rewards",
    "generate/avg_tokens_non_zero_rewards",
    "timing/step",
    "timing/generate",
    "timing/policy_train",
    "timing/fwd_logprobs_values_reward",
    "timing/sync_weights",
    "timing/save_checkpoint",
    "timing/save_hf_model",
    "w8/reward_group_variance_mean",
    "w8/reward_group_variance_max",
    "w8/zero_variance_group_fraction",
    "w8/zero_advantage_token_fraction",
)

IMPORTANT_PARAM_KEYS = (
    "data/train_data/0",
    "data/val_data/0",
    "environment/env_class",
    "environment/skyrl_gym/max_env_workers",
    "generator/n_samples_per_prompt",
    "generator/eval_n_samples_per_prompt",
    "generator/sampling_params/max_generate_length",
    "generator/eval_sampling_params/max_generate_length",
    "generator/inference_engine/num_engines",
    "generator/inference_engine/gpu_memory_utilization",
    "trainer/algorithm/use_kl_loss",
    "trainer/algorithm/kl_loss_coef",
    "trainer/algorithm/use_entropy_loss",
    "trainer/algorithm/entropy_loss_coef",
    "trainer/algorithm/use_kl_in_reward",
    "trainer/ckpt_interval",
    "trainer/hf_save_interval",
    "trainer/max_ckpts_to_keep",
    "trainer/micro_train_batch_size_per_gpu",
    "trainer/train_batch_size",
    "trainer/epochs",
    "trainer/update_epochs_per_batch",
    "trainer/placement/policy_num_nodes",
    "trainer/placement/policy_num_gpus_per_node",
    "trainer/policy/fsdp_config/fsdp_size",
    "trainer/ref/fsdp_config/fsdp_size",
    "trainer/policy/model/path",
)


def read_mlflow_metrics(
    db_path: str | Path,
    *,
    metric_keys: list[str] | tuple[str, ...] | None = None,
    last: int = 100,
) -> dict[str, Any]:
    """Return latest values and short series from an MLflow SQLite backend store."""

    path = Path(db_path)
    if not path.exists():
        return _unavailable(path, "db_missing")
    if path.stat().st_size == 0:
        return _unavailable(path, "db_empty")

    keys = tuple(dict.fromkeys(metric_keys or DEFAULT_METRIC_KEYS))
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            experiments = _experiments(conn)
            runs = _runs(conn)
            active_run = _active_run(runs)
            run_uuid = active_run.get("run_uuid") if active_run else None
            params = _key_values(conn, "params", run_uuid=run_uuid)
            tags = _key_values(conn, "tags", run_uuid=run_uuid)
            metric_table_available = _table_exists(conn, "metrics")
            available_keys = _available_keys(conn, run_uuid=run_uuid) if metric_table_available else []
            metric_row_count = _count_rows(conn, "metrics", run_uuid=run_uuid)
            selected_keys = [key for key in keys if key in available_keys] if keys else available_keys
            latest = _latest_metrics(conn, selected_keys, run_uuid=run_uuid)
            series = _metric_series(conn, selected_keys, last=last, run_uuid=run_uuid)
    except sqlite3.Error as exc:
        return _unavailable(path, f"sqlite_error:{exc}")

    latest_step = max((item["step"] for item in latest.values() if item.get("step") is not None), default=None)
    tracking_state = _tracking_state(
        run=active_run,
        experiments=experiments,
        available_keys=available_keys,
        metric_table_available=metric_table_available,
    )
    return {
        "available": True,
        "reason": None,
        "backend": "mlflow",
        "tracking_state": tracking_state,
        "source": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        },
        "experiments": {
            "count": len(experiments),
            "items": experiments,
        },
        "runs": {
            "count": len(runs),
            "items": runs,
        },
        "run": {
            "available": active_run is not None,
            **(active_run or {}),
        },
        "params": {
            "count": len(params),
            "available_keys": list(params),
            "selected": {key: params[key] for key in IMPORTANT_PARAM_KEYS if key in params},
        },
        "tags": {
            "count": len(tags),
            "items": tags,
        },
        "latest_step": latest_step,
        "metric_row_count": metric_row_count,
        "metric_count": len(available_keys),
        "available_keys": available_keys,
        "selected_keys": selected_keys,
        "latest": latest,
        "series": series,
    }


def _unavailable(path: Path, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "backend": "mlflow",
        "tracking_state": reason,
        "source": {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
        },
        "experiments": {"count": 0, "items": []},
        "runs": {"count": 0, "items": []},
        "run": {"available": False},
        "params": {"count": 0, "available_keys": [], "selected": {}},
        "tags": {"count": 0, "items": {}},
        "latest_step": None,
        "metric_row_count": 0,
        "metric_count": 0,
        "available_keys": [],
        "selected_keys": [],
        "latest": {},
        "series": {},
    }


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type = 'table' and name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {str(row["name"]) for row in conn.execute(f"pragma table_info({table})").fetchall()}


def _count_rows(conn: sqlite3.Connection, table: str, *, run_uuid: str | None = None) -> int:
    if not _table_exists(conn, table):
        return 0
    columns = _table_columns(conn, table)
    if run_uuid and "run_uuid" in columns:
        row = conn.execute(f"select count(*) as count from {table} where run_uuid = ?", (run_uuid,)).fetchone()
    else:
        row = conn.execute(f"select count(*) as count from {table}").fetchone()
    return int(row["count"]) if row is not None else 0


def _experiments(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _select_dicts(
        conn,
        "experiments",
        (
            "experiment_id",
            "name",
            "lifecycle_stage",
            "artifact_location",
            "creation_time",
            "last_update_time",
        ),
        order_by="experiment_id",
    )


def _runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    columns = _table_columns(conn, "runs")
    order_by = "start_time desc" if "start_time" in columns else None
    return _select_dicts(
        conn,
        "runs",
        (
            "run_uuid",
            "name",
            "status",
            "start_time",
            "end_time",
            "experiment_id",
            "lifecycle_stage",
            "artifact_uri",
        ),
        order_by=order_by,
        limit=20,
    )


def _active_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for run in runs:
        if str(run.get("status", "")).upper() == "RUNNING":
            return run
    return runs[0] if runs else None


def _select_dicts(
    conn: sqlite3.Connection,
    table: str,
    preferred_columns: tuple[str, ...],
    *,
    order_by: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    columns = [column for column in preferred_columns if column in _table_columns(conn, table)]
    if not columns:
        return []
    query = f"select {', '.join(columns)} from {table}"
    if order_by:
        query += f" order by {order_by}"
    if limit is not None:
        query += " limit ?"
        rows = conn.execute(query, (limit,)).fetchall()
    else:
        rows = conn.execute(query).fetchall()
    return [{key: _coerce_value(row[key]) for key in row.keys()} for row in rows]


def _key_values(conn: sqlite3.Connection, table: str, *, run_uuid: str | None = None) -> dict[str, str]:
    columns = _table_columns(conn, table)
    if not {"key", "value"}.issubset(columns):
        return {}
    query = f"select key, value from {table}"
    params: tuple[str, ...] = ()
    if run_uuid and "run_uuid" in columns:
        query += " where run_uuid = ?"
        params = (run_uuid,)
    query += " order by key"
    rows = conn.execute(query, params).fetchall()
    return {str(row["key"]): str(row["value"]) for row in rows}


def _available_keys(conn: sqlite3.Connection, *, run_uuid: str | None = None) -> list[str]:
    where, params = _metric_run_filter(conn, run_uuid)
    rows = conn.execute(f"select distinct key from metrics{where} order by key", params).fetchall()
    return [str(row["key"]) for row in rows]


def _latest_metrics(
    conn: sqlite3.Connection, keys: list[str], *, run_uuid: str | None = None
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    run_filter, run_params = _metric_run_filter(conn, run_uuid, prefix="and")
    for key in keys:
        row = conn.execute(
            f"""
            select key, value, timestamp, step
            from metrics
            where key = ?
            {run_filter}
            order by step desc, timestamp desc
            limit 1
            """,
            (key, *run_params),
        ).fetchone()
        if row is not None:
            payload[key] = _row_payload(row)
    return payload


def _metric_series(
    conn: sqlite3.Connection,
    keys: list[str],
    *,
    last: int,
    run_uuid: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if last <= 0:
        return {}
    series: dict[str, list[dict[str, Any]]] = {}
    run_filter, run_params = _metric_run_filter(conn, run_uuid, prefix="and")
    for key in keys:
        rows = conn.execute(
            f"""
            select key, value, timestamp, step
            from metrics
            where key = ?
            {run_filter}
            order by step desc, timestamp desc
            limit ?
            """,
            (key, *run_params, last),
        ).fetchall()
        if rows:
            series[key] = [_row_payload(row) for row in reversed(rows)]
    return series


def _metric_run_filter(
    conn: sqlite3.Connection,
    run_uuid: str | None,
    *,
    prefix: str = "where",
) -> tuple[str, tuple[str, ...]]:
    if run_uuid and "run_uuid" in _table_columns(conn, "metrics"):
        return f" {prefix} run_uuid = ?", (run_uuid,)
    return "", ()


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    step = row["step"]
    timestamp = row["timestamp"]
    return {
        "step": int(step) if step is not None else None,
        "value": float(row["value"]),
        "timestamp_ms": int(timestamp) if timestamp is not None else None,
    }


def _tracking_state(
    *,
    run: dict[str, Any] | None,
    experiments: list[dict[str, Any]],
    available_keys: list[str],
    metric_table_available: bool,
) -> str:
    if available_keys:
        return "metrics_available"
    if run is not None:
        return "run_active_no_metrics" if str(run.get("status", "")).upper() == "RUNNING" else "run_finished_no_metrics"
    if experiments:
        return "experiments_no_runs"
    if metric_table_available:
        return "mlflow_schema_empty"
    return "metrics_table_missing"


def _coerce_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
