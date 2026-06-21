"""Read headless MLflow tracking-server metrics for ops status."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_METRIC_KEYS = (
    "eval/all/avg_score",
    "eval/all/pass_at_1",
    "eval/all/mean_positive_reward",
    "loss/avg_final_rewards",
    "reward/avg_pass_at_1",
    "reward/avg_pass_at_8",
    "reward/avg_raw_reward",
    "policy/policy_kl",
    "loss/avg_kl",
    "loss/avg_kl_max",
    "policy/policy_entropy",
    "policy/response_length",
    "generate/avg_assistant_tokens",
    "generate/std_assistant_tokens",
    "generate/min_assistant_tokens",
    "generate/max_assistant_tokens",
    "policy/grad_norm",
    "policy/policy_loss",
    "loss/avg_raw_advantages_abs",
    "policy/loss_metrics/clip_ratio",
    "policy/policy_lr",
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
            if not _table_exists(conn, "metrics"):
                return _unavailable(path, "metrics_table_missing")
            available_keys = _available_keys(conn)
            selected_keys = [key for key in keys if key in available_keys] if keys else available_keys
            latest = _latest_metrics(conn, selected_keys)
            series = _metric_series(conn, selected_keys, last=last)
    except sqlite3.Error as exc:
        return _unavailable(path, f"sqlite_error:{exc}")

    latest_step = max((item["step"] for item in latest.values() if item.get("step") is not None), default=None)
    return {
        "available": True,
        "reason": None,
        "backend": "mlflow",
        "source": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
        },
        "latest_step": latest_step,
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
        "source": {
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
        },
        "latest_step": None,
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


def _available_keys(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("select distinct key from metrics order by key").fetchall()
    return [str(row["key"]) for row in rows]


def _latest_metrics(conn: sqlite3.Connection, keys: list[str]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for key in keys:
        row = conn.execute(
            """
            select key, value, timestamp, step
            from metrics
            where key = ?
            order by step desc, timestamp desc
            limit 1
            """,
            (key,),
        ).fetchone()
        if row is not None:
            payload[key] = _row_payload(row)
    return payload


def _metric_series(conn: sqlite3.Connection, keys: list[str], *, last: int) -> dict[str, list[dict[str, Any]]]:
    if last <= 0:
        return {}
    series: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        rows = conn.execute(
            """
            select key, value, timestamp, step
            from metrics
            where key = ?
            order by step desc, timestamp desc
            limit ?
            """,
            (key, last),
        ).fetchall()
        if rows:
            series[key] = [_row_payload(row) for row in reversed(rows)]
    return series


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    step = row["step"]
    timestamp = row["timestamp"]
    return {
        "step": int(step) if step is not None else None,
        "value": float(row["value"]),
        "timestamp_ms": int(timestamp) if timestamp is not None else None,
    }
