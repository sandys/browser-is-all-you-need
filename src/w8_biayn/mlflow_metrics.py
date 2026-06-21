"""Read headless MLflow tracking-server metrics for ops status."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


def read_mlflow_api(
    base_url: str,
    *,
    metric_keys: list[str] | tuple[str, ...] | None = None,
    last: int = 100,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Return latest values and short series from an MLflow Tracking Server API."""

    normalized_url = base_url.rstrip("/")
    keys = tuple(dict.fromkeys(metric_keys or DEFAULT_METRIC_KEYS))
    try:
        experiments = _api_experiments(normalized_url, timeout_s=timeout_s)
        experiment_ids = [str(item["experiment_id"]) for item in experiments if item.get("experiment_id") is not None]
        runs = _api_runs(normalized_url, experiment_ids=experiment_ids, timeout_s=timeout_s)
        active_run_raw = _active_run(runs)
        run_id = active_run_raw.get("run_uuid") if active_run_raw else None
        params = active_run_raw.get("_params", {}) if active_run_raw else {}
        tags = active_run_raw.get("_tags", {}) if active_run_raw else {}
        latest_from_run = active_run_raw.get("_latest", {}) if active_run_raw else {}
        public_runs = [_public_run_payload(run) for run in runs]
        active_run = _public_run_payload(active_run_raw) if active_run_raw else None
        available_keys = sorted(latest_from_run)
        selected_keys = [key for key in keys if key in latest_from_run] if keys else available_keys
        series = _api_metric_series(
            normalized_url,
            run_id=str(run_id) if run_id else None,
            keys=selected_keys,
            last=last,
            timeout_s=timeout_s,
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return _unavailable_api(normalized_url, f"api_error:{exc}")

    latest = {
        key: series[key][-1] if series.get(key) else latest_from_run[key]
        for key in selected_keys
        if key in latest_from_run or series.get(key)
    }
    latest_step = max((item["step"] for item in latest.values() if item.get("step") is not None), default=None)
    tracking_state = _tracking_state(
        run=active_run,
        experiments=experiments,
        available_keys=available_keys,
        metric_table_available=True,
    )
    return {
        "available": True,
        "reason": None,
        "backend": "mlflow_api",
        "tracking_state": tracking_state,
        "source": {
            "base_url": normalized_url,
        },
        "experiments": {
            "count": len(experiments),
            "items": experiments,
        },
        "runs": {
            "count": len(public_runs),
            "items": public_runs,
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
        "metric_row_count": sum(len(points) for points in series.values()),
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


def _unavailable_api(base_url: str, reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "backend": "mlflow_api",
        "tracking_state": reason,
        "source": {"base_url": base_url.rstrip("/")},
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


def _api_experiments(base_url: str, *, timeout_s: float) -> list[dict[str, Any]]:
    try:
        payload = _api_post_json(base_url, "/api/2.0/mlflow/experiments/search", {"max_results": 1000}, timeout_s)
        experiments = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
    except HTTPError as exc:
        if exc.code != 404:
            raise
        payload = _api_get_json(base_url, "/api/2.0/mlflow/experiments/list", {}, timeout_s)
        experiments = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
    return [_api_experiment_payload(item) for item in experiments if isinstance(item, dict)]


def _api_runs(base_url: str, *, experiment_ids: list[str], timeout_s: float) -> list[dict[str, Any]]:
    if not experiment_ids:
        return []
    payload = _api_post_json(
        base_url,
        "/api/2.0/mlflow/runs/search",
        {
            "experiment_ids": experiment_ids,
            "max_results": 1000,
            "order_by": ["attributes.start_time DESC"],
        },
        timeout_s,
    )
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    return [_api_run_payload(item) for item in runs if isinstance(item, dict)]


def _api_metric_series(
    base_url: str,
    *,
    run_id: str | None,
    keys: list[str],
    last: int,
    timeout_s: float,
) -> dict[str, list[dict[str, Any]]]:
    if not run_id or last <= 0:
        return {}
    series: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        payload = _api_get_json(
            base_url,
            "/api/2.0/mlflow/metrics/get-history",
            {"run_id": run_id, "metric_key": key},
            timeout_s,
        )
        points = payload.get("metrics") if isinstance(payload.get("metrics"), list) else []
        normalized = [_api_metric_payload(item) for item in points if isinstance(item, dict)]
        normalized.sort(key=lambda item: (item.get("step") or -1, item.get("timestamp_ms") or -1))
        if normalized:
            series[key] = normalized[-last:]
    return series


def _api_experiment_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": _coerce_value(item.get("experiment_id")),
        "name": item.get("name"),
        "lifecycle_stage": item.get("lifecycle_stage"),
        "artifact_location": item.get("artifact_location"),
        "creation_time": _to_int_or_none(item.get("creation_time")),
        "last_update_time": _to_int_or_none(item.get("last_update_time")),
    }


def _api_run_payload(item: dict[str, Any]) -> dict[str, Any]:
    info = item.get("info") if isinstance(item.get("info"), dict) else {}
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    run_id = info.get("run_id") or info.get("run_uuid")
    metrics = _api_keyed_metrics(data.get("metrics") if isinstance(data.get("metrics"), list) else [])
    payload = {
        "run_uuid": run_id,
        "name": info.get("run_name") or _api_tag_value(data, "mlflow.runName"),
        "status": info.get("status"),
        "start_time": _to_int_or_none(info.get("start_time")),
        "end_time": _to_int_or_none(info.get("end_time")),
        "experiment_id": _coerce_value(info.get("experiment_id")),
        "lifecycle_stage": info.get("lifecycle_stage"),
        "artifact_uri": info.get("artifact_uri"),
        "_params": _api_key_values(data.get("params") if isinstance(data.get("params"), list) else []),
        "_tags": _api_key_values(data.get("tags") if isinstance(data.get("tags"), list) else []),
        "_latest": metrics,
    }
    return payload


def _public_run_payload(item: dict[str, Any] | None) -> dict[str, Any]:
    if item is None:
        return {}
    return {key: value for key, value in item.items() if not str(key).startswith("_")}


def _api_key_values(items: list[Any]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict) and item.get("key") is not None:
            payload[str(item["key"])] = str(item.get("value"))
    return payload


def _api_keyed_metrics(items: list[Any]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get("key") is not None:
            metric = _api_metric_payload(item)
            key = str(item["key"])
            existing = payload.get(key)
            if existing is None or (
                (metric.get("step") or -1, metric.get("timestamp_ms") or -1)
                >= (existing.get("step") or -1, existing.get("timestamp_ms") or -1)
            ):
                payload[key] = metric
    return payload


def _api_metric_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": _to_int_or_none(item.get("step")),
        "value": float(item.get("value")),
        "timestamp_ms": _to_int_or_none(item.get("timestamp")),
    }


def _api_tag_value(data: dict[str, Any], key: str) -> str | None:
    tags = data.get("tags") if isinstance(data.get("tags"), list) else []
    for item in tags:
        if isinstance(item, dict) and item.get("key") == key:
            return str(item.get("value"))
    return None


def _api_post_json(base_url: str, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _read_json_response(request, timeout_s)


def _api_get_json(base_url: str, path: str, query: dict[str, str], timeout_s: float) -> dict[str, Any]:
    suffix = "?" + urlencode(query) if query else ""
    request = Request(base_url.rstrip("/") + path + suffix, method="GET")
    return _read_json_response(request, timeout_s)


def _read_json_response(request: Request, timeout_s: float) -> dict[str, Any]:
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read()
    if not raw:
        return {}
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("MLflow API returned a non-object JSON response")
    return payload


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
