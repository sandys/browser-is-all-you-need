from __future__ import annotations

import sqlite3

from w8_biayn.mlflow_metrics import read_mlflow_metrics


def test_read_mlflow_metrics_latest_and_series(tmp_path):
    db = tmp_path / "mlflow.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table metrics (key text, value real, timestamp integer, step integer)")
        conn.executemany(
            "insert into metrics (key, value, timestamp, step) values (?, ?, ?, ?)",
            [
                ("policy/policy_entropy", 0.5, 10, 1),
                ("policy/policy_entropy", 0.25, 20, 2),
                ("loss/avg_final_rewards", -0.2, 11, 1),
                ("loss/avg_final_rewards", 0.8, 21, 2),
                ("unused/key", 42.0, 22, 2),
            ],
        )

    payload = read_mlflow_metrics(db, metric_keys=["policy/policy_entropy", "loss/avg_final_rewards"], last=2)

    assert payload["available"] is True
    assert payload["latest_step"] == 2
    assert payload["latest"]["policy/policy_entropy"]["value"] == 0.25
    assert payload["series"]["policy/policy_entropy"] == [
        {"step": 1, "value": 0.5, "timestamp_ms": 10},
        {"step": 2, "value": 0.25, "timestamp_ms": 20},
    ]
    assert "unused/key" in payload["available_keys"]
    assert "unused/key" not in payload["selected_keys"]
    assert payload["tracking_state"] == "metrics_available"
    assert payload["metric_row_count"] == 5


def test_read_mlflow_metrics_missing_db_is_unavailable(tmp_path):
    payload = read_mlflow_metrics(tmp_path / "missing.db")

    assert payload["available"] is False
    assert payload["reason"] == "db_missing"


def test_read_mlflow_metrics_reports_active_run_before_first_scalar(tmp_path):
    db = tmp_path / "mlflow.db"
    run_uuid = "run-123"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "create table experiments (experiment_id integer, name text, lifecycle_stage text, artifact_location text)"
        )
        conn.execute(
            """
            create table runs (
                run_uuid text, name text, status text, start_time integer, end_time integer,
                experiment_id integer, lifecycle_stage text, artifact_uri text
            )
            """
        )
        conn.execute("create table params (key text, value text, run_uuid text)")
        conn.execute("create table tags (key text, value text, run_uuid text)")
        conn.execute("create table metrics (key text, value real, timestamp integer, step integer, run_uuid text)")
        conn.execute(
            "insert into experiments values (?, ?, ?, ?)",
            (1, "skyrl", "active", "/artifacts/tracking/mlflow/artifacts/1"),
        )
        conn.execute(
            "insert into runs values (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_uuid, "test_run", "RUNNING", 1234, None, 1, "active", "/artifacts/run"),
        )
        conn.executemany(
            "insert into params values (?, ?, ?)",
            [
                ("trainer/algorithm/use_kl_loss", "True", run_uuid),
                ("trainer/algorithm/use_entropy_loss", "True", run_uuid),
                ("trainer/placement/policy_num_nodes", "2", run_uuid),
                ("unselected", "kept-out", run_uuid),
            ],
        )
        conn.execute("insert into tags values (?, ?, ?)", ("mlflow.runName", "test_run", run_uuid))

    payload = read_mlflow_metrics(db)

    assert payload["available"] is True
    assert payload["tracking_state"] == "run_active_no_metrics"
    assert payload["latest_step"] is None
    assert payload["metric_count"] == 0
    assert payload["metric_row_count"] == 0
    assert payload["experiments"]["count"] == 1
    assert payload["run"]["available"] is True
    assert payload["run"]["run_uuid"] == run_uuid
    assert payload["run"]["status"] == "RUNNING"
    assert payload["params"]["count"] == 4
    assert payload["params"]["selected"] == {
        "trainer/algorithm/use_kl_loss": "True",
        "trainer/algorithm/use_entropy_loss": "True",
        "trainer/placement/policy_num_nodes": "2",
    }
    assert payload["tags"]["items"] == {"mlflow.runName": "test_run"}
