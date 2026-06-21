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


def test_read_mlflow_metrics_missing_db_is_unavailable(tmp_path):
    payload = read_mlflow_metrics(tmp_path / "missing.db")

    assert payload["available"] is False
    assert payload["reason"] == "db_missing"
