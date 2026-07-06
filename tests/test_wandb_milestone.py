"""Tests for the standalone pipeline-milestone script (no wandb/network needed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "wandb_milestone.py"

spec = importlib.util.spec_from_file_location("wandb_milestone", SCRIPT)
milestone = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(milestone)


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.summary: dict = {}
        self.defined: list = []

    def log(self, metrics: dict) -> None:
        self.logged.append(metrics)

    def define_metric(self, *args, **kwargs) -> None:
        self.defined.append((args, kwargs))


def test_append_milestone_measures_elapsed_from_first_entry(tmp_path) -> None:
    path = tmp_path / "pipeline_milestones.jsonl"

    first = milestone.append_milestone(path, event="host_preflight_done", unix=1000.0)
    second = milestone.append_milestone(path, event="model_download_finished", unix=1090.5)

    assert first["elapsed_s"] == 0.0 and first["step"] == 0
    assert second["elapsed_s"] == 90.5 and second["step"] == 1
    rows = milestone.read_milestones(path)
    assert [row["event"] for row in rows] == ["host_preflight_done", "model_download_finished"]
    # corrupt lines are skipped, not fatal
    path.write_text(path.read_text() + "not-json\n{}\n", encoding="utf-8")
    assert len(milestone.read_milestones(path)) == 2


def test_log_milestone_uses_step_axis_and_never_raw_unix() -> None:
    run = FakeRun()
    entry = {"event": "sft_started", "unix": 2000.0, "iso": "x", "elapsed_s": 120.0, "step": 3}

    metrics = milestone.log_milestone(run, entry)

    assert metrics == {"pipeline/step": 3, "pipeline/elapsed_seconds": 120.0}
    assert run.logged == [metrics]
    # the old malformed shape must be gone: no per-event metric keys, no unix values
    assert all("_at" not in key and "_unix" not in key for key in metrics)
    assert run.summary["pipeline/latest"] == "sft_started"
    assert run.summary["pipeline/sft_started_elapsed_s"] == 120.0
    assert run.summary["pipeline/first_ts"] == 1880.0
    assert ("pipeline/elapsed_seconds",) in [args for args, _ in run.defined]


def test_finalize_publishes_single_timeline_table(tmp_path) -> None:
    path = tmp_path / "m.jsonl"
    milestone.append_milestone(path, event="a", unix=10.0)
    milestone.append_milestone(path, event="b", unix=25.0)
    run = FakeRun()

    count = milestone.log_finalize(
        run,
        milestone.read_milestones(path),
        table_factory=lambda columns, data: {"columns": list(columns), "data": data},
    )

    assert count == 2
    table = run.logged[0]["pipeline/timeline"]
    assert table["columns"] == ["event", "iso_time", "elapsed_s"]
    assert [row[0] for row in table["data"]] == ["a", "b"]
    assert table["data"][1][2] == 15.0
    assert run.summary["pipeline/total_elapsed_s"] == 15.0
    assert run.summary["pipeline/event_count"] == 2
    # empty log: no table, no crash
    assert milestone.log_finalize(FakeRun(), [], table_factory=dict) == 0


def test_main_skips_wandb_but_still_appends_local_log(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    path = tmp_path / "m.jsonl"

    code = milestone.main(
        ["--project", "p", "--run-id", "r1", "--event", "weird event!", "--milestones-file", str(path)]
    )

    assert code == 0
    assert "wandb_milestone_skipped" in capsys.readouterr().err
    rows = milestone.read_milestones(path)
    assert len(rows) == 1
    assert rows[0]["event"] == "weird-event-"  # sanitized
    assert json.loads(path.read_text().splitlines()[0])["step"] == 0


def test_script_has_no_raw_unix_metric_logging() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "_at" not in text.replace("elapsed_s", "").replace("_attr", "")  # no pipeline/<event>_at metrics
    assert "run.log({f" not in text  # no per-event dynamic metric keys
