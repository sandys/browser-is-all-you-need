"""Log pipeline milestones to a shared per-launch W&B run.

Standalone by design: invoked as a path script from the remote host
(`uv run --with wandb python scripts/wandb_milestone.py ...`) and from
inside the SLIME container (wandb is preinstalled there), so it must not
import anything from the w8_biayn package.

Data shape (the fix for the old one-point unix-timestamp panels): every event
appends to a local ``pipeline_milestones.jsonl`` (launch evidence, works even
without W&B) and logs ONE metric -- ``pipeline/elapsed_seconds`` against the
``pipeline/step`` axis -- so the pipeline run renders a single monotonic
"how far along / how long" curve. Event names live in ``run.summary``
(``pipeline/latest``, ``pipeline/<event>_elapsed_s``) where they belong.
``--finalize`` turns the jsonl into a single ``pipeline/timeline`` wandb.Table
(event, iso_time, elapsed_s). The jsonl (first line's timestamp) is the source
of truth for elapsed math; host and container share it via the repo mount.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TIMELINE_COLUMNS = ("event", "iso_time", "elapsed_s")


def _default_milestones_file() -> Path:
    return Path(__file__).resolve().parents[1] / ".w8-biayn" / "pipeline_milestones.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True, help="Launch run id; milestones share the run <run-id>-pipeline.")
    parser.add_argument("--event", required=True, help="Milestone name, e.g. model_download_finished.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument(
        "--milestones-file",
        default=os.environ.get("W8_MILESTONES_FILE") or str(_default_milestones_file()),
        help="Shared per-launch event log (jsonl); elapsed times are measured from its first entry.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Also publish the whole event log as one pipeline/timeline wandb.Table.",
    )
    return parser.parse_args(argv)


def _wandb_safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)[:120]


def _iso(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat(timespec="seconds")


def read_milestones(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("unix"), (int, float)):
            rows.append(row)
    return rows


def append_milestone(path: Path, *, event: str, unix: float) -> dict:
    """Append one event; elapsed_s is measured from the file's first entry."""

    rows = read_milestones(path)
    first = float(rows[0]["unix"]) if rows else unix
    entry = {
        "event": event,
        "unix": unix,
        "iso": _iso(unix),
        "elapsed_s": round(unix - first, 3),
        "step": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def log_milestone(run, entry: dict) -> dict:
    """One point on the pipeline/elapsed_seconds curve + event names in summary."""

    try:
        run.define_metric("pipeline/step")
        run.define_metric("pipeline/elapsed_seconds", step_metric="pipeline/step")
    except Exception:  # noqa: BLE001 - older clients; axis then falls back to _step
        pass
    metrics = {"pipeline/step": entry["step"], "pipeline/elapsed_seconds": entry["elapsed_s"]}
    run.log(metrics)
    run.summary["pipeline/latest"] = entry["event"]
    run.summary[f"pipeline/{entry['event']}_elapsed_s"] = entry["elapsed_s"]
    if "pipeline/first_ts" not in run.summary:
        run.summary["pipeline/first_ts"] = entry["unix"] - entry["elapsed_s"]
    return metrics


def timeline_rows(rows: list[dict]) -> list[list[object]]:
    return [[row.get("event"), row.get("iso"), row.get("elapsed_s")] for row in rows]


def log_finalize(run, rows: list[dict], *, table_factory=None) -> int:
    """Publish the whole event log as one pipeline/timeline table."""

    if not rows:
        return 0
    if table_factory is None:  # pragma: no cover - real wandb path
        import wandb

        def table_factory(columns, data):  # noqa: ANN001
            return wandb.Table(columns=list(columns), data=data)

    run.log({"pipeline/timeline": table_factory(TIMELINE_COLUMNS, timeline_rows(rows))})
    run.summary["pipeline/total_elapsed_s"] = rows[-1].get("elapsed_s")
    run.summary["pipeline/event_count"] = len(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    event = _wandb_safe_id(args.event)
    milestones_path = Path(args.milestones_file)
    # Local evidence first: the jsonl is written even when W&B is unavailable.
    entry = append_milestone(milestones_path, event=event, unix=time.time())

    if not os.environ.get("WANDB_API_KEY"):
        print("wandb_milestone_skipped: WANDB_API_KEY not set", file=sys.stderr)
        return 0
    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=_wandb_safe_id(f"{args.run_id}-pipeline"),
        name=f"{args.run_id}-pipeline",
        group=args.run_id,
        job_type="pipeline",
        resume="allow",
        settings=wandb.Settings(silent=True),
    )
    log_milestone(run, entry)
    if args.finalize:
        log_finalize(run, read_milestones(milestones_path))
    run.finish()
    print(f"wandb_milestone_logged: {event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
