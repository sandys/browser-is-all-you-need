"""Log pipeline milestones to a shared per-launch W&B run.

Standalone by design: invoked as a path script from the remote host
(`uv run --with wandb python src/w8_biayn/wandb_milestones.py ...`) and from
inside the SLIME container (wandb is preinstalled there), so it must not
import anything from the w8_biayn package.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True, help="Launch run id; milestones share the run <run-id>-pipeline.")
    parser.add_argument("--event", required=True, help="Milestone name, e.g. model_download_finished.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    return parser.parse_args(argv)


def _wandb_safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "-", value)[:120]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    now = time.time()
    event = _wandb_safe_id(args.event)
    run.log({f"pipeline/{event}_at": now})
    run.summary["pipeline/latest"] = event
    run.summary[f"pipeline/{event}_unix"] = now
    run.finish()
    print(f"wandb_milestone_logged: {event}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
