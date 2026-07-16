#!/usr/bin/env python3
"""Log canonical PIE C++ post-training evidence to W&B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from glm47_posttraining.integrations.wandb_posttraining import (
    log_comparison_run,
    log_eval_run,
    log_pipeline_milestone,
    log_stage_finalization,
    read_json,
    read_jsonl,
    resolve_experiment_id,
    safe_identifier,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill = subparsers.add_parser(
        "backfill-eval",
        help="Create a canonical eval run from preserved JSON/JSONL artifacts.",
    )
    _add_wandb_args(backfill)
    backfill.add_argument("--records", required=True)
    backfill.add_argument("--generated", required=True)
    backfill.add_argument("--summary", required=True)
    backfill.add_argument("--receipt", default="")
    backfill.add_argument("--artifact-path", action="append", default=[])
    backfill.add_argument("--label", default="")
    backfill.add_argument("--run-id", default="")
    backfill.add_argument("--name", default="")
    backfill.add_argument("--group", default="")
    backfill.add_argument("--job-type", default="eval")
    backfill.add_argument("--timing-status", default=os.environ.get("GLM47_TIMING_STATUS", "unverified"))
    backfill.add_argument("--output-dir", default="")

    compare = subparsers.add_parser(
        "compare",
        help="Create the base/SFT/GRPO comparison table from preserved summaries.",
    )
    _add_wandb_args(compare)
    compare.add_argument("--summary", action="append", required=True)
    compare.add_argument("--run-id", default="")
    compare.add_argument("--timing-status", default=os.environ.get("GLM47_TIMING_STATUS", "unverified"))
    compare.add_argument("--output-dir", required=True)

    finalize = subparsers.add_parser(
        "finalize-stage",
        help="Resume a Miles run and attach its curated final summary and artifacts.",
    )
    _add_wandb_args(finalize)
    finalize.add_argument("--run-id", required=True)
    finalize.add_argument("--group", default="")
    finalize.add_argument("--stage", required=True)
    finalize.add_argument("--status", choices=("success", "failed"), required=True)
    finalize.add_argument("--receipt", required=True)
    finalize.add_argument("--artifact-path", action="append", default=[])
    finalize.add_argument("--run-log", default="")
    finalize.add_argument("--rollout-dump-dir", default="")
    finalize.add_argument("--sync-metrics-dir", default="")
    finalize.add_argument("--checkpoint-dir", default="")
    finalize.add_argument(
        "--max-table-rows",
        type=int,
        default=int(os.environ.get("GLM47_WANDB_MAX_TABLE_ROWS", "5000")),
    )
    finalize.add_argument("--timing-status", default=os.environ.get("GLM47_TIMING_STATUS", "unverified"))
    finalize.add_argument("--output-dir", required=True)

    milestone = subparsers.add_parser(
        "milestone",
        help="Append one resumable event to the experiment pipeline run.",
    )
    _add_wandb_args(milestone)
    milestone.add_argument("--stage", required=True)
    milestone.add_argument("--run-id", default="")
    milestone.add_argument("--event", required=True)
    milestone.add_argument("--status", choices=("started", "success", "failed"), required=True)
    milestone.add_argument("--wall-s", type=float, default=None)
    milestone.add_argument("--repo-sha", default="")
    milestone.add_argument("--image", default="")
    milestone.add_argument("--receipt", default="")
    milestone.add_argument("--error", default="")
    milestone.add_argument("--event-time", type=float, default=None)
    return parser.parse_args(argv)


def _add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", ""), required=False)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--experiment-id", default=os.environ.get("GLM47_EXPERIMENT_ID", ""))
    parser.add_argument("--mode", default=os.environ.get("WANDB_MODE", "online"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.project:
        raise SystemExit("--project or WANDB_PROJECT is required")
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is required for this command; install it or use `uv run --with wandb`") from exc

    if args.command == "backfill-eval":
        result = _backfill_eval(wandb, args)
    elif args.command == "compare":
        result = _compare(wandb, args)
    elif args.command == "finalize-stage":
        result = _finalize_stage(wandb, args)
    else:
        result = _milestone(wandb, args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _backfill_eval(wandb: Any, args: argparse.Namespace) -> dict[str, str]:
    records_path = Path(args.records)
    generated_path = Path(args.generated)
    summary_path = Path(args.summary)
    receipt_path = Path(args.receipt) if args.receipt else None
    records = read_jsonl(records_path)
    generations = read_jsonl(generated_path)
    summary = read_json(summary_path)
    receipt = read_json(receipt_path) if receipt_path else {}
    source_label = str(summary.get("label") or receipt.get("label") or "eval")
    label = args.label or source_label
    if args.label:
        records = [{**row, "label": label} for row in records]
        generations = [{**row, "label": label} for row in generations]
    experiment_id = resolve_experiment_id(
        explicit=args.experiment_id,
        run_id=args.run_id,
        label=label,
    )
    run_id = safe_identifier(args.run_id or f"{experiment_id}-{label}-eval")
    output_dir = Path(args.output_dir) if args.output_dir else summary_path.parent
    summary.update(
        {
            "label": label,
            "experiment_id": experiment_id,
            "stage": "eval",
            "status": "success",
            "source_label": source_label,
            "timing_status": args.timing_status,
            "timing_trustworthy": args.timing_status == "verified",
            "elapsed_seconds": receipt.get("wall_s", receipt.get("elapsed_seconds")),
        }
    )
    artifact_paths: list[Path] = [records_path, generated_path, summary_path]
    if receipt_path:
        artifact_paths.append(receipt_path)
    generation_summary = summary_path.with_name(summary_path.name.replace(".summary.json", ".generation_summary.json"))
    if generation_summary.exists():
        artifact_paths.append(generation_summary)
    artifact_paths.extend(Path(path) for path in args.artifact_path)
    return log_eval_run(
        wandb,
        project=args.project,
        entity=args.entity or None,
        experiment_id=experiment_id,
        run_id=run_id,
        name=args.name or run_id,
        group=args.group or experiment_id,
        job_type=args.job_type,
        mode=args.mode,
        timing_status=args.timing_status,
        summary=summary,
        records=records,
        generations=generations,
        config={
            **receipt,
            "label": label,
            "source_label": source_label,
            "source_records": records_path.name,
            "source_generated": generated_path.name,
            "source_summary": summary_path.name,
            "backfill": True,
        },
        artifact_paths=artifact_paths,
        manifest_dir=output_dir,
        tags=("backfill",),
    )


def _compare(wandb: Any, args: argparse.Namespace) -> dict[str, str]:
    summary_paths = [Path(path) for path in args.summary]
    summaries = []
    for path in summary_paths:
        summary = read_json(path)
        source_label = str(summary.get("label") or "unknown")
        summary["source_label"] = source_label
        summary["label"] = _canonical_checkpoint_label(source_label)
        summaries.append(summary)
    experiment_id = resolve_experiment_id(explicit=args.experiment_id, run_id=args.run_id)
    run_id = safe_identifier(args.run_id or f"{experiment_id}-comparison")
    return log_comparison_run(
        wandb,
        project=args.project,
        entity=args.entity or None,
        experiment_id=experiment_id,
        run_id=run_id,
        mode=args.mode,
        timing_status=args.timing_status,
        summaries=summaries,
        summary_paths=summary_paths,
        output_dir=args.output_dir,
    )


def _finalize_stage(wandb: Any, args: argparse.Namespace) -> dict[str, str]:
    experiment_id = resolve_experiment_id(explicit=args.experiment_id, run_id=args.run_id)
    return log_stage_finalization(
        wandb,
        project=args.project,
        entity=args.entity or None,
        experiment_id=experiment_id,
        run_id=args.run_id,
        group=args.group or experiment_id,
        stage=args.stage,
        status=args.status,
        mode=args.mode,
        timing_status=args.timing_status,
        receipt=args.receipt,
        artifact_paths=args.artifact_path,
        manifest_dir=args.output_dir,
        run_log=args.run_log or None,
        rollout_dump_dir=args.rollout_dump_dir or None,
        sync_metrics_dir=args.sync_metrics_dir or None,
        checkpoint_dir=args.checkpoint_dir or None,
        max_table_rows=args.max_table_rows,
    )


def _milestone(wandb: Any, args: argparse.Namespace) -> dict[str, str]:
    experiment_id = resolve_experiment_id(explicit=args.experiment_id, label=args.stage)
    return log_pipeline_milestone(
        wandb,
        project=args.project,
        entity=args.entity or None,
        experiment_id=experiment_id,
        stage=args.stage,
        event=args.event,
        status=args.status,
        mode=args.mode,
        run_id=args.run_id,
        wall_s=args.wall_s,
        repo_sha=args.repo_sha,
        image=args.image,
        receipt=args.receipt or None,
        error=args.error,
        event_time=args.event_time,
    )


def _canonical_checkpoint_label(label: str) -> str:
    lowered = label.lower()
    for canonical in ("base", "sft", "grpo"):
        suffix = lowered[len(canonical) :] if lowered.startswith(canonical) else ""
        if lowered == canonical or (suffix and suffix[0] in "_-0123456789"):
            return canonical
    return label


if __name__ == "__main__":
    raise SystemExit(main())
