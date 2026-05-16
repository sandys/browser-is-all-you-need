"""CLI entrypoint for w8-biayn."""

from __future__ import annotations

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import datasets, upstreams
from .constants import DEFAULT_CREDENTIALS_PATH, DEFAULT_RENDER_DIR
from .secrets import CredentialError, default_bucket_for_project, get_project_id
from .shell import run_command
from .sky_config import Pipeline, RenderOptions, write_sky_yaml

app = typer.Typer(help="Command and control for BrowserGym RL on rLLM, SkyRL, and GCP.")
upstreams_app = typer.Typer(help="Manage pinned upstream source clones.")
data_app = typer.Typer(help="Prepare BrowserGym datasets.")
config_app = typer.Typer(help="Render SkyPilot and training configs.")
app.add_typer(upstreams_app, name="upstreams")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
console = Console()


def _project_id(credentials: str) -> str:
    try:
        return get_project_id(credentials)
    except CredentialError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _render_options(
    pipeline: Pipeline,
    credentials: str,
    bucket: Optional[str],
    accelerators: str,
    num_nodes: int,
    cluster: Optional[str],
    logger: str,
    wandb_secret: bool,
    webarena_archives_gcs: Optional[str] = None,
) -> RenderOptions:
    project_id = _project_id(credentials)
    return RenderOptions(
        pipeline=pipeline,
        project_id=project_id,
        bucket=bucket,
        credentials_path=credentials,
        accelerators=accelerators,
        num_nodes=num_nodes,
        cluster_name=cluster,
        logger=logger,
        wandb_secret=wandb_secret,
        webarena_archives_gcs=webarena_archives_gcs,
    )


@app.command()
def doctor(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    cloud: bool = typer.Option(False, help="Run cloud checks such as `sky check gcp`."),
) -> None:
    """Validate local prerequisites without printing secrets."""
    table = Table(title="w8-biayn doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    for tool in ("uv", "git", "gcloud", "sky"):
        found = shutil.which(tool)
        table.add_row(tool, "ok" if found else "missing", found or "not on PATH")

    try:
        project_id = get_project_id(credentials)
        table.add_row("credentials", "ok", f"{credentials} project={project_id}")
        table.add_row("artifact bucket", "default", default_bucket_for_project(project_id))
    except CredentialError as exc:
        table.add_row("credentials", "error", str(exc))

    console.print(table)

    if cloud:
        project_id = _project_id(credentials)
        env = {**os.environ, "GOOGLE_APPLICATION_CREDENTIALS": str(Path(credentials).resolve())}
        run_command(["gcloud", "auth", "activate-service-account", f"--key-file={credentials}"], env=env)
        run_command(["gcloud", "config", "set", "project", project_id], env=env)
        run_command(["sky", "check", "gcp"], env=env)
        check_json = subprocess.run(
            ["sky", "check", "gcp", "-o", "json"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            enabled_summary = json.loads(check_json.stdout)
        except json.JSONDecodeError:
            enabled_summary = {}
        if "gcp" not in json.dumps(enabled_summary).lower():
            console.print("[red]GCP is not enabled for SkyPilot. Fix the IAM errors above and rerun doctor.[/red]")
            raise typer.Exit(1)


@upstreams_app.command("clone")
def upstreams_clone(
    name: Optional[str] = typer.Argument(None, help="One upstream key, or omit for all."),
    dry_run: bool = typer.Option(False, help="Print git commands without running them."),
) -> None:
    """Clone or update pinned upstream repos into `.cache/upstreams`."""
    keys = [name] if name else list(upstreams.UPSTREAMS)
    for key in keys:
        if key not in upstreams.UPSTREAMS:
            raise typer.BadParameter(f"Unknown upstream: {key}")
        path = upstreams.clone_or_update(key, dry_run=dry_run)
        console.print(f"{key}: {path}")


@upstreams_app.command("status")
def upstreams_status() -> None:
    """Show pinned upstream clone state."""
    table = Table(title="Upstreams")
    for column in ("key", "state", "head", "pin", "path"):
        table.add_column(column)
    for row in upstreams.status():
        table.add_row(row["key"], row["state"], row["head"], row["pin"], row["path"])
    console.print(table)


@data_app.command("prepare")
def data_prepare(
    benchmark: datasets.Benchmark = typer.Argument(..., help="BrowserGym benchmark."),
    out: str = typer.Option(..., "--out", help="Output directory for train/validation parquet."),
    train_ratio: float = typer.Option(0.8, help="Train split ratio."),
    seed: int = typer.Option(42, help="Deterministic split seed."),
    limit: Optional[int] = typer.Option(None, help="Optional max env ids for smoke tests."),
) -> None:
    """Prepare BrowserGym MiniWoB/WebArena parquet datasets."""
    train_path, val_path = datasets.prepare_dataset(
        benchmark,
        out,
        train_ratio=train_ratio,
        seed=seed,
        limit=limit,
    )
    console.print(f"train: {train_path}")
    console.print(f"validation: {val_path}")


@config_app.command("render")
def config_render(
    pipeline: Pipeline = typer.Argument(..., help="Pipeline to render."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output YAML path."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI."),
    accelerators: str = typer.Option("A100:4", help="SkyPilot accelerator request."),
    num_nodes: int = typer.Option(1, help="SkyPilot node count."),
    cluster: Optional[str] = typer.Option(None, help="SkyPilot cluster name."),
    logger: str = typer.Option("console", help="SkyRL logger: console or wandb."),
    wandb_secret: bool = typer.Option(False, help="Include WANDB_API_KEY as a SkyPilot secret."),
    webarena_archives_gcs: Optional[str] = typer.Option(
        None,
        help="GCS prefix containing official WebArena archives for VM-side service provisioning.",
    ),
) -> None:
    """Render a SkyPilot YAML file."""
    options = _render_options(
        pipeline,
        credentials,
        bucket,
        accelerators,
        num_nodes,
        cluster,
        logger,
        wandb_secret,
        webarena_archives_gcs,
    )
    output_path = output or f"{DEFAULT_RENDER_DIR}/{pipeline}.sky.yaml"
    written = write_sky_yaml(options, output_path)
    console.print(str(written))


@app.command()
def launch(
    pipeline: Pipeline = typer.Argument(..., help="Pipeline to launch."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI."),
    accelerators: str = typer.Option("A100:4", help="SkyPilot accelerator request."),
    num_nodes: int = typer.Option(1, help="SkyPilot node count."),
    cluster: Optional[str] = typer.Option(None, help="SkyPilot cluster name."),
    logger: str = typer.Option("console", help="SkyRL logger: console or wandb."),
    wandb_secret: bool = typer.Option(False, help="Pass WANDB_API_KEY through SkyPilot secrets."),
    webarena_archives_gcs: Optional[str] = typer.Option(
        None,
        help="GCS prefix containing official WebArena archives for VM-side service provisioning.",
    ),
    yes: bool = typer.Option(True, help="Pass -y to SkyPilot to skip confirmation prompts."),
    down_after: bool = typer.Option(True, help="Pass --down so successful smoke runs tear down the cluster."),
    dry_run: bool = typer.Option(False, help="Render and print commands without launching."),
    skip_auth: bool = typer.Option(False, help="Do not run gcloud service-account activation first."),
) -> None:
    """Render config and launch a SkyPilot job."""
    options = _render_options(
        pipeline,
        credentials,
        bucket,
        accelerators,
        num_nodes,
        cluster,
        logger,
        wandb_secret,
        webarena_archives_gcs,
    )
    output = write_sky_yaml(options, f"{DEFAULT_RENDER_DIR}/{pipeline}.sky.yaml")
    env = {
        **os.environ,
        "GOOGLE_APPLICATION_CREDENTIALS": str(Path(credentials).resolve()),
    }
    if not skip_auth:
        run_command(
            ["gcloud", "auth", "activate-service-account", f"--key-file={credentials}"],
            env=env,
            dry_run=dry_run,
        )
        run_command(["gcloud", "config", "set", "project", options.project_id], env=env, dry_run=dry_run)
    sky_args = ["sky", "launch", "-c", options.name]
    if yes:
        sky_args.append("-y")
    if down_after:
        sky_args.append("--down")
    sky_args.append(str(output))
    run_command(sky_args, env=env, dry_run=dry_run)


@app.command()
def status(cluster: Optional[str] = typer.Option(None, help="Optional cluster name.")) -> None:
    """Show SkyPilot status."""
    args = ["sky", "status"]
    if cluster:
        args.append(cluster)
    run_command(args)


@app.command()
def logs(cluster: str = typer.Argument("w8-biayn-miniwob", help="Cluster name.")) -> None:
    """Tail SkyPilot logs."""
    run_command(["sky", "logs", cluster])


@app.command()
def down(cluster: str = typer.Argument("w8-biayn-miniwob", help="Cluster name.")) -> None:
    """Tear down a SkyPilot cluster."""
    run_command(["sky", "down", "-y", cluster])


if __name__ == "__main__":
    app()
