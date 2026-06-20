"""CLI entrypoint for w8-biayn."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import benchmarks, datasets, domdiff, osworld, upstreams
from .constants import (
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_ACCELERATORS,
    DEFAULT_DOMDIFF_ARTIFACT_IMAGE,
    DEFAULT_DOMDIFF_ARTIFACT_LOCATION,
    DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY,
    DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
    DEFAULT_DOMDIFF_IMAGE,
    DEFAULT_DOMDIFF_LOCAL_IMAGE,
    DEFAULT_DOMDIFF_MACHINE_TYPE,
    DEFAULT_DOMDIFF_MIN_CPU_PLATFORM,
    DEFAULT_DOMDIFF_VOLUME_GB,
    DEFAULT_DOMDIFF_ZONE,
    DEFAULT_GPU_CONTAINER_IMAGE,
    DEFAULT_HARBOR_R3_ACCELERATORS,
    DEFAULT_RENDER_DIR,
)
from .gcp_auth import GcpAuthError, check_project_permissions, service_account_env
from .harbor import tasks as harbor_tasks
from .harbor.docker_runner import HarborDockerTaskRunner, run_oracle_smoke
from .harbor.skyrl_dataset import prepare_harbor_skyrl_dataset
from .secrets import CredentialError, default_bucket_for_project, get_project_id
from .shell import run_command
from .sky_config import Pipeline, RenderOptions, default_accelerators_for, validate_remote_runtime_urls, write_sky_yaml

app = typer.Typer(help="Command and control for BrowserGym RL on rLLM, SkyRL, and GCP.")
upstreams_app = typer.Typer(help="Manage pinned upstream source clones.")
data_app = typer.Typer(help="Prepare BrowserGym datasets.")
config_app = typer.Typer(help="Render SkyPilot and training configs.")
domdiff_app = typer.Typer(help="Manage DOMDiff reward images and hosts.")
domdiff_local_app = typer.Typer(help="Run a local DOMDiff container and expose it with quick tunnels.")
benchmarks_app = typer.Typer(help="Inspect benchmark targets for the R3 pipeline.")
harbor_app = typer.Typer(help="Inspect and smoke packaged Harbor DOMDiff tasks.")
osworld_app = typer.Typer(help="Validate and smoke OSWorld desktop tasks from the upstream clone.")
app.add_typer(upstreams_app, name="upstreams")
app.add_typer(data_app, name="data")
app.add_typer(config_app, name="config")
app.add_typer(domdiff_app, name="domdiff")
domdiff_app.add_typer(domdiff_local_app, name="local")
app.add_typer(benchmarks_app, name="benchmarks")
app.add_typer(harbor_app, name="harbor")
app.add_typer(osworld_app, name="osworld")
console = Console()

SKYPILOT_GCP_LAUNCH_PERMISSIONS = (
    "compute.disks.create",
    "compute.disks.list",
    "compute.firewalls.create",
    "compute.firewalls.delete",
    "compute.firewalls.get",
    "compute.globalOperations.get",
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.list",
    "compute.instances.setLabels",
    "compute.instances.setServiceAccount",
    "compute.instances.start",
    "compute.instances.stop",
    "compute.networks.get",
    "compute.networks.getEffectiveFirewalls",
    "compute.networks.list",
    "compute.projects.get",
    "compute.reservations.list",
    "compute.subnetworks.list",
    "compute.subnetworks.use",
    "compute.subnetworks.useExternalIp",
    "compute.zoneOperations.get",
    "iam.roles.get",
    "iam.serviceAccounts.create",
    "iam.serviceAccounts.get",
    "resourcemanager.projects.get",
    "resourcemanager.projects.getIamPolicy",
    "resourcemanager.projects.setIamPolicy",
    "serviceusage.services.enable",
    "serviceusage.services.list",
    "serviceusage.services.use",
    "storage.buckets.create",
    "storage.buckets.delete",
)


def _project_id(credentials: str) -> str:
    try:
        return get_project_id(credentials)
    except CredentialError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _service_account_env(credentials: str, *, project_id: Optional[str] = None) -> dict[str, str]:
    try:
        return service_account_env(credentials, project_id=project_id)
    except GcpAuthError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _render_options(
    pipeline: Pipeline,
    credentials: str,
    bucket: Optional[str],
    accelerators: Optional[str],
    num_nodes: int,
    cluster: Optional[str],
    logger: str,
    wandb_secret: bool,
    webarena_archives_gcs: Optional[str] = None,
    chromiumrl_url: Optional[str] = None,
    cdp_url: Optional[str] = None,
    domdiff_reward_image: str = DEFAULT_DOMDIFF_IMAGE,
    benchmark: Optional[str] = None,
    harbor_task_ids: Optional[list[str]] = None,
    harbor_oracle: bool = True,
    gpu_container_image: str = DEFAULT_GPU_CONTAINER_IMAGE,
) -> RenderOptions:
    project_id = _project_id(credentials)
    normalized_harbor_task_ids = tuple(harbor_task_ids or harbor_tasks.DEFAULT_HARBOR_TASK_IDS)
    if benchmark == "harbor-domdiff-browser-swe":
        _validate_harbor_tasks(normalized_harbor_task_ids)
    effective_accelerators = accelerators or default_accelerators_for(pipeline, benchmark)
    options = RenderOptions(
        pipeline=pipeline,
        project_id=project_id,
        bucket=bucket,
        credentials_path=credentials,
        accelerators=effective_accelerators,
        num_nodes=num_nodes,
        cluster_name=cluster,
        logger=logger,
        wandb_secret=wandb_secret,
        webarena_archives_gcs=webarena_archives_gcs,
        chromiumrl_url=chromiumrl_url,
        cdp_url=cdp_url,
        domdiff_reward_image=domdiff_reward_image,
        benchmark=benchmark,
        harbor_task_ids=normalized_harbor_task_ids,
        harbor_oracle=harbor_oracle,
        gpu_container_image=gpu_container_image,
    )
    try:
        validate_remote_runtime_urls(options)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return options


def _validate_harbor_tasks(task_ids: tuple[str, ...] | list[str]) -> None:
    try:
        harbor_tasks.validate_tasks(task_ids=list(task_ids))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _warn_if_harbor_r3_accelerator_risk(options: RenderOptions) -> None:
    """Warn before spending a paid launch on a topology known to OOM."""

    accelerator_name = options.accelerators.split(":", maxsplit=1)[0].strip().upper()
    if options.benchmark == "harbor-domdiff-browser-swe" and accelerator_name == "A100":
        console.print(
            "[yellow]warning:[/yellow] Harbor DOMDiff R3 on 40GB A100 GPUs can reach DOMDiff reward "
            "scoring but needs CPU optimizer offload for the Megatron optimizer step. "
            "This render enables offload automatically; use the default "
            f"{DEFAULT_HARBOR_R3_ACCELERATORS} topology when H100 quota is available."
        )


def _domdiff_config(
    *,
    run_id: Optional[str],
    credentials: str,
    project: Optional[str],
    zone: str,
    machine_type: str,
    min_cpu_platform: str,
    network: str,
    boot_disk_type: str,
    volume_gb: int,
    reward_image: str,
    keep: bool = False,
) -> domdiff.DomdiffConfig:
    return domdiff.DomdiffConfig(
        run_id=run_id or domdiff.default_run_id(),
        credentials_path=credentials,
        project_id=project or "",
        zone=zone,
        machine_type=machine_type,
        min_cpu_platform=min_cpu_platform,
        network=network,
        boot_disk_type=boot_disk_type,
        volume_gb=volume_gb,
        reward_image=reward_image,
        keep=keep,
    )


def _local_domdiff_config(
    *,
    run_id: Optional[str],
    image: str,
    container_name: str,
    host_health_port: int,
    host_cdp_port: int,
    reward_port: int,
    keep_container: bool = False,
    publish_cdp: bool = False,
) -> domdiff.LocalDomdiffConfig:
    return domdiff.LocalDomdiffConfig(
        run_id=run_id or domdiff.default_run_id("domdiff-local"),
        image=image,
        container_name=container_name,
        host_health_port=host_health_port,
        host_cdp_port=host_cdp_port,
        reward_port=reward_port,
        keep_container=keep_container,
        publish_cdp=publish_cdp,
    )


def _artifact_image_config(
    *,
    source_image: str,
    credentials: str,
    project: Optional[str],
    location: str,
    repository: str,
    image_name: str,
    tag: Optional[str],
    create_repository: bool,
) -> domdiff.ArtifactImageConfig:
    return domdiff.ArtifactImageConfig(
        source_image=source_image,
        credentials_path=credentials,
        project_id=project or "",
        location=location,
        repository=repository,
        image_name=image_name,
        tag=tag or "",
        create_repository=create_repository,
        gcloud_config_dir=f".w8-biayn/domdiff/image-push/{domdiff.safe_name(repository, max_len=40)}-gcloud-config",
    )


def _maybe_push_local_reward_image(
    *,
    local_reward_image: Optional[str],
    reward_image: str,
    credentials: str,
    project: Optional[str],
    artifact_location: str,
    artifact_repository: str,
    artifact_image_name: str,
    artifact_tag: Optional[str],
    create_repository: bool,
    dry_run: bool,
) -> str:
    if not local_reward_image:
        return reward_image
    config = _artifact_image_config(
        source_image=local_reward_image,
        credentials=credentials,
        project=project,
        location=artifact_location,
        repository=artifact_repository,
        image_name=artifact_image_name,
        tag=artifact_tag,
        create_repository=create_repository,
    )
    if dry_run:
        console.print(domdiff.artifact_push_plan(config))
        return domdiff.push_local_image_to_artifact_registry(config, dry_run=True)
    target = domdiff.push_local_image_to_artifact_registry(config)
    console.print(f"pushed_reward_image: {target}")
    return target


@app.command()
def doctor(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    cloud: bool = typer.Option(False, help="Run cloud checks such as `sky check gcp`."),
    domdiff_checks: bool = typer.Option(False, "--domdiff", help="Include DOMDiff reward-host prerequisites."),
) -> None:
    """Validate local prerequisites without printing secrets."""
    table = Table(title="w8-biayn doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    tools = ["uv", "git", "gcloud", "sky"]
    if domdiff_checks:
        tools.extend(["docker", "ssh", "scp", "ssh-keygen", "curl"])
    for tool in tools:
        found = shutil.which(tool)
        table.add_row(tool, "ok" if found else "missing", found or "not on PATH")
    if domdiff_checks:
        cloudflared = domdiff.find_cloudflared()
        table.add_row("cloudflared", "ok" if cloudflared else "missing", str(cloudflared or "not on PATH or .local/cloudflared"))
        docker = shutil.which("docker")
        if docker:
            image_ok = domdiff.local_docker_image_exists(DEFAULT_DOMDIFF_LOCAL_IMAGE)
            table.add_row(
                DEFAULT_DOMDIFF_LOCAL_IMAGE,
                "ok" if image_ok else "missing",
                "local Docker image is available" if image_ok else "build/tag it locally or use the prebuilt remote image",
            )
        table.add_row("/dev/kvm", "ok" if Path("/dev/kvm").exists() else "missing", "required for local DOMDiff container")

    try:
        project_id = get_project_id(credentials)
        table.add_row("credentials", "ok", f"{credentials} project={project_id}")
        table.add_row("artifact bucket", "default", default_bucket_for_project(project_id))
    except CredentialError as exc:
        table.add_row("credentials", "error", str(exc))

    console.print(table)

    if cloud:
        project_id = _project_id(credentials)
        env = _service_account_env(credentials, project_id=project_id)
        console.print("cloud credentials: service-account JSON env override; gcloud auth/config is not modified.")
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
        sky_gcp_enabled = "gcp" in json.dumps(enabled_summary).lower()
        try:
            missing_launch_permissions = check_project_permissions(
                credentials,
                project_id=project_id,
                permissions=SKYPILOT_GCP_LAUNCH_PERMISSIONS,
            )
        except GcpAuthError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if missing_launch_permissions:
            console.print(
                "[red]GCP service account lacks SkyPilot launch permissions: "
                + ", ".join(missing_launch_permissions)
                + "[/red]"
            )
            raise typer.Exit(1)
        if domdiff_checks:
            config = _domdiff_config(
                run_id="doctor",
                credentials=credentials,
                project=None,
                zone=DEFAULT_DOMDIFF_ZONE,
                machine_type=DEFAULT_DOMDIFF_MACHINE_TYPE,
                min_cpu_platform=DEFAULT_DOMDIFF_MIN_CPU_PLATFORM,
                network="default",
                boot_disk_type=DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
                volume_gb=DEFAULT_DOMDIFF_VOLUME_GB,
                reward_image=DEFAULT_DOMDIFF_IMAGE,
            )
            host = domdiff.create_host(config)
            host.preflight_permissions()
            console.print("[green]DOMDiff GCP permission preflight passed.[/green]")
        if not sky_gcp_enabled:
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
    accelerators: Optional[str] = typer.Option(
        None,
        help=(
            "SkyPilot accelerator request. Defaults to "
            f"{DEFAULT_ACCELERATORS}, or {DEFAULT_HARBOR_R3_ACCELERATORS} for Harbor DOMDiff R3."
        ),
    ),
    num_nodes: int = typer.Option(1, help="SkyPilot node count."),
    cluster: Optional[str] = typer.Option(None, help="SkyPilot cluster name."),
    logger: str = typer.Option("console", help="SkyRL logger: console or wandb."),
    wandb_secret: bool = typer.Option(False, help="Include WANDB_API_KEY as a SkyPilot secret."),
    webarena_archives_gcs: Optional[str] = typer.Option(
        None,
        help="GCS prefix containing official WebArena archives for VM-side service provisioning.",
    ),
    chromiumrl_url: Optional[str] = typer.Option(None, help="DOMDiff ChromiumRL reward service URL to inject."),
    cdp_url: Optional[str] = typer.Option(None, help="DOMDiff CDP WebSocket URL to inject."),
    domdiff_reward_image: str = typer.Option(DEFAULT_DOMDIFF_IMAGE, help="DOMDiff reward image metadata to render."),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark key to record in the rendered config."),
    harbor_task: Optional[list[str]] = typer.Option(None, "--harbor-task", help="Packaged Harbor task id to include. Repeat to select multiple tasks."),
    harbor_oracle: bool = typer.Option(True, "--harbor-oracle/--no-harbor-oracle", help="Run packaged Harbor oracle patches before verification."),
    gpu_container_image: str = typer.Option(DEFAULT_GPU_CONTAINER_IMAGE, help="Google GPU Docker image used for Harbor R3 training."),
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
        chromiumrl_url,
        cdp_url,
        domdiff_reward_image,
        benchmark,
        harbor_task,
        harbor_oracle,
        gpu_container_image,
    )
    output_path = output or f"{DEFAULT_RENDER_DIR}/{pipeline}.sky.yaml"
    written = write_sky_yaml(options, output_path)
    console.print(str(written))


@app.command()
def launch(
    pipeline: Pipeline = typer.Argument(..., help="Pipeline to launch."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI."),
    accelerators: Optional[str] = typer.Option(
        None,
        help=(
            "SkyPilot accelerator request. Defaults to "
            f"{DEFAULT_ACCELERATORS}, or {DEFAULT_HARBOR_R3_ACCELERATORS} for Harbor DOMDiff R3."
        ),
    ),
    num_nodes: int = typer.Option(1, help="SkyPilot node count."),
    cluster: Optional[str] = typer.Option(None, help="SkyPilot cluster name."),
    logger: str = typer.Option("console", help="SkyRL logger: console or wandb."),
    wandb_secret: bool = typer.Option(False, help="Pass WANDB_API_KEY through SkyPilot secrets."),
    webarena_archives_gcs: Optional[str] = typer.Option(
        None,
        help="GCS prefix containing official WebArena archives for VM-side service provisioning.",
    ),
    with_domdiff: bool = typer.Option(False, help="Provision a GCP DOMDiff reward host before launching R3."),
    with_local_domdiff: bool = typer.Option(False, help="Start a local DOMDiff container and expose the reward service through a Cloudflare quick tunnel."),
    chromiumrl_url: Optional[str] = typer.Option(None, help="Existing DOMDiff ChromiumRL reward service URL."),
    cdp_url: Optional[str] = typer.Option(None, help="Existing DOMDiff CDP WebSocket URL."),
    domdiff_run_id: Optional[str] = typer.Option(None, help="Run id for the DOMDiff reward host."),
    domdiff_project: Optional[str] = typer.Option(None, help="Override GCP project for the DOMDiff reward host."),
    domdiff_zone: str = typer.Option(DEFAULT_DOMDIFF_ZONE, help="GCP zone for the DOMDiff reward host."),
    domdiff_machine_type: str = typer.Option(DEFAULT_DOMDIFF_MACHINE_TYPE, help="GCP machine type for DOMDiff."),
    domdiff_min_cpu_platform: str = typer.Option(DEFAULT_DOMDIFF_MIN_CPU_PLATFORM, help="GCP minimum CPU platform for nested virtualization."),
    domdiff_network: str = typer.Option("default", help="GCP VPC network for the DOMDiff reward host."),
    domdiff_boot_disk_type: str = typer.Option(DEFAULT_DOMDIFF_BOOT_DISK_TYPE, help="DOMDiff boot disk type."),
    domdiff_volume_gb: int = typer.Option(DEFAULT_DOMDIFF_VOLUME_GB, help="DOMDiff boot disk size in GB."),
    domdiff_reward_image: str = typer.Option(DEFAULT_DOMDIFF_IMAGE, help="Prebuilt DOMDiff reward image."),
    local_reward_image: Optional[str] = typer.Option(
        None,
        "--local-reward-image",
        help="Local Docker image to push to Google Artifact Registry and use as the DOMDiff reward image.",
    ),
    artifact_location: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_LOCATION, help="Artifact Registry location for --local-reward-image."),
    artifact_repository: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY, help="Artifact Registry repository for --local-reward-image."),
    artifact_image_name: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_IMAGE, help="Artifact Registry image name for --local-reward-image."),
    artifact_tag: Optional[str] = typer.Option(None, help="Artifact Registry tag for --local-reward-image. Defaults to local-<image-id>."),
    create_artifact_repository: bool = typer.Option(True, "--create-artifact-repository/--no-create-artifact-repository", help="Create the Artifact Registry repository if it is missing."),
    keep_domdiff: bool = typer.Option(False, help="Leave the DOMDiff VM running after launch exits."),
    local_domdiff_run_id: Optional[str] = typer.Option(None, help="Run id for --with-local-domdiff state."),
    local_domdiff_image: str = typer.Option(DEFAULT_DOMDIFF_LOCAL_IMAGE, help="Local Docker image for --with-local-domdiff."),
    local_domdiff_container: str = typer.Option("android-world-domdiff", help="Local DOMDiff container name."),
    local_domdiff_health_port: int = typer.Option(5080, help="Local host port for DOMDiff /health."),
    local_domdiff_cdp_port: int = typer.Option(9224, help="Local host port for DOMDiff CDP."),
    local_chromiumrl_port: int = typer.Option(8080, help="Local host port for the ChromiumRL reward service."),
    local_publish_cdp: bool = typer.Option(False, "--local-publish-cdp", help="Also expose local CDP through a Cloudflare quick tunnel for debugging."),
    keep_local_domdiff: bool = typer.Option(False, help="Leave local DOMDiff processes/container running after launch exits."),
    benchmark: Optional[str] = typer.Option(None, help="Benchmark key to record for this run."),
    harbor_task: Optional[list[str]] = typer.Option(None, "--harbor-task", help="Packaged Harbor task id to include. Repeat to select multiple tasks."),
    harbor_oracle: bool = typer.Option(True, "--harbor-oracle/--no-harbor-oracle", help="Run packaged Harbor oracle patches before verification."),
    gpu_container_image: str = typer.Option(DEFAULT_GPU_CONTAINER_IMAGE, help="Google GPU Docker image used for Harbor R3 training."),
    yes: bool = typer.Option(True, help="Pass -y to SkyPilot to skip confirmation prompts."),
    down_after: bool = typer.Option(True, help="Pass --down so successful smoke runs tear down the cluster."),
    dry_run: bool = typer.Option(False, help="Render and print commands without launching."),
) -> None:
    """Render config and launch a SkyPilot job."""
    domdiff_state: domdiff.DomdiffState | None = None
    local_domdiff_state: domdiff.LocalDomdiffState | None = None
    if with_domdiff and pipeline != "r3":
        raise typer.BadParameter("--with-domdiff is currently supported only for the r3 pipeline")
    if with_local_domdiff and pipeline != "r3":
        raise typer.BadParameter("--with-local-domdiff is currently supported only for the r3 pipeline")
    if with_domdiff and with_local_domdiff:
        raise typer.BadParameter("Use only one of --with-domdiff or --with-local-domdiff")
    if with_local_domdiff and local_reward_image:
        raise typer.BadParameter("--local-reward-image uploads to Artifact Registry; use --local-domdiff-image for local tunnel mode")
    if benchmark:
        try:
            benchmarks.get_benchmark(benchmark)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from exc
    is_harbor_benchmark = benchmark == "harbor-domdiff-browser-swe"
    selected_harbor_tasks = tuple(harbor_task or harbor_tasks.DEFAULT_HARBOR_TASK_IDS)
    if is_harbor_benchmark:
        if pipeline != "r3":
            raise typer.BadParameter("harbor-domdiff-browser-swe is supported only for the r3 pipeline")
        _validate_harbor_tasks(selected_harbor_tasks)
    domdiff_reward_image = _maybe_push_local_reward_image(
        local_reward_image=local_reward_image,
        reward_image=domdiff_reward_image,
        credentials=credentials,
        project=domdiff_project,
        artifact_location=artifact_location,
        artifact_repository=artifact_repository,
        artifact_image_name=artifact_image_name,
        artifact_tag=artifact_tag,
        create_repository=create_artifact_repository,
        dry_run=dry_run,
    )
    if with_domdiff and not (chromiumrl_url and cdp_url):
        domdiff_config = _domdiff_config(
            run_id=domdiff_run_id,
            credentials=credentials,
            project=domdiff_project,
            zone=domdiff_zone,
            machine_type=domdiff_machine_type,
            min_cpu_platform=domdiff_min_cpu_platform,
            network=domdiff_network,
            boot_disk_type=domdiff_boot_disk_type,
            volume_gb=domdiff_volume_gb,
            reward_image=domdiff_reward_image,
            keep=keep_domdiff,
        )
        if dry_run:
            console.print(domdiff.dry_run_plan(domdiff_config))
            if benchmark:
                console.print(f"benchmark: {benchmark}")
            chromiumrl_url = chromiumrl_url or "https://<domdiff-reward-tunnel>"
            cdp_url = cdp_url or "wss://<domdiff-cdp-tunnel>"
        else:
            domdiff_state = domdiff.up(domdiff_config)
            chromiumrl_url = chromiumrl_url or domdiff_state.chromiumrl_url
            cdp_url = cdp_url or domdiff_state.cdp_url
    if with_local_domdiff and not chromiumrl_url:
        local_config = _local_domdiff_config(
            run_id=local_domdiff_run_id,
            image=local_domdiff_image,
            container_name=local_domdiff_container,
            host_health_port=local_domdiff_health_port,
            host_cdp_port=local_domdiff_cdp_port,
            reward_port=local_chromiumrl_port,
            keep_container=keep_local_domdiff,
            publish_cdp=local_publish_cdp,
        )
        if dry_run:
            console.print(domdiff.local_dry_run_plan(local_config))
            if benchmark:
                console.print(f"benchmark: {benchmark}")
            chromiumrl_url = chromiumrl_url or "https://<local-domdiff-reward-tunnel>"
            if local_publish_cdp:
                cdp_url = cdp_url or "wss://<local-domdiff-cdp-tunnel>"
            console.print(f"CHROMIUMRL_URL={chromiumrl_url}")
            if cdp_url:
                console.print(f"CDP_URL={cdp_url}")
        else:
            local_domdiff_state = domdiff.local_up(local_config)
            chromiumrl_url = chromiumrl_url or local_domdiff_state.chromiumrl_url
            cdp_url = cdp_url or local_domdiff_state.cdp_url

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
        chromiumrl_url,
        cdp_url,
        domdiff_reward_image,
        benchmark,
        list(selected_harbor_tasks),
        harbor_oracle,
        gpu_container_image,
    )
    _warn_if_harbor_r3_accelerator_risk(options)
    output = write_sky_yaml(options, f"{DEFAULT_RENDER_DIR}/{pipeline}.sky.yaml")
    env = _service_account_env(credentials, project_id=options.project_id)
    sky_args = ["sky", "launch", "-c", options.name]
    if yes:
        sky_args.append("-y")
    if down_after:
        sky_args.append("--down")
    sky_args.append(str(output))
    try:
        run_command(sky_args, env=env, dry_run=dry_run)
    finally:
        if domdiff_state is not None and not keep_domdiff:
            cleanup_config = _domdiff_config(
                run_id=domdiff_state.run_id,
                credentials=credentials,
                project=domdiff_state.project_id,
                zone=domdiff_state.zone,
                machine_type=domdiff_machine_type,
                min_cpu_platform=domdiff_min_cpu_platform,
                network=domdiff_network,
                boot_disk_type=domdiff_boot_disk_type,
                volume_gb=domdiff_volume_gb,
                reward_image=domdiff_state.reward_image,
                keep=False,
            )
            host = domdiff.create_host_from_state(cleanup_config, domdiff.run_dir_for(domdiff_state.run_id))
            host.cleanup()
        if local_domdiff_state is not None and not keep_local_domdiff:
            local_cleanup_config = _local_domdiff_config(
                run_id=local_domdiff_state.run_id,
                image=local_domdiff_state.image,
                container_name=local_domdiff_state.container_name,
                host_health_port=local_domdiff_health_port,
                host_cdp_port=local_domdiff_cdp_port,
                reward_port=local_chromiumrl_port,
                keep_container=False,
                publish_cdp=bool(local_domdiff_state.cdp_url),
            )
            host = domdiff.create_local_host_from_state(
                local_cleanup_config,
                domdiff.local_run_dir_for(local_domdiff_state.run_id),
            )
            host.cleanup()


@domdiff_app.command("push-image")
def domdiff_push_image(
    source_image: str = typer.Option(DEFAULT_DOMDIFF_LOCAL_IMAGE, help="Local Docker image tag to upload."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    project: Optional[str] = typer.Option(None, help="Override GCP project."),
    location: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_LOCATION, help="Artifact Registry location."),
    repository: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY, help="Artifact Registry repository name."),
    image_name: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_IMAGE, help="Artifact Registry image name."),
    tag: Optional[str] = typer.Option(None, help="Artifact Registry tag. Defaults to local-<image-id>."),
    create_repository: bool = typer.Option(True, "--create-repository/--no-create-repository", help="Create the repository if it is missing."),
    dry_run: bool = typer.Option(False, help="Print the push plan without uploading."),
) -> None:
    """Upload a local DOMDiff Docker image to Google Artifact Registry."""
    config = _artifact_image_config(
        source_image=source_image,
        credentials=credentials,
        project=project,
        location=location,
        repository=repository,
        image_name=image_name,
        tag=tag,
        create_repository=create_repository,
    )
    if dry_run:
        console.print(domdiff.artifact_push_plan(config))
        return
    target = domdiff.push_local_image_to_artifact_registry(config)
    console.print(f"image: {target}")
    console.print(f"use: uv run w8-biayn domdiff smoke --reward-image {target}")


@domdiff_local_app.command("up")
def domdiff_local_up(
    run_id: Optional[str] = typer.Option(None, help="Run id for state under .w8-biayn/domdiff-local."),
    image: str = typer.Option(DEFAULT_DOMDIFF_LOCAL_IMAGE, help="Local DOMDiff Docker image."),
    container_name: str = typer.Option("android-world-domdiff", help="Local Docker container name."),
    health_port: int = typer.Option(5080, help="Host port for DOMDiff /health."),
    cdp_port: int = typer.Option(9224, help="Host port for CDP /json/version."),
    reward_port: int = typer.Option(8080, help="Host port for the local ChromiumRL reward service."),
    publish_cdp: bool = typer.Option(False, "--publish-cdp", help="Also expose local CDP through a Cloudflare quick tunnel for debugging."),
    keep_container: bool = typer.Option(False, "--keep-container/--no-keep-container", help="Keep the Docker container if startup fails before tunnels are ready."),
    dry_run: bool = typer.Option(False, help="Print the local container/tunnel plan without starting anything."),
) -> None:
    """Start a local DOMDiff container, reward service, and Cloudflare quick tunnel."""
    config = _local_domdiff_config(
        run_id=run_id,
        image=image,
        container_name=container_name,
        host_health_port=health_port,
        host_cdp_port=cdp_port,
        reward_port=reward_port,
        keep_container=keep_container,
        publish_cdp=publish_cdp,
    )
    if dry_run:
        console.print(domdiff.local_dry_run_plan(config))
        return
    state = domdiff.local_up(config)
    console.print(f"run_id: {state.run_id}")
    console.print(f"chromiumrl_url: {state.chromiumrl_url}")
    if state.cdp_url:
        console.print(f"cdp_url: {state.cdp_url}")
    console.print(f"local_reward_url: {state.reward_url}")
    console.print(f"state: {domdiff.local_run_dir_for(state.run_id)}")


@domdiff_local_app.command("verify")
def domdiff_local_verify(
    run_id: Optional[str] = typer.Option(None, help="Run id to verify. Defaults to latest local run."),
    image: str = typer.Option(DEFAULT_DOMDIFF_LOCAL_IMAGE, help="Local DOMDiff Docker image metadata for state reconstruction."),
    health_port: int = typer.Option(5080, help="Host port for DOMDiff /health."),
    cdp_port: int = typer.Option(9224, help="Host port for CDP /json/version."),
    reward_port: int = typer.Option(8080, help="Host port for the local ChromiumRL reward service."),
) -> None:
    """Verify local KVM, CDP, reward tunnel, and one live DOMDiff evaluation."""
    run_dir = domdiff.resolve_local_run_dir(run_id)
    state = domdiff.LocalDomdiffState.read(run_dir)
    config = _local_domdiff_config(
        run_id=state.run_id,
        image=image,
        container_name=state.container_name,
        host_health_port=health_port,
        host_cdp_port=cdp_port,
        reward_port=reward_port,
    )
    report = domdiff.create_local_host_from_state(config, run_dir).verify()
    console.print_json(data=report)


@domdiff_local_app.command("logs")
def domdiff_local_logs(
    run_id: Optional[str] = typer.Option(None, help="Run id to collect logs from. Defaults to latest local run."),
) -> None:
    """Collect local Docker, reward service, and cloudflared logs."""
    run_dir = domdiff.resolve_local_run_dir(run_id)
    state = domdiff.LocalDomdiffState.read(run_dir)
    config = _local_domdiff_config(
        run_id=state.run_id,
        image=state.image,
        container_name=state.container_name,
        host_health_port=5080,
        host_cdp_port=9224,
        reward_port=8080,
    )
    domdiff.create_local_host_from_state(config, run_dir).collect_logs()
    console.print(str(run_dir))


@domdiff_local_app.command("down")
def domdiff_local_down(
    run_id: Optional[str] = typer.Option(None, help="Run id to stop. Defaults to latest local run."),
    keep_container: bool = typer.Option(False, "--keep-container", help="Stop tunnels/reward service but keep the Docker container."),
) -> None:
    """Stop local DOMDiff quick tunnels, reward service, and optionally the container."""
    run_dir = domdiff.resolve_local_run_dir(run_id)
    state = domdiff.LocalDomdiffState.read(run_dir)
    config = _local_domdiff_config(
        run_id=state.run_id,
        image=state.image,
        container_name=state.container_name,
        host_health_port=5080,
        host_cdp_port=9224,
        reward_port=8080,
        keep_container=keep_container,
    )
    domdiff.create_local_host_from_state(config, run_dir).cleanup(keep_container=keep_container)
    console.print(f"stopped: {state.run_id}")


@domdiff_local_app.command("smoke")
def domdiff_local_smoke(
    run_id: Optional[str] = typer.Option(None, help="Run id for state under .w8-biayn/domdiff-local."),
    image: str = typer.Option(DEFAULT_DOMDIFF_LOCAL_IMAGE, help="Local DOMDiff Docker image."),
    container_name: str = typer.Option("android-world-domdiff", help="Local Docker container name."),
    health_port: int = typer.Option(5080, help="Host port for DOMDiff /health."),
    cdp_port: int = typer.Option(9224, help="Host port for CDP /json/version."),
    reward_port: int = typer.Option(8080, help="Host port for the local ChromiumRL reward service."),
    publish_cdp: bool = typer.Option(False, "--publish-cdp", help="Also expose local CDP through a Cloudflare quick tunnel for debugging."),
    keep_running: bool = typer.Option(False, help="Leave local container, reward service, and tunnels running after verification."),
    dry_run: bool = typer.Option(False, help="Print the local container/tunnel plan without starting anything."),
) -> None:
    """Run a local DOMDiff smoke and tear it down by default."""
    config = _local_domdiff_config(
        run_id=run_id,
        image=image,
        container_name=container_name,
        host_health_port=health_port,
        host_cdp_port=cdp_port,
        reward_port=reward_port,
        keep_container=keep_running,
        publish_cdp=publish_cdp,
    )
    if dry_run:
        console.print(domdiff.local_dry_run_plan(config))
        return
    state: domdiff.LocalDomdiffState | None = None
    try:
        state = domdiff.local_up(config)
        host = domdiff.create_local_host_from_state(config, domdiff.local_run_dir_for(state.run_id))
        report = host.verify()
        console.print_json(data=report)
    finally:
        if state is not None and not keep_running:
            host = domdiff.create_local_host_from_state(config, domdiff.local_run_dir_for(state.run_id))
            host.cleanup()


@domdiff_app.command("up")
def domdiff_up(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    run_id: Optional[str] = typer.Option(None, help="Run id for state under .w8-biayn/domdiff."),
    project: Optional[str] = typer.Option(None, help="Override GCP project."),
    zone: str = typer.Option(DEFAULT_DOMDIFF_ZONE, help="GCP zone."),
    machine_type: str = typer.Option(DEFAULT_DOMDIFF_MACHINE_TYPE, help="GCP machine type."),
    min_cpu_platform: str = typer.Option(DEFAULT_DOMDIFF_MIN_CPU_PLATFORM, help="GCP minimum CPU platform."),
    network: str = typer.Option("default", help="GCP VPC network."),
    boot_disk_type: str = typer.Option(DEFAULT_DOMDIFF_BOOT_DISK_TYPE, help="Boot disk type."),
    volume_gb: int = typer.Option(DEFAULT_DOMDIFF_VOLUME_GB, help="Boot disk size in GB."),
    reward_image: str = typer.Option(DEFAULT_DOMDIFF_IMAGE, help="Prebuilt DOMDiff reward image."),
    local_reward_image: Optional[str] = typer.Option(
        None,
        "--local-reward-image",
        help="Local Docker image to push to Google Artifact Registry and use as the DOMDiff reward image.",
    ),
    artifact_location: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_LOCATION, help="Artifact Registry location for --local-reward-image."),
    artifact_repository: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY, help="Artifact Registry repository for --local-reward-image."),
    artifact_image_name: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_IMAGE, help="Artifact Registry image name for --local-reward-image."),
    artifact_tag: Optional[str] = typer.Option(None, help="Artifact Registry tag for --local-reward-image. Defaults to local-<image-id>."),
    create_artifact_repository: bool = typer.Option(True, "--create-artifact-repository/--no-create-artifact-repository", help="Create the Artifact Registry repository if it is missing."),
    keep: bool = typer.Option(False, help="Record the host as intentionally kept for debugging."),
    dry_run: bool = typer.Option(False, help="Print the GCP/container plan without launching."),
) -> None:
    """Create a GCP nested-virtualization DOMDiff reward host."""
    reward_image = _maybe_push_local_reward_image(
        local_reward_image=local_reward_image,
        reward_image=reward_image,
        credentials=credentials,
        project=project,
        artifact_location=artifact_location,
        artifact_repository=artifact_repository,
        artifact_image_name=artifact_image_name,
        artifact_tag=artifact_tag,
        create_repository=create_artifact_repository,
        dry_run=dry_run,
    )
    config = _domdiff_config(
        run_id=run_id,
        credentials=credentials,
        project=project,
        zone=zone,
        machine_type=machine_type,
        min_cpu_platform=min_cpu_platform,
        network=network,
        boot_disk_type=boot_disk_type,
        volume_gb=volume_gb,
        reward_image=reward_image,
        keep=keep,
    )
    if dry_run:
        console.print(domdiff.dry_run_plan(config))
        return
    state = domdiff.up(config)
    console.print(f"run_id: {state.run_id}")
    console.print(f"chromiumrl_url: {state.chromiumrl_url}")
    console.print(f"cdp_url: {state.cdp_url}")


@domdiff_app.command("verify")
def domdiff_verify(
    run_id: Optional[str] = typer.Option(None, help="Run id to verify. Defaults to latest run."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
) -> None:
    """Verify KVM, CDP, reward adapter health, and one live DOMDiff evaluation."""
    run_dir = domdiff.resolve_run_dir(run_id)
    state = domdiff.DomdiffState.read(run_dir)
    config = _domdiff_config(
        run_id=state.run_id,
        credentials=credentials,
        project=state.project_id,
        zone=state.zone,
        machine_type=DEFAULT_DOMDIFF_MACHINE_TYPE,
        min_cpu_platform=DEFAULT_DOMDIFF_MIN_CPU_PLATFORM,
        network="default",
        boot_disk_type=DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
        volume_gb=DEFAULT_DOMDIFF_VOLUME_GB,
        reward_image=state.reward_image,
    )
    report = domdiff.create_host_from_state(config, run_dir).verify()
    console.print_json(data=report)


@domdiff_app.command("logs")
def domdiff_logs(
    run_id: Optional[str] = typer.Option(None, help="Run id to collect logs from. Defaults to latest run."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
) -> None:
    """Collect DOMDiff VM, Docker, tunnel, and reward adapter logs."""
    run_dir = domdiff.resolve_run_dir(run_id)
    state = domdiff.DomdiffState.read(run_dir)
    config = _domdiff_config(
        run_id=state.run_id,
        credentials=credentials,
        project=state.project_id,
        zone=state.zone,
        machine_type=DEFAULT_DOMDIFF_MACHINE_TYPE,
        min_cpu_platform=DEFAULT_DOMDIFF_MIN_CPU_PLATFORM,
        network="default",
        boot_disk_type=DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
        volume_gb=DEFAULT_DOMDIFF_VOLUME_GB,
        reward_image=state.reward_image,
    )
    domdiff.create_host_from_state(config, run_dir).collect_logs()
    console.print(str(run_dir))


@domdiff_app.command("down")
def domdiff_down(
    run_id: Optional[str] = typer.Option(None, help="Run id to delete. Defaults to latest run."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
) -> None:
    """Delete a DOMDiff reward host and stop its tunnels/container."""
    run_dir = domdiff.resolve_run_dir(run_id)
    state = domdiff.DomdiffState.read(run_dir)
    config = _domdiff_config(
        run_id=state.run_id,
        credentials=credentials,
        project=state.project_id,
        zone=state.zone,
        machine_type=DEFAULT_DOMDIFF_MACHINE_TYPE,
        min_cpu_platform=DEFAULT_DOMDIFF_MIN_CPU_PLATFORM,
        network="default",
        boot_disk_type=DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
        volume_gb=DEFAULT_DOMDIFF_VOLUME_GB,
        reward_image=state.reward_image,
    )
    domdiff.create_host_from_state(config, run_dir).cleanup()
    console.print(f"deleted: {state.run_id}")


@domdiff_app.command("smoke")
def domdiff_smoke(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    run_id: Optional[str] = typer.Option(None, help="Run id for state under .w8-biayn/domdiff."),
    project: Optional[str] = typer.Option(None, help="Override GCP project."),
    zone: str = typer.Option(DEFAULT_DOMDIFF_ZONE, help="GCP zone."),
    machine_type: str = typer.Option(DEFAULT_DOMDIFF_MACHINE_TYPE, help="GCP machine type."),
    min_cpu_platform: str = typer.Option(DEFAULT_DOMDIFF_MIN_CPU_PLATFORM, help="GCP minimum CPU platform."),
    network: str = typer.Option("default", help="GCP VPC network."),
    reward_image: str = typer.Option(DEFAULT_DOMDIFF_IMAGE, help="Prebuilt DOMDiff reward image."),
    local_reward_image: Optional[str] = typer.Option(
        None,
        "--local-reward-image",
        help="Local Docker image to push to Google Artifact Registry and use as the DOMDiff reward image.",
    ),
    artifact_location: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_LOCATION, help="Artifact Registry location for --local-reward-image."),
    artifact_repository: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY, help="Artifact Registry repository for --local-reward-image."),
    artifact_image_name: str = typer.Option(DEFAULT_DOMDIFF_ARTIFACT_IMAGE, help="Artifact Registry image name for --local-reward-image."),
    artifact_tag: Optional[str] = typer.Option(None, help="Artifact Registry tag for --local-reward-image. Defaults to local-<image-id>."),
    create_artifact_repository: bool = typer.Option(True, "--create-artifact-repository/--no-create-artifact-repository", help="Create the Artifact Registry repository if it is missing."),
    keep: bool = typer.Option(False, help="Leave the VM running after verification."),
    dry_run: bool = typer.Option(False, help="Print the GCP/container plan without launching."),
) -> None:
    """Run the real GCP DOMDiff smoke and tear down by default."""
    reward_image = _maybe_push_local_reward_image(
        local_reward_image=local_reward_image,
        reward_image=reward_image,
        credentials=credentials,
        project=project,
        artifact_location=artifact_location,
        artifact_repository=artifact_repository,
        artifact_image_name=artifact_image_name,
        artifact_tag=artifact_tag,
        create_repository=create_artifact_repository,
        dry_run=dry_run,
    )
    config = _domdiff_config(
        run_id=run_id,
        credentials=credentials,
        project=project,
        zone=zone,
        machine_type=machine_type,
        min_cpu_platform=min_cpu_platform,
        network=network,
        boot_disk_type=DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
        volume_gb=DEFAULT_DOMDIFF_VOLUME_GB,
        reward_image=reward_image,
        keep=keep,
    )
    if dry_run:
        console.print(domdiff.dry_run_plan(config))
        return
    state: domdiff.DomdiffState | None = None
    try:
        state = domdiff.up(config)
        host = domdiff.create_host_from_state(config, domdiff.run_dir_for(state.run_id))
        report = host.verify()
        console.print_json(data=report)
    finally:
        if state is not None and not keep:
            host = domdiff.create_host_from_state(config, domdiff.run_dir_for(state.run_id))
            host.cleanup()


@osworld_app.command("list")
def osworld_list(
    domain: Optional[str] = typer.Option(None, help="Only list tasks from one OSWorld domain."),
    smoke_candidates: bool = typer.Option(
        False,
        "--smoke-candidates",
        help="Only show low-risk tasks with proxy=false and fixed_ip=false.",
    ),
    limit: Optional[int] = typer.Option(None, help="Maximum number of tasks to print."),
) -> None:
    """List OSWorld tasks discovered from the pinned upstream clone."""
    tasks = osworld.list_tasks(domain=domain, smoke_candidates=smoke_candidates)
    if limit is not None:
        tasks = tasks[:limit]
    table = Table(title="OSWorld tasks")
    for column in ("task", "proxy", "fixed_ip", "env_change", "instruction"):
        table.add_column(column)
    for task_info in tasks:
        instruction = task_info.instruction.replace("\n", " ")
        if len(instruction) > 100:
            instruction = instruction[:97] + "..."
        table.add_row(
            task_info.task,
            "" if task_info.proxy is None else str(task_info.proxy).lower(),
            "" if task_info.fixed_ip is None else str(task_info.fixed_ip).lower(),
            task_info.env_change,
            instruction,
        )
    console.print(table)
    console.print(f"tasks: {len(tasks)}")

@osworld_app.command("setup")
def osworld_setup(
    dry_run: bool = typer.Option(False, help="Print the upstream uv sync command without running it."),
) -> None:
    """Create/update the OSWorld-specific upstream Python environment."""
    try:
        osworld.setup(dry_run=dry_run)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print("OSWorld environment command completed." if not dry_run else "OSWorld setup dry run rendered.")

@osworld_app.command("validate")
def osworld_validate(
    task: str = typer.Option(
        osworld.DEFAULT_OSWORLD_TASK, help="Task formatted as <domain>/<task-id>."
    ),
    enable_proxy: bool = typer.Option(
        False,
        "--enable-proxy/--no-enable-proxy",
        help="Also validate the OSWorld proxy config file.",
    ),
    proxy_config_file: Optional[str] = typer.Option(
        None,
        help="Proxy JSON path. Defaults to upstream evaluation_examples/settings/proxy/dataimpulse.json.",
    ),
    parse_error_policy: str = typer.Option(
        "fail",
        help="SkyRL env policy for unparseable model actions: fail or retry.",
    ),
    fail_on_env_error: bool = typer.Option(
        True,
        "--fail-on-env-error/--retry-on-env-error",
        help="Terminate a trajectory when OSWorld reset/action execution raises an env error.",
    ),
) -> None:
    """Validate local OSWorld upstream, env, Docker, KVM, and task metadata."""
    rows = osworld.validate(
        task=task,
        enable_proxy=enable_proxy,
        proxy_config_file=proxy_config_file,
    )
    table = Table(title="OSWorld validation")
    for column in ("check", "status", "detail"):
        table.add_column(column)
    for row in rows:
        table.add_row(row.check, row.status, row.detail)
    console.print(table)
    if osworld.has_errors(rows):
        raise typer.Exit(1)

@osworld_app.command("run")
def osworld_run(
    tasks: list[str] = typer.Option(
        [], "--task", help="Task formatted as <domain>/<task-id>. Can be repeated."
    ),
    suite: Optional[str] = typer.Option(None, help="Named OSWorld suite to run, such as 'tiny'."),
    domain: Optional[str] = typer.Option(None, help="Run tasks from one OSWorld domain."),
    taskset: Optional[str] = typer.Option(
        None, help="Named OSWorld taskset or JSON path, such as arpo-subset32."
    ),
    limit: Optional[int] = typer.Option(None, help="Maximum number of selected tasks."),
    provider: str = typer.Option(
        osworld.DEFAULT_PROVIDER, help="OSWorld provider name. Docker is the current local default."
    ),
    observation_type: str = typer.Option(
        osworld.DEFAULT_OBSERVATION_TYPE, help="OSWorld observation type."
    ),
    model: str = typer.Option(
        osworld.DEFAULT_MODEL, help="OSWorld model argument passed through to upstream run.py."
    ),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL for local Qwen/Kimi-style OSWorld runs."
    ),
    api_key: Optional[str] = typer.Option(
        None, help="OpenAI-compatible API key. Use EMPTY for local vLLM servers that ignore it."
    ),
    max_steps: int = typer.Option(osworld.DEFAULT_MAX_STEPS, help="Max OSWorld agent steps."),
    max_tokens: int = typer.Option(osworld.DEFAULT_MAX_TOKENS, help="Max model output tokens per OSWorld action request."),
    max_trajectory_length: int = typer.Option(
        1, help="Number of previous OSWorld observations to keep in the upstream prompt history."
    ),
    a11y_tree_max_items: int = typer.Option(
        osworld.DEFAULT_A11Y_TREE_MAX_ITEMS,
        help="Maximum deduped accessibility-tree rows kept before token trimming.",
    ),
    a11y_iou_threshold: float = typer.Option(
        osworld.DEFAULT_A11Y_IOU_THRESHOLD,
        help="IoU threshold for removing duplicate accessibility-tree rows with the same text.",
    ),
    headless: bool = typer.Option(
        True, "--headless/--headed", help="Run OSWorld in headless mode."
    ),
    enable_proxy: bool = typer.Option(
        False,
        "--enable-proxy/--no-enable-proxy",
        help="Enable OSWorld proxy setup for tasks with proxy=true.",
    ),
    proxy_config_file: Optional[str] = typer.Option(
        None,
        help="Proxy JSON path. Defaults to upstream evaluation_examples/settings/proxy/dataimpulse.json.",
    ),
    client_password: str = typer.Option(
        osworld.DEFAULT_CLIENT_PASSWORD,
        help="Ubuntu VM password used by OSWorld proxy setup for sudo operations.",
    ),
    dry_run: bool = typer.Option(
        False, help="Print the upstream OSWorld command without running it."
    ),
) -> None:
    """Run or render local OSWorld tasks through upstream run.py."""
    try:
        metadata, task_keys, run_id = osworld.run(
            tasks=tuple(tasks),
            suite=suite,
            domain=domain,
            taskset=taskset,
            limit=limit,
            provider=provider,
            observation_type=observation_type,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_trajectory_length=max_trajectory_length,
            a11y_tree_max_items=a11y_tree_max_items,
            a11y_iou_threshold=a11y_iou_threshold,
            headless=headless,
            enable_proxy=enable_proxy,
            proxy_config_file=proxy_config_file,
            client_password=client_password,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not dry_run:
        record = osworld.read_run_record(run_id) if run_id is not None else {}
        result_dir = record.get("results")
        table = Table(title="OSWorld run result")
        for column in ("task", "status", "score", "result"):
            table.add_column(column)
        for task_key in task_keys:
            result = osworld.read_task_result(
                task_key, observation_type=observation_type, model=model, result_dir=result_dir
            )
            table.add_row(
                result.task,
                result.status,
                "" if result.score is None else str(result.score),
                str(result.result_file),
            )
        console.print(table)
        if run_id is not None:
            console.print(f"run_id: {run_id}")
        console.print(f"metadata: {metadata}")
        console.print(f"results: {result_dir or osworld.results_path()}")

@osworld_app.command("benchmark")
def osworld_benchmark(
    domains: list[str] = typer.Option(
        [], "--domain", help="OSWorld domain to include. Can be repeated. Defaults to all domains."
    ),
    taskset: Optional[str] = typer.Option(
        None, help="Named OSWorld taskset or JSON path, such as arpo-subset32."
    ),
    limit_per_domain: Optional[int] = typer.Option(
        None, help="Maximum number of tasks to run per domain."
    ),
    smoke_candidates: bool = typer.Option(
        False,
        "--smoke-candidates",
        help="Only run low-risk proxy=false/fixed_ip=false/env_change=low tasks.",
    ),
    provider: str = typer.Option(
        osworld.DEFAULT_PROVIDER, help="OSWorld provider name. Docker is the current local default."
    ),
    observation_type: str = typer.Option(
        osworld.DEFAULT_OBSERVATION_TYPE, help="OSWorld observation type."
    ),
    model: str = typer.Option(
        osworld.DEFAULT_MODEL, help="OSWorld model argument passed through to upstream run.py."
    ),
    base_url: Optional[str] = typer.Option(
        None, help="OpenAI-compatible base URL for local Qwen/Kimi-style OSWorld runs."
    ),
    api_key: Optional[str] = typer.Option(
        None, help="OpenAI-compatible API key. Use EMPTY for local vLLM servers that ignore it."
    ),
    max_steps: int = typer.Option(osworld.DEFAULT_MAX_STEPS, help="Max OSWorld agent steps."),
    max_tokens: int = typer.Option(osworld.DEFAULT_MAX_TOKENS, help="Max model output tokens per OSWorld action request."),
    max_trajectory_length: int = typer.Option(
        1, help="Number of previous OSWorld observations to keep in the upstream prompt history."
    ),
    a11y_tree_max_items: int = typer.Option(
        osworld.DEFAULT_A11Y_TREE_MAX_ITEMS,
        help="Maximum deduped accessibility-tree rows kept before token trimming.",
    ),
    a11y_iou_threshold: float = typer.Option(
        osworld.DEFAULT_A11Y_IOU_THRESHOLD,
        help="IoU threshold for removing duplicate accessibility-tree rows with the same text.",
    ),
    headless: bool = typer.Option(
        True, "--headless/--headed", help="Run OSWorld in headless mode."
    ),
    enable_proxy: bool = typer.Option(
        False,
        "--enable-proxy/--no-enable-proxy",
        help="Enable OSWorld proxy setup for tasks with proxy=true.",
    ),
    proxy_config_file: Optional[str] = typer.Option(
        None,
        help="Proxy JSON path. Defaults to upstream evaluation_examples/settings/proxy/dataimpulse.json.",
    ),
    client_password: str = typer.Option(
        osworld.DEFAULT_CLIENT_PASSWORD,
        help="Ubuntu VM password used by OSWorld proxy setup for sudo operations.",
    ),
    dry_run: bool = typer.Option(
        False, help="Print the benchmark plan without running domains."
    ),
) -> None:
    """Run all or selected OSWorld domains as a benchmark sequence."""
    try:
        result = osworld.benchmark(
            domains=tuple(domains),
            taskset=taskset,
            limit_per_domain=limit_per_domain,
            smoke_candidates=smoke_candidates,
            provider=provider,
            observation_type=observation_type,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_trajectory_length=max_trajectory_length,
            a11y_tree_max_items=a11y_tree_max_items,
            a11y_iou_threshold=a11y_iou_threshold,
            headless=headless,
            enable_proxy=enable_proxy,
            proxy_config_file=proxy_config_file,
            client_password=client_password,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if dry_run or result is None:
        return

    table = Table(title="OSWorld benchmark result")
    for column in ("domain", "run_id", "tasks", "completed", "successes", "failures", "average", "results"):
        table.add_column(column)
    for domain in result.domains:
        average = "" if domain.average_score is None else f"{domain.average_score:.3f}"
        table.add_row(
            domain.domain,
            domain.run_id,
            str(len(domain.tasks)),
            str(domain.completed),
            str(domain.successes),
            str(domain.failures),
            average,
            str(domain.results),
        )
    console.print(table)
    average = "" if result.average_score is None else f"{result.average_score:.3f}"
    console.print(
        f"domains: {len(result.domains)} tasks: {result.total_tasks} "
        f"completed: {result.completed} successes: {result.successes} "
        f"failures: {result.failures} average_score: {average}"
    )

@osworld_app.command("results")
def osworld_results(
    run_id: Optional[str] = typer.Option(
        None, help="Run id to summarize. Defaults to the latest OSWorld run."
    ),
    observation_type: str = typer.Option(
        osworld.DEFAULT_OBSERVATION_TYPE, help="OSWorld observation type used by the run."
    ),
    model: str = typer.Option(osworld.DEFAULT_MODEL, help="OSWorld model used by the run."),
) -> None:
    """Summarize OSWorld result files for a recorded run."""
    try:
        record = osworld.read_run_record(run_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    task_keys = tuple(record.get("tasks", []))
    result_dir = record.get("results")
    observation_type = record.get("observation_type") or observation_type
    model = record.get("model") or model
    summary = osworld.summarize_task_results(
        task_keys, observation_type=observation_type, model=model, result_dir=result_dir
    )
    table = Table(title=f"OSWorld results {record.get('run_id', '')}")
    for column in ("task", "status", "score", "result"):
        table.add_column(column)
    for result in summary.tasks:
        table.add_row(
            result.task,
            result.status,
            "" if result.score is None else str(result.score),
            str(result.result_file),
        )
    console.print(table)
    average = "" if summary.average_score is None else f"{summary.average_score:.3f}"
    console.print(
        f"completed: {summary.completed}/{len(summary.tasks)} "
        f"successes: {summary.successes} failures: {summary.failures} average_score: {average}"
    )
    if result_dir:
        console.print(f"results: {result_dir}")

@osworld_app.command("status")
def osworld_status() -> None:
    """List recorded OSWorld runs."""
    records = osworld.list_run_records()
    table = Table(title="OSWorld runs")
    for column in ("run_id", "status", "command", "tasks", "created_at"):
        table.add_column(column)
    for record in records:
        table.add_row(
            str(record.get("run_id", "")),
            str(record.get("status", "")),
            str(record.get("command", "")),
            str(len(record.get("tasks", []))),
            str(record.get("created_at", "")),
        )
    console.print(table)

@osworld_app.command("smoke")
def osworld_smoke(
    task: str = typer.Option(
        osworld.DEFAULT_OSWORLD_TASK, help="Task formatted as <domain>/<task-id>."
    ),
    suite: Optional[str] = typer.Option(
        None, help="Named OSWorld smoke suite to run, such as 'tiny'."
    ),
    provider: str = typer.Option(
        osworld.DEFAULT_PROVIDER,
        help="OSWorld provider name. Docker is the current local smoke default.",
    ),
    observation_type: str = typer.Option(
        osworld.DEFAULT_OBSERVATION_TYPE, help="OSWorld observation type."
    ),
    model: str = typer.Option(
        osworld.DEFAULT_MODEL, help="OSWorld model argument passed through to upstream run.py."
    ),
    max_steps: int = typer.Option(osworld.DEFAULT_MAX_STEPS, help="Max OSWorld agent steps."),
    max_tokens: int = typer.Option(osworld.DEFAULT_MAX_TOKENS, help="Max model output tokens per OSWorld action request."),
    max_trajectory_length: int = typer.Option(
        1, help="Number of previous OSWorld observations to keep in the upstream prompt history."
    ),
    a11y_tree_max_items: int = typer.Option(
        osworld.DEFAULT_A11Y_TREE_MAX_ITEMS,
        help="Maximum deduped accessibility-tree rows kept before token trimming.",
    ),
    a11y_iou_threshold: float = typer.Option(
        osworld.DEFAULT_A11Y_IOU_THRESHOLD,
        help="IoU threshold for removing duplicate accessibility-tree rows with the same text.",
    ),
    headless: bool = typer.Option(
        True, "--headless/--headed", help="Run OSWorld in headless mode."
    ),
    enable_proxy: bool = typer.Option(
        False,
        "--enable-proxy/--no-enable-proxy",
        help="Enable OSWorld proxy setup for tasks with proxy=true.",
    ),
    proxy_config_file: Optional[str] = typer.Option(
        None,
        help="Proxy JSON path. Defaults to upstream evaluation_examples/settings/proxy/dataimpulse.json.",
    ),
    client_password: str = typer.Option(
        osworld.DEFAULT_CLIENT_PASSWORD,
        help="Ubuntu VM password used by OSWorld proxy setup for sudo operations.",
    ),
    dry_run: bool = typer.Option(
        False, help="Print the upstream OSWorld command without running it."
    ),
) -> None:
    """Run or render a one-task OSWorld smoke through upstream run.py."""
    try:
        metadata = osworld.smoke(
            task=task,
            suite=suite,
            provider=provider,
            observation_type=observation_type,
            model=model,
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_trajectory_length=max_trajectory_length,
            a11y_tree_max_items=a11y_tree_max_items,
            a11y_iou_threshold=a11y_iou_threshold,
            headless=headless,
            enable_proxy=enable_proxy,
            proxy_config_file=proxy_config_file,
            client_password=client_password,
            dry_run=dry_run,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not dry_run:
        task_keys = osworld.tasks_for_suite(suite) if suite else (task,)
        table = Table(title="OSWorld smoke result")
        for column in ("task", "status", "score", "result"):
            table.add_column(column)
        for task_key in task_keys:
            result = osworld.read_task_result(
                task_key, observation_type=observation_type, model=model
            )
            table.add_row(
                result.task,
                result.status,
                "" if result.score is None else str(result.score),
                str(result.result_file),
            )
        console.print(table)
        console.print(f"metadata: {metadata}")
        console.print(f"results: {osworld.results_path()}")

@benchmarks_app.command("list")
def benchmarks_list() -> None:
    """List benchmark targets and the command to run each one."""
    table = Table(title="w8-biayn benchmarks")
    table.add_column("key", no_wrap=True)
    for column in ("tier", "harness", "metric", "command"):
        table.add_column(column)
    for benchmark in benchmarks.list_benchmarks():
        table.add_row(
            benchmark.key,
            benchmark.tier,
            benchmark.harness,
            benchmark.metric,
            benchmark.command,
        )
    console.print(table)


@benchmarks_app.command("show")
def benchmarks_show(key: str = typer.Argument(..., help="Benchmark key.")) -> None:
    """Show why a benchmark exists and how it should be run."""
    try:
        benchmark = benchmarks.get_benchmark(key)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=benchmark.__dict__)


@harbor_app.command("list")
def harbor_list(
    task_root: Optional[str] = typer.Option(None, help="Harbor task root. Defaults to the packaged smoke task root."),
) -> None:
    """List packaged Harbor DOMDiff tasks."""
    table = Table(title="w8-biayn Harbor DOMDiff tasks")
    table.add_column("task_id", no_wrap=True, overflow="ignore")
    for column in ("difficulty", "language", "build", "preview_path"):
        table.add_column(column)
    for task_id in harbor_tasks.discover_task_ids(task_root):
        task = harbor_tasks.load_task(task_id, task_root)
        metadata = task.task_config.get("metadata", {})
        env = task.task_config.get("verifier", {}).get("env", {})
        build = str(env.get("BUILD_COMMAND", "") if isinstance(env, dict) else "")
        table.add_row(
            task.task_id,
            str(metadata.get("difficulty", "")),
            str(metadata.get("language", "")),
            build,
            task.preview_path,
        )
    console.print(table)


@harbor_app.command("validate")
def harbor_validate(
    task: Optional[list[str]] = typer.Option(None, "--task", help="Task id to validate. Repeat to validate multiple tasks."),
    task_root: Optional[str] = typer.Option(None, help="Harbor task root. Defaults to the packaged smoke task root."),
) -> None:
    """Validate packaged Harbor task structure and DOMDiff rubrics."""
    table = Table(title="Harbor validation")
    table.add_column("task_id", no_wrap=True, overflow="ignore")
    for column in ("status", "preview_path"):
        table.add_column(column)
    for item in harbor_tasks.validate_tasks(task_root=task_root, task_ids=task):
        table.add_row(item.task_id, "ok", item.preview_path)
    console.print(table)


@harbor_app.command("prepare-data")
def harbor_prepare_data(
    out: str = typer.Option(..., "--out", help="Output directory for SkyRL train/validation parquet files."),
    task: Optional[list[str]] = typer.Option(None, "--task", help="Task id to include. Repeat to include multiple tasks."),
    task_root: Optional[str] = typer.Option(None, help="Harbor task root. Defaults to the packaged smoke task root."),
    oracle: bool = typer.Option(False, "--oracle/--no-oracle", help="Mark dataset rows to use packaged oracle patches for infrastructure smoke runs."),
) -> None:
    """Prepare packaged Harbor tasks as a SkyRL-Gym dataset."""
    train_path, val_path = prepare_harbor_skyrl_dataset(
        out,
        task_root=task_root,
        task_ids=task,
        oracle=oracle,
    )
    console.print(f"train: {train_path}")
    console.print(f"validation: {val_path}")


@harbor_app.command("oracle-smoke")
def harbor_oracle_smoke(
    task: str = typer.Option(harbor_tasks.DEFAULT_HARBOR_TASK_IDS[0], "--task", help="Task id to run."),
    task_root: Optional[str] = typer.Option(None, help="Harbor task root. Defaults to the packaged smoke task root."),
    chromiumrl_url: Optional[str] = typer.Option(None, help="ChromiumRL reward URL. Omit to run verifier setup without live DOMDiff scoring."),
    keep_container: bool = typer.Option(False, "--keep-container", help="Leave the task container running for debugging."),
    dry_run: bool = typer.Option(False, help="Print Docker build/run/verifier commands without running them."),
) -> None:
    """Run one Harbor task with the packaged oracle patch in Docker."""
    loaded = harbor_tasks.load_task(task, task_root)
    if dry_run:
        runner = HarborDockerTaskRunner(chromiumrl_url=chromiumrl_url, oracle=True, keep_containers=keep_container)
        console.print(runner.dry_run_plan(loaded))
        return
    outcome = run_oracle_smoke(
        loaded.task_id,
        task_root=task_root,
        chromiumrl_url=chromiumrl_url,
        keep_container=keep_container,
    )
    console.print_json(
        data={
            "task_id": outcome.task_id,
            "container": outcome.container,
            "image": outcome.image,
            "preview_url": outcome.preview_url,
            "terminal_state": outcome.terminal_state,
            "reward": outcome.reward_doc,
        }
    )


@app.command()
def status(
    cluster: Optional[str] = typer.Option(None, help="Optional cluster name."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
) -> None:
    """Show SkyPilot status."""
    args = ["sky", "status"]
    if cluster:
        args.append(cluster)
    run_command(args, env=_service_account_env(credentials))


@app.command()
def logs(
    cluster: str = typer.Argument("w8-biayn-miniwob", help="Cluster name."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
) -> None:
    """Tail SkyPilot logs."""
    run_command(["sky", "logs", cluster], env=_service_account_env(credentials))


@app.command()
def down(
    cluster: str = typer.Argument("w8-biayn-miniwob", help="Cluster name."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
) -> None:
    """Tear down a SkyPilot cluster."""
    run_command(["sky", "down", "-y", cluster], env=_service_account_env(credentials))


if __name__ == "__main__":
    app()
