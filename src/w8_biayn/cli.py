"""CLI entrypoint for w8-biayn C++ performance RL."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import benchmarks, upstreams
from .slime_integration.doctor import run_slime_doctor
from .slime_integration.setup import DEFAULT_SLIME_IMAGE, build_slime_setup_plan, write_slime_setup_files
from .constants import (
    CPP_DATA_SCHEMA_VERSION,
    DEFAULT_CPP_CONTAINER_IMAGE,
    DEFAULT_CPP_EVAL_ACCELERATORS,
    DEFAULT_CPP_SMOKE_ACCELERATORS,
    DEFAULT_CPP_TRAIN_ACCELERATORS,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_DATA_ROOT,
    DEFAULT_RENDER_DIR,
    DEFAULT_SKYRL_DATA_DIR,
    SUPERCODER_DATASET,
)
from .cpp_perf.data import (
    PIE_DOWNLOADS,
    SUPERCODER_ALLOW_PATTERNS,
    build_tests_manifest_with_report,
    build_tests_manifest_from_io,
    download_pie,
    download_supercoder,
    discover_problem_tests,
    inspect_supercoder_parquet,
    prepare_full_pie,
    verify_data_manifest,
    write_data_manifest,
)
from .cpp_perf.coverage import measure_cpp_coverage
from .cpp_perf.eval import aggregate_eval_records, compare_eval_summaries, read_jsonl
from .cpp_perf.pie import build_tasks, build_tasks_with_report, load_tests_manifest, read_pie_pairs
from .cpp_perf.reward import compute_reward, extract_reward_code, valid_model_output
from .cpp_perf.sandbox import (
    DEFAULT_CPU,
    DEFAULT_DOCKER_IMAGE,
    build_sandbox_image,
    dry_run_plan,
    runtime_preflight_plan,
    run_in_sandbox,
    run_runtime_preflight,
    sandbox_image_build_plan,
)
from .cpp_perf.schema import CppTask, TestCase
from .cpp_perf.slime_dataset import build_slime_datasets
from .cpp_perf.skyrl_dataset import build_skyrl_datasets
from .gcp_auth import GcpAuthError, check_project_permissions, service_account_env
from .grpo_readiness import build_grpo_readiness, readiness_blocks_launch
from .mlflow_metrics import DEFAULT_METRIC_KEYS, read_mlflow_api, read_mlflow_metrics
from .reporting import build_raw_run_report
from .run_status import PIPELINES as STATUS_PIPELINES
from .run_status import build_run_status
from .secrets import CredentialError, default_bucket_for_project, get_project_id
from .shell import run_command
from .sky_config import (
    GRPO_MIN_SAMPLES_PER_GPU_STEP,
    GRPO_MULTINODE_RESUME_MIN_DISK_GB,
    Pipeline,
    RenderOptions,
    SUPPORTED_TRACKING_BACKENDS,
    write_sky_yaml,
)

app = typer.Typer(help="Command and control for C++ performance RL on rLLM, SkyRL, and GCP.")
upstreams_app = typer.Typer(help="Manage pinned upstream source clones.")
slime_app = typer.Typer(help="Inspect and operate the pinned SLIME upstream sidecar.")
data_app = typer.Typer(help="Download, convert, validate, and cache training datasets.")
data_pie_app = typer.Typer(help="Prepare PIE C++ data.")
data_supercoder_app = typer.Typer(help="Download and inspect SuperCoder reference data.")
data_skyrl_app = typer.Typer(help="Build SkyRL/rLLM dataset files from validated tasks.")
data_slime_app = typer.Typer(help="Build SLIME dataset files from validated C++ tasks.")
data_cache_app = typer.Typer(help="Upload and restore versioned dataset bundles from GCS.")
cpp_app = typer.Typer(help="Build, run, and score C++ performance-RL tasks.")
task_app = typer.Typer(help="Build PIE-derived C++ task JSON.")
harness_app = typer.Typer(help="Run C++ candidates in the sandbox harness.")
reward_app = typer.Typer(help="Score model outputs with the C++ reward function.")
config_app = typer.Typer(help="Render cloud backend configs.")
benchmarks_app = typer.Typer(help="Inspect the C++ performance-RL benchmark ladder.")
gcp_app = typer.Typer(help="Run-scoped GCP cleanup helpers.")
ops_app = typer.Typer(help="Inspect and manage cloud runs without calling the backend directly.")
eval_app = typer.Typer(help="Aggregate C++ performance-RL evaluation outputs.")

app.add_typer(upstreams_app, name="upstreams")
app.add_typer(slime_app, name="slime")
app.add_typer(data_app, name="data")
data_app.add_typer(data_pie_app, name="pie")
data_app.add_typer(data_supercoder_app, name="supercoder")
data_app.add_typer(data_skyrl_app, name="skyrl")
data_app.add_typer(data_slime_app, name="slime")
data_app.add_typer(data_cache_app, name="cache")
app.add_typer(cpp_app, name="cpp")
cpp_app.add_typer(task_app, name="task")
cpp_app.add_typer(harness_app, name="harness")
cpp_app.add_typer(reward_app, name="reward")
app.add_typer(config_app, name="config")
app.add_typer(benchmarks_app, name="benchmarks")
app.add_typer(gcp_app, name="gcp")
app.add_typer(ops_app, name="ops")
app.add_typer(eval_app, name="eval")
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


def _backend_env(credentials: str) -> dict[str, str]:
    project_id = _project_id(credentials)
    return _service_account_env(credentials, project_id=project_id)


def _run_backend_command(args: list[str], *, credentials: str, dry_run: bool = False) -> None:
    run_command(args, env=_backend_env(credentials), dry_run=dry_run)


def _data_gcs_prefix(credentials: str, override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    project_id = _project_id(credentials)
    return f"{default_bucket_for_project(project_id)}/datasets/cpp-perf/{CPP_DATA_SCHEMA_VERSION}/skyrl"


def _bucket_uri_from_gcs_prefix(prefix: str) -> str:
    if not prefix.startswith("gs://"):
        raise typer.BadParameter(f"GCS prefix must start with gs://, got {prefix}")
    parts = prefix.removeprefix("gs://").split("/", 1)
    if not parts[0]:
        raise typer.BadParameter(f"GCS prefix must include a bucket name: {prefix}")
    return f"gs://{parts[0]}"


def _build_tasks_command(
    *,
    pie_rows: str,
    tests_json: str,
    out: str,
    split: str,
    limit: Optional[int],
    compiler_flags: str,
) -> None:
    pairs = read_pie_pairs(pie_rows, limit=limit)
    manifest = load_tests_manifest(tests_json)
    tasks = build_tasks(pairs, manifest, split=split, compiler_flags=compiler_flags)  # type: ignore[arg-type]
    output = Path(out)
    for task in tasks:
        task.write_json(output / f"{task.task_id}.json")
    console.print(f"wrote_tasks: {len(tasks)}")
    console.print(f"out: {output}")


def _read_split_pairs(prepared_root: str | Path, split: str, *, limit: Optional[int]) -> list:
    base = Path(prepared_root)
    for suffix in (".jsonl", ".json", ".tsv"):
        path = base / f"{split}{suffix}"
        if path.exists():
            return read_pie_pairs(path, limit=limit, skip_invalid=True)
    raise typer.BadParameter(f"missing prepared split file for {split} under {base}")


def _write_json(path: str | Path, payload: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _write_json_atomic(path: str | Path, payload: object) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f"{output.name}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(output)
    return output


def _new_run_id() -> str:
    return time.strftime("r%Y%m%d%H%M%S", time.gmtime())


def _require_launch_labels(options: RenderOptions) -> None:
    labels = options.labels
    required = {"project", "phase", "pipeline", "run_id", "owner", "ttl"}
    missing = sorted(key for key in required if not labels.get(key))
    if missing:
        raise typer.BadParameter(f"launch labels missing required keys: {', '.join(missing)}")


def _require_grpo_multinode_utilization(options: RenderOptions) -> None:
    if (
        options.pipeline != "cpp-grpo"
        or options.num_nodes <= 1
        or options.allow_low_multinode_utilization
        or options.grpo_multinode_utilization_ok
    ):
        return
    failures = []
    if options.effective_samples_per_step < options.min_multinode_effective_samples_per_step:
        failures.append(
            "effective samples per step "
            f"{options.effective_samples_per_step} < {options.min_multinode_effective_samples_per_step} "
            f"({GRPO_MIN_SAMPLES_PER_GPU_STEP} samples/GPU/step across {options.total_gpu_count} GPUs)"
        )
    if options.max_env_workers < options.effective_samples_per_step:
        failures.append(
            f"max env workers {options.max_env_workers} < effective samples per step "
            f"{options.effective_samples_per_step}"
        )
    hint = (
        "For 2x[A100:8], use for example "
        "--train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256, "
        "or pass --allow-low-multinode-utilization for an intentional experiment."
    )
    raise typer.BadParameter("; ".join(failures) + f". {hint}")


def _require_grpo_multinode_resume_disk(options: RenderOptions) -> None:
    if options.pipeline != "cpp-grpo" or options.num_nodes <= 1 or not options.resume_from.strip():
        return
    if options.effective_disk_size >= GRPO_MULTINODE_RESUME_MIN_DISK_GB:
        return
    raise typer.BadParameter(
        "multi-node GRPO resume requires "
        f"--disk-size {GRPO_MULTINODE_RESUME_MIN_DISK_GB} or larger; 1024 GB failed during "
        "FSDP checkpoint restore with [Errno 28] No space left on device."
    )


def _normalize_tracking_backends(backends: Optional[list[str]]) -> tuple[str, ...] | None:
    if not backends:
        return None
    normalized: list[str] = []
    for backend in backends:
        value = backend.strip().lower()
        if value not in SUPPORTED_TRACKING_BACKENDS:
            raise typer.BadParameter(
                f"unknown tracking backend {backend!r}; supported: {', '.join(SUPPORTED_TRACKING_BACKENDS)}"
            )
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _require_grpo_readiness(rendered_config_path: str | Path) -> None:
    payload = build_grpo_readiness(rendered_config_path)
    if readiness_blocks_launch(payload):
        console.print_json(data=payload)
        raise typer.Exit(1)


@app.command()
def doctor(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    cloud: bool = typer.Option(False, help="Run GCP/cloud backend checks."),
    cpp_perf: bool = typer.Option(False, "--cpp-perf", help="Check C++ performance-harness prerequisites."),
    preflight: bool = typer.Option(True, "--preflight/--no-preflight", help="Run the C++ runtime preflight."),
    image: str = typer.Option(DEFAULT_DOCKER_IMAGE, help="Docker image used for the C++ runtime preflight."),
    cpu: str = typer.Option(DEFAULT_CPU, help="CPU id used for taskset in the C++ runtime preflight."),
    build_image: bool = typer.Option(True, "--build-image/--no-build-image", help="Build the default runtime sandbox image before preflight."),
) -> None:
    """Validate local prerequisites without printing secrets."""

    table = Table(title="w8-biayn doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    tools = ["uv", "git"]
    if cpp_perf:
        tools.extend(["docker", "g++", "python3", "taskset", "timeout"])
    if cloud:
        tools.extend(["gcloud", "sky"])
    for tool in tools:
        found = shutil.which(tool)
        table.add_row(tool, "ok" if found else "missing", found or "not on PATH")

    preflight_failed = False
    if cpp_perf:
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if governor.exists():
            value = governor.read_text(encoding="utf-8").strip()
            table.add_row("cpu governor", "ok" if value == "performance" else "warn", value)
        else:
            table.add_row("cpu governor", "warn", "cpufreq governor file not found")
        if preflight:
            if shutil.which("docker"):
                build_ok = True
                if build_image:
                    build = build_sandbox_image(image=image)
                    build_ok = build.returncode == 0
                    table.add_row(
                        "sandbox image build",
                        "ok" if build_ok else "error",
                        image if build_ok else (build.stderr or build.stdout)[-500:],
                    )
                if build_ok:
                    result = run_runtime_preflight(image=image, cpu=cpu)
                    status = "ok" if result.ok else "error"
                    detail = (
                        f"{result.reason}; runtime_cpu_ns={result.runtime_cpu_ns}; "
                        f"runtime_wall_ns={result.runtime_wall_ns}; image={image}; cpu={cpu}"
                    )
                    table.add_row("runtime preflight", status, detail)
                    preflight_failed = not result.ok
                else:
                    preflight_failed = True
            else:
                table.add_row("runtime preflight", "error", "docker missing")
                preflight_failed = True

    try:
        project_id = get_project_id(credentials)
        table.add_row("credentials", "ok", f"{credentials} project={project_id}")
        table.add_row("artifact bucket", "default", default_bucket_for_project(project_id))
    except CredentialError as exc:
        table.add_row("credentials", "error", str(exc))
        project_id = ""

    console.print(table)
    if preflight_failed:
        raise typer.Exit(1)

    if cloud:
        project_id = project_id or _project_id(credentials)
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
                "[red]GCP service account lacks cloud backend launch permissions: "
                + ", ".join(missing_launch_permissions)
                + "[/red]"
            )
            raise typer.Exit(1)
        if not sky_gcp_enabled:
            console.print("[red]GCP is not enabled for the cloud backend. Fix the IAM errors above and rerun doctor.[/red]")
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


@slime_app.command("setup")
def slime_setup(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root containing .cache/upstreams/slime.",
    ),
    image: str = typer.Option(
        DEFAULT_SLIME_IMAGE,
        "--image",
        help="Docker image to use for the SLIME quick-start container.",
    ),
    container_name: str = typer.Option(
        "w8-slime",
        "--container-name",
        help="Container name for the generated Docker launcher.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite generated launcher/bootstrap scripts if they already exist.",
    ),
) -> None:
    """Generate Docker-first SLIME quick-start scripts for this repo checkout."""

    upstreams.clone_or_update("slime", repo_root=repo_root)
    plan = build_slime_setup_plan(repo_root, image=image, container_name=container_name)
    launcher_path, bootstrap_path = write_slime_setup_files(plan, force=force)

    report = run_slime_doctor(repo_root)
    if not report.ok:
        console.print("slime_clone_status: missing_or_incomplete")
        if report.next_action:
            console.print(f"next_action: {report.next_action}")
        raise typer.Exit(1)

    console.print("SLIME setup files generated")
    console.print(f"slime_root: {report.root}")
    console.print(f"docker_image: {plan.image}")
    console.print(f"repo_mount: {plan.repo_mount}")
    console.print(f"launcher: {launcher_path}")
    console.print(f"bootstrap: {bootstrap_path}")
    console.print("next_steps:")
    console.print(f"  1. run {launcher_path}")
    console.print("     this does docker pull + docker run + in-container bootstrap")
    console.print(f"  2. optional manual in-container rerun: {plan.bootstrap_command}")


@slime_app.command("doctor")
def slime_doctor(
    repo_root: Path = typer.Option(
        Path("."),
        "--repo-root",
        help="Repository root containing .cache/upstreams/slime.",
    ),
) -> None:
    """Validate the pinned SLIME upstream clone and top-level layout."""

    report = run_slime_doctor(repo_root)
    table = Table(title="SLIME upstream doctor")
    table.add_column("Path")
    table.add_column("Status")
    table.add_column("Detail")
    for check in report.checks:
        table.add_row(check.path, check.status, check.detail)
    console.print(table)
    console.print(f"slime_root: {report.root}")
    if not report.ok:
        if report.next_action:
            console.print(f"next_action: {report.next_action}")
        raise typer.Exit(1)


@data_app.command("doctor")
def data_doctor() -> None:
    """Validate local dataset-conversion tooling."""

    table = Table(title="w8-biayn data doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for module_name in ("pyarrow", "datasets", "huggingface_hub", "gdown"):
        try:
            module = __import__(module_name)
            detail = getattr(module, "__version__", "installed")
            table.add_row(module_name, "ok", str(detail))
        except ImportError:
            table.add_row(module_name, "missing", f"run ./scripts/bootstrap.sh to install {module_name}")
    for tool in ("gcloud", "git", "g++", "gcov", "timeout"):
        found = shutil.which(tool)
        table.add_row(tool, "ok" if found else "missing", found or "not on PATH")
    table.add_row("schema", "ok", CPP_DATA_SCHEMA_VERSION)
    console.print(table)


@data_pie_app.command("download")
def data_pie_download(
    out: str = typer.Option(f"{DEFAULT_DATA_ROOT}/pie", "--out", help="Local PIE data root."),
    force: bool = typer.Option(False, help="Re-download files that already exist."),
    dry_run: bool = typer.Option(False, help="Print source URLs and destinations without downloading."),
) -> None:
    """Download PIE source archives into the project data cache."""

    if dry_run:
        for target in PIE_DOWNLOADS:
            console.print(f"{target.name}: {target.url} -> {Path(out) / target.relative_path}")
        return
    written = download_pie(out, force=force)
    for path in written:
        console.print(str(path))


@data_pie_app.command("prepare-full")
def data_pie_prepare_full(
    source_root: str = typer.Option(f"{DEFAULT_DATA_ROOT}/pie", "--source-root", help="PIE download root."),
    out: str = typer.Option(f"{DEFAULT_DATA_ROOT}/pie-full", "--out", help="Prepared full PIE root."),
    force: bool = typer.Option(False, help="Replace an existing prepared tree."),
) -> None:
    """Unpack official PIE archives into normalized full-run splits and cases."""

    result = prepare_full_pie(source_root, out, force=force)
    console.print_json(
        data={
            "root": str(result["root"]),
            "manifest": str(result["manifest"]),
            "splits": {key: str(path) for key, path in result["splits"].items()},
            "problem_count_with_tests": result["problem_count_with_tests"],
            "test_count": result["test_count"],
        }
    )


@data_pie_app.command("measure-coverage")
def data_pie_measure_coverage(
    prepared_root: str = typer.Option(f"{DEFAULT_DATA_ROOT}/pie-full", "--prepared-root", help="Prepared PIE root."),
    out: str = typer.Option(f"{DEFAULT_DATA_ROOT}/pie-full/coverage.json", "--out", help="Coverage JSON path."),
    report_out: str = typer.Option(
        f"{DEFAULT_DATA_ROOT}/pie-full/coverage-report.json",
        "--report-out",
        help="Detailed coverage admission report.",
    ),
    split: Optional[list[str]] = typer.Option(
        None,
        "--split",
        help="Prepared split to inspect. Repeatable. Defaults to train, validation, test.",
    ),
    limit_per_split: Optional[int] = typer.Option(None, help="Optional row limit per split."),
    timeout_s: int = typer.Option(5, help="Timeout per oracle test run."),
    jobs: int = typer.Option(1, help="Parallel coverage workers."),
    checkpoint_every: int = typer.Option(25, help="Write progress after this many completed problems."),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Reuse completed problems from an existing report."),
) -> None:
    """Measure oracle gcov coverage for prepared PIE problems."""

    base = Path(prepared_root)
    splits = split or ["train", "validation", "test"]
    unique_pairs = {}
    for split_name in splits:
        pairs = _read_split_pairs(base, split_name, limit=limit_per_split)
        for pair in pairs:
            unique_pairs.setdefault(pair.problem_id, pair)
    report: dict[str, object] = {"splits": splits, "problems": {}, "accepted": 0, "rejected": 0}
    if resume and Path(report_out).exists():
        loaded_report = json.loads(Path(report_out).read_text(encoding="utf-8"))
        if isinstance(loaded_report, dict) and isinstance(loaded_report.get("problems"), dict):
            completed = {
                problem_id: payload
                for problem_id, payload in loaded_report["problems"].items()
                if problem_id in unique_pairs and isinstance(payload, dict)
            }
            report["problems"] = completed
            console.print(f"resuming_completed: {len(completed)}")

    def rebuild_coverage() -> dict[str, dict[str, float]]:
        rebuilt: dict[str, dict[str, float]] = {}
        problems = report["problems"]
        if not isinstance(problems, dict):
            return rebuilt
        for problem_id, payload in problems.items():
            if not isinstance(payload, dict) or not payload.get("ok") or not isinstance(payload.get("coverage"), dict):
                continue
            rebuilt[str(problem_id)] = payload["coverage"]  # type: ignore[assignment]
        report["accepted"] = len(rebuilt)
        report["rejected"] = len(problems) - len(rebuilt)
        report["completed"] = len(problems)
        report["total"] = len(unique_pairs)
        return rebuilt

    def checkpoint() -> None:
        current_coverage = rebuild_coverage()
        _write_json_atomic(out, current_coverage)
        _write_json_atomic(report_out, report)

    def measure_one(problem_id: str):
        pair = unique_pairs[problem_id]
        tests = [TestCase.model_validate(item) for item in discover_problem_tests(base / "cases" / problem_id)]
        return problem_id, measure_cpp_coverage(pair.oracle_solution, tests, timeout_s=timeout_s)

    checkpoint_interval = max(1, checkpoint_every)
    pending = [problem_id for problem_id in sorted(unique_pairs) if problem_id not in report["problems"]]
    checkpoint()
    console.print(f"coverage_total: {len(unique_pairs)}")
    console.print(f"coverage_pending: {len(pending)}")
    worker_count = max(1, jobs)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(measure_one, problem_id): problem_id for problem_id in pending}
        completed_since_checkpoint = 0
        for future in as_completed(futures):
            submitted_problem_id = futures[future]
            try:
                problem_id, measurement = future.result()
                problem_report = measurement.as_dict()
            except Exception as exc:  # pragma: no cover - defensive for messy upstream data
                problem_id = submitted_problem_id
                problem_report = {
                    "coverage": None,
                    "tests_total": 0,
                    "tests_passed": 0,
                    "ok": False,
                    "reason": "worker_failed",
                    "logs": repr(exc),
                }
            report["problems"][problem_id] = problem_report  # type: ignore[index]
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_interval:
                checkpoint()
                completed_since_checkpoint = 0
                console.print(
                    "coverage_progress: "
                    f"{report['completed']}/{report['total']} "
                    f"accepted={report['accepted']} rejected={report['rejected']}"
                )
    checkpoint()
    console.print_json(
        data={
            "accepted": report["accepted"],
            "rejected": report["rejected"],
            "completed": report["completed"],
            "total": report["total"],
            "out": out,
            "report_out": report_out,
        }
    )


@data_pie_app.command("build-tests-manifest")
def data_pie_build_tests_manifest(
    inputs_outputs_basepath: str = typer.Option(..., "--inputs-outputs-basepath", help="Directory keyed by problem id."),
    coverage_json: str = typer.Option(..., "--coverage-json", help="Problem-id keyed coverage JSON."),
    out: str = typer.Option(..., "--out", help="Output tests manifest JSON path."),
    problem_id: Optional[list[str]] = typer.Option(None, "--problem-id", help="Problem id to include. Repeatable."),
    problem_ids_json: Optional[str] = typer.Option(None, "--problem-ids-json", help="JSON list of problem ids."),
    visible_count: int = typer.Option(1, help="Visible tests per problem."),
    hidden_count: int = typer.Option(1, help="Hidden tests per problem."),
) -> None:
    """Build a task tests manifest from PIE/CodeNet input-output files."""

    selected: list[str] = list(problem_id or [])
    if problem_ids_json:
        loaded_ids = json.loads(Path(problem_ids_json).read_text(encoding="utf-8"))
        if not isinstance(loaded_ids, list):
            raise typer.BadParameter("--problem-ids-json must contain a JSON list")
        selected.extend(str(item) for item in loaded_ids)
    if not selected:
        selected = [path.name for path in sorted(Path(inputs_outputs_basepath).iterdir()) if path.is_dir()]
    coverage = json.loads(Path(coverage_json).read_text(encoding="utf-8"))
    manifest = build_tests_manifest_from_io(
        selected,
        inputs_outputs_basepath,
        coverage,
        visible_count=visible_count,
        hidden_count=hidden_count,
    )
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(f"wrote_problems: {len(manifest)}")
    console.print(f"out: {output}")


@data_pie_app.command("build-full-tasks")
def data_pie_build_full_tasks(
    prepared_root: str = typer.Option(f"{DEFAULT_DATA_ROOT}/pie-full", "--prepared-root", help="Prepared PIE root."),
    coverage_json: str = typer.Option(
        f"{DEFAULT_DATA_ROOT}/pie-full/coverage.json",
        "--coverage-json",
        help="Problem-id keyed measured coverage JSON.",
    ),
    out: str = typer.Option(f"{DEFAULT_DATA_ROOT}/tasks-full", "--out", help="Output task JSON root."),
    visible_count: int = typer.Option(1, help="Visible tests per problem."),
    hidden_count: int = typer.Option(1, help="Hidden tests per problem."),
    min_train: int = typer.Option(1000, help="Minimum train tasks for a full official run."),
    min_validation: int = typer.Option(100, help="Minimum validation tasks for a full official run."),
    min_test: int = typer.Option(100, help="Minimum test tasks for a full official run."),
    limit_per_split: Optional[int] = typer.Option(None, help="Optional row limit per split."),
    compiler_flags: str = typer.Option("-O3 -std=c++20", help="Compiler flags recorded in each task."),
    force: bool = typer.Option(False, help="Replace an existing task tree."),
) -> None:
    """Build full official PIE task JSON with count and coverage gates."""

    base = Path(prepared_root)
    output = Path(out)
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    coverage = json.loads(Path(coverage_json).read_text(encoding="utf-8"))
    report: dict[str, object] = {
        "prepared_root": str(base),
        "coverage_json": coverage_json,
        "visible_count": visible_count,
        "hidden_count": hidden_count,
        "splits": {},
    }
    counts: dict[str, int] = {}
    for split_name in ("train", "validation", "test"):
        pairs = _read_split_pairs(base, split_name, limit=limit_per_split)
        problem_ids = [pair.problem_id for pair in pairs]
        tests_manifest, tests_report = build_tests_manifest_with_report(
            problem_ids,
            base / "cases",
            coverage,
            visible_count=visible_count,
            hidden_count=hidden_count,
        )
        tasks, task_report = build_tasks_with_report(
            pairs,
            tests_manifest,
            split=split_name,  # type: ignore[arg-type]
            task_id_prefix=f"pie_cpp_{split_name}",
            compiler_flags=compiler_flags,
        )
        split_out = output / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            task.write_json(split_out / f"{task.task_id}.json")
        counts[split_name] = len(tasks)
        report["splits"][split_name] = {  # type: ignore[index]
            "tests": tests_report,
            "tasks": task_report.as_dict(),
        }

    report["counts"] = counts
    report_path = _write_json(output / "_w8_task_build_report.json", report)
    manifest_path = write_data_manifest(
        output,
        kind="pie-full-tasks",
        sources=[str(base), coverage_json],
        options={
            "visible_count": visible_count,
            "hidden_count": hidden_count,
            "min_train": min_train,
            "min_validation": min_validation,
            "min_test": min_test,
            "counts": counts,
            "report": str(report_path),
        },
    )
    console.print_json(data={"counts": counts, "report": str(report_path), "manifest": str(manifest_path)})
    if counts.get("train", 0) < min_train:
        raise typer.BadParameter(f"full-run train gate failed: {counts.get('train', 0)} < {min_train}")
    if counts.get("validation", 0) < min_validation:
        raise typer.BadParameter(
            f"full-run validation gate failed: {counts.get('validation', 0)} < {min_validation}"
        )
    if counts.get("test", 0) < min_test:
        raise typer.BadParameter(f"full-run test gate failed: {counts.get('test', 0)} < {min_test}")


@data_pie_app.command("build-tasks")
def data_pie_build_tasks(
    pairs: str = typer.Option(..., "--pairs", help="PIE TSV or JSONL rows."),
    tests_json: str = typer.Option(..., "--tests-json", help="Problem-id keyed tests/coverage manifest."),
    out: str = typer.Option(..., "--out", help="Output directory for task JSON files."),
    split: str = typer.Option("train", help="Task split to write: train, test, or validation."),
    limit: Optional[int] = typer.Option(None, help="Optional maximum PIE rows to inspect."),
    compiler_flags: str = typer.Option("-O3 -std=c++20", help="Compiler flags recorded in each task."),
) -> None:
    """Build validated task JSON from PIE rows and a tests manifest."""

    _build_tasks_command(
        pie_rows=pairs,
        tests_json=tests_json,
        out=out,
        split=split,
        limit=limit,
        compiler_flags=compiler_flags,
    )


@data_skyrl_app.command("build")
def data_skyrl_build(
    tasks_dir: str = typer.Option(..., "--tasks-dir", help="Directory containing validated task JSON."),
    out: str = typer.Option(DEFAULT_SKYRL_DATA_DIR, "--out", help="Output SkyRL dataset bundle directory."),
    profile: str = typer.Option("smoke", help="Dataset profile recorded in the manifest."),
    run_id: Optional[str] = typer.Option(None, help="Run id recorded in the manifest."),
    min_train_tasks: int = typer.Option(1, help="Minimum train tasks required."),
    min_validation_tasks: int = typer.Option(1, help="Minimum validation/test tasks required."),
) -> None:
    """Build SkyRL GRPO parquet and SFT JSONL from task JSON."""

    written = build_skyrl_datasets(
        tasks_dir,
        out,
        profile=profile,
        run_id=run_id,
        min_train_tasks=min_train_tasks,
        min_validation_tasks=min_validation_tasks,
    )
    for key, path in written.items():
        console.print(f"{key}: {path}")


@data_slime_app.command("build")
def data_slime_build(
    tasks_dir: str = typer.Option(..., "--tasks-dir", help="Directory containing validated task JSON."),
    out: str = typer.Option(
        f"{DEFAULT_DATA_ROOT}/slime-pie",
        "--out",
        help="Output SLIME dataset bundle directory.",
    ),
    profile: str = typer.Option("smoke", help="Dataset profile recorded in the manifest."),
    run_id: Optional[str] = typer.Option(None, help="Run id recorded in the manifest."),
    min_train_tasks: int = typer.Option(1, help="Minimum train tasks required."),
    min_validation_tasks: int = typer.Option(1, help="Minimum validation/test tasks required."),
    absolute_task_paths: bool = typer.Option(
        False,
        "--absolute-task-paths",
        help="Write absolute task paths into metadata instead of relocatable bundle-relative paths.",
    ),
) -> None:
    """Build SLIME JSONL prompt files from task JSON."""

    written = build_slime_datasets(
        tasks_dir,
        out,
        profile=profile,
        run_id=run_id,
        min_train_tasks=min_train_tasks,
        min_validation_tasks=min_validation_tasks,
        absolute_task_paths=absolute_task_paths,
    )
    for key, path in written.items():
        console.print(f"{key}: {path}")


@data_supercoder_app.command("download")
def data_supercoder_download(
    out: str = typer.Option(f"{DEFAULT_DATA_ROOT}/supercoder", "--out", help="Local SuperCoder data root."),
    dataset: str = typer.Option(SUPERCODER_DATASET, help="Hugging Face dataset id."),
    force: bool = typer.Option(False, help="Re-download resolved files."),
    dry_run: bool = typer.Option(False, help="Print Hugging Face dataset/patterns without downloading."),
) -> None:
    """Download SuperCoder reference parquet splits for study/eval conversion."""

    if dry_run:
        console.print(f"dataset: {dataset}")
        console.print(f"out: {out}")
        console.print(f"allow_patterns: {list(SUPERCODER_ALLOW_PATTERNS)}")
        return
    path = download_supercoder(out, dataset=dataset, force=force)
    console.print(str(path))


@data_supercoder_app.command("inspect")
def data_supercoder_inspect(
    parquet: str = typer.Option(..., "--parquet", help="Resolved SuperCoder parquet file."),
) -> None:
    """Inspect a SuperCoder parquet schema and row count."""

    console.print_json(data=inspect_supercoder_parquet(parquet))


@data_cache_app.command("upload")
def data_cache_upload(
    path: str = typer.Option(DEFAULT_SKYRL_DATA_DIR, "--path", help="Local data bundle directory."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    gcs_prefix: Optional[str] = typer.Option(None, "--gcs-prefix", help="Destination GCS prefix."),
    dry_run: bool = typer.Option(False, help="Print gcloud command without uploading."),
) -> None:
    """Upload a versioned dataset bundle to GCS using scoped credentials."""

    local_path = Path(path)
    if not dry_run:
        errors = verify_data_manifest(local_path)
        if errors:
            raise typer.BadParameter("; ".join(errors))
    prefix = _data_gcs_prefix(credentials, gcs_prefix)
    env = _service_account_env(credentials)
    bucket_uri = _bucket_uri_from_gcs_prefix(prefix)
    if dry_run:
        run_command(["gcloud", "storage", "buckets", "describe", bucket_uri], env=env, dry_run=True)
        run_command(
            ["gcloud", "storage", "buckets", "create", bucket_uri, "--project", _project_id(credentials)],
            env=env,
            dry_run=True,
        )
    else:
        describe = subprocess.run(
            ["gcloud", "storage", "buckets", "describe", bucket_uri],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if describe.returncode != 0:
            run_command(["gcloud", "storage", "buckets", "create", bucket_uri, "--project", _project_id(credentials)], env=env)
    run_command(["gcloud", "storage", "rsync", "--recursive", str(local_path), prefix], env=env, dry_run=dry_run)


@data_cache_app.command("restore")
def data_cache_restore(
    path: str = typer.Option(DEFAULT_SKYRL_DATA_DIR, "--path", help="Local restore directory."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    gcs_prefix: Optional[str] = typer.Option(None, "--gcs-prefix", help="Source GCS prefix."),
    dry_run: bool = typer.Option(False, help="Print gcloud command without restoring."),
) -> None:
    """Restore and verify a versioned dataset bundle from GCS."""

    prefix = _data_gcs_prefix(credentials, gcs_prefix)
    local_path = Path(path)
    local_path.mkdir(parents=True, exist_ok=True)
    env = _service_account_env(credentials)
    run_command(["gcloud", "storage", "rsync", "--recursive", prefix, str(local_path)], env=env, dry_run=dry_run)
    if not dry_run:
        errors = verify_data_manifest(local_path)
        if errors:
            raise typer.BadParameter("; ".join(errors))
        console.print("manifest: ok")


@task_app.command("build")
def cpp_task_build(
    pie_tsv: str = typer.Option(..., "--pie-tsv", help="PIE trajectory TSV to parse."),
    tests_json: str = typer.Option(..., "--tests-json", help="Problem-id keyed tests/coverage manifest."),
    out: str = typer.Option(..., "--out", help="Output directory for task JSON files."),
    split: str = typer.Option("train", help="Task split to write: train, test, or validation."),
    limit: Optional[int] = typer.Option(None, help="Optional maximum PIE rows to inspect."),
    compiler_flags: str = typer.Option("-O3 -std=c++20", help="Compiler flags recorded in each task."),
) -> None:
    """Build validated PIE-derived C++ task JSON files."""

    _build_tasks_command(
        pie_rows=pie_tsv,
        tests_json=tests_json,
        out=out,
        split=split,
        limit=limit,
        compiler_flags=compiler_flags,
    )


@harness_app.command("run")
def cpp_harness_run(
    task: str = typer.Option(..., "--task", help="Task JSON path."),
    candidate: str = typer.Option(..., "--candidate", help="Candidate C++ file path."),
    image: str = typer.Option(DEFAULT_DOCKER_IMAGE, help="Docker image containing g++, python3, taskset, and bash."),
    cpu: str = typer.Option(DEFAULT_CPU, help="Isolated CPU id used for taskset."),
    work_dir: Optional[str] = typer.Option(None, help="Optional scratch directory to keep logs/artifacts."),
    dry_run: bool = typer.Option(False, help="Print sandbox commands without running Docker."),
) -> None:
    """Compile, test, and measure a candidate."""

    loaded_task = CppTask.read_json(task)
    if dry_run:
        console.print(dry_run_plan(loaded_task, image=image, cpu=cpu), markup=False, soft_wrap=True)
        return
    code = Path(candidate).read_text(encoding="utf-8")
    result = run_in_sandbox(loaded_task, code, image=image, cpu=cpu, work_dir=work_dir)
    console.print_json(data=result.model_dump())


@harness_app.command("preflight")
def cpp_harness_preflight(
    image: str = typer.Option(DEFAULT_DOCKER_IMAGE, help="Docker image containing g++, python3, taskset, and bash."),
    cpu: str = typer.Option(DEFAULT_CPU, help="CPU id used for taskset."),
    work_dir: Optional[str] = typer.Option(None, help="Optional scratch directory to keep logs/artifacts."),
    build_image: bool = typer.Option(True, "--build-image/--no-build-image", help="Build the default runtime sandbox image before running."),
    dry_run: bool = typer.Option(False, help="Print the sandbox preflight command without running Docker."),
) -> None:
    """Verify that the sandbox can compile, run, and measure CPU time."""

    if dry_run:
        if build_image:
            console.print(sandbox_image_build_plan(image=image), markup=False, soft_wrap=True)
        console.print(runtime_preflight_plan(image=image, cpu=cpu), markup=False, soft_wrap=True)
        return
    if build_image:
        build = build_sandbox_image(image=image)
        if build.returncode != 0:
            console.print_json(
                data={
                    "ok": False,
                    "reason": "sandbox_image_build_failed",
                    "image": image,
                    "logs": build.stderr or build.stdout,
                }
            )
            raise typer.Exit(1)
    result = run_runtime_preflight(image=image, cpu=cpu, work_dir=work_dir)
    console.print_json(data=result.as_dict())
    if not result.ok:
        raise typer.Exit(1)


@reward_app.command("score")
def cpp_reward_score(
    task: str = typer.Option(..., "--task", help="Task JSON path."),
    model_output: str = typer.Option(..., "--model-output", help="Model output Markdown path."),
    image: str = typer.Option(DEFAULT_DOCKER_IMAGE, help="Docker image for harness execution."),
    cpu: str = typer.Option(DEFAULT_CPU, help="Isolated CPU id used for taskset."),
    dry_run: bool = typer.Option(False, help="Validate format and print harness commands without scoring."),
) -> None:
    """Score one model output with the correctness-gated efficiency reward."""

    loaded_task = CppTask.read_json(task)
    output_text = Path(model_output).read_text(encoding="utf-8")
    if dry_run:
        valid = valid_model_output(output_text)
        console.print(f"valid_format: {valid}")
        try:
            code, _format_valid = extract_reward_code(output_text)
        except ValueError:
            code = None
        if code is not None:
            console.print(f"candidate_bytes: {len(code.encode('utf-8'))}")
            console.print(dry_run_plan(loaded_task, image=image, cpu=cpu), markup=False, soft_wrap=True)
        return

    def runner(candidate_task: CppTask, code: str):
        return run_in_sandbox(candidate_task, code, image=image, cpu=cpu)

    breakdown = compute_reward(loaded_task, output_text, runner=runner)
    payload = {"reward": breakdown.reward, "reason": breakdown.reason}
    if breakdown.harness is not None:
        payload["harness"] = breakdown.harness.model_dump()
    console.print_json(data=payload)


@config_app.command("render")
def config_render(
    pipeline: Pipeline = typer.Argument(..., help="Pipeline to render: cpp-smoke, cpp-sft, cpp-grpo, or cpp-eval."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output YAML path."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI."),
    accelerators: Optional[str] = typer.Option(None, help="Cloud backend accelerator request."),
    num_nodes: int = typer.Option(1, help="Cloud backend node count."),
    region: str = typer.Option("", "--region", help="Optional GCP region to pin in rendered SkyPilot infra."),
    zone: str = typer.Option("", "--zone", help="Optional GCP zone to pin in rendered SkyPilot infra."),
    disk_size: Optional[int] = typer.Option(None, help="Cloud backend boot disk size in GB."),
    cluster: Optional[str] = typer.Option(None, help="Cloud backend cluster name."),
    model: Optional[str] = typer.Option(None, help="Open model to load/train."),
    gpu_container_image: str = typer.Option(DEFAULT_CPP_CONTAINER_IMAGE, help="GPU Docker image for the smoke."),
    dataset_gcs_prefix: Optional[str] = typer.Option(None, help="Versioned SkyRL dataset GCS prefix."),
    train_batch_size: int = typer.Option(16, help="Training batch size for cpp-sft/cpp-grpo."),
    micro_train_batch_size_per_gpu: int = typer.Option(
        1,
        "--micro-train-batch-size-per-gpu",
        help="Per-GPU micro train batch for cpp-sft/cpp-grpo; try 2 only as a memory-checked tuning experiment.",
    ),
    grpo_use_kl_loss: bool = typer.Option(
        True,
        "--grpo-use-kl-loss/--no-grpo-use-kl-loss",
        help="Apply a KL loss anchor during cpp-grpo to reduce policy drift from the SFT/reference model.",
    ),
    grpo_kl_loss_coef: float = typer.Option(0.001, help="KL loss coefficient for cpp-grpo."),
    grpo_use_entropy_loss: bool = typer.Option(
        True,
        "--grpo-use-entropy-loss/--no-grpo-use-entropy-loss",
        help="Apply an entropy bonus during cpp-grpo to reduce deterministic policy collapse.",
    ),
    grpo_entropy_loss_coef: float = typer.Option(0.001, help="Entropy loss coefficient for cpp-grpo."),
    grpo_vllm_gpu_memory_utilization: float = typer.Option(
        0.7,
        "--grpo-vllm-gpu-memory-utilization",
        help="vLLM GPU memory reservation for colocated cpp-grpo rollout engines.",
    ),
    n_samples_per_prompt: int = typer.Option(4, help="GRPO samples per prompt."),
    train_epochs: int = typer.Option(1, help="Training epochs for cpp-sft/cpp-grpo."),
    eval_before_train: bool = typer.Option(
        True,
        "--eval-before-train/--no-eval-before-train",
        help="Run the initial GRPO eval pass before training updates.",
    ),
    eval_interval: int = typer.Option(50, help="Evaluation interval for cpp-sft/cpp-grpo."),
    max_env_workers: int = typer.Option(32, help="Maximum SkyRL Gym environment workers for cpp-grpo rewards."),
    ckpt_interval: int = typer.Option(-1, help="Checkpoint interval; negative disables smoke checkpoints."),
    hf_save_interval: int = typer.Option(-1, help="HF export interval; negative disables smoke exports."),
    ckpt_path: str = typer.Option("~/ckpts/", help="Checkpoint root for full training runs."),
    export_path: str = typer.Option("~/exports/", help="HF export root for full training runs."),
    max_ckpts_to_keep: int = typer.Option(-1, help="Maximum checkpoints to retain; -1 keeps all."),
    resume_from: str = typer.Option("", help="Training checkpoint resume source: empty, latest, or a global_step_N path."),
    export_checkpoint: str = typer.Option("", help="SFT/GRPO export-only checkpoint source: a global_step_N path."),
    sandbox_image: str = typer.Option(DEFAULT_DOCKER_IMAGE, help="C++ sandbox Docker image."),
    sandbox_cpu: str = typer.Option(DEFAULT_CPU, help="CPU id used by sandbox taskset."),
    tracking_backend: Optional[list[str]] = typer.Option(
        None,
        "--tracking-backend",
        help=(
            "Tracking backend for training. Repeatable. Supported: "
            f"{', '.join(SUPPORTED_TRACKING_BACKENDS)}. Full SFT/GRPO default to console+mlflow."
        ),
    ),
    run_id: Optional[str] = typer.Option(None, help="Run id for labels, cluster name, and artifacts."),
    owner: str = typer.Option("sss", help="Owner label for GCP resources."),
    eval_label: str = typer.Option("model", help="Evaluation label for cpp-eval output files."),
    eval_max_tasks: Optional[int] = typer.Option(None, help="Optional validation task limit for cpp-eval."),
    allow_low_multinode_utilization: bool = typer.Option(
        False,
        "--allow-low-multinode-utilization",
        help="Permit multi-node GRPO with low samples/GPU or too few reward workers.",
    ),
) -> None:
    """Render a cloud backend YAML file."""

    options = _render_options(
        pipeline=pipeline,
        credentials=credentials,
        bucket=bucket,
        accelerators=accelerators,
        num_nodes=num_nodes,
        region=region,
        zone=zone,
        disk_size=disk_size,
        cluster=cluster,
        model=model,
        gpu_container_image=gpu_container_image,
        dataset_gcs_prefix=dataset_gcs_prefix,
        train_batch_size=train_batch_size,
        micro_train_batch_size_per_gpu=micro_train_batch_size_per_gpu,
        grpo_use_kl_loss=grpo_use_kl_loss,
        grpo_kl_loss_coef=grpo_kl_loss_coef,
        grpo_use_entropy_loss=grpo_use_entropy_loss,
        grpo_entropy_loss_coef=grpo_entropy_loss_coef,
        grpo_vllm_gpu_memory_utilization=grpo_vllm_gpu_memory_utilization,
        n_samples_per_prompt=n_samples_per_prompt,
        train_epochs=train_epochs,
        eval_before_train=eval_before_train,
        eval_interval=eval_interval,
        max_env_workers=max_env_workers,
        ckpt_interval=ckpt_interval,
        hf_save_interval=hf_save_interval,
        ckpt_path=ckpt_path,
        export_path=export_path,
        max_ckpts_to_keep=max_ckpts_to_keep,
        resume_from=resume_from,
        export_checkpoint=export_checkpoint,
        sandbox_image=sandbox_image,
        sandbox_cpu=sandbox_cpu,
        tracking_backends=tracking_backend,
        run_id=run_id,
        owner=owner,
        eval_label=eval_label,
        eval_max_tasks=eval_max_tasks,
        allow_low_multinode_utilization=allow_low_multinode_utilization,
    )
    _require_grpo_multinode_utilization(options)
    _require_grpo_multinode_resume_disk(options)
    written = write_sky_yaml(options, output)
    console.print(str(written))


@app.command()
def launch(
    pipeline: Pipeline = typer.Argument(..., help="Pipeline to launch: cpp-smoke, cpp-sft, cpp-grpo, or cpp-eval."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI."),
    accelerators: Optional[str] = typer.Option(None, help="Cloud backend accelerator request."),
    num_nodes: int = typer.Option(1, help="Cloud backend node count."),
    region: str = typer.Option("", "--region", help="Optional GCP region to pin in rendered SkyPilot infra."),
    zone: str = typer.Option("", "--zone", help="Optional GCP zone to pin in rendered SkyPilot infra."),
    disk_size: Optional[int] = typer.Option(None, help="Cloud backend boot disk size in GB."),
    cluster: Optional[str] = typer.Option(None, help="Cloud backend cluster name."),
    model: Optional[str] = typer.Option(None, help="Open model to load/train."),
    gpu_container_image: str = typer.Option(DEFAULT_CPP_CONTAINER_IMAGE, help="GPU Docker image for the smoke."),
    dataset_gcs_prefix: Optional[str] = typer.Option(None, help="Versioned SkyRL dataset GCS prefix."),
    train_batch_size: int = typer.Option(16, help="Training batch size for cpp-sft/cpp-grpo."),
    micro_train_batch_size_per_gpu: int = typer.Option(
        1,
        "--micro-train-batch-size-per-gpu",
        help="Per-GPU micro train batch for cpp-sft/cpp-grpo; try 2 only as a memory-checked tuning experiment.",
    ),
    grpo_use_kl_loss: bool = typer.Option(
        True,
        "--grpo-use-kl-loss/--no-grpo-use-kl-loss",
        help="Apply a KL loss anchor during cpp-grpo to reduce policy drift from the SFT/reference model.",
    ),
    grpo_kl_loss_coef: float = typer.Option(0.001, help="KL loss coefficient for cpp-grpo."),
    grpo_use_entropy_loss: bool = typer.Option(
        True,
        "--grpo-use-entropy-loss/--no-grpo-use-entropy-loss",
        help="Apply an entropy bonus during cpp-grpo to reduce deterministic policy collapse.",
    ),
    grpo_entropy_loss_coef: float = typer.Option(0.001, help="Entropy loss coefficient for cpp-grpo."),
    grpo_vllm_gpu_memory_utilization: float = typer.Option(
        0.7,
        "--grpo-vllm-gpu-memory-utilization",
        help="vLLM GPU memory reservation for colocated cpp-grpo rollout engines.",
    ),
    n_samples_per_prompt: int = typer.Option(4, help="GRPO samples per prompt."),
    train_epochs: int = typer.Option(1, help="Training epochs for cpp-sft/cpp-grpo."),
    eval_before_train: bool = typer.Option(
        True,
        "--eval-before-train/--no-eval-before-train",
        help="Run the initial GRPO eval pass before training updates.",
    ),
    eval_interval: int = typer.Option(50, help="Evaluation interval for cpp-sft/cpp-grpo."),
    max_env_workers: int = typer.Option(32, help="Maximum SkyRL Gym environment workers for cpp-grpo rewards."),
    ckpt_interval: int = typer.Option(-1, help="Checkpoint interval; negative disables smoke checkpoints."),
    hf_save_interval: int = typer.Option(-1, help="HF export interval; negative disables smoke exports."),
    ckpt_path: str = typer.Option("~/ckpts/", help="Checkpoint root for full training runs."),
    export_path: str = typer.Option("~/exports/", help="HF export root for full training runs."),
    max_ckpts_to_keep: int = typer.Option(-1, help="Maximum checkpoints to retain; -1 keeps all."),
    resume_from: str = typer.Option("", help="Training checkpoint resume source: empty, latest, or a global_step_N path."),
    export_checkpoint: str = typer.Option("", help="SFT/GRPO export-only checkpoint source: a global_step_N path."),
    sandbox_image: str = typer.Option(DEFAULT_DOCKER_IMAGE, help="C++ sandbox Docker image."),
    sandbox_cpu: str = typer.Option(DEFAULT_CPU, help="CPU id used by sandbox taskset."),
    tracking_backend: Optional[list[str]] = typer.Option(
        None,
        "--tracking-backend",
        help=(
            "Tracking backend for training. Repeatable. Supported: "
            f"{', '.join(SUPPORTED_TRACKING_BACKENDS)}. Full SFT/GRPO default to console+mlflow."
        ),
    ),
    run_id: Optional[str] = typer.Option(None, help="Run id for labels, cluster name, and artifacts."),
    owner: str = typer.Option("sss", help="Owner label for GCP resources."),
    eval_label: str = typer.Option("model", help="Evaluation label for cpp-eval output files."),
    eval_max_tasks: Optional[int] = typer.Option(None, help="Optional validation task limit for cpp-eval."),
    yes: bool = typer.Option(True, help="Skip cloud backend confirmation prompts."),
    down_after: bool = typer.Option(True, help="Pass --down so successful smoke runs tear down the cluster."),
    detach_run: bool = typer.Option(
        False,
        "--detach-run/--no-detach-run",
        help="Submit the backend job and return after it starts instead of streaming logs until completion.",
    ),
    dry_run: bool = typer.Option(False, help="Render and print commands without launching."),
    allow_low_multinode_utilization: bool = typer.Option(
        False,
        "--allow-low-multinode-utilization",
        help="Permit multi-node GRPO with low samples/GPU or too few reward workers.",
    ),
) -> None:
    """Render config and launch a cloud backend job."""

    effective_run_id = run_id or _new_run_id()
    options = _render_options(
        pipeline=pipeline,
        credentials=credentials,
        bucket=bucket,
        accelerators=accelerators,
        num_nodes=num_nodes,
        region=region,
        zone=zone,
        disk_size=disk_size,
        cluster=cluster,
        model=model,
        gpu_container_image=gpu_container_image,
        dataset_gcs_prefix=dataset_gcs_prefix,
        train_batch_size=train_batch_size,
        micro_train_batch_size_per_gpu=micro_train_batch_size_per_gpu,
        grpo_use_kl_loss=grpo_use_kl_loss,
        grpo_kl_loss_coef=grpo_kl_loss_coef,
        grpo_use_entropy_loss=grpo_use_entropy_loss,
        grpo_entropy_loss_coef=grpo_entropy_loss_coef,
        grpo_vllm_gpu_memory_utilization=grpo_vllm_gpu_memory_utilization,
        n_samples_per_prompt=n_samples_per_prompt,
        train_epochs=train_epochs,
        eval_before_train=eval_before_train,
        eval_interval=eval_interval,
        max_env_workers=max_env_workers,
        ckpt_interval=ckpt_interval,
        hf_save_interval=hf_save_interval,
        ckpt_path=ckpt_path,
        export_path=export_path,
        max_ckpts_to_keep=max_ckpts_to_keep,
        resume_from=resume_from,
        export_checkpoint=export_checkpoint,
        sandbox_image=sandbox_image,
        sandbox_cpu=sandbox_cpu,
        tracking_backends=tracking_backend,
        run_id=effective_run_id,
        owner=owner,
        eval_label=eval_label,
        eval_max_tasks=eval_max_tasks,
        allow_low_multinode_utilization=allow_low_multinode_utilization,
    )
    _require_launch_labels(options)
    _require_grpo_multinode_utilization(options)
    _require_grpo_multinode_resume_disk(options)
    output = write_sky_yaml(options, f"{DEFAULT_RENDER_DIR}/{pipeline}.sky.yaml")
    if options.pipeline == "cpp-grpo" and options.num_nodes > 1 and not options.sft_export_checkpoint:
        _require_grpo_readiness(output)
    env = _service_account_env(credentials, project_id=options.project_id)
    sky_args = ["sky", "launch", "-c", options.name]
    if yes:
        sky_args.append("-y")
    if down_after:
        sky_args.append("--down")
    if detach_run:
        sky_args.append("--detach-run")
    sky_args.append(str(output))
    console.print(f"run_id: {options.run_id}")
    console.print(f"cluster: {options.name}")
    run_command(sky_args, env=env, dry_run=dry_run)


@benchmarks_app.command("list")
def benchmarks_list() -> None:
    """List the benchmark ladder."""

    table = Table(title="w8-biayn C++ performance-RL benchmarks")
    for column in ("key", "tier", "harness", "metric"):
        table.add_column(column)
    for benchmark in benchmarks.list_benchmarks():
        table.add_row(benchmark.key, benchmark.tier, benchmark.harness, benchmark.metric)
    console.print(table)


@benchmarks_app.command("show")
def benchmarks_show(key: str = typer.Argument(..., help="Benchmark key.")) -> None:
    """Show benchmark details."""

    try:
        benchmark = benchmarks.get_benchmark(key)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print_json(data=benchmark.__dict__)


@eval_app.command("cpp")
def eval_cpp(
    records: list[str] = typer.Option(
        ...,
        "--records",
        help="Evaluation records as LABEL=path.jsonl. Repeat for base/SFT/GRPO.",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Optional JSON summary path."),
) -> None:
    """Aggregate C++ eval JSONL records and compare model labels."""

    summaries = []
    for item in records:
        if "=" not in item:
            raise typer.BadParameter("--records must use LABEL=path.jsonl")
        label, path = item.split("=", 1)
        summary = aggregate_eval_records(read_jsonl(path), label=label)
        summary.pop("best_records", None)
        summaries.append(summary)
    comparison = compare_eval_summaries(summaries)
    if out:
        _write_json(out, comparison)
        console.print(out)
    else:
        console.print_json(data=comparison)


@eval_app.command("raw-report")
def eval_raw_report(
    run_id: str = typer.Option(..., "--run-id", help="Run id to report on."),
    run_root: Optional[str] = typer.Option(None, "--run-root", help="Local run artifact root. Defaults to .w8-biayn/runs/RUN_ID."),
    records: Optional[list[str]] = typer.Option(
        None,
        "--records",
        help="Optional eval records as LABEL=path.jsonl. Repeat to override run-root defaults.",
    ),
    metrics_json: Optional[str] = typer.Option(
        None,
        "--metrics-json",
        help="Optional MLflow metrics JSON. Defaults to RUN_ROOT/metrics.api.json when present.",
    ),
    status_json: Optional[str] = typer.Option(
        None,
        "--status-json",
        help="Optional run-status JSON. Defaults to RUN_ROOT/status.json when present.",
    ),
    out: Optional[str] = typer.Option(
        None,
        "--out",
        help="Markdown report path. Defaults to RUN_REPORT_RAW_RUN_ID.md.",
    ),
    assets_dir: Optional[str] = typer.Option(
        None,
        "--assets-dir",
        help="Directory for CSV/JSON/SVG assets. Defaults to RUN_REPORT_RAW_RUN_ID_assets.",
    ),
    companion_report: Optional[str] = typer.Option(
        None,
        "--companion-report",
        help="Optional interpretation report to link from the raw report.",
    ),
) -> None:
    """Create a raw Markdown report with CSV/JSON assets and SVG curves for one run."""

    record_map = _parse_label_paths(records or [], option_name="--records") if records else None
    result = build_raw_run_report(
        run_id=run_id,
        run_root=run_root,
        output=out,
        assets_dir=assets_dir,
        records=record_map,
        metrics_json=metrics_json,
        status_json=status_json,
        companion_report=companion_report,
    )
    console.print_json(data=result.as_dict())


def _parse_label_paths(items: list[str], *, option_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"{option_name} must use LABEL=path")
        label, path = item.split("=", 1)
        if not label:
            raise typer.BadParameter(f"{option_name} label must not be empty")
        parsed[label] = path
    return parsed


@gcp_app.command("cleanup")
def gcp_cleanup(
    run_id: str = typer.Option(..., "--run-id", help="Run id to tear down."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    owner: str = typer.Option("sss", help="Owner label used for the run."),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Print cleanup commands unless --execute is used."),
) -> None:
    """Tear down run clusters and list GCP instances for one run id."""

    project_id = _project_id(credentials)
    env = _service_account_env(credentials, project_id=project_id)
    console.print(f"cleanup_run_id: {run_id}")
    for pipeline_name in ("cpp-smoke", "cpp-sft", "cpp-grpo", "cpp-eval"):
        cluster = f"w8-biayn-{pipeline_name}-{run_id}"
        run_command(["sky", "down", "-y", cluster], env=env, dry_run=dry_run)
    label_filter = (
        f"labels.project=w8-biayn AND labels.phase=cpp-perf-rl "
        f"AND labels.run_id={run_id.lower()} AND labels.owner={owner.lower()}"
    )
    run_command(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project_id,
            "--filter",
            label_filter,
            "--format",
            "table(name,zone,status,labels)",
        ],
        env=env,
        dry_run=dry_run,
    )


@ops_app.command("status")
def ops_status(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    refresh: bool = typer.Option(False, help="Refresh cloud cluster state before printing status."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Show current run clusters and managed jobs."""

    args = ["sky", "status"]
    if refresh:
        args.append("--refresh")
    _run_backend_command(args, credentials=credentials, dry_run=dry_run)


@ops_app.command("logs")
def ops_logs(
    cluster: str = typer.Argument(..., help="w8-biayn cluster name."),
    job_id: Optional[str] = typer.Argument(None, help="Optional job id; omit for latest job."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    follow: bool = typer.Option(False, "--follow/--no-follow", help="Stream logs instead of printing the current tail."),
    tail: int = typer.Option(1000, help="Number of trailing log lines; pass 0 for the full log."),
    status_only: bool = typer.Option(False, "--status", help="Return only the job status code."),
    provision: bool = typer.Option(False, help="Show provisioning logs instead of job logs."),
    autostop: bool = typer.Option(False, help="Show autostop hook logs."),
    sync_down: bool = typer.Option(False, "--sync-down", help="Download logs under the backend logs directory."),
    worker: Optional[int] = typer.Option(None, help="Worker id to read logs from."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Show logs for a run cluster without invoking the backend directly."""

    args = ["sky", "logs"]
    if provision:
        args.append("--provision")
    if autostop:
        args.append("--autostop")
    if sync_down:
        args.append("--sync-down")
    if status_only:
        args.append("--status")
    if worker is not None:
        args.extend(["--worker", str(worker)])
    args.extend(["--tail", str(tail)])
    if not follow:
        args.append("--no-follow")
    args.append(cluster)
    if job_id is not None:
        args.append(job_id)
    _run_backend_command(args, credentials=credentials, dry_run=dry_run)


@ops_app.command("queue")
def ops_queue(
    cluster: Optional[str] = typer.Argument(None, help="Optional cluster name; omit to show all queues."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    skip_finished: bool = typer.Option(False, "--skip-finished", help="Show only pending or running jobs."),
    all_users: bool = typer.Option(False, "--all-users", help="Show jobs for all users."),
    output: str = typer.Option("table", "--output", help="Output format: table or json."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Show queued jobs for run clusters."""

    args = ["sky", "queue"]
    if all_users:
        args.append("--all-users")
    if skip_finished:
        args.append("--skip-finished")
    args.extend(["--output", output])
    if cluster is not None:
        args.append(cluster)
    _run_backend_command(args, credentials=credentials, dry_run=dry_run)


@ops_app.command("cancel")
def ops_cancel(
    cluster: str = typer.Argument(..., help="w8-biayn cluster name or glob."),
    job_id: Optional[str] = typer.Argument(None, help="Optional job id; omit for latest running job."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    all_jobs: bool = typer.Option(False, "--all", help="Cancel all jobs from the current user."),
    all_users: bool = typer.Option(False, "--all-users", help="Cancel matching jobs for all users."),
    yes: bool = typer.Option(True, "--yes/--no-yes", help="Skip backend confirmation prompts."),
    async_: bool = typer.Option(False, "--async/--no-async", help="Return without waiting for cancellation."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Cancel a queued or running job."""

    args = ["sky", "cancel", cluster]
    if job_id is not None:
        args.append(job_id)
    if all_jobs:
        args.append("--all")
    if all_users:
        args.append("--all-users")
    if yes:
        args.append("--yes")
    if async_:
        args.append("--async")
    _run_backend_command(args, credentials=credentials, dry_run=dry_run)


@ops_app.command("down")
def ops_down(
    cluster: str = typer.Argument(..., help="w8-biayn cluster name."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    yes: bool = typer.Option(True, "--yes/--no-yes", help="Skip backend confirmation prompts."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Tear down one run cluster."""

    args = ["sky", "down"]
    if yes:
        args.append("-y")
    args.append(cluster)
    _run_backend_command(args, credentials=credentials, dry_run=dry_run)


@ops_app.command("gpus")
def ops_gpus(
    accelerator: Optional[str] = typer.Argument(None, help="Optional accelerator name such as A100 or B200."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    infra: str = typer.Option("gcp", help="Infrastructure selector passed to the backend."),
    all_regions: bool = typer.Option(False, "--all-regions", help="Show every region for the accelerator."),
    all_offerings: bool = typer.Option(False, "--all", help="Show all accelerator offerings."),
    output: str = typer.Option("table", "--output", help="Output format: table or json."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """List accelerator offerings through the project CLI."""

    args = ["sky", "gpus", "list"]
    if accelerator:
        args.append(accelerator)
    args.extend(["--infra", infra])
    if all_regions:
        args.append("--all-regions")
    if all_offerings:
        args.append("--all")
    args.extend(["--output", output])
    _run_backend_command(args, credentials=credentials, dry_run=dry_run)


@ops_app.command("run-status")
def ops_run_status(
    run_id: str = typer.Option(..., "--run-id", help="Run id to inspect."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI. Defaults from the credentials project."),
    dataset_gcs_prefix: Optional[str] = typer.Option(None, help="SkyRL dataset GCS prefix to verify."),
    pipeline: Optional[list[str]] = typer.Option(
        None,
        "--pipeline",
        help="Pipeline to inspect. Repeatable. Defaults to cpp-sft, cpp-grpo, and cpp-eval.",
    ),
    owner: str = typer.Option("sss", help="Owner label used for run-scoped GCP resources."),
    job_id: Optional[str] = typer.Option(None, "--job-id", help="Optional backend job id for log inspection."),
    log_tail: int = typer.Option(800, "--log-tail", help="Backend log tail lines to scan for status signals."),
    expected_world_size: int = typer.Option(8, help="Expected FSDP world size for checkpoint shard validation."),
    expected_sft_final_step: Optional[int] = typer.Option(
        None,
        "--expected-sft-final-step",
        help="Expected final SFT step used to check final HF export readiness.",
    ),
    expected_grpo_final_step: Optional[int] = typer.Option(
        None,
        "--expected-grpo-final-step",
        help="Expected final GRPO step used to check final HF export readiness.",
    ),
    check_timeout: int = typer.Option(
        180,
        "--check-timeout",
        help="Seconds allowed for each backend/GCS check before returning a partial status.",
    ),
    check_retries: int = typer.Option(
        0,
        "--check-retries",
        min=0,
        help="Retries for timed-out or transiently failed read-only backend/GCS/SSH checks.",
    ),
    node_health: bool = typer.Option(
        False,
        "--node-health",
        help="Include read-only SSH GPU/disk/process health for running cluster nodes.",
    ),
    baseline_status: Optional[list[str]] = typer.Option(
        None,
        "--baseline-status",
        help="Prior run-status JSON path to compare throughput against. Repeatable.",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Optional path to also write the JSON snapshot."),
    dry_run: bool = typer.Option(False, help="Emit planned checks without calling cloud backends."),
) -> None:
    """Emit one JSON status snapshot for a run, suitable for dashboards."""

    selected_pipelines = pipeline or list(STATUS_PIPELINES)
    unknown = sorted(set(selected_pipelines) - set(STATUS_PIPELINES))
    if unknown:
        raise typer.BadParameter(f"unknown pipeline(s): {', '.join(unknown)}")
    project_id = _project_id(credentials)
    artifact_bucket = (bucket or default_bucket_for_project(project_id)).rstrip("/")
    baseline_payloads = []
    for status_path in baseline_status or []:
        payload = json.loads(Path(status_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["_status_source"] = status_path
            baseline_payloads.append(payload)
    payload = build_run_status(
        run_id=run_id,
        project_id=project_id,
        artifact_bucket=artifact_bucket,
        env=_service_account_env(credentials, project_id=project_id),
        credentials_path=credentials,
        dataset_gcs_prefix=dataset_gcs_prefix,
        pipelines=selected_pipelines,
        owner=owner,
        log_tail=log_tail,
        job_id=job_id,
        expected_world_size=expected_world_size,
        expected_sft_final_step=expected_sft_final_step,
        expected_grpo_final_step=expected_grpo_final_step,
        include_node_health=node_health,
        baseline_statuses=baseline_payloads,
        command_timeout_s=check_timeout,
        command_retries=check_retries,
        dry_run=dry_run,
    )
    if out:
        _write_json(out, payload)
    console.print_json(data=payload)


@ops_app.command("metrics")
def ops_metrics(
    run_id: str = typer.Option(..., "--run-id", help="Run id to inspect."),
    pipeline: str = typer.Option("cpp-grpo", "--pipeline", help="Pipeline to inspect, usually cpp-sft or cpp-grpo."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    bucket: Optional[str] = typer.Option(None, help="Artifact bucket URI. Defaults from the credentials project."),
    source: str = typer.Option(
        "auto",
        "--source",
        help="Metric source: auto prefers live MLflow API through SSH tunnel, api requires it, sqlite reads GCS snapshot.",
    ),
    cluster: Optional[str] = typer.Option(
        None,
        "--cluster",
        help="Optional SkyPilot cluster name for live MLflow API tunneling. Defaults to w8-biayn-<pipeline>-<run-id>.",
    ),
    mlflow_port: int = typer.Option(5000, "--mlflow-port", min=1, max=65535, help="Remote MLflow server port."),
    api_timeout: int = typer.Option(20, "--api-timeout", min=1, help="Seconds to wait for live API/tunnel."),
    metric: Optional[list[str]] = typer.Option(
        None,
        "--metric",
        help="Metric key to include. Repeatable. Defaults to the standard GRPO/SFT post-training metric set.",
    ),
    last: int = typer.Option(100, "--last", min=1, help="Number of points per selected metric to return."),
    out: Optional[str] = typer.Option(None, "--out", help="Optional path to also write the JSON payload."),
    dry_run: bool = typer.Option(False, help="Print the planned GCS download without reading cloud state."),
) -> None:
    """Fetch MLflow metrics through live API or synced SQLite and emit headless metric JSON."""

    if pipeline not in STATUS_PIPELINES:
        raise typer.BadParameter(f"unknown pipeline: {pipeline}")
    if source not in {"auto", "api", "sqlite"}:
        raise typer.BadParameter("source must be one of: auto, api, sqlite")
    project_id = _project_id(credentials)
    artifact_bucket = (bucket or default_bucket_for_project(project_id)).rstrip("/")
    db_uri = f"{artifact_bucket}/runs/cpp-perf/{run_id}/{pipeline}/tracking/mlflow/mlflow.db"
    local_db = Path(".w8-biayn/runs") / run_id / pipeline / "mlflow.db"
    local_db.parent.mkdir(parents=True, exist_ok=True)
    cluster_name = cluster or f"w8-biayn-{pipeline}-{run_id}"
    checks: list[dict[str, object]] = []
    metrics_payload: dict[str, object] | None = None
    resolved_source = source
    if dry_run:
        if source in {"auto", "api"}:
            checks.append(
                {
                    "name": "ssh_tunnel:mlflow_api",
                    "command": _mlflow_tunnel_command(
                        cluster=cluster_name,
                        local_port="<auto>",
                        remote_port=mlflow_port,
                    ),
                    "ok": None,
                    "returncode": None,
                    "skipped": True,
                    "timed_out": False,
                    "attempt_count": 1,
                }
            )
        if source in {"auto", "sqlite"}:
            checks.append(_gcloud_mlflow_db_check(["gcloud", "storage", "cp", db_uri, str(local_db)], dry_run=True))
        metrics_payload = {
            "available": False,
            "reason": "dry_run",
            "backend": "mlflow_api" if source == "api" else "mlflow",
            "tracking_state": "dry_run",
            "source": {"path": str(local_db), "size_bytes": None},
            "experiments": {"count": 0, "items": []},
            "runs": {"count": 0, "items": []},
            "run": {"available": False},
            "params": {"count": 0, "available_keys": [], "selected": {}},
            "tags": {"count": 0, "items": {}},
            "latest_step": None,
            "metric_row_count": 0,
            "metric_count": 0,
            "available_keys": [],
            "selected_keys": list(metric or DEFAULT_METRIC_KEYS),
            "latest": {},
            "series": {},
        }
    else:
        if source in {"auto", "api"}:
            api_payload, api_checks = _read_mlflow_metrics_via_tunnel(
                cluster=cluster_name,
                remote_port=mlflow_port,
                env=_service_account_env(credentials, project_id=project_id),
                metric_keys=metric,
                last=last,
                timeout_s=api_timeout,
            )
            checks.extend(api_checks)
            if api_payload.get("available") is True or source == "api":
                metrics_payload = api_payload
                resolved_source = "api"
        if metrics_payload is None:
            sqlite_payload, sqlite_check = _read_mlflow_metrics_from_gcs(
                db_uri=db_uri,
                local_db=local_db,
                credentials=credentials,
                project_id=project_id,
                metric_keys=metric,
                last=last,
            )
            checks.append(sqlite_check)
            metrics_payload = sqlite_payload
            resolved_source = "sqlite"
    payload = {
        "schema_version": "w8-mlflow-metrics-v1",
        "run_id": run_id,
        "pipeline": pipeline,
        "source": resolved_source,
        "cluster": cluster_name,
        "mlflow_api": {
            "remote_port": mlflow_port,
            "tunnel": "ssh",
            "available": metrics_payload.get("backend") == "mlflow_api" and metrics_payload.get("available") is True,
        },
        "artifact_bucket": artifact_bucket,
        "mlflow_db_uri": db_uri,
        "local_db_path": str(local_db),
        "metrics": metrics_payload,
        "checks": checks,
    }
    if out:
        _write_json(out, payload)
    console.print_json(data=payload)


def _read_mlflow_metrics_from_gcs(
    *,
    db_uri: str,
    local_db: Path,
    credentials: str,
    project_id: str,
    metric_keys: list[str] | None,
    last: int,
) -> tuple[dict[str, object], dict[str, object]]:
    command = ["gcloud", "storage", "cp", db_uri, str(local_db)]
    proc = subprocess.run(
        command,
        env=_service_account_env(credentials, project_id=project_id),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    check = _gcloud_mlflow_db_check(command, proc=proc)
    if proc.returncode != 0:
        return (
            {
                "available": False,
                "reason": "mlflow_db_unavailable",
                "backend": "mlflow",
                "tracking_state": "mlflow_db_unavailable",
                "source": {"path": str(local_db), "size_bytes": None},
                "experiments": {"count": 0, "items": []},
                "runs": {"count": 0, "items": []},
                "run": {"available": False},
                "params": {"count": 0, "available_keys": [], "selected": {}},
                "tags": {"count": 0, "items": {}},
                "latest_step": None,
                "metric_row_count": 0,
                "metric_count": 0,
                "available_keys": [],
                "selected_keys": list(metric_keys or DEFAULT_METRIC_KEYS),
                "latest": {},
                "series": {},
            },
            check,
        )
    return read_mlflow_metrics(local_db, metric_keys=metric_keys, last=last), check


def _gcloud_mlflow_db_check(
    command: list[str],
    *,
    proc: subprocess.CompletedProcess[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    check: dict[str, object] = {
        "name": "gcloud_storage_cp:mlflow_db",
        "command": command,
        "ok": None if dry_run else proc is not None and proc.returncode == 0,
        "returncode": None if dry_run or proc is None else proc.returncode,
        "skipped": dry_run,
        "timed_out": False,
        "attempt_count": 1,
    }
    if proc is not None and proc.stderr:
        check["stderr_tail"] = proc.stderr[-2000:]
    return check


def _read_mlflow_metrics_via_tunnel(
    *,
    cluster: str,
    remote_port: int,
    env: dict[str, str],
    metric_keys: list[str] | None,
    last: int,
    timeout_s: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    local_port = _free_local_port()
    command = _mlflow_tunnel_command(cluster=cluster, local_port=local_port, remote_port=remote_port)
    check: dict[str, object] = {
        "name": "ssh_tunnel:mlflow_api",
        "command": command,
        "ok": False,
        "returncode": None,
        "skipped": False,
        "timed_out": False,
        "attempt_count": 1,
    }
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{local_port}"
    payload: dict[str, object] = {
        "available": False,
        "reason": "mlflow_api_unavailable",
        "backend": "mlflow_api",
        "tracking_state": "mlflow_api_unavailable",
        "source": {"base_url": base_url},
        "experiments": {"count": 0, "items": []},
        "runs": {"count": 0, "items": []},
        "run": {"available": False},
        "params": {"count": 0, "available_keys": [], "selected": {}},
        "tags": {"count": 0, "items": {}},
        "latest_step": None,
        "metric_row_count": 0,
        "metric_count": 0,
        "available_keys": [],
        "selected_keys": list(metric_keys or DEFAULT_METRIC_KEYS),
        "latest": {},
        "series": {},
    }
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            returncode = proc.poll()
            if returncode is not None:
                check["returncode"] = returncode
                _, stderr = proc.communicate(timeout=1)
                if stderr:
                    check["stderr_tail"] = stderr[-2000:]
                payload["reason"] = f"ssh_tunnel_failed:{returncode}"
                payload["tracking_state"] = payload["reason"]
                return payload, [check]
            payload = read_mlflow_api(base_url, metric_keys=metric_keys, last=last, timeout_s=3.0)
            if payload.get("available") is True:
                check["ok"] = True
                return payload, [check]
            time.sleep(0.5)
        check["timed_out"] = True
        payload["reason"] = "mlflow_api_tunnel_timeout"
        payload["tracking_state"] = "mlflow_api_tunnel_timeout"
        return payload, [check]
    finally:
        _stop_process(proc)


def _mlflow_tunnel_command(*, cluster: str, local_port: int | str, remote_port: int) -> list[str]:
    return [
        "ssh",
        "-N",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        cluster,
    ]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


@ops_app.command("grpo-readiness")
def ops_grpo_readiness(
    rendered_config: str = typer.Option(
        f"{DEFAULT_RENDER_DIR}/cpp-grpo.sky.yaml",
        "--rendered-config",
        help="Rendered cpp-grpo SkyPilot YAML to validate.",
    ),
    status_json: Optional[str] = typer.Option(
        None,
        "--status-json",
        help="Optional ops run-status JSON snapshot to validate live multi-node evidence.",
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Optional path to also write readiness JSON."),
) -> None:
    """Validate rendered GRPO config and optional live status before launch or claims."""

    status_payload = None
    if status_json:
        status_payload = json.loads(Path(status_json).read_text(encoding="utf-8"))
        if not isinstance(status_payload, dict):
            raise typer.BadParameter("--status-json must contain a JSON object")
    payload = build_grpo_readiness(rendered_config, status_payload=status_payload)
    if out:
        _write_json(out, payload)
    console.print_json(data=payload)
    if readiness_blocks_launch(payload):
        raise typer.Exit(1)


@app.command()
def status(
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    refresh: bool = typer.Option(False, help="Refresh cloud cluster state before printing status."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Alias for `w8-biayn ops status`."""

    ops_status(credentials=credentials, refresh=refresh, dry_run=dry_run)


@app.command()
def logs(
    cluster: str = typer.Argument("w8-biayn-cpp-smoke", help="w8-biayn cluster name."),
    job_id: Optional[str] = typer.Argument(None, help="Optional job id; omit for latest job."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    follow: bool = typer.Option(False, "--follow/--no-follow", help="Stream logs instead of printing the current tail."),
    tail: int = typer.Option(1000, help="Number of trailing log lines; pass 0 for the full log."),
    status_only: bool = typer.Option(False, "--status", help="Return only the job status code."),
    provision: bool = typer.Option(False, help="Show provisioning logs instead of job logs."),
    autostop: bool = typer.Option(False, help="Show autostop hook logs."),
    sync_down: bool = typer.Option(False, "--sync-down", help="Download logs under the backend logs directory."),
    worker: Optional[int] = typer.Option(None, help="Worker id to read logs from."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Alias for `w8-biayn ops logs`."""

    ops_logs(
        cluster,
        job_id,
        credentials=credentials,
        follow=follow,
        tail=tail,
        status_only=status_only,
        provision=provision,
        autostop=autostop,
        sync_down=sync_down,
        worker=worker,
        dry_run=dry_run,
    )


@app.command()
def down(
    cluster: str = typer.Argument("w8-biayn-cpp-smoke", help="w8-biayn cluster name."),
    credentials: str = typer.Option(DEFAULT_CREDENTIALS_PATH, help="Path to local GCP service-account JSON."),
    yes: bool = typer.Option(True, "--yes/--no-yes", help="Skip backend confirmation prompts."),
    dry_run: bool = typer.Option(False, help="Print the backend command without running it."),
) -> None:
    """Alias for `w8-biayn ops down`."""

    ops_down(cluster, credentials=credentials, yes=yes, dry_run=dry_run)



def _render_options(
    *,
    pipeline: Pipeline,
    credentials: str,
    bucket: Optional[str],
    accelerators: Optional[str],
    num_nodes: int,
    region: str,
    zone: str,
    disk_size: Optional[int],
    cluster: Optional[str],
    model: Optional[str],
    gpu_container_image: str,
    dataset_gcs_prefix: Optional[str],
    train_batch_size: int,
    micro_train_batch_size_per_gpu: int,
    grpo_use_kl_loss: bool,
    grpo_kl_loss_coef: float,
    grpo_use_entropy_loss: bool,
    grpo_entropy_loss_coef: float,
    grpo_vllm_gpu_memory_utilization: float,
    n_samples_per_prompt: int,
    train_epochs: int,
    eval_before_train: bool,
    eval_interval: int,
    max_env_workers: int,
    ckpt_interval: int,
    hf_save_interval: int,
    ckpt_path: str,
    export_path: str,
    max_ckpts_to_keep: int,
    resume_from: str,
    export_checkpoint: str,
    sandbox_image: str,
    sandbox_cpu: str,
    tracking_backends: Optional[list[str]],
    run_id: Optional[str],
    owner: str,
    eval_label: str,
    eval_max_tasks: Optional[int],
    allow_low_multinode_utilization: bool,
) -> RenderOptions:
    project_id = _project_id(credentials)
    is_training = pipeline in ("cpp-sft", "cpp-grpo")
    is_eval = pipeline == "cpp-eval"
    default_accelerators = DEFAULT_CPP_TRAIN_ACCELERATORS if is_training else DEFAULT_CPP_SMOKE_ACCELERATORS
    if is_eval:
        default_accelerators = DEFAULT_CPP_EVAL_ACCELERATORS
    if micro_train_batch_size_per_gpu < 1:
        raise typer.BadParameter("--micro-train-batch-size-per-gpu must be at least 1")
    if grpo_kl_loss_coef < 0:
        raise typer.BadParameter("--grpo-kl-loss-coef must be non-negative")
    if grpo_entropy_loss_coef < 0:
        raise typer.BadParameter("--grpo-entropy-loss-coef must be non-negative")
    if not 0 < grpo_vllm_gpu_memory_utilization <= 1:
        raise typer.BadParameter("--grpo-vllm-gpu-memory-utilization must be in (0, 1]")
    normalized_tracking_backends = _normalize_tracking_backends(tracking_backends)
    return RenderOptions(
        pipeline=pipeline,
        project_id=project_id,
        bucket=bucket,
        credentials_path=credentials,
        accelerators=accelerators or default_accelerators,
        num_nodes=num_nodes,
        region=region,
        zone=zone,
        disk_size=disk_size,
        cluster_name=cluster,
        model=model,
        gpu_container_image=gpu_container_image,
        dataset_gcs_prefix=dataset_gcs_prefix,
        train_batch_size=train_batch_size,
        micro_train_batch_size_per_gpu=micro_train_batch_size_per_gpu,
        grpo_use_kl_loss=grpo_use_kl_loss,
        grpo_kl_loss_coef=grpo_kl_loss_coef,
        grpo_use_entropy_loss=grpo_use_entropy_loss,
        grpo_entropy_loss_coef=grpo_entropy_loss_coef,
        grpo_vllm_gpu_memory_utilization=grpo_vllm_gpu_memory_utilization,
        n_samples_per_prompt=n_samples_per_prompt,
        train_epochs=train_epochs,
        eval_before_train=eval_before_train,
        eval_interval=eval_interval,
        max_env_workers=max_env_workers,
        ckpt_interval=ckpt_interval,
        hf_save_interval=hf_save_interval,
        ckpt_path=ckpt_path,
        export_path=export_path,
        max_ckpts_to_keep=max_ckpts_to_keep,
        resume_from=resume_from,
        sft_export_checkpoint=export_checkpoint,
        sandbox_image=sandbox_image,
        sandbox_cpu=sandbox_cpu,
        tracking_backends=normalized_tracking_backends,
        run_id=run_id,
        owner=owner,
        eval_label=eval_label,
        eval_max_tasks=eval_max_tasks,
        allow_low_multinode_utilization=allow_low_multinode_utilization,
    )
