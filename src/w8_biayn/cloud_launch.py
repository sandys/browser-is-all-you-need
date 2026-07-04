"""Provision GCP GPU hardware with SkyPilot and run the full GLM C++ SLIME lane.

`w8-biayn launch glm47-full` is the supported entrypoint. SkyPilot manages the
cloud hardware (provision, workdir sync, artifact download, teardown); all
training runs through SLIME inside the lane container on the provisioned node.
The SkyPilot client is imported lazily at the pin in
`w8_biayn.constants.SKYPILOT_PIN`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from typing import IO, Any

from .constants import DEFAULT_CREDENTIALS_PATH, SKYPILOT_PIN, SLIME_PIN
from .gcp_auth import service_account_env
from .secrets import default_bucket_for_project, get_project_id

ALLOWED_REGIONS = ("asia-southeast1", "asia-south1", "asia-south2")
DEFAULT_REGION = "asia-southeast1"
DEFAULT_ACCELERATORS = "H100:8"
DEFAULT_SLIME_IMAGE = "slimerl/slime:latest"
DEFAULT_WANDB_PROJECT = "slime-glm47-cpp-perf"
DEFAULT_LOCAL_ROOT = Path(".w8-biayn/slime/glm47-cpp-perf")
FULL_LIMIT_SENTINEL = 1_000_000
TERMINAL_JOB_STATUS_SUFFIXES = ("SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED")
DEFAULT_ENV_FILE = ".env"
# Bump when the tasks-full build semantics change so old cache entries are not
# reused. Combined with the admission gates, this makes the cache path a pure
# function of what produced it — shared across users, never colliding.
TASKS_CACHE_VERSION = "cpp-perf-v1"


class LaunchError(RuntimeError):
    """Raised when a paid launch cannot start or track its remote job."""


@dataclass
class LaunchOptions:
    """Inputs for one full GLM C++ lane run on provisioned GCP hardware."""

    run_id: str = ""
    regions: list[str] = field(default_factory=lambda: [DEFAULT_REGION])
    accelerators: str = DEFAULT_ACCELERATORS
    use_spot: bool = False
    retry_sleep_seconds: int = 300
    max_attempts: int = 0
    cluster_name: str = ""
    credentials: str = DEFAULT_CREDENTIALS_PATH
    slime_image: str = DEFAULT_SLIME_IMAGE
    wandb_api_key: str = ""
    wandb_api_key_file: str = ""
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_entity: str = ""
    wandb_base_url: str = ""
    hf_token: str = ""
    hf_token_file: str = ""
    local_output_root: str = str(DEFAULT_LOCAL_ROOT)
    train_limit: int = FULL_LIMIT_SENTINEL
    eval_limit: int = FULL_LIMIT_SENTINEL
    min_train_tasks: int = 1000
    min_validation_tasks: int = 100
    min_test_tasks: int = 100
    coverage_jobs: int = 32
    grpo_num_rollout: int = 8
    grpo_rollout_batch_size: int = 4
    grpo_global_batch_size: int = 4
    eval_samples_per_prompt: int = 2
    disk_size: int = 1024
    dry_run: bool = False
    artifact_bucket: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = default_run_id()
        invalid = [region for region in self.regions if region not in ALLOWED_REGIONS]
        if invalid:
            raise LaunchError(f"regions {invalid} are not in the allowed set {ALLOWED_REGIONS}")
        if not self.cluster_name:
            self.cluster_name = default_cluster_name(self.run_id)


class Tee:
    """Write stream output to multiple file-like objects."""

    def __init__(self, *streams: IO[str]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def import_sky() -> Any:
    """Import the pinned SkyPilot client, with a w8-biayn-native install hint."""

    try:
        import sky
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SkyPilot with GCP support is required for paid launches. Run: "
            "uv run --extra cloud w8-biayn launch glm47-full ... "
            f"(the cloud extra pins {SKYPILOT_PIN}; keep it matched to any locally "
            "running sky API server)"
        ) from exc
    return sky


def apply_scoped_gcp_env(credentials: str) -> None:
    """Export scoped service-account env vars for this process only."""

    os.environ.update(service_account_env(credentials))


def resolve_wandb_api_key(
    *,
    direct: str = "",
    file_path: str = "",
    env_file: str = DEFAULT_ENV_FILE,
    required: bool = True,
) -> str:
    """Resolve the W&B key from flag, key file, environment, or the repo .env.

    Never prints or logs the key value.
    """

    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    value = direct or os.environ.get("WANDB_API_KEY", "") or os.environ.get("WANDB_KEY", "")
    if not value and Path(env_file).exists():
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition("=")
            if key.strip() in ("WANDB_API_KEY", "WANDB_KEY"):
                value = raw.strip().strip("'\"")
                if value:
                    break
    if required and not value:
        raise LaunchError(
            "Missing W&B API key: pass --wandb-api-key/--wandb-api-key-file, set "
            "WANDB_API_KEY, or add WANDB_KEY to .env."
        )
    return value


def resolve_optional_secret(*, direct: str, file_path: str, env_name: str) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    return direct or os.environ.get(env_name, "")


def launch_glm47_full(options: LaunchOptions) -> int:
    """Run the full GLM lane on provisioned hardware; returns a process exit code."""

    wandb_api_key = "" if options.dry_run else resolve_wandb_api_key(
        direct=options.wandb_api_key,
        file_path=options.wandb_api_key_file,
    )
    hf_token = resolve_optional_secret(
        direct=options.hf_token, file_path=options.hf_token_file, env_name="HF_TOKEN"
    )

    # Resolve the shared cache bucket from the credentials' project so every
    # user in that project restores the same cache (bare bucket name; the run
    # script prepends gs://). Falls back to a placeholder for dry-run rendering.
    if not options.artifact_bucket:
        try:
            project_id = get_project_id(options.credentials)
            options.artifact_bucket = default_bucket_for_project(project_id).removeprefix("gs://")
        except Exception:  # noqa: BLE001 - dry-run and missing-cred paths render a placeholder.
            options.artifact_bucket = "<project>-w8-biayn"

    local_root = Path(options.local_output_root)
    local_cloud_root = local_root / "cloud-runs" / options.run_id
    local_run_root = local_root / "runs" / options.run_id
    local_cloud_root.mkdir(parents=True, exist_ok=True)
    local_run_root.mkdir(parents=True, exist_ok=True)
    log_path = local_cloud_root / "orchestrator.log"
    config_path = local_cloud_root / "launch_config.json"
    _write_config(config_path, options=options)

    setup_script = build_setup_script()
    run_script = build_run_script()

    if options.dry_run:
        print(
            json.dumps(
                {
                    "cluster": options.cluster_name,
                    "regions": options.regions,
                    "accelerators": options.accelerators,
                    "use_spot": options.use_spot,
                    "config": str(config_path),
                },
                indent=2,
            )
        )
        print("\n--- setup ---")
        print(setup_script)
        print("\n--- run ---")
        print(run_script)
        return 0

    apply_scoped_gcp_env(options.credentials)

    cluster_acquired = False
    training_status = 1
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"run_id={options.run_id}")
            print(f"cluster={options.cluster_name}")
            print(f"regions={','.join(options.regions)}")
            print(f"local_run_root={local_run_root}")
            try:
                cluster_acquired = _acquire_and_run(
                    options=options,
                    setup_script=setup_script,
                    run_script=run_script,
                    local_run_root=local_run_root,
                    local_cloud_root=local_cloud_root,
                    wandb_api_key=wandb_api_key,
                    hf_token=hf_token,
                )
                training_status = 0
            except KeyboardInterrupt:
                print("\nInterrupted; attempting artifact download and cluster teardown.", flush=True)
                training_status = 130
            except Exception as exc:  # noqa: BLE001 - top-level orchestrator.
                print(f"\nTraining/provisioning failed: {exc!r}", flush=True)
                training_status = 1
            finally:
                # Never rely on cluster_acquired alone: a failure mid-wait
                # leaves it False while the cluster is still live and billing.
                # Always tear down anything that still exists for this run.
                cluster_live = False
                try:
                    cluster_live = _cluster_exists(import_sky(), options.cluster_name)
                except Exception:  # noqa: BLE001 - teardown must run even if status check fails.
                    cluster_live = cluster_acquired
                if cluster_acquired or cluster_live:
                    download_artifacts(
                        cluster_name=options.cluster_name,
                        run_id=options.run_id,
                        local_run_root=local_run_root,
                        local_cloud_root=local_cloud_root,
                    )
                    down_cluster(options.cluster_name)
                else:
                    print("No cluster was acquired; nothing to tear down.", flush=True)

            print(f"orchestrator_log={log_path}")
            print(f"downloaded_run_root={local_run_root}")
            print(f"cloud_log_root={local_cloud_root}")
            comparison = local_run_root / "eval" / "comparison.json"
            if comparison.exists():
                print(f"comparison_json={comparison}")
    return training_status


def _acquire_and_run(
    *,
    options: LaunchOptions,
    setup_script: str,
    run_script: str,
    local_run_root: Path,
    local_cloud_root: Path,
    wandb_api_key: str,
    hf_token: str,
) -> bool:
    sky = import_sky()

    attempt = 0
    while options.max_attempts == 0 or attempt < options.max_attempts:
        for region in options.regions:
            attempt += 1
            print(
                f"\n=== Provision attempt {attempt}: {options.accelerators} in {region} ===",
                flush=True,
            )
            task = sky.Task(
                name=f"glm47-cpp-full-{options.run_id}",
                setup=setup_script,
                run=run_script,
                workdir=".",
                envs={
                    "W8_GLM47_RUN_ID": options.run_id,
                    "W8_GLM47_WANDB_PROJECT": options.wandb_project,
                    "W8_GLM47_WANDB_GROUP": options.run_id,
                    "W8_GLM47_WANDB_ENTITY": options.wandb_entity,
                    "W8_GLM47_WANDB_BASE_URL": options.wandb_base_url,
                    "W8_GLM47_SLIME_IMAGE": options.slime_image,
                    "W8_GLM47_SLIME_PIN": SLIME_PIN,
                    "W8_GLM47_ARTIFACT_BUCKET": options.artifact_bucket,
                    "W8_GLM47_CACHE_VERSION": TASKS_CACHE_VERSION,
                    "W8_GLM47_TRAIN_LIMIT": str(options.train_limit),
                    "W8_GLM47_EVAL_LIMIT": str(options.eval_limit),
                    "W8_GLM47_MIN_TRAIN": str(options.min_train_tasks),
                    "W8_GLM47_MIN_VALIDATION": str(options.min_validation_tasks),
                    "W8_GLM47_MIN_TEST": str(options.min_test_tasks),
                    "W8_GLM47_COVERAGE_JOBS": str(options.coverage_jobs),
                    "W8_GLM47_GRPO_NUM_ROLLOUT": str(options.grpo_num_rollout),
                    "W8_GLM47_GRPO_ROLLOUT_BATCH_SIZE": str(options.grpo_rollout_batch_size),
                    "W8_GLM47_GRPO_GLOBAL_BATCH_SIZE": str(options.grpo_global_batch_size),
                    "W8_GLM47_EVAL_SAMPLES_PER_PROMPT": str(options.eval_samples_per_prompt),
                },
                secrets=_secrets(wandb_api_key=wandb_api_key, hf_token=hf_token),
            )
            task.set_resources(
                sky.Resources(
                    infra=f"gcp/{region}",
                    accelerators=options.accelerators,
                    use_spot=options.use_spot,
                    # The 30B model is staged four times (HF, torch_dist, SFT
                    # ckpt, SFT/GRPO HF export); the 256GB default boot disk
                    # runs out mid-export.
                    disk_size=options.disk_size,
                    labels=_resource_labels(options.run_id),
                )
            )
            try:
                request_id = sky.launch(
                    task,
                    cluster_name=options.cluster_name,
                    down=False,
                    no_setup=False,
                    _need_confirmation=False,
                )
                launch_result = sky.stream_and_get(request_id)
                # On API-server SkyPilot builds, launch resolves at job SUBMISSION,
                # not completion; block until the job actually reaches a terminal
                # state or teardown would kill a freshly started run.
                job_id = launch_result[0] if isinstance(launch_result, tuple) and launch_result else None
                if job_id is None:
                    raise LaunchError("SkyPilot launch returned no job id; cannot track run completion.")
                print(f"Job {job_id} submitted on {options.cluster_name}; waiting for terminal job state.", flush=True)
                final_status = _wait_for_job_completion(sky, options.cluster_name, job_id)
                if not final_status.endswith("SUCCEEDED"):
                    raise LaunchError(f"Remote GLM job {job_id} finished with status {final_status}.")
                print(f"Completed remote GLM run on {options.cluster_name} in {region}", flush=True)
                return True
            except KeyboardInterrupt:
                print("Interrupted during launch/run; checking for artifacts before teardown.", flush=True)
                if _cluster_exists(sky, options.cluster_name):
                    download_artifacts(
                        cluster_name=options.cluster_name,
                        run_id=options.run_id,
                        local_run_root=local_run_root,
                        local_cloud_root=local_cloud_root,
                    )
                    down_cluster(options.cluster_name)
                raise
            except Exception as exc:  # noqa: BLE001 - retry capacity/provisioning failures.
                print(f"Attempt failed in {region}: {exc!r}", flush=True)
                if _cluster_exists(sky, options.cluster_name):
                    print(
                        "Cluster exists after this failure; treating it as remote setup/training failure, not capacity miss.",
                        flush=True,
                    )
                    download_artifacts(
                        cluster_name=options.cluster_name,
                        run_id=options.run_id,
                        local_run_root=local_run_root,
                        local_cloud_root=local_cloud_root,
                    )
                    down_cluster(options.cluster_name)
                    raise
                down_cluster(options.cluster_name)
                if options.max_attempts and attempt >= options.max_attempts:
                    raise
        print(f"Sleeping {options.retry_sleep_seconds}s before next provisioning pass.", flush=True)
        time.sleep(options.retry_sleep_seconds)
    return False


def _wait_for_job_completion(sky: Any, cluster_name: str, job_id: int, *, poll_seconds: int = 30) -> str:
    """Stream job logs and poll status until the job reaches a terminal state."""

    while True:
        try:
            request_id = sky.tail_logs(cluster_name=cluster_name, job_id=job_id, follow=True)
            sky.stream_and_get(request_id)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - log streams drop on long runs; status poll is authoritative.
            print(f"Log stream interrupted ({exc!r}); re-checking job status.", flush=True)
        status_name = ""
        try:
            request_id = sky.job_status(cluster_name, job_ids=[job_id])
            statuses = sky.stream_and_get(request_id)
            status = statuses.get(job_id)
            status_name = getattr(status, "name", "") or (str(status) if status is not None else "")
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - transient API-server errors should not abort tracking.
            print(f"Job status check failed ({exc!r}); retrying in {poll_seconds}s.", flush=True)
        if status_name and any(status_name.endswith(suffix) for suffix in TERMINAL_JOB_STATUS_SUFFIXES):
            return status_name
        time.sleep(poll_seconds)


def build_setup_script() -> str:
    return textwrap.dedent(
        """\
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        # SkyPilot GPU images preinstall docker-ce/containerd.io; a blanket
        # docker.io install conflicts with them, so only install missing tools.
        missing_packages=""
        command -v docker >/dev/null 2>&1 || missing_packages="$missing_packages docker.io"
        command -v curl >/dev/null 2>&1 || missing_packages="$missing_packages curl"
        command -v git >/dev/null 2>&1 || missing_packages="$missing_packages git"
        command -v rsync >/dev/null 2>&1 || missing_packages="$missing_packages rsync"
        if [ -n "$missing_packages" ] && command -v apt-get >/dev/null 2>&1; then
          sudo apt-get update
          # shellcheck disable=SC2086
          sudo DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates $missing_packages
        fi
        if command -v systemctl >/dev/null 2>&1; then
          sudo systemctl enable --now docker || true
        fi
        sudo chmod 666 /var/run/docker.sock || true
        if ! command -v uv >/dev/null 2>&1; then
          curl -LsSf https://astral.sh/uv/install.sh | sh
        fi
        export PATH="$HOME/.local/bin:$PATH"
        """
    )


def build_run_script() -> str:
    repo_mount = "/workspace/$(basename \"$PWD\")"
    remote_cloud_root = '${PWD}/.w8-biayn/slime/glm47-cpp-perf/cloud-runs/${W8_GLM47_RUN_ID}'
    remote_run_root = '${PWD}/.w8-biayn/slime/glm47-cpp-perf/runs/${W8_GLM47_RUN_ID}'
    remote_export_root = '${HOME}/w8-glm47-artifacts/${W8_GLM47_RUN_ID}'
    # The container script is substituted AFTER dedent: interpolating a
    # column-0 block into the f-string first would make dedent a no-op, leave
    # the heredoc terminator indented, and let the heredoc swallow the rest of
    # the script (the docker run included) into the entrypoint file.
    container_script = "@W8_CONTAINER_SCRIPT@"
    template = textwrap.dedent(
        f"""\
        set -euo pipefail
        export PATH="$HOME/.local/bin:$PATH"
        export W8_REMOTE_CLOUD_ROOT="{remote_cloud_root}"
        export W8_REMOTE_RUN_ROOT="{remote_run_root}"
        export W8_REMOTE_EXPORT_ROOT="{remote_export_root}"
        mkdir -p "$W8_REMOTE_CLOUD_ROOT" "$W8_REMOTE_EXPORT_ROOT"
        exec > >(tee -a "$W8_REMOTE_CLOUD_ROOT/remote_host.log") 2>&1

        collect_artifacts() {{
          set +e
          mkdir -p "$W8_REMOTE_EXPORT_ROOT/run" "$W8_REMOTE_EXPORT_ROOT/cloud"
          if [ -d "$W8_REMOTE_RUN_ROOT" ]; then
            cp -a "$W8_REMOTE_RUN_ROOT/." "$W8_REMOTE_EXPORT_ROOT/run/"
          fi
          if [ -d "$W8_REMOTE_CLOUD_ROOT" ]; then
            cp -a "$W8_REMOTE_CLOUD_ROOT/." "$W8_REMOTE_EXPORT_ROOT/cloud/"
          fi
          mkdir -p "$W8_REMOTE_EXPORT_ROOT/data-build"
          for evidence in .w8-biayn/data/pie-full/coverage-report.json .w8-biayn/data/pie-full/coverage.json .w8-biayn/data/tasks-full/_w8_task_build_report.json; do
            [ -f "$evidence" ] && cp "$evidence" "$W8_REMOTE_EXPORT_ROOT/data-build/" || true
          done
          find "$W8_REMOTE_EXPORT_ROOT" -maxdepth 3 -type f | sort > "$W8_REMOTE_EXPORT_ROOT/artifact_manifest.txt" || true
        }}
        trap collect_artifacts EXIT

        echo "remote_pwd=$PWD"
        echo "run_id=$W8_GLM47_RUN_ID"
        echo "slime_image=$W8_GLM47_SLIME_IMAGE"
        nvidia-smi -L

        # Best-effort pipeline milestones into the <run-id>-pipeline W&B run.
        w8_milestone() {{
          uv run --with wandb python scripts/wandb_milestone.py \
            --project "$W8_GLM47_WANDB_PROJECT" --run-id "$W8_GLM47_RUN_ID" --event "$1" >/dev/null 2>&1 || true
        }}

        ./scripts/bootstrap.sh
        uv run w8-biayn data doctor
        uv run w8-biayn upstreams clone slime
        uv run w8-biayn slime doctor
        uv run w8-biayn cpp harness preflight --cpu 3
        w8_milestone host_preflight_done

        # Structural dataset cache: a deterministic, project-scoped,
        # gate-keyed GCS path shared by everyone in the project. Restore first;
        # only build (and re-upload) on a cache miss. The gate values are part
        # of the key so different admission thresholds never collide.
        W8_TASKS_CACHE_PREFIX="cache/${{W8_GLM47_CACHE_VERSION}}/tasks-full/mintrain${{W8_GLM47_MIN_TRAIN}}-minval${{W8_GLM47_MIN_VALIDATION}}-mintest${{W8_GLM47_MIN_TEST}}"
        if [ ! -d .w8-biayn/data/tasks-full ]; then
          w8_milestone data_restore_started
          if uv run w8-biayn data cache restore \
               --path .w8-biayn/data/tasks-full \
               --gcs-prefix "gs://${{W8_GLM47_ARTIFACT_BUCKET}}/${{W8_TASKS_CACHE_PREFIX}}"; then
            w8_milestone data_restore_hit
          else
            echo "tasks-full cache miss; building from PIE" >&2
            rm -rf .w8-biayn/data/tasks-full
          fi
        fi
        if [ ! -d .w8-biayn/data/tasks-full ]; then
          w8_milestone data_build_started
          uv run w8-biayn data pie download --out .w8-biayn/data/pie
          uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
          uv run w8-biayn data pie measure-coverage \
            --prepared-root .w8-biayn/data/pie-full \
            --out .w8-biayn/data/pie-full/coverage.json \
            --report-out .w8-biayn/data/pie-full/coverage-report.json \
            --jobs "$W8_GLM47_COVERAGE_JOBS"
          uv run w8-biayn data pie build-full-tasks \
            --prepared-root .w8-biayn/data/pie-full \
            --coverage-json .w8-biayn/data/pie-full/coverage.json \
            --out .w8-biayn/data/tasks-full \
            --min-train "$W8_GLM47_MIN_TRAIN" \
            --min-validation "$W8_GLM47_MIN_VALIDATION" \
            --min-test "$W8_GLM47_MIN_TEST" \
            --force
          # Populate the shared cache for the next run/user. Best-effort: a
          # node without bucket-write permission must not fail the training run.
          uv run w8-biayn data cache upload \
            --path .w8-biayn/data/tasks-full \
            --gcs-prefix "gs://${{W8_GLM47_ARTIFACT_BUCKET}}/${{W8_TASKS_CACHE_PREFIX}}" || \
            echo "tasks-full cache upload skipped (no bucket write permission?)" >&2
        fi
        w8_milestone data_build_done

        uv run w8-biayn slime setup --force --image "$W8_GLM47_SLIME_IMAGE"
        w8_milestone slime_setup_done
        cat > "$W8_REMOTE_CLOUD_ROOT/container_entrypoint.sh" <<'W8_GLM47_CONTAINER'
        {container_script}
        W8_GLM47_CONTAINER
        chmod +x "$W8_REMOTE_CLOUD_ROOT/container_entrypoint.sh"

        docker pull "$W8_GLM47_SLIME_IMAGE"
        w8_milestone container_image_pulled
        docker rm -f "w8-slime-glm47-$W8_GLM47_RUN_ID" >/dev/null 2>&1 || true
        docker run --rm --gpus all --ipc=host --shm-size=16g \
          --ulimit stack=67108864 \
          --name "w8-slime-glm47-$W8_GLM47_RUN_ID" \
          -v "$PWD":{repo_mount} \
          -v /var/run/docker.sock:/var/run/docker.sock \
          -v /tmp:/tmp \
          -e WANDB_API_KEY \
          -e WANDB_ENTITY="$W8_GLM47_WANDB_ENTITY" \
          -e WANDB_BASE_URL="$W8_GLM47_WANDB_BASE_URL" \
          -e HF_TOKEN \
          -e HUGGING_FACE_HUB_TOKEN="${{HF_TOKEN:-}}" \
          -e W8_GLM47_RUN_ID \
          -e W8_GLM47_SLIME_PIN \
          -e W8_GLM47_REPO_DIR={repo_mount} \
          -e W8_GLM47_WANDB_PROJECT \
          -e W8_GLM47_WANDB_GROUP \
          -e W8_GLM47_TRAIN_LIMIT \
          -e W8_GLM47_EVAL_LIMIT \
          -e W8_GLM47_MIN_TRAIN \
          -e W8_GLM47_MIN_VALIDATION \
          -e W8_GLM47_MIN_TEST \
          -e W8_GLM47_GRPO_NUM_ROLLOUT \
          -e W8_GLM47_GRPO_ROLLOUT_BATCH_SIZE \
          -e W8_GLM47_GRPO_GLOBAL_BATCH_SIZE \
          -e W8_GLM47_EVAL_SAMPLES_PER_PROMPT \
          -e SLIME_NOFILE_SOFT_LIMIT="${{SLIME_NOFILE_SOFT_LIMIT:-65536}}" \
          -w {repo_mount} \
          "$W8_GLM47_SLIME_IMAGE" \
          /bin/bash -lc "bash {repo_mount}/.w8-biayn/slime/glm47-cpp-perf/cloud-runs/$W8_GLM47_RUN_ID/container_entrypoint.sh" \
          2>&1 | tee "$W8_REMOTE_CLOUD_ROOT/container_outer.log"

        # The job only counts as succeeded when the lane actually produced its
        # final comparison artifact; a truncated script must not exit 0.
        test -f "$W8_REMOTE_RUN_ROOT/eval/comparison.json"
        """
    )
    return template.replace("@W8_CONTAINER_SCRIPT@", build_container_script().rstrip())


def build_container_script() -> str:
    return textwrap.dedent(
        """\
        set -euo pipefail
        ulimit -Sn "${SLIME_NOFILE_SOFT_LIMIT:-65536}" 2>/dev/null || true
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}
        # wandb's pydantic Settings validates any WANDB_* env var that is
        # merely present; an empty WANDB_BASE_URL/WANDB_ENTITY crashes it.
        [ -z "${WANDB_ENTITY:-}" ] && unset WANDB_ENTITY || true
        [ -z "${WANDB_BASE_URL:-}" ] && unset WANDB_BASE_URL || true
        if ! command -v docker >/dev/null 2>&1; then
          if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io rsync
          else
            echo "docker CLI is required inside the SLIME container for C++ rewards" >&2
            exit 2
          fi
        fi
        docker version
        # Fail fast when W&B capture cannot work; silent no-metric runs are
        # worse than an early abort on a paid node.
        python - <<'W8_WANDB_CHECK'
        import os
        import sys

        key = os.environ.get("WANDB_API_KEY", "")
        if not key:
            print("WANDB_API_KEY missing inside the training container", file=sys.stderr)
            sys.exit(2)
        import wandb

        viewer = wandb.Api(timeout=30).viewer
        print(f"wandb_auth_ok entity={viewer.entity}")
        W8_WANDB_CHECK
        python "$W8_GLM47_REPO_DIR/scripts/wandb_milestone.py" \
          --project "$W8_GLM47_WANDB_PROJECT" --run-id "$W8_GLM47_RUN_ID" --event container_wandb_ok >/dev/null 2>&1 || true
        cd /root/slime
        # Pin the container's SLIME to the repo pin instead of drifting to HEAD.
        git fetch origin "$W8_GLM47_SLIME_PIN"
        git checkout "$W8_GLM47_SLIME_PIN"
        pip install -e . --no-deps

        # SLIME's bundled megatron.patch allows partial dist-ckpt loads (skips
        # keys absent from the checkpoint, e.g. TE _extra_state objects that
        # HF conversion never writes), but skipped object entries leave their
        # bare BytesIO placeholders unlisted and
        # _replace_sharded_keys_with_state_dict_keys crashes on len(). Wrap
        # bare placeholders; genuine mismatches still assert.
        python - <<'W8_MEGATRON_PATCH'
        from pathlib import Path

        path = Path("/root/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py")
        src = path.read_text()
        guard = "# w8: bare object placeholder from partial load"
        if guard not in src:
            anchor = (
                "    for k, tensors in state_dict.items():\\n"
                "        assert len(tensors) == len(rename_mapping[k])"
            )
            replacement = (
                "    for k, tensors in state_dict.items():\\n"
                "        if not isinstance(tensors, list):  # w8: bare object placeholder from partial load\\n"
                "            # An empty BytesIO means the partial load skipped this object\\n"
                "            # (e.g. TE _extra_state absent from a converted checkpoint);\\n"
                "            # pass None so TE's set_extra_state guard skips it instead of\\n"
                "            # choking on a placeholder.\\n"
                "            if hasattr(tensors, 'getbuffer') and tensors.getbuffer().nbytes == 0:\\n"
                "                tensors = [None]\\n"
                "            else:\\n"
                "                tensors = [tensors]\\n"
                "        assert len(tensors) == len(rename_mapping[k])"
            )
            assert anchor in src, "w8 megatron patch anchor missing; inspect torch.py before proceeding"
            src = src.replace(anchor, replacement, 1)
            # TE set_extra_state requires tensor-format state and calls
            # .numel() on anything non-None, but bridge-based conversion
            # writes BytesIO-format extra_state. Null every _extra_state
            # entry so TE's `if state is None: return` guard skips them;
            # FP8/quantizer metadata is irrelevant for BF16 training.
            return_anchor = "    return unflatten_state_dict(recovered_sd, flat_mapping)"
            return_replacement = (
                "    recovered_sd = {  # w8: drop converter extra_state formats TE cannot read\\n"
                "        k: (None if '_extra_state' in k else v) for k, v in recovered_sd.items()\\n"
                "    }\\n"
                "    return unflatten_state_dict(recovered_sd, flat_mapping)"
            )
            assert return_anchor in src, "w8 megatron patch return anchor missing"
            path.write_text(src.replace(return_anchor, return_replacement, 1))
        print("w8 megatron partial-object patch active")
        W8_MEGATRON_PATCH
        python train.py --help >/tmp/slime-train-help.txt
        sed -n '1,80p' /tmp/slime-train-help.txt

        cd "${W8_GLM47_REPO_DIR:?missing repo mount}"
        export SLIME_RUN_ID="$W8_GLM47_RUN_ID"
        export SLIME_CPP_TASKS_DIR="${W8_GLM47_REPO_DIR}/.w8-biayn/data/tasks-full"
        export SLIME_CPP_TRAIN_LIMIT="$W8_GLM47_TRAIN_LIMIT"
        export SLIME_CPP_EVAL_LIMIT="$W8_GLM47_EVAL_LIMIT"
        export SLIME_HF_CHECKPOINT="${W8_GLM47_REPO_DIR}/.w8-biayn/models/GLM-4.7-Flash"
        export SLIME_REF_LOAD_DIR="${W8_GLM47_REPO_DIR}/.w8-biayn/models/GLM-4.7-Flash_torch_dist"
        export SLIME_DOWNLOAD_HF_CHECKPOINT=1
        export SLIME_CONVERT_IF_MISSING=1
        export SLIME_CONVERT_NPROC=8
        export SLIME_NUM_GPUS=8
        export SLIME_WANDB_PROJECT="$W8_GLM47_WANDB_PROJECT"
        export SLIME_WANDB_GROUP="$W8_GLM47_WANDB_GROUP"
        export SLIME_GRPO_NUM_ROLLOUT="$W8_GLM47_GRPO_NUM_ROLLOUT"
        export SLIME_GRPO_SKIP_WEIGHT_UPDATE=0
        export SLIME_GRPO_LOAD_WEIGHTS_ONLY=1
        export SLIME_GRPO_ROLLOUT_BATCH_SIZE="$W8_GLM47_GRPO_ROLLOUT_BATCH_SIZE"
        export SLIME_GRPO_GLOBAL_BATCH_SIZE="$W8_GLM47_GRPO_GLOBAL_BATCH_SIZE"
        export SLIME_GRPO_N_SAMPLES_PER_PROMPT=1
        export SLIME_EVAL_N_SAMPLES_PER_PROMPT="$W8_GLM47_EVAL_SAMPLES_PER_PROMPT"
        export SLIME_SAVE_HF_EXPORTS=1
        export SLIME_STANDALONE_HF_EXPORTS=1
        export SLIME_INLINE_SAVE_HF_EXPORTS=0
        export SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP=1
        export SLIME_GRPO_SKIP_FINAL_TRAIN_SLEEP=1
        export WANDB_MODE="${WANDB_MODE:-online}"

        bash examples/slime/glm47_cpp_perf/prepare_data.sh
        bash examples/slime/glm47_cpp_perf/eval_base.sh
        bash examples/slime/glm47_cpp_perf/sft.sh
        bash examples/slime/glm47_cpp_perf/eval_sft.sh
        bash examples/slime/glm47_cpp_perf/grpo.sh
        bash examples/slime/glm47_cpp_perf/eval_grpo.sh
        bash examples/slime/glm47_cpp_perf/compare.sh
        """
    )


def download_artifacts(*, cluster_name: str, run_id: str, local_run_root: Path, local_cloud_root: Path) -> None:
    remote_root = f"~/w8-glm47-artifacts/{run_id}"
    print(f"\nDownloading remote artifacts from {cluster_name}:{remote_root}", flush=True)
    local_run_root.mkdir(parents=True, exist_ok=True)
    local_cloud_root.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        _run(["rsync", "-az", f"{cluster_name}:{remote_root}/run/", f"{local_run_root}/"], check=False)
        _run(["rsync", "-az", f"{cluster_name}:{remote_root}/cloud/", f"{local_cloud_root}/remote/"], check=False)
        _run(
            ["rsync", "-az", f"{cluster_name}:{remote_root}/artifact_manifest.txt", f"{local_cloud_root}/artifact_manifest.txt"],
            check=False,
        )
    else:
        _run(["scp", "-r", f"{cluster_name}:{remote_root}/run/.", str(local_run_root)], check=False)
        remote_cloud = local_cloud_root / "remote"
        remote_cloud.mkdir(parents=True, exist_ok=True)
        _run(["scp", "-r", f"{cluster_name}:{remote_root}/cloud/.", str(remote_cloud)], check=False)
        _run(["scp", f"{cluster_name}:{remote_root}/artifact_manifest.txt", str(local_cloud_root / "artifact_manifest.txt")], check=False)


def down_cluster(cluster_name: str) -> None:
    try:
        sky = import_sky()

        print(f"\nTearing down cluster {cluster_name}", flush=True)
        request_id = sky.down(cluster_name)
        sky.stream_and_get(request_id)
    except Exception as exc:  # noqa: BLE001 - cleanup is best effort.
        print(f"Cluster teardown failed or cluster was absent: {exc!r}", flush=True)


def _cluster_exists(sky: Any, cluster_name: str) -> bool:
    try:
        request_id = sky.status(cluster_names=[cluster_name])
        statuses = sky.stream_and_get(request_id)
    except Exception as exc:  # noqa: BLE001 - absence checks should not mask cleanup.
        print(f"Could not refresh SkyPilot cluster status for {cluster_name}: {exc!r}", flush=True)
        return False
    return bool(statuses)


def _resource_labels(run_id: str) -> dict[str, str]:
    return {
        "project": "w8-biayn",
        "phase": "training",
        "pipeline": "glm47-cpp-perf",
        "run_id": _label_safe(run_id),
        "owner": _label_safe(os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"),
        "ttl": "down-after-job",
    }


def _label_safe(value: str) -> str:
    chars = []
    for ch in value.lower():
        if "a" <= ch <= "z" or "0" <= ch <= "9":
            chars.append(ch)
        else:
            chars.append("-")
    normalized = "-".join(part for part in "".join(chars).split("-") if part)
    if not normalized or not ("a" <= normalized[0] <= "z"):
        normalized = "x-" + (normalized or "unknown")
    return normalized[:63].strip("-") or "unknown"


def _secrets(*, wandb_api_key: str, hf_token: str) -> dict[str, str]:
    secrets = {"WANDB_API_KEY": wandb_api_key}
    if hf_token:
        secrets["HF_TOKEN"] = hf_token
        secrets["HUGGING_FACE_HUB_TOKEN"] = hf_token
    return secrets


def _write_config(path: Path, *, options: LaunchOptions) -> None:
    payload: dict[str, Any] = {
        key: value for key, value in vars(options).items() if key not in ("wandb_api_key", "hf_token")
    }
    payload["wandb_api_key"] = "<redacted>" if options.wandb_api_key else ""
    payload["hf_token"] = "<redacted>" if options.hf_token else ""
    payload["labels"] = _resource_labels(options.run_id)
    payload["skypilot_pin"] = SKYPILOT_PIN
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    return subprocess.run(command, check=check, text=True)


def _cluster_safe(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized[:40] or default_run_id()


def default_run_id() -> str:
    return "glm47full-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def default_cluster_name(run_id: str) -> str:
    """The SkyPilot cluster name for a run id (shared by launch and teardown)."""

    return f"w8-glm47-h100-{_cluster_safe(run_id)}"


def run_label_value(run_id: str) -> str:
    """The GCE `run_id` label value SkyPilot applies (matches the W&B group)."""

    return _label_safe(run_id)
