#!/usr/bin/env python3
"""Provision one GCP H100:8 node and run the full GLM C++ SLIME lane.

This is a local orchestrator. It loops over allowed GCP regions until
SkyPilot can provision an H100:8 instance, runs the existing GLM lane inside
the repo-owned SLIME container, downloads local run artifacts back to this
machine, and then tears the cluster down.
"""

from __future__ import annotations

import argparse
import contextlib
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


SKYPILOT_PIN = "skypilot-nightly[gcp]==1.0.0.dev20260516"
ALLOWED_REGIONS = ("asia-southeast1", "asia-south1", "asia-south2")
DEFAULT_REGION = "asia-southeast1"
DEFAULT_ACCELERATORS = "H100:8"
DEFAULT_SLIME_IMAGE = "slimerl/slime:latest"
DEFAULT_WANDB_PROJECT = "slime-glm47-cpp-perf"
DEFAULT_LOCAL_ROOT = Path(".w8-biayn/slime/glm47-cpp-perf")
FULL_LIMIT_SENTINEL = 1_000_000


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=_default_run_id(), help="Stable run id for cluster, W&B group, and artifacts.")
    parser.add_argument(
        "--region",
        action="append",
        choices=ALLOWED_REGIONS,
        help=(
            "Allowed GCP region to try. Repeat to add regions. "
            f"Default: {DEFAULT_REGION}. Choices are restricted to {', '.join(ALLOWED_REGIONS)}."
        ),
    )
    parser.add_argument("--accelerators", default=DEFAULT_ACCELERATORS, help="SkyPilot accelerator request.")
    parser.add_argument("--retry-sleep-seconds", type=int, default=300, help="Sleep between provisioning passes.")
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means retry forever until capacity is acquired.")
    parser.add_argument(
        "--use-spot",
        action="store_true",
        help="Request spot (preemptible) capacity instead of on-demand. Needed when the project only holds preemptible GPU quota.",
    )
    parser.add_argument("--cluster-name", default="", help="Override cluster name. Defaults to w8-glm47-h100-<run-id>.")
    parser.add_argument("--slime-image", default=DEFAULT_SLIME_IMAGE, help="SLIME Docker image for the GPU container.")
    parser.add_argument("--wandb-api-key", default="", help="W&B API key. Prefer WANDB_API_KEY env or --wandb-api-key-file.")
    parser.add_argument("--wandb-api-key-file", default="", help="File containing the W&B API key.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT, help="W&B project name.")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""), help="Optional W&B entity.")
    parser.add_argument("--wandb-base-url", default=os.environ.get("WANDB_BASE_URL", ""), help="Optional W&B base URL.")
    parser.add_argument("--hf-token", default="", help="Optional Hugging Face token. Prefer HF_TOKEN env when possible.")
    parser.add_argument("--hf-token-file", default="", help="File containing an optional Hugging Face token.")
    parser.add_argument(
        "--local-output-root",
        default=str(DEFAULT_LOCAL_ROOT),
        help="Local ignored artifact root for cloud logs and downloaded run results.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=FULL_LIMIT_SENTINEL,
        help="SLIME C++ train row cap. Default is intentionally huge to include all admitted train tasks.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=FULL_LIMIT_SENTINEL,
        help="SLIME C++ eval row cap. Default is intentionally huge to include all admitted eval tasks.",
    )
    parser.add_argument("--grpo-num-rollout", type=int, default=8, help="GRPO rollout count for the full run.")
    parser.add_argument("--grpo-rollout-batch-size", type=int, default=4, help="GRPO rollout batch size.")
    parser.add_argument("--grpo-global-batch-size", type=int, default=4, help="GRPO global batch size.")
    parser.add_argument("--eval-samples-per-prompt", type=int, default=2, help="Eval samples per held-out prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned SkyPilot task and exit without launching.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    regions = args.region or [DEFAULT_REGION]
    cluster_name = args.cluster_name or f"w8-glm47-h100-{_cluster_safe(args.run_id)}"
    wandb_api_key = _resolve_secret(
        direct=args.wandb_api_key,
        file_path=args.wandb_api_key_file,
        env_name="WANDB_API_KEY",
        required=not args.dry_run,
    )
    hf_token = _resolve_secret(
        direct=args.hf_token,
        file_path=args.hf_token_file,
        env_name="HF_TOKEN",
        required=False,
    )

    local_root = Path(args.local_output_root)
    local_cloud_root = local_root / "cloud-runs" / args.run_id
    local_run_root = local_root / "runs" / args.run_id
    local_cloud_root.mkdir(parents=True, exist_ok=True)
    local_run_root.mkdir(parents=True, exist_ok=True)
    log_path = local_cloud_root / "orchestrator.log"
    config_path = local_cloud_root / "launch_config.json"
    _write_config(config_path, args=args, regions=regions, cluster_name=cluster_name)

    setup_script = build_setup_script()
    run_script = build_run_script()

    if args.dry_run:
        print(json.dumps({"cluster": cluster_name, "regions": regions, "config": str(config_path)}, indent=2))
        print("\n--- setup ---")
        print(setup_script)
        print("\n--- run ---")
        print(run_script)
        return 0

    cluster_acquired = False
    training_status = 1
    with log_path.open("a", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"run_id={args.run_id}")
            print(f"cluster={cluster_name}")
            print(f"regions={','.join(regions)}")
            print(f"local_run_root={local_run_root}")
            try:
                cluster_acquired = acquire_and_run(
                    args=args,
                    regions=regions,
                    cluster_name=cluster_name,
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
            except Exception as exc:  # noqa: BLE001 - this is a top-level orchestrator.
                print(f"\nTraining/provisioning failed: {exc!r}", flush=True)
                training_status = 1
            finally:
                if cluster_acquired:
                    download_artifacts(
                        cluster_name=cluster_name,
                        run_id=args.run_id,
                        local_run_root=local_run_root,
                        local_cloud_root=local_cloud_root,
                    )
                    down_cluster(cluster_name)
                else:
                    print("No cluster was acquired; nothing to tear down.", flush=True)

            print(f"orchestrator_log={log_path}")
            print(f"downloaded_run_root={local_run_root}")
            print(f"cloud_log_root={local_cloud_root}")
            comparison = local_run_root / "eval" / "comparison.json"
            if comparison.exists():
                print(f"comparison_json={comparison}")
    return training_status


def acquire_and_run(
    *,
    args: argparse.Namespace,
    regions: list[str],
    cluster_name: str,
    setup_script: str,
    run_script: str,
    local_run_root: Path,
    local_cloud_root: Path,
    wandb_api_key: str,
    hf_token: str,
) -> bool:
    sky = _import_sky()

    attempt = 0
    while args.max_attempts == 0 or attempt < args.max_attempts:
        for region in regions:
            attempt += 1
            print(f"\n=== Provision attempt {attempt}: {args.accelerators} in {region} ===", flush=True)
            task = sky.Task(
                name=f"glm47-cpp-full-{args.run_id}",
                setup=setup_script,
                run=run_script,
                workdir=".",
                envs={
                    "W8_GLM47_RUN_ID": args.run_id,
                    "W8_GLM47_WANDB_PROJECT": args.wandb_project,
                    "W8_GLM47_WANDB_GROUP": args.run_id,
                    "W8_GLM47_WANDB_ENTITY": args.wandb_entity,
                    "W8_GLM47_WANDB_BASE_URL": args.wandb_base_url,
                    "W8_GLM47_SLIME_IMAGE": args.slime_image,
                    "W8_GLM47_TRAIN_LIMIT": str(args.train_limit),
                    "W8_GLM47_EVAL_LIMIT": str(args.eval_limit),
                    "W8_GLM47_GRPO_NUM_ROLLOUT": str(args.grpo_num_rollout),
                    "W8_GLM47_GRPO_ROLLOUT_BATCH_SIZE": str(args.grpo_rollout_batch_size),
                    "W8_GLM47_GRPO_GLOBAL_BATCH_SIZE": str(args.grpo_global_batch_size),
                    "W8_GLM47_EVAL_SAMPLES_PER_PROMPT": str(args.eval_samples_per_prompt),
                },
                secrets=_secrets(wandb_api_key=wandb_api_key, hf_token=hf_token),
            )
            task.set_resources(
                sky.Resources(
                    infra=f"gcp/{region}",
                    accelerators=args.accelerators,
                    use_spot=args.use_spot,
                    labels=_resource_labels(args.run_id),
                )
            )
            try:
                request_id = sky.launch(
                    task,
                    cluster_name=cluster_name,
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
                    raise RuntimeError("SkyPilot launch returned no job id; cannot track run completion.")
                print(f"Job {job_id} submitted on {cluster_name}; waiting for terminal job state.", flush=True)
                final_status = _wait_for_job_completion(sky, cluster_name, job_id)
                if not final_status.endswith("SUCCEEDED"):
                    raise RuntimeError(f"Remote GLM job {job_id} finished with status {final_status}.")
                print(f"Completed remote GLM run on {cluster_name} in {region}", flush=True)
                return True
            except KeyboardInterrupt:
                print("Interrupted during launch/run; checking for artifacts before teardown.", flush=True)
                if _cluster_exists(sky, cluster_name):
                    download_artifacts(
                        cluster_name=cluster_name,
                        run_id=args.run_id,
                        local_run_root=local_run_root,
                        local_cloud_root=local_cloud_root,
                    )
                    down_cluster(cluster_name)
                raise
            except Exception as exc:  # noqa: BLE001 - retry capacity/provisioning failures.
                print(f"Attempt failed in {region}: {exc!r}", flush=True)
                if _cluster_exists(sky, cluster_name):
                    print(
                        "Cluster exists after this failure; treating it as remote setup/training failure, not capacity miss.",
                        flush=True,
                    )
                    download_artifacts(
                        cluster_name=cluster_name,
                        run_id=args.run_id,
                        local_run_root=local_run_root,
                        local_cloud_root=local_cloud_root,
                    )
                    down_cluster(cluster_name)
                    raise
                down_cluster(cluster_name)
                if args.max_attempts and attempt >= args.max_attempts:
                    raise
        print(f"Sleeping {args.retry_sleep_seconds}s before next provisioning pass.", flush=True)
        time.sleep(args.retry_sleep_seconds)
    return False


TERMINAL_JOB_STATUS_SUFFIXES = ("SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED")


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
    container_script = build_container_script()
    return textwrap.dedent(
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
          find "$W8_REMOTE_EXPORT_ROOT" -maxdepth 3 -type f | sort > "$W8_REMOTE_EXPORT_ROOT/artifact_manifest.txt" || true
        }}
        trap collect_artifacts EXIT

        echo "remote_pwd=$PWD"
        echo "run_id=$W8_GLM47_RUN_ID"
        echo "slime_image=$W8_GLM47_SLIME_IMAGE"
        nvidia-smi -L

        ./scripts/bootstrap.sh
        uv run w8-biayn data doctor
        uv run w8-biayn upstreams clone slime
        uv run w8-biayn slime doctor
        uv run w8-biayn cpp harness preflight --cpu 3

        if [ ! -d .w8-biayn/data/tasks-full ]; then
          uv run w8-biayn data pie download --out .w8-biayn/data/pie
          uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
          uv run w8-biayn data pie measure-coverage \
            --prepared-root .w8-biayn/data/pie-full \
            --out .w8-biayn/data/pie-full/coverage.json \
            --report-out .w8-biayn/data/pie-full/coverage-report.json
          uv run w8-biayn data pie build-full-tasks \
            --prepared-root .w8-biayn/data/pie-full \
            --coverage-json .w8-biayn/data/pie-full/coverage.json \
            --out .w8-biayn/data/tasks-full \
            --min-train 1000 \
            --min-validation 100 \
            --min-test 100 \
            --force
        fi

        uv run w8-biayn slime setup --force --image "$W8_GLM47_SLIME_IMAGE"
        cat > "$W8_REMOTE_CLOUD_ROOT/container_entrypoint.sh" <<'W8_GLM47_CONTAINER'
        {container_script}
        W8_GLM47_CONTAINER
        chmod +x "$W8_REMOTE_CLOUD_ROOT/container_entrypoint.sh"

        docker pull "$W8_GLM47_SLIME_IMAGE"
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
          -e W8_GLM47_REPO_DIR={repo_mount} \
          -e W8_GLM47_WANDB_PROJECT \
          -e W8_GLM47_WANDB_GROUP \
          -e W8_GLM47_TRAIN_LIMIT \
          -e W8_GLM47_EVAL_LIMIT \
          -e W8_GLM47_GRPO_NUM_ROLLOUT \
          -e W8_GLM47_GRPO_ROLLOUT_BATCH_SIZE \
          -e W8_GLM47_GRPO_GLOBAL_BATCH_SIZE \
          -e W8_GLM47_EVAL_SAMPLES_PER_PROMPT \
          -e SLIME_NOFILE_SOFT_LIMIT="${{SLIME_NOFILE_SOFT_LIMIT:-65536}}" \
          -w {repo_mount} \
          "$W8_GLM47_SLIME_IMAGE" \
          /bin/bash -lc "bash {repo_mount}/.w8-biayn/slime/glm47-cpp-perf/cloud-runs/$W8_GLM47_RUN_ID/container_entrypoint.sh" \
          2>&1 | tee "$W8_REMOTE_CLOUD_ROOT/container_outer.log"
        """
    )


def build_container_script() -> str:
    return textwrap.dedent(
        """\
        set -euo pipefail
        ulimit -Sn "${SLIME_NOFILE_SOFT_LIMIT:-65536}" 2>/dev/null || true
        export PYTHONUNBUFFERED=1
        export PYTHONPATH=/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}
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
        cd /root/slime
        git pull
        pip install -e . --no-deps
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
        _run(["rsync", "-az", f"{cluster_name}:{remote_root}/artifact_manifest.txt", f"{local_cloud_root}/artifact_manifest.txt"], check=False)
    else:
        _run(["scp", "-r", f"{cluster_name}:{remote_root}/run/.", str(local_run_root)], check=False)
        remote_cloud = local_cloud_root / "remote"
        remote_cloud.mkdir(parents=True, exist_ok=True)
        _run(["scp", "-r", f"{cluster_name}:{remote_root}/cloud/.", str(remote_cloud)], check=False)
        _run(["scp", f"{cluster_name}:{remote_root}/artifact_manifest.txt", str(local_cloud_root / "artifact_manifest.txt")], check=False)


def down_cluster(cluster_name: str) -> None:
    try:
        sky = _import_sky()

        print(f"\nTearing down cluster {cluster_name}", flush=True)
        request_id = sky.down(cluster_name)
        sky.stream_and_get(request_id)
    except Exception as exc:  # noqa: BLE001 - cleanup is best effort.
        print(f"Cluster teardown failed or cluster was absent: {exc!r}", flush=True)


def _import_sky() -> Any:
    try:
        import sky
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SkyPilot with GCP support is required. Run: "
            f"uv run --with '{SKYPILOT_PIN}' python examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py "
            "(keep the pin matched to any locally running sky API server)"
        ) from exc
    return sky


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


def _resolve_secret(*, direct: str, file_path: str, env_name: str, required: bool) -> str:
    value = direct or os.environ.get(env_name, "")
    if file_path:
        value = Path(file_path).read_text(encoding="utf-8").strip()
    if required and not value:
        option = env_name.lower().replace("_", "-")
        raise SystemExit(f"Missing required secret: pass --{option} or set {env_name}.")
    return value


def _secrets(*, wandb_api_key: str, hf_token: str) -> dict[str, str]:
    secrets = {"WANDB_API_KEY": wandb_api_key}
    if hf_token:
        secrets["HF_TOKEN"] = hf_token
        secrets["HUGGING_FACE_HUB_TOKEN"] = hf_token
    return secrets


def _write_config(path: Path, *, args: argparse.Namespace, regions: list[str], cluster_name: str) -> None:
    payload: dict[str, Any] = vars(args).copy()
    payload["wandb_api_key"] = "<redacted>" if payload.get("wandb_api_key") else ""
    payload["hf_token"] = "<redacted>" if payload.get("hf_token") else ""
    payload["regions"] = regions
    payload["cluster_name"] = cluster_name
    payload["labels"] = _resource_labels(args.run_id)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    return subprocess.run(command, check=check, text=True)


def _cluster_safe(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized[:40] or _default_run_id()


def _default_run_id() -> str:
    return "glm47full-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
