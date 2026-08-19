#!/usr/bin/env python3
"""Four official fixed26 trials with one compiler-feedback repair turn."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


RUN_ID = "issue111-execution-bank-rl-v2-think-r2-iter19-fixed26-mt2-4x-20260818"
SOURCE = Path(
    "/workspace/local-fixed26/runs/execution-bank-RL-v2-think-r2/"
    "checkpoints/grpo_lora_r16/iter_0000019/adapter"
)
SERVING = SOURCE.parent / "adapter-fixed26-serving"
ARTIFACT_ROOT = Path(
    "/workspace/runs/execution-bank-RL-v2-think-r2/"
    "fixed26-mt2-4x-20260818"
)
SCRATCH = Path(f"/tmp/{RUN_ID}")
MODEL_PATH = Path("/workspace/local-models/GLM-4.7-Flash")
SOURCE_SHA256 = "f5c298f5554c6f5fa4a27f3329e4a08c6c2fef76834c0f10147a270ede41148d"
MODEL_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
LORA_NAME = "glm-4.7-flash-grpo"
PORTS = (8000, 8001)
CUDA_GROUPS = ("0,1,2,3", "4,5,6,7")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checked(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def verify_identity() -> dict[str, object]:
    if ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty artifact root: {ARTIFACT_ROOT}")
    if SCRATCH.exists():
        raise FileExistsError(f"refusing to reuse scratch root: {SCRATCH}")
    required = (SOURCE / "adapter_model.bin", SOURCE / "adapter_config.json")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"source adapter is incomplete: {SOURCE}")
    observed_sha = sha256_path(required[0])
    if observed_sha != SOURCE_SHA256:
        raise RuntimeError(f"source adapter SHA mismatch: {observed_sha}")
    config = json.loads(required[1].read_text(encoding="utf-8"))
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise RuntimeError(f"unexpected LoRA config: {config}")
    aider = run_checked(["git", "-C", "/aider", "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    polyglot = run_checked(
        ["git", "-C", "/aider/tmp.benchmarks/polyglot-benchmark", "rev-parse", "HEAD"],
        capture_output=True,
    ).stdout.strip()
    if aider != AIDER_COMMIT or polyglot != POLYGLOT_COMMIT:
        raise RuntimeError(f"benchmark commit drift: aider={aider} polyglot={polyglot}")
    revision_file = MODEL_PATH / "MODEL_REVISION"
    revision = revision_file.read_text(encoding="utf-8").strip()
    if revision != MODEL_REVISION:
        raise RuntimeError(f"model revision drift: {revision}")
    gpu_names = run_checked(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True
    ).stdout.splitlines()
    if len(gpu_names) != 8 or any("H100" not in name for name in gpu_names):
        raise RuntimeError(f"expected 8x H100, observed {gpu_names}")
    compute_processes = run_checked(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
    ).stdout.strip()
    if compute_processes:
        raise RuntimeError(f"GPU processes already active before eval: {compute_processes}")
    for port in PORTS:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port {port} is already occupied")
    return {
        "source_adapter_path": str(SOURCE),
        "source_adapter_sha256": observed_sha,
        "source_adapter_config_sha256": sha256_path(required[1]),
        "source_native_shards": len(list(SOURCE.glob("adapter_megatron*.pt"))),
        "source_training_state_files": len(list(SOURCE.glob("training_state_rank*.pt"))),
        "lora_rank": config["r"],
        "lora_alpha": config["lora_alpha"],
        "model_path": str(MODEL_PATH),
        "model_revision": revision,
        "aider_commit": aider,
        "polyglot_commit": polyglot,
        "gpu_names": gpu_names,
        "preexisting_gpu_processes": [],
    }


def prepare_serving_adapter() -> dict[str, object]:
    if SERVING.exists():
        manifest_path = SERVING / "mtp_strip_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"existing serving path lacks manifest: {SERVING}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_adapter_model_sha256") != SOURCE_SHA256
            or manifest.get("source_tensor_count") != 9741
            or manifest.get("stripped_tensor_count") != 207
            or manifest.get("kept_tensor_count") != 9534
            or sha256_path(SERVING / "adapter_model.bin")
            != manifest.get("output_adapter_model_sha256")
        ):
            raise RuntimeError("existing serving adapter does not match the exact step-10 source")
        status = "reused"
    else:
        run_checked(
            [
                "python3",
                "/root/sky_workdir/scripts/prepare_grpo_adapter.py",
                "--expected-source-sha256",
                SOURCE_SHA256,
                "--expected-source-tensors",
                "9741",
                "--expected-stripped-tensors",
                "207",
                str(SOURCE),
                str(SERVING),
            ]
        )
        manifest = json.loads((SERVING / "mtp_strip_manifest.json").read_text(encoding="utf-8"))
        status = "created"
    return {
        "status": status,
        "path": str(SERVING),
        "adapter_model_sha256": sha256_path(SERVING / "adapter_model.bin"),
        "adapter_config_sha256": sha256_path(SERVING / "adapter_config.json"),
        "manifest": manifest,
    }


def install_overlay_assets() -> None:
    destination = Path("/opt/fixed26")
    for name in ("aider_fixed26_contract_overlay.py", "aider_fixed26_originals.json"):
        if not (destination / name).is_file():
            raise FileNotFoundError(destination / name)
    sys.path.insert(0, str(destination))


def apply_overlay(destination: Path) -> dict[str, object]:
    import aider_fixed26_contract_overlay as overlay

    manifest = overlay.apply(destination)
    audit = overlay.audit(destination)
    task_count = len([path for path in destination.iterdir() if path.is_dir()])
    if (
        manifest.get("overlay_version") != "fixed26-contract-v2"
        or int(manifest["tasks"]) != task_count
        or int(audit["tasks"]) != task_count
        or int(audit["unexplained_deterministic_requirements"]) != 0
    ):
        raise RuntimeError("fixed26 contract overlay or prompt-test audit failed")
    manifest["prompt_test_audit"] = audit
    manifest["prompt_test_audit_sha256"] = audit["audit_sha256"]
    return manifest


def create_shard(trial: int, shard: int) -> tuple[Path, list[str], dict[str, object]]:
    source = Path("/aider/tmp.benchmarks/polyglot-benchmark/cpp/exercises/practice")
    tasks = sorted(path for path in source.iterdir() if path.is_dir())
    if len(tasks) != 26:
        raise RuntimeError(f"fixed26 source has {len(tasks)} tasks, expected 26")
    selected = tasks[shard * 13 : (shard + 1) * 13]
    root = SCRATCH / f"trial-{trial}" / f"shard-{shard}-input"
    destination = root / "cpp/exercises/practice"
    destination.mkdir(parents=True, exist_ok=False)
    for task in selected:
        shutil.copytree(task, destination / task.name)
    manifest = apply_overlay(destination)
    return root, [task.name for task in selected], manifest


def model_settings() -> str:
    return (
        "- name: openai/glm-4.7-flash-grpo\n"
        "  edit_format: whole\n"
        "  use_repo_map: false\n"
        "  use_temperature: true\n"
        "  streaming: false\n"
        "  extra_params:\n"
        "    max_tokens: 32768\n"
        "    temperature: 0.7\n"
        "    top_p: 1.0\n"
        "    extra_body:\n"
        f"      lora_path: {LORA_NAME}\n"
    )


def server_command(port: int) -> list[str]:
    return [
        "python3", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_PATH),
        "--tp-size", "4",
        "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
        "--mem-fraction-static", "0.8",
        "--max-running-requests", "16",
        "--served-model-name", LORA_NAME,
        "--api-key", "local-eval",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--enable-lora",
        "--max-lora-rank", "16",
        "--lora-backend", "triton",
        "--lora-target-modules", "q_a_proj", "kv_a_proj_with_mqa", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "--experts-shared-outer-loras",
        "--lora-use-virtual-experts",
    ]


def wait_for_server(proc: subprocess.Popen[str], port: int, timeout: int = 1800) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang port {port} exited during startup: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang port {port} did not become healthy in {timeout}s")


def load_adapter(port: int) -> str:
    payload = json.dumps({"lora_name": LORA_NAME, "lora_path": str(SERVING)}).encode()
    for endpoint in ("/load_lora_adapter", "/v1/load_lora_adapter"):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{endpoint}",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local-eval"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                if response.status == 200:
                    return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise RuntimeError(exc.read().decode("utf-8", errors="replace")) from exc
    raise RuntimeError(f"SGLang port {port} exposes no LoRA load endpoint")


def probe_completion(port: int, with_lora: bool) -> dict[str, object]:
    body: dict[str, object] = {
        "model": LORA_NAME,
        "messages": [{
            "role": "user",
            "content": "Write a C++17 function `int answer()` that returns 42. Reply with only the code.",
        }],
        "temperature": 0.0,
        "max_tokens": 48,
        "logprobs": True,
    }
    if with_lora:
        body["lora_path"] = LORA_NAME
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-eval"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    choice = payload["choices"][0]
    logprobs = [
        round(entry["logprob"], 6)
        for entry in (choice.get("logprobs") or {}).get("content") or []
    ]
    return {"content": choice["message"]["content"], "logprobs": logprobs}


def verify_lora_activation(port: int) -> dict[str, object]:
    with_lora = probe_completion(port, True)
    without_lora = probe_completion(port, False)
    if (
        with_lora["content"] == without_lora["content"]
        and with_lora["logprobs"] == without_lora["logprobs"]
    ):
        raise RuntimeError(f"LoRA activation probe did not diverge on port {port}")
    return {"status": "diverged", "with_lora": with_lora, "without_lora": without_lora}


def benchmark(trial: int, shard: int, port: int) -> dict[str, object]:
    shard_root, selected_tasks, overlay = create_shard(trial, shard)
    label = f"{RUN_ID}-trial-{trial}-shard-{shard}"
    command = [
        "/opt/aider-venv/bin/python", "/aider/benchmark/benchmark.py", label,
        "--model", "openai/glm-4.7-flash-grpo",
        "--edit-format", "whole",
        "--languages", "cpp",
        "--tries", "2",
        "--threads", "8",
        "--exercises-dir", str(shard_root),
        "--read-model-settings", str(SCRATCH / "model-settings.yml"),
    ]
    env = os.environ.copy()
    env.update({
        "AIDER_DOCKER": "1",
        "OPENAI_API_BASE": f"http://127.0.0.1:{port}/v1",
        "OPENAI_API_KEY": "local-eval",
    })
    log_path = SCRATCH / f"trial-{trial}" / f"shard-{shard}-benchmark.log"
    print(f"TRIAL_{trial}_SHARD_{shard}_START port={port}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        run_checked(command, cwd="/aider", env=env, stdout=log, stderr=subprocess.STDOUT)
    candidates = sorted(Path("/aider/tmp.benchmarks").glob(f"*--{label}"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one benchmark output for {label}, got {candidates}")
    output_dir = candidates[0]
    paths = sorted(output_dir.rglob(".aider.results.json"))
    if len(paths) != 13:
        raise RuntimeError(f"trial {trial} shard {shard}: {len(paths)} results, expected 13")
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        outcomes = payload.get("tests_outcomes")
        if (
            payload.get("model") != "openai/glm-4.7-flash-grpo"
            or payload.get("edit_format") != "whole"
            or not isinstance(payload.get("testcase"), str)
            or not isinstance(outcomes, list)
            or not 1 <= len(outcomes) <= 2
            or any(not isinstance(outcome, bool) for outcome in outcomes)
            or (len(outcomes) == 1 and not outcomes[0])
            or (len(outcomes) == 2 and outcomes[0])
        ):
            raise RuntimeError(f"malformed official two-turn result: {path}")
        rows.append({
            "task": Path(payload["testcase"]).name,
            "tests_outcomes": outcomes,
            "passed_first": outcomes[0],
            "passed_by_turn_2": any(outcomes),
            "result_path": str(path.relative_to(output_dir)),
            "result_sha256": sha256_path(path),
            "malformed_responses": int(payload.get("num_malformed_responses", 0)),
            "error_outputs": int(payload.get("num_error_outputs", 0)),
            "context_exhaustions": int(payload.get("num_exhausted_context_windows", 0)),
            "test_timeouts": int(payload.get("test_timeouts", 0)),
        })
    observed = sorted(row["task"] for row in rows)
    if observed != sorted(selected_tasks) or len(set(observed)) != 13:
        raise RuntimeError(f"trial {trial} shard {shard}: task identity mismatch")
    destination = ARTIFACT_ROOT / f"trial-{trial}" / f"shard-{shard}"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(output_dir, destination / output_dir.name)
    shutil.copy2(log_path, destination / "benchmark.log")
    receipt = {
        "trial": trial,
        "shard": shard,
        "port": port,
        "selected_tasks": selected_tasks,
        "tries": 2,
        "threads": 8,
        "temperature": 0.7,
        "top_p": 1.0,
        "thinking_disabled": False,
        "overlay": overlay,
        "command": command,
        "rows": rows,
        "pass_at_1": sum(row["passed_first"] for row in rows),
        "multi_turn_with_error_feedback_at_2": sum(
            row["passed_by_turn_2"] for row in rows
        ),
        "completed_at_utc": utc_now(),
    }
    (destination / "shard_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"TRIAL_{trial}_SHARD_{shard}_COMPLETE "
        f"pass_at_1={receipt['pass_at_1']}/13 "
        f"turn_2={receipt['multi_turn_with_error_feedback_at_2']}/13",
        flush=True,
    )
    return receipt


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


def main() -> None:
    started = utc_now()
    identity = verify_identity()
    SCRATCH.mkdir(parents=True, exist_ok=False)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    install_overlay_assets()
    (SCRATCH / "model-settings.yml").write_text(model_settings(), encoding="utf-8")
    serving = prepare_serving_adapter()
    print(
        "SERVING_ADAPTER_READY "
        f"source={identity['source_adapter_sha256']} serving={serving['adapter_model_sha256']}",
        flush=True,
    )

    servers: list[subprocess.Popen[str]] = []
    server_receipts = []
    try:
        for shard, (port, cuda) in enumerate(zip(PORTS, CUDA_GROUPS)):
            command = server_command(port)
            log_path = ARTIFACT_ROOT / f"sglang-shard-{shard}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            proc._issue110_log = log  # type: ignore[attr-defined]
            servers.append(proc)
            print(f"SGLANG_SHARD_{shard}_STARTING port={port} cuda={cuda}", flush=True)
        for shard, (proc, port) in enumerate(zip(servers, PORTS)):
            wait_for_server(proc, port)
            adapter_load = load_adapter(port)
            activation = {"status": "not-run"}
            server_receipts.append({
                "shard": shard,
                "port": port,
                "cuda_visible_devices": CUDA_GROUPS[shard],
                "command": server_command(port),
                "adapter_load": adapter_load,
                "activation_probe": activation,
            })
            print(f"SGLANG_SHARD_{shard}_READY adapter=loaded", flush=True)

        trials = []
        for trial in range(1, 5):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(benchmark, trial, shard, PORTS[shard])
                    for shard in range(2)
                ]
                shards = [future.result() for future in futures]
            rows = [row for shard in shards for row in shard["rows"]]
            if len(rows) != 26 or len({row["task"] for row in rows}) != 26:
                raise RuntimeError(f"trial {trial}: merged task set is not exactly fixed26")
            trial_receipt = {
                "trial": trial,
                "pass_at_1": sum(row["passed_first"] for row in rows),
                "multi_turn_with_error_feedback_at_2": sum(
                    row["passed_by_turn_2"] for row in rows
                ),
                "denominator": 26,
                "rows": sorted(rows, key=lambda row: row["task"]),
                "shards": shards,
                "completed_at_utc": utc_now(),
            }
            trial_dir = ARTIFACT_ROOT / f"trial-{trial}"
            (trial_dir / "run_receipt.json").write_text(
                json.dumps(trial_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            trials.append(trial_receipt)
            print(
                f"TRIAL_{trial}_COMPLETE pass_at_1={trial_receipt['pass_at_1']}/26 "
                f"turn_2={trial_receipt['multi_turn_with_error_feedback_at_2']}/26",
                flush=True,
            )

        task_first_frequency: dict[str, int] = {}
        task_turn_2_frequency: dict[str, int] = {}
        for trial in trials:
            for row in trial["rows"]:
                task_first_frequency.setdefault(row["task"], 0)
                task_turn_2_frequency.setdefault(row["task"], 0)
                task_first_frequency[row["task"]] += int(row["passed_first"])
                task_turn_2_frequency[row["task"]] += int(row["passed_by_turn_2"])
        total_first_passes = sum(trial["pass_at_1"] for trial in trials)
        total_turn_2_passes = sum(
            trial["multi_turn_with_error_feedback_at_2"] for trial in trials
        )
        union_first = sorted(task for task, count in task_first_frequency.items() if count)
        union_turn_2 = sorted(task for task, count in task_turn_2_frequency.items() if count)
        summary = {
            "kind": "fixed26-official-four-trial-two-turn-evaluation",
            "status": "complete",
            "run_id": RUN_ID,
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "model_identity": "base GLM-4.7-Flash + cumulative execution-bank-RL-v2-think-r2 iter19 adapter",
            "identity": identity,
            "serving_adapter": serving,
            "servers": server_receipts,
            "contract": {
                "trials": 4,
                "tasks_per_trial": 26,
                "task_evaluations": 104,
                "tries": 2,
                "temperature": 0.7,
                "top_p": 1.0,
                "thinking_disabled": False,
                "parallel_topology": "2x TP4 on one 8xH100 node",
                "overlay_version": "fixed26-contract-v2",
            },
            "pass_at_1": {
                "per_trial": [trial["pass_at_1"] for trial in trials],
                "total_passes": total_first_passes,
                "total_task_evaluations": 104,
                "mean_tasks_out_of_26": total_first_passes / 4,
                "rate": total_first_passes / 104,
                "task_pass_frequency_over_4": dict(sorted(task_first_frequency.items())),
                "union_tasks": union_first,
            },
            "multi_turn_with_error_feedback_at_2": {
                "per_trial": [
                    trial["multi_turn_with_error_feedback_at_2"] for trial in trials
                ],
                "total_passes": total_turn_2_passes,
                "total_task_evaluations": 104,
                "mean_tasks_out_of_26": total_turn_2_passes / 4,
                "rate": total_turn_2_passes / 104,
                "task_pass_frequency_over_4": dict(sorted(task_turn_2_frequency.items())),
                "union_tasks": union_turn_2,
            },
            "trials": trials,
        }
        summary_path = ARTIFACT_ROOT / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (ARTIFACT_ROOT / "SUMMARY_SHA256").write_text(
            sha256_path(summary_path) + "  summary.json\n", encoding="utf-8"
        )
        print(
            "FIXED26_MT2_4X_COMPLETE "
            f"pass1={summary['pass_at_1']['per_trial']} "
            f"turn2={summary['multi_turn_with_error_feedback_at_2']['per_trial']} "
            f"pass1_mean={summary['pass_at_1']['mean_tasks_out_of_26']}/26 "
            f"turn2_mean={summary['multi_turn_with_error_feedback_at_2']['mean_tasks_out_of_26']}/26",
            flush=True,
        )
    finally:
        for proc in servers:
            stop_server(proc)
            log = getattr(proc, "_issue110_log", None)
            if log is not None:
                log.close()


if __name__ == "__main__":
    main()
