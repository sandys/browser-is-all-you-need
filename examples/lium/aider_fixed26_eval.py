#!/usr/bin/env python3
"""Run the pinned fixed-26 Aider C++ evaluation directly on one 8xH100 Lium pod."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("GLM47_EVAL_ROOT", "/workspace/eval-final-iter10"))
AIDER = ROOT / "aider"
VENV_PYTHON = ROOT / "aider-venv/bin/python"
POLYGLOT = AIDER / "tmp.benchmarks/polyglot-benchmark"
MODEL_PATH = Path(os.environ.get("GLM47_EVAL_MODEL_PATH", "/workspace/models/GLM-4.7-Flash"))
RUN_ID = os.environ.get(
    "GLM47_EVAL_RUN_ID",
    f"glm47-aider-rl-final-iter10-fixed26-repro-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
)
TRAIN_RUN_ID = os.environ.get(
    "GLM47_EVAL_TRAIN_RUN_ID",
    "glm47-aider-grpo169-merge1211-530-r32-2ep-fixed-20260722-lium-r3",
)
ADAPTER = Path(
    os.environ.get(
        "GLM47_EVAL_ADAPTER_PATH",
        f"/workspace/runs/{TRAIN_RUN_ID}/checkpoints/grpo_lora_r16/iter_0000010/adapter",
    )
)
SERVING_ADAPTER = Path(f"{ADAPTER}-serving")
RESULTS = ROOT / "results" / RUN_ID
MODEL_SETTINGS = ROOT / "model-settings.yml"
IMAGE = os.environ.get("GLM47_EVAL_IMAGE", "glm47-fixed:c5cb63f")
AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
ADAPTER_SHA256 = os.environ.get(
    "GLM47_EVAL_ADAPTER_SHA256",
    "046a1018b605aa29f8b8c4f2677f47ce55489105f6766155f4c009798f48abe2",
)
DATA_MANIFEST_SHA256 = os.environ.get(
    "GLM47_EVAL_DATA_MANIFEST_SHA256",
    "a7e54c0245b97ae78f9b2fa57ff5278844585cf03004254137b6cfc8e91ef157",
)
SOURCE_TENSORS = 9_741
REMOVED_TENSORS = 207
SERVING_TENSORS = 9_534
PORTS = (18000, 18001)
CONTAINERS = tuple(
    os.environ.get(
        "GLM47_EVAL_CONTAINER_NAMES",
        "glm47-final-eval-shard0,glm47-final-eval-shard1",
    ).split(",")
)
POD_NAME = os.environ.get("GLM47_EVAL_POD_NAME", "unspecified")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def git_head(path: Path) -> str:
    return checked(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True
    ).stdout.strip()


def validate_inputs() -> None:
    expected_files = (
        VENV_PYTHON,
        MODEL_PATH / "config.json",
        ADAPTER / "adapter_model.bin",
        ADAPTER / "adapter_config.json",
    )
    missing = [str(path) for path in expected_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required inputs: {missing}")
    if git_head(AIDER) != AIDER_COMMIT or git_head(POLYGLOT) != POLYGLOT_COMMIT:
        raise RuntimeError("benchmark checkout does not match the pinned commits")
    if sha256(ADAPTER / "adapter_model.bin") != ADAPTER_SHA256:
        raise RuntimeError("final adapter SHA-256 mismatch")
    if RESULTS.exists():
        raise FileExistsError(f"refusing to reuse result path: {RESULTS}")
    for executable in ("docker", "cmake", "make", "g++"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"missing required executable: {executable}")


def prepare_serving_adapter() -> dict[str, object]:
    receipt_path = SERVING_ADAPTER / "conversion_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected = {
            "source_adapter_sha256": ADAPTER_SHA256,
            "source_tensor_count": SOURCE_TENSORS,
            "removed_layer_47_tensor_count": REMOVED_TENSORS,
            "serving_tensor_count": SERVING_TENSORS,
            "training_data_manifest_sha256": DATA_MANIFEST_SHA256,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise RuntimeError("existing serving conversion receipt is stale")
        if receipt.get("serving_adapter_sha256") != sha256(
            SERVING_ADAPTER / "adapter_model.bin"
        ):
            raise RuntimeError("existing serving adapter hash mismatch")
        return receipt
    if SERVING_ADAPTER.exists():
        raise FileExistsError(f"incomplete serving adapter exists: {SERVING_ADAPTER}")

    conversion = r'''
import hashlib, json, os, shutil, torch, uuid
from pathlib import Path
source = Path(os.environ["SOURCE_ADAPTER"])
destination = Path(os.environ["SERVING_ADAPTER"])
temporary = Path(f"{destination}-preparing-{uuid.uuid4().hex[:8]}")
temporary.mkdir(parents=True, exist_ok=False)
state = torch.load(source / "adapter_model.bin", map_location="cpu", weights_only=True, mmap=True)
removed = [key for key in state if ".layers.47." in key]
if len(state) != 9741 or len(removed) != 207:
    raise RuntimeError(f"unexpected tensor structure: {len(state)} total, {len(removed)} layer-47")
filtered = {key: value for key, value in state.items() if ".layers.47." not in key}
if len(filtered) != 9534:
    raise RuntimeError(f"unexpected serving tensor count: {len(filtered)}")
torch.save(filtered, temporary / "adapter_model.bin")
shutil.copy2(source / "adapter_config.json", temporary / "adapter_config.json")
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
receipt = {
    "schema_version": 2,
    "kind": "glm47-serving-adapter-conversion",
    "source_adapter_path": str(source),
    "source_adapter_sha256": os.environ["ADAPTER_SHA256"],
    "source_adapter_config_sha256": digest(source / "adapter_config.json"),
    "source_tensor_count": len(state),
    "removed_layer_47_tensor_count": len(removed),
    "serving_tensor_count": len(filtered),
    "serving_adapter_sha256": digest(temporary / "adapter_model.bin"),
    "serving_adapter_config_sha256": digest(temporary / "adapter_config.json"),
    "training_data_manifest_sha256": os.environ["DATA_MANIFEST_SHA256"],
    "training_gate_run_id": os.environ["TRAIN_RUN_ID"],
}
(temporary / "conversion_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
'''
    env_args = [
        "-e", f"SOURCE_ADAPTER={ADAPTER}",
        "-e", f"SERVING_ADAPTER={SERVING_ADAPTER}",
        "-e", f"ADAPTER_SHA256={ADAPTER_SHA256}",
        "-e", f"DATA_MANIFEST_SHA256={DATA_MANIFEST_SHA256}",
        "-e", f"TRAIN_RUN_ID={TRAIN_RUN_ID}",
    ]
    checked(
        [
            "docker", "run", "--rm", "-v", "/workspace:/workspace",
            *env_args, IMAGE, "python3", "-c", conversion,
        ]
    )
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def create_shards() -> list[tuple[Path, list[str]]]:
    source = POLYGLOT / "cpp/exercises/practice"
    tasks = sorted(path for path in source.iterdir() if path.is_dir())
    if len(tasks) != 26:
        raise RuntimeError(f"fixed C++ task count mismatch: {len(tasks)}")
    shard_root = ROOT / f"shards-{RUN_ID}"
    if shard_root.exists():
        raise FileExistsError(f"refusing to reuse shard path: {shard_root}")
    outputs = []
    for shard_index in range(2):
        selected = tasks[shard_index * 13 : (shard_index + 1) * 13]
        destination = shard_root / f"shard-{shard_index}/cpp/exercises/practice"
        destination.mkdir(parents=True, exist_ok=False)
        for task in selected:
            shutil.copytree(task, destination / task.name)
        outputs.append((shard_root / f"shard-{shard_index}", [p.name for p in selected]))
    return outputs


def server_command(shard_index: int) -> list[str]:
    gpu_ids = "0,1,2,3" if shard_index == 0 else "4,5,6,7"
    port = PORTS[shard_index]
    return [
        "docker", "run", "--rm", "--name", CONTAINERS[shard_index],
        "--gpus", "all", "--network", "host", "--ipc", "host",
        "-e", f"CUDA_VISIBLE_DEVICES={gpu_ids}",
        "-v", "/workspace:/workspace", IMAGE,
        "python3", "-m", "sglang.launch_server",
        "--model-path", str(MODEL_PATH),
        "--tp-size", "4",
        "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
        "--mem-fraction-static", "0.8",
        "--max-running-requests", "16",
        "--served-model-name", "glm-4.7-flash-grpo",
        "--api-key", "local-eval",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--enable-lora",
        "--max-lora-rank", "32",
        "--lora-backend", "triton",
        "--lora-target-modules",
        "q_a_proj", "kv_a_proj_with_mqa", "o_proj", "gate_proj", "up_proj", "down_proj",
        "--experts-shared-outer-loras",
        "--lora-use-virtual-experts",
    ]


def wait_for_server(process: subprocess.Popen[str], port: int, log_path: Path) -> None:
    deadline = time.time() + 1800
    while time.time() < deadline:
        if process.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise RuntimeError(f"SGLang on port {port} exited with {process.returncode}\n{tail}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError(f"SGLang on port {port} did not become healthy")


def load_adapter(port: int) -> str:
    payload = json.dumps(
        {"lora_name": "glm-4.7-flash-grpo", "lora_path": str(SERVING_ADAPTER)}
    ).encode()
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
    raise RuntimeError(f"SGLang on port {port} exposes no LoRA load endpoint")


def benchmark_command(shard_index: int, shard_root: Path) -> list[str]:
    return [
        str(VENV_PYTHON), str(AIDER / "benchmark/benchmark.py"),
        f"{RUN_ID}-shard-{shard_index}",
        "--model", "openai/glm-4.7-flash-grpo",
        "--edit-format", "whole",
        "--languages", "cpp",
        "--tries", "2",
        "--threads", "8",
        "--exercises-dir", str(shard_root),
        "--read-model-settings", str(MODEL_SETTINGS),
    ]


def validate_output(output_dir: Path, selected_tasks: list[str]) -> dict[str, object]:
    result_paths = sorted(output_dir.rglob(".aider.results.json"))
    if len(result_paths) != 13:
        raise RuntimeError(f"terminal result count mismatch: {len(result_paths)} != 13")
    rows = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        outcomes = payload.get("tests_outcomes")
        if (
            payload.get("model") != "openai/glm-4.7-flash-grpo"
            or payload.get("edit_format") != "whole"
            or not isinstance(payload.get("testcase"), str)
            or not isinstance(outcomes, list)
            or not 1 <= len(outcomes) <= 2
            or any(not isinstance(value, bool) for value in outcomes)
            or (len(outcomes) < 2 and not outcomes[-1])
            or (len(outcomes) > 1 and outcomes[0])
        ):
            raise RuntimeError(f"malformed result: {path}")
        rows.append((path, payload))
    testcases = sorted(payload["testcase"] for _, payload in rows)
    if testcases != sorted(selected_tasks):
        raise RuntimeError("terminal task identities do not match the fixed shard")
    return {
        "terminal_tasks": len(rows),
        "terminal_attempts": sum(len(p["tests_outcomes"]) for _, p in rows),
        "maximum_attempts": 26,
        "short_circuited_after_first_pass": sum(
            len(p["tests_outcomes"]) == 1 and p["tests_outcomes"][0] for _, p in rows
        ),
        "unique_testcases": len(set(testcases)),
        "testcases": testcases,
        "pass_at_1": sum(bool(p["tests_outcomes"][0]) for _, p in rows),
        "pass_at_k": sum(any(p["tests_outcomes"]) for _, p in rows),
        "well_formed_tasks": sum(int(p.get("num_malformed_responses", 0)) == 0 for _, p in rows),
        "malformed_responses": sum(int(p.get("num_malformed_responses", 0)) for _, p in rows),
        "error_outputs": sum(int(p.get("num_error_outputs", 0)) for _, p in rows),
        "context_exhaustions": sum(int(p.get("num_exhausted_context_windows", 0)) for _, p in rows),
        "test_timeouts": sum(int(p.get("test_timeouts", 0)) for _, p in rows),
        "prompt_tokens": sum(int(p.get("prompt_tokens", 0)) for _, p in rows),
        "completion_tokens": sum(int(p.get("completion_tokens", 0)) for _, p in rows),
        "result_sha256": {str(path.relative_to(output_dir)): sha256(path) for path, _ in rows},
        "outcomes": {p["testcase"]: p["tests_outcomes"] for _, p in rows},
    }


def main() -> None:
    started = utc_now()
    validate_inputs()
    conversion_receipt = prepare_serving_adapter()
    shards = create_shards()
    RESULTS.mkdir(parents=True, exist_ok=False)
    MODEL_SETTINGS.write_text(
        """- name: openai/glm-4.7-flash-grpo
  edit_format: whole
  use_repo_map: false
  use_temperature: true
  streaming: false
  extra_params:
    max_tokens: 32768
    temperature: 0.7
    top_p: 1.0
""",
        encoding="utf-8",
    )

    servers: list[subprocess.Popen[str]] = []
    server_logs = []
    benchmark_processes: list[subprocess.Popen[str]] = []
    benchmark_logs = []
    adapter_loads = []
    try:
        for shard_index in range(2):
            log_path = RESULTS / f"sglang-shard-{shard_index}.log"
            handle = log_path.open("w", encoding="utf-8")
            server_logs.append(handle)
            servers.append(
                subprocess.Popen(
                    server_command(shard_index), stdout=handle, stderr=subprocess.STDOUT, text=True
                )
            )
        for shard_index, server in enumerate(servers):
            wait_for_server(server, PORTS[shard_index], RESULTS / f"sglang-shard-{shard_index}.log")
            adapter_loads.append(load_adapter(PORTS[shard_index]))

        for shard_index, (shard_root, _) in enumerate(shards):
            log_path = RESULTS / f"benchmark-shard-{shard_index}.log"
            handle = log_path.open("w", encoding="utf-8")
            benchmark_logs.append(handle)
            env = os.environ.copy()
            env.update(
                {
                    "AIDER_DOCKER": "1",
                    "OPENAI_API_BASE": f"http://127.0.0.1:{PORTS[shard_index]}/v1",
                    "OPENAI_API_KEY": "local-eval",
                }
            )
            benchmark_processes.append(
                subprocess.Popen(
                    benchmark_command(shard_index, shard_root),
                    cwd=AIDER,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        codes = [process.wait() for process in benchmark_processes]
        if codes != [0, 0]:
            raise RuntimeError(f"benchmark shard exit codes: {codes}")
    finally:
        for handle in benchmark_logs:
            handle.close()
        for name in CONTAINERS:
            subprocess.run(["docker", "stop", "-t", "30", name], capture_output=True, text=True)
        for server in servers:
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
        for handle in server_logs:
            handle.close()

    shard_receipts = []
    for shard_index, (_, selected_tasks) in enumerate(shards):
        candidates = sorted((AIDER / "tmp.benchmarks").glob(f"*--{RUN_ID}-shard-{shard_index}"))
        if not candidates:
            raise RuntimeError(f"missing benchmark output for shard {shard_index}")
        output_dir = candidates[-1]
        validation = validate_output(output_dir, selected_tasks)
        shard_receipt = {
            "status": "complete",
            "run_id": RUN_ID,
            "shard_index": shard_index,
            "selected_tasks": selected_tasks,
            "output_dir": str(output_dir),
            "gpu_ids": "0-3" if shard_index == 0 else "4-7",
            "gpu_requested": "4x NVIDIA H100 80GB HBM3",
            "server_command": server_command(shard_index),
            "benchmark_command": benchmark_command(shard_index, shards[shard_index][0]),
            "adapter_load": adapter_loads[shard_index],
            "validation": validation,
        }
        (RESULTS / f"shard-{shard_index}-receipt.json").write_text(
            json.dumps(shard_receipt, indent=2) + "\n", encoding="utf-8"
        )
        shard_receipts.append(shard_receipt)

    fields = (
        "terminal_tasks", "terminal_attempts", "maximum_attempts",
        "short_circuited_after_first_pass", "pass_at_1", "pass_at_k",
        "well_formed_tasks", "malformed_responses", "error_outputs",
        "context_exhaustions", "test_timeouts", "prompt_tokens", "completion_tokens",
    )
    validation = {
        field: sum(int(receipt["validation"][field]) for receipt in shard_receipts)
        for field in fields
    }
    testcases = sorted(
        testcase for receipt in shard_receipts for testcase in receipt["validation"]["testcases"]
    )
    if len(testcases) != 26 or len(set(testcases)) != 26:
        raise RuntimeError("merged fixed-26 task set is incomplete or duplicated")
    validation.update(
        {
            "unique_testcases": 26,
            "testcases": testcases,
            "outcomes": {
                testcase: outcomes
                for receipt in shard_receipts
                for testcase, outcomes in receipt["validation"]["outcomes"].items()
            },
        }
    )
    receipt = {
        "status": "complete",
        "run_id": RUN_ID,
        "benchmark": "aider-polyglot-cpp-grpo-eval",
        "parallel_topology": "2x TP4 on one Lium 8xH100 pod",
        "pod": POD_NAME,
        "adapter_path": str(ADAPTER),
        "adapter_sha256": ADAPTER_SHA256,
        "adapter_config_sha256": sha256(ADAPTER / "adapter_config.json"),
        "training_data_manifest_sha256": DATA_MANIFEST_SHA256,
        "serving_conversion_receipt": conversion_receipt,
        "aider_commit": AIDER_COMMIT,
        "polyglot_commit": POLYGLOT_COMMIT,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 32768,
        "tries": 2,
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "validation": validation,
        "shards": shard_receipts,
    }
    receipt_path = RESULTS / "run_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(f"receipt_sha256={sha256(receipt_path)}")


if __name__ == "__main__":
    main()
