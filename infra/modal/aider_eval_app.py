"""Pinned fixed-26 Aider C++ evaluation for a provenance-gated GRPO adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal


MODEL_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
MODEL_PATH = f"/models/zai-org--GLM-4.7-Flash/{MODEL_REVISION}"
AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SOURCE_TENSORS = 9_741
EXPECTED_LAYER_47_TENSORS = 207
EXPECTED_SERVING_TENSORS = 9_534
EVAL_TAG = os.environ.get("GLM47_EVAL_TAG", "shadow-grpo")
EXPECTED_TRAINING_GATE_KIND = os.environ.get(
    "GLM47_TRAINING_GATE_KIND", "glm47-aider-grpo-training-gate"
)
EXPECTED_TRAINING_PHASE = os.environ.get("GLM47_EXPECTED_TRAINING_PHASE", "full")
EXPECTED_TRAINING_TASK_COUNT = int(
    os.environ.get("GLM47_EXPECTED_TRAINING_TASK_COUNT", "253")
)
EVAL_LORA_RANK = int(os.environ.get("GLM47_EVAL_LORA_RANK", "16"))

app = modal.App(f"glm47-aider-{EVAL_TAG}-grpo-eval")
image = (
    modal.Image.from_registry(
        "radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37",
    )
    .env(
        {
            "FLASHINFER_VERSION": "0.6.12",
            "FLASHINFER_CUDA_INDEX": "129",
            "GLM47_EXPECTED_TRAINING_TASK_COUNT": str(EXPECTED_TRAINING_TASK_COUNT),
            "GLM47_EVAL_LORA_RANK": str(EVAL_LORA_RANK),
        }
    )
    .run_commands(
        "python3 -m pip install --no-cache-dir --no-deps --upgrade "
        "flashinfer-python==0.6.12 flashinfer-cubin==0.6.12",
        "python3 -m pip install --no-cache-dir --no-deps --upgrade "
        "flashinfer-jit-cache==0.6.12 --index-url https://flashinfer.ai/whl/cu129/",
        "python3 -m pip install --no-cache-dir --no-deps --force-reinstall "
        "sglang-kernel==0.4.4 --index-url https://docs.sglang.ai/whl/cu129/",
        "python3 -m pip install --no-cache-dir --no-deps --upgrade "
        "torch-memory-saver==0.0.9.post1",
    )
    .apt_install("git", "cmake", "make", "g++", "curl", "python3-venv")
    .run_commands(
        f"git clone https://github.com/Aider-AI/aider.git /aider && git -C /aider checkout {AIDER_COMMIT}",
        f"git clone https://github.com/Aider-AI/polyglot-benchmark.git /aider/tmp.benchmarks/polyglot-benchmark && git -C /aider/tmp.benchmarks/polyglot-benchmark checkout {POLYGLOT_COMMIT}",
        "python3 -m venv /opt/aider-venv && /opt/aider-venv/bin/pip install -e '/aider[dev]'",
    )
)

models = modal.Volume.from_name("w8-glm47-flash-models", create_if_missing=False)
runs = modal.Volume.from_name("glm47-runs", create_if_missing=False)
results = modal.Volume.from_name("w8-aider-polyglot-cpp-results", create_if_missing=False)


def validate_adapter_path(adapter_path: str) -> Path:
    path = PurePosixPath(adapter_path)
    if not path.is_absolute() or len(path.parts) < 4 or path.parts[1] != "runs" or ".." in path.parts:
        raise ValueError("adapter_path must be an absolute checkpoint path beneath /runs")
    return Path(str(path))


def validate_run_id(run_id: str) -> str:
    value = run_id.strip() if run_id else (
        f"glm47-aider-{EVAL_TAG}-grpo-eval-"
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    )
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("run_id must contain only letters, digits, dot, underscore, and hyphen")
    return value


def validate_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be an explicit lowercase SHA-256")
    return normalized


def serving_adapter_path(adapter_path: str) -> Path:
    return Path(f"{validate_adapter_path(adapter_path)}-serving")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_adapter_files(adapter_path: Path) -> None:
    for name in ("adapter_model.bin", "adapter_config.json"):
        if not (adapter_path / name).is_file():
            raise FileNotFoundError(f"adapter is incomplete: {adapter_path / name}")


def verify_training_binding(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
) -> tuple[Path, dict[str, object]]:
    source = validate_adapter_path(adapter_path)
    validate_adapter_files(source)
    adapter_sha256 = validate_sha256(expected_adapter_sha256, "expected_adapter_sha256")
    data_manifest_sha256 = validate_sha256(
        expected_data_manifest_sha256, "expected_data_manifest_sha256"
    )
    if sha256_path(source / "adapter_model.bin") != adapter_sha256:
        raise RuntimeError("selected adapter bytes do not match the caller-bound SHA-256")
    run_id = source.parts[2]
    gate_path = Path("/runs", run_id, "grpo_lora_r16", "grpo_training_gate.json")
    if not gate_path.is_file():
        raise FileNotFoundError(f"missing full-run training gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = {
        "kind": EXPECTED_TRAINING_GATE_KIND,
        "status": "passed",
        "phase": expected_training_phase,
        "run_id": run_id,
        "data_manifest_sha256": data_manifest_sha256,
        "training_task_count": EXPECTED_TRAINING_TASK_COUNT,
        "official_26_role": "external fixed evaluation only",
        "gpus_per_node": 8,
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise RuntimeError("adapter is not bound to the exact caller-selected GRPO gate")
    if not gate.get("source_commit") or gate.get("source_commit") == "unbound":
        raise RuntimeError("GRPO gate does not bind the training source commit")
    checkpoints = gate.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise RuntimeError("GRPO gate has no checkpoint catalog")
    checkpoint = next(
        (
            item
            for item in checkpoints
            if isinstance(item, dict)
            and item.get("adapter_model_sha256") == adapter_sha256
            and item.get("adapter_config_sha256")
            == sha256_path(source / "adapter_config.json")
        ),
        None,
    )
    if checkpoint is None:
        raise RuntimeError("selected adapter is absent from the GRPO checkpoint catalog")
    if (
        checkpoint.get("tensor_count") != EXPECTED_SOURCE_TENSORS
        or checkpoint.get("layer_47_tensor_count") != EXPECTED_LAYER_47_TENSORS
    ):
        raise RuntimeError("selected GRPO adapter has an unexpected tensor domain")
    return source, gate


@app.function(image=image, cpu=1.0, memory=2_048, volumes={"/runs": runs})
def inspect_adapter(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
) -> dict[str, object]:
    import torch

    source, training_gate = verify_training_binding(
        adapter_path,
        expected_adapter_sha256,
        expected_data_manifest_sha256,
        expected_training_phase,
    )
    state = torch.load(source / "adapter_model.bin", map_location="cpu", weights_only=True, mmap=True)
    keys = list(state)
    layer_47_keys = [key for key in keys if ".layers.47." in key]
    if len(keys) != EXPECTED_SOURCE_TENSORS or len(layer_47_keys) != EXPECTED_LAYER_47_TENSORS:
        raise RuntimeError("adapter tensor structure does not match the proven GLM-4.7 serving conversion")
    return {
        "adapter_path": str(source),
        "adapter_model_sha256": sha256_path(source / "adapter_model.bin"),
        "adapter_config_sha256": sha256_path(source / "adapter_config.json"),
        "tensor_count": len(keys),
        "layer_47_tensor_count": len(layer_47_keys),
        "layer_47_keys": layer_47_keys,
        "sample_keys": keys[:20],
        "training_gate": training_gate,
    }


@app.function(image=image, cpu=1.0, memory=2_048, volumes={"/runs": runs}, timeout=1800)
def prepare_adapter(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
) -> dict[str, object]:
    import torch

    source, training_gate = verify_training_binding(
        adapter_path,
        expected_adapter_sha256,
        expected_data_manifest_sha256,
        expected_training_phase,
    )
    destination = serving_adapter_path(adapter_path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing serving adapter: {destination}")
    temporary = Path(f"{destination}-preparing-{uuid.uuid4().hex[:8]}")
    temporary.mkdir(parents=True, exist_ok=False)
    state = torch.load(source / "adapter_model.bin", map_location="cpu", weights_only=True, mmap=True)
    layer_47_keys = [key for key in state if ".layers.47." in key]
    if len(state) != EXPECTED_SOURCE_TENSORS or len(layer_47_keys) != EXPECTED_LAYER_47_TENSORS:
        raise RuntimeError("adapter tensor structure does not match the proven GLM-4.7 serving conversion")
    filtered = {key: value for key, value in state.items() if ".layers.47." not in key}
    if len(filtered) != EXPECTED_SERVING_TENSORS or any(".layers.47." in key for key in filtered):
        raise RuntimeError("serving adapter conversion did not produce the exact proven tensor domain")
    torch.save(filtered, temporary / "adapter_model.bin")
    shutil.copy2(source / "adapter_config.json", temporary / "adapter_config.json")
    conversion_receipt = {
        "schema_version": 2,
        "kind": "glm47-serving-adapter-conversion",
        "source_adapter_path": str(source),
        "source_adapter_sha256": sha256_path(source / "adapter_model.bin"),
        "source_adapter_config_sha256": sha256_path(source / "adapter_config.json"),
        "source_tensor_count": len(state),
        "removed_layer_47_tensor_count": len(layer_47_keys),
        "serving_tensor_count": len(filtered),
        "serving_adapter_sha256": sha256_path(temporary / "adapter_model.bin"),
        "serving_adapter_config_sha256": sha256_path(temporary / "adapter_config.json"),
        "training_data_manifest_sha256": training_gate["data_manifest_sha256"],
        "training_gate_run_id": training_gate["run_id"],
    }
    (temporary / "conversion_receipt.json").write_text(
        json.dumps(conversion_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    runs.commit()
    return {
        "adapter_path": str(source),
        "source_adapter_sha256": sha256_path(source / "adapter_model.bin"),
        "source_tensor_count": len(state),
        "serving_tensor_count": len(filtered),
        "removed_tensor_count": len(state) - len(filtered),
        "serving_adapter_path": str(destination),
        "serving_adapter_sha256": sha256_path(destination / "adapter_model.bin"),
        "serving_adapter_config_sha256": sha256_path(destination / "adapter_config.json"),
        "conversion_receipt_sha256": sha256_path(destination / "conversion_receipt.json"),
        "training_gate": training_gate,
    }


def verify_serving_binding(
    serving_path: Path,
    source: Path,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
) -> dict[str, object]:
    receipt_path = serving_path / "conversion_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing serving conversion receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "kind": "glm47-serving-adapter-conversion",
        "source_adapter_path": str(source),
        "source_adapter_sha256": validate_sha256(expected_adapter_sha256, "expected_adapter_sha256"),
        "source_adapter_config_sha256": sha256_path(source / "adapter_config.json"),
        "source_tensor_count": EXPECTED_SOURCE_TENSORS,
        "removed_layer_47_tensor_count": EXPECTED_LAYER_47_TENSORS,
        "serving_tensor_count": EXPECTED_SERVING_TENSORS,
        "serving_adapter_sha256": sha256_path(serving_path / "adapter_model.bin"),
        "serving_adapter_config_sha256": sha256_path(serving_path / "adapter_config.json"),
        "training_data_manifest_sha256": validate_sha256(
            expected_data_manifest_sha256, "expected_data_manifest_sha256"
        ),
        "training_gate_run_id": source.parts[2],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("serving adapter is stale or not bound to the selected full-run adapter")
    return receipt


@app.local_entrypoint()
def main(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
    run_id: str = "",
    inspect: bool = False,
    prepare: bool = False,
    parallel: bool = False,
) -> None:
    if sum((inspect, prepare, parallel)) > 1:
        raise ValueError("choose at most one of --inspect, --prepare, and --parallel")
    if inspect:
        payload = inspect_adapter.remote(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
        )
    elif prepare:
        payload = prepare_adapter.remote(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
        )
    elif parallel:
        resolved_run_id = validate_run_id(run_id)
        calls = [
            evaluate_shard.spawn(
                adapter_path=adapter_path,
                expected_adapter_sha256=expected_adapter_sha256,
                expected_data_manifest_sha256=expected_data_manifest_sha256,
                expected_training_phase=expected_training_phase,
                run_id=resolved_run_id,
                shard_index=index,
            )
            for index in range(2)
        ]
        shard_receipts = [call.get() for call in calls]
        payload = merge_shards.remote(
            run_id=resolved_run_id,
            shard_receipts=shard_receipts,
        )
    else:
        payload = evaluate.remote(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
            run_id=validate_run_id(run_id),
        )
    print(json.dumps(payload, indent=2))


def _wait_for_server(proc: subprocess.Popen[str], timeout: int = 1800, port: int = 8000) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang exited during startup with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError("SGLang did not become healthy within 30 minutes")


def _load_adapter(serving_path: Path, port: int = 8000) -> str:
    payload = json.dumps({"lora_name": "glm-4.7-flash-grpo", "lora_path": str(serving_path)}).encode()
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
    raise RuntimeError("SGLang does not expose a LoRA adapter loading endpoint")


def _benchmark(
    label: str,
    num_tests: int | None,
    tries: int,
    threads: int,
    *,
    exercises_dir: str = "polyglot-benchmark",
    api_base: str = "http://127.0.0.1:8000/v1",
) -> dict[str, object]:
    command = [
        "/opt/aider-venv/bin/python",
        "/aider/benchmark/benchmark.py",
        label,
        "--model",
        "openai/glm-4.7-flash-grpo",
        "--edit-format",
        "whole",
        "--languages",
        "cpp",
        "--tries",
        str(tries),
        "--threads",
        str(threads),
        "--exercises-dir",
        exercises_dir,
        "--read-model-settings",
        "/run/model-settings.yml",
    ]
    if num_tests is not None:
        command.extend(["--num-tests", str(num_tests)])
    env = os.environ.copy()
    env.update(
        {
            "AIDER_DOCKER": "1",
            "OPENAI_API_BASE": api_base,
            "OPENAI_API_KEY": "local-eval",
        }
    )
    subprocess.run(command, cwd="/aider", check=True, env=env)
    candidates = sorted(Path("/aider/tmp.benchmarks").glob(f"*--{label}"))
    if not candidates:
        raise RuntimeError(f"No benchmark output found for {label}")
    output_dir = candidates[-1]
    stats = subprocess.run(
        ["/opt/aider-venv/bin/python", "/aider/benchmark/benchmark.py", "--stats", str(output_dir)],
        cwd="/aider",
        check=True,
        capture_output=True,
        text=True,
    )
    return {"command": command, "output_dir": str(output_dir), "stats_stdout": stats.stdout}


def _create_cpp_shard(shard_index: int) -> tuple[Path, list[str]]:
    if shard_index not in (0, 1):
        raise ValueError("shard_index must be 0 or 1")
    source = Path("/aider/tmp.benchmarks/polyglot-benchmark/cpp/exercises/practice")
    tasks = sorted(path for path in source.iterdir() if path.is_dir())
    if len(tasks) != 26:
        raise RuntimeError(f"fixed C++ benchmark task count mismatch: {len(tasks)} != 26")
    selected = tasks[shard_index * 13 : (shard_index + 1) * 13]
    shard_root = Path(f"/tmp/polyglot-benchmark-shard-{shard_index}")
    destination = shard_root / "cpp/exercises/practice"
    destination.mkdir(parents=True, exist_ok=False)
    for task in selected:
        shutil.copytree(task, destination / task.name)
    return shard_root, [task.name for task in selected]


def _validate_benchmark_results(
    phase: dict[str, object],
    *,
    expected_tasks: int,
    expected_tries: int,
) -> dict[str, object]:
    output_dir = Path(str(phase["output_dir"]))
    result_paths = sorted(output_dir.rglob(".aider.results.json"))
    if len(result_paths) != expected_tasks:
        raise RuntimeError(
            f"benchmark terminal-result count mismatch: {len(result_paths)} != {expected_tasks}"
        )
    rows = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        outcomes = payload.get("tests_outcomes")
        if (
            payload.get("model") != "openai/glm-4.7-flash-grpo"
            or payload.get("edit_format") != "whole"
            or not isinstance(payload.get("testcase"), str)
            or not payload["testcase"]
            or not isinstance(outcomes, list)
            or not 1 <= len(outcomes) <= expected_tries
            or any(not isinstance(value, bool) for value in outcomes)
            or (len(outcomes) < expected_tries and not outcomes[-1])
            or (len(outcomes) > 1 and outcomes[0])
        ):
            raise RuntimeError(f"malformed or incomplete benchmark result: {path}")
        rows.append((path, payload))
    testcases = [payload["testcase"] for _, payload in rows]
    if len(testcases) != len(set(testcases)):
        raise RuntimeError("benchmark contains duplicate task identities")
    return {
        "terminal_tasks": len(rows),
        "terminal_attempts": sum(len(payload["tests_outcomes"]) for _, payload in rows),
        "maximum_attempts": expected_tasks * expected_tries,
        "short_circuited_after_first_pass": sum(
            len(payload["tests_outcomes"]) == 1 and payload["tests_outcomes"][0]
            for _, payload in rows
        ),
        "unique_testcases": len(set(testcases)),
        "testcases": sorted(testcases),
        "pass_at_1": sum(bool(payload["tests_outcomes"][0]) for _, payload in rows),
        "pass_at_k": sum(any(payload["tests_outcomes"]) for _, payload in rows),
        "well_formed_tasks": sum(int(payload.get("num_malformed_responses", 0)) == 0 for _, payload in rows),
        "malformed_responses": sum(int(payload.get("num_malformed_responses", 0)) for _, payload in rows),
        "error_outputs": sum(int(payload.get("num_error_outputs", 0)) for _, payload in rows),
        "context_exhaustions": sum(int(payload.get("num_exhausted_context_windows", 0)) for _, payload in rows),
        "test_timeouts": sum(int(payload.get("test_timeouts", 0)) for _, payload in rows),
        "prompt_tokens": sum(int(payload.get("prompt_tokens", 0)) for _, payload in rows),
        "completion_tokens": sum(int(payload.get("completion_tokens", 0)) for _, payload in rows),
        "result_sha256": {
            str(path.relative_to(output_dir)): sha256_path(path) for path, _ in rows
        },
    }


@app.function(
    image=image,
    gpu="H100:4",
    timeout=7200,
    volumes={"/models": models, "/runs": runs, "/results": results},
)
def evaluate_shard(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str,
    run_id: str,
    shard_index: int,
) -> dict[str, object]:
    source, training_gate = verify_training_binding(
        adapter_path,
        expected_adapter_sha256,
        expected_data_manifest_sha256,
        expected_training_phase,
    )
    serving_path = serving_adapter_path(adapter_path)
    validate_adapter_files(serving_path)
    conversion_receipt = verify_serving_binding(
        serving_path, source, expected_adapter_sha256, expected_data_manifest_sha256
    )
    resolved_run_id = validate_run_id(run_id)
    destination = Path("/results/runs") / f"{resolved_run_id}-shard-{shard_index}"
    if destination.exists():
        raise FileExistsError(f"refusing to reuse shard evaluation path: {destination}")

    Path("/run/model-settings.yml").write_text(
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
    shard_root, selected_tasks = _create_cpp_shard(shard_index)
    server_command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--tp-size",
        "4",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "16",
        "--served-model-name",
        "glm-4.7-flash-grpo",
        "--api-key",
        "local-eval",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--enable-lora",
        "--max-lora-rank",
        str(EVAL_LORA_RANK),
        "--lora-backend",
        "triton",
        "--lora-target-modules",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "--experts-shared-outer-loras",
        "--lora-use-virtual-experts",
    ]
    started = datetime.now(timezone.utc)
    log_path = Path(f"/tmp/sglang-grpo-shard-{shard_index}.log")
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            try:
                _wait_for_server(server)
            except Exception as exc:
                log.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                raise RuntimeError(f"{exc}\nSGLang log tail:\n{tail}") from exc
            adapter_load = _load_adapter(serving_path)
            full = _benchmark(
                f"{resolved_run_id}-shard-{shard_index}",
                num_tests=None,
                tries=2,
                threads=8,
                exercises_dir=str(shard_root),
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()

    validation = _validate_benchmark_results(full, expected_tasks=13, expected_tries=2)
    observed_names = sorted(Path(testcase).name for testcase in validation["testcases"])
    if observed_names != sorted(selected_tasks):
        raise RuntimeError("shard benchmark task identities do not match the fixed partition")
    destination.mkdir(parents=True, exist_ok=False)
    benchmark_source = Path(str(full["output_dir"]))
    shutil.copytree(benchmark_source, destination / benchmark_source.name, dirs_exist_ok=False)
    shutil.copy2(log_path, destination / "sglang.log")
    receipt = {
        "status": "complete",
        "run_id": resolved_run_id,
        "shard_index": shard_index,
        "artifact_path": str(destination),
        "selected_tasks": selected_tasks,
        "model_revision": MODEL_REVISION,
        "adapter_path": str(source),
        "adapter_sha256": sha256_path(source / "adapter_model.bin"),
        "training_data_manifest_sha256": expected_data_manifest_sha256,
        "training_gate": training_gate,
        "serving_conversion_receipt": conversion_receipt,
        "aider_commit": AIDER_COMMIT,
        "polyglot_commit": POLYGLOT_COMMIT,
        "gpu_requested": "H100:4",
        "lora_rank": EVAL_LORA_RANK,
        "temperature": 0.7,
        "top_p": 1.0,
        "tries": 2,
        "threads": 8,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_command": server_command,
        "adapter_load": adapter_load,
        "full": full,
        "validation": validation,
    }
    (destination / "shard_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    results.commit()
    return receipt


def _merge_shard_receipts(
    run_id: str, shard_receipts: list[dict[str, object]]
) -> dict[str, object]:
    resolved_run_id = validate_run_id(run_id)
    if sorted(receipt.get("shard_index") for receipt in shard_receipts) != [0, 1]:
        raise RuntimeError("parallel evaluation did not return both shards")
    if any(receipt.get("status") != "complete" for receipt in shard_receipts):
        raise RuntimeError("parallel evaluation contains a non-terminal shard")
    testcases = [
        testcase
        for receipt in shard_receipts
        for testcase in receipt["validation"]["testcases"]
    ]
    if len(testcases) != 26 or len(set(testcases)) != 26:
        raise RuntimeError("merged fixed-26 evaluation is incomplete or duplicated")
    destination = Path("/results/runs") / resolved_run_id
    if destination.exists():
        raise FileExistsError(f"refusing to reuse merged evaluation path: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    for receipt in shard_receipts:
        shard_index = int(receipt["shard_index"])
        shutil.copytree(
            Path(str(receipt["artifact_path"])),
            destination / f"shard-{shard_index}",
            dirs_exist_ok=False,
        )
    fields = (
        "terminal_tasks",
        "terminal_attempts",
        "maximum_attempts",
        "short_circuited_after_first_pass",
        "pass_at_1",
        "pass_at_k",
        "well_formed_tasks",
        "malformed_responses",
        "error_outputs",
        "context_exhaustions",
        "test_timeouts",
        "prompt_tokens",
        "completion_tokens",
    )
    validation = {
        field: sum(int(receipt["validation"][field]) for receipt in shard_receipts)
        for field in fields
    }
    validation.update({"unique_testcases": 26, "testcases": sorted(testcases)})
    receipt = {
        "status": "complete",
        "run_id": resolved_run_id,
        "benchmark": "aider-polyglot-cpp-grpo-eval",
        "parallel_topology": "2x TP4 on 8x H100 total",
        "adapter_path": shard_receipts[0]["adapter_path"],
        "adapter_sha256": shard_receipts[0]["adapter_sha256"],
        "training_data_manifest_sha256": shard_receipts[0][
            "training_data_manifest_sha256"
        ],
        "aider_commit": AIDER_COMMIT,
        "polyglot_commit": POLYGLOT_COMMIT,
        "temperature": 0.7,
        "top_p": 1.0,
        "tries": 2,
        "validation": validation,
        "shards": shard_receipts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    results.commit()
    return receipt


@app.function(image=image, cpu=1.0, memory=2_048, volumes={"/results": results})
def merge_shards(run_id: str, shard_receipts: list[dict[str, object]]) -> dict[str, object]:
    return _merge_shard_receipts(run_id, shard_receipts)


@app.function(image=image, cpu=1.0, memory=2_048, volumes={"/results": results})
def merge_saved_shards(run_id: str) -> dict[str, object]:
    resolved_run_id = validate_run_id(run_id)
    shard_receipts = []
    for shard_index in range(2):
        receipt_path = (
            Path("/results/runs")
            / f"{resolved_run_id}-shard-{shard_index}"
            / "shard_receipt.json"
        )
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing completed shard receipt: {receipt_path}")
        shard_receipts.append(json.loads(receipt_path.read_text(encoding="utf-8")))
    return _merge_shard_receipts(resolved_run_id, shard_receipts)


@app.function(
    image=image,
    gpu="H100:4",
    timeout=7200,
    volumes={"/models": models, "/runs": runs, "/results": results},
)
def evaluate(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
    run_id: str = "",
) -> dict[str, object]:
    source, training_gate = verify_training_binding(
        adapter_path,
        expected_adapter_sha256,
        expected_data_manifest_sha256,
        expected_training_phase,
    )
    serving_path = serving_adapter_path(adapter_path)
    validate_adapter_files(serving_path)
    conversion_receipt = verify_serving_binding(
        serving_path, source, expected_adapter_sha256, expected_data_manifest_sha256
    )
    resolved_run_id = validate_run_id(run_id)
    destination = Path("/results/runs") / resolved_run_id
    if destination.exists():
        raise FileExistsError(f"refusing to reuse existing evaluation run ID: {resolved_run_id}")

    Path("/run/model-settings.yml").write_text(
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
    env = os.environ.copy()
    env.update(
        {
            "AIDER_DOCKER": "1",
            "OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "local-eval",
        }
    )
    os.environ.update(env)
    server_command = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--tp-size",
        "4",
        "--tool-call-parser",
        "glm47",
        "--reasoning-parser",
        "glm45",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "16",
        "--served-model-name",
        "glm-4.7-flash-grpo",
        "--api-key",
        "local-eval",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--enable-lora",
        "--max-lora-rank",
        str(EVAL_LORA_RANK),
        "--lora-backend",
        "triton",
        "--lora-target-modules",
        "q_a_proj",
        "kv_a_proj_with_mqa",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "--experts-shared-outer-loras",
        "--lora-use-virtual-experts",
    ]
    started = datetime.now(timezone.utc)
    log_path = Path("/tmp/sglang-grpo.log")
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            try:
                _wait_for_server(server)
            except Exception as exc:
                log.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                raise RuntimeError(f"{exc}\nSGLang log tail:\n{tail}") from exc
            adapter_load = _load_adapter(serving_path)
            smoke = _benchmark(f"{resolved_run_id}-smoke", num_tests=2, tries=1, threads=1)
            full = _benchmark(f"{resolved_run_id}-full", num_tests=None, tries=2, threads=8)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()

    smoke_validation = _validate_benchmark_results(smoke, expected_tasks=2, expected_tries=1)
    full_validation = _validate_benchmark_results(full, expected_tasks=26, expected_tries=2)
    destination.mkdir(parents=True, exist_ok=False)
    for phase in (smoke, full):
        benchmark_source = Path(str(phase["output_dir"]))
        shutil.copytree(benchmark_source, destination / benchmark_source.name, dirs_exist_ok=False)
    shutil.copy2(log_path, destination / "sglang.log")
    receipt = {
        "status": "complete",
        "run_id": resolved_run_id,
        "benchmark": "aider-polyglot-cpp-grpo-eval",
        "model_revision": MODEL_REVISION,
        "adapter_path": str(source),
        "adapter_sha256": sha256_path(source / "adapter_model.bin"),
        "training_data_manifest_sha256": validate_sha256(
            expected_data_manifest_sha256, "expected_data_manifest_sha256"
        ),
        "training_gate": training_gate,
        "serving_adapter_path": str(serving_path),
        "serving_adapter_sha256": sha256_path(serving_path / "adapter_model.bin"),
        "serving_conversion_receipt": conversion_receipt,
        "aider_commit": AIDER_COMMIT,
        "polyglot_commit": POLYGLOT_COMMIT,
        "gpu_requested": "H100:4",
        "lora_rank": EVAL_LORA_RANK,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 32768,
        "tries": 2,
        "threads": 8,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_command": server_command,
        "adapter_load": adapter_load,
        "smoke": smoke,
        "smoke_validation": smoke_validation,
        "full": full,
        "full_validation": full_validation,
    }
    (destination / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    results.commit()
    return receipt
