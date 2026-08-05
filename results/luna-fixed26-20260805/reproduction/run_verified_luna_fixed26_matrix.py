#!/usr/bin/env python3
"""Run a fail-closed GPT-5.6 Luna Low fixed26 matrix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
ARTIFACT = Path(
    os.environ.get(
        "LUNA_EVAL_ARTIFACT_ROOT",
        WORKSPACE / "artifacts" / "gpt56-luna-low-fixed26-v2contract-20260803",
    )
).resolve()
SOURCE_ARTIFACT = Path(
    os.environ.get(
        "LUNA_EVAL_SOURCE_ARTIFACT",
        WORKSPACE / "artifacts" / "luna-low-fixed26-v2contract-20260803",
    )
).resolve()
AIDER_ROOT = Path(os.environ.get("LUNA_EVAL_AIDER_ROOT", "/private/tmp/aider-fixed26-luna-src"))
PYTHON = Path("/private/tmp/aider-luna-venv/bin/python")
BENCHMARK = AIDER_ROOT / "benchmark" / "benchmark.py"
MODEL_SETTINGS = ARTIFACT / "provenance" / "model-settings.yml"
EXERCISES_ROOT = Path(
    os.environ.get(
        "LUNA_EVAL_EXERCISES_ROOT",
        SOURCE_ARTIFACT / "provenance" / "polyglot-benchmark",
    )
).resolve()
OVERLAY_AUDIT = Path(
    os.environ.get(
        "LUNA_EVAL_OVERLAY_AUDIT",
        SOURCE_ARTIFACT / "provenance" / "luna-fixed26-audit.json",
    )
).resolve()
TEST_COMMAND = Path(
    os.environ.get(
        "LUNA_EVAL_TEST_COMMAND",
        WORKSPACE / "scripts" / "luna_fixed26_modal_test_command.py",
    )
).resolve()
PROXY_RECEIPT = ARTIFACT / "transport" / "proxy_receipt.json"
PROXY_HEALTH = "http://127.0.0.1:8765/health"
EXPECTED_MODEL = "gpt-5.6-luna"
EXPECTED_EFFORT = os.environ.get("LUNA_EXPECTED_REASONING_EFFORT", "low")
EXPECTED_AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
EXPECTED_TREE_SHA256 = os.environ.get(
    "LUNA_EXPECTED_TREE_SHA256",
    "c0541864071b5df862e735aa6063d121c8154df3dbd652ef3b9d2ce101ba515e",
)
EXPECTED_OVERLAY_VERSION = os.environ.get(
    "LUNA_EXPECTED_OVERLAY_VERSION", "fixed26-contract-v2"
)
EXPECTED_SCORER_APP = os.environ.get(
    "LUNA_MODAL_SCORER_APP_NAME", "luna-low-fixed26-cpu-scorer-v3-boost"
)
MODEL_ALIAS = "openai/gpt-5.6-luna"
LABEL_PREFIX = os.environ.get("LUNA_EVAL_LABEL_PREFIX", "gpt56-luna-low-fixed26-v2c")
EXPECTED_BENCHMARK_SHA256 = os.environ.get("LUNA_EVAL_EXPECTED_BENCHMARK_SHA256")
SINGLE_ONLY = os.environ.get("LUNA_EVAL_SINGLE_ONLY") == "1"
FEEDBACK_ONLY = os.environ.get("LUNA_EVAL_FEEDBACK_ONLY") == "1"
SINGLE_TRIALS = int(os.environ.get("LUNA_EVAL_SINGLE_TRIALS", "4"))
if SINGLE_TRIALS < 1:
    raise ValueError("LUNA_EVAL_SINGLE_TRIALS must be positive")
if SINGLE_ONLY and FEEDBACK_ONLY:
    raise ValueError("single-only and feedback-only modes are mutually exclusive")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> tuple[int, str]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        content = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    return len(records), hashlib.sha256(_canonical_json_bytes(records)).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _health() -> dict[str, Any]:
    with urllib.request.urlopen(PROXY_HEALTH, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    expected = {
        "status": "ok",
        "model": EXPECTED_MODEL,
        "reasoning_effort": EXPECTED_EFFORT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"proxy identity gate failed: {payload!r}")
    return payload


def _preflight() -> dict[str, Any]:
    if not BENCHMARK.is_file() or not PYTHON.is_file() or not TEST_COMMAND.is_file():
        raise FileNotFoundError("pinned Aider checkout or Python environment is missing")
    aider_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=AIDER_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if aider_commit != EXPECTED_AIDER_COMMIT:
        raise RuntimeError(f"Aider commit drift: {aider_commit}")
    benchmark_sha256 = _sha256_file(BENCHMARK)
    if EXPECTED_BENCHMARK_SHA256 and benchmark_sha256 != EXPECTED_BENCHMARK_SHA256:
        raise RuntimeError(f"benchmark harness drift: {benchmark_sha256}")
    practice_root = EXERCISES_ROOT / "cpp" / "exercises" / "practice"
    task_count = len([path for path in practice_root.iterdir() if path.is_dir()])
    file_count, tree_sha256 = _tree_sha256(practice_root)
    if task_count != 26 or tree_sha256 != EXPECTED_TREE_SHA256:
        raise RuntimeError(
            f"source tree gate failed: tasks={task_count} files={file_count} sha={tree_sha256}"
        )
    overlay = json.loads(OVERLAY_AUDIT.read_text(encoding="utf-8"))
    if overlay.get("overlay_version") != EXPECTED_OVERLAY_VERSION or overlay.get("tasks") != 26:
        raise RuntimeError("overlay audit gate failed")
    if EXPECTED_OVERLAY_VERSION == "fixed26-contract-v2":
        if overlay.get("unexplained_deterministic_requirements") != 0:
            raise RuntimeError("v2 overlay audit gate failed")
    elif EXPECTED_OVERLAY_VERSION == "v1-contract":
        if (
            overlay.get("verdict") != "pass"
            or overlay.get("historical_prompt_matches") != 26
            or overlay.get("candidate_test_file_exposures") != 0
            or overlay.get("candidate_test_content_exposures") != 0
        ):
            raise RuntimeError("historical v1-contract freeze gate failed")
    else:
        if (
            overlay.get("verdict") != "pass"
            or overlay.get("candidate_test_content_exposures") != 0
        ):
            raise RuntimeError("ablation overlay exposure gate failed")
        if (
            EXPECTED_OVERLAY_VERSION == "fixed26-interface-only-ablation-v1"
            and overlay.get("candidate_behavioral_addendum_exposures") != 0
        ):
            raise RuntimeError("interface-only behavioral exposure gate failed")
        if EXPECTED_OVERLAY_VERSION == "fixed26-v1-supported-hybrid-v1" and (
            overlay.get("help_enabled_tasks") != 15
            or overlay.get("no_help_tasks") != 11
        ):
            raise RuntimeError("v1-supported hybrid task-policy gate failed")
        if EXPECTED_OVERLAY_VERSION == "fixed26-v1-top10-hybrid-v1" and (
            overlay.get("help_enabled_tasks") != 10
            or overlay.get("no_help_tasks") != 16
            or overlay.get("tie_break") != "task-id-ascending"
            or overlay.get("cutoff_tie_tasks")
            != ["allergies", "bank-account", "perfect-numbers"]
        ):
            raise RuntimeError("v1-top10 hybrid task-policy gate failed")
        if EXPECTED_OVERLAY_VERSION == "pristine-original-v1" and (
            overlay.get("official_prompts_match_pinned_git") != 26
            or overlay.get("prompts_differing_from_assisted_v1_contract") != 26
            or overlay.get("overlay_marker_hits") != 0
            or overlay.get("semantic_oracle_support_present") is not False
            or overlay.get("candidate_test_file_exposures") != 0
        ):
            raise RuntimeError("pristine original prompt-purity gate failed")
    proxy = _health()
    proxy_receipt = json.loads(PROXY_RECEIPT.read_text(encoding="utf-8"))
    identity_gate = proxy_receipt.get("identity_gate", {})
    if (
        proxy_receipt.get("model") != EXPECTED_MODEL
        or proxy_receipt.get("reasoning_effort") != EXPECTED_EFFORT
        or identity_gate.get("passed") is not True
    ):
        raise RuntimeError("proxy receipt identity gate failed")
    model_settings = MODEL_SETTINGS.read_text(encoding="utf-8")
    if f"name: {MODEL_ALIAS}\n" not in model_settings:
        raise RuntimeError("Aider model alias drifted")
    receipt = {
        "schema_version": 1,
        "kind": "gpt56-luna-low-fixed26-preflight",
        "status": "passed",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": EXPECTED_MODEL,
        "reasoning_effort": EXPECTED_EFFORT,
        "contract": EXPECTED_OVERLAY_VERSION,
        "scorer_app": EXPECTED_SCORER_APP,
        "test_command": str(TEST_COMMAND),
        "test_command_sha256": _sha256_file(TEST_COMMAND),
        "aider_commit": aider_commit,
        "benchmark_sha256": benchmark_sha256,
        "task_count": task_count,
        "source_tree_file_count": file_count,
        "source_tree_sha256": tree_sha256,
        "overlay_audit_sha256": _sha256_file(OVERLAY_AUDIT),
        "model_settings_sha256": _sha256_file(MODEL_SETTINGS),
        "proxy_receipt_sha256": _sha256_file(PROXY_RECEIPT),
        "proxy_health": proxy,
    }
    _write_json(ARTIFACT / "preflight_receipt.json", receipt)
    return receipt


def _command(label: str, tries: int) -> list[str]:
    return [
        str(PYTHON),
        str(BENCHMARK),
        label,
        "--model",
        MODEL_ALIAS,
        "--edit-format",
        "whole",
        "--languages",
        "cpp",
        "--tries",
        str(tries),
        "--threads",
        "8",
        "--num-tests",
        "26",
        "--exercises-dir",
        str(EXERCISES_ROOT),
        "--read-model-settings",
        str(MODEL_SETTINGS),
        "--reasoning-effort",
        EXPECTED_EFFORT,
    ]


def main() -> int:
    preflight = _preflight()
    (AIDER_ROOT / "tmp.benchmarks").mkdir(parents=True, exist_ok=True)
    benchmark_destination = ARTIFACT / "benchmark"
    logs = ARTIFACT / "logs"
    modal_receipts = ARTIFACT / "modal-receipts"
    runner_receipts = ARTIFACT / "attempt-receipts"
    for directory in (benchmark_destination, logs, modal_receipts, runner_receipts):
        directory.mkdir(parents=True, exist_ok=True)

    attempts = [] if FEEDBACK_ONLY else [
        (f"{LABEL_PREFIX}-single-a{index}", 1)
        for index in range(1, SINGLE_TRIALS + 1)
    ]
    if not SINGLE_ONLY:
        attempts += [
            (f"{LABEL_PREFIX}-feedback-a{index}", 2)
            for index in range(1, 5)
        ]
    duplicate_outputs = [
        path
        for label, _ in attempts
        for path in (AIDER_ROOT / "tmp.benchmarks").glob(f"*--{label}")
    ]
    occupied_destinations = [
        benchmark_destination / label
        for label, _ in attempts
        if (benchmark_destination / label).exists()
    ]
    if duplicate_outputs or occupied_destinations:
        raise FileExistsError(
            "refusing to reuse an evaluation identity: "
            f"outputs={duplicate_outputs!r} destinations={occupied_destinations!r}"
        )

    environment = {
        **os.environ,
        "AIDER_DOCKER": "1",
        "OPENAI_API_BASE": "http://127.0.0.1:8765/v1",
        "OPENAI_API_KEY": "local-eval",
        "AIDER_CPP_TEST_COMMAND": str(TEST_COMMAND),
        "LUNA_MODAL_SCORER_APP_NAME": EXPECTED_SCORER_APP,
        "LUNA_EXPECTED_TREE_SHA256": EXPECTED_TREE_SHA256,
    }
    processes: dict[str, dict[str, Any]] = {}
    suite_started = datetime.now(timezone.utc)
    for label, tries in attempts:
        attempt_receipts = modal_receipts / label
        attempt_receipts.mkdir(exist_ok=False)
        command = _command(label, tries)
        log_path = logs / f"{label}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=AIDER_ROOT,
            env={
                **environment,
                "LUNA_MODAL_RECEIPTS_DIR": str(attempt_receipts),
                "LUNA_SCORE_RECEIPTS_DIR": str(attempt_receipts),
                "LUNA_LOCAL_SOURCE_PRACTICE_ROOT": str(
                    EXERCISES_ROOT / "cpp" / "exercises" / "practice"
                ),
            },
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes[label] = {
            "process": process,
            "log_handle": log_handle,
            "log_path": log_path,
            "command": command,
            "tries": tries,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(f"started {label} pid={process.pid} tries={tries}", flush=True)

    pending = set(processes)
    failures = []
    last_progress = 0.0
    while pending:
        for label in sorted(list(pending)):
            record = processes[label]
            process = record["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            record["log_handle"].close()
            record["returncode"] = returncode
            record["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            pending.remove(label)
            print(f"completed {label} exit={returncode}", flush=True)
            if returncode != 0:
                failures.append(label)
        now = time.monotonic()
        if pending and now - last_progress >= 60:
            health = _health()
            completed_results = len(
                list(
                    (AIDER_ROOT / "tmp.benchmarks").glob(
                        f"*--{LABEL_PREFIX}-*/**/.aider.results.json"
                    )
                )
            )
            print(
                f"progress pending={len(pending)} proxy_calls={health.get('calls')} "
                f"result_files={completed_results}",
                flush=True,
            )
            last_progress = now
        if pending:
            time.sleep(2)

    for label, record in processes.items():
        matches = sorted((AIDER_ROOT / "tmp.benchmarks").glob(f"*--{label}"))
        destination = benchmark_destination / label
        if len(matches) == 1 and not destination.exists():
            shutil.copytree(matches[0], destination)
        else:
            failures.append(label)
        attempt_receipt = {
            "schema_version": 1,
            "kind": "gpt56-luna-low-fixed26-local-aider-attempt",
            "status": (
                "complete"
                if record.get("returncode") == 0 and len(matches) == 1
                else "failed"
            ),
            "label": label,
            "actual_upstream_model": EXPECTED_MODEL,
            "reasoning_effort": EXPECTED_EFFORT,
            "tries": record["tries"],
            "threads": 8,
            "command": record["command"],
            "started_at_utc": record["started_at_utc"],
            "completed_at_utc": record.get("completed_at_utc"),
            "returncode": record.get("returncode"),
            "source_output": str(matches[0]) if len(matches) == 1 else None,
            "artifact_output": str(destination) if destination.exists() else None,
            "log": str(record["log_path"]),
        }
        _write_json(runner_receipts / f"{label}.json", attempt_receipt)

    health = _health()
    suite_receipt = {
        "schema_version": 1,
        "kind": "gpt56-luna-low-fixed26-matrix-runner",
        "status": "complete" if not failures else "failed",
        "started_at_utc": suite_started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_upstream_model": EXPECTED_MODEL,
        "actual_upstream_model": health.get("model"),
        "reasoning_effort": health.get("reasoning_effort"),
        "contract": EXPECTED_OVERLAY_VERSION,
        "scorer_app": EXPECTED_SCORER_APP,
        "single_trials": SINGLE_TRIALS,
        "single_only": SINGLE_ONLY,
        "feedback_only": FEEDBACK_ONLY,
        "preflight_receipt_sha256": _sha256_file(ARTIFACT / "preflight_receipt.json"),
        "attempts": [label for label, _ in attempts],
        "failed_attempts": sorted(set(failures)),
        "proxy_health_final": health,
    }
    _write_json(ARTIFACT / "matrix_runner_receipt.json", suite_receipt)
    print(json.dumps(suite_receipt, indent=2, sort_keys=True), flush=True)
    return 0 if suite_receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
