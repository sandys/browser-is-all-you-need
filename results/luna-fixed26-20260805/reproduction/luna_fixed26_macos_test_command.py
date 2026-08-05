#!/usr/bin/env python3
"""Hash-bound, network-denied macOS scorer for the pinned fixed26 tree."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


TREE_SHA256 = os.environ["LUNA_EXPECTED_TREE_SHA256"]
PRACTICE_ROOT = Path(os.environ["LUNA_LOCAL_SOURCE_PRACTICE_ROOT"]).resolve()
RECEIPTS_DIR = Path(
    os.environ.get("LUNA_SCORE_RECEIPTS_DIR")
    or os.environ["LUNA_MODAL_RECEIPTS_DIR"]
).resolve()
APP_NAME = os.environ.get(
    "LUNA_MODAL_SCORER_APP_NAME",
    "luna-fixed26-pristine-original-v1-macos-scorer",
)
CONTRACT_VARIANT = os.environ.get(
    "LUNA_FIXED26_SCORER_VARIANT", "pristine-original-v1"
)
TASK_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SANDBOX_PROFILE = "(version 1) (allow default) (deny network*)"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        content = path.read_bytes()
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": digest(content),
            "size": len(content),
        })
    return rows, digest(canonical(rows))


def version(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return (result.stdout or result.stderr).splitlines()[0]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    started = time.time()
    candidate_root = Path.cwd().resolve()
    task = candidate_root.name
    request_id = f"{task}-{time.time_ns()}-{uuid.uuid4().hex[:12]}"
    receipt_path = RECEIPTS_DIR / f"{request_id}.json"
    try:
        if not TASK_PATTERN.fullmatch(task):
            raise RuntimeError("invalid task id")
        source_task = PRACTICE_ROOT / task
        if not source_task.is_dir():
            raise RuntimeError("task is outside the pinned fixed26 tree")
        tree_rows, actual_tree = manifest(PRACTICE_ROOT)
        if actual_tree != TREE_SHA256:
            raise RuntimeError(
                f"fixed26 source-tree digest mismatch: {actual_tree} != {TREE_SHA256}"
            )
        config = json.loads((source_task / ".meta" / "config.json").read_text())
        solution_files = config.get("files", {}).get("solution")
        if not isinstance(solution_files, list) or not solution_files:
            raise RuntimeError("invalid solution-file contract")
        files = {
            name: (candidate_root / name).read_text(encoding="utf-8")
            for name in solution_files
        }
        with TemporaryDirectory(prefix=f"fixed26-{task}-") as temporary:
            temporary_root = Path(temporary)
            task_root = temporary_root / task
            shutil.copytree(source_task, task_root)
            for name, content in files.items():
                target = task_root / name
                if target.resolve().parent != task_root.resolve():
                    raise RuntimeError(f"escaping solution path: {name}")
                target.write_text(content, encoding="utf-8")
            command = [
                "/usr/bin/sandbox-exec", "-p", SANDBOX_PROFILE,
                "/bin/bash", "-lc",
                "set -e\nmkdir build\ncd build\n"
                "cmake -DEXERCISM_RUN_ALL_TESTS=1 -G 'Unix Makefiles' ..\n"
                "make\n",
            ]
            completed = subprocess.run(
                command,
                cwd=task_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env={
                    **os.environ,
                    "CMAKE_PREFIX_PATH": "/opt/homebrew/opt/boost",
                },
            )
            output = completed.stdout.replace(str(task_root), task).replace(
                str(temporary_root), task
            )
        receipt = {
            "schema_version": 1,
            "kind": "luna-fixed26-macos-score-v1",
            "contract_variant": CONTRACT_VARIANT,
            "status": "passed" if completed.returncode == 0 else "failed",
            "request_id": request_id,
            "task": task,
            "returncode": completed.returncode,
            "output": output,
            "output_sha256": digest(output.encode()),
            "candidate_files": [
                {
                    "path": name,
                    "sha256": digest(content.encode()),
                    "size": len(content.encode()),
                }
                for name, content in sorted(files.items())
            ],
            "source_tree": {
                "sha256": actual_tree,
                "expected_sha256": TREE_SHA256,
                "file_count": len(tree_rows),
            },
            "environment": {
                "provider": "Local Mac",
                "app": APP_NAME,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "compiler": version(["g++", "--version"]),
                "cmake": version(["cmake", "--version"]),
                "boost_date_time_package": version(
                    ["brew", "list", "--versions", "boost"]
                ),
                "cpu": os.cpu_count(),
                "gpu": None,
                "network_blocked": True,
                "network_block_profile": SANDBOX_PROFILE,
            },
            "duration_seconds": time.time() - started,
        }
        write_json(receipt_path, receipt)
        sys.stdout.write(output)
        if output and not output.endswith("\n"):
            sys.stdout.write("\n")
        return completed.returncode
    except Exception as exc:
        receipt = {
            "schema_version": 1,
            "kind": "luna-fixed26-macos-score-v1",
            "status": "transport_failure",
            "request_id": request_id,
            "task": task,
            "returncode": 125,
            "error": f"{type(exc).__name__}: {exc}",
        }
        write_json(receipt_path, receipt)
        print(f"Local Mac scorer failure: {receipt['error']}")
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
