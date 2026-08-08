#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from aider import models
from aider.coders import Coder
from aider.io import InputOutput

MODEL = "openai/glm-4.7-flash-grpo"
TARGET = "include/fmt/chrono.h"
BASE_COMMIT = "06f1c0d725855861535e9e65cd4d502aca7c61ed"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wait_for(path: Path, alternate: Path, timeout: int = 3600) -> Path:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return path
        if alternate.exists():
            return alternate
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {path.name} or {alternate.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--edit-format", choices=("diff", "udiff", "whole"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    template = root / "workspace-template"
    trial = root / "trials" / args.trial
    workspace = trial / "workspace"
    if trial.exists():
        raise SystemExit(f"refusing to reuse trial path: {trial}")
    if not (template / TARGET).is_file():
        raise SystemExit("workspace template is missing")
    trial.mkdir(parents=True)
    shutil.copytree(template, workspace)

    prompt_path = root / "task-prompt.md"
    settings_path = root / "model-settings.yml"
    prompt = prompt_path.read_text(encoding="utf-8")
    models.register_models([str(settings_path)])
    os.environ.update(
        {
            "AIDER_DOCKER": "1",
            "OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
            "OPENAI_API_KEY": "local-eval",
        }
    )
    history = trial / ".aider.chat.history.md"
    io = InputOutput(pretty=False, yes=True, chat_history_file=history)
    main_model = models.Model(MODEL, weak_model=None, verbose=False)
    os.chdir(workspace)
    coder = Coder.create(
        main_model,
        args.edit_format,
        io,
        fnames=[str((workspace / TARGET).resolve())],
        use_git=False,
        stream=False,
        verbose=False,
        auto_lint=False,
        cache_prompts=True,
        suggest_shell_commands=False,
        ignore_mentions={"test/chrono-test.cc", "CMakeLists.txt"},
    )
    coder.max_reflections = 0
    coder.show_announcements()
    coder.get_file_mentions = lambda _: set()

    started = datetime.now(timezone.utc).isoformat()
    response1 = coder.run(with_message=prompt, preproc=False)
    (trial / "raw-response-turn1.txt").write_text(response1, encoding="utf-8")
    (trial / "turn1.ready").write_text("ready\n", encoding="utf-8")

    signal = wait_for(trial / "feedback-turn1.txt", trial / "stop-after-turn1")
    response2_path = None
    if signal.name == "feedback-turn1.txt":
        feedback = signal.read_text(encoding="utf-8", errors="replace")
        response2 = coder.run(with_message=feedback, preproc=False)
        response2_path = trial / "raw-response-turn2.txt"
        response2_path.write_text(response2, encoding="utf-8")
        (trial / "turn2.ready").write_text("ready\n", encoding="utf-8")

    completed = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "trial": args.trial,
        "started_at_utc": started,
        "completed_at_utc": completed,
        "model": MODEL,
        "base_model_revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        "checkpoint": "Synth v1 epoch 50",
        "iteration": 649,
        "lora_rank": 16,
        "source_adapter_sha256": "4acb7f23c295f45380155c5d9ee6bc59422262f0cb51f0c02f7e550d405b575a",
        "aider_commit": "5dc9490bb35f9729ef2c95d00a19ccd30c26339c",
        "edit_format": args.edit_format,
        "repo_map": False,
        "thinking_disabled": False,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_completion_tokens": 32768,
        "auto_lint": False,
        "max_internal_reflections": 0,
        "explicit_seed": None,
        "base_commit": BASE_COMMIT,
        "editable_file": TARGET,
        "task_prompt_sha256": sha256(prompt_path),
        "accepted_implementation_disclosed": False,
        "turn1_response_sha256": sha256(trial / "raw-response-turn1.txt"),
        "turn2_response_sha256": sha256(response2_path) if response2_path else None,
        "chat_history_sha256": sha256(history),
    }
    (trial / "generation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
