#!/usr/bin/env python3
"""Preflight the Polyglot schema, format, reward, and optional Catch2 runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from glm47_posttraining.cpp_perf.dataset import aider_sft_output, build_aider_prompt, load_aider_tasks
from glm47_posttraining.cpp_perf.reward import compute_aider_reward, valid_aider_model_output
from glm47_posttraining.cpp_perf.sandbox import build_sandbox_image
from glm47_posttraining.cpp_perf.schema import Catch2HarnessResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--skip-sandbox", choices=("0", "1"), default="0")
    parser.add_argument("--backend", choices=("local", "docker"), default="local")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument(
        "--skip-thread-sanitizer",
        action="store_true",
        help="Skip the state-task TSan preflight (never recommended for training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_image:
        built = build_sandbox_image()
        if built.returncode != 0:
            raise SystemExit(f"sandbox image build failed:\n{built.stdout}\n{built.stderr}")
    loaded = load_aider_tasks(args.tasks_dir)
    if not loaded:
        raise SystemExit(f"no tasks found under {args.tasks_dir}")
    task = next((item for _path, item in loaded if item.task_id == "allergies"), loaded[0][1])

    prompt = build_aider_prompt(task)
    assert task.task_id in prompt
    assert all(name in prompt for name in task.solution_files)
    assert not any(content in prompt for content in task.test_files.values())

    oracle_output = aider_sft_output(task)
    assert valid_aider_model_output(oracle_output, task)
    invalid = compute_aider_reward(task, "```cpp\nint main() {}\n```")
    assert invalid.reward == -1.0 and invalid.reason == "rubric1_invalid_format"

    fake = compute_aider_reward(
        task,
        oracle_output,
        runner=lambda _task, _files: Catch2HarnessResult(
            passed_test_cases=2,
            total_test_cases=2,
            passed_assertions=4,
            total_assertions=4,
        ),
    )
    assert fake.reason == "correct_all_rubrics_passed" and fake.reward == 0.325

    if args.skip_sandbox == "0":
        os.environ["GLM47_CPP_SANDBOX_BACKEND"] = args.backend
        real = compute_aider_reward(task, oracle_output)
        if real.reason != "correct_all_rubrics_passed":
            details = real.harness.logs if real.harness else {}
            raise SystemExit(f"sandbox preflight failed: {real.reason}: {details}")

        state_task = next(
            (
                item
                for _path, item in loaded
                if item.effective_rubric_category == "state_concurrency"
            ),
            None,
        )
        if state_task is not None and not args.skip_thread_sanitizer:
            state_result = compute_aider_reward(state_task, aider_sft_output(state_task))
            state_harness = state_result.harness
            if (
                state_result.reason != "correct_all_rubrics_passed"
                or state_harness is None
                or not state_harness.thread_sanitizer_ran
                or state_harness.thread_sanitizer_error
                or state_harness.thread_sanitizer_timeout
            ):
                details = state_harness.logs if state_harness else {}
                raise SystemExit(
                    "thread-sanitizer preflight failed for "
                    f"{state_task.task_id}: {state_result.reason}: {details}"
                )

    print(
        f"Aider Polyglot preflight passed: task={task.task_id} "
        f"sandbox={'skipped' if args.skip_sandbox == '1' else args.backend}"
    )


if __name__ == "__main__":
    main()
