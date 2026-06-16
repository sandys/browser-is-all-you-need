"""Generate and score C++ optimization candidates for evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from w8_biayn.cpp_perf.eval import aggregate_eval_records, write_json
from w8_biayn.cpp_perf.reward import compute_reward
from w8_biayn.cpp_perf.sandbox import DEFAULT_CPU, DEFAULT_DOCKER_IMAGE, run_in_sandbox
from w8_biayn.cpp_perf.schema import CppTask


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(Path(args.data_dir) / "grpo" / "validation.parquet").to_pylist()
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.max_model_len, trust_remote_code=True)
    sampling = SamplingParams(
        n=args.samples_per_task,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    prompts = [_prompt_text(row) for row in rows]
    outputs = llm.generate(prompts, sampling)

    records_path = output_dir / f"{args.label}.records.jsonl"
    records: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as handle:
        for row, request_output in zip(rows, outputs, strict=True):
            task = CppTask.read_json(Path(args.data_dir) / row["extra_info"]["task_path"])
            for sample_index, generated in enumerate(request_output.outputs):
                record = score_generation(
                    task,
                    generated.text,
                    label=args.label,
                    sample_index=sample_index,
                    image=args.sandbox_image,
                    cpu=args.sandbox_cpu,
                )
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
    summary = aggregate_eval_records(records, label=args.label)
    summary.pop("best_records", None)
    write_json(output_dir / f"{args.label}.summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--output-dir", default="/tmp/w8-cpp-eval")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--sandbox-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument("--sandbox-cpu", default=DEFAULT_CPU)
    return parser.parse_args()


def score_generation(
    task: CppTask,
    model_output: str,
    *,
    label: str,
    sample_index: int,
    image: str,
    cpu: str,
) -> dict[str, Any]:
    """Score one generated answer and flatten key harness fields."""

    def runner(candidate_task: CppTask, code: str):
        return run_in_sandbox(candidate_task, code, image=image, cpu=cpu)

    breakdown = compute_reward(task, model_output, runner=runner)
    harness = breakdown.harness
    record: dict[str, Any] = {
        "label": label,
        "task_id": task.task_id,
        "problem_id": task.problem_id,
        "sample_index": sample_index,
        "reward": breakdown.reward,
        "reason": breakdown.reason,
        "reference_metric": task.reference.metric,
        "reference_value": task.reference.value,
        "runtime_cpu_ns": harness.runtime_cpu_ns if harness else None,
        "runtime_wall_ns": harness.runtime_wall_ns if harness else None,
        "reference_runtime_cpu_ns": harness.reference_runtime_cpu_ns if harness else None,
        "reference_runtime_wall_ns": harness.reference_runtime_wall_ns if harness else None,
        "runtime_speedup": harness.runtime_speedup if harness else None,
        "all_tests_pass": harness.all_tests_pass if harness else False,
        "compile_error": harness.compile_error if harness else False,
        "sanitizer_error": harness.sanitizer_error if harness else False,
        "timeout": harness.timeout if harness else False,
        "tests_passed": harness.tests_passed if harness else 0,
        "tests_total": harness.tests_total if harness else 0,
    }
    return record


def _prompt_text(row: dict[str, Any]) -> str:
    prompt = row["prompt"]
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
    return str(prompt)


if __name__ == "__main__":
    main()
