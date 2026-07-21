from __future__ import annotations

import argparse
import concurrent.futures
import gc
import inspect
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from glm47_posttraining.cpp_perf.eval import aggregate_aider_eval_records, aggregate_eval_records, write_json
from glm47_posttraining.cpp_perf.schema import AiderPolyglotTask, CppTask
from glm47_posttraining.integrations.scoring import score_aider_generation, score_generation
from glm47_posttraining.integrations.wandb_posttraining import log_eval_run, resolve_experiment_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PIE C++ prompts with SGLang and an optional LoRA adapter.")
    parser.add_argument("--data-dir", required=True, help="Prepared PIE dataset directory.")
    parser.add_argument(
        "--task-kind",
        choices=("auto", "pie", "polyglot"),
        default="auto",
        help="Task schema; auto reads manifest.kind",
    )
    parser.add_argument("--model", default="", help="Base HF model path; required unless --generated is used.")
    parser.add_argument(
        "--generated",
        default="",
        help="Replay a preserved generated JSONL through scoring instead of running model generation.",
    )
    parser.add_argument(
        "--task-allowlist",
        default="",
        help="JSON list of reference-valid task ids to retain for generation or replay.",
    )
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory to apply during generation.")
    parser.add_argument("--lora-target-modules", default="gate_proj,up_proj,down_proj")
    parser.add_argument(
        "--experts-shared-outer-loras",
        action="store_true",
        help="Serve adapters trained with the shared-outer expert LoRA contract (GLM-4.7-Flash).",
    )
    parser.add_argument(
        "--lora-use-virtual-experts",
        action="store_true",
        help="Enable SGLang virtual-expert LoRA modules for MoE expert adapters.",
    )
    parser.add_argument("--label", default="grpo")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=("sglang", "transformers"), default="sglang")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.35)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=16)
    parser.add_argument(
        "--attention-backend",
        default="",
        help="SGLang attention backend override (e.g. flashinfer, triton); "
        "empty keeps SGLang's auto choice.",
    )
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--chat-template-kwargs", default="{}")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--score-workers", type=int, default=16)
    parser.add_argument("--sandbox-image", default=os.environ.get("GLM47_CPP_SANDBOX_IMAGE", "glm47-cpp-perf:latest"))
    parser.add_argument("--sandbox-cpu", default=os.environ.get("GLM47_CPP_SANDBOX_CPU", "1"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--wandb-name", default="")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-notes", default="")
    parser.add_argument("--wandb-experiment-id", default=os.environ.get("GLM47_EXPERIMENT_ID", ""))
    parser.add_argument("--wandb-job-type", default=os.environ.get("WANDB_JOB_TYPE", "eval"))
    parser.add_argument(
        "--wandb-timing-status",
        default=os.environ.get("GLM47_TIMING_STATUS", "unverified"),
        help="Trust marker for timing-derived metrics.",
    )
    parser.add_argument("--wandb-log-artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-truncated-ratio", type=float, default=None)
    parser.add_argument("--min-valid-format-rate", type=float, default=None)
    return parser.parse_args()


def resolve_task_kind(data_dir: Path, configured: str) -> str:
    if configured != "auto":
        return configured
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "polyglot" in str(manifest.get("kind", "")).lower():
            return "polyglot"
    return "pie"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    args.task_kind = resolve_task_kind(data_dir, args.task_kind)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_dir / "eval" / "validation.jsonl")
    source_task_count = len(rows)
    task_allowlist_path = Path(args.task_allowlist) if args.task_allowlist else None
    allowed_task_ids = load_task_allowlist(task_allowlist_path) if task_allowlist_path else None
    if allowed_task_ids is not None:
        rows = filter_rows_by_task_ids(rows, allowed_task_ids)
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]
    if not rows:
        raise ValueError(f"No eval rows found in {data_dir}")

    generated_path = output_dir / f"{args.label}.generated.jsonl"
    records_path = output_dir / f"{args.label}.records.jsonl"
    summary_path = output_dir / f"{args.label}.summary.json"
    generation_summary_path = output_dir / f"{args.label}.generation_summary.json"
    receipt_path = output_dir / f"{args.label}.receipt.json"

    started_at = time.time()
    source_generated_path = Path(args.generated) if args.generated else None
    if source_generated_path is not None:
        generations = read_jsonl(source_generated_path)
        if not generations:
            raise ValueError(f"No generated rows found in {source_generated_path}")
        source_generation_sample_count = len(generations)
        if allowed_task_ids is not None:
            generations = filter_rows_by_task_ids(generations, allowed_task_ids)
        if args.max_tasks is not None:
            generations = limit_generations_by_task_count(generations, args.max_tasks)
        print(
            f"PIE eval generation reused: label={args.label} samples={len(generations)} "
            f"source={source_generated_path}",
            flush=True,
        )
    else:
        if not args.model:
            raise ValueError("--model is required unless --generated is used")
        print(
            f"PIE eval generation start: backend={args.backend} label={args.label} tasks={len(rows)} "
            f"samples_per_task={args.samples_per_task}",
            flush=True,
        )
        generations = generate_rows(args, rows)
        source_generation_sample_count = len(generations)
    write_jsonl(generated_path, generations)
    generation_summary = summarize_generations(generations, max_tokens=args.max_tokens)
    write_json(generation_summary_path, generation_summary)
    print(
        f"PIE eval generation complete: backend={args.backend} label={args.label} samples={len(generations)} "
        f"path={generated_path}",
        flush=True,
    )
    _release_cuda_memory()

    print(
        f"PIE eval scoring start: label={args.label} samples={len(generations)} "
        f"workers={args.score_workers}",
        flush=True,
    )
    records = score_rows(args, data_dir, generations)
    write_jsonl(records_path, records)
    summary = (
        aggregate_aider_eval_records(records, label=args.label)
        if args.task_kind == "polyglot"
        else aggregate_eval_records(records, label=args.label)
    )
    summary.pop("best_records", None)
    summary.update(generation_summary)
    summary.setdefault("valid_format_rate", 1.0 - float(summary.get("invalid_format_rate", 0.0)))
    experiment_id = resolve_experiment_id(
        explicit=args.wandb_experiment_id,
        run_id=args.wandb_run_id,
        label=args.label,
    )
    elapsed_seconds = time.time() - started_at
    gate_failures = eval_gate_failures(args, summary)
    summary.update(
        {
            "experiment_id": experiment_id,
            "stage": "eval",
            "status": "quality_gate_failed" if gate_failures else "success",
            "quality_gate_status": "failed" if gate_failures else "passed",
            "quality_gate_failures": gate_failures,
            "timing_status": args.wandb_timing_status,
            "timing_trustworthy": args.wandb_timing_status == "verified",
            "elapsed_seconds": elapsed_seconds,
            "source_task_count": source_task_count,
            "task_allowlist_count": len(allowed_task_ids) if allowed_task_ids is not None else None,
            "task_allowlist_excluded_count": source_task_count - len(rows),
            "task_allowlist_path": str(task_allowlist_path) if task_allowlist_path else "",
            "source_generation_sample_count": source_generation_sample_count,
            "task_allowlist_excluded_sample_count": source_generation_sample_count - len(generations),
        }
    )
    write_json(summary_path, summary)
    receipt = {
        "label": args.label,
        "experiment_id": experiment_id,
        "stage": "eval",
        "status": summary["status"],
        "quality_gate_status": summary["quality_gate_status"],
        "quality_gate_failures": gate_failures,
        "timing_status": args.wandb_timing_status,
        "data_dir": str(data_dir),
        "model": args.model,
        "adapter": args.adapter,
        "lora_target_modules": parse_lora_target_modules(args.lora_target_modules),
        "backend": args.backend,
        "output_dir": str(output_dir),
        "task_count": len(rows),
        "sample_count": len(generations),
        "samples_per_task": args.samples_per_task,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "tp_size": args.tp_size,
        "apply_chat_template": args.apply_chat_template,
        "chat_template_kwargs": parse_chat_template_kwargs(args.chat_template_kwargs),
        "elapsed_seconds": elapsed_seconds,
        "summary_path": str(summary_path),
        "generation_summary_path": str(generation_summary_path),
        "records_path": str(records_path),
        "generated_path": str(generated_path),
        "source_generated_path": str(source_generated_path) if source_generated_path is not None else "",
        "source_task_count": source_task_count,
        "task_allowlist_count": len(allowed_task_ids) if allowed_task_ids is not None else None,
        "task_allowlist_excluded_count": source_task_count - len(rows),
        "task_allowlist_path": str(task_allowlist_path) if task_allowlist_path else "",
        "source_generation_sample_count": source_generation_sample_count,
        "task_allowlist_excluded_sample_count": source_generation_sample_count - len(generations),
    }
    write_json(receipt_path, receipt)
    artifact_paths = [generated_path, records_path, summary_path, generation_summary_path, receipt_path]
    if task_allowlist_path is not None:
        artifact_paths.append(task_allowlist_path)
    log_wandb(
        args,
        summary=summary,
        records=records,
        generations=generations,
        receipt=receipt,
        artifact_paths=artifact_paths,
    )
    enforce_eval_gates(args, summary)
    print(f"PIE eval scoring complete: label={args.label} path={summary_path}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_task_allowlist(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Task allowlist must be a JSON list of strings: {path}")
    return set(data)


def filter_rows_by_task_ids(rows: list[dict[str, Any]], allowed_task_ids: set[str]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        task_id = str(row.get("task_id") or metadata.get("task_id") or "")
        if task_id in allowed_task_ids:
            filtered.append(row)
    return filtered


def limit_generations_by_task_count(
    generations: list[dict[str, Any]], max_tasks: int
) -> list[dict[str, Any]]:
    """Keep every sample for the first ``max_tasks`` task ids in a replay file."""

    selected: set[str] = set()
    limited: list[dict[str, Any]] = []
    for index, row in enumerate(generations):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        task_id = str(row.get("task_id") or metadata.get("task_id") or f"row-{index}")
        if task_id not in selected:
            if len(selected) >= max_tasks:
                continue
            selected.add(task_id)
        limited.append(row)
    return limited


def generate_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.backend == "transformers":
        return generate_rows_transformers(args, rows)
    return generate_rows_sglang(args, rows)


def generate_rows_sglang(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from sglang import Engine

    prompt_formatter = PromptFormatter(args)
    engine_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "trust_remote_code": True,
        "tp_size": args.tp_size,
        "dtype": "bfloat16",
        "mem_fraction_static": args.mem_fraction_static,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "moe_runner_backend": "triton",
        "log_level": "warning",
    }
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.adapter:
        engine_kwargs.update(
            {
                "enable_lora": True,
                "max_lora_rank": 16,
                "lora_target_modules": parse_lora_target_modules(args.lora_target_modules),
                "lora_backend": "triton",
            }
        )
        if args.experts_shared_outer_loras:
            engine_kwargs["experts_shared_outer_loras"] = True
        if args.lora_use_virtual_experts:
            engine_kwargs["lora_use_virtual_experts"] = True

    engine = Engine(**sglang_engine_kwargs(engine_kwargs))
    try:
        if args.adapter:
            engine.load_lora_adapter(args.label, args.adapter)

        generations: list[dict[str, Any]] = []
        expanded = []
        for row in rows:
            for sample_index in range(args.samples_per_task):
                expanded.append((row, sample_index))

        sampling_params = {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        for start in range(0, len(expanded), args.batch_size):
            batch = expanded[start : start + args.batch_size]
            prompts = [prompt_formatter.format(str(row["prompt"])) for row, _sample_index in batch]
            lora_paths = [args.label] * len(prompts) if args.adapter else None
            outputs = engine.generate(prompts, sampling_params, lora_path=lora_paths)
            if isinstance(outputs, dict):
                outputs = [outputs]
            for (row, sample_index), output in zip(batch, outputs, strict=True):
                generations.append(
                    {
                        "label": args.label,
                        "task_id": row.get("task_id"),
                        "problem_id": row.get("problem_id"),
                        "split": row.get("split"),
                        "sample_index": sample_index,
                        "metadata": row.get("metadata", {}),
                        "response": output_text(output),
                        "finish_reason": output_finish_reason(output),
                        "completion_tokens": output_token_count(output, "completion"),
                        "prompt_tokens": output_token_count(output, "prompt"),
                        "truncated": output_is_truncated(output, args.max_tokens),
                    }
                )
            if len(generations) == len(batch) or len(generations) % 100 == 0 or len(generations) == len(expanded):
                print(
                    "SGLang PIE eval generation progress: "
                    f"{len(generations)}/{len(expanded)}",
                    flush=True,
                )
        return generations
    finally:
        engine.shutdown()


def generate_rows_transformers(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.adapter:
        raise ValueError("--backend transformers does not support --adapter in this eval script")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt_formatter = PromptFormatter(args)
    tokenizer = prompt_formatter.tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    eos_token_ids = _eos_token_ids(tokenizer)
    generations: list[dict[str, Any]] = []
    expanded = []
    for row in rows:
        for sample_index in range(args.samples_per_task):
            expanded.append((row, sample_index))

    do_sample = args.temperature > 0.0
    with torch.inference_mode():
        for index, (row, sample_index) in enumerate(expanded, start=1):
            prompt = prompt_formatter.format(str(row["prompt"]))
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = generated_ids[0, input_ids.shape[-1] :]
            completion_tokens = int(completion_ids.numel())
            response = tokenizer.decode(completion_ids, skip_special_tokens=True)
            finish_reason = "length" if completion_tokens >= args.max_tokens else "unknown"
            if completion_tokens and int(completion_ids[-1]) in eos_token_ids:
                finish_reason = "stop"
            generations.append(
                {
                    "label": args.label,
                    "task_id": row.get("task_id"),
                    "problem_id": row.get("problem_id"),
                    "split": row.get("split"),
                    "sample_index": sample_index,
                    "metadata": row.get("metadata", {}),
                    "response": response,
                    "finish_reason": finish_reason,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "truncated": completion_tokens >= args.max_tokens,
                }
            )
            if index == 1 or index == len(expanded) or index % 10 == 0:
                print(
                    "Transformers PIE eval generation progress: "
                    f"{index}/{len(expanded)}",
                    flush=True,
                )
    return generations


def _eos_token_ids(tokenizer: Any) -> set[int]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, list):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def output_text(output: Any) -> str:
    if isinstance(output, dict):
        for key in ("text", "output_text", "content"):
            value = output.get(key)
            if isinstance(value, str):
                return value
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return output_text(outputs[0])
    return str(output)


def sglang_engine_kwargs(engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs accepted by the installed SGLang ServerArgs class."""

    try:
        from sglang.srt.server_args import ServerArgs
    except ImportError:
        return engine_kwargs
    signature = inspect.signature(ServerArgs.__init__)
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return engine_kwargs
    return _filter_sglang_server_args(engine_kwargs, set(parameters))


def _filter_sglang_server_args(engine_kwargs: dict[str, Any], server_arg_names: set[str]) -> dict[str, Any]:
    compatible = dict(engine_kwargs)
    cuda_graph_max_bs = compatible.get("cuda_graph_max_bs")
    if cuda_graph_max_bs is not None and "cuda_graph_max_bs" not in server_arg_names:
        compatible.pop("cuda_graph_max_bs", None)
        if "cuda_graph_max_bs_decode" in server_arg_names:
            compatible["cuda_graph_max_bs_decode"] = cuda_graph_max_bs
        if "cuda_graph_max_bs_prefill" in server_arg_names:
            compatible["cuda_graph_max_bs_prefill"] = cuda_graph_max_bs
        if "cuda_graph_max_bs_for_capture" in server_arg_names:
            compatible["cuda_graph_max_bs_for_capture"] = cuda_graph_max_bs
    return {key: value for key, value in compatible.items() if key in server_arg_names}


class PromptFormatter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.apply_chat_template = bool(args.apply_chat_template)
        self.system_prompt = str(args.system_prompt or "")
        self.chat_template_kwargs = parse_chat_template_kwargs(args.chat_template_kwargs)
        self.tokenizer = None
        if self.apply_chat_template:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def format(self, prompt: str) -> str:
        if not self.apply_chat_template:
            return prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        assert self.tokenizer is not None
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        )


def parse_chat_template_kwargs(raw_value: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise ValueError("--chat-template-kwargs must decode to a JSON object")
    return value


def parse_lora_target_modules(raw_value: str) -> list[str]:
    modules = [item.strip() for item in raw_value.replace(",", " ").split()]
    if not modules:
        raise ValueError("--lora-target-modules must name at least one module")
    return modules


def output_finish_reason(output: Any) -> str | None:
    if isinstance(output, dict):
        meta_info = output.get("meta_info")
        if isinstance(meta_info, dict):
            finish_reason = meta_info.get("finish_reason")
            if isinstance(finish_reason, dict):
                reason_type = finish_reason.get("type") or finish_reason.get("reason")
                return str(reason_type) if reason_type is not None else json.dumps(finish_reason, sort_keys=True)
            if finish_reason is not None:
                return str(finish_reason)
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return output_finish_reason(outputs[0])
    return None


def output_token_count(output: Any, token_kind: str) -> int | None:
    keys = (
        ("completion_tokens", "num_completion_tokens", "output_tokens", "num_output_tokens")
        if token_kind == "completion"
        else ("prompt_tokens", "num_prompt_tokens", "input_tokens", "num_input_tokens")
    )
    if isinstance(output, dict):
        meta_info = output.get("meta_info")
        if isinstance(meta_info, dict):
            for key in keys:
                value = meta_info.get(key)
                if isinstance(value, int):
                    return value
        for key in keys:
            value = output.get(key)
            if isinstance(value, int):
                return value
        outputs = output.get("outputs")
        if isinstance(outputs, list) and outputs:
            return output_token_count(outputs[0], token_kind)
    return None


def output_is_truncated(output: Any, max_tokens: int) -> bool:
    finish_reason = output_finish_reason(output)
    if finish_reason and finish_reason.lower() in {"length", "max_tokens", "abort_length"}:
        return True
    completion_tokens = output_token_count(output, "completion")
    return completion_tokens is not None and completion_tokens >= max_tokens


def summarize_generations(generations: list[dict[str, Any]], *, max_tokens: int) -> dict[str, Any]:
    count = len(generations)
    truncated = [item for item in generations if item.get("truncated") is True]
    response_lengths = [len(str(item.get("response", ""))) for item in generations]
    completion_tokens = [
        int(item["completion_tokens"]) for item in generations if isinstance(item.get("completion_tokens"), int)
    ]
    prompt_tokens = [int(item["prompt_tokens"]) for item in generations if isinstance(item.get("prompt_tokens"), int)]
    finish_reasons = Counter(str(item.get("finish_reason") or "unknown") for item in generations)
    return {
        "generation_sample_count": count,
        "max_tokens": max_tokens,
        "truncated_count": len(truncated),
        "truncated_ratio": len(truncated) / count if count else 0.0,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "mean_response_chars": _mean_float(response_lengths),
        "mean_completion_tokens": _mean_float(completion_tokens),
        "mean_prompt_tokens": _mean_float(prompt_tokens),
    }


def _mean_float(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def log_wandb(
    args: argparse.Namespace,
    *,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    receipt: dict[str, Any],
    artifact_paths: list[Path],
) -> None:
    if not args.wandb_project:
        return
    try:
        import wandb
    except ImportError:
        print("W&B logging requested but wandb is not installed; continuing without W&B.", flush=True)
        return
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    experiment_id = resolve_experiment_id(
        explicit=args.wandb_experiment_id,
        run_id=args.wandb_run_id,
        label=args.label,
    )
    result = log_eval_run(
        wandb,
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        experiment_id=experiment_id,
        run_id=args.wandb_run_id or f"{experiment_id}-{args.label}-eval",
        name=args.wandb_name or args.wandb_run_id or f"{experiment_id}-{args.label}-eval",
        group=args.wandb_group or experiment_id,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        timing_status=args.wandb_timing_status,
        summary=summary,
        records=records,
        generations=generations,
        config={
            **receipt,
            "notes": args.wandb_notes,
            "max_tasks": args.max_tasks,
            "batch_size": args.batch_size,
            "mem_fraction_static": args.mem_fraction_static,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "system_prompt": args.system_prompt,
        },
        artifact_paths=artifact_paths if args.wandb_log_artifacts else [],
        manifest_dir=args.output_dir,
        tags=tags,
    )
    print(f"wandb_proof_run id={result['run_id']} url={result['url']}", flush=True)


def enforce_eval_gates(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    failures = eval_gate_failures(args, summary)
    if failures:
        raise SystemExit("; ".join(failures))


def eval_gate_failures(args: argparse.Namespace, summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if args.max_truncated_ratio is not None and float(summary["truncated_ratio"]) > args.max_truncated_ratio:
        failures.append(
            f"truncated_ratio {summary['truncated_ratio']:.4f} exceeds gate {args.max_truncated_ratio:.4f}"
        )
    if args.min_valid_format_rate is not None and float(summary["valid_format_rate"]) < args.min_valid_format_rate:
        failures.append(
            f"valid_format_rate {summary['valid_format_rate']:.4f} below gate {args.min_valid_format_rate:.4f}"
        )
    return failures


def score_rows(args: argparse.Namespace, data_dir: Path, generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def score_one(item: dict[str, Any]) -> dict[str, Any]:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        task_path = Path(str(metadata["task_path"]))
        if not task_path.is_absolute():
            task_path = data_dir / task_path
        if args.task_kind == "polyglot":
            task = AiderPolyglotTask.read_json(task_path)
            record = score_aider_generation(
                task,
                str(item.get("response", "")),
                label=args.label,
                sample_index=int(item.get("sample_index", 0)),
                image=args.sandbox_image,
                context_exhausted=bool(item.get("truncated")) or str(item.get("finish_reason") or "").lower()
                in {"length", "max_tokens", "abort_length", "context_exhausted"},
            )
        else:
            task = CppTask.read_json(task_path)
            record = score_generation(
                task,
                str(item.get("response", "")),
                label=args.label,
                sample_index=int(item.get("sample_index", 0)),
                image=args.sandbox_image,
                cpu=args.sandbox_cpu,
            )
        record["task_id"] = item.get("task_id") or record.get("task_id")
        record["problem_id"] = item.get("problem_id") or record.get("problem_id")
        record["split"] = item.get("split") or record.get("split")
        return record

    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.score_workers)) as executor:
        futures = [executor.submit(score_one, item) for item in generations]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            records.append(future.result())
            if index == 1 or index == len(futures) or index % 25 == 0:
                print(f"SGLang PIE eval scoring progress: {index}/{len(futures)}", flush=True)
    records.sort(key=lambda row: (str(row.get("task_id")), int(row.get("sample_index") or 0)))
    return records


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
