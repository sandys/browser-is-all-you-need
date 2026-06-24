"""LoRA SFT utilities for ScaleCUA rows converted to OSWorld tool calls."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm ships with the SFT stack, but keep a fallback.
    tqdm = None


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1024 * 28 * 28
AUTO_DOCSTRING_NOISE_RE = re.compile(
    r"^\[ERROR\] `[^`]+` is part of .* signature, but not documented\."
)


@dataclass(frozen=True)
class SftConfig:
    model: str
    train: Path
    output: Path
    eval: Path | None = None
    max_steps: int = 100
    batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    limit: int | None = None
    bf16: bool = True
    gradient_checkpointing: bool = True
    min_pixels: int = DEFAULT_MIN_PIXELS
    max_pixels: int = DEFAULT_MAX_PIXELS
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None


def load_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} did not contain any training rows")
    return rows


def filter_rows_with_valid_images(
    rows: list[dict[str, Any]],
    *,
    image_loader: Callable[[str], None],
    source: Path,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    skipped = 0
    iterator: Iterable[dict[str, Any]]
    if tqdm is not None:
        iterator = tqdm(
            rows,
            desc=f"Verifying images: {source.name}",
            unit="row",
            dynamic_ncols=True,
            smoothing=0.05,
        )
    else:
        print(f"Verifying images from {source} ({len(rows)} rows)...")
        iterator = rows
    for row in iterator:
        image = row.get("image")
        if not isinstance(image, str) or not image:
            skipped += 1
            continue
        try:
            image_loader(image)
        except (FileNotFoundError, OSError, ValueError):
            skipped += 1
            continue
        kept.append(row)
    if not kept:
        raise ValueError(f"{source} did not contain any rows with readable images")
    print(
        f"Verified {len(kept)} usable rows from {source}; skipped {skipped} missing/unreadable rows."
    )
    return kept


def _message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError(f"row is missing {role!r} message content")


def qwen_messages_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    image = row.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("row is missing image path")
    user_text = _message_content(row, "user").replace("<image>", "").strip()
    assistant_text = _message_content(row, "assistant").strip()
    if "<tool_call>" not in assistant_text:
        raise ValueError("assistant target must contain an OSWorld <tool_call>")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": assistant_text},
    ]


def mask_prompt_tokens(labels: Any, prompt_lengths: Iterable[int], pad_token_id: int | None = None) -> Any:
    """Mask prompt and padding tokens so loss is only computed on assistant targets."""

    for row_idx, prompt_len in enumerate(prompt_lengths):
        labels[row_idx, :prompt_len] = -100
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    return labels


def configure_tracking(config: SftConfig) -> tuple[list[str], str | None]:
    report_to: list[str] = []
    run_name = config.wandb_run_name or config.mlflow_run_name
    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        report_to.append("wandb")
    if config.mlflow_tracking_uri or config.mlflow_experiment or config.mlflow_run_name:
        if config.wandb_run_name and config.mlflow_run_name and config.wandb_run_name != config.mlflow_run_name:
            raise ValueError("wandb_run_name and mlflow_run_name must match when both are set")
        if config.mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = config.mlflow_tracking_uri
        if config.mlflow_experiment:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = config.mlflow_experiment
        report_to.append("mlflow")
    return report_to, run_name


def training_args_eval_kwargs(training_args_cls: Any, *, has_eval: bool, eval_steps: int) -> dict[str, Any]:
    if not has_eval:
        return {}
    signature = inspect.signature(training_args_cls.__init__)
    kwargs: dict[str, Any] = {"eval_steps": eval_steps}
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("Installed transformers TrainingArguments has no eval strategy parameter")
    return kwargs


def _filtered_import_output(buffer: io.StringIO) -> str:
    kept_lines = [
        line
        for line in buffer.getvalue().splitlines()
        if not AUTO_DOCSTRING_NOISE_RE.match(line.strip())
    ]
    return "\n".join(kept_lines).strip()


def import_sft_dependencies() -> dict[str, Any]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        torch = importlib.import_module("torch")
        peft = importlib.import_module("peft")
        pil_image = importlib.import_module("PIL.Image")
        qwen_vl_utils = importlib.import_module("qwen_vl_utils")
        transformers = importlib.import_module("transformers")

    stdout_text = _filtered_import_output(stdout_buffer)
    stderr_text = _filtered_import_output(stderr_buffer)
    if stdout_text:
        print(stdout_text)
    if stderr_text:
        print(stderr_text, file=os.sys.stderr)

    return {
        "torch": torch,
        "LoraConfig": peft.LoraConfig,
        "get_peft_model": peft.get_peft_model,
        "ImageModule": pil_image,
        "process_vision_info": qwen_vl_utils.process_vision_info,
        "Dataset": importlib.import_module("torch.utils.data").Dataset,
        "AutoProcessor": transformers.AutoProcessor,
        "Qwen2_5_VLForConditionalGeneration": transformers.Qwen2_5_VLForConditionalGeneration,
        "Trainer": transformers.Trainer,
        "TrainingArguments": transformers.TrainingArguments,
    }


class ScaleCuaDataCollator:
    """Create Qwen2.5-VL batches and assistant-only labels from converted JSONL rows."""

    def __init__(
        self,
        *,
        processor: Any,
        process_vision_info: Callable[[list[dict[str, Any]]], tuple[list[Any] | None, list[Any] | None]],
    ) -> None:
        self.processor = processor
        self.process_vision_info = process_vision_info

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[Any] = []
        videos: list[Any] = []
        skipped_rows = 0
        for row in rows:
            try:
                messages = qwen_messages_from_row(row)
                full_texts.append(
                    self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                )
                prompt_texts.append(
                    self.processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
                )
                image_inputs, video_inputs = self.process_vision_info(messages)
            except (FileNotFoundError, OSError, ValueError):
                skipped_rows += 1
                continue
            if image_inputs:
                images.extend(image_inputs)
            if video_inputs:
                videos.extend(video_inputs)

        if not full_texts:
            raise ValueError("All rows in the batch had missing or unreadable images")

        batch = self.processor(
            text=full_texts,
            images=images or None,
            videos=videos or None,
            padding=True,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images or None,
            videos=videos or None,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        prompt_lengths = [int(mask.sum().item()) for mask in prompt_batch["attention_mask"]]
        batch["labels"] = mask_prompt_tokens(labels, prompt_lengths, self.processor.tokenizer.pad_token_id)
        return batch


def run_sft(config: SftConfig) -> Path:
    """Run single-node Qwen2.5-VL LoRA SFT and return the adapter output path."""

    try:
        deps = import_sft_dependencies()
    except ImportError as exc:
        raise RuntimeError(
            "ScaleCUA SFT dependencies are missing. Install them with `uv sync --extra sft` "
            "or run through `uv run --extra sft w8-biayn scalecua sft ...`."
        ) from exc
    torch = deps["torch"]
    LoraConfig = deps["LoraConfig"]
    get_peft_model = deps["get_peft_model"]
    ImageModule = deps["ImageModule"]
    process_vision_info = deps["process_vision_info"]
    Dataset = deps["Dataset"]
    AutoProcessor = deps["AutoProcessor"]
    Qwen2_5_VLForConditionalGeneration = deps["Qwen2_5_VLForConditionalGeneration"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    class JsonlDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.rows[index]

    report_to, run_name = configure_tracking(config)

    def verify_image(path: str) -> None:
        with ImageModule.open(path) as image:
            image.load()

    rows = filter_rows_with_valid_images(
        load_jsonl(config.train, limit=config.limit),
        image_loader=verify_image,
        source=config.train,
    )
    eval_rows = None
    if config.eval:
        eval_rows = filter_rows_with_valid_images(
            load_jsonl(config.eval),
            image_loader=verify_image,
            source=config.eval,
        )
    processor = AutoProcessor.from_pretrained(
        config.model,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(DEFAULT_TARGET_MODULES),
    )
    model = get_peft_model(model, lora_config)

    eval_kwargs = training_args_eval_kwargs(
        TrainingArguments, has_eval=eval_rows is not None, eval_steps=config.eval_steps
    )
    training_args = TrainingArguments(
        output_dir=str(config.output),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        bf16=config.bf16,
        fp16=not config.bf16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
        **eval_kwargs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=JsonlDataset(rows),
        eval_dataset=JsonlDataset(eval_rows) if eval_rows is not None else None,
        data_collator=ScaleCuaDataCollator(processor=processor, process_vision_info=process_vision_info),
    )
    trainer.train()
    config.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output)
    processor.save_pretrained(config.output)
    return config.output
