"""LoRA SFT utilities for ScaleCUA rows converted to OSWorld tool calls."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


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
    for row in rows:
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
    if skipped:
        print(f"Skipped {skipped} rows with missing/unreadable images from {source}.")
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
        import torch
        from peft import LoraConfig, get_peft_model
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        from torch.utils.data import Dataset
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ScaleCUA SFT dependencies are missing. Install them with `uv sync --extra sft` "
            "or run through `uv run --extra sft w8-biayn scalecua sft ...`."
        ) from exc

    class JsonlDataset(Dataset):
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.rows[index]

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    def verify_image(path: str) -> None:
        with Image.open(path) as image:
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
        report_to=["wandb"] if config.wandb_project else [],
        run_name=config.wandb_run_name,
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
