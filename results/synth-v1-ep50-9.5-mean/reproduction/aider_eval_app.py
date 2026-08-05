"""Pinned fixed-26 Aider C++ evaluation for a base model or gated adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal


OVERLAY_EXPECTED_VERSION = "fixed26-contract-v2"
MODEL_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
MODEL_PATH = f"/models/zai-org--GLM-4.7-Flash/{MODEL_REVISION}"
AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SOURCE_TENSORS = 9_741
EXPECTED_LAYER_47_TENSORS = 207
EXPECTED_SERVING_TENSORS = 9_534
EVAL_TAG = os.environ.get("GLM47_EVAL_TAG", "rl-grpo")
EXPECTED_TRAINING_GATE_KIND = os.environ.get(
    "GLM47_TRAINING_GATE_KIND", "glm47-aider-grpo-training-gate"
)
EXPECTED_TRAINING_PHASE = os.environ.get("GLM47_EXPECTED_TRAINING_PHASE", "full")
EXPECTED_TRAINING_TASK_COUNT = int(os.environ.get("GLM47_EXPECTED_TRAINING_TASK_COUNT", "253"))
DISABLE_THINKING = os.environ.get("GLM47_EVAL_DISABLE_THINKING", "0") == "1"
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
            # Baked from the launching shell so the container agrees with the caller.
            "GLM47_EVAL_DISABLE_THINKING": "1" if DISABLE_THINKING else "0",
        }
    )
    .run_commands(
        "python3 -m pip install --no-cache-dir --no-deps --upgrade "
        "flashinfer-python==0.6.12 flashinfer-cubin==0.6.12",
        "python3 -m pip install --no-cache-dir --no-deps --upgrade "
        "flashinfer-jit-cache==0.6.12 --index-url https://flashinfer.ai/whl/cu129/",
        "python3 -m pip install --no-cache-dir --no-deps --force-reinstall "
        "sglang-kernel==0.4.4 --index-url https://docs.sglang.ai/whl/cu129/",
        "python3 -m pip install --no-cache-dir --no-deps --upgrade torch-memory-saver==0.0.9.post1",
    )
    .apt_install("git", "cmake", "make", "g++", "curl", "python3-venv")
    .run_commands(
        f"git clone https://github.com/Aider-AI/aider.git /aider && git -C /aider checkout {AIDER_COMMIT}",
        f"git clone https://github.com/Aider-AI/polyglot-benchmark.git /aider/tmp.benchmarks/polyglot-benchmark && git -C /aider/tmp.benchmarks/polyglot-benchmark checkout {POLYGLOT_COMMIT}",
        "python3 -m venv /opt/aider-venv && /opt/aider-venv/bin/pip install -e '/aider[dev]'",
    )
    # Fairness overlay: the polyglot instructions never state the interface the
    # hidden test file requires, so every exercise is patched with an explicit
    # contract before it is served. Hash-pinned; see the module docstring.
    .add_local_file(
        Path(__file__).with_name("aider_fixed26_contract_overlay.py"),
        "/opt/fixed26/aider_fixed26_contract_overlay.py",
    )
    .add_local_file(
        Path(__file__).with_name("aider_fixed26_originals.json"),
        "/opt/fixed26/aider_fixed26_originals.json",
    )
)

models = modal.Volume.from_name("w8-glm47-flash-models", create_if_missing=False)
runs = modal.Volume.from_name("glm47-runs", create_if_missing=False)
results = modal.Volume.from_name("w8-aider-polyglot-cpp-results", create_if_missing=False)
training_assets = modal.Volume.from_name("glm47-assets", create_if_missing=False)


def _model_settings_yaml(lora_name: str | None = None) -> str:
    """Aider model settings. Thinking is on by default, matching the frozen contract.

    RL rollouts generate with thinking suppressed while this evaluation generates with
    it enabled, so the trained and measured regimes differ. Setting
    GLM47_EVAL_DISABLE_THINKING=1 measures the same checkpoint in the regime it was
    actually trained in. Everything else stays identical so the two are comparable.
    """

    extra_body_lines = []
    if lora_name is not None:
        # SGLang only applies a loaded LoRA when the request selects it, either via
        # "base:adapter" model syntax or an explicit top-level lora_path field.
        # Without this line every request silently runs the base weights.
        extra_body_lines.append(f"      lora_path: {lora_name}")
    if DISABLE_THINKING:
        extra_body_lines.append("      chat_template_kwargs:\n        enable_thinking: false")
    extra_body = (
        "\n    extra_body:\n" + "\n".join(extra_body_lines) if extra_body_lines else ""
    )
    return (
        "- name: openai/glm-4.7-flash-grpo\n"
        "  edit_format: whole\n"
        "  use_repo_map: false\n"
        "  use_temperature: true\n"
        "  streaming: false\n"
        "  extra_params:\n"
        "    max_tokens: 32768\n"
        "    temperature: 0.7\n"
        "    top_p: 1.0"
        f"{extra_body}\n"
    )


def validate_adapter_path(adapter_path: str) -> Path:
    path = PurePosixPath(adapter_path)
    if (
        not path.is_absolute()
        or len(path.parts) < 4
        or path.parts[1] != "runs"
        or ".." in path.parts
    ):
        raise ValueError("adapter_path must be an absolute checkpoint path beneath /runs")
    return Path(str(path))


def validate_run_id(run_id: str) -> str:
    value = (
        run_id.strip()
        if run_id
        else (
            f"glm47-aider-{EVAL_TAG}-grpo-eval-"
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
        )
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
    expected_training_task_count: int = EXPECTED_TRAINING_TASK_COUNT,
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

    merged_receipt_path = source / "merged_receipt.json"
    if merged_receipt_path.is_file():
        # A merged adapter blends two independently gated fine-tunes, so it belongs
        # to no single run's checkpoint catalog. Verify its own bytes, then verify
        # every parent through this same gate check against that parent's own run.
        merged = json.loads(merged_receipt_path.read_text(encoding="utf-8"))
        if merged.get("kind") != "glm47-merged-adapter":
            raise RuntimeError("merged adapter receipt has an unknown kind")
        if merged.get("merged_adapter_model_sha256") != adapter_sha256:
            raise RuntimeError("merged adapter bytes do not match their receipt")
        if merged.get("merged_adapter_config_sha256") != sha256_path(
            source / "adapter_config.json"
        ):
            raise RuntimeError("merged adapter config does not match its receipt")
        if merged.get("merged_data_manifest_sha256") != data_manifest_sha256:
            raise RuntimeError("merged adapter is not bound to the caller's data manifest")
        parents = merged.get("parents")
        if not isinstance(parents, list) or len(parents) < 2:
            raise RuntimeError("merged adapter receipt must record at least two parents")
        parent_gates = [
            verify_training_binding(
                str(parent["adapter_path"]),
                str(parent["adapter_model_sha256"]),
                str(parent["data_manifest_sha256"]),
                expected_training_phase,
                int(parent["training_task_count"]),
            )[1]
            for parent in parents
        ]
        return source, {
            "kind": "glm47-merged-adapter-gate",
            "status": "passed",
            "phase": expected_training_phase,
            "run_id": run_id,
            "data_manifest_sha256": data_manifest_sha256,
            "merged_receipt": merged,
            "parent_gates": parent_gates,
        }

    is_sft = expected_training_phase == "sft"
    expected_gate_kind = "glm47-aider-sft-training-gate" if is_sft else EXPECTED_TRAINING_GATE_KIND
    if is_sft:
        gate_path = Path("/runs", run_id, "sft_lora_r16", "sft_training_gate.json")
    else:
        gate_path = Path("/runs", run_id, "grpo_lora_r16", "grpo_training_gate.json")
    if not gate_path.is_file():
        raise FileNotFoundError(f"missing full-run training gate: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = {
        "kind": expected_gate_kind,
        "status": "passed",
        "phase": expected_training_phase,
        "run_id": run_id,
        "data_manifest_sha256": data_manifest_sha256,
        "training_task_count": expected_training_task_count,
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
    config_sha256 = sha256_path(source / "adapter_config.json")

    def _catalog_entry(model_sha: str, cfg_sha: str) -> dict[str, object] | None:
        return next(
            (
                item
                for item in checkpoints
                if isinstance(item, dict)
                and item.get("adapter_model_sha256") == model_sha
                and item.get("adapter_config_sha256") == cfg_sha
            ),
            None,
        )

    checkpoint = _catalog_entry(adapter_sha256, config_sha256)
    derived_receipt_path = source / "derived_receipt.json"
    if checkpoint is None and derived_receipt_path.is_file():
        # A derived adapter (e.g. a scaled interpolation toward base) is accepted
        # only when its receipt binds the exact bytes served to a parent that is
        # itself in the gate's checkpoint catalog.
        derived = json.loads(derived_receipt_path.read_text(encoding="utf-8"))
        if derived.get("kind") != "glm47-derived-scaled-adapter":
            raise RuntimeError("derived adapter receipt has an unknown kind")
        scale = derived.get("scale")
        if not isinstance(scale, (int, float)) or not 0.0 < float(scale) <= 1.0:
            raise RuntimeError("derived adapter scale must be in (0, 1]")
        if derived.get("derived_adapter_model_sha256") != adapter_sha256:
            raise RuntimeError("derived adapter bytes do not match their receipt")
        if derived.get("derived_adapter_config_sha256") != config_sha256:
            raise RuntimeError("derived adapter config does not match its receipt")
        parent = validate_adapter_path(str(derived.get("parent_adapter_path")))
        if parent.parts[2] != run_id:
            raise RuntimeError("derived adapter parent belongs to a different training run")
        parent_model_sha = str(derived.get("parent_adapter_model_sha256"))
        parent_config_sha = str(derived.get("parent_adapter_config_sha256"))
        if sha256_path(Path(parent) / "adapter_model.bin") != parent_model_sha:
            raise RuntimeError("derived adapter parent bytes no longer match their receipt")
        if sha256_path(Path(parent) / "adapter_config.json") != parent_config_sha:
            raise RuntimeError("derived adapter parent config no longer matches its receipt")
        checkpoint = _catalog_entry(parent_model_sha, parent_config_sha)
        if checkpoint is None:
            raise RuntimeError("derived adapter parent is absent from the checkpoint catalog")
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
    state = torch.load(
        source / "adapter_model.bin", map_location="cpu", weights_only=True, mmap=True
    )
    keys = list(state)
    layer_47_keys = [key for key in keys if ".layers.47." in key]
    if len(keys) != EXPECTED_SOURCE_TENSORS or len(layer_47_keys) != EXPECTED_LAYER_47_TENSORS:
        raise RuntimeError(
            "adapter tensor structure does not match the proven GLM-4.7 serving conversion"
        )
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
    expected_training_task_count: int = EXPECTED_TRAINING_TASK_COUNT,
) -> dict[str, object]:
    source, training_gate = verify_training_binding(
        adapter_path,
        expected_adapter_sha256,
        expected_data_manifest_sha256,
        expected_training_phase,
        expected_training_task_count,
    )
    destination = serving_adapter_path(adapter_path)
    payload = ensure_serving_adapter(
        source=source,
        destination=destination,
        training_gate=training_gate,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
    )
    if payload["preparation_status"] == "created":
        runs.commit()
    return {**payload, "training_gate": training_gate}


@app.function(image=image, cpu=4.0, memory=32_768, timeout=3600, volumes={"/runs": runs})
def derive_scaled_adapter(
    parent_adapter_path: str,
    expected_parent_sha256: str,
    expected_data_manifest_sha256: str,
    scale: float,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
    expected_training_task_count: int = EXPECTED_TRAINING_TASK_COUNT,
) -> dict[str, object]:
    """Create base + scale*delta as a gated derived adapter under the parent's run.

    Scaling every lora_B tensor by `scale` scales the low-rank weight delta exactly,
    which is the linear interpolation between the base model and the fine-tune.
    """

    import torch

    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be in (0, 1]")
    parent, training_gate = verify_training_binding(
        parent_adapter_path,
        expected_parent_sha256,
        expected_data_manifest_sha256,
        expected_training_phase,
        expected_training_task_count,
    )
    tag = f"scale{int(round(scale * 100)):02d}"
    destination = parent.parent.parent / f"{parent.parent.name}-{tag}" / "adapter"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite derived adapter: {destination}")
    state = torch.load(parent / "adapter_model.bin", map_location="cpu", weights_only=True)
    if len(state) != EXPECTED_SOURCE_TENSORS:
        raise RuntimeError("parent adapter tensor structure is not the proven GLM-4.7 layout")
    scaled_count = 0
    for name in state:
        if "lora_B" in name:
            state[name] = state[name].to(torch.float32).mul_(scale).to(state[name].dtype)
            scaled_count += 1
    if not scaled_count:
        raise RuntimeError("no lora_B tensors found; refusing to write an unscaled copy")
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(state, destination / "adapter_model.bin")
    shutil.copy2(parent / "adapter_config.json", destination / "adapter_config.json")
    receipt = {
        "kind": "glm47-derived-scaled-adapter",
        "scale": scale,
        "parent_adapter_path": str(parent),
        "parent_adapter_model_sha256": sha256_path(parent / "adapter_model.bin"),
        "parent_adapter_config_sha256": sha256_path(parent / "adapter_config.json"),
        "derived_adapter_path": str(destination),
        "derived_adapter_model_sha256": sha256_path(destination / "adapter_model.bin"),
        "derived_adapter_config_sha256": sha256_path(destination / "adapter_config.json"),
        "scaled_tensor_count": scaled_count,
        "tensor_count": len(state),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "derived_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    runs.commit()
    verify_training_binding(
        str(destination),
        receipt["derived_adapter_model_sha256"],
        expected_data_manifest_sha256,
        expected_training_phase,
        expected_training_task_count,
    )
    return receipt


@app.function(image=image, cpu=4.0, memory=32_768, timeout=3600, volumes={"/runs": runs})
def derive_merged_adapter(
    primary_adapter_path: str,
    primary_sha256: str,
    primary_data_manifest_sha256: str,
    primary_training_task_count: int,
    secondary_adapter_path: str,
    secondary_sha256: str,
    secondary_data_manifest_sha256: str,
    secondary_training_task_count: int,
    tag: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
) -> dict[str, object]:
    """Blend two gated rank-16 fine-tunes into their exact 50/50 linear merge.

    A LoRA delta is B @ A, so averaging A and B separately is not the average of
    the deltas. Concatenating along the rank axis is exact: with lora_alpha held
    at 32 while r doubles to 32, PEFT's alpha/r scaling halves, and the served
    delta becomes base + 0.5*delta_primary + 0.5*delta_secondary.
    """

    import torch

    primary, primary_gate = verify_training_binding(
        primary_adapter_path,
        primary_sha256,
        primary_data_manifest_sha256,
        expected_training_phase,
        primary_training_task_count,
    )
    secondary, secondary_gate = verify_training_binding(
        secondary_adapter_path,
        secondary_sha256,
        secondary_data_manifest_sha256,
        expected_training_phase,
        secondary_training_task_count,
    )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
        raise ValueError("tag must contain only letters, digits, dot, underscore, and hyphen")

    primary_config = json.loads((primary / "adapter_config.json").read_text(encoding="utf-8"))
    secondary_config = json.loads((secondary / "adapter_config.json").read_text(encoding="utf-8"))
    if primary_config != secondary_config:
        raise RuntimeError("refusing to merge adapters with differing LoRA configurations")
    rank = int(primary_config["r"])
    alpha = int(primary_config["lora_alpha"])

    destination = primary.parent.parent / f"{primary.parent.name}-{tag}" / "adapter"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite merged adapter: {destination}")

    first = torch.load(primary / "adapter_model.bin", map_location="cpu", weights_only=True)
    second = torch.load(secondary / "adapter_model.bin", map_location="cpu", weights_only=True)
    if set(first) != set(second):
        raise RuntimeError("merged parents do not share an identical tensor key set")
    if len(first) != EXPECTED_SOURCE_TENSORS:
        raise RuntimeError("parent adapters do not use the proven GLM-4.7 tensor layout")

    merged_state: dict[str, object] = {}
    concatenated = 0
    for name, tensor in first.items():
        other = second[name]
        if tensor.shape != other.shape:
            raise RuntimeError(f"shape mismatch between parents for {name}")
        # Shared MoE expert tensors carry a leading expert axis, so concatenate on
        # the rank axis relative to the end: lora_A is (..., r, in), lora_B is
        # (..., out, r). Concatenating on absolute axes would widen the expert
        # dimension instead, which the shared-outer-LoRA serving path rejects.
        if "lora_A" in name:
            axis = tensor.dim() - 2
        elif "lora_B" in name:
            axis = tensor.dim() - 1
        else:
            raise RuntimeError(f"unexpected non-LoRA tensor in adapter: {name}")
        if tensor.shape[axis] != rank:
            raise RuntimeError(
                f"{name} has {tensor.shape[axis]} on its rank axis, expected {rank}"
            )
        merged = torch.cat([tensor, other], dim=axis)
        expected_shape = list(tensor.shape)
        expected_shape[axis] = rank * 2
        if list(merged.shape) != expected_shape:
            raise RuntimeError(f"merged tensor {name} has an unexpected shape")
        merged_state[name] = merged
        concatenated += 1

    destination.mkdir(parents=True, exist_ok=False)
    torch.save(merged_state, destination / "adapter_model.bin")
    merged_config = {**primary_config, "r": rank * 2, "lora_alpha": alpha}
    (destination / "adapter_config.json").write_text(
        json.dumps(merged_config, indent=2) + "\n", encoding="utf-8"
    )
    parents = [
        {
            "role": "primary",
            "adapter_path": str(primary),
            "adapter_model_sha256": sha256_path(primary / "adapter_model.bin"),
            "data_manifest_sha256": primary_data_manifest_sha256,
            "training_task_count": primary_training_task_count,
            "training_run_id": str(primary_gate["run_id"]),
            "weight": 0.5,
        },
        {
            "role": "secondary",
            "adapter_path": str(secondary),
            "adapter_model_sha256": sha256_path(secondary / "adapter_model.bin"),
            "data_manifest_sha256": secondary_data_manifest_sha256,
            "training_task_count": secondary_training_task_count,
            "training_run_id": str(secondary_gate["run_id"]),
            "weight": 0.5,
        },
    ]
    merged_manifest_sha256 = hashlib.sha256(
        json.dumps(
            [parent["data_manifest_sha256"] for parent in parents], sort_keys=True
        ).encode()
    ).hexdigest()
    receipt = {
        "kind": "glm47-merged-adapter",
        "method": "rank-concatenation linear merge (exact 0.5/0.5 delta average)",
        "merged_adapter_path": str(destination),
        "merged_adapter_model_sha256": sha256_path(destination / "adapter_model.bin"),
        "merged_adapter_config_sha256": sha256_path(destination / "adapter_config.json"),
        "merged_data_manifest_sha256": merged_manifest_sha256,
        "merged_rank": rank * 2,
        "lora_alpha": alpha,
        "tensor_count": len(merged_state),
        "concatenated_tensor_count": concatenated,
        "parents": parents,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "merged_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    runs.commit()
    verify_training_binding(
        str(destination),
        receipt["merged_adapter_model_sha256"],
        merged_manifest_sha256,
        expected_training_phase,
        primary_training_task_count,
    )
    return receipt


def verify_serving_binding(
    serving_path: Path,
    source: Path,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    *,
    expected_source_tensors: int = EXPECTED_SOURCE_TENSORS,
    expected_layer_47_tensors: int = EXPECTED_LAYER_47_TENSORS,
    expected_serving_tensors: int = EXPECTED_SERVING_TENSORS,
    expected_training_run_id: str | None = None,
) -> dict[str, object]:
    validate_adapter_files(serving_path)
    receipt_path = serving_path / "conversion_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing serving conversion receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
        "kind": "glm47-serving-adapter-conversion",
        "source_adapter_path": str(source),
        "source_adapter_sha256": validate_sha256(
            expected_adapter_sha256, "expected_adapter_sha256"
        ),
        "source_adapter_config_sha256": sha256_path(source / "adapter_config.json"),
        "source_tensor_count": expected_source_tensors,
        "removed_layer_47_tensor_count": expected_layer_47_tensors,
        "serving_tensor_count": expected_serving_tensors,
        "serving_adapter_sha256": sha256_path(serving_path / "adapter_model.bin"),
        "serving_adapter_config_sha256": sha256_path(serving_path / "adapter_config.json"),
        "training_data_manifest_sha256": validate_sha256(
            expected_data_manifest_sha256, "expected_data_manifest_sha256"
        ),
        "training_gate_run_id": expected_training_run_id or source.parts[2],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("serving adapter is stale or not bound to the selected full-run adapter")
    return receipt


def ensure_serving_adapter(
    *,
    source: Path,
    destination: Path,
    training_gate: dict[str, object],
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_source_tensors: int = EXPECTED_SOURCE_TENSORS,
    expected_layer_47_tensors: int = EXPECTED_LAYER_47_TENSORS,
    expected_serving_tensors: int = EXPECTED_SERVING_TENSORS,
) -> dict[str, object]:
    """Atomically create or verify the serving adapter for one training checkpoint."""

    import torch

    validate_adapter_files(source)
    adapter_sha256 = validate_sha256(expected_adapter_sha256, "expected_adapter_sha256")
    data_manifest_sha256 = validate_sha256(
        expected_data_manifest_sha256, "expected_data_manifest_sha256"
    )
    if sha256_path(source / "adapter_model.bin") != adapter_sha256:
        raise RuntimeError("source adapter does not match the caller-bound SHA-256")
    if training_gate.get("data_manifest_sha256") != data_manifest_sha256:
        raise RuntimeError("training gate does not match the caller-bound data manifest")
    training_run_id = str(training_gate["run_id"])

    if destination.exists():
        receipt = verify_serving_binding(
            destination,
            source,
            expected_adapter_sha256,
            expected_data_manifest_sha256,
            expected_source_tensors=expected_source_tensors,
            expected_layer_47_tensors=expected_layer_47_tensors,
            expected_serving_tensors=expected_serving_tensors,
            expected_training_run_id=training_run_id,
        )
        return {
            "preparation_status": "reused",
            "adapter_path": str(source),
            "source_adapter_sha256": sha256_path(source / "adapter_model.bin"),
            "source_tensor_count": expected_source_tensors,
            "serving_tensor_count": expected_serving_tensors,
            "removed_tensor_count": expected_layer_47_tensors,
            "serving_adapter_path": str(destination),
            "serving_adapter_sha256": sha256_path(destination / "adapter_model.bin"),
            "serving_adapter_config_sha256": sha256_path(destination / "adapter_config.json"),
            "conversion_receipt_sha256": sha256_path(destination / "conversion_receipt.json"),
            "conversion_receipt": receipt,
        }

    temporary = Path(f"{destination}-preparing-{uuid.uuid4().hex[:8]}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        state = torch.load(
            source / "adapter_model.bin",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        layer_47_keys = [key for key in state if ".layers.47." in key]
        if len(state) != expected_source_tensors or len(layer_47_keys) != expected_layer_47_tensors:
            raise RuntimeError(
                "adapter tensor structure does not match the proven GLM-4.7 serving conversion"
            )
        filtered = {key: value for key, value in state.items() if ".layers.47." not in key}
        if len(filtered) != expected_serving_tensors or any(
            ".layers.47." in key for key in filtered
        ):
            raise RuntimeError(
                "serving adapter conversion did not produce the exact proven tensor domain"
            )
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
            "training_data_manifest_sha256": validate_sha256(
                data_manifest_sha256, "expected_data_manifest_sha256"
            ),
            "training_gate_run_id": training_run_id,
        }
        (temporary / "conversion_receipt.json").write_text(
            json.dumps(conversion_receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    verified_receipt = verify_serving_binding(
        destination,
        source,
        expected_adapter_sha256,
        expected_data_manifest_sha256,
        expected_source_tensors=expected_source_tensors,
        expected_layer_47_tensors=expected_layer_47_tensors,
        expected_serving_tensors=expected_serving_tensors,
        expected_training_run_id=training_run_id,
    )
    return {
        "preparation_status": "created",
        "adapter_path": str(source),
        "source_adapter_sha256": sha256_path(source / "adapter_model.bin"),
        "source_tensor_count": len(state),
        "serving_tensor_count": len(filtered),
        "removed_tensor_count": len(state) - len(filtered),
        "serving_adapter_path": str(destination),
        "serving_adapter_sha256": sha256_path(destination / "adapter_model.bin"),
        "serving_adapter_config_sha256": sha256_path(destination / "adapter_config.json"),
        "conversion_receipt_sha256": sha256_path(destination / "conversion_receipt.json"),
        "conversion_receipt": verified_receipt,
    }


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
        preparation = prepare_adapter.remote(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
        )
        print(json.dumps({"serving_preflight": preparation}, indent=2), flush=True)
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
        print(
            json.dumps(
                {"shard_function_call_ids": [call.object_id for call in calls]},
                indent=2,
            ),
            flush=True,
        )
        shard_receipts = [call.get() for call in calls]
        payload = merge_shards.remote(
            run_id=resolved_run_id,
            shard_receipts=shard_receipts,
        )
    else:
        preparation = prepare_adapter.remote(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
        )
        print(json.dumps({"serving_preflight": preparation}, indent=2), flush=True)
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


LORA_NAME = "glm-4.7-flash-grpo"


def _load_adapter(serving_path: Path, port: int = 8000) -> str:
    payload = json.dumps(
        {"lora_name": LORA_NAME, "lora_path": str(serving_path)}
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
    raise RuntimeError("SGLang does not expose a LoRA adapter loading endpoint")


def _lora_probe_completion(lora: bool, port: int = 8000) -> dict[str, object]:
    body: dict[str, object] = {
        "model": LORA_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Write a C++17 function `int answer()` that returns 42. Reply with only the code.",
            }
        ],
        "temperature": 0.0,
        "max_tokens": 48,
        "logprobs": True,
    }
    if lora:
        body["lora_path"] = LORA_NAME
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-eval"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    choice = payload["choices"][0]
    logprobs = [
        round(entry["logprob"], 6)
        for entry in (choice.get("logprobs") or {}).get("content") or []
    ]
    return {"content": choice["message"]["content"], "logprobs": logprobs}


def _verify_lora_activation(port: int = 8000) -> dict[str, object]:
    """Fail closed unless requests that select the adapter behave differently from base.

    A loaded adapter that is never selected by requests is indistinguishable from a
    healthy server; only a behavioral divergence under greedy decoding proves the
    benchmark traffic will actually exercise the trained weights.
    """
    with_lora = _lora_probe_completion(lora=True, port=port)
    without_lora = _lora_probe_completion(lora=False, port=port)
    diverged = (
        with_lora["content"] != without_lora["content"]
        or with_lora["logprobs"] != without_lora["logprobs"]
    )
    if not diverged:
        raise RuntimeError(
            "LoRA activation probe failed: greedy completions with and without "
            "lora_path are identical, so benchmark requests would measure the base model"
        )
    return {"status": "diverged", "with_lora": with_lora, "without_lora": without_lora}


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


def _apply_contract_overlay(destination: Path) -> dict[str, object]:
    """Patch the copied exercises with their stated interface contracts.

    Applied to the shard copy, never to the pinned clone, so the upstream tree in
    the image stays byte-identical to POLYGLOT_COMMIT. The overlay verifies the
    sha256 of every original instructions.md and raises if upstream text drifted.
    """

    sys.path.insert(0, "/opt/fixed26")
    import aider_fixed26_contract_overlay as overlay

    manifest = overlay.apply(destination)
    if int(manifest["tasks"]) != len(list(destination.iterdir())):
        raise RuntimeError("contract overlay did not cover every shard task")
    audit_manifest = overlay.audit(destination)
    if (
        int(audit_manifest["tasks"]) != len(list(destination.iterdir()))
        or int(audit_manifest["unexplained_deterministic_requirements"]) != 0
    ):
        raise RuntimeError("contract audit did not explain every deterministic requirement")
    manifest["prompt_test_audit"] = audit_manifest
    manifest["prompt_test_audit_sha256"] = audit_manifest["audit_sha256"]
    return manifest


def _create_cpp_shard(shard_index: int, run_id: str = "") -> tuple[Path, list[str], dict[str, object]]:
    if shard_index not in (0, 1):
        raise ValueError("shard_index must be 0 or 1")
    source = Path("/aider/tmp.benchmarks/polyglot-benchmark/cpp/exercises/practice")
    tasks = sorted(path for path in source.iterdir() if path.is_dir())
    if len(tasks) != 26:
        raise RuntimeError(f"fixed C++ benchmark task count mismatch: {len(tasks)} != 26")
    selected = tasks[shard_index * 13 : (shard_index + 1) * 13]
    # Scoped by run_id: Modal reuses warm containers across sequential attempts,
    # so a fixed path collides with the previous attempt's tree in the same
    # container. The exist_ok=False below still guards against reuse within a run.
    suffix = f"-{run_id}" if run_id else ""
    shard_root = Path(f"/tmp/polyglot-benchmark-shard-{shard_index}{suffix}")
    if shard_root.exists():
        shutil.rmtree(shard_root)
    destination = shard_root / "cpp/exercises/practice"
    destination.mkdir(parents=True, exist_ok=False)
    for task in selected:
        shutil.copytree(task, destination / task.name)
    overlay_manifest = _apply_contract_overlay(destination)
    return shard_root, [task.name for task in selected], overlay_manifest


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
        # Per-task outcomes, needed to take the union across independent samples.
        "passed_testcases_first": sorted(
            Path(payload["testcase"]).name
            for _, payload in rows
            if payload["tests_outcomes"][0]
        ),
        "passed_testcases_any": sorted(
            Path(payload["testcase"]).name
            for _, payload in rows
            if any(payload["tests_outcomes"])
        ),
        "pass_at_1": sum(bool(payload["tests_outcomes"][0]) for _, payload in rows),
        "multi_turn_with_error_feedback_at_2": sum(
            any(payload["tests_outcomes"]) for _, payload in rows
        ),
        "well_formed_tasks": sum(
            int(payload.get("num_malformed_responses", 0)) == 0 for _, payload in rows
        ),
        "malformed_responses": sum(
            int(payload.get("num_malformed_responses", 0)) for _, payload in rows
        ),
        "error_outputs": sum(int(payload.get("num_error_outputs", 0)) for _, payload in rows),
        "context_exhaustions": sum(
            int(payload.get("num_exhausted_context_windows", 0)) for _, payload in rows
        ),
        "test_timeouts": sum(int(payload.get("test_timeouts", 0)) for _, payload in rows),
        "prompt_tokens": sum(int(payload.get("prompt_tokens", 0)) for _, payload in rows),
        "completion_tokens": sum(int(payload.get("completion_tokens", 0)) for _, payload in rows),
        "result_sha256": {str(path.relative_to(output_dir)): sha256_path(path) for path, _ in rows},
    }


@app.function(
    image=image,
    gpu="H100:4",
    timeout=7200,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, max_delay=30.0),
    volumes={"/models": models, "/runs": runs, "/results": results},
)
def evaluate_shard(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str,
    run_id: str,
    shard_index: int,
    tries: int = 2,
    base_model: bool = False,
    expected_training_task_count: int = EXPECTED_TRAINING_TASK_COUNT,
) -> dict[str, object]:
    if tries not in (1, 2):
        raise ValueError("tries must be 1 (single-turn) or 2 (one repair round)")
    if base_model:
        if adapter_path or expected_adapter_sha256 or expected_data_manifest_sha256:
            raise ValueError("base-model evaluation must not receive adapter binding arguments")
        source = None
        serving_path = None
        training_gate = None
        conversion_receipt = None
    else:
        source, training_gate = verify_training_binding(
            adapter_path,
            expected_adapter_sha256,
            expected_data_manifest_sha256,
            expected_training_phase,
            expected_training_task_count,
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
        _model_settings_yaml(lora_name=None if base_model else LORA_NAME),
        encoding="utf-8",
    )
    shard_root, selected_tasks, overlay_manifest = _create_cpp_shard(shard_index, resolved_run_id)
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
    ]
    if not base_model:
        # A merged adapter has a wider rank than a single fine-tune, so read the
        # rank the server must support from the adapter that will be served rather
        # than from the image environment.
        served_rank = max(
            EVAL_LORA_RANK,
            int(
                json.loads(
                    (serving_path / "adapter_config.json").read_text(encoding="utf-8")
                )["r"]
            ),
        )
        server_command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(served_rank),
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
        )
    started = datetime.now(timezone.utc)
    log_path = Path(f"/tmp/sglang-grpo-shard-{shard_index}.log")
    print(f"[shard {shard_index}] starting SGLang", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            try:
                _wait_for_server(server)
            except Exception as exc:
                log.flush()
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                print(f"[shard {shard_index}] SGLang startup failure:\n{tail}", flush=True)
                raise RuntimeError(f"{exc}\nSGLang log tail:\n{tail}") from exc
            if base_model:
                adapter_load = {"mode": "base", "loaded_adapters": []}
                activation_probe = None
                print(f"[shard {shard_index}] base model healthy; starting benchmark", flush=True)
            else:
                print(f"[shard {shard_index}] SGLang healthy; loading adapter", flush=True)
                try:
                    adapter_load = _load_adapter(serving_path)
                except Exception:
                    log.flush()
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                    print(f"[shard {shard_index}] adapter load failure:\n{tail}", flush=True)
                    raise
                try:
                    activation_probe = _verify_lora_activation()
                except Exception:
                    log.flush()
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                    print(
                        f"[shard {shard_index}] activation probe failure:\n{tail}",
                        flush=True,
                    )
                    raise
                print(
                    f"[shard {shard_index}] adapter loaded and activation verified; "
                    "starting benchmark",
                    flush=True,
                )
            full = _benchmark(
                f"{resolved_run_id}-shard-{shard_index}",
                num_tests=None,
                tries=tries,
                threads=8,
                exercises_dir=str(shard_root),
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()

    validation = _validate_benchmark_results(full, expected_tasks=13, expected_tries=tries)
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
        "model_kind": "base" if base_model else "adapter",
        "model_revision": MODEL_REVISION,
        "adapter_path": None if source is None else str(source),
        "adapter_sha256": None if source is None else sha256_path(source / "adapter_model.bin"),
        "training_data_manifest_sha256": (
            None if base_model else expected_data_manifest_sha256
        ),
        "training_gate": training_gate,
        "serving_conversion_receipt": conversion_receipt,
        "aider_commit": AIDER_COMMIT,
        "polyglot_commit": POLYGLOT_COMMIT,
        "tries": tries,
        "eval_set_version": overlay_manifest["overlay_version"],
        "contract_overlay_sha256": overlay_manifest["overlay_sha256"],
        "contract_overlay": overlay_manifest,
        "prompt_test_audit_sha256": overlay_manifest["prompt_test_audit_sha256"],
        "prompt_test_audit": overlay_manifest["prompt_test_audit"],
        "gpu_requested": "H100:4",
        "lora_rank": None if base_model else EVAL_LORA_RANK,
        "temperature": 0.7,
        "top_p": 1.0,
        "thinking_disabled": DISABLE_THINKING,
        "threads": 8,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_command": server_command,
        "adapter_load": adapter_load,
        "lora_activation_probe": activation_probe,
        "full": full,
        "validation": validation,
    }
    (destination / "shard_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
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
    identity_fields = (
        "model_kind",
        "model_revision",
        "adapter_sha256",
        "training_data_manifest_sha256",
        "aider_commit",
        "polyglot_commit",
        "tries",
        "eval_set_version",
        "temperature",
        "top_p",
        "thinking_disabled",
    )
    for field in identity_fields:
        if len({json.dumps(receipt.get(field), sort_keys=True) for receipt in shard_receipts}) != 1:
            raise RuntimeError(f"parallel evaluation shard identity mismatch: {field}")
    testcases = [
        testcase for receipt in shard_receipts for testcase in receipt["validation"]["testcases"]
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
        "multi_turn_with_error_feedback_at_2",
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
    for key in ("passed_testcases_first", "passed_testcases_any"):
        validation[key] = sorted(
            {name for receipt in shard_receipts for name in receipt["validation"].get(key, [])}
        )
    receipt = {
        "status": "complete",
        "run_id": resolved_run_id,
        "benchmark": "aider-polyglot-cpp-grpo-eval",
        "parallel_topology": "2x TP4 on 8x H100 total",
        "model_kind": shard_receipts[0].get("model_kind", "adapter"),
        "model_revision": shard_receipts[0].get("model_revision", MODEL_REVISION),
        "adapter_path": shard_receipts[0]["adapter_path"],
        "adapter_sha256": shard_receipts[0]["adapter_sha256"],
        "training_data_manifest_sha256": shard_receipts[0]["training_data_manifest_sha256"],
        "aider_commit": AIDER_COMMIT,
        "polyglot_commit": POLYGLOT_COMMIT,
        "temperature": 0.7,
        "top_p": 1.0,
        "thinking_disabled": shard_receipts[0].get("thinking_disabled"),
        "tries": shard_receipts[0].get("tries", 2),
        "eval_set_version": shard_receipts[0].get("eval_set_version"),
        "contract_overlay_sha256": shard_receipts[0].get("contract_overlay_sha256"),
        "prompt_test_audit_sha256": shard_receipts[0].get("prompt_test_audit_sha256"),
        "contract_overlay_shard_sha256": {
            str(receipt["shard_index"]): receipt["contract_overlay_sha256"]
            for receipt in shard_receipts
        },
        "prompt_test_audit_shard_sha256": {
            str(receipt["shard_index"]): receipt["prompt_test_audit_sha256"]
            for receipt in shard_receipts
        },
        "lora_activation_verified": (
            None
            if shard_receipts[0].get("model_kind") == "base"
            else all(
                receipt.get("lora_activation_probe", {}).get("status") == "diverged"
                for receipt in shard_receipts
            )
        ),
        "validation": validation,
        "shards": shard_receipts,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    results.commit()
    return receipt


@app.function(image=image, cpu=1.0, memory=2_048, volumes={"/results": results})
def merge_shards(run_id: str, shard_receipts: list[dict[str, object]]) -> dict[str, object]:
    return _merge_shard_receipts(run_id, shard_receipts)


@app.function(image=image, cpu=2.0, memory=4_096, timeout=1800)
def verify_contract_overlay() -> dict[str, object]:
    """CPU-only preflight: apply the overlay to both shards inside the real image.

    Catches a drifted upstream clone or a missing overlay file before any GPU is
    provisioned.
    """

    out = []
    for shard_index in range(2):
        shard_root, tasks, manifest = _create_cpp_shard(shard_index, 'preflight')
        sample = (
            Path(shard_root) / "cpp/exercises/practice" / tasks[0] / ".docs/instructions.md"
        ).read_text(encoding="utf-8")
        out.append({
            "shard_index": shard_index,
            "tasks": tasks,
            "overlay_sha256": manifest["overlay_sha256"],
            "overlay_version": manifest["overlay_version"],
            "prompt_test_audit_sha256": manifest["prompt_test_audit_sha256"],
            "unexplained_deterministic_requirements": manifest[
                "prompt_test_audit"
            ]["unexplained_deterministic_requirements"],
            "marker_present": "## C++ interface contract" in sample,
        })
    covered = sorted(task for shard in out for task in shard["tasks"])
    if len(covered) != 26 or len(set(covered)) != 26:
        raise RuntimeError("prompt/test preflight did not cover exactly 26 unique tasks")
    return {
        "status": "passed",
        "tasks": len(covered),
        "unique_tasks": len(set(covered)),
        "unexplained_deterministic_requirements": sum(
            int(shard["unexplained_deterministic_requirements"]) for shard in out
        ),
        "shards": out,
        "polyglot_commit": POLYGLOT_COMMIT,
    }


def _extract_training_file_listings(
    content: str,
    expected_names: set[str],
) -> dict[str, str]:
    """Extract exact whole-edit targets from a validated SFT assistant message."""

    lines = content.splitlines()
    files: dict[str, str] = {}
    index = 0
    while index < len(lines):
        name = lines[index]
        if name not in expected_names:
            index += 1
            continue
        if name in files or index + 1 >= len(lines) or lines[index + 1] != "```":
            raise RuntimeError(f"malformed or duplicate file listing for {name}")
        closing = index + 2
        while closing < len(lines) and lines[closing] != "```":
            closing += 1
        if closing == len(lines):
            raise RuntimeError(f"unterminated file listing for {name}")
        files[name] = "\n".join(lines[index + 2 : closing]) + "\n"
        index = closing + 1
    if set(files) != expected_names:
        raise RuntimeError(
            f"file listing mismatch: got={sorted(files)}, expected={sorted(expected_names)}"
        )
    return files


@app.function(
    image=image,
    cpu=8.0,
    memory=16_384,
    timeout=3_600,
    volumes={"/training-assets": training_assets, "/results": results},
)
def revalidate_fixed26_direct_dataset_linux(
    receipt_name: str = "fixed26-direct-v1-linux-gcc13",
) -> dict[str, object]:
    """Replay the 26 direct SFT targets under the Linux fixed-26 test harness."""

    import tempfile

    expected_dataset_sha = (
        "cc85ca4efff202b0ce687e8d712aa98b38df0f2458eec1401af3f666e3641c32"
    )
    dataset = Path(
        "/training-assets/fixed26-training-arms-v1/"
        "fixed26-direct-v1/sft/train.jsonl"
    )
    if not dataset.is_file() or sha256_path(dataset) != expected_dataset_sha:
        raise RuntimeError("direct SFT dataset is missing or differs from the training-bound SHA")

    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    direct = [
        row
        for row in rows
        if row.get("metadata", {}).get("fixed26_training_role") == "direct"
    ]
    if len(rows) != 31 or len(direct) != 26:
        raise RuntimeError(f"unexpected dataset composition: rows={len(rows)}, direct={len(direct)}")

    source_root = Path(
        "/aider/tmp.benchmarks/polyglot-benchmark/cpp/exercises/practice"
    )
    destination = Path("/results/dataset-revalidation") / receipt_name
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite revalidation receipt: {destination}")
    destination.mkdir(parents=True)

    task_receipts: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="fixed26-direct-linux-") as temporary:
        temporary_root = Path(temporary)
        for row in sorted(direct, key=lambda item: str(item["label"])):
            task = str(row["label"])
            stem = task.replace("-", "_")
            expected_names = {f"{stem}.h", f"{stem}.cpp"}
            messages = row.get("messages")
            if (
                not isinstance(messages, list)
                or not messages
                or messages[-1].get("role") != "assistant"
            ):
                raise RuntimeError(f"{task}: missing final assistant target")
            files = _extract_training_file_listings(
                str(messages[-1].get("content", "")),
                expected_names,
            )

            exercise = temporary_root / task
            shutil.copytree(source_root / task, exercise)
            file_hashes = {}
            for name, content in files.items():
                path = exercise / name
                path.write_text(content, encoding="utf-8")
                file_hashes[name] = sha256_path(path)

            build = exercise / "build"
            commands = [
                [
                    "cmake",
                    "-S",
                    str(exercise),
                    "-B",
                    str(build),
                    "-DEXERCISM_RUN_ALL_TESTS=ON",
                    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                ],
                ["cmake", "--build", str(build), "--target", task, "--parallel", "4"],
                [str(build / task)],
            ]
            log_parts = []
            exits = []
            for command in commands:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=900,
                    check=False,
                )
                exits.append(completed.returncode)
                log_parts.extend(
                    [
                        f"$ {' '.join(command)}",
                        completed.stdout,
                        completed.stderr,
                        f"exit={completed.returncode}",
                    ]
                )
                if completed.returncode != 0:
                    break
            task_log = destination / f"{task}.log"
            task_log.write_text("\n".join(log_parts) + "\n", encoding="utf-8")
            task_receipts.append(
                {
                    "task": task,
                    "passed": exits == [0, 0, 0],
                    "stage_exit_statuses": exits,
                    "candidate_files_sha256": dict(sorted(file_hashes.items())),
                    "log_sha256": sha256_path(task_log),
                }
            )

    passed = sum(bool(item["passed"]) for item in task_receipts)
    receipt = {
        "schema_version": 1,
        "kind": "fixed26-direct-sft-linux-revalidation",
        "status": "passed" if passed == 26 else "failed",
        "dataset_sha256": expected_dataset_sha,
        "dataset_rows": len(rows),
        "direct_rows": len(direct),
        "passed": passed,
        "failed_tasks": [
            item["task"] for item in task_receipts if not item["passed"]
        ],
        "platform": {
            "polyglot_commit": POLYGLOT_COMMIT,
            "compiler": subprocess.run(
                ["g++", "--version"], text=True, capture_output=True, check=True
            ).stdout.splitlines()[0],
            "cmake": subprocess.run(
                ["cmake", "--version"], text=True, capture_output=True, check=True
            ).stdout.splitlines()[0],
        },
        "tasks": task_receipts,
    }
    (destination / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results.commit()
    if passed != 26:
        raise RuntimeError(json.dumps(receipt, sort_keys=True))
    return receipt


@app.local_entrypoint()
def preflight() -> None:
    print(json.dumps(verify_contract_overlay.remote(), indent=2))


@app.local_entrypoint()
def revalidate_direct_dataset(
    receipt_name: str = "fixed26-direct-v1-linux-gcc13",
) -> None:
    print(
        json.dumps(
            revalidate_fixed26_direct_dataset_linux.remote(receipt_name),
            indent=2,
        )
    )


def _run_pass_at_k(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
    run_id: str = "",
    samples: int = 8,
    tries: int = 1,
    first_index: int = 1,
    expected_training_task_count: int = EXPECTED_TRAINING_TASK_COUNT,
) -> None:
    if not 1 <= samples <= 16:
        raise ValueError("samples must be between 1 and 16")
    if tries not in (1, 2):
        raise ValueError("tries must be 1 (single-turn) or 2 (one repair round)")
    base = validate_run_id(run_id)
    preparation = prepare_adapter.remote(
        adapter_path=adapter_path,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_training_phase=expected_training_phase,
        expected_training_task_count=expected_training_task_count,
    )
    print(json.dumps({"serving_preflight": preparation}, indent=2), flush=True)
    attempts = []
    for index in range(first_index, first_index + samples):
        attempt_id = validate_run_id(f"{base}-a{index}")
        print(f"=== attempt {index}/{samples}: {attempt_id} (tries={tries})", flush=True)
        calls = [
            evaluate_shard.spawn(
                adapter_path=adapter_path,
                expected_adapter_sha256=expected_adapter_sha256,
                expected_data_manifest_sha256=expected_data_manifest_sha256,
                expected_training_phase=expected_training_phase,
                run_id=attempt_id,
                shard_index=shard,
                tries=tries,
                expected_training_task_count=expected_training_task_count,
            )
            for shard in range(2)
        ]
        shard_receipts = [call.get() for call in calls]
        receipt = merge_shards.remote(run_id=attempt_id, shard_receipts=shard_receipts)
        attempts.append({
            "attempt": index,
            "run_id": attempt_id,
            "pass_at_1": receipt["validation"]["pass_at_1"],
            "cumulative_at_tries": receipt["validation"]["multi_turn_with_error_feedback_at_2"],
            "passed_tasks": receipt["validation"]["passed_testcases_any"],
        })
        print(json.dumps(attempts[-1], indent=2), flush=True)

    union = sorted({task for a in attempts for task in a["passed_tasks"]})
    summary = {
        "base_run_id": base,
        "samples": samples,
        "tries": tries,
        "eval_set_version": OVERLAY_EXPECTED_VERSION,
        "per_attempt_pass_at_1": [a["pass_at_1"] for a in attempts],
        "pass_at_k": len(union),
        "pass_at_k_tasks": union,
        "attempts": attempts,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def pass_at_k(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str = EXPECTED_TRAINING_PHASE,
    run_id: str = "",
    samples: int = 8,
    tries: int = 1,
    first_index: int = 1,
) -> None:
    """Run independent evaluations while the local Modal client stays connected."""

    _run_pass_at_k(
        adapter_path=adapter_path,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_training_phase=expected_training_phase,
        run_id=run_id,
        samples=samples,
        tries=tries,
        first_index=first_index,
    )


def _collect_pass_at_k_receipts(base_run_id: str, samples: int = 8) -> dict[str, object]:
    base = validate_run_id(base_run_id)
    attempts = []
    for index in range(1, samples + 1):
        run_id = validate_run_id(f"{base}-a{index}")
        receipt_path = Path("/results/runs") / run_id / "run_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"missing completed pass@k receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validation = receipt["validation"]
        attempts.append(
            {
                "attempt": index,
                "run_id": run_id,
                "pass_at_1": validation["pass_at_1"],
                "cumulative_at_tries": validation[
                    "multi_turn_with_error_feedback_at_2"
                ],
                "passed_tasks": validation["passed_testcases_any"],
            }
        )
    union = sorted({task for attempt in attempts for task in attempt["passed_tasks"]})
    return {
        "base_run_id": base,
        "samples": samples,
        "per_attempt_pass_at_1": [attempt["pass_at_1"] for attempt in attempts],
        "per_attempt_cumulative": [
            attempt["cumulative_at_tries"] for attempt in attempts
        ],
        "pass_at_k": len(union),
        "pass_at_k_tasks": union,
        "attempts": attempts,
    }


@app.function(
    cpu=1.0,
    memory=2_048,
    timeout=86_400,
    volumes={"/results": results},
)
def run_pass_at_k_suite(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str,
    suite_run_id: str,
    single_turn_run_id: str,
    two_turn_run_id: str,
    single_turn_first_index: int = 1,
    single_turn_samples_to_run: int = 8,
    two_turn_samples: int = 8,
    expected_training_task_count: int = EXPECTED_TRAINING_TASK_COUNT,
) -> dict[str, object]:
    """Run both pass@k arms entirely server-side and persist a suite receipt."""

    resolved_suite_id = validate_run_id(suite_run_id)
    destination = Path("/results/runs") / resolved_suite_id
    if destination.exists():
        raise FileExistsError(f"refusing to reuse suite result path: {destination}")

    single_turn_total = single_turn_first_index + single_turn_samples_to_run - 1
    if single_turn_samples_to_run:
        _run_pass_at_k(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
            run_id=single_turn_run_id,
            samples=single_turn_samples_to_run,
            tries=1,
            first_index=single_turn_first_index,
            expected_training_task_count=expected_training_task_count,
        )
    results.reload()
    single_turn = _collect_pass_at_k_receipts(single_turn_run_id, samples=single_turn_total)

    if two_turn_samples:
        _run_pass_at_k(
            adapter_path=adapter_path,
            expected_adapter_sha256=expected_adapter_sha256,
            expected_data_manifest_sha256=expected_data_manifest_sha256,
            expected_training_phase=expected_training_phase,
            run_id=two_turn_run_id,
            samples=two_turn_samples,
            tries=2,
            first_index=1,
            expected_training_task_count=expected_training_task_count,
        )
        results.reload()
        two_turn = _collect_pass_at_k_receipts(two_turn_run_id, samples=two_turn_samples)
    else:
        two_turn = None

    payload = {
        "schema_version": 1,
        "kind": "fixed26-pass-at-8-suite",
        "status": "complete",
        "suite_run_id": resolved_suite_id,
        "adapter_path": str(validate_adapter_path(adapter_path)),
        "adapter_sha256": validate_sha256(
            expected_adapter_sha256, "expected_adapter_sha256"
        ),
        "training_data_manifest_sha256": validate_sha256(
            expected_data_manifest_sha256, "expected_data_manifest_sha256"
        ),
        "expected_training_phase": expected_training_phase,
        "eval_set_version": OVERLAY_EXPECTED_VERSION,
        "single_turn": single_turn,
        "two_turn_with_error_feedback": two_turn,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "suite_receipt.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results.commit()
    return payload


@app.local_entrypoint()
def launch_pass_at_k_suite(
    adapter_path: str,
    expected_adapter_sha256: str,
    expected_data_manifest_sha256: str,
    expected_training_phase: str,
    suite_run_id: str,
    single_turn_run_id: str,
    two_turn_run_id: str,
    single_turn_first_index: int = 1,
    single_turn_samples_to_run: int = 8,
) -> None:
    """Detach one durable server-side controller and print its function-call ID."""

    call = run_pass_at_k_suite.spawn(
        adapter_path=adapter_path,
        expected_adapter_sha256=expected_adapter_sha256,
        expected_data_manifest_sha256=expected_data_manifest_sha256,
        expected_training_phase=expected_training_phase,
        suite_run_id=suite_run_id,
        single_turn_run_id=single_turn_run_id,
        two_turn_run_id=two_turn_run_id,
        single_turn_first_index=single_turn_first_index,
        single_turn_samples_to_run=single_turn_samples_to_run,
    )
    print(
        json.dumps(
            {
                "suite_run_id": validate_run_id(suite_run_id),
                "function_call_id": call.object_id,
            },
            indent=2,
        ),
        flush=True,
    )


PLAN_ENTRY_KEYS = {
    "checkpoint_label",
    "adapter_path",
    "expected_adapter_sha256",
    "expected_data_manifest_sha256",
    "expected_training_phase",
    "expected_training_task_count",
    "suite_run_id",
    "single_turn_run_id",
    "two_turn_run_id",
    "single_turn_samples",
    "two_turn_samples",
}


@app.function(
    cpu=1.0,
    memory=2_048,
    timeout=86_400,
    volumes={"/results": results},
)
def run_fixed26_eval_plan(plan: list[dict], index: int = 0) -> dict[str, object]:
    """Run one plan entry's suite, then spawn the next entry as a fresh call.

    Self-chaining keeps every call far below the function timeout and keeps the
    whole multi-checkpoint sequence alive with no local client attached.
    """

    for position, entry in enumerate(plan):
        missing = PLAN_ENTRY_KEYS - set(entry)
        if missing:
            raise ValueError(f"plan entry {position} missing keys: {sorted(missing)}")
    entry = plan[index]
    print(
        f"=== plan entry {index + 1}/{len(plan)}: {entry['checkpoint_label']}",
        flush=True,
    )
    payload = run_pass_at_k_suite.local(
        adapter_path=entry["adapter_path"],
        expected_adapter_sha256=entry["expected_adapter_sha256"],
        expected_data_manifest_sha256=entry["expected_data_manifest_sha256"],
        expected_training_phase=entry["expected_training_phase"],
        suite_run_id=entry["suite_run_id"],
        single_turn_run_id=entry["single_turn_run_id"],
        two_turn_run_id=entry["two_turn_run_id"],
        single_turn_samples_to_run=entry["single_turn_samples"],
        two_turn_samples=entry["two_turn_samples"],
        expected_training_task_count=entry["expected_training_task_count"],
    )
    if index + 1 < len(plan):
        call = run_fixed26_eval_plan.spawn(plan=plan, index=index + 1)
        print(
            json.dumps(
                {"next_entry": plan[index + 1]["checkpoint_label"], "function_call_id": call.object_id},
                indent=2,
            ),
            flush=True,
        )
    return payload


@app.local_entrypoint()
def launch_fixed26_eval_plan(plan_json: str) -> None:
    """Detach the self-chaining plan runner and print the first function-call ID."""

    plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a non-empty JSON list")
    call = run_fixed26_eval_plan.spawn(plan=plan, index=0)
    print(
        json.dumps(
            {
                "entries": [entry["checkpoint_label"] for entry in plan],
                "first_function_call_id": call.object_id,
            },
            indent=2,
        ),
        flush=True,
    )


@app.local_entrypoint()
def base_pass_at_k(
    run_id: str,
    samples: int = 8,
    tries: int = 1,
    first_index: int = 1,
) -> None:
    """Run independent fixed-26 evaluations without loading an adapter."""

    if not 1 <= samples <= 16:
        raise ValueError("samples must be between 1 and 16")
    if tries not in (1, 2):
        raise ValueError("tries must be 1 (single-turn) or 2 (one repair round)")
    base = validate_run_id(run_id)
    print(
        json.dumps(
            {
                "base_model_preflight": {
                    "model_kind": "base",
                    "model_revision": MODEL_REVISION,
                    "model_path": MODEL_PATH,
                    "adapter_loaded": False,
                    "eval_set_version": OVERLAY_EXPECTED_VERSION,
                    "thinking_disabled": DISABLE_THINKING,
                }
            },
            indent=2,
        ),
        flush=True,
    )
    attempts = []
    for index in range(first_index, first_index + samples):
        attempt_id = validate_run_id(f"{base}-a{index}")
        print(f"=== attempt {index}/{samples}: {attempt_id} (tries={tries})", flush=True)
        calls = [
            evaluate_shard.spawn(
                adapter_path="",
                expected_adapter_sha256="",
                expected_data_manifest_sha256="",
                expected_training_phase="",
                run_id=attempt_id,
                shard_index=shard,
                tries=tries,
                base_model=True,
            )
            for shard in range(2)
        ]
        shard_receipts = [call.get() for call in calls]
        receipt = merge_shards.remote(run_id=attempt_id, shard_receipts=shard_receipts)
        attempts.append({
            "attempt": index,
            "run_id": attempt_id,
            "pass_at_1": receipt["validation"]["pass_at_1"],
            "cumulative_at_tries": receipt["validation"]["multi_turn_with_error_feedback_at_2"],
            "passed_tasks": receipt["validation"]["passed_testcases_any"],
        })
        print(json.dumps(attempts[-1], indent=2), flush=True)

    union = sorted({task for attempt in attempts for task in attempt["passed_tasks"]})
    print(json.dumps({
        "base_run_id": base,
        "model_kind": "base",
        "model_revision": MODEL_REVISION,
        "adapter_loaded": False,
        "samples": samples,
        "tries": tries,
        "eval_set_version": OVERLAY_EXPECTED_VERSION,
        "per_attempt_pass_at_1": [attempt["pass_at_1"] for attempt in attempts],
        "pass_at_k": len(union),
        "pass_at_k_tasks": union,
        "attempts": attempts,
    }, indent=2))


@app.function(image=image, cpu=1.0, memory=2_048, volumes={"/results": results})
def merge_saved_shards(run_id: str) -> dict[str, object]:
    resolved_run_id = validate_run_id(run_id)
    shard_receipts = []
    for shard_index in range(2):
        receipt_path = (
            Path("/results/runs") / f"{resolved_run_id}-shard-{shard_index}" / "shard_receipt.json"
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
        _model_settings_yaml(lora_name=LORA_NAME),
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
            activation_probe = _verify_lora_activation()
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
        "lora_activation_probe": activation_probe,
        "smoke": smoke,
        "smoke_validation": smoke_validation,
        "full": full,
        "full_validation": full_validation,
    }
    (destination / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    results.commit()
    return receipt
