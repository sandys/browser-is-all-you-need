"""Tier-B integration discovery: how to interpose the Triton grouped-GEMM into the
Megatron MoE experts on the tiny model (single GPU). Introspection-first — prints the
MoE layer / experts module structure + the layer-spec submodules so we can write a
custom experts ModuleSpec that calls `grouped_gemm`. Build recipe is the validated
identity-swap one (gradient_accumulation_fusion=False, provide_distributed_model).
"""

from __future__ import annotations

import inspect
import os
import traceback

import torch


def show(t: str) -> None:
    print(f"\n===== {t} =====", flush=True)


os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
torch.cuda.set_device(0)
import torch.distributed as dist

dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state as mpu

mpu.initialize_model_parallel(1, 1, expert_model_parallel_size=1, expert_tensor_parallel_size=None, context_parallel_size=1)

MODEL = os.environ.get("W8_TINY_MODEL", "eatang/qwen3-moe-tiny-random")
show(f"build {MODEL}")
provider = bridge = None
try:
    from megatron.bridge import AutoBridge
    from megatron.core.distributed import DistributedDataParallelConfig

    bridge = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    for a, v in (("tensor_model_parallel_size", 1), ("pipeline_model_parallel_size", 1),
                 ("expert_model_parallel_size", 1), ("context_parallel_size", 1),
                 ("gradient_accumulation_fusion", False)):
        if hasattr(provider, a):
            setattr(provider, a, v)
    provider.finalize()
    model = provider.provide_distributed_model(ddp_config=DistributedDataParallelConfig(), wrap_with_ddp=False, bf16=True)
    model = model[0] if isinstance(model, (list, tuple)) else model
    print("model built:", type(model).__name__)
except Exception:
    traceback.print_exc()
    raise SystemExit("build failed")

show("locate MoE layer + experts module")
moe_mods = []
for name, mod in model.named_modules():
    tn = type(mod).__name__
    if any(k in tn.lower() for k in ("moe", "expert", "groupedmlp", "sequentialmlp")):
        moe_mods.append((name, tn))
for name, tn in moe_mods[:30]:
    print(f"  {name}: {tn}")

show("experts module detail")
experts = None
for name, mod in model.named_modules():
    if type(mod).__name__ in ("GroupedMLP", "TEGroupedMLP", "SequentialMLP"):
        experts = mod
        print("experts module:", name, "->", type(mod).__module__ + "." + type(mod).__name__)
        print("  forward sig:", str(inspect.signature(mod.forward)))
        params = [(pn, tuple(p.shape)) for pn, p in mod.named_parameters()]
        print("  params:", params[:8])
        break
if experts is None:
    print("no GroupedMLP/TEGroupedMLP/SequentialMLP found; full module type list:")
    print(sorted({type(m).__name__ for _, m in model.named_modules()}))

show("layer spec submodules")
try:
    spec = provider.transformer_layer_spec
    built = spec(provider) if callable(spec) else spec
    print("layer spec type:", type(built).__name__)
    subs = getattr(built, "submodules", None)
    if subs is not None:
        mlp = getattr(subs, "mlp", None)
        print("  submodules.mlp:", mlp)
        mlp_subs = getattr(mlp, "submodules", None) if mlp is not None else None
        if mlp_subs is not None:
            print("  mlp.submodules:", mlp_subs)
            print("  mlp.submodules.experts:", getattr(mlp_subs, "experts", "<none>"))
except Exception:
    traceback.print_exc()

show("DONE")
