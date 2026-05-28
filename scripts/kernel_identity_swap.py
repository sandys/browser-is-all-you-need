"""Phase-0 proof: interpose a no-op pass-through `transformer_layer_spec` on the
Megatron-Bridge provider and confirm the tiny MoE model builds + forwards identically
(single GPU, TP=PP=CP=EP=1). Diagnostic-heavy so one run teaches us the next fix.
"""

from __future__ import annotations

import inspect
import os
import traceback

import torch


def show(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


SEED = 1234
MODEL = os.environ.get("W8_TINY_MODEL", "eatang/qwen3-moe-tiny-random")

show("single-process distributed + parallel state")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29577")
torch.cuda.set_device(0)
import torch.distributed as dist

dist.init_process_group(backend="nccl", world_size=1, rank=0)
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel

mpu.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    expert_tensor_parallel_size=None,
    context_parallel_size=1,
)
print("model-parallel initialized; world_size", dist.get_world_size())


def build(interpose: bool):
    from megatron.bridge import AutoBridge
    from megatron.core.distributed import DistributedDataParallelConfig

    tensor_parallel.model_parallel_cuda_manual_seed(SEED)
    torch.manual_seed(SEED)
    bridge = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
    provider = bridge.to_megatron_provider()
    for attr, val in (
        ("tensor_model_parallel_size", 1),
        ("pipeline_model_parallel_size", 1),
        ("expert_model_parallel_size", 1),
        ("context_parallel_size", 1),
        # APEX fused grad-accum extension isn't built here; not needed for eval.
        ("gradient_accumulation_fusion", False),
    ):
        if hasattr(provider, attr):
            setattr(provider, attr, val)
    if interpose:
        orig = provider.transformer_layer_spec

        def passthrough(*a, **k):
            return orig(*a, **k)

        provider.transformer_layer_spec = passthrough
        print("  interposed pass-through transformer_layer_spec over", orig)
    provider.finalize()
    model = provider.provide_distributed_model(
        ddp_config=DistributedDataParallelConfig(), wrap_with_ddp=False, bf16=True
    )
    if isinstance(model, (list, tuple)):
        model = model[0]
    model.eval()
    return model


def forward(model, input_ids, position_ids):
    with torch.no_grad():
        try:
            return model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        except TypeError:
            return model(input_ids, position_ids, None)


show(f"build baseline model for {MODEL}")
try:
    m1 = build(interpose=False)
    print("model type:", type(m1).__module__ + "." + type(m1).__name__)
    cfg = getattr(m1, "config", None)
    vocab = getattr(cfg, "vocab_size", None) or getattr(cfg, "padded_vocab_size", None) or 1024
    print("vocab_size:", vocab, "| forward sig:", str(inspect.signature(m1.forward)))
except Exception:
    traceback.print_exc()
    raise SystemExit("baseline build failed")

show("forward baseline")
try:
    B, S = 1, 16
    torch.manual_seed(0)
    input_ids = torch.randint(0, int(vocab), (B, S), device="cuda")
    position_ids = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S)
    out1 = forward(m1, input_ids, position_ids)
    out1 = out1[0] if isinstance(out1, (list, tuple)) else out1
    print("out1 type/shape:", type(out1).__name__, tuple(out1.shape), out1.dtype)
except Exception:
    traceback.print_exc()
    raise SystemExit("baseline forward failed")

show("build with pass-through spec + forward + compare")
try:
    del m1
    torch.cuda.empty_cache()
    m2 = build(interpose=True)
    out2 = forward(m2, input_ids, position_ids)
    out2 = out2[0] if isinstance(out2, (list, tuple)) else out2
    diff = (out1.float() - out2.float()).abs().max().item()
    print("max_abs_diff:", diff)
    print("bit_identical:", bool(torch.equal(out1, out2)))
    print("allclose(atol=1e-3):", bool(torch.allclose(out1.float(), out2.float(), atol=1e-3, rtol=0)))
    print("IDENTITY_SWAP_OK" if diff < 1e-2 else "IDENTITY_SWAP_DIFF_TOO_LARGE")
except Exception:
    traceback.print_exc()
    raise SystemExit("pass-through build/forward failed")

show("DONE")
