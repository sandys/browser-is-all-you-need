"""Registry of custom-kernel R&D targets (metadata only; torch-free).

This module deliberately imports nothing heavy: it describes *what* kernels exist,
which SkyRL/Megatron op each one replaces, how it is injected, and its maturity.
``w8-biayn kernels list`` and CLI validation use this on a plain CPU host. The actual
Triton implementations live in the ``triton_*`` siblings and import ``torch`` lazily.

Tiers (see the approved plan):
- ``A`` — SkyRL non-distributed numerics. Warm-up that proves the harness + gating.
  These are NOT the headline result (they cannot move the generation-bound step).
- ``B`` — Megatron/TE local ops in the single-device lab. The actual R&D target.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    name: str
    tier: str  # "A" or "B"
    target: str  # upstream function/module the kernel replaces
    seam: str  # how the replacement is injected
    status: str  # "reference-only" | "planned" | "validated"


KERNELS: tuple[KernelSpec, ...] = (
    KernelSpec(
        "logprob",
        "A",
        "skyrl_train.utils.torch_utils.logprobs_from_logits_v2",
        "module monkeypatch (+ bound name in workers.model_wrapper)",
        "reference-only",
    ),
    KernelSpec(
        "entropy",
        "A",
        "skyrl_train.utils.torch_utils.chunked_entropy_from_logits",
        "module monkeypatch + instance attr chunked_entropy_from_logits_fn",
        "reference-only",
    ),
    KernelSpec(
        "ppo",
        "A",
        "skyrl_train.utils.ppo_utils.ppo_policy_loss / safe_exp_delta",
        "PolicyLossRegistry.register(PolicyLossType.REGULAR, fn)",
        "reference-only",
    ),
    KernelSpec(
        "mla",
        "B",
        "Megatron MLA attention (fused backend; FlashAttention has no MLA path)",
        "Megatron-Bridge provider transformer-layer ModuleSpec (single-device lab)",
        "planned",
    ),
    KernelSpec(
        "moe-gemm",
        "B",
        "Megatron MoE grouped GEMM / experts module",
        "Megatron-Bridge provider experts ModuleSpec (single-device lab, EP=1)",
        "fwd validated (parity + 3.4x on A100); bwd + integration pending",
    ),
)

# Tier-A kernels safe to activate inside the full Harbor R3 trainer (correctness gate).
TIER_A = tuple(k.name for k in KERNELS if k.tier == "A")


def kernel_names() -> tuple[str, ...]:
    return tuple(k.name for k in KERNELS)


def get_kernel(name: str) -> KernelSpec:
    for spec in KERNELS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown kernel {name!r}; known kernels: {', '.join(kernel_names())}")
