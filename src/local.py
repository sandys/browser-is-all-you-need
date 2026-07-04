"""Megatron local-layer spec shim for the Moonlight SLIME example.

Megatron validates ``--attention-backend local`` by requiring ``--spec`` to
start with the literal module name ``local``.  The SLIME runner imports this
module from the repo's ``src`` path and asks it for a non-Transformer-Engine
decoder block spec that still preserves Moonlight MLA/MoE settings.
"""

from __future__ import annotations

import inspect


MoonlightLocalDotProductAttention = None
SequenceParallelSafeWrappedTorchNorm = None


def _moonlight_local_dot_product_attention():
    """Return a local attention class compatible with Megatron MLA constructors."""
    global MoonlightLocalDotProductAttention
    if MoonlightLocalDotProductAttention is not None:
        return MoonlightLocalDotProductAttention

    from megatron.core.transformer.dot_product_attention import DotProductAttention

    class _MoonlightLocalDotProductAttention(DotProductAttention):
        def __init__(self, *args, k_channels=None, v_channels=None, **kwargs):
            del k_channels, v_channels
            super().__init__(*args, **kwargs)

        def forward(self, query, key, value, *args, packed_seq_params=None, **kwargs):
            del packed_seq_params
            packed_mla = query.ndim == key.ndim == value.ndim == 3
            if packed_mla:
                query = query.unsqueeze(1)
                key = key.unsqueeze(1)
                value = value.unsqueeze(1)
            return super().forward(query, key, value, *args, packed_seq_params=None, **kwargs)

    _MoonlightLocalDotProductAttention.__name__ = "MoonlightLocalDotProductAttention"
    _MoonlightLocalDotProductAttention.__qualname__ = "MoonlightLocalDotProductAttention"
    _MoonlightLocalDotProductAttention.__module__ = __name__
    MoonlightLocalDotProductAttention = _MoonlightLocalDotProductAttention
    return MoonlightLocalDotProductAttention


def _patch_mla_core_attention(block_spec):
    core_attention = _moonlight_local_dot_product_attention()
    for layer_spec in getattr(block_spec, "layer_specs", []):
        self_attention = getattr(getattr(layer_spec, "submodules", None), "self_attention", None)
        attention_submodules = getattr(self_attention, "submodules", None)
        if hasattr(attention_submodules, "core_attention"):
            attention_submodules.core_attention = core_attention
    return block_spec


def _sequence_parallel_safe_wrapped_torch_norm():
    """Return a torch norm wrapper that tolerates Megatron sequence parallel."""
    global SequenceParallelSafeWrappedTorchNorm
    if SequenceParallelSafeWrappedTorchNorm is not None:
        return SequenceParallelSafeWrappedTorchNorm

    import torch

    class _SequenceParallelSafeWrappedTorchNorm:
        def __new__(
            cls,
            config,
            hidden_size: int,
            eps: float = 1e-5,
            persist_layer_norm: bool = False,
            zero_centered_gamma: bool = False,
            normalization: str = "LayerNorm",
        ):
            del persist_layer_norm, zero_centered_gamma
            norm_name = getattr(config, "normalization", normalization)
            if norm_name == "LayerNorm":
                return torch.nn.LayerNorm(normalized_shape=hidden_size, eps=eps)
            if norm_name == "RMSNorm":
                return torch.nn.RMSNorm(normalized_shape=hidden_size, eps=eps)
            raise ValueError(f"Unsupported torch norm for Moonlight local spec: {norm_name}")

    _SequenceParallelSafeWrappedTorchNorm.__name__ = "SequenceParallelSafeWrappedTorchNorm"
    _SequenceParallelSafeWrappedTorchNorm.__qualname__ = "SequenceParallelSafeWrappedTorchNorm"
    _SequenceParallelSafeWrappedTorchNorm.__module__ = __name__
    SequenceParallelSafeWrappedTorchNorm = _SequenceParallelSafeWrappedTorchNorm
    return SequenceParallelSafeWrappedTorchNorm


def _patch_sequence_parallel_rmsnorm(block_spec):
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    safe_norm = _sequence_parallel_safe_wrapped_torch_norm()

    def replace_wrapped_norm(container, name):
        if getattr(container, name, None) is WrappedTorchNorm:
            setattr(container, name, safe_norm)

    for layer_spec in getattr(block_spec, "layer_specs", []):
        layer_submodules = getattr(layer_spec, "submodules", None)
        if layer_submodules is None:
            continue
        replace_wrapped_norm(layer_submodules, "input_layernorm")
        replace_wrapped_norm(layer_submodules, "pre_mlp_layernorm")
        self_attention = getattr(layer_submodules, "self_attention", None)
        attention_submodules = getattr(self_attention, "submodules", None)
        if attention_submodules is not None:
            replace_wrapped_norm(attention_submodules, "q_layernorm")
            replace_wrapped_norm(attention_submodules, "kv_layernorm")
    block_spec.layer_norm = safe_norm
    return block_spec


def moonlight_local_decoder_block_spec(args, config, vp_stage=None):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.torch_norm import WrappedTorchNorm

    kwargs = {
        "use_transformer_engine": False,
        "normalization": getattr(args, "normalization", None),
        "qk_l2_norm": getattr(args, "qk_l2_norm", False),
    }
    if "vp_stage" in inspect.signature(get_gpt_decoder_block_spec).parameters:
        kwargs["vp_stage"] = vp_stage
    block_spec = _patch_mla_core_attention(get_gpt_decoder_block_spec(config, **kwargs))
    if getattr(args, "normalization", None) == "RMSNorm":
        if getattr(args, "sequence_parallel", False):
            block_spec = _patch_sequence_parallel_rmsnorm(block_spec)
        else:
            block_spec.layer_norm = WrappedTorchNorm
    return block_spec
