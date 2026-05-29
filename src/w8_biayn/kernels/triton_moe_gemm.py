"""Tier-B Triton kernel: MoE grouped GEMM (forward).

One fused kernel does E independent matmuls (per-expert FFN) over tokens grouped by
expert, replacing E separate cuBLAS launches. Validated on an A100-SXM4-40GB
(torch 2.11+cu128, triton 3.6): forward parity rel=2.99e-3 (bf16) and **3.44x** faster
than the per-expert torch loop at E=8, T=8192, K=512, N=1024 (see
`scripts/kernel_moe_gemm_bench.py` and `reference.grouped_gemm_ref`).

Forward only for now. Backward (dX, dW grouped GEMMs) and integration into Megatron's
experts module via `provider.transformer_layer_spec` are the next Tier-B steps; the
kernel is registered as a `patch.py` installer only once those pass parity.

`triton` imports lazily/guarded so this module stays importable on a CPU host.
"""

from __future__ import annotations

from typing import Any

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001 - CPU host without triton
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _grouped_gemm_fwd(
        x_ptr, w_ptr, y_ptr, group_off_ptr, group_size_ptr,
        K, N,
        stride_xm, stride_xk,
        stride_we, stride_wk, stride_wn,
        stride_ym, stride_yn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        e = tl.program_id(0)
        mt = tl.program_id(1)
        nt = tl.program_id(2)
        gsize = tl.load(group_size_ptr + e)
        m0 = mt * BLOCK_M
        if m0 >= gsize:
            return
        goff = tl.load(group_off_ptr + e)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < gsize
        n_mask = offs_n < N
        global_m = goff + offs_m
        x_ptrs = x_ptr + global_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + e * stride_we + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_mask = (offs_k + k0) < K
            x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        y_ptrs = y_ptr + global_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=row_mask[:, None] & n_mask[None, :])


def grouped_gemm(x: Any, w: Any, group_sizes: Any, block_m: int = 64, block_n: int = 64, block_k: int = 32) -> Any:
    """MoE grouped GEMM (forward). x: [T,K] grouped by expert; w: [E,K,N]; group_sizes: [E].

    Returns y: [T, N]. Requires a CUDA GPU + triton.
    """
    if not _HAVE_TRITON:
        raise RuntimeError("grouped_gemm requires triton + a CUDA GPU (run on the kernel-lab A100).")
    import torch

    T, K = x.shape
    E, K2, N = w.shape
    assert K == K2, (K, K2)
    y = torch.empty((T, N), device=x.device, dtype=x.dtype)
    group_off = torch.cumsum(group_sizes, 0) - group_sizes
    max_m = int(group_sizes.max().item()) if T > 0 else 1
    grid = (E, triton.cdiv(max(max_m, 1), block_m), triton.cdiv(N, block_n))
    _grouped_gemm_fwd[grid](
        x, w, y, group_off, group_sizes,
        K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
    )
    return y
