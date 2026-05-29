"""Tier-B Triton kernel: MoE grouped GEMM (forward + backward).

One fused kernel does E independent matmuls (per-expert FFN) over tokens grouped by
expert, replacing E separate cuBLAS launches.

Forward:  Y_e = X_e @ W[e].
Backward: dX_e = dY_e @ W[e]^T  (the forward kernel with W transposed)
          dW[e] = X_e^T @ dY_e  (per-expert reduction over the group's tokens)

Validated on an A100-SXM4-40GB (torch 2.11+cu128, triton 3.6) at E=8, T=8192, K=512,
N=1024 (bf16, variable group sizes): forward parity rel=2.99e-3 and **3.3x** faster than
the per-expert torch loop; backward grad parity rel dX=2.2e-3 / dW=3.3e-3 and **2.4x**
faster fwd+bwd. See `scripts/kernel_moe_gemm_bench.py` / `kernel_moe_gemm_grad.py` and
`reference.grouped_gemm_ref`.

Remaining Tier-B step: integration into Megatron's experts module via
`provider.transformer_layer_spec`. `triton` imports lazily/guarded so this module stays
importable on a CPU host.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001 - CPU host without torch/triton
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _fwd_kernel(
        x_ptr, w_ptr, y_ptr, goff_ptr, gsz_ptr, K, N,
        sxm, sxk, swe, swk, swn, sym, syn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        e = tl.program_id(0)
        mt = tl.program_id(1)
        nt = tl.program_id(2)
        gsize = tl.load(gsz_ptr + e)
        m0 = mt * BLOCK_M
        if m0 >= gsize:
            return
        goff = tl.load(goff_ptr + e)
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        rmask = offs_m < gsize
        nmask = offs_n < N
        gm = goff + offs_m
        xp = x_ptr + gm[:, None] * sxm + offs_k[None, :] * sxk
        wp = w_ptr + e * swe + offs_k[:, None] * swk + offs_n[None, :] * swn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kmask = (offs_k + k0) < K
            x = tl.load(xp, mask=rmask[:, None] & kmask[None, :], other=0.0)
            w = tl.load(wp, mask=kmask[:, None] & nmask[None, :], other=0.0)
            acc += tl.dot(x, w)
            xp += BLOCK_K * sxk
            wp += BLOCK_K * swk
        yp = y_ptr + gm[:, None] * sym + offs_n[None, :] * syn
        tl.store(yp, acc.to(y_ptr.dtype.element_ty), mask=rmask[:, None] & nmask[None, :])

    @triton.jit
    def _dw_kernel(
        x_ptr, dy_ptr, dw_ptr, goff_ptr, gsz_ptr, K, N,
        sxm, sxk, sdym, sdyn, sdwe, sdwk, sdwn,
        BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        # dW[e] = X_e^T @ dY_e -> tile [BLOCK_K, BLOCK_N], contracting over group tokens.
        e = tl.program_id(0)
        kt = tl.program_id(1)
        nt = tl.program_id(2)
        gsize = tl.load(gsz_ptr + e)
        goff = tl.load(goff_ptr + e)
        offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        kmask = offs_k < K
        nmask = offs_n < N
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        for m0 in range(0, gsize, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            mmask = offs_m < gsize
            gm = goff + offs_m
            xp = x_ptr + gm[:, None] * sxm + offs_k[None, :] * sxk
            dyp = dy_ptr + gm[:, None] * sdym + offs_n[None, :] * sdyn
            x = tl.load(xp, mask=mmask[:, None] & kmask[None, :], other=0.0)
            dy = tl.load(dyp, mask=mmask[:, None] & nmask[None, :], other=0.0)
            acc += tl.dot(tl.trans(x), dy)
        dwp = dw_ptr + e * sdwe + offs_k[:, None] * sdwk + offs_n[None, :] * sdwn
        tl.store(dwp, acc.to(dw_ptr.dtype.element_ty), mask=kmask[:, None] & nmask[None, :])

    def _grouped_off(group_sizes):
        return torch.cumsum(group_sizes, 0) - group_sizes

    def _fwd(x, w, group_sizes, bm=64, bn=64, bk=32):
        T, K = x.shape
        E, _, N = w.shape
        y = torch.empty((T, N), device=x.device, dtype=x.dtype)
        goff = _grouped_off(group_sizes)
        max_m = max(int(group_sizes.max().item()), 1)
        grid = (E, triton.cdiv(max_m, bm), triton.cdiv(N, bn))
        _fwd_kernel[grid](
            x, w, y, goff, group_sizes, K, N,
            x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
            y.stride(0), y.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        )
        return y

    def _dw(x, dy, group_sizes, bk=64, bn=64, bm=32):
        T, K = x.shape
        _, N = dy.shape
        E = group_sizes.shape[0]
        dw = torch.empty((E, K, N), device=x.device, dtype=x.dtype)
        goff = _grouped_off(group_sizes)
        grid = (E, triton.cdiv(K, bk), triton.cdiv(N, bn))
        _dw_kernel[grid](
            x, dy, dw, goff, group_sizes, K, N,
            x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
            dw.stride(0), dw.stride(1), dw.stride(2),
            BLOCK_K=bk, BLOCK_N=bn, BLOCK_M=bm,
        )
        return dw

    class _GroupedGEMM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w, group_sizes):
            ctx.save_for_backward(x, w, group_sizes)
            return _fwd(x, w, group_sizes)

        @staticmethod
        def backward(ctx, dy):
            x, w, group_sizes = ctx.saved_tensors
            dy = dy.contiguous()
            dx = _fwd(dy, w.transpose(1, 2).contiguous(), group_sizes)
            dw = _dw(x, dy, group_sizes)
            return dx, dw, None


def grouped_gemm(x: Any, w: Any, group_sizes: Any) -> Any:
    """Differentiable MoE grouped GEMM. x: [T,K] grouped by expert; w: [E,K,N]; sizes: [E].

    Returns y: [T, N]. Requires a CUDA GPU + triton. Autograd-enabled (dX, dW).
    """
    if not _HAVE_TRITON:
        raise RuntimeError("grouped_gemm requires triton + a CUDA GPU (run on the kernel-lab A100).")
    return _GroupedGEMM.apply(x, w, group_sizes)
