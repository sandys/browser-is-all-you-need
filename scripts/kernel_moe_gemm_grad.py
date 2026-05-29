"""Tier-B: MoE grouped-GEMM backward — gradient parity + fwd/bwd microbench (GPU).

Forward: per expert e, Y_e = X_e @ W[e].
Backward:  dX_e = dY_e @ W[e]^T   (a grouped GEMM with W transposed -> reuse fwd kernel)
           dW[e] = X_e^T @ dY_e   (per-expert reduction over the group's tokens -> new kernel)

Wrapped in an autograd.Function and checked against autograd through a cat-based torch
reference. Diagnostic-heavy so one run pinpoints the next fix.
"""

from __future__ import annotations

import traceback

import torch
import triton
import triton.language as tl


def show(t: str) -> None:
    print(f"\n===== {t} =====", flush=True)


show("env")
print("torch", torch.__version__, "triton", triton.__version__, "cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))


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
    # dW[e] = X_e^T @ dY_e  -> tile [BLOCK_K, BLOCK_N], contract over the group's tokens.
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
        xp = x_ptr + gm[:, None] * sxm + offs_k[None, :] * sxk   # [BLOCK_M, BLOCK_K]
        dyp = dy_ptr + gm[:, None] * sdym + offs_n[None, :] * sdyn  # [BLOCK_M, BLOCK_N]
        x = tl.load(xp, mask=mmask[:, None] & kmask[None, :], other=0.0)
        dy = tl.load(dyp, mask=mmask[:, None] & nmask[None, :], other=0.0)
        acc += tl.dot(tl.trans(x), dy)  # [BLOCK_K, BLOCK_M] @ [BLOCK_M, BLOCK_N]
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
    _fwd_kernel[grid](x, w, y, goff, group_sizes, K, N,
                      x.stride(0), x.stride(1), w.stride(0), w.stride(1), w.stride(2),
                      y.stride(0), y.stride(1), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)
    return y


def _dw(x, dy, group_sizes, bk=64, bn=64, bm=32):
    T, K = x.shape
    _, N = dy.shape
    E = group_sizes.shape[0]
    dw = torch.empty((E, K, N), device=x.device, dtype=x.dtype)
    goff = _grouped_off(group_sizes)
    grid = (E, triton.cdiv(K, bk), triton.cdiv(N, bn))
    _dw_kernel[grid](x, dy, dw, goff, group_sizes, K, N,
                     x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
                     dw.stride(0), dw.stride(1), dw.stride(2),
                     BLOCK_K=bk, BLOCK_N=bn, BLOCK_M=bm)
    return dw


class GroupedGEMM(torch.autograd.Function):
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


def ref_cat(x, w, group_sizes):
    outs, off = [], 0
    for e in range(group_sizes.shape[0]):
        n = int(group_sizes[e].item())
        outs.append(x[off : off + n] @ w[e])
        off += n
    return torch.cat(outs, 0)


def main():
    dtype = torch.bfloat16
    E, K, N = 8, 512, 1024
    sizes = torch.tensor([308, 877, 851, 14, 783, 628, 996, 3735], dtype=torch.int32, device="cuda")
    T = int(sizes.sum())
    show(f"grad parity  E={E} T={T} K={K} N={N} {dtype}")
    torch.manual_seed(0)
    x0 = torch.randn(T, K, device="cuda", dtype=dtype)
    w0 = torch.randn(E, K, N, device="cuda", dtype=dtype) * (K**-0.5)
    g = torch.randn(T, N, device="cuda", dtype=dtype)

    xr = x0.clone().requires_grad_(True)
    wr = w0.clone().requires_grad_(True)
    ref_cat(xr, wr, sizes).backward(g)

    xt = x0.clone().requires_grad_(True)
    wt = w0.clone().requires_grad_(True)
    GroupedGEMM.apply(xt, wt, sizes).backward(g)

    def rel(a, b):
        return ((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-6)).item()

    rdx = rel(xr.grad, xt.grad)
    rdw = rel(wr.grad, wt.grad)
    print(f"rel dX={rdx:.2e}  rel dW={rdw:.2e}")
    ok = rdx < 3e-2 and rdw < 3e-2
    print("MOE_GEMM_BWD_PARITY_OK" if ok else "MOE_GEMM_BWD_PARITY_FAIL")

    if ok:
        def fb_triton():
            xt2 = x0.clone().requires_grad_(True)
            wt2 = w0.clone().requires_grad_(True)
            GroupedGEMM.apply(xt2, wt2, sizes).backward(g)

        def fb_ref():
            xr2 = x0.clone().requires_grad_(True)
            wr2 = w0.clone().requires_grad_(True)
            ref_cat(xr2, wr2, sizes).backward(g)

        for f in (fb_triton, fb_ref):
            f()
        torch.cuda.synchronize()
        import time

        def timed(f, n=30):
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                f()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / n * 1e3

        t_tri = timed(fb_triton)
        t_ref = timed(fb_ref)
        print(f"fwd+bwd  ref {t_ref:.3f} ms  triton {t_tri:.3f} ms  SPEEDUP x{t_ref / t_tri:.2f}")


try:
    main()
    show("DONE")
except Exception:
    traceback.print_exc()
    print("MOE_GEMM_GRAD_RUN_FAILED")
