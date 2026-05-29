"""Tier-B: Triton MoE grouped-GEMM — forward parity + microbench (runs on a GPU VM).

A MoE expert FFN applies, per expert e, Y_rows(e) = X_rows(e) @ W[e]. With tokens grouped
by expert this is a "grouped GEMM": one kernel doing E independent matmuls of variable row
counts, avoiding E separate cuBLAS launches. This script generates grouped data, runs a
pure-torch reference and the Triton kernel, checks numeric parity, and microbenches both.

This is the isolated-op half of Tier-B acceptance (forward). Backward + model integration
(via provider.transformer_layer_spec) are the next iterations.
"""

from __future__ import annotations

import traceback

import torch


def show(t: str) -> None:
    print(f"\n===== {t} =====", flush=True)


show("env")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import triton
import triton.language as tl

print("triton", triton.__version__, "| device", torch.cuda.get_device_name(0))


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


def grouped_gemm_triton(x, w, group_sizes, block_m=64, block_n=64, block_k=32):
    T, K = x.shape
    E, K2, N = w.shape
    assert K == K2
    y = torch.empty((T, N), device=x.device, dtype=x.dtype)
    group_off = torch.cumsum(group_sizes, 0) - group_sizes
    max_m = int(group_sizes.max().item()) if T > 0 else 0
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


def grouped_gemm_ref(x, w, group_sizes):
    T, K = x.shape
    E, _, N = w.shape
    y = torch.empty((T, N), device=x.device, dtype=x.dtype)
    off = 0
    for e in range(E):
        n = int(group_sizes[e].item())
        if n:
            y[off : off + n] = x[off : off + n] @ w[e]
        off += n
    return y


def make_data(E, T, K, N, dtype):
    torch.manual_seed(0)
    sizes = torch.randint(1, max(2, T // E), (E,), device="cuda")
    sizes[-1] = max(1, T - int(sizes[:-1].sum().item()))  # make them sum to ~T
    sizes = sizes.clamp(min=0).to(torch.int32)
    total = int(sizes.sum().item())
    x = torch.randn(total, K, device="cuda", dtype=dtype)
    w = torch.randn(E, K, N, device="cuda", dtype=dtype) * (K**-0.5)
    return x, w, sizes


def bench(fn, iters=50):
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 1e6


def main():
    dtype = torch.bfloat16
    E, T, K, N = 8, 8192, 512, 1024
    show(f"grouped GEMM  E={E} T={T} K={K} N={N} dtype={dtype}")
    x, w, sizes = make_data(E, T, K, N, dtype)
    print("group_sizes:", sizes.tolist(), "sum", int(sizes.sum()))

    y_ref = grouped_gemm_ref(x, w, sizes)
    y_tri = grouped_gemm_triton(x, w, sizes)
    diff = (y_ref.float() - y_tri.float()).abs().max().item()
    denom = y_ref.float().abs().max().item() + 1e-6
    rel = diff / denom
    print(f"max_abs_diff={diff:.4f}  rel={rel:.2e}")
    parity = rel < 2e-2
    print("MOE_GEMM_PARITY_OK" if parity else "MOE_GEMM_PARITY_FAIL")

    if parity:
        t_ref, m_ref = bench(lambda: grouped_gemm_ref(x, w, sizes))
        t_tri, m_tri = bench(lambda: grouped_gemm_triton(x, w, sizes))
        print(f"ref   : {t_ref:.3f} ms  peak {m_ref:.0f} MB")
        print(f"triton: {t_tri:.3f} ms  peak {m_tri:.0f} MB")
        print(f"SPEEDUP x{t_ref / t_tri:.2f}  MEM x{m_ref / max(m_tri,1e-6):.2f}")


try:
    main()
    show("DONE")
except Exception:
    traceback.print_exc()
    print("MOE_GEMM_RUN_FAILED")
