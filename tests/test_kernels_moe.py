from __future__ import annotations

import os

import pytest

from w8_biayn.kernels import reference, triton_moe_gemm
from w8_biayn.kernels.registry import get_kernel


def test_registry_marks_moe_gemm_validated():
    assert "validated" in get_kernel("moe-gemm").status


def test_kernel_modules_import_without_torch_or_triton():
    # On a CPU host neither torch nor triton is installed; the modules must still import.
    assert hasattr(triton_moe_gemm, "grouped_gemm")
    assert hasattr(reference, "grouped_gemm_ref")


def test_grouped_gemm_requires_triton(monkeypatch):
    # Without triton the wrapper must fail fast with a clear, actionable error
    # (before touching torch), so a CPU invocation never silently misbehaves.
    monkeypatch.setattr(triton_moe_gemm, "_HAVE_TRITON", False)
    with pytest.raises(RuntimeError, match="triton"):
        triton_moe_gemm.grouped_gemm(None, None, None)


@pytest.mark.skipif(
    os.environ.get("W8_BIAYN_RUN_GPU_TESTS") != "1",
    reason="needs a CUDA GPU + triton (set W8_BIAYN_RUN_GPU_TESTS=1 on the kernel-lab A100)",
)
def test_grouped_gemm_parity_gpu():
    import torch

    from w8_biayn.kernels.reference import grouped_gemm_ref
    from w8_biayn.kernels.triton_moe_gemm import grouped_gemm

    torch.manual_seed(0)
    E, K, N = 8, 256, 512
    sizes = torch.tensor([3, 64, 1, 128, 50, 200, 7, 99], dtype=torch.int32, device="cuda")
    total = int(sizes.sum())
    x0 = torch.randn(total, K, device="cuda", dtype=torch.bfloat16)
    w0 = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * (K**-0.5)
    g = torch.randn(total, N, device="cuda", dtype=torch.bfloat16)

    def rel(a, b):
        return ((a.float() - b.float()).abs().max() / (a.float().abs().max() + 1e-6)).item()

    # forward parity
    xr, wr = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
    yr = grouped_gemm_ref(xr, wr, sizes)
    xt, wt = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
    yt = grouped_gemm(xt, wt, sizes)
    assert rel(yr, yt) < 2e-2

    # backward grad parity (dX, dW)
    yr.backward(g)
    yt.backward(g)
    assert rel(xr.grad, xt.grad) < 3e-2
    assert rel(wr.grad, wt.grad) < 3e-2
