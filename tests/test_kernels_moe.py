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
    x = torch.randn(total, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * (K**-0.5)

    y_ref = grouped_gemm_ref(x, w, sizes)
    y_tri = grouped_gemm(x, w, sizes)
    rel = (y_ref.float() - y_tri.float()).abs().max() / (y_ref.float().abs().max() + 1e-6)
    assert rel.item() < 2e-2, rel.item()
