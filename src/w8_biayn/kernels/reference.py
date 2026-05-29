"""Pure-torch reference oracles for the custom kernels (lazy torch import).

These mirror the upstream math and are the parity ground truth the Triton kernels are
judged against. Imported only on a GPU host (or wherever torch is installed); the kernels
package stays importable on a CPU box because nothing here is imported eagerly.
"""

from __future__ import annotations

from typing import Any


def grouped_gemm_ref(x: Any, w: Any, group_sizes: Any) -> Any:
    """Reference MoE grouped GEMM: per expert e, Y_rows(e) = X_rows(e) @ W[e].

    x: [T, K] tokens grouped (sorted) by expert; w: [E, K, N]; group_sizes: [E] int.
    Returns y: [T, N]. This is the per-expert torch loop (E separate matmuls).
    """
    import torch

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
