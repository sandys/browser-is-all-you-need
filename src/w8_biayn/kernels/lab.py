"""Single-device kernel lab runner (executes on a CUDA GPU, e.g. a GCP A100).

Invoked as ``w8-biayn kernels <mode> --local`` on the lab VM. The body is implemented and
validated during the A100 session — it cannot run on a CPU-only host. Keeping it a thin,
import-safe dispatcher means the CLI and the rendered lab plan stay coherent locally while
the GPU numerics are written where they can actually be tested.
"""

from __future__ import annotations


def run_local(*, mode: str, kernel: str, model: str, dtype: str, out: str) -> None:
    """Run a kernel microbench (``mode='bench'``) or tiny-model integration (``'lab'``).

    Requires a CUDA build of ``torch``; raises a clear error otherwise.
    """

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only off-GPU
        raise RuntimeError(
            "kernels --local needs a CUDA build of torch and runs on a GPU host. "
            "Use `w8-biayn kernels lab --remote` to provision a GCP A100."
        ) from exc

    raise NotImplementedError(
        f"kernel '{kernel}' {mode} runs in the A100 lab session: discover the Megatron-Bridge "
        "provider key, prove the no-op ModuleSpec identity swap, then implement and validate the "
        "Triton kernel (numeric-tolerance parity + speed/memory). Tracked as the paid Phase-0/Tier-B step. "
        f"(model={model}, dtype={dtype}, out={out})"
    )
