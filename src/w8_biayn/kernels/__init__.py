"""Custom Triton-kernel R&D lane for w8-biayn.

This package is import-safe on a CPU host without ``torch``: only :mod:`registry`
(plain metadata) is imported eagerly. The Triton implementations and pure-torch
references live in sibling modules that import ``torch`` lazily, so they only run on
a CUDA host (provisioned on GCP — this workstation has no GPU).
"""

from __future__ import annotations

from .registry import KERNELS, KernelSpec, get_kernel, kernel_names

__all__ = ["KERNELS", "KernelSpec", "get_kernel", "kernel_names"]
