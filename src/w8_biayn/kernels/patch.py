"""Gated activation of custom kernels (control-flow is torch-free).

Reads ``W8_BIAYN_KERNELS``, validates the names against the registry, and applies each
requested kernel via its installer. Installers live in the ``triton_*`` modules and import
``torch`` lazily; until a kernel has a *validated* Triton implementation it has no installer
and is skipped (logged), so activation is always safe — an unset or partially-implemented
spec never changes behavior.

Each installer is responsible for self-checking parity against the pure-torch reference
before patching and skipping on mismatch (guards ``SKYRL_PIN`` drift): bit-identical only
for a pass-through, numeric tolerances (fwd/bwd/optimizer-step) for a real kernel. Any
installer exception is caught here so a bad kernel leaves upstream untouched rather than
breaking the trainer.
"""

from __future__ import annotations

import os
from typing import Callable, Mapping

from .registry import kernel_names

# name -> installer callable. Populated by the triton_* modules once a kernel is validated
# on a GPU. Empty by default, so activation is a safe no-op until real kernels land.
INSTALLERS: dict[str, Callable[[], None]] = {}


def parse_kernel_spec(value: str | None) -> tuple[str, ...]:
    """Parse a ``W8_BIAYN_KERNELS`` value into validated kernel names."""

    if not value:
        return ()
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    known = set(kernel_names())
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(
            "unknown kernel(s) in W8_BIAYN_KERNELS: "
            + ", ".join(unknown)
            + "; known: "
            + ", ".join(sorted(known))
        )
    return names


def apply_kernel_patches(names: tuple[str, ...], *, log: Callable[[str], None] = print) -> list[str]:
    """Apply each named kernel via its installer; return the kernels actually activated."""

    applied: list[str] = []
    for name in names:
        installer = INSTALLERS.get(name)
        if installer is None:
            log(
                f"[w8-biayn kernels] '{name}' requested but has no validated implementation "
                "yet; skipping (implement + validate in the A100 lab)."
            )
            continue
        try:
            installer()
        except Exception as exc:  # noqa: BLE001 - keep upstream on any failure
            log(f"[w8-biayn kernels] '{name}' install failed ({exc!r}); keeping upstream.")
            continue
        applied.append(name)
        log(f"[w8-biayn kernels] '{name}' active.")
    return applied


def apply_kernel_patches_from_env(
    env: Mapping[str, str] | None = None,
    *,
    log: Callable[[str], None] = print,
) -> list[str]:
    """Activate kernels named in ``W8_BIAYN_KERNELS``. No-op when unset.

    Called inside SkyRL's Ray entrypoint (beside ``register_harbor_env``) so patches reach
    the trainer worker process, never the local driver only.
    """

    environ = os.environ if env is None else env
    names = parse_kernel_spec(environ.get("W8_BIAYN_KERNELS"))
    if not names:
        return []
    return apply_kernel_patches(names, log=log)
