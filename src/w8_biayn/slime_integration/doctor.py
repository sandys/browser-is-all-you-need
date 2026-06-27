"""Health checks for the pinned SLIME upstream clone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from w8_biayn import upstreams
from w8_biayn.constants import UPSTREAMS

SLIME_UPSTREAM_KEY = "slime"
REQUIRED_SLIME_PATHS = (
    "README.md",
    "train.py",
    "train_async.py",
    "slime",
    "examples",
    "docs",
)


@dataclass(frozen=True)
class SlimeDoctorCheck:
    path: str
    status: str
    detail: str


@dataclass(frozen=True)
class SlimeDoctorReport:
    root: Path
    ok: bool
    checks: tuple[SlimeDoctorCheck, ...]
    next_action: str | None = None


def slime_root(repo_root: str | Path = ".") -> Path:
    """Return the expected local path of the pinned SLIME clone."""

    return upstreams.upstream_path(UPSTREAMS[SLIME_UPSTREAM_KEY], repo_root)


def run_slime_doctor(repo_root: str | Path = ".") -> SlimeDoctorReport:
    """Validate that the pinned SLIME clone has the expected top-level layout."""

    root = slime_root(repo_root)
    checks: list[SlimeDoctorCheck] = []

    if not root.exists():
        return SlimeDoctorReport(
            root=root,
            ok=False,
            checks=(SlimeDoctorCheck(str(root), "missing", "SLIME upstream clone is absent"),),
            next_action="uv run w8-biayn upstreams clone slime",
        )

    for relative in REQUIRED_SLIME_PATHS:
        path = root / relative
        checks.append(
            SlimeDoctorCheck(
                relative,
                "ok" if path.exists() else "missing",
                str(path),
            )
        )

    ok = all(check.status == "ok" for check in checks)
    return SlimeDoctorReport(
        root=root,
        ok=ok,
        checks=tuple(checks),
        next_action=None if ok else "uv run w8-biayn upstreams clone slime",
    )
