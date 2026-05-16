"""Small subprocess wrapper for CLI commands."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


def format_command(args: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in args)


def run_command(
    args: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Run a command or print it in dry-run mode."""
    if dry_run:
        print(format_command(args))
        return None

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        [str(part) for part in args],
        cwd=Path(cwd) if cwd is not None else None,
        env=merged_env,
        check=True,
        text=True,
    )

