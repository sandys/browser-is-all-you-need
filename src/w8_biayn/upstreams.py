"""Manage local ignored clones of pinned upstream repositories."""

from __future__ import annotations

from pathlib import Path

from .constants import UPSTREAMS, Upstream
from .shell import run_command


def cache_root(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / ".cache" / "upstreams"


def upstream_path(upstream: Upstream, repo_root: str | Path = ".") -> Path:
    return cache_root(repo_root) / upstream.name


def clone_or_update(name: str, repo_root: str | Path = ".", dry_run: bool = False) -> Path:
    """Clone or update one pinned upstream into `.cache/upstreams`."""
    upstream = UPSTREAMS[name]
    destination = upstream_path(upstream, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        run_command(["git", "clone", upstream.repo, str(destination)], dry_run=dry_run)
    else:
        run_command(["git", "-C", str(destination), "fetch", "--all", "--tags"], dry_run=dry_run)

    run_command(["git", "-C", str(destination), "checkout", upstream.pin], dry_run=dry_run)
    if (destination / ".gitmodules").exists():
        run_command(["git", "-C", str(destination), "submodule", "update", "--init", "--recursive"], dry_run=dry_run)
    return destination


def status(repo_root: str | Path = ".") -> list[dict[str, str]]:
    """Return status rows for all configured upstreams."""
    rows: list[dict[str, str]] = []
    for key, upstream in UPSTREAMS.items():
        path = upstream_path(upstream, repo_root)
        head = ""
        state = "missing"
        if path.exists():
            import subprocess

            proc = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                head = proc.stdout.strip()
                state = "pinned" if head == upstream.pin else "different"
            else:
                state = "invalid"
        rows.append(
            {
                "key": key,
                "name": upstream.name,
                "path": str(path),
                "pin": upstream.pin,
                "head": head,
                "state": state,
            }
        )
    return rows

