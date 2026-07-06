"""Manage local ignored clones of pinned upstream repositories."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .constants import UPSTREAMS, Upstream
from .shell import run_command


def cache_root(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / ".cache" / "upstreams"


def upstream_path(upstream: Upstream, repo_root: str | Path = ".") -> Path:
    return cache_root(repo_root) / upstream.name


def clone_or_update(name: str, repo_root: str | Path = ".", dry_run: bool = False) -> Path:
    """Clone or update one pinned upstream into `.cache/upstreams`.

    The pin is a fixed commit: when the local clone already contains it, no
    network is touched at all -- a transient GitHub outage on a cloud node once
    killed an entire paid run at `git fetch` even though the pinned commit was
    already present locally. When a fetch IS needed, it retries.
    """

    upstream = UPSTREAMS[name]
    destination = upstream_path(upstream, repo_root)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        run_command(["git", "clone", upstream.repo, str(destination)], dry_run=dry_run)
    elif not dry_run and not _commit_present(destination, upstream.pin):
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                run_command(["git", "-C", str(destination), "fetch", "--all", "--tags"], dry_run=dry_run)
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - transient network fetches
                last_error = exc
                print(f"upstream fetch attempt {attempt + 1} failed for {name}: {exc}", flush=True)
                time.sleep(10)
        if last_error is not None:
            raise last_error
    elif dry_run:
        run_command(["git", "-C", str(destination), "fetch", "--all", "--tags"], dry_run=dry_run)

    run_command(["git", "-C", str(destination), "checkout", upstream.pin], dry_run=dry_run)
    return destination


def _commit_present(destination: Path, pin: str) -> bool:
    """True when the pinned ref already resolves to a commit in the clone."""

    proc = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "--verify", "--quiet", f"{pin}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


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
