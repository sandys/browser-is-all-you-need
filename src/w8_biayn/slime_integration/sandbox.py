"""Sandbox helpers for repo-owned SLIME integrations."""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_ENV = "SLIME_AGENT_SANDBOX_BACKEND"
DEFAULT_BACKEND = "e2b"
VALID_BACKENDS = {"e2b", "docker"}


class SandboxError(RuntimeError):
    """Raised when sandbox setup or execution fails."""


@dataclass(frozen=True)
class CompletedExec:
    returncode: int
    stdout: str
    stderr: str


class DockerSandbox:
    """Disposable local Docker implementation of SLIME's sandbox contract."""

    keepalive_command = ("sh", "-lc", "trap 'exit 0' TERM INT; while :; do sleep 3600; done")

    def __init__(self, image: str, *, timeout: int | None = None, name: str | None = None) -> None:
        self.image = image
        self.timeout = timeout or 3600
        self.container_name = name or f"slime-agent-{uuid.uuid4().hex[:12]}"
        self.sandbox_id = self.container_name
        self.container_id = ""

    async def __aenter__(self) -> DockerSandbox:
        result = await self._run_subprocess(self._build_run_command(), check=True)
        self.container_id = result.stdout.strip()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._run_subprocess(["docker", "rm", "-f", self.container_name], check=False)

    def _build_run_command(self) -> list[str]:
        return [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            self.image,
            *self.keepalive_command,
        ]

    def _build_exec_command(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
    ) -> list[str]:
        args = ["docker", "exec", "--user", user]
        if env:
            for key, value in env.items():
                args.extend(["-e", f"{key}={value}"])
        args.extend([self.container_name, "sh", "-lc", cmd])
        return args

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> tuple[int, str, str]:
        result = await self._run_subprocess(
            self._build_exec_command(cmd, user=user, env=env),
            timeout=timeout,
            check=check,
        )
        return result.returncode, result.stdout, result.stderr

    async def write_file(self, sandbox_path: str, content: str | bytes | Path, *, user: str = "root") -> None:
        parent = shlex.quote(str(Path(sandbox_path).parent))
        await self.exec(f"mkdir -p {parent}", user="root", timeout=30, check=True)

        temp_path: Path | None = None
        source_path: Path
        if isinstance(content, Path):
            source_path = content
        else:
            fd, raw_path = tempfile.mkstemp(prefix="slime-agent-", suffix=".tmp")
            temp_path = Path(raw_path)
            os.close(fd)
            if isinstance(content, bytes):
                temp_path.write_bytes(content)
            else:
                temp_path.write_text(content, encoding="utf-8")
            source_path = temp_path

        try:
            target = f"{self.container_name}:{sandbox_path}"
            await self._run_subprocess(["docker", "cp", str(source_path), target], timeout=120, check=True)
            if user != "root":
                quoted = shlex.quote(sandbox_path)
                await self.exec(f"chown {user}:{user} {quoted}", user="root", timeout=30, check=True)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        quoted = shlex.quote(sandbox_path)
        _, stdout, _ = await self.exec(f"cat {quoted}", user=user, timeout=30, check=True)
        return stdout

    async def _run_subprocess(
        self,
        args: list[str],
        *,
        timeout: int | None = None,
        check: bool = False,
    ) -> CompletedExec:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise SandboxError(f"command timed out: {' '.join(args)}") from exc

        result = CompletedExec(
            proc.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
        if check and result.returncode != 0:
            raise SandboxError(
                f"command failed (exit={result.returncode}): {' '.join(args)}\n{result.stderr[:400]}"
            )
        return result


def sandbox_backend_from_env(default: str = DEFAULT_BACKEND) -> str:
    backend = os.environ.get(BACKEND_ENV, default).strip().lower()
    if backend not in VALID_BACKENDS:
        raise SandboxError(f"unsupported {BACKEND_ENV}={backend!r}; expected one of {sorted(VALID_BACKENDS)}")
    return backend


def create_sandbox(image: str, *, backend: str | None = None, **kwargs: Any) -> Any:
    backend_name = (backend or sandbox_backend_from_env()).strip().lower()
    if backend_name == "docker":
        return DockerSandbox(image, timeout=kwargs.get("timeout"))
    if backend_name == "e2b":
        try:
            from slime.agent.sandbox import E2BSandbox  # type: ignore
        except ImportError as exc:
            raise SandboxError(
                "E2B backend requested but the upstream slime package is not importable in this environment"
            ) from exc
        return E2BSandbox(
            image,
            timeout=kwargs.get("timeout"),
            image_metadata_key=kwargs.get("image_metadata_key"),
            rpc_retries=kwargs.get("rpc_retries"),
        )
    raise SandboxError(f"unsupported backend: {backend_name}")
