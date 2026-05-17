"""Run packaged Harbor browser tasks in Docker on the trainer host."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import HarborTask, load_task


@dataclass(frozen=True)
class HarborDockerOutcome:
    task_id: str
    task_dir: Path
    image: str
    container: str
    reward_doc: dict[str, Any]
    diagnostics: dict[str, Any]
    preview_headers: dict[str, str]
    logs: str

    @property
    def preview_url(self) -> str:
        return str(self.diagnostics.get("preview_url") or "")

    @property
    def terminal_state(self) -> str:
        return str(self.diagnostics.get("terminal_state") or "")


class HarborDockerTaskRunner:
    def __init__(
        self,
        *,
        chromiumrl_url: str | None = None,
        oracle: bool = True,
        keep_containers: bool = False,
        build_timeout_sec: int = 1800,
        test_timeout_sec: int = 900,
    ) -> None:
        self.chromiumrl_url = chromiumrl_url or ""
        self.oracle = oracle
        self.keep_containers = keep_containers
        self.build_timeout_sec = build_timeout_sec
        self.test_timeout_sec = test_timeout_sec

    def dry_run_plan(self, task: HarborTask) -> str:
        tag = image_tag_for_task(task.path)
        solution_step = (
            "docker exec w8-harbor-<session> bash /task/solution/solve.sh  # oracle smoke"
            if self.oracle
            else "docker exec -i w8-harbor-<session> sh -lc 'cat > /tmp/w8_solution.sh && bash /tmp/w8_solution.sh'"
        )
        return "\n".join(
            [
                f"Harbor Docker task dry run: {task.task_id}",
                f"docker build -t {tag} {task.path / 'environment'}",
                f"docker run -d --name w8-harbor-<session> --network host {tag}",
                "docker cp <task assets> w8-harbor-<session>:/task",
                solution_step,
                "docker exec w8-harbor-<session> bash /tests/test.sh",
                "ChromiumRL evaluation: "
                + (self.chromiumrl_url or "disabled until --chromiumrl-url is provided"),
            ]
        )

    def run(
        self,
        task_id: str,
        *,
        task_root: str | Path | None = None,
        solution_script: str | None = None,
    ) -> HarborDockerOutcome:
        task = load_task(task_id, task_root)
        image = image_tag_for_task(task.path)
        container = container_name_for_task(task.task_id)
        self._build_image(task, image)
        self._start_container(image, container)
        try:
            self._copy_task_assets(task, container)
            if self.oracle:
                self._exec(container, ["bash", "/task/solution/solve.sh"], workdir="/testbed")
            elif solution_script:
                self._copy_generated_solution(container, solution_script)
                self._exec(container, ["bash", "/tmp/w8_solution.sh"], workdir="/testbed")
            self._run_verifier(task, container)
            reward_doc = self._read_json(container, "/logs/verifier/reward.json")
            diagnostics = self._read_json(container, "/logs/verifier/diagnostics.json")
            preview_headers = self._read_optional_json(container, "/trial/preview_headers.json")
            logs = self._logs(container)
            return HarborDockerOutcome(
                task_id=task.task_id,
                task_dir=task.path,
                image=image,
                container=container,
                reward_doc=reward_doc,
                diagnostics=diagnostics,
                preview_headers={str(k): str(v) for k, v in preview_headers.items()},
                logs=logs,
            )
        finally:
            if not self.keep_containers:
                subprocess.run(["docker", "rm", "-f", container], check=False)

    def _build_image(self, task: HarborTask, image: str) -> None:
        self._run(
            ["docker", "build", "-t", image, str(task.path / "environment")],
            timeout=self.build_timeout_sec,
        )

    def _start_container(self, image: str, container: str) -> None:
        subprocess.run(["docker", "rm", "-f", container], check=False, stdout=subprocess.DEVNULL)
        self._run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "--network",
                "host",
                image,
            ],
            timeout=120,
        )

    def _copy_task_assets(self, task: HarborTask, container: str) -> None:
        self._exec(container, ["mkdir", "-p", "/task", "/tests", "/logs/verifier", "/trial"])
        for name in ("instruction.md", "task.toml", "rubric.json", "gold_signals.json"):
            self._run(["docker", "cp", str(task.path / name), f"{container}:/task/{name}"])
        self._run(["docker", "cp", str(task.path / "solution"), f"{container}:/task/solution"])
        self._copy_directory_contents(task.path / "tests", container, "/tests")
        self._exec(container, ["sh", "-lc", "test -f /tests/test.sh && chmod +x /tests/test.sh"], timeout=30)

    def _copy_directory_contents(self, source_dir: Path, container: str, target_dir: str) -> None:
        for source in sorted(source_dir.rglob("*")):
            relative = source.relative_to(source_dir).as_posix()
            target = f"{target_dir.rstrip('/')}/{relative}"
            if source.is_dir():
                self._exec(container, ["mkdir", "-p", target], timeout=30)
            elif source.is_file():
                self._exec(container, ["mkdir", "-p", str(Path(target).parent)], timeout=30)
                self._run(["docker", "cp", str(source), f"{container}:{target}"])

    def _copy_generated_solution(self, container: str, solution_script: str) -> None:
        self._run(
            [
                "docker",
                "exec",
                "-i",
                container,
                "sh",
                "-lc",
                "cat > /tmp/w8_solution.sh && chmod +x /tmp/w8_solution.sh",
            ],
            input_text=solution_script,
            timeout=30,
        )

    def _run_verifier(self, task: HarborTask, container: str) -> None:
        verifier_env = task.task_config.get("verifier", {}).get("env", {})
        env: dict[str, str] = {
            "CHROMIUMRL_API_URL": self.chromiumrl_url,
            "W8_HARBOR_PREVIEW_TUNNEL": "1",
            "W8_HARBOR_PREVIEW_TUNNEL_TIMEOUT_SEC": "120",
        }
        if isinstance(verifier_env, dict):
            env.update({str(k): str(v) for k, v in verifier_env.items()})
        self._exec(
            container,
            ["bash", "/tests/test.sh"],
            workdir="/tests",
            env=env,
            timeout=self.test_timeout_sec,
        )

    def _exec(
        self,
        container: str,
        command: list[str],
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        cmd = ["docker", "exec"]
        for key, value in (env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        if workdir:
            cmd.extend(["-w", workdir])
        cmd.append(container)
        cmd.extend(command)
        return self._run(cmd, timeout=timeout)

    def _read_json(self, container: str, path: str) -> dict[str, Any]:
        result = self._exec(container, ["sh", "-lc", f"cat {path}"], timeout=30)
        doc = json.loads(result.stdout)
        if not isinstance(doc, dict):
            raise RuntimeError(f"Expected object JSON in {container}:{path}")
        return doc

    def _read_optional_json(self, container: str, path: str) -> dict[str, Any]:
        result = self._exec(container, ["sh", "-lc", f"cat {path} 2>/dev/null || echo '{{}}'"], timeout=30)
        doc = json.loads(result.stdout)
        return doc if isinstance(doc, dict) else {}

    def _logs(self, container: str) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", "400", container],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return result.stdout

    def _run(
        self,
        command: list[str],
        *,
        input_text: str | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            check=False,
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(command)}\n{_tail(result.stdout)}"
            )
        return result


def image_tag_for_task(task_dir: str | Path) -> str:
    path = Path(task_dir)
    digest = hashlib.sha256()
    for candidate in sorted((path / "environment").rglob("*")):
        if candidate.is_file():
            digest.update(str(candidate.relative_to(path / "environment")).encode())
            digest.update(candidate.read_bytes())
    safe = _safe_name(path.name, max_len=48)
    return f"w8-biayn-harbor-{safe}:{digest.hexdigest()[:12]}"


def container_name_for_task(task_id: str) -> str:
    suffix = f"{os.getpid()}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}"
    return _safe_name(f"w8-harbor-{task_id}-{suffix}", max_len=63)


def run_oracle_smoke(
    task_id: str,
    *,
    task_root: str | Path | None = None,
    chromiumrl_url: str | None = None,
    keep_container: bool = False,
) -> HarborDockerOutcome:
    runner = HarborDockerTaskRunner(
        chromiumrl_url=chromiumrl_url,
        oracle=True,
        keep_containers=keep_container,
    )
    return runner.run(task_id, task_root=task_root)


def _safe_name(value: str, *, max_len: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-").lower()
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha256(safe.encode()).hexdigest()[:8]
    return f"{safe[: max_len - 9]}-{digest}"


def _tail(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[-limit:]
