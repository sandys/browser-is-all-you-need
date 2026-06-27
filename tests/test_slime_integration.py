from __future__ import annotations

import pytest

from w8_biayn.slime_integration import sandbox as sandbox_module


def test_sandbox_backend_from_env_defaults_to_e2b(monkeypatch):
    monkeypatch.delenv(sandbox_module.BACKEND_ENV, raising=False)

    assert sandbox_module.sandbox_backend_from_env() == "e2b"


def test_sandbox_backend_from_env_accepts_docker(monkeypatch):
    monkeypatch.setenv(sandbox_module.BACKEND_ENV, "Docker")

    assert sandbox_module.sandbox_backend_from_env() == "docker"


def test_sandbox_backend_from_env_rejects_unknown(monkeypatch):
    monkeypatch.setenv(sandbox_module.BACKEND_ENV, "podman")

    with pytest.raises(sandbox_module.SandboxError):
        sandbox_module.sandbox_backend_from_env()


def test_create_sandbox_returns_docker_backend():
    sandbox = sandbox_module.create_sandbox("repo-image:latest", backend="docker", timeout=99)

    assert isinstance(sandbox, sandbox_module.DockerSandbox)
    assert sandbox.image == "repo-image:latest"
    assert sandbox.timeout == 99


def test_docker_sandbox_builds_exec_command():
    sandbox = sandbox_module.DockerSandbox("repo-image:latest", name="slime-agent-test")

    command = sandbox._build_exec_command("echo hi", user="agent", env={"A": "B", "C": "D"})

    assert command[:4] == ["docker", "exec", "--user", "agent"]
    assert "-e" in command
    assert "A=B" in command
    assert "C=D" in command
    assert command[-4:] == ["slime-agent-test", "sh", "-lc", "echo hi"]
