from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import w8_biayn.kernels.patch as patch_mod
from w8_biayn.cli import app
from w8_biayn.kernels.patch import (
    apply_kernel_patches,
    apply_kernel_patches_from_env,
    parse_kernel_spec,
)


def test_parse_kernel_spec_validates_and_splits():
    assert parse_kernel_spec(None) == ()
    assert parse_kernel_spec("") == ()
    assert parse_kernel_spec(" logprob , entropy ") == ("logprob", "entropy")
    with pytest.raises(ValueError, match="unknown kernel"):
        parse_kernel_spec("logprob,bogus")


def test_apply_from_env_is_noop_when_unset():
    logs: list[str] = []
    assert apply_kernel_patches_from_env({}, log=logs.append) == []
    assert logs == []


def test_apply_skips_kernels_without_installer():
    logs: list[str] = []
    applied = apply_kernel_patches(("logprob", "mla"), log=logs.append)
    assert applied == []
    assert any("no validated implementation" in message for message in logs)


def test_apply_runs_installer_and_survives_failure(monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(patch_mod.INSTALLERS, "logprob", lambda: calls.append("logprob"))

    def boom() -> None:
        raise RuntimeError("nope")

    monkeypatch.setitem(patch_mod.INSTALLERS, "entropy", boom)
    logs: list[str] = []
    applied = apply_kernel_patches(("logprob", "entropy"), log=logs.append)

    assert applied == ["logprob"]
    assert calls == ["logprob"]
    assert any("keeping upstream" in message for message in logs)


def test_apply_from_env_reads_w8_biayn_kernels():
    logs: list[str] = []
    applied = apply_kernel_patches_from_env({"W8_BIAYN_KERNELS": "logprob"}, log=logs.append)
    assert applied == []  # no installer registered by default -> safe no-op
    assert any("logprob" in message for message in logs)


def test_cli_kernels_local_without_gpu_errors_clearly():
    result = CliRunner().invoke(app, ["kernels", "lab", "--kernel", "mla", "--local"])
    assert result.exit_code != 0
    assert "CUDA" in result.output


def test_cli_kernels_rejects_remote_and_local_together(tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text(
        json.dumps({"type": "service_account", "project_id": "proj", "private_key": "X"}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        app,
        ["kernels", "lab", "--kernel", "mla", "--remote", "--local", "--credentials", str(sa)],
    )
    assert result.exit_code != 0
    assert "not both" in result.output
