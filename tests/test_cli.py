from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from w8_biayn.cli import app


def service_account(tmp_path):
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps({"type": "service_account", "project_id": "proj", "private_key": "SECRET"}),
        encoding="utf-8",
    )
    return path


def test_cli_config_render(tmp_path):
    credentials = service_account(tmp_path)
    output = tmp_path / "task.yaml"
    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "miniwob",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["name"] == "w8-biayn-miniwob"


def test_cli_launch_dry_run_prints_commands(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "miniwob",
            "--credentials",
            str(credentials),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "gcloud auth activate-service-account" in result.output
    assert "sky launch" in result.output

