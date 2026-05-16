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


def test_cli_launch_r3_with_domdiff_dry_run_prints_domdiff_plan(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "r3",
            "--credentials",
            str(credentials),
            "--with-domdiff",
            "--benchmark",
            "webvoyager-domdiff-heldout",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DOMDiff dry run" in result.output
    assert "sky launch" in result.output
    assert "webvoyager-domdiff-heldout" in result.output


def test_cli_domdiff_push_image_dry_run(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "domdiff",
            "push-image",
            "--credentials",
            str(credentials),
            "--source-image",
            "android-world-domdiff:local",
            "--tag",
            "test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DOMDiff Artifact Registry image push dry run" in result.output
    assert "docker push us-central1-docker.pkg.dev/proj/w8-biayn/android-world-domdiff:test" in result.output


def test_cli_launch_r3_with_local_domdiff_image_dry_run_prints_push_and_domdiff_plan(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "r3",
            "--credentials",
            str(credentials),
            "--with-domdiff",
            "--local-reward-image",
            "android-world-domdiff:local",
            "--artifact-tag",
            "test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "docker push us-central1-docker.pkg.dev/proj/w8-biayn/android-world-domdiff:test" in result.output
    assert "reward_image=us-central1-docker.pkg.dev/proj/w8-biayn/android-world-domdiff:test" in result.output


def test_cli_domdiff_local_up_dry_run():
    result = CliRunner().invoke(
        app,
        [
            "domdiff",
            "local",
            "up",
            "--image",
            "android-world-domdiff:local",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Local DOMDiff dry run" in result.output
    assert "docker run -d --name android-world-domdiff" in result.output
    assert "w8_biayn.rewards.chromiumrl_service:app" in result.output


def test_cli_launch_r3_with_local_domdiff_dry_run_prints_local_plan(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "r3",
            "--credentials",
            str(credentials),
            "--with-local-domdiff",
            "--benchmark",
            "webvoyager-domdiff-heldout",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Local DOMDiff dry run" in result.output
    assert "CHROMIUMRL_URL" in result.output
    assert "https://<local-domdiff-reward-tunnel>" in result.output
    assert "CDP_URL=wss://<local-domdiff-cdp-tunnel>" not in result.output
    assert "wss://<local-domdiff-cdp-tunnel>" not in result.output
    assert "sky launch" in result.output


def test_cli_launch_r3_with_local_domdiff_can_publish_cdp_for_debug(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "r3",
            "--credentials",
            str(credentials),
            "--with-local-domdiff",
            "--local-publish-cdp",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "CDP_URL=wss://<local-domdiff-cdp-tunnel>" in result.output


def test_cli_launch_r3_rejects_push_option_with_local_domdiff(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "r3",
            "--credentials",
            str(credentials),
            "--with-local-domdiff",
            "--local-reward-image",
            "android-world-domdiff:local",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--local-reward-image uploads to Artifact Registry" in result.output


def test_cli_benchmarks_list():
    result = CliRunner().invoke(app, ["benchmarks", "list"])

    assert result.exit_code == 0, result.output
    assert "webvoyager-domdiff-heldout" in result.output
    assert "androidworld-transfer" in result.output
