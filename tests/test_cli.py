from __future__ import annotations

import json
import subprocess

import yaml
from typer.testing import CliRunner

import w8_biayn.cli as cli_mod
from w8_biayn.cli import SKYPILOT_GCP_LAUNCH_PERMISSIONS, app
from w8_biayn.constants import DEFAULT_GPU_CONTAINER_IMAGE
from w8_biayn.harbor.tasks import DEFAULT_HARBOR_TASK_IDS


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
    assert "gcloud auth activate-service-account" not in result.output
    assert "gcloud config set project" not in result.output
    assert "sky launch" in result.output


def test_cli_doctor_cloud_rejects_missing_skypilot_launch_permission(monkeypatch, tmp_path):
    credentials = service_account(tmp_path)

    monkeypatch.setattr(cli_mod, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, '{"gcp": true}', ""),
    )
    monkeypatch.setattr(
        cli_mod,
        "check_project_permissions",
        lambda *args, **kwargs: ["resourcemanager.projects.setIamPolicy"],
    )

    result = CliRunner().invoke(app, ["doctor", "--credentials", str(credentials), "--cloud"])

    assert result.exit_code == 1
    assert "resourcemanager.projects.setIamPolicy" in result.output


def test_skypilot_launch_preflight_covers_project_permissions():
    permissions = set(SKYPILOT_GCP_LAUNCH_PERMISSIONS)

    assert "compute.instances.create" in permissions
    assert "compute.firewalls.create" in permissions
    assert "compute.reservations.list" in permissions
    assert "iam.serviceAccounts.create" in permissions
    assert "resourcemanager.projects.getIamPolicy" in permissions
    assert "resourcemanager.projects.setIamPolicy" in permissions
    assert "serviceusage.services.enable" in permissions
    assert "storage.buckets.create" in permissions


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


def test_cli_config_render_rejects_private_domdiff_url(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "r3",
            "--credentials",
            str(credentials),
            "--chromiumrl-url",
            "http://127.0.0.1:8080",
        ],
    )

    assert result.exit_code != 0
    assert "local/private URL" in result.output


def test_cli_config_render_harbor_r3_smoke(tmp_path):
    credentials = service_account(tmp_path)
    output = tmp_path / "harbor.sky.yaml"
    task_id = DEFAULT_HARBOR_TASK_IDS[0]
    result = CliRunner().invoke(
        app,
        [
            "config",
            "render",
            "r3",
            "--credentials",
            str(credentials),
            "--output",
            str(output),
            "--benchmark",
            "harbor-domdiff-browser-swe",
            "--chromiumrl-url",
            "https://reward.trycloudflare.com",
            "--harbor-task",
            task_id,
        ],
    )

    assert result.exit_code == 0, result.output
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert "secrets" not in config
    assert config["envs"]["CHROMIUMRL_URL"] == "https://reward.trycloudflare.com"
    assert "CDP_URL" not in config["envs"]
    assert DEFAULT_GPU_CONTAINER_IMAGE in config["run"]
    assert "docker run --rm --gpus all --network host --shm-size=32g" in config["run"]
    assert "w8-biayn harbor prepare-data" in config["run"]
    assert "w8_biayn.integrations.skyrl_harbor_main" in config["run"]
    assert "TINKER_API_KEY" not in config["run"]
    assert task_id in config["run"]
    assert DEFAULT_HARBOR_TASK_IDS[1] not in config["run"]


def test_cli_launch_harbor_with_local_domdiff_dry_run(tmp_path):
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
            "harbor-domdiff-browser-swe",
            "--harbor-task",
            DEFAULT_HARBOR_TASK_IDS[0],
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Local DOMDiff dry run" in result.output
    assert "https://<local-domdiff-reward-tunnel>" in result.output
    assert "wss://<local-domdiff-cdp-tunnel>" not in result.output
    assert "sky launch" in result.output


def test_cli_launch_harbor_rejects_non_r3_pipeline(tmp_path):
    credentials = service_account(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "miniwob",
            "--credentials",
            str(credentials),
            "--benchmark",
            "harbor-domdiff-browser-swe",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "supported only for the r3" in result.output
    assert "pipeline" in result.output


def test_cli_harbor_commands_validate_and_dry_run():
    list_result = CliRunner().invoke(app, ["harbor", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert DEFAULT_HARBOR_TASK_IDS[0] in list_result.output
    assert DEFAULT_HARBOR_TASK_IDS[1] in list_result.output

    validate_result = CliRunner().invoke(app, ["harbor", "validate", "--task", DEFAULT_HARBOR_TASK_IDS[0]])
    assert validate_result.exit_code == 0, validate_result.output
    assert "ok" in validate_result.output

    smoke_result = CliRunner().invoke(
        app,
        [
            "harbor",
            "oracle-smoke",
            "--task",
            DEFAULT_HARBOR_TASK_IDS[0],
            "--chromiumrl-url",
            "https://reward.trycloudflare.com",
            "--dry-run",
        ],
    )
    assert smoke_result.exit_code == 0, smoke_result.output
    assert "Harbor Docker task dry run" in smoke_result.output
    assert "docker build" in smoke_result.output


def test_cli_benchmarks_list():
    result = CliRunner().invoke(app, ["benchmarks", "list"])

    assert result.exit_code == 0, result.output
    assert "webvoyager-domdiff-heldout" in result.output
    assert "androidworld-transfer" in result.output
