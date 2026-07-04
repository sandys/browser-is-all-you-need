from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from w8_biayn.cli import app


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_MODULE = ROOT / "src/w8_biayn/cloud_launch.py"
EXAMPLE_SHIM = ROOT / "examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py"
RUNNER = ROOT / "examples/slime/glm47_cpp_perf/glm47_cpp_perf.sh"


def test_launch_module_defaults_to_single_h100_8_in_allowed_gcp_regions() -> None:
    text = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert 'ALLOWED_REGIONS = ("asia-southeast1", "asia-south1", "asia-south2")' in text
    assert 'DEFAULT_REGION = "asia-southeast1"' in text
    assert 'DEFAULT_ACCELERATORS = "H100:8"' in text
    assert "use_spot=options.use_spot" in text


def test_launch_module_runs_full_glm_stage_sequence_and_not_smoke_grpo() -> None:
    text = LAUNCH_MODULE.read_text(encoding="utf-8")
    stages = [
        "prepare_data.sh",
        "eval_base.sh",
        "sft.sh",
        "eval_sft.sh",
        "grpo.sh",
        "eval_grpo.sh",
        "compare.sh",
    ]
    positions = [text.index(f"bash examples/slime/glm47_cpp_perf/{stage}") for stage in stages]

    assert positions == sorted(positions)
    assert "FULL_LIMIT_SENTINEL = 1_000_000" in text
    assert 'export SLIME_GRPO_NUM_ROLLOUT="$W8_GLM47_GRPO_NUM_ROLLOUT"' in text
    assert "export SLIME_GRPO_SKIP_WEIGHT_UPDATE=0" in text


def test_launch_module_downloads_artifacts_and_tears_down_cluster() -> None:
    text = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert "download_artifacts(" in text
    assert "sky.down(cluster_name)" in text
    assert "labels=_resource_labels(options.run_id)" in text
    assert "WANDB_API_KEY" in text
    assert "<redacted>" in text


def test_launch_module_pins_skypilot_and_tracks_job_to_terminal_state() -> None:
    text = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert "SKYPILOT_PIN" in text
    assert "_wait_for_job_completion(" in text
    assert 'final_status.endswith("SUCCEEDED")' in text
    # Setup must probe for preinstalled tools instead of blanket apt installs;
    # docker.io conflicts with the docker-ce shipped on SkyPilot GPU images.
    assert "missing_packages" in text


def test_example_launcher_is_a_shim_that_delegates_to_the_cli() -> None:
    text = EXAMPLE_SHIM.read_text(encoding="utf-8")

    assert "w8-biayn launch glm47-full" in text
    assert "from w8_biayn.cli import app" in text
    assert "sky.Task" not in text


def test_cli_launch_glm47_full_dry_run_renders_without_skypilot(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "glm47-full",
            "--run-id",
            "unittest-dry",
            "--accelerators",
            "A100-80GB:8",
            "--use-spot",
            "--local-output-root",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"cluster": "w8-glm47-h100-unittest-dry"' in result.output
    assert "--- setup ---" in result.output
    assert "--- run ---" in result.output
    config = json.loads((tmp_path / "cloud-runs" / "unittest-dry" / "launch_config.json").read_text())
    assert config["use_spot"] is True
    assert config["accelerators"] == "A100-80GB:8"
    assert config["skypilot_pin"].startswith("skypilot-nightly[gcp]==")
    assert config["wandb_api_key"] == ""


def test_cli_launch_glm47_full_rejects_disallowed_region(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "launch",
            "glm47-full",
            "--region",
            "us-central1",
            "--local-output-root",
            str(tmp_path),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "us-central1" in result.output


def test_glm_runner_keeps_wandb_files_inside_run_root() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'WANDB_DIR_ROOT="${SLIME_WANDB_DIR:-${RUN_ROOT}/wandb}"' in text
    assert 'WANDB_DIR_ROOT="$(absolute_path "${WANDB_DIR_ROOT}")"' in text
    assert 'export WANDB_DIR="${WANDB_DIR_ROOT}/${STAGE}"' in text
    assert '"WANDB_DIR",' in text
    assert 'wandb_dir=${WANDB_DIR:-}' in text
