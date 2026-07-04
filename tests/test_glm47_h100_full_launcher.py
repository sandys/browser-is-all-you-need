from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py"
RUNNER = ROOT / "examples/slime/glm47_cpp_perf/glm47_cpp_perf.sh"


def test_launcher_defaults_to_single_h100_8_in_allowed_gcp_regions() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'ALLOWED_REGIONS = ("asia-southeast1", "asia-south1", "asia-south2")' in text
    assert 'DEFAULT_REGION = "asia-southeast1"' in text
    assert 'DEFAULT_ACCELERATORS = "H100:8"' in text
    assert """sky.Resources(
                    infra=f"gcp/{region}",
                    accelerators=args.accelerators,""" in text


def test_launcher_runs_full_glm_stage_sequence_and_not_smoke_grpo() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
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


def test_launcher_downloads_artifacts_and_tears_down_cluster() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "download_artifacts(" in text
    assert "sky.down(cluster_name)" in text
    assert "labels=_resource_labels(args.run_id)" in text
    assert "WANDB_API_KEY" in text
    assert "<redacted>" in text


def test_launcher_pins_skypilot_and_tracks_job_to_terminal_state() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'SKYPILOT_PIN = "skypilot-nightly[gcp]==' in text
    assert "use_spot=args.use_spot" in text
    assert "_wait_for_job_completion(" in text
    assert 'final_status.endswith("SUCCEEDED")' in text
    # Setup must probe for preinstalled tools instead of blanket apt installs;
    # docker.io conflicts with the docker-ce shipped on SkyPilot GPU images.
    assert "missing_packages" in text


def test_glm_runner_keeps_wandb_files_inside_run_root() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'WANDB_DIR_ROOT="${SLIME_WANDB_DIR:-${RUN_ROOT}/wandb}"' in text
    assert 'WANDB_DIR_ROOT="$(absolute_path "${WANDB_DIR_ROOT}")"' in text
    assert 'export WANDB_DIR="${WANDB_DIR_ROOT}/${STAGE}"' in text
    assert '"WANDB_DIR",' in text
    assert 'wandb_dir=${WANDB_DIR:-}' in text
