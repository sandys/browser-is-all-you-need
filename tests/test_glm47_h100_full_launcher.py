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


def test_run_script_heredoc_terminates_and_docker_runs_after_it() -> None:
    from w8_biayn.cloud_launch import build_run_script

    script = build_run_script()
    # The terminator must sit at column 0 or bash swallows the rest of the
    # script (docker pull/run included) into the entrypoint file and exits 0.
    assert "\nW8_GLM47_CONTAINER\n" in script
    after_heredoc = script.split("\nW8_GLM47_CONTAINER\n", 1)[1]
    assert "docker pull" in after_heredoc
    assert "docker run" in after_heredoc
    assert 'test -f "$W8_REMOTE_RUN_ROOT/eval/comparison.json"' in after_heredoc


def test_teardown_uses_the_run_id_tag_shared_with_wandb() -> None:
    from typer.testing import CliRunner

    from w8_biayn.cli import app
    from w8_biayn.cloud_launch import default_cluster_name, run_label_value

    # The cluster name and the GCE run_id label are both deterministic from the
    # run id, which is also the W&B group — one id ties launch, tags, and W&B.
    rid = "w8pilot-a100r-20260704203227"
    assert default_cluster_name(rid) == f"w8-glm47-h100-{rid}"
    assert run_label_value(rid) == rid
    # The launch always tears down anything still live, not just on the success
    # path (cluster_acquired alone would leak a box that fails mid-wait).
    src = (ROOT / "src/w8_biayn/cloud_launch.py").read_text(encoding="utf-8")
    assert "cluster_acquired or cluster_live" in src
    # down-run is registered and renders a delete plan without executing.
    result = CliRunner().invoke(app, ["ops", "down-run", "--help"])
    assert result.exit_code == 0
    assert "tagged with this run id" in result.output


def test_dataset_cache_is_restore_first_and_gate_keyed() -> None:
    from w8_biayn.cloud_launch import TASKS_CACHE_VERSION, LaunchOptions, build_run_script

    run = build_run_script()
    restore_at = run.index("data cache restore")
    build_at = run.index("data pie build-full-tasks")
    upload_at = run.index("data cache upload")
    # Restore must be attempted before building, and the cache repopulated after.
    assert restore_at < build_at < upload_at
    # The cache key includes the version and every admission gate so different
    # inputs never collide and identical inputs are shared across users.
    prefix_line = run.split("W8_TASKS_CACHE_PREFIX=", 1)[1].splitlines()[0]
    for keyed in ("W8_GLM47_CACHE_VERSION", "W8_GLM47_MIN_TRAIN", "W8_GLM47_MIN_VALIDATION", "W8_GLM47_MIN_TEST"):
        assert keyed in prefix_line
    # The version literal is carried into the run via the env var above.
    assert LaunchOptions(run_id="t", dry_run=True).artifact_bucket == ""  # unresolved until launch
    assert TASKS_CACHE_VERSION == "cpp-perf-v1"
    # Cache upload is best-effort: a node without bucket write must not fail.
    assert "cache upload skipped" in run


def test_container_script_dedents_and_embedded_python_parses() -> None:
    import ast

    from w8_biayn.cloud_launch import build_container_script

    script = build_container_script()
    # A single template line reaching column 0 (e.g. an unescaped \n inside a
    # nested string literal) makes dedent a no-op and indents every heredoc
    # terminator, silently truncating the script at execution time.
    assert script.splitlines()[0] == "set -euo pipefail"
    for terminator in ("W8_WANDB_CHECK", "W8_MEGATRON_PATCH"):
        assert f"\n{terminator}\n" in script, f"{terminator} heredoc must terminate at column 0"
    embedded = script.split("python - <<'W8_MEGATRON_PATCH'\n", 1)[1].split("\nW8_MEGATRON_PATCH", 1)[0]
    ast.parse(embedded)
    assert "\\n" in embedded, "patch string escapes must reach the embedded python intact"


def test_glm_lane_disables_fully_parallel_ckpt_load_and_pins_slime() -> None:
    from w8_biayn.cloud_launch import build_container_script

    entry_text = (ROOT / "src/w8_biayn/integrations/slime_train_entry.py").read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    container = build_container_script()

    # SLIME force-enables ckpt_fully_parallel_load post-parse; the wrapper
    # crashes on TE extra_state object shards (BytesIO has no len), so the
    # repo-owned train entry must be able to disable it via env.
    assert "W8_SLIME_NO_FULLY_PARALLEL_CKPT_LOAD" in entry_text
    assert "args.ckpt_fully_parallel_load = False" in entry_text
    assert "W8_SLIME_NO_FULLY_PARALLEL_CKPT_LOAD" in runner_text
    # The container must run the pinned SLIME, not a git-pull moving target.
    assert 'git fetch origin "$W8_GLM47_SLIME_PIN"' in container
    assert "git pull" not in container
    # Every env the container script reads must be passed through docker run.
    from w8_biayn.cloud_launch import build_run_script
    import re

    run_script = build_run_script()
    for var in sorted(set(re.findall(r"\$\{?(W8_GLM47_[A-Z_]+)", container))):
        assert f"-e {var}" in run_script or f'-e {var}=' in run_script, f"missing docker passthrough for {var}"


def test_pipeline_milestones_use_the_non_shadowed_script_path() -> None:
    from w8_biayn.cloud_launch import build_container_script, build_run_script

    # scripts/ placement matters: invoked by path, a script inside
    # src/w8_biayn/ would shadow stdlib `secrets` with w8_biayn/secrets.py
    # via sys.path[0] and crash wandb on import.
    assert "scripts/wandb_milestone.py" in build_run_script()
    assert "scripts/wandb_milestone.py" in build_container_script()
    assert "src/w8_biayn/wandb_milestones.py" not in build_run_script()
    runner_text = RUNNER.read_text(encoding="utf-8")
    assert "scripts/wandb_milestone.py" in runner_text


def test_launch_module_parallelizes_coverage_and_verifies_wandb() -> None:
    text = LAUNCH_MODULE.read_text(encoding="utf-8")

    assert '--jobs "$W8_GLM47_COVERAGE_JOBS"' in text
    assert '--min-train "$W8_GLM47_MIN_TRAIN"' in text
    assert "wandb_auth_ok" in text
    assert "WANDB_API_KEY missing inside the training container" in text
    # Empty WANDB_* env vars crash wandb's pydantic Settings validation.
    assert "unset WANDB_BASE_URL" in text
    assert "unset WANDB_ENTITY" in text


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


def test_glm_runner_default_parallelism_fits_eight_gpus() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    # ETP*EP*PP must divide the default world size of 8 (Megatron asserts it).
    assert 'EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"' in text
    assert 'PP_SIZE="${SLIME_PIPELINE_MODEL_PARALLEL_SIZE:-2}"' in text
    assert 'ETP_SIZE="${SLIME_EXPERT_TENSOR_PARALLEL_SIZE:-1}"' in text
    # 512-token responses truncated 100% of generations (W&B truncated_ratio=1)
    # and would zero out GRPO advantages; budgets must fit reasoning + full C++.
    assert 'GRPO_MAX_RESPONSE_LEN="${SLIME_GRPO_MAX_RESPONSE_LEN:-2048}"' in text
    assert 'EVAL_MAX_RESPONSE_LEN="${SLIME_EVAL_MAX_RESPONSE_LEN:-2048}"' in text
    assert 'SEQ_LENGTH="${SLIME_SEQ_LENGTH:-4096}"' in text


def test_glm_runner_keeps_wandb_files_inside_run_root() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert 'WANDB_DIR_ROOT="${SLIME_WANDB_DIR:-${RUN_ROOT}/wandb}"' in text
    assert 'WANDB_DIR_ROOT="$(absolute_path "${WANDB_DIR_ROOT}")"' in text
    assert 'export WANDB_DIR="${WANDB_DIR_ROOT}/${STAGE}"' in text
    assert '"WANDB_DIR",' in text
    assert 'wandb_dir=${WANDB_DIR:-}' in text
