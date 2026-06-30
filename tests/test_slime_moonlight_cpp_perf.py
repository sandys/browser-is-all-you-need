from __future__ import annotations

import os
import subprocess
from pathlib import Path


EXAMPLE_ROOT = Path("examples/slime/moonlight_cpp_perf")
RUNNER = EXAMPLE_ROOT / "moonlight_cpp_perf.sh"
README = EXAMPLE_ROOT / "README.md"


def test_moonlight_cpp_perf_example_files_are_present_and_executable() -> None:
    expected = {
        "README.md",
        "moonlight_cpp_perf.sh",
        "prepare_data.sh",
        "eval_base.sh",
        "sft.sh",
        "eval_sft.sh",
        "grpo.sh",
        "eval_grpo.sh",
        "compare.sh",
    }

    assert expected.issubset({path.name for path in EXAMPLE_ROOT.iterdir()})
    for script in expected - {"README.md"}:
        assert os.access(EXAMPLE_ROOT / script, os.X_OK), script


def test_moonlight_cpp_perf_scripts_are_bash_syntax_valid() -> None:
    for script in sorted(EXAMPLE_ROOT.glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_moonlight_cpp_perf_runner_pins_stage_sequence_and_artifacts() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "prepare-data|base-eval|sft|sft-eval|grpo|grpo-eval|compare" in text
    assert 'RUN_ID="${SLIME_RUN_ID:-moonlight_cpp_perf}"' in text
    assert 'TASKS_DIR="${SLIME_CPP_TASKS_DIR:-${REPO_ROOT}/.w8-biayn/data/tasks-full}"' in text
    assert "w8_biayn.integrations.slime_cpp_perf build-data" in text
    assert "--train-limit" in text
    assert "--eval-limit" in text
    assert "base.records.jsonl" not in text
    assert "--save-debug-rollout-data" in text
    assert "rollout_dumps/${STAGE_LABEL}_{rollout_id}.pt" in text
    assert "vram_usage.csv" in text
    assert "vram_peak.txt" in text
    assert "run_receipt.txt" in text
    assert '"${SLIME_ROOT}/train.py"' in text
    assert "-u train.py" not in text


def test_moonlight_cpp_perf_runner_has_base_sft_grpo_slime_modes() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "Moonlight-16B-A3B-Instruct" in text
    assert "moonshotai/Moonlight-16B-A3B-Instruct" in text
    assert 'source "${SLIME_ROOT}/scripts/models/moonlight.sh"' in text
    assert "--num-rollout 0" in text
    assert "--eval-prompt-data cpp_perf" in text
    assert "--rollout-function-path slime.rollout.sft_rollout.generate_rollout" in text
    assert "--loss-type sft_loss" in text
    assert "--disable-compute-advantages-and-returns" in text
    assert 'require_checkpoint "${SFT_SAVE_DIR}" "SFT"' in text
    assert '--ref-load "${SLIME_GRPO_REF_LOAD_DIR:-${SFT_SAVE_DIR}}"' in text
    assert '--load "${SLIME_GRPO_ACTOR_LOAD_DIR:-${SFT_SAVE_DIR}}"' in text
    assert 'require_checkpoint "${GRPO_SAVE_DIR}" "GRPO"' in text
    assert "--advantage-estimator grpo" in text
    assert "--reward-key score" in text
    assert "--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func" in text


def test_moonlight_cpp_perf_runner_configures_wandb_and_local_sandbox() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in text.lower().splitlines() if line.strip() and not line.lstrip().startswith("#")
    )

    assert "--use-wandb" in text
    assert "--wandb-project" in text
    assert "slime-moonlight-cpp-perf" in text
    assert "--wandb-run-id" in text
    assert "W8_BIAYN_DATA_DIR" in text
    assert "W8_CPP_SANDBOX_IMAGE" in text
    assert "W8_CPP_SANDBOX_CPU" in text
    assert "DOCKER_HOST" in text
    assert "e2b" not in executable
    assert "browsergym" not in executable
    assert "webarena" not in executable


def test_moonlight_cpp_perf_readme_has_exact_operator_order() -> None:
    text = README.read_text(encoding="utf-8")

    assert "base checkpoint" in text
    assert "after SLIME SFT" in text
    assert "after SLIME GRPO" in text
    assert "does not use E2B" in text
    assert "local Docker sandbox" in text
    assert "uv run w8-biayn data pie download" in text
    assert "uv run w8-biayn data pie build-full-tasks" in text
    assert "bash examples/slime/moonlight_cpp_perf/prepare_data.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/eval_base.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/sft.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/eval_sft.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/grpo.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/eval_grpo.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/compare.sh" in text
    assert "comparison.json" in text
    assert "Do not treat a completed launch as proof of learning" in text
