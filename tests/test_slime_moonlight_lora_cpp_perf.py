from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from w8_biayn.slime_integration.lora import available_flags, build_lora_args


EXAMPLE_ROOT = Path("examples/slime/moonlight_lora_cpp_perf")
RUNNER = EXAMPLE_ROOT / "moonlight_lora_cpp_perf.sh"
README = EXAMPLE_ROOT / "README.md"
BASE_RUNNER = Path("examples/slime/moonlight_cpp_perf/moonlight_cpp_perf.sh")


def test_lora_resolver_maps_rank_16_and_optional_flags() -> None:
    help_text = """
    --enable-lora
    --lora-rank LORA_RANK
    --lora-alpha LORA_ALPHA
    --lora-dropout LORA_DROPOUT
    --lora-target-modules LORA_TARGET_MODULES
    """

    plan = build_lora_args(
        help_text,
        rank=16,
        alpha="32",
        dropout="0.0",
        target_modules="q_proj,k_proj,v_proj,o_proj",
    )

    assert plan.args == [
        "--enable-lora",
        "--lora-rank",
        "16",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.0",
        "--lora-target-modules",
        "q_proj,k_proj,v_proj,o_proj",
    ]
    assert plan.rank_flag == "--lora-rank"
    assert plan.enable_flag == "--enable-lora"
    assert "--lora-rank" in available_flags(help_text)


def test_lora_resolver_fails_without_supported_rank_flag() -> None:
    with pytest.raises(ValueError, match="does not expose a supported LoRA rank flag"):
        build_lora_args("--enable-lora --other-flag", rank=16)


def test_moonlight_lora_cpp_perf_example_files_are_present_and_executable() -> None:
    expected = {
        "README.md",
        "moonlight_lora_cpp_perf.sh",
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


def test_moonlight_lora_cpp_perf_scripts_are_bash_syntax_valid() -> None:
    for script in sorted(EXAMPLE_ROOT.glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_moonlight_lora_wrapper_defaults_and_guards() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "BASE_RUNNER" in text
    assert "examples/slime/moonlight_cpp_perf/moonlight_cpp_perf.sh" in text
    assert 'SLIME_RUN_ID="${SLIME_RUN_ID:-moonlight_lora16_cpp_perf}"' in text
    assert "moonlight-lora-cpp-perf" in text
    assert 'SLIME_CPP_PROFILE="${SLIME_CPP_PROFILE:-moonlight-lora16-cpp-perf}"' in text
    assert 'SLIME_WANDB_PROJECT="${SLIME_WANDB_PROJECT:-slime-moonlight-lora-cpp-perf}"' in text
    assert 'SLIME_LORA_RANK="${SLIME_LORA_RANK:-16}"' in text
    assert "stage_uses_lora" in text
    assert "sft | sft-eval | grpo | grpo-eval" in text
    assert "base-eval" not in text.split("stage_uses_lora", 1)[1].split("}", 1)[0]
    assert "w8_biayn.slime_integration.lora resolve" in text
    assert "--help-command" in text
    assert "SLIME_LORA_EXTRA_ARGS" in text
    assert "W8_RESOLVED_LORA_ARGS" in text


def test_base_moonlight_runner_receipts_carry_lora_metadata() -> None:
    text = BASE_RUNNER.read_text(encoding="utf-8")

    assert "lora_enabled=${SLIME_LORA_ENABLED:-}" in text
    assert "lora_rank=${SLIME_LORA_RANK:-}" in text
    assert "lora_extra_args=${SLIME_LORA_EXTRA_ARGS:-}" in text
    assert "resolved_lora_extra_args=${W8_RESOLVED_LORA_ARGS:-}" in text


def test_moonlight_lora_readme_has_container_commands_and_wandb() -> None:
    text = README.read_text(encoding="utf-8")

    assert "Kimi Moonlight" in text
    assert "LoRA rank 16" in text
    assert "SLIME_LORA_RANK=16" in text
    assert "WANDB_API_KEY" in text
    assert "SLIME_WANDB_PROJECT=slime-moonlight-lora-cpp-perf" in text
    assert "bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh" in text
    assert "bash examples/slime/moonlight_lora_cpp_perf/grpo.sh" in text
    assert "bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh" in text
    assert "bash examples/slime/moonlight_lora_cpp_perf/compare.sh" in text
    assert "comparison.json" in text
    assert "resolved_lora_extra_args" in text
