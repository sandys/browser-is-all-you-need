from __future__ import annotations

from pathlib import Path


EXAMPLE_ROOT = Path("examples/slime/retool")
SFT_LAUNCHER = EXAMPLE_ROOT / "retool_moonlight_sft.sh"
RL_LAUNCHER = EXAMPLE_ROOT / "retool_moonlight_rl.sh"
README = EXAMPLE_ROOT / "README.md"


def test_retool_example_keeps_upstream_helper_structure() -> None:
    expected_files = {
        "README.md",
        "generate_with_retool.py",
        "tool_sandbox.py",
        "sft_data_processing.py",
        "rl_data_preprocess.py",
        "requirements.txt",
        "retool_moonlight_sft.sh",
        "retool_moonlight_rl.sh",
    }

    assert expected_files.issubset({path.name for path in EXAMPLE_ROOT.iterdir()})


def test_retool_launchers_use_moonlight_model_wiring() -> None:
    for launcher in (SFT_LAUNCHER, RL_LAUNCHER):
        text = launcher.read_text(encoding="utf-8")

        assert 'source "${SLIME_ROOT}/scripts/models/moonlight.sh"' in text
        assert 'HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/Moonlight-16B-A3B-Instruct}"' in text
        assert 'REF_LOAD_DIR="${SLIME_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist}"' in text
        assert 'SLIME_ATTENTION_BACKEND:-' in text
        assert "--attention-backend flash" not in text
        assert "Qwen" not in text
        assert "qwen" not in text
        assert "qwen3-4B.sh" not in text


def test_retool_rl_launcher_keeps_custom_retool_hooks() -> None:
    text = RL_LAUNCHER.read_text(encoding="utf-8")

    assert "--custom-generate-function-path generate_with_retool.generate" in text
    assert "--custom-rm-path generate_with_retool.reward_func" in text
    assert 'PYTHONPATH": ":".join(' in text
    assert '"${SCRIPT_DIR}"' in text


def test_retool_launchers_configure_wandb_diagnostics_like_multi_agent() -> None:
    for launcher in (SFT_LAUNCHER, RL_LAUNCHER):
        text = launcher.read_text(encoding="utf-8")

        assert 'WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"' in text
        assert "WANDB_ALREADY_LOGGED_IN" in text
        assert '${HOME}/.config/wandb/settings' in text
        assert "--wandb-project" in text
        assert "--wandb-group" in text
        assert "--disable-wandb-random-suffix" in text
        assert "--wandb-run-id" in text
        assert "SLIME_WANDB_RUN_ID" in text


def test_retool_readme_documents_moonlight_paths() -> None:
    text = README.read_text(encoding="utf-8")

    assert "SLIME ReTool Moonlight Example" in text
    assert "SLIME_HF_CHECKPOINT" in text
    assert "SLIME_REF_LOAD_DIR" in text
    assert "SLIME_WANDB_PROJECT" in text
    assert "SLIME_WANDB_RUN_ID" in text
    assert "retool_moonlight_sft.sh" in text
    assert "retool_moonlight_rl.sh" in text
