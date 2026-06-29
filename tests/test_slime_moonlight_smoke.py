from __future__ import annotations

import json
from pathlib import Path


EXAMPLE_ROOT = Path("examples/slime/moonlight_moe_smoke")
LAUNCHER = EXAMPLE_ROOT / "run_moonlight_16b_a3b_int4_smoke.sh"
DATA = EXAMPLE_ROOT / "moonlight_math_smoke.jsonl"
README = EXAMPLE_ROOT / "README.md"


def test_moonlight_smoke_launcher_uses_local_int4_moe_defaults() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "scripts/models/moonlight.sh" in text
    assert "Moonlight-16B-A3B-Instruct" in text
    assert "moonlight_math_smoke.jsonl" in text
    assert 'NUM_GPUS="${SLIME_NUM_GPUS:-4}"' in text
    assert 'TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"' in text
    assert 'EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"' in text
    assert 'NUM_ROLLOUT="${SLIME_NUM_ROLLOUT:-1}"' in text
    assert 'MAX_RESPONSE_LEN="${SLIME_MAX_RESPONSE_LEN:-128}"' in text
    assert "OPEN_TRAINING_INT4_FAKE_QAT_FLAG" in text
    assert "OPEN_TRAINING_INT4_GROUP_SIZE" in text
    assert "vram_usage.csv" in text
    assert "vram_peak.txt" in text
    assert "--query-gpu=timestamp,index,name,memory.used,memory.total" in text
    assert "--ci-test" in text


def test_moonlight_smoke_launcher_configures_optional_wandb_diagnostics() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"' in text
    assert "WANDB_ALREADY_LOGGED_IN" in text
    assert '${HOME}/.config/wandb/settings' in text
    assert "--wandb-project" in text
    assert "--wandb-group" in text
    assert "--disable-wandb-random-suffix" in text
    assert "--wandb-run-id" in text
    assert "SLIME_WANDB_RUN_ID" in text
    assert '"${WANDB_ARGS[@]}"' in text


def test_moonlight_smoke_launcher_has_no_external_task_dependencies() -> None:
    text = LAUNCHER.read_text(encoding="utf-8").lower()
    executable_text = "\n".join(
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    )

    assert "e2b" not in executable_text
    assert "dapo" not in executable_text
    assert "hf download" not in executable_text
    assert "custom-generate-function-path" not in executable_text


def test_moonlight_smoke_data_is_tiny_local_math_jsonl() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 4
    assert all(sorted(row) == ["label", "prompt"] for row in rows)
    assert {row["label"] for row in rows} == {"7", "30", "42", "45"}


def test_moonlight_smoke_readme_documents_prereqs_and_run_command() -> None:
    text = README.read_text(encoding="utf-8")

    assert "4x A100 80 GB" in text
    assert "SLIME_HF_CHECKPOINT" in text
    assert "SLIME_REF_LOAD_DIR" in text
    assert "SLIME_CONVERT_IF_MISSING=1" in text
    assert "SLIME_WANDB_PROJECT" in text
    assert "SLIME_WANDB_RUN_ID" in text
    assert "vram_peak.txt" in text
    assert "run_moonlight_16b_a3b_int4_smoke.sh" in text
