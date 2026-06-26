from __future__ import annotations

from pathlib import Path


def test_multi_agent_launcher_uses_text_only_default() -> None:
    text = Path("examples/slime/multi_agent/run_multi_agent_text.sh").read_text(encoding="utf-8")

    assert "Qwen3-4B" in text
    assert "Qwen3-VL" not in text
    assert "custom-generate-function-path" in text
    assert "examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents" in text
    assert "dapo-math-17k.jsonl" in text
    assert "convert_hf_to_torch_dist.py" in text
    assert "SLIME_CONVERT_NPROC" in text
    assert "--rollout-max-context-len" in text
    assert "--use-rollout-logprobs" in text
    assert "SLIME_WANDB_PROJECT" in text
    assert "SLIME_WANDB_RUN_ID" in text
    assert "SLIME_MAX_TOKENS_PER_GPU:-}" in text
    assert "WANDB_ALREADY_LOGGED_IN" in text
    assert "${HOME}/.netrc" in text
    assert "${HOME}/.config/wandb/settings" in text


def test_multi_agent_readme_documents_smoke_and_moe() -> None:
    text = Path("examples/slime/multi_agent/README.md").read_text(encoding="utf-8")

    assert "scripts/prepare_dapo_math_dataset.py" in text
    assert "--limit 32" in text
    assert "run_multi_agent_text.sh" in text
    assert "Qwen3-30B-A3B" in text
