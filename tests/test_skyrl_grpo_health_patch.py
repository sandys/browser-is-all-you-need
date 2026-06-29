from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from w8_biayn.integrations.skyrl_grpo_health_patch import PATCH_MARKER, patch_skyrl_grpo_health


SAMPLE_TRAINER = '''import math
from collections import defaultdict

import numpy as np
from loguru import logger


class Trainer:
    async def train(self):
        log_payload = {
            **self.all_metrics,
            **{f"timing/{k}": v for k, v in self.all_timings.items()},
        }
        if self._vllm_metrics_scraper is not None:
            log_payload.update(await self._vllm_metrics_scraper.sample())
        self.tracker.log(log_payload, step=self.global_step, commit=True)

    def compute_advantages_and_returns(self, data):
        token_level_rewards = data["rewards"]
        pad_size = data.metadata.get("pad_size", 0)
        num_samples = len(token_level_rewards)
        return_sums = token_level_rewards.sum(dim=-1)[: num_samples - pad_size]
        avg_response_length = data.metadata["avg_response_length"]
        data = data.to("cpu")
        valid_advantages = torch.masked_select(
            data["advantages"][: num_samples - pad_size, ...], data["response_mask"][: num_samples - pad_size].bool()
        )
        avg_advantages: float = valid_advantages.mean().item()
        avg_advantages_abs: float = valid_advantages.abs().mean().item()

        if "metrics" not in data.metadata:
            data.metadata["metrics"] = {}
        data.metadata["metrics"].update(
            {
                "avg_final_rewards": avg_rewards,
                "avg_response_length": avg_response_length,
                "avg_advantages": avg_advantages,
                "avg_advantages_abs": avg_advantages_abs,
            }
        )

        logger.info(f"avg_final_rewards: {avg_rewards}, avg_response_length: {avg_response_length}")
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": avg_rewards,
                "loss/avg_raw_advantages": avg_advantages,
                "loss/avg_raw_advantages_abs": avg_advantages_abs,
            }
        )
        return data
'''


def test_patch_skyrl_grpo_health_emits_json_and_reward_variance_metrics(tmp_path: Path) -> None:
    trainer_path = tmp_path / "trainer.py"
    trainer_path.write_text(SAMPLE_TRAINER, encoding="utf-8")

    assert patch_skyrl_grpo_health(trainer_path) == "patched"

    patched = trainer_path.read_text(encoding="utf-8")
    compile(patched, str(trainer_path), "exec")
    assert PATCH_MARKER in patched
    assert "import json" in patched
    assert "W8_GRPO_HEALTH " in patched
    assert "w8/reward_group_variance_mean" in patched
    assert "w8/zero_variance_group_fraction" in patched
    assert "w8/zero_advantage_token_fraction" in patched


def test_patch_skyrl_grpo_health_is_idempotent(tmp_path: Path) -> None:
    trainer_path = tmp_path / "trainer.py"
    trainer_path.write_text(SAMPLE_TRAINER, encoding="utf-8")

    assert patch_skyrl_grpo_health(trainer_path) == "patched"
    once = trainer_path.read_text(encoding="utf-8")
    assert patch_skyrl_grpo_health(trainer_path) == "already_patched"
    assert trainer_path.read_text(encoding="utf-8") == once


def test_patch_skyrl_grpo_health_fails_closed_when_target_moves(tmp_path: Path) -> None:
    trainer_path = tmp_path / "trainer.py"
    trainer_path.write_text("import math\n\nclass Trainer:\n    pass\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="patch target not found"):
        patch_skyrl_grpo_health(trainer_path)


def test_patch_skyrl_grpo_health_applies_to_pinned_skyrl_source(tmp_path: Path) -> None:
    upstream = Path(".cache/upstreams/SkyRL/skyrl/train/trainer.py")
    if not upstream.exists():
        pytest.skip("pinned SkyRL checkout is not available")

    trainer_path = tmp_path / "trainer.py"
    shutil.copyfile(upstream, trainer_path)

    assert patch_skyrl_grpo_health(trainer_path) == "patched"
    patched = trainer_path.read_text(encoding="utf-8")
    compile(patched, str(trainer_path), "exec")
    assert PATCH_MARKER in patched
