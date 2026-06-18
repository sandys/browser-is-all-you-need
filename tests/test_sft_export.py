from __future__ import annotations

from pathlib import Path

import pytest

from w8_biayn.integrations.skyrl_sft_export_checkpoint_main import _validate_policy_checkpoint


def write_policy_shards(policy_dir: Path, world_size: int, *, skip: tuple[str, int] | None = None) -> None:
    policy_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(world_size):
        for prefix in ("model", "extra_state"):
            if skip == (prefix, rank):
                continue
            (policy_dir / f"{prefix}_world_size_{world_size}_rank_{rank}.pt").write_text("", encoding="utf-8")


def test_validate_policy_checkpoint_accepts_single_node_shards(tmp_path):
    policy_dir = tmp_path / "single-node-policy"
    write_policy_shards(policy_dir, 8)

    _validate_policy_checkpoint(policy_dir, 8)


def test_validate_policy_checkpoint_accepts_multinode_total_world_size(tmp_path):
    policy_dir = tmp_path / "multinode-policy"
    write_policy_shards(policy_dir, 16)

    _validate_policy_checkpoint(policy_dir, 16)


def test_validate_policy_checkpoint_reports_missing_shard(tmp_path):
    policy_dir = tmp_path / "missing-policy"
    write_policy_shards(policy_dir, 4, skip=("extra_state", 3))

    with pytest.raises(FileNotFoundError) as exc_info:
        _validate_policy_checkpoint(policy_dir, 4)

    message = str(exc_info.value)
    assert "missing 1 expected rank file" in message
    assert "extra_state_world_size_4_rank_3.pt" in message
