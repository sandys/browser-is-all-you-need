from __future__ import annotations

from w8_biayn.datasets import make_record, split_env_ids


def test_make_record_uses_rllm_browsergym_shape():
    record = make_record("miniwob", "browsergym/miniwob.click-button", "train", 3)

    assert record["data_source"] == "miniwob"
    assert record["ability"] == "web"
    assert record["reward_model"]["style"] == "rule"
    assert record["extra_info"]["env_id"] == "browsergym/miniwob.click-button"
    assert record["prompt"][0]["role"] == "system"


def test_split_env_ids_is_deterministic_and_non_empty_validation():
    train_a, val_a = split_env_ids(["a", "b", "c", "d"], 0.8, 7)
    train_b, val_b = split_env_ids(["a", "b", "c", "d"], 0.8, 7)

    assert (train_a, val_a) == (train_b, val_b)
    assert train_a
    assert val_a

