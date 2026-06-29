from __future__ import annotations

import pytest

from w8_biayn.mlflow_tracking import parse_mlflow_tags


def test_parse_mlflow_tags_parses_repeated_key_values():
    assert parse_mlflow_tags(["model=qwen", "run=smoke", "seed=001"]) == (
        ("model", "qwen"),
        ("run", "smoke"),
        ("seed", "001"),
    )


def test_parse_mlflow_tags_accepts_empty_value():
    assert parse_mlflow_tags(["run="]) == (("run", ""),)


def test_parse_mlflow_tags_rejects_invalid_format():
    with pytest.raises(ValueError, match="key=value"):
        parse_mlflow_tags(["invalid-tag"])


def test_parse_mlflow_tags_rejects_empty_key():
    with pytest.raises(ValueError, match="non-empty key"):
        parse_mlflow_tags(["=missing"])
