from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_jsonl_train_eval.py"
spec = importlib.util.spec_from_file_location("split_jsonl_train_eval", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
splitter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = splitter
spec.loader.exec_module(splitter)


def test_split_jsonl_is_deterministic(tmp_path):
    input_path = tmp_path / "input.jsonl"
    rows = [{"i": i} for i in range(10)]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    train_a = tmp_path / "train-a.jsonl"
    eval_a = tmp_path / "eval-a.jsonl"
    train_b = tmp_path / "train-b.jsonl"
    eval_b = tmp_path / "eval-b.jsonl"

    assert splitter.split_jsonl(input_path, train_a, eval_a, eval_size=3, seed=7) == (7, 3)
    assert splitter.split_jsonl(input_path, train_b, eval_b, eval_size=3, seed=7) == (7, 3)

    assert train_a.read_text() == train_b.read_text()
    assert eval_a.read_text() == eval_b.read_text()
    assert len(train_a.read_text().splitlines()) == 7
    assert len(eval_a.read_text().splitlines()) == 3
