from __future__ import annotations

import json
from pathlib import Path

from w8_biayn.cpp_perf.schema import CppTask, ReferencePerformance, TestCase, TestCoverage
from w8_biayn.cpp_perf.slime_dataset import build_slime_datasets


def _write_task(root: Path, *, task_id: str, split: str) -> None:
    task = CppTask(
        task_id=task_id,
        problem_id=f"problem-{task_id}",
        prompt_code="#include <bits/stdc++.h>\nint main(){int x; std::cin>>x; std::cout<<x+1<<'\\n';}\n",
        unit_tests=[TestCase(input="1\n", expected="2\n")],
        hidden_tests=[TestCase(input="2\n", expected="3\n")],
        oracle_solution=(
            "#include <bits/stdc++.h>\n"
            "int main(){int x; std::cin>>x; std::cout<<x+1<<'\\n';} // ORACLE_FAST\n"
        ),
        test_coverage=TestCoverage(line=0.95, branch=0.85),
        reference=ReferencePerformance(value=100),
        split=split,  # type: ignore[arg-type]
    )
    task.write_json(root / split / f"{task_id}.json")


def test_build_slime_datasets_writes_prompt_label_metadata_bundle(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _write_task(tasks, task_id="train-1", split="train")
    _write_task(tasks, task_id="validation-1", split="validation")

    out = tmp_path / "slime"
    written = build_slime_datasets(tasks, out, profile="unit", run_id="run-test")

    assert written["train"] == out / "train.jsonl"
    train_row = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    validation_row = json.loads((out / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert train_row["label"] == "train-1"
    assert train_row["metadata"]["task_path"] == "tasks/train/train-1.json"
    assert train_row["metadata"]["hidden_test_count"] == 1
    assert "Visible tests:" in train_row["prompt"]
    assert "ORACLE_FAST" not in train_row["prompt"]
    assert "Input:\n2\nExpected output:\n3" not in train_row["prompt"]
    assert validation_row["label"] == "validation-1"
    assert (out / train_row["metadata"]["task_path"]).exists()

    manifest = json.loads((out / "_w8_data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "slime-dataset"
    assert manifest["options"]["counts"]["train"] == 1
    assert manifest["options"]["counts"]["validation_effective"] == 1
