from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from w8_biayn.cpp_perf.data import (
    build_tests_manifest_from_io,
    inspect_supercoder_parquet,
    verify_data_manifest,
)
from w8_biayn.cpp_perf.schema import CppTask, ReferencePerformance, TestCase, TestCoverage
from w8_biayn.cpp_perf.skyrl_dataset import build_skyrl_datasets, build_prompt


def sample_task(task_id: str, split: str) -> CppTask:
    return CppTask(
        task_id=task_id,
        problem_id=f"p_{task_id}",
        prompt_code="#include <iostream>\nint main(){int n; std::cin>>n; std::cout<<n<<\"\\n\";}\n",
        unit_tests=[TestCase(input="1\n", expected="1\n")],
        hidden_tests=[TestCase(input="2\n", expected="2\n")],
        oracle_solution="#include <iostream>\nint main(){int n; std::cin>>n; std::cout<<n<<\"\\n\";}\n",
        test_coverage=TestCoverage(line=0.96, branch=0.86),
        reference=ReferencePerformance(value=100),
        split=split,  # type: ignore[arg-type]
    )


def test_tests_manifest_from_input_output_dirs(tmp_path):
    problem = tmp_path / "cases" / "p1"
    problem.mkdir(parents=True)
    (problem / "input.0.txt").write_text("1\n", encoding="utf-8")
    (problem / "output.0.txt").write_text("1\n", encoding="utf-8")
    (problem / "input.1.txt").write_text("2\n", encoding="utf-8")
    (problem / "output.1.txt").write_text("2\n", encoding="utf-8")

    manifest = build_tests_manifest_from_io(
        ["p1", "missing"],
        tmp_path / "cases",
        {"p1": {"line": 0.96, "branch": 0.86}},
    )

    assert list(manifest) == ["p1"]
    assert manifest["p1"]["unit_tests"] == [{"input": "1\n", "expected": "1\n"}]
    assert manifest["p1"]["hidden_tests"] == [{"input": "2\n", "expected": "2\n"}]


def test_build_skyrl_datasets_writes_grpo_sft_and_manifest(tmp_path):
    tasks = tmp_path / "tasks"
    sample_task("train_1", "train").write_json(tasks / "train_1.json")
    sample_task("val_1", "validation").write_json(tasks / "val_1.json")

    out = tmp_path / "skyrl"
    written = build_skyrl_datasets(tasks, out)

    assert written["grpo_train"].exists()
    assert written["grpo_validation"].exists()
    assert written["sft_train"].exists()
    assert written["sft_validation"].exists()
    assert verify_data_manifest(out) == []

    row = pq.read_table(written["grpo_train"]).to_pylist()[0]
    assert row["env_class"] == "cpp-perf"
    assert row["extra_info"]["task_path"] == "tasks/train_1.json"
    assert row["extra_info"]["hidden_test_count"] == 1
    assert "oracle_solution" not in json.dumps(row)

    sft_line = json.loads(written["sft_train"].read_text(encoding="utf-8").splitlines()[0])
    assert sft_line["instruction"].startswith("Optimize the following C++20 program")
    assert "<reasoning>" in sft_line["output"]
    assert "```cpp" in sft_line["output"]


def test_build_prompt_exposes_visible_tests_not_hidden_tests():
    task = sample_task("train_1", "train")
    prompt = build_prompt(task)

    assert "Visible tests" in prompt
    assert "1\n" in prompt
    assert "Do not mention or rely on hidden tests" in prompt
    assert "2\nExpected output:\n2" not in prompt


def test_supercoder_inspect_rejects_lfs_pointer(tmp_path):
    pointer = tmp_path / "val.parquet"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Git LFS pointer"):
        inspect_supercoder_parquet(pointer)
