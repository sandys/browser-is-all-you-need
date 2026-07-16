from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from glm47_posttraining.cpp_perf.schema import CppTask, HarnessResult, ReferencePerformance, TestCase, TestCoverage
from glm47_posttraining.cpp_perf.reward import RewardBreakdown
import glm47_posttraining.integrations.miles_cpp_perf as miles_cpp_perf_module
from glm47_posttraining.integrations.miles_cpp_perf import (
    build_miles_cpp_perf_datasets,
    record_from_debug_sample,
    reward_func,
    reward_record_from_breakdown,
    write_comparison_artifact,
    write_eval_artifacts,
)
def sample_task(task_id: str, split: str) -> CppTask:
    return CppTask(
        task_id=task_id,
        problem_id=f"problem-{task_id}",
        prompt_code="#include <bits/stdc++.h>\nint main(){int x; std::cin>>x; std::cout<<x<<'\\n';}\n",
        unit_tests=[TestCase(input="1\n", expected="1\n")],
        hidden_tests=[TestCase(input="2\n", expected="2\n")],
        oracle_solution="#include <bits/stdc++.h>\nint main(){int x; std::cin>>x; std::cout<<x<<'\\n';}\n",
        test_coverage=TestCoverage(line=0.96, branch=0.86),
        reference=ReferencePerformance(value=1000, gem5_cycles=2000),
        split=split,  # type: ignore[arg-type]
    )


def test_build_miles_cpp_perf_datasets_writes_train_and_eval_shapes(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    sample_task("train-1", "train").write_json(tasks / "train" / "train-1.json")
    sample_task("validation-1", "validation").write_json(tasks / "validation" / "validation-1.json")

    paths = build_miles_cpp_perf_datasets(tasks, tmp_path / "miles", profile="unit")

    sft_rows = _read_jsonl(paths["sft_train"])
    grpo_rows = _read_jsonl(paths["grpo_train"])
    eval_rows = _read_jsonl(paths["eval"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert manifest["counts"] == {"copied_tasks": 2, "eval": 1, "train": 1}
    assert sft_rows[0]["messages"][0]["role"] == "user"
    assert sft_rows[0]["messages"][1]["role"] == "assistant"
    assert "<reasoning>" in sft_rows[0]["messages"][1]["content"]
    assert grpo_rows[0]["prompt"].startswith("Optimize the following C++20 program")
    assert eval_rows[0]["metadata"]["task_path"] == "tasks/validation/validation-1.json"
    assert eval_rows[0]["metadata"]["subset"] == "eval"
    assert manifest["sort_by_size"] is False


def test_build_miles_cpp_perf_datasets_can_sort_train_rows_by_size(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    long_task = sample_task("train-long", "train")
    long_task.prompt_code = long_task.prompt_code * 12
    sample_task("train-short", "train").write_json(tasks / "train" / "train-short.json")
    long_task.write_json(tasks / "train" / "train-long.json")
    sample_task("validation-1", "validation").write_json(tasks / "validation" / "validation-1.json")

    paths = build_miles_cpp_perf_datasets(
        tasks,
        tmp_path / "miles",
        train_limit=1,
        sort_by_size=True,
        profile="unit",
    )

    sft_rows = _read_jsonl(paths["sft_train"])
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert sft_rows[0]["task_id"] == "train-short"
    assert manifest["sort_by_size"] is True


def test_build_miles_cpp_perf_datasets_can_keep_only_full_mark_oracles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = tmp_path / "tasks"
    sample_task("train-good", "train").write_json(tasks / "train" / "train-good.json")
    sample_task("train-bad", "train").write_json(tasks / "train" / "train-bad.json")
    sample_task("validation-1", "validation").write_json(tasks / "validation" / "validation-1.json")

    def fake_compute_reward(task, model_output, *, runner=None):
        assert "<reasoning>" in model_output
        if task.task_id == "train-good":
            return RewardBreakdown(
                reward=1.5,
                reason="correct",
                harness=HarnessResult(tests_passed=2, tests_total=2, runtime_cpu_ns=8, reference_runtime_cpu_ns=10),
                format_valid=True,
            )
        return RewardBreakdown(
            reward=-0.1,
            reason="tests_failed",
            harness=HarnessResult(tests_passed=1, tests_total=2, runtime_cpu_ns=12, reference_runtime_cpu_ns=10),
            format_valid=True,
        )

    monkeypatch.setattr(miles_cpp_perf_module, "compute_reward", fake_compute_reward)

    paths = build_miles_cpp_perf_datasets(
        tasks,
        tmp_path / "miles",
        profile="unit",
        filter_train_oracle_full_marks=True,
        oracle_filter_workers=1,
    )

    sft_rows = _read_jsonl(paths["sft_train"])
    grpo_rows = _read_jsonl(paths["grpo_train"])
    oracle_filter_rows = _read_jsonl(paths["oracle_filter"])
    oracle_keep_task_ids = json.loads(paths["oracle_keep_task_ids"].read_text(encoding="utf-8"))
    oracle_drop_task_ids = json.loads(paths["oracle_drop_task_ids"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    assert [row["task_id"] for row in sft_rows] == ["train-good"]
    assert [row["task_id"] for row in grpo_rows] == ["train-good"]
    assert {row["task_id"]: row["keep"] for row in oracle_filter_rows} == {"train-good": True, "train-bad": False}
    assert oracle_keep_task_ids == ["train-good"]
    assert oracle_drop_task_ids == ["train-bad"]
    assert manifest["counts"] == {"copied_tasks": 2, "eval": 1, "train": 1}
    assert manifest["filter_train_oracle_full_marks"] is True
    assert manifest["oracle_filter"]["scored"] == 2
    assert manifest["oracle_filter"]["kept"] == 1
    assert manifest["oracle_filter"]["reason_counts"] == {"correct": 1, "tests_failed": 1}
    assert manifest["files"]["oracle_keep_task_ids"] == "oracle_filter/keep_task_ids.json"
    assert manifest["files"]["oracle_drop_task_ids"] == "oracle_filter/drop_task_ids.json"


def test_reward_func_invalid_format_returns_score_dict_without_running_sandbox(tmp_path: Path, monkeypatch) -> None:
    task_path = sample_task("validation-1", "validation").write_json(tmp_path / "tasks" / "task.json")
    monkeypatch.setenv("GLM47_DATA_DIR", str(tmp_path))
    sample = SimpleNamespace(
        index=3,
        rollout_id=7,
        response="not a formatted C++ answer",
        metadata={
            "task_id": "validation-1",
            "problem_id": "problem-validation-1",
            "split": "validation",
            "task_path": str(task_path.relative_to(tmp_path)),
        },
    )

    reward = asyncio.run(reward_func(SimpleNamespace(), sample))

    assert reward["score"] == -1.0
    assert reward["reward"] == -1.0
    assert reward["reason"] == "invalid_format"
    assert reward["task_id"] == "validation-1"
    assert reward["sample_index"] == 3


def test_record_from_debug_sample_uses_reward_payload_and_label() -> None:
    sample = {
        "index": 1,
        "rollout_id": 0,
        "response": "<reasoning>x</reasoning>\n```cpp\nint main(){return 0;}\n```\n",
        "metadata": {"task_id": "t1", "problem_id": "p1", "split": "validation"},
        "reward": {
            "score": 1.5,
            "reward": 1.5,
            "reason": "correct",
            "task_id": "t1",
            "problem_id": "p1",
            "split": "validation",
            "tests_passed": 2,
            "tests_total": 2,
            "runtime_cpu_ns": 50,
            "reference_runtime_cpu_ns": 100,
        },
    }

    record = record_from_debug_sample(sample, label="base")

    assert record["label"] == "base"
    assert record["reward"] == 1.5
    assert record["all_tests_pass"] is True
    assert record["response"].startswith("<reasoning>")


def test_write_eval_artifacts_and_comparison_from_json_samples(tmp_path: Path) -> None:
    samples_path = tmp_path / "base.samples.jsonl"
    samples_path.write_text(
        json.dumps(
            {
                "index": 0,
                "rollout_id": 0,
                "metadata": {"task_id": "t1", "problem_id": "p1", "split": "validation"},
                "response": "bad",
                "reward": {
                    "score": -1.0,
                    "reward": -1.0,
                    "reason": "invalid_format",
                    "task_id": "t1",
                    "problem_id": "p1",
                    "split": "validation",
                    "tests_passed": 0,
                    "tests_total": 0,
                    "all_tests_pass": False,
                    "runtime_cpu_ns": None,
                    "reference_runtime_cpu_ns": 100,
                },
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "index": 1,
                "rollout_id": 0,
                "metadata": {"task_id": "t1", "problem_id": "p1", "split": "validation"},
                "response": "good",
                "reward": {
                    "score": 1.5,
                    "reward": 1.5,
                    "reason": "correct",
                    "task_id": "t1",
                    "problem_id": "p1",
                    "split": "validation",
                    "tests_passed": 2,
                    "tests_total": 2,
                    "all_tests_pass": True,
                    "runtime_cpu_ns": 50,
                    "reference_runtime_cpu_ns": 100,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = write_eval_artifacts(label="base", debug_samples_path=samples_path, output_dir=tmp_path / "eval")
    summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))
    records = _read_jsonl(artifacts["records"])

    assert len(records) == 2
    assert summary["task_count"] == 1
    assert summary["correct_and_faster_rate"] == 1.0
    comparison_path = write_comparison_artifact([artifacts["summary"]], tmp_path / "eval" / "comparison.json")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["uplift_gate"]["missing_required_labels"] == ["sft", "grpo"]


def test_reward_record_from_breakdown_carries_cxx_metrics() -> None:
    sample = SimpleNamespace(index=0, rollout_id=0, response="ok", metadata={})
    task = sample_task("validation-1", "validation")
    breakdown = SimpleNamespace(
        reward=1.5,
        reason="correct",
        code="int main(){return 0;}\n",
        format_valid=True,
        harness=HarnessResult(
            tests_passed=2,
            tests_total=2,
            runtime_cpu_ns=50,
            reference_runtime_cpu_ns=100,
        ),
    )

    record = reward_record_from_breakdown(sample, task, breakdown)  # type: ignore[arg-type]

    assert record["score"] == 1.5
    assert record["all_tests_pass"] is True
    assert record["runtime_speedup"] is None
    assert record["candidate_bytes"] > 0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
