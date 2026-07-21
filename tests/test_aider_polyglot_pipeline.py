from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm47_posttraining.cpp_perf.dataset import (
    ComplexityWireSampler,
    aider_sft_output,
    build_aider_prompt,
)
from glm47_posttraining.cpp_perf.eval import aggregate_aider_eval_records
from glm47_posttraining.cpp_perf.reward import (
    compute_aider_reward,
    compute_multi_try_aider_reward,
    extract_aider_files,
    valid_aider_model_output,
)
from glm47_posttraining.cpp_perf.rubrics import (
    normalize_risks,
    rubric_pass_rates,
    update_risks_ema,
)
from glm47_posttraining.cpp_perf.sandbox import _aider_compile_command, parse_catch2_xml
from glm47_posttraining.cpp_perf.schema import AiderPolyglotTask, Catch2HarnessResult
from glm47_posttraining.integrations.miles_polyglot_cpp import (
    build_miles_polyglot_datasets,
    compute_calibrated_grpo_advantages,
    sample_context_exhausted,
    sample_reasoning_token_count,
    task_with_live_risks,
)
from scripts.build_task_taxonomy import compute_cyclomatic_complexity
from scripts.ingest_exercism_aider import ingest_all
from scripts.update_rubric_risks import update_taxonomy_risks


def sample_task(task_id: str = "allergies", split: str = "train") -> AiderPolyglotTask:
    return AiderPolyglotTask(
        task_id=task_id,
        instructions_md=f"Implement {task_id}.",
        solution_files={
            f"{task_id}.h": "#pragma once\nint answer();\n",
            f"{task_id}.cpp": f'#include "{task_id}.h"\nint answer() {{ return 0; }}\n',
        },
        oracle_files={
            f"{task_id}.h": "#pragma once\nint answer();\n",
            f"{task_id}.cpp": f'#include "{task_id}.h"\nint answer() {{ return 42; }}\n',
        },
        test_files={f"{task_id}_test.cpp": "TEST_CASE(\"answer\") { REQUIRE(answer() == 42); }\n"},
        support_files={"test/catch.hpp": "// catch", "test/tests-main.cpp": "// main"},
        split=split,  # type: ignore[arg-type]
    )


def passing_harness() -> Catch2HarnessResult:
    return Catch2HarnessResult(
        passed_test_cases=2,
        total_test_cases=2,
        passed_assertions=4,
        total_assertions=4,
    )


def test_schema_prompt_and_sft_target_preserve_both_editable_files() -> None:
    task = sample_task()
    prompt = build_aider_prompt(task)
    output = aider_sft_output(task)

    assert "allergies.h" in prompt and "allergies.cpp" in prompt
    assert task.test_files["allergies_test.cpp"] not in prompt
    assert valid_aider_model_output(output, task)
    assert extract_aider_files(output, set(task.solution_files)) == task.oracle_files


@pytest.mark.parametrize(
    "output",
    [
        "```cpp\n// ===== FILE: allergies.cpp =====\nint x;\n```",
        "<reasoning>x</reasoning>\n```cpp\nint x;\n```",
        "<reasoning>x</reasoning>\n```cpp\n// ===== FILE: other.cpp =====\nint x;\n```",
    ],
)
def test_multifile_format_rejects_missing_reasoning_markers_or_files(output: str) -> None:
    assert not valid_aider_model_output(output, sample_task())


def test_reward_boundaries_and_attempt_discount() -> None:
    task = sample_task().model_copy(update={"difficulty_index": 0.8})
    output = aider_sft_output(task)
    correct = compute_aider_reward(task, output, runner=lambda _task, _files: passing_harness())
    partial = compute_aider_reward(
        task,
        output,
        runner=lambda _task, _files: Catch2HarnessResult(
            passed_test_cases=1,
            total_test_cases=2,
            passed_assertions=3,
            total_assertions=4,
        ),
    )
    retry = compute_multi_try_aider_reward(
        task,
        output,
        attempt_index=2,
        runner=lambda _task, _files: passing_harness(),
    )

    assert correct.reason == "correct_all_rubrics_passed"
    assert correct.reward == 0.325
    assert correct.rubric_scores == {
        "correctness": 1.0,
        "reasoning": 0.1,
        "memory_safety": 0.1,
        "thread_safety": 0.0,
        "runtime": 0.0,
        "cpp_quality": 0.1,
    }
    expected_partial_correctness = -0.2 + 0.7 * (0.5 * 0.5 + 0.5 * 0.75)
    assert partial.reward == pytest.approx(
        0.25 * (expected_partial_correctness + 0.1 + 0.1 + 0.1)
    )
    assert correct.rubric_weights == {
        "correctness": 0.25,
        "reasoning": 0.25,
        "memory_safety": 0.25,
        "cpp_quality": 0.25,
    }
    assert retry.reason == "correct_try2"
    assert retry.reward == pytest.approx(correct.reward * 0.8)


def test_rubric_reasoning_budget_context_exhaustion_and_static_antipattern() -> None:
    task = sample_task()
    concise = aider_sft_output(task)
    exhausted = compute_aider_reward(task, concise, context_exhausted=True)
    verbose = compute_aider_reward(
        task,
        concise,
        reasoning_token_count=1001,
        runner=lambda _task, _files: passing_harness(),
    )
    bad_output = concise.replace(
        "int answer() { return 42; }",
        "void copy(std::vector<int> data) {}\nint answer() { return 42; }",
    )
    bad_design = compute_aider_reward(
        task,
        bad_output,
        runner=lambda _task, _files: passing_harness(),
    )

    assert exhausted.reason == "rubric1_context_exhausted" and exhausted.reward == -0.125
    assert verbose.rubric_scores["reasoning"] == -0.5
    assert bad_design.rubric_scores["cpp_quality"] == -0.1
    assert "introduced_heavy_pass_by_value" in bad_design.observations


def test_concurrency_rubric_uses_tsan_evidence_not_mutex_keywords() -> None:
    task = sample_task().model_copy(update={"topic_category": "State & concurrency"})
    output = aider_sft_output(task)
    clean = compute_aider_reward(
        task,
        output,
        runner=lambda _task, _files: passing_harness().model_copy(
            update={"thread_sanitizer_ran": True}
        ),
    )
    race = compute_aider_reward(
        task,
        output,
        runner=lambda _task, _files: passing_harness().model_copy(
            update={"thread_sanitizer_ran": True, "thread_sanitizer_error": True}
        ),
    )
    unavailable = compute_aider_reward(
        task,
        output,
        runner=lambda _task, _files: passing_harness(),
    )

    assert clean.rubric_scores["thread_safety"] == 0.1
    assert clean.reward == 0.28
    assert race.rubric_scores["thread_safety"] == -0.5
    assert race.reason == "rubric5_thread_safety_failed"
    assert unavailable.reason == "correct_thread_sanitizer_not_run"


def test_bank_account_risk_weights_match_documented_example() -> None:
    task = sample_task("bank-account").model_copy(
        update={
            "rubric_category": "state_concurrency",
            "rubric_risks": {
                "correctness": 0.55,
                "thread_safety": 0.30,
                "reasoning": 0.10,
                "memory_safety": 0.05,
            },
        }
    )
    result = compute_aider_reward(
        task,
        aider_sft_output(task),
        runner=lambda _task, _files: passing_harness(),
    )

    assert result.rubric_weights == {
        "correctness": 0.55,
        "reasoning": 0.10,
        "memory_safety": 0.05,
        "thread_safety": 0.30,
        "cpp_quality": 0.0,
    }
    assert result.reward == 0.565


def test_performance_template_uses_runtime_without_thread_rubric() -> None:
    task = sample_task("knapsack").model_copy(
        update={
            "rubric_category": "performance_intensive",
            "rubric_risks": {"correctness": 0.5, "runtime": 0.5},
        }
    )
    result = compute_aider_reward(
        task,
        aider_sft_output(task),
        runner=lambda _task, _files: passing_harness().model_copy(
            update={"runtime_cpu_ns": 50, "reference_runtime_cpu_ns": 100}
        ),
    )

    assert "runtime" in result.active_rubrics
    assert "thread_safety" not in result.active_rubrics
    assert result.rubric_scores["runtime"] == 0.1
    assert result.reward == 0.55


def test_simple_risk_normalization_and_ema_update() -> None:
    assert normalize_risks(
        "standard", {"correctness": 0.35, "reasoning": 0.45, "memory_safety": 0.20}
    ) == {
        "correctness": 0.35,
        "reasoning": 0.45,
        "memory_safety": 0.20,
        "cpp_quality": 0.0,
    }
    updated = update_risks_ema(
        "standard",
        {"correctness": 0.50},
        {"correctness": 0.80},
        alpha=0.8,
    )
    assert updated["correctness"] == pytest.approx(0.44)
    assert rubric_pass_rates(
        [
            {
                "active_rubrics": ["correctness", "reasoning"],
                "rubric_scores": {"correctness": 0.4, "reasoning": 0.1},
            }
        ]
    ) == {"correctness": 0.0, "reasoning": 1.0}


def test_evaluation_epoch_updates_risk_sidecar(tmp_path: Path) -> None:
    taxonomy = tmp_path / "taxonomy.json"
    records = tmp_path / "records.jsonl"
    output = tmp_path / "risks.json"
    taxonomy.write_text(
        json.dumps(
            {
                "allergies": {
                    "rubric_category": "standard",
                    "rubric_risks": {"correctness": 0.5, "reasoning": 0.5},
                }
            }
        ),
        encoding="utf-8",
    )
    records.write_text(
        json.dumps(
            {
                "task_id": "allergies",
                "active_rubrics": ["correctness", "reasoning"],
                "rubric_scores": {"correctness": 1.0, "reasoning": -0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    updated = update_taxonomy_risks(taxonomy, records, output, alpha=0.8)

    assert updated["allergies"]["rubric_risks"]["correctness"] == 0.4
    assert updated["allergies"]["rubric_risks"]["reasoning"] == 0.6
    assert sum(updated["allergies"]["rubric_weights"].values()) == pytest.approx(1.0)


def test_miles_overlays_live_risk_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "risks.json"
    sidecar.write_text(
        json.dumps(
            {
                "allergies": {
                    "rubric_category": "standard",
                    "rubric_risks": {"correctness": 0.35, "reasoning": 0.45},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MILES_CPP_RUBRIC_RISK_FILE", str(sidecar))

    updated = task_with_live_risks(sample_task())

    assert updated.rubric_risks == {"correctness": 0.35, "reasoning": 0.45}


def test_rollout_context_exhaustion_signal_detection() -> None:
    assert sample_context_exhausted({"truncated": True})
    assert sample_context_exhausted({"finish_reason": "length"})
    assert not sample_context_exhausted({"finish_reason": "stop", "truncated": False})
    assert sample_reasoning_token_count({"reasoning_tokens": 321}) == 321
    assert sample_reasoning_token_count({"metadata": {"num_reasoning_tokens": 123}}) == 123
    assert sample_reasoning_token_count({"completion_tokens": 999}) is None


def test_tsan_compile_uses_separate_instrumented_cpp17_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GLM47_CPP_SANDBOX_BACKEND", "local")
    command = _aider_compile_command(
        sample_task(),
        tmp_path,
        output="test_runner_tsan",
        sanitizer=False,
        thread_sanitizer=True,
        image="unused",
        memory="2g",
        timeout_s=180,
    )
    rendered = " ".join(command)

    assert "-O1" in rendered and "-g" in rendered
    assert "-std=c++17" in rendered and "-fsanitize=thread" in rendered
    assert "-pthread" in rendered and "test_runner_tsan" in rendered


def test_catch2_v2_aggregate_xml_parser(tmp_path: Path) -> None:
    xml = tmp_path / "results.xml"
    xml.write_text(
        """<Catch2TestRun><TestGroup><OverallResults successes="7" failures="2" expectedFailures="1"/>"
        "<OverallResultsCases successes="3" failures="1" expectedFailures="1"/>"
        "</TestGroup></Catch2TestRun>""",
        encoding="utf-8",
    )
    assert parse_catch2_xml(xml) == (4, 5, 8, 10)


def test_complexity_ignores_comments_and_string_literals() -> None:
    assert compute_cyclomatic_complexity('/* if while */ "for &&"; if (x || y) {}') == 3


def test_wire_sampler_exact_size_and_expected_allocation() -> None:
    tasks = [sample_task(f"task-{index}") for index in range(4)]
    taxonomy = {
        tasks[0].task_id: {"difficulty_index_D_i": 0.1},
        tasks[1].task_id: {"difficulty_index_D_i": 0.5},
        tasks[2].task_id: {"difficulty_index_D_i": 0.8},
        tasks[3].task_id: {"difficulty_index_D_i": 0.95},
    }
    sampled = ComplexityWireSampler(tasks, taxonomy, seed=7).sample_batch(20)
    counts = {task.task_id: sampled.count(task) for task in tasks}

    assert len(sampled) == 20
    assert counts == {"task-0": 4, "task-1": 7, "task-2": 6, "task-3": 3}


def test_dataset_builder_and_manifest(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    sample_task("train-one", "train").write_json(tasks_dir / "train" / "train-one.json")
    sample_task("eval-one", "validation").write_json(tasks_dir / "validation" / "eval-one.json")
    paths = build_miles_polyglot_datasets(
        tasks_dir,
        tmp_path / "data",
        sampling_strategy="uniform",
        sample_count=3,
    )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in paths["grpo_train"].read_text().splitlines()]

    assert manifest["counts"] == {"eval": 1, "grpo_rows": 3, "train_unique": 1}
    assert len(rows) == 3
    assert rows[0]["metadata"]["data_source"] == "aider-polyglot-cpp-v1"


def test_eval_aggregation_pass_at_k_and_strata() -> None:
    records = [
        {"task_id": "a", "sample_index": 0, "all_tests_pass": False, "difficulty": "Easy", "topic_category": "Text", "format_valid": True},
        {"task_id": "a", "sample_index": 1, "all_tests_pass": True, "difficulty": "Easy", "topic_category": "Text", "format_valid": True, "passed_assertions": 2, "total_assertions": 2},
        {"task_id": "b", "sample_index": 0, "all_tests_pass": True, "difficulty": "Hard", "topic_category": "State", "format_valid": True, "passed_assertions": 1, "total_assertions": 1},
    ]
    summary = aggregate_aider_eval_records(records, label="base")

    assert summary["pass_at_1"] == 0.5
    assert summary["pass_at_2"] == 1.0
    assert summary["assertion_pass_ratio"] == 1.0
    assert summary["stratified_pass_at_2"]["difficulty"] == {"Easy": 1.0, "Hard": 1.0}


def test_eval_aggregation_counts_rubric_context_exhaustion_without_fake_compile_success() -> None:
    summary = aggregate_aider_eval_records(
        [
            {
                "task_id": "a",
                "sample_index": 0,
                "all_tests_pass": False,
                "difficulty": "Hard",
                "topic_category": "State & concurrency",
                "format_valid": False,
                "sandbox_ran": False,
                "reason": "rubric1_context_exhausted",
            }
        ],
        label="base",
    )

    assert summary["context_exhausted_rate"] == 1.0
    assert summary["compile_success_rate"] == 0.0
    assert summary["sanitizer_pass_rate"] == 0.0


def test_calibrated_advantage_supplies_cross_group_signal_for_zero_variance_groups() -> None:
    advantages = compute_calibrated_grpo_advantages(
        [[1.0, 1.0], [0.0, 0.0]],
        [0.2, 0.9],
    )
    assert advantages[0][0] > 0
    assert advantages[1][0] < 0


def test_real_cloned_benchmark_ingests_all_26_tasks(tmp_path: Path) -> None:
    practice = Path(__file__).resolve().parents[1] / "polyglot-benchmark/cpp/exercises/practice"
    if not practice.exists():
        pytest.skip("polyglot-benchmark clone is not present")
    outputs = ingest_all(practice, tmp_path / "ingested", split_mode="curriculum")
    splits = [path.parent.name for path in outputs]
    assert len(outputs) == 26
    assert splits.count("train") == 20
    assert splits.count("validation") == 3
    assert splits.count("test") == 3
    ingested = {path.stem: AiderPolyglotTask.read_json(path) for path in outputs}
    assert ingested["allergies"].effective_rubric_category == "standard"
    assert ingested["bank-account"].effective_rubric_category == "state_concurrency"
    assert ingested["knapsack"].effective_rubric_category == "performance_intensive"
