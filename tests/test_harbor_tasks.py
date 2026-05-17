from __future__ import annotations

import pytest

from w8_biayn.harbor.skyrl_dataset import make_harbor_record, prepare_harbor_skyrl_dataset
from w8_biayn.harbor.tasks import (
    DEFAULT_HARBOR_TASK_IDS,
    build_harbor_rows,
    discover_task_ids,
    load_task,
    resolve_task_root,
    validate_tasks,
)
from w8_biayn.harbor.docker_runner import HarborDockerTaskRunner, container_name_for_task, image_tag_for_task
from w8_biayn.harbor.rubric import evaluate_harbor_rubric
from w8_biayn.integrations.harbor_env import _extract_solution_script


def test_packaged_harbor_tasks_validate():
    task_root = resolve_task_root()
    assert set(discover_task_ids(task_root)) == set(DEFAULT_HARBOR_TASK_IDS)

    tasks = validate_tasks(task_root)
    assert {task.task_id for task in tasks} == set(DEFAULT_HARBOR_TASK_IDS)
    assert all(task.preview_path.startswith("/") for task in tasks)


def test_harbor_rows_have_local_task_paths():
    rows = build_harbor_rows(task_ids=[DEFAULT_HARBOR_TASK_IDS[0]])

    assert len(rows) == 1
    row = rows[0]
    assert row.get("task_path") or row.get("metadata", {}).get("task_path")
    assert DEFAULT_HARBOR_TASK_IDS[0] in str(row)


def test_domdiff_core_rubric_scores_passing_summary():
    task = load_task(DEFAULT_HARBOR_TASK_IDS[0])
    evaluation = evaluate_harbor_rubric(
        task.rubric,
        domdiff_summary={
            "structuralSimilarity": 0.95,
            "textSimilarity": 0.95,
            "layoutSimilarity": 0.95,
            "styleSimilarity": 0.95,
        },
        dom_state={},
        signal_metrics={},
    )

    assert evaluation.passed
    assert evaluation.score >= task.rubric["pass_threshold"]


def test_harbor_docker_dry_run_names_build_and_verifier_steps():
    task = load_task(DEFAULT_HARBOR_TASK_IDS[0])
    runner = HarborDockerTaskRunner(chromiumrl_url="https://reward.trycloudflare.com")
    plan = runner.dry_run_plan(task)

    assert "docker build" in plan
    assert str(task.path / "environment") in plan
    assert "bash /tests/test.sh" in plan
    assert image_tag_for_task(task.path).startswith("w8-biayn-harbor-")


def test_harbor_docker_dry_run_can_use_generated_solution():
    task = load_task(DEFAULT_HARBOR_TASK_IDS[0])
    runner = HarborDockerTaskRunner(chromiumrl_url="https://reward.trycloudflare.com", oracle=False)
    plan = runner.dry_run_plan(task)

    assert "/tmp/w8_solution.sh" in plan
    assert "bash /task/solution/solve.sh" not in plan


def test_harbor_container_names_are_unique_for_parallel_samples():
    names = {container_name_for_task("chakra-ui__chakra-ui-8905") for _ in range(100)}

    assert len(names) == 100
    assert all(name.startswith("w8-harbor-chakra-ui__chakra-ui-8905-") for name in names)
    assert all(len(name) <= 63 for name in names)


def test_harbor_skyrl_record_shape():
    task = load_task(DEFAULT_HARBOR_TASK_IDS[0])
    record = make_harbor_record(task, "train", 0, oracle=True)

    assert record["env_class"] == "harbor-domdiff"
    assert record["task_id"] == DEFAULT_HARBOR_TASK_IDS[0]
    assert record["oracle"] is True
    assert record["prompt"][0]["role"] == "system"
    assert "<solution>" in record["prompt"][0]["content"]


def test_harbor_solution_extraction():
    assert _extract_solution_script("before <solution>\necho ok\n</solution> after") == "echo ok\n"
    assert _extract_solution_script("```bash\necho fenced\n```") == "echo fenced\n"


def test_prepare_harbor_skyrl_dataset(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    train_path, val_path = prepare_harbor_skyrl_dataset(
        tmp_path,
        task_ids=[DEFAULT_HARBOR_TASK_IDS[0]],
        oracle=True,
    )

    assert train_path.exists()
    assert val_path.exists()
    assert (tmp_path / "metadata.json").exists()
