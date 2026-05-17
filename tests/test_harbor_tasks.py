from __future__ import annotations

from w8_biayn.harbor.tasks import (
    DEFAULT_HARBOR_TASK_IDS,
    build_harbor_rows,
    discover_task_ids,
    load_task,
    resolve_task_root,
    validate_tasks,
)
from w8_biayn.harbor.docker_runner import HarborDockerTaskRunner, image_tag_for_task
from w8_biayn.harbor.rubric import evaluate_harbor_rubric


def test_packaged_harbor_tasks_validate():
    task_root = resolve_task_root()
    assert set(discover_task_ids(task_root)) == set(DEFAULT_HARBOR_TASK_IDS)

    tasks = validate_tasks(task_root)
    assert {task.task_id for task in tasks} == set(DEFAULT_HARBOR_TASK_IDS)
    assert all(task.preview_path.startswith("/") for task in tasks)


def test_harbor_rows_have_local_task_paths_without_rllm():
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
