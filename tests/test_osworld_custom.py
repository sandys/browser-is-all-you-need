from __future__ import annotations

import json
from pathlib import Path

from w8_biayn.osworld_custom import harness, registry, validate


def test_iter_custom_tasks_finds_imported_tasks():
    tasks = registry.iter_custom_tasks()

    assert len(tasks) >= 100
    assert any(task.name == 'add_todo_comment' for task in tasks)


def test_load_task_returns_expected_metadata():
    task = registry.load_task('src/w8_biayn/osworld_custom/tasks/add_todo_comment')

    assert task.name == 'add_todo_comment'
    assert task.domain == 'vscode'
    assert task.snapshot == 'vscode'
    assert task.source == 'custom/w8-biayn'
    assert task.instruction.startswith('Add the comment # TODO: implement')


def test_load_taskset_supports_domain_mapping(tmp_path: Path):
    sample = registry.iter_custom_tasks()[:2]
    taskset_path = tmp_path / 'taskset.json'
    taskset_path.write_text(
        json.dumps({'tasks': [{'domain': sample[0].domain, 'task_id': sample[0].task_id}, {'domain': sample[1].domain, 'task_id': sample[1].task_id}]}),
        encoding='utf-8',
    )

    loaded = registry.load_taskset(taskset_path)

    assert [task.task_id for task in loaded] == [sample[0].task_id, sample[1].task_id]


def test_validate_task_paths_accepts_representative_task():
    report = validate.validate_task_paths(['src/w8_biayn/osworld_custom/tasks/add_todo_comment'])

    assert report.ok is True
    assert report.errors == []


def test_validate_taskset_reports_bad_proxy_without_allow_proxy(tmp_path: Path):
    task_dir = tmp_path / 'proxy-task'
    task_dir.mkdir()
    (task_dir / 'task.json').write_text(json.dumps({
        'id': 'proxy-task',
        'snapshot': 'chrome',
        'instruction': 'Do a thing.',
        'source': 'custom',
        'config': [],
        'trajectory': 'trajectories/',
        'related_apps': ['chrome'],
        'evaluator': {
            'func': 'check_include_exclude',
            'result': {'type': 'vm_command_line', 'command': 'echo Evaluation successful.'},
            'expected': {'type': 'rule', 'rules': {'include': ['Evaluation successful.'], 'exclude': ['Evaluation failed.']}},
        },
        'proxy': True,
        'fixed_ip': False,
    }), encoding='utf-8')

    report = validate.validate_task_paths([task_dir])

    assert report.ok is False
    assert any('proxy is true' in issue.message for issue in report.errors)


def test_resolve_env_tasks_supports_task_paths_and_limit():
    tasks = harness.resolve_env_tasks(task_paths=['src/w8_biayn/osworld_custom/tasks/add_todo_comment', 'src/w8_biayn/osworld_custom/tasks/python_file_manifest'], limit=1)

    assert len(tasks) == 1
    assert tasks[0].name == 'add_todo_comment'
