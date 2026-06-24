from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .registry import CustomTask, load_task, load_taskset, repo_path
from .validate import validate_tasks

STATE_DIR = Path('.w8-biayn/osworld-custom')
ENV_RUNS_DIR = STATE_DIR / 'env-runs'


@dataclass(frozen=True)
class HarnessConfig:
    provider_name: str = 'docker'
    headless: bool = True
    screen_width: int = 1920
    screen_height: int = 1080
    os_type: str = 'Ubuntu'
    osworld_path: Path | None = None
    cache_dir: Path | None = None
    path_to_vm: str | None = None
    upstream_python: str = '3.12'
    action_format: str = 'pyautogui'
    max_steps: int = 1
    step_pause: float = 0.5
    cleanup_containers: bool = True
    cleanup_scope: str = 'created'
    remove_containers: bool = True
    artifact_root: Path = field(default_factory=lambda: ENV_RUNS_DIR)

    def as_worker_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data['osworld_path'] = str(_resolve_osworld_path(self.osworld_path))
        data['cache_dir'] = str(self.cache_dir) if self.cache_dir is not None else None
        data['artifact_root'] = str(self.artifact_root)
        return data


def _path_or_none(value: str | None) -> Path | None:
    return Path(value) if value else None


def config_from_dict(data: dict[str, Any]) -> HarnessConfig:
    return HarnessConfig(
        provider_name=data.get('provider_name', 'docker'),
        headless=bool(data.get('headless', True)),
        screen_width=int(data.get('screen_width', 1920)),
        screen_height=int(data.get('screen_height', 1080)),
        os_type=data.get('os_type', 'Ubuntu'),
        osworld_path=_path_or_none(data.get('osworld_path')),
        cache_dir=_path_or_none(data.get('cache_dir')),
        path_to_vm=data.get('path_to_vm'),
        upstream_python=data.get('upstream_python', '3.12'),
        action_format=data.get('action_format', 'pyautogui'),
        max_steps=int(data.get('max_steps', 1)),
        step_pause=float(data.get('step_pause', 0.5)),
        cleanup_containers=bool(data.get('cleanup_containers', True)),
        cleanup_scope=data.get('cleanup_scope', 'created'),
        remove_containers=bool(data.get('remove_containers', True)),
        artifact_root=Path(data.get('artifact_root') or ENV_RUNS_DIR),
    )


def _resolve_osworld_path(path: Path | None) -> Path:
    if path is not None:
        return path
    from w8_biayn import osworld
    return osworld.upstream_path()


def _uv_bin() -> str:
    found = shutil.which('uv')
    if found:
        return found
    local = Path.home() / '.local' / 'bin' / 'uv'
    if local.exists():
        return str(local)
    return 'uv'


def import_desktop_env(osworld_path: Path):
    if osworld_path.exists():
        sys.path.insert(0, str(osworld_path))
    try:
        module = importlib.import_module('desktop_env.desktop_env')
    except ModuleNotFoundError as exc:
        raise RuntimeError('could not import desktop_env.desktop_env; clone/install OSWorld or pass --osworld-path') from exc
    return module.DesktopEnv


def desktop_env_kwargs(config: HarnessConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'provider_name': config.provider_name,
        'action_space': 'pyautogui',
        'screen_size': (config.screen_width, config.screen_height),
        'headless': config.headless,
        'os_type': config.os_type,
        'require_a11y_tree': False,
    }
    if config.cache_dir is not None:
        kwargs['cache_dir'] = str(config.cache_dir)
    if config.path_to_vm:
        kwargs['path_to_vm'] = config.path_to_vm
    return kwargs


def create_desktop_env(DesktopEnv, config: HarnessConfig):
    kwargs = desktop_env_kwargs(config)
    try:
        return DesktopEnv(**kwargs)
    except TypeError:
        minimal = {
            'provider_name': kwargs['provider_name'],
            'action_space': kwargs['action_space'],
            'screen_size': kwargs['screen_size'],
            'headless': kwargs['headless'],
        }
        return DesktopEnv(**minimal)


def maybe_call(obj: Any, name: str) -> None:
    fn = getattr(obj, name, None)
    if callable(fn):
        with contextlib.suppress(Exception):
            fn()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {'bytes': len(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return repr(value)


def _save_screenshot(obs: Any, path: Path) -> str | None:
    if not isinstance(obs, dict):
        return None
    screenshot = obs.get('screenshot')
    if screenshot is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(screenshot, (bytes, bytearray)):
        path.write_bytes(bytes(screenshot))
        return repo_path(path)
    save = getattr(screenshot, 'save', None)
    if callable(save):
        save(path)
        return repo_path(path)
    return None


def _step_env(env: Any, action: str, pause: float):
    try:
        return env.step(action, pause=pause)
    except TypeError:
        return env.step(action)


def _container_ids() -> set[str]:
    from w8_biayn.osworld import clean_interrupted_docker_runs
    _ = clean_interrupted_docker_runs  # imported for parity / future use
    return set()


def _cleanup_record(config: HarnessConfig, before_containers: set[str]) -> dict[str, Any]:
    return {
        'enabled': config.cleanup_containers,
        'scope': config.cleanup_scope,
        'remove': config.remove_containers,
        'before_count': len(before_containers),
        'after_count': len(before_containers),
        'target_ids': [],
        'cleaned': [],
        'error': None,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def make_run_id(prefix: str) -> str:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    return f'{prefix}-{stamp}'


class OSWorldSession:
    def __init__(self, env: Any, task: CustomTask, artifact_dir: Path, *, action_format: str = 'pyautogui', max_steps: int = 1, step_pause: float = 0.5) -> None:
        self.env = env
        self.task = task
        self.artifact_dir = artifact_dir
        self.action_format = action_format
        self.max_steps = max_steps
        self.step_pause = step_pause
        self.steps: list[dict[str, Any]] = []
        self.reset_ok = False
        self.done = False
        self.last_obs: Any = None

    def reset(self) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.artifact_dir / 'task.json', self.task.data)
        self.last_obs = self.env.reset(self.task.data)
        self.reset_ok = True
        screenshot_path = _save_screenshot(self.last_obs, self.artifact_dir / 'step-000-reset.png')
        observation = {
            'task_id': self.task.task_id,
            'domain': self.task.domain,
            'screenshot_path': screenshot_path,
            'observation_keys': sorted(self.last_obs.keys()) if isinstance(self.last_obs, dict) else [],
        }
        write_json(self.artifact_dir / 'reset.json', observation)
        maybe_call(self.env, 'pause')
        return observation

    def step(self, model_response: str) -> dict[str, Any]:
        step_number = len(self.steps) + 1
        record: dict[str, Any] = {
            'step': step_number,
            'raw_response': model_response,
            'execution_ok': False,
            'env_reward': None,
            'env_done': False,
            'done': False,
            'format_reward': 0.0,
            'info': None,
            'screenshot_path': None,
            'error': None,
            'executed_actions': [],
        }
        try:
            maybe_call(self.env, 'unpause')
            obs, reward, env_done, info = _step_env(self.env, model_response, self.step_pause)
            self.last_obs = obs
            record['executed_actions'].append(str(model_response))
            record['env_reward'] = _json_safe(reward)
            record['env_done'] = bool(env_done)
            record['info'] = _json_safe(info)
            record['screenshot_path'] = _save_screenshot(obs, self.artifact_dir / f'step-{step_number:03d}.png')
            record['execution_ok'] = True
            record['done'] = bool(env_done) or step_number >= self.max_steps
            self.done = bool(record['done'])
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            record['traceback'] = traceback.format_exc()
            record['done'] = True
            self.done = True
        finally:
            maybe_call(self.env, 'pause')

        self.steps.append(record)
        write_json(self.artifact_dir / f'step-{step_number:03d}.json', record)
        return record

    def evaluate(self) -> float:
        maybe_call(self.env, 'unpause')
        score_raw = self.env.evaluate()
        try:
            score = float(score_raw)
        except Exception as exc:
            raise RuntimeError(f'evaluator returned non-numeric score: {score_raw!r}') from exc
        write_json(self.artifact_dir / 'evaluator_result.json', {'score': score})
        return score


def _reward_summary(eval_score: float | None, steps: list[dict[str, Any]]) -> dict[str, Any]:
    format_reward = min((float(step.get('format_reward', 0.0)) for step in steps), default=0.0)
    return {
        'format_reward': format_reward,
        'eval_reward': eval_score,
        'total_reward': (eval_score or 0.0) + format_reward,
    }


def run_episode_in_process(env: Any, task: CustomTask, actions: list[str], artifact_dir: Path, config: HarnessConfig) -> dict[str, Any]:
    session = OSWorldSession(env, task, artifact_dir, action_format=config.action_format, max_steps=config.max_steps, step_pause=config.step_pause)
    eval_score: float | None = None
    error: str | None = None
    reset_observation: dict[str, Any] | None = None
    try:
        reset_observation = session.reset()
        for action in actions[: config.max_steps]:
            step = session.step(action)
            if step.get('done'):
                break
        eval_score = session.evaluate()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        (artifact_dir / 'error.txt').write_text(error + '\n' + traceback.format_exc(), encoding='utf-8')

    reward = _reward_summary(eval_score, session.steps)
    result = {
        'domain': task.domain,
        'task_id': task.task_id,
        'task_config_path': repo_path(task.path),
        'artifact_dir': repo_path(artifact_dir),
        'reset_ok': session.reset_ok,
        'evaluate_ok': eval_score is not None,
        'steps': len(session.steps),
        'done': session.done,
        'eval_score': eval_score,
        'env_error': error,
        'ok': error is None and session.reset_ok and eval_score is not None and all(step.get('execution_ok') for step in session.steps),
        'reset_observation': reset_observation,
        'step_records': session.steps,
        **reward,
    }
    write_json(artifact_dir / 'episode_result.json', result)
    return result


def resolve_env_tasks(*, taskset: str | Path | None = None, task_paths: list[str | Path] | None = None, task_id: str | None = None, limit: int | None = None) -> list[CustomTask]:
    tasks: list[CustomTask] = []
    if taskset:
        tasks.extend(load_taskset(taskset))
    for path in task_paths or []:
        tasks.append(load_task(path))
    if task_id:
        tasks = [task for task in tasks if task.task_id == task_id]
    if not tasks:
        raise ValueError('provide --taskset or task path(s)')
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def run_env_smoke(tasks: list[CustomTask], config: HarnessConfig, *, actions: list[str] | None = None, prefix: str = 'env-smoke') -> dict[str, Any]:
    validation = validate_tasks(tasks)
    run_dir = config.artifact_root / make_run_id(prefix)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / 'harness_config.json', config.as_worker_dict())
    write_json(run_dir / 'validation.json', validation.as_dict())
    action_list = actions or ['WAIT']

    if validation.errors:
        summary = _summary(run_dir, validation, [], action_list, worker_ok=False)
        write_json(run_dir / 'summary.json', summary)
        write_jsonl(run_dir / 'results.jsonl', [])
        return summary

    rows = _run_worker_subprocess(tasks, config, run_dir, action_list)
    summary = _summary(run_dir, validation, rows, action_list, worker_ok=True)
    write_json(run_dir / 'summary.json', summary)
    write_jsonl(run_dir / 'results.jsonl', rows)
    return summary


def _run_worker_subprocess(tasks: list[CustomTask], config: HarnessConfig, run_dir: Path, actions: list[str]) -> list[dict[str, Any]]:
    osworld_path = _resolve_osworld_path(config.osworld_path)
    if not osworld_path.exists():
        raise FileNotFoundError(f'OSWorld upstream checkout not found: {osworld_path}')

    input_path = run_dir / 'worker_input.json'
    output_path = run_dir / 'worker_results.json'
    write_json(input_path, {
        'run_dir': str(run_dir),
        'config': config.as_worker_dict(),
        'actions': actions,
        'tasks': [
            {'domain': task.domain, 'task_config_path': str(task.path)}
            for task in tasks
        ],
    })

    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    cmd = [_uv_bin(), 'run', '--python', config.upstream_python, 'python', '-m', 'w8_biayn.osworld_custom.worker', str(input_path), str(output_path)]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (run_dir / 'worker_stdout.log').write_text(proc.stdout, encoding='utf-8')
    (run_dir / 'worker_stderr.log').write_text(proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f"env worker failed with exit code {proc.returncode}; see {repo_path(run_dir / 'worker_stderr.log')}")
    return json.loads(output_path.read_text(encoding='utf-8'))


def _summary(run_dir: Path, validation, rows: list[dict[str, Any]], actions: list[str], *, worker_ok: bool) -> dict[str, Any]:
    passed = sum(1 for row in rows if row.get('ok'))
    failed = len(rows) - passed
    return {
        'run_dir': repo_path(run_dir),
        'task_count': len(validation.tasks),
        'validated': validation.ok,
        'validation_errors': len(validation.errors),
        'validation_warnings': len(validation.warnings),
        'episodes': len(rows),
        'passed': passed,
        'failed': failed,
        'worker_ok': worker_ok,
        'ok': validation.ok and worker_ok and len(rows) == len(validation.tasks) and failed == 0,
        'actions': actions,
        'results': rows,
    }
