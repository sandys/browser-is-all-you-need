from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
CUSTOM_TASKS_DIR = Path(__file__).resolve().parent / "tasks"


@dataclass(frozen=True)
class CustomTask:
    domain: str
    task_id: str
    name: str
    path: Path
    data: dict[str, Any]

    @property
    def instruction(self) -> str:
        value = self.data.get("instruction")
        return value.strip() if isinstance(value, str) else ""

    @property
    def snapshot(self) -> str:
        value = self.data.get("snapshot")
        return value.strip() if isinstance(value, str) else ""

    @property
    def source(self) -> str:
        value = self.data.get("source")
        return value.strip() if isinstance(value, str) else "custom"

    def as_row(self, *, max_turns: int = 15, observation_type: str = "screenshot_a11y_tree") -> dict[str, Any]:
        return {
            "env_id": "osworld_custom",
            "domain": self.domain,
            "task_id": self.task_id,
            "name": self.name,
            "source": self.source,
            "snapshot": self.snapshot,
            "instruction": self.instruction,
            "task_config_path": repo_path(self.path),
            "max_turns": max_turns,
            "observation_type": observation_type,
            "screen_width": 1920,
            "screen_height": 1080,
            "tags": ["custom", self.domain],
        }

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["path"] = str(self.path)
        return row


def repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_task_path(path: str | Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    if raw.is_dir():
        raw = raw / "task.json"
    return raw


def infer_domain(data: dict[str, Any], *, fallback: str | None = None) -> str:
    related = data.get("related_apps")
    if isinstance(related, list):
        for item in related:
            if isinstance(item, str) and item.strip():
                return item.strip()
    snapshot = data.get("snapshot")
    if isinstance(snapshot, str) and snapshot.strip():
        return snapshot.strip()
    return fallback or "custom"


def load_task(path: str | Path, *, domain: str | None = None, name: str | None = None) -> CustomTask:
    task_path = normalize_task_path(path)
    if not task_path.exists():
        raise FileNotFoundError(f"task.json not found: {task_path}")
    data = load_json(task_path)
    if not isinstance(data, dict):
        raise ValueError(f"task JSON must be an object: {task_path}")
    task_id = data.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = task_path.parent.name
    return CustomTask(
        domain=domain or infer_domain(data, fallback=task_path.parent.name),
        task_id=task_id.strip(),
        name=name or task_path.parent.name,
        path=task_path,
        data=data,
    )


def iter_custom_tasks(root: Path = CUSTOM_TASKS_DIR) -> list[CustomTask]:
    if not root.exists():
        return []
    return [load_task(path) for path in sorted(root.glob("*/task.json"))]


def index_custom_tasks(root: Path = CUSTOM_TASKS_DIR) -> dict[str, CustomTask]:
    index: dict[str, CustomTask] = {}
    for task in iter_custom_tasks(root):
        if task.task_id in index:
            raise ValueError(f"duplicate custom task id {task.task_id}: {index[task.task_id].path} and {task.path}")
        index[task.task_id] = task
    return index


def _task_from_domain_id(domain: str, task_id: str, index: dict[str, CustomTask]) -> CustomTask:
    if task_id not in index:
        raise ValueError(f"taskset references unknown custom task id {domain}/{task_id}")
    task = index[task_id]
    return CustomTask(domain=domain, task_id=task.task_id, name=task.name, path=task.path, data=task.data)


def _task_from_item(item: Any, idx: int, index: dict[str, CustomTask]) -> CustomTask:
    if isinstance(item, str):
        if "/" in item:
            domain, task_id = item.split("/", 1)
            return _task_from_domain_id(domain, task_id, index)
        return load_task(item)
    if not isinstance(item, dict):
        raise ValueError(f"taskset item {idx} must be an object or string")

    path = item.get("path") or item.get("task_config_path")
    domain = item.get("domain")
    task_id = item.get("task_id") or item.get("id")
    if path:
        if not isinstance(path, str):
            raise ValueError(f"taskset item {idx} has non-string path")
        return load_task(path, domain=domain if isinstance(domain, str) else None)
    if not isinstance(domain, str) or not isinstance(task_id, str):
        raise ValueError(f"taskset item {idx} needs domain/task_id or path")
    return _task_from_domain_id(domain, task_id, index)


def normalize_taskset_path(path: str | Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw


def load_taskset(path: str | Path, *, custom_root: Path = CUSTOM_TASKS_DIR) -> list[CustomTask]:
    taskset_path = normalize_taskset_path(path)
    data = load_json(taskset_path)
    index = index_custom_tasks(custom_root)
    tasks: list[CustomTask] = []

    if isinstance(data, dict) and isinstance(data.get("tasks"), list):
        for idx, item in enumerate(data["tasks"]):
            tasks.append(_task_from_item(item, idx, index))
        return tasks

    if isinstance(data, list):
        for idx, item in enumerate(data):
            tasks.append(_task_from_item(item, idx, index))
        return tasks

    if isinstance(data, dict):
        for domain, task_ids in data.items():
            if not isinstance(domain, str) or not isinstance(task_ids, list):
                raise ValueError(f"taskset must be domain -> list, tasks list, or list: {taskset_path}")
            for task_id in task_ids:
                if not isinstance(task_id, str):
                    raise ValueError(f"task id in domain {domain} is not a string")
                tasks.append(_task_from_domain_id(domain, task_id, index))
        return tasks

    raise ValueError(f"unsupported taskset shape: {taskset_path}")


def unique_by_id(tasks: Iterable[CustomTask]) -> list[CustomTask]:
    seen: set[str] = set()
    unique: list[CustomTask] = []
    for task in tasks:
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        unique.append(task)
    return unique
