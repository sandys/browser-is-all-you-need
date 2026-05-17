"""Discover and validate packaged Harbor DOMDiff tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib  # type: ignore[no-redef]

from .rubric import validate_harbor_rubric

DEFAULT_HARBOR_TASK_ROOT = "benchmarks/harbor_tasks/github-arbitrary-domdiff-smoke"
DEFAULT_HARBOR_TASK_IDS = ("radix-ui__primitives-3548", "chakra-ui__chakra-ui-8905")

REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "rubric.json",
    "source_rubric.json",
    "gold_signals.json",
    "environment/Dockerfile",
    "tests/test.sh",
)


@dataclass(frozen=True)
class HarborTask:
    task_id: str
    path: Path
    instruction: str
    task_config: dict[str, Any]
    rubric: dict[str, Any]
    gold_signals: dict[str, Any]

    @property
    def preview_path(self) -> str:
        verifier_env = self.task_config.get("verifier", {}).get("env", {})
        if not isinstance(verifier_env, dict):
            return ""
        return str(verifier_env.get("PREVIEW_PATH", "") or "")


def repo_root() -> Path:
    return Path(os.environ.get("W8_REPO_ROOT", Path(__file__).resolve().parents[3])).resolve()


def resolve_task_root(root: str | Path | None = None) -> Path:
    raw = root or os.environ.get("W8_HARBOR_TASK_ROOT") or DEFAULT_HARBOR_TASK_ROOT
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def env_task_ids() -> tuple[str, ...] | None:
    raw = os.environ.get("W8_HARBOR_TASK_IDS", "").strip()
    if not raw:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def discover_task_ids(task_root: str | Path | None = None) -> tuple[str, ...]:
    root = resolve_task_root(task_root)
    if not root.exists():
        raise FileNotFoundError(f"Harbor task root does not exist: {root}")
    return tuple(
        child.name
        for child in sorted(root.iterdir())
        if child.is_dir() and (child / "task.toml").exists()
    )


def selected_task_ids(
    task_root: str | Path | None = None,
    task_ids: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    requested = tuple(str(task_id).strip() for task_id in (task_ids or ()) if str(task_id).strip())
    if requested:
        return requested
    return env_task_ids() or discover_task_ids(task_root)


def load_task(task_id: str, task_root: str | Path | None = None) -> HarborTask:
    root = resolve_task_root(task_root)
    path = root / task_id
    validate_task_dir(path)
    task_config = _read_toml(path / "task.toml")
    rubric = _read_json(path / "rubric.json")
    gold_signals = _read_json(path / "gold_signals.json")
    return HarborTask(
        task_id=task_id,
        path=path,
        instruction=(path / "instruction.md").read_text(encoding="utf-8").strip(),
        task_config=task_config,
        rubric=rubric,
        gold_signals=gold_signals,
    )


def validate_task_dir(task_dir: str | Path) -> None:
    path = Path(task_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Harbor task directory does not exist: {path}")
    missing = [name for name in REQUIRED_TASK_FILES if not (path / name).exists()]
    if missing:
        raise ValueError(f"{path.name} is missing required Harbor files: {', '.join(missing)}")

    task_config = _read_toml(path / "task.toml")
    instance_id = str(task_config.get("metadata", {}).get("instance_id", "") or "")
    if instance_id and instance_id != path.name:
        raise ValueError(f"{path.name} task.toml instance_id mismatch: {instance_id}")

    verifier_env = task_config.get("verifier", {}).get("env", {})
    if not isinstance(verifier_env, dict):
        raise ValueError(f"{path.name} task.toml verifier.env must be an object")
    if str(verifier_env.get("CHROMIUMRL_ELIGIBLE", "")).lower() != "true":
        raise ValueError(f"{path.name} must be ChromiumRL eligible")
    if not str(verifier_env.get("PREVIEW_PATH", "") or "").startswith("/"):
        raise ValueError(f"{path.name} must define verifier.env.PREVIEW_PATH")

    rubric = _read_json(path / "rubric.json")
    validate_harbor_rubric(rubric)
    gold = _read_json(path / "gold_signals.json")
    reference = gold.get("dom_state", gold)
    if not isinstance(reference, dict):
        raise ValueError(f"{path.name} gold_signals.json must contain an object reference state")
    nodes = reference.get("nodes", [])
    if not isinstance(nodes, list) or len(nodes) < 10:
        raise ValueError(f"{path.name} gold_signals.json is missing a meaningful DOM node list")


def validate_tasks(
    task_root: str | Path | None = None,
    task_ids: tuple[str, ...] | list[str] | None = None,
) -> list[HarborTask]:
    return [load_task(task_id, task_root) for task_id in selected_task_ids(task_root, task_ids)]


def build_harbor_rows(
    *,
    task_root: str | Path | None = None,
    task_ids: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert local Harbor task dirs to portable task metadata rows."""

    root = resolve_task_root(task_root)
    ids = selected_task_ids(root, task_ids)
    if not ids:
        raise RuntimeError(f"No Harbor tasks found under {root}")

    rows: list[dict[str, Any]] = []
    for task_id in ids:
        task = load_task(task_id, root)
        rows.append(
            {
                "id": task_id,
                "task_path": str(task.path),
                "task": task.instruction,
                "metadata": {
                    "task_path": str(task.path),
                    "harbor_task_name": task_id,
                    "preview_path": task.preview_path,
                },
            }
        )
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return doc


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        doc = tomllib.load(fh)
    if not isinstance(doc, dict):
        raise ValueError(f"Expected object TOML at {path}")
    return doc
