from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .registry import CustomTask, load_task, load_taskset, repo_path

SUPPORTED_CONFIG_TYPES = {
    "execute",
    "launch",
    "activate_window",
    "sleep",
    "open",
    "download_file",
    "upload_file",
    "chrome_open_tabs",
}

LOCAL_ONLY_MARKERS = (
    "/data/",
    "/Users/",
    "/home/ubuntu/",
    "/home/saad/",
    "C:\\Users\\",
)

NETWORK_MARKERS = ("http://", "https://")


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class ValidationReport:
    tasks: list[CustomTask]
    issues: list[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "task_count": len(self.tasks),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "tasks": [task.as_row() for task in self.tasks],
            "issues": [issue.as_dict() for issue in self.issues],
        }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _issue(level: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(level=level, path=path, message=message)


def _json_walk(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _json_walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_walk(child)


def _validate_config_steps(task: CustomTask, key: str, value: Any, issues: list[ValidationIssue]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        issues.append(_issue("error", repo_path(task.path), f"{key} must be a list"))
        return
    for idx, step in enumerate(value):
        prefix = f"{key}[{idx}]"
        if not isinstance(step, dict):
            issues.append(_issue("error", repo_path(task.path), f"{prefix} must be an object"))
            continue
        step_type = step.get("type")
        if not isinstance(step_type, str) or not step_type.strip():
            issues.append(_issue("error", repo_path(task.path), f"{prefix}.type is required"))
            continue
        if step_type not in SUPPORTED_CONFIG_TYPES:
            issues.append(_issue("warning", repo_path(task.path), f"{prefix}.type '{step_type}' is not in the local known-type list"))
        params = step.get("parameters")
        if params is not None and not isinstance(params, dict):
            issues.append(_issue("error", repo_path(task.path), f"{prefix}.parameters must be an object when present"))


def validate_task(task: CustomTask, *, allow_proxy: bool = False, strict_network: bool = True) -> list[ValidationIssue]:
    data = task.data
    path = repo_path(task.path)
    issues: list[ValidationIssue] = []

    required = ["id", "snapshot", "instruction", "source", "config", "trajectory", "related_apps", "evaluator", "proxy", "fixed_ip"]
    for key in required:
        if key not in data:
            issues.append(_issue("error", path, f"missing required key: {key}"))

    if not isinstance(data.get("id"), str) or not str(data.get("id", "")).strip():
        issues.append(_issue("error", path, "id must be a non-empty string"))
    if data.get("id") != task.task_id:
        issues.append(_issue("error", path, f"task id mismatch: registry={task.task_id} task.json={data.get('id')}"))
    if not task.instruction:
        issues.append(_issue("error", path, "instruction must be a non-empty string"))
    if not isinstance(data.get("related_apps"), list) or not data.get("related_apps"):
        issues.append(_issue("error", path, "related_apps must be a non-empty list"))

    if _truthy(data.get("proxy")) and not allow_proxy:
        issues.append(_issue("error", path, "proxy is true; pass --allow-proxy only for intentional proxy tasks"))

    _validate_config_steps(task, "config", data.get("config"), issues)

    evaluator = data.get("evaluator")
    if not isinstance(evaluator, dict):
        issues.append(_issue("error", path, "evaluator must be an object"))
        return issues

    postconfig = evaluator.get("postconfig")
    if postconfig is not None:
        _validate_config_steps(task, "evaluator.postconfig", postconfig, issues)

    if evaluator.get("func") != "check_include_exclude":
        issues.append(_issue("error", path, "evaluator.func must be check_include_exclude for deterministic custom tasks"))

    result = evaluator.get("result")
    if not isinstance(result, dict):
        issues.append(_issue("error", path, "evaluator.result must be an object"))
    else:
        if result.get("type") != "vm_command_line":
            issues.append(_issue("error", path, "evaluator.result.type must be vm_command_line"))
        command = result.get("command")
        if not isinstance(command, (str, list)) or not command:
            issues.append(_issue("error", path, "evaluator.result.command must be a non-empty string or argv list"))

    expected = evaluator.get("expected")
    if not isinstance(expected, dict):
        issues.append(_issue("error", path, "evaluator.expected must be an object"))
    else:
        if expected.get("type") != "rule":
            issues.append(_issue("error", path, "evaluator.expected.type must be rule"))
        rules = expected.get("rules")
        if not isinstance(rules, dict):
            issues.append(_issue("error", path, "evaluator.expected.rules must be an object"))
        else:
            include = rules.get("include")
            exclude = rules.get("exclude")
            if not isinstance(include, list) or "Evaluation successful." not in include:
                issues.append(_issue("error", path, "expected.rules.include must contain Evaluation successful."))
            if not isinstance(exclude, list) or "Evaluation failed." not in exclude:
                issues.append(_issue("error", path, "expected.rules.exclude must contain Evaluation failed."))

    strings = list(_json_walk(data))
    for marker in LOCAL_ONLY_MARKERS:
        if any(marker in value for value in strings):
            issues.append(_issue("error", path, f"task JSON contains local-only host path marker: {marker}"))
    if strict_network and not _truthy(data.get("proxy")):
        for marker in NETWORK_MARKERS:
            if any(marker in value for value in strings):
                issues.append(_issue("error", path, f"non-proxy task JSON contains network URL marker: {marker}"))

    return issues


def validate_tasks(tasks: list[CustomTask], *, allow_proxy: bool = False, strict_network: bool = True) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seen: dict[str, CustomTask] = {}
    for task in tasks:
        if task.task_id in seen:
            issues.append(_issue("error", repo_path(task.path), f"duplicate task id also seen at {repo_path(seen[task.task_id].path)}"))
        else:
            seen[task.task_id] = task
        issues.extend(validate_task(task, allow_proxy=allow_proxy, strict_network=strict_network))
    return ValidationReport(tasks=tasks, issues=issues)


def validate_task_paths(paths: list[str | Path], *, allow_proxy: bool = False, strict_network: bool = True) -> ValidationReport:
    tasks = [load_task(path) for path in paths]
    return validate_tasks(tasks, allow_proxy=allow_proxy, strict_network=strict_network)


def validate_taskset(path: str | Path, *, allow_proxy: bool = False, strict_network: bool = True) -> ValidationReport:
    return validate_tasks(load_taskset(path), allow_proxy=allow_proxy, strict_network=strict_network)


def format_report(report: ValidationReport) -> str:
    status = "ok" if report.ok else "failed"
    lines = [f"validation {status}: {len(report.tasks)} task(s), {len(report.errors)} error(s), {len(report.warnings)} warning(s)"]
    for issue in report.issues:
        lines.append(f"{issue.level.upper():<7} {issue.path}  {issue.message}")
    return "\n".join(lines)


def report_to_json(report: ValidationReport) -> str:
    return json.dumps(report.as_dict(), indent=2, sort_keys=True)
