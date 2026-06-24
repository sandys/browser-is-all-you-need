"""OSWorld upstream smoke helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from importlib import resources
from typing import Callable
from dataclasses import dataclass
from pathlib import Path

from . import upstreams
from .constants import UPSTREAMS
from .mlflow_tracking import ensure_tracker
from .shell import format_command, run_command

OSWORLD_UPSTREAM_KEY = "osworld"
DEFAULT_OSWORLD_TASK = "os/e0df059f-28a6-4169-924f-b9623e7184cc"
DEFAULT_PROVIDER = "docker"
DEFAULT_DOCKER_IMAGE = "happysixd/osworld-docker"
DEFAULT_OBSERVATION_TYPE = "screenshot"
DEFAULT_ACTION_SPACE = "pyautogui"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_STEPS = 15
DEFAULT_MAX_TOKENS = 256
DEFAULT_SMOKE_SUITE = "tiny"
DEFAULT_CLIENT_PASSWORD = "password"
DEFAULT_PROXY_CONFIG = Path("evaluation_examples/settings/proxy/dataimpulse.json")
LOCAL_OPENAI_PROVIDER_PATCH_MARKER = "# w8-biayn local OpenAI-compatible provider patch"
LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER = "# w8-biayn local OpenAI-compatible provider patch v2"
KIMI_PROVIDER_PATCH_MARKER = LOCAL_OPENAI_PROVIDER_PATCH_MARKER
PROXY_RUN_PATCH_MARKER = "# w8-biayn OSWorld proxy patch"
A11Y_COMPACTION_PATCH_MARKER = "# w8-biayn OSWorld a11y compaction patch"
RESULT_ARTIFACT_PATCH_MARKER = "# w8-biayn OSWorld result artifact patch"
DEFAULT_A11Y_TREE_MAX_ITEMS = 300
DEFAULT_A11Y_IOU_THRESHOLD = 0.2
TASKSET_PACKAGE = "w8_biayn.data.osworld_tasksets"
BUILTIN_TASKSETS: dict[str, str] = {}
SMOKE_SUITES = {
    "tiny": (
        DEFAULT_OSWORLD_TASK,
        "os/28cc3b7e-b194-4bc9-8353-d04c0f4d56d2",
        "os/bedcedc4-4d72-425e-ad62-21960b11fe0d",
    ),
}
KVM_DEVICE = Path("/dev/kvm")
STATE_DIR = Path(".w8-biayn/osworld")
METADATA_DIR = STATE_DIR / "metadata"
RESULTS_DIR = STATE_DIR / "results"
RUNS_DIR = STATE_DIR / "runs"
LATEST_RUN = STATE_DIR / "latest-run.txt"


def uv_command() -> list[str]:
    uv_bin = shutil.which("uv")
    if uv_bin:
        return [uv_bin]
    local_uv = Path.home() / ".local" / "bin" / "uv"
    if local_uv.exists():
        return [str(local_uv)]
    raise FileNotFoundError("Could not find `uv`. Install it or add it to PATH.")


@dataclass(frozen=True)
class ValidationRow:
    check: str
    status: str
    detail: str
    required: bool = False


@dataclass(frozen=True)
class ResultSummary:
    tasks: tuple[TaskResult, ...]

    @property
    def completed(self) -> int:
        return sum(task.score is not None for task in self.tasks)

    @property
    def successes(self) -> int:
        return sum(task.status == "success" for task in self.tasks)

    @property
    def failures(self) -> int:
        return sum(task.status == "failure" for task in self.tasks)

    @property
    def average_score(self) -> float | None:
        scores = [task.score for task in self.tasks if task.score is not None]
        return sum(scores) / len(scores) if scores else None


@dataclass(frozen=True)
class TaskInfo:
    task: str
    domain: str
    task_id: str
    instruction: str
    proxy: bool | None
    fixed_ip: bool | None
    env_change: str
    path: Path


@dataclass(frozen=True)
class TaskResult:
    task: str
    status: str
    score: float | None
    result_file: Path


@dataclass(frozen=True)
class BenchmarkDomainResult:
    domain: str
    run_id: str
    tasks: tuple[str, ...]
    completed: int
    successes: int
    failures: int
    average_score: float | None
    results: Path


@dataclass(frozen=True)
class BenchmarkResult:
    domains: tuple[BenchmarkDomainResult, ...]

    @property
    def total_tasks(self) -> int:
        return sum(len(domain.tasks) for domain in self.domains)

    @property
    def completed(self) -> int:
        return sum(domain.completed for domain in self.domains)

    @property
    def successes(self) -> int:
        return sum(domain.successes for domain in self.domains)

    @property
    def failures(self) -> int:
        return sum(domain.failures for domain in self.domains)

    @property
    def average_score(self) -> float | None:
        scored_domains = [domain for domain in self.domains if domain.average_score is not None]
        total_completed = sum(domain.completed for domain in scored_domains)
        if total_completed == 0:
            return None
        weighted_sum = sum((domain.average_score or 0.0) * domain.completed for domain in scored_domains)
        return weighted_sum / total_completed


@dataclass(frozen=True)
class BenchmarkProgress:
    total_tasks: int
    completed_tasks: int
    remaining_tasks: int
    elapsed_seconds: float
    eta_seconds: float | None
    current_domain: str | None = None


@dataclass(frozen=True)
class TaskRef:
    domain: str
    task_id: str

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task_id}"


def upstream_path(repo_root: str | Path = ".") -> Path:
    return upstreams.upstream_path(UPSTREAMS[OSWORLD_UPSTREAM_KEY], repo_root)


def venv_path(repo_root: str | Path = ".") -> Path:
    return upstream_path(repo_root) / ".venv"


def upstream_agent_path(repo_root: str | Path = ".") -> Path:
    return upstream_path(repo_root) / "mm_agents" / "agent.py"


def upstream_run_path(repo_root: str | Path = ".") -> Path:
    return upstream_path(repo_root) / "run.py"


def upstream_lib_run_single_path(repo_root: str | Path = ".") -> Path:
    return upstream_path(repo_root) / "lib_run_single.py"


def uses_local_openai_provider(model: str, *, base_url: str | None = None) -> bool:
    model_name = model.lower()
    return (
        model_name.startswith(("kimi", "qwen"))
        or bool(base_url)
        or bool(os.environ.get("LOCAL_OPENAI_BASE_URL"))
        or bool(os.environ.get("QWEN_BASE_URL"))
        or bool(os.environ.get("KIMI_BASE_URL"))
    )


def uses_kimi_openai_provider(model: str) -> bool:
    return uses_local_openai_provider(model)


def _legacy_local_openai_response_block(indent: str) -> str:
    child = indent + "    "
    return (
        f"{indent}if response.status_code != 200:\n"
        f"{child}if response.json()['error']['code'] == \"context_length_exceeded\":\n"
        f"{child}    logger.error(\"Context length exceeded. Retrying with a smaller context.\")\n"
        f"{child}    payload[\"messages\"] = [payload[\"messages\"][0]] + payload[\"messages\"][-1:]\n"
        f"{child}    retry_response = requests.post(\n"
        f"{child}        api_url,\n"
        f"{child}        headers=headers,\n"
        f"{child}        json=payload\n"
        f"{child}    )\n"
        f"{child}    if retry_response.status_code != 200:\n"
        f"{child}        logger.error(\n"
        f"{child}            \"Failed to call LLM even after attempt on shortening the history: \" + retry_response.text)\n"
        f"{child}        return \"\"\n"
        "\n"
        f"{child}logger.error(\"Failed to call LLM: \" + response.text)\n"
        f"{child}time.sleep(5)\n"
        f"{child}return \"\"\n"
        f"{indent}else:\n"
        f"{child}return response.json()['choices'][0]['message']['content']\n"
    )


def _hardened_local_openai_response_block(indent: str) -> str:
    child = indent + "    "
    return (
        f"{indent}if response.status_code != 200:\n"
        f"{child}response_text = response.text\n"
        f"{child}try:\n"
        f"{child}    error_doc = response.json()\n"
        f"{child}except ValueError:\n"
        f"{child}    error_doc = {{}}\n"
        f"{child}error_obj = error_doc.get(\"error\") if isinstance(error_doc, dict) else None\n"
        f"{child}error_code = error_obj.get(\"code\") if isinstance(error_obj, dict) else None\n"
        f"{child}if error_code == \"context_length_exceeded\":\n"
        f"{child}    logger.error(\"Context length exceeded. Retrying with a smaller context.\")\n"
        f"{child}    payload[\"messages\"] = [payload[\"messages\"][0]] + payload[\"messages\"][-1:]\n"
        f"{child}    retry_response = requests.post(\n"
        f"{child}        api_url,\n"
        f"{child}        headers=headers,\n"
        f"{child}        json=payload\n"
        f"{child}    )\n"
        f"{child}    if retry_response.status_code != 200:\n"
        f"{child}        logger.error(\n"
        f"{child}            \"Failed to call LLM even after attempt on shortening the history: \" + retry_response.text)\n"
        f"{child}        return \"\"\n"
        "\n"
        f"{child}logger.error(\"Failed to call LLM: \" + response_text)\n"
        f"{child}time.sleep(5)\n"
        f"{child}return \"\"\n"
        f"{indent}else:\n"
        f"{child}response_doc = response.json()\n"
        f"{child}content = response_doc[\"choices\"][0][\"message\"].get(\"content\", \"\")\n"
        f"{child}if isinstance(content, list):\n"
        f"{child}    return \"\".join(\n"
        f"{child}        part.get(\"text\", \"\") for part in content if isinstance(part, dict)\n"
        f"{child}    )\n"
        f"{child}return content or \"\"\n"
    )


def _harden_local_openai_response_handling(source: str) -> str:
    anchor = 'logger.info("Generating content with OpenAI-compatible model: %s", self.model)'
    start = source.find(anchor)
    if start == -1:
        raise RuntimeError(
            "Could not harden local OpenAI-compatible OSWorld provider patch; "
            "the patched provider logger was not found."
        )
    for indent in ("            ", "        "):
        old_block = _legacy_local_openai_response_block(indent)
        block_start = source.find(old_block, start)
        if block_start != -1:
            new_block = _hardened_local_openai_response_block(indent)
            return source[:block_start] + new_block + source[block_start + len(old_block) :]
    if "response_text = response.text" in source[start:]:
        return source
    raise RuntimeError(
        "Could not harden local OpenAI-compatible OSWorld provider patch; "
        "the upstream response handling block changed."
    )


def ensure_local_openai_provider(
    *,
    model: str,
    base_url: str | None = None,
    repo_root: str | Path = ".",
) -> bool:
    """Patch ignored OSWorld upstream to route local models through an OpenAI-compatible API."""
    if not uses_local_openai_provider(model, base_url=base_url):
        return False
    agent_path = upstream_agent_path(repo_root)
    if not agent_path.exists():
        raise FileNotFoundError(f"OSWorld agent module not found: {agent_path}")
    source = agent_path.read_text(encoding="utf-8")
    if LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER in source:
        return False
    if LOCAL_OPENAI_PROVIDER_PATCH_MARKER in source:
        patched = source.replace(
            LOCAL_OPENAI_PROVIDER_PATCH_MARKER,
            f"{LOCAL_OPENAI_PROVIDER_PATCH_MARKER}\n            {LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER}",
            1,
        )
        patched = _harden_local_openai_response_handling(patched)
        agent_path.write_text(patched, encoding="utf-8")
        return True

    replacements = {
        'elif self.model.startswith("gpt"):': (
            'elif self.model.startswith("gpt") or self.model.lower().startswith(("kimi", "qwen")) '
            'or os.environ.get("LOCAL_OPENAI_BASE_URL") '
            'or os.environ.get("QWEN_BASE_URL") '
            'or os.environ.get("KIMI_BASE_URL"):\n'
            f"            {LOCAL_OPENAI_PROVIDER_PATCH_MARKER}\n"
            f"            {LOCAL_OPENAI_PROVIDER_PATCH_V2_MARKER}"
        ),
        "base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com')": (
            'is_local_openai_provider = (\n'
            '                self.model.lower().startswith(("kimi", "qwen"))\n'
            '                or bool(os.environ.get("LOCAL_OPENAI_BASE_URL"))\n'
            '                or bool(os.environ.get("QWEN_BASE_URL"))\n'
            '                or bool(os.environ.get("KIMI_BASE_URL"))\n'
            '            )\n'
            '            base_url = (\n'
            '                os.environ.get("LOCAL_OPENAI_BASE_URL")\n'
            '                or os.environ.get("QWEN_BASE_URL")\n'
            '                or os.environ.get("KIMI_BASE_URL")\n'
            '                or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")\n'
            '                if is_local_openai_provider\n'
            '                else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")\n'
            '            )\n'
            '            api_key = (\n'
            '                os.environ.get("LOCAL_OPENAI_API_KEY")\n'
            '                or os.environ.get("QWEN_API_KEY")\n'
            '                or os.environ.get("KIMI_API_KEY")\n'
            '                or os.environ.get("OPENAI_API_KEY", "sk-no-key-required")\n'
            '                if is_local_openai_provider\n'
            '                else os.environ["OPENAI_API_KEY"]\n'
            '            )'
        ),
        "\"Authorization\": f\"Bearer {os.environ['OPENAI_API_KEY']}\"": (
            '"Authorization": f"Bearer {api_key}"'
        ),
        'logger.info("Generating content with GPT model: %s", self.model)': (
            'logger.info("Generating content with OpenAI-compatible model: %s", self.model)'
        ),
    }
    patched = source
    missing: list[str] = []
    for old, new in replacements.items():
        if old not in patched:
            missing.append(old)
            continue
        patched = patched.replace(old, new, 1)
    if missing:
        raise RuntimeError(
            "Could not apply local OpenAI-compatible provider patch to OSWorld upstream "
            "agent.py; the pinned upstream provider branch changed."
        )
    patched = _harden_local_openai_response_handling(patched)
    agent_path.write_text(patched, encoding="utf-8")
    return True


def ensure_kimi_openai_provider(*, model: str, repo_root: str | Path = ".") -> bool:
    return ensure_local_openai_provider(model=model, repo_root=repo_root)


def _a11y_compaction_trim_function() -> str:
    return r"""
def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    # w8-biayn OSWorld a11y compaction patch
    def _parse_pair(value):
        nums = re.findall(r"-?\d+", value or "")
        if len(nums) < 2:
            return None
        return int(nums[0]), int(nums[1])

    def _box(row):
        parts = row.split("\t")
        if len(parts) < 7:
            return None
        xy = _parse_pair(parts[-2])
        wh = _parse_pair(parts[-1])
        if xy is None or wh is None:
            return None
        return (xy[0], xy[1], wh[0], wh[1])

    def _iou(a, b):
        if a is None or b is None:
            return 0.0
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        inter_w = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        inter_h = max(0, min(ay + ah, by + bh) - max(ay, by))
        inter = inter_w * inter_h
        if inter <= 0:
            return 0.0
        union = max(0, aw * ah) + max(0, bw * bh) - inter
        return 0.0 if union <= 0 else inter / union

    def _key(row):
        parts = row.split("\t")
        useful = parts[1:5] if len(parts) >= 5 else parts
        return re.sub(r"\s+", " ", " ".join(useful).strip().lower())

    max_items = int(os.environ.get("OSWORLD_A11Y_TREE_MAX_ITEMS", "0") or "0")
    iou_threshold = float(os.environ.get("OSWORLD_A11Y_IOU_THRESHOLD", "0.2") or "0.2")
    if max_items > 0:
        lines = linearized_accessibility_tree.splitlines()
        if len(lines) > 1:
            header, rows = lines[0], lines[1:]
            kept = []
            for row in rows:
                key = _key(row)
                row_box = _box(row)
                duplicate = False
                for existing in kept:
                    if key and key == _key(existing) and _iou(row_box, _box(existing)) > iou_threshold:
                        duplicate = True
                        break
                if not duplicate:
                    kept.append(row)
            truncated = len(kept) > max_items
            kept = kept[:max_items]
            lines = [header] + kept
            if truncated:
                lines.append("[... accessibility tree truncated by item cap ...]")
            linearized_accessibility_tree = "\n".join(lines)

    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree
"""


def ensure_a11y_compaction_support(*, repo_root: str | Path = ".") -> bool:
    """Patch ignored OSWorld upstream to cap/dedupe linearized a11y rows for local runs."""
    agent_path = upstream_agent_path(repo_root)
    if not agent_path.exists():
        raise FileNotFoundError(f"OSWorld agent module not found: {agent_path}")
    source = agent_path.read_text(encoding="utf-8")
    if A11Y_COMPACTION_PATCH_MARKER in source:
        return False
    old = r"""def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree
"""
    if old not in source:
        raise RuntimeError(
            "Could not apply OSWorld a11y compaction patch; upstream trim_accessibility_tree changed."
        )
    agent_path.write_text(source.replace(old, _a11y_compaction_trim_function(), 1), encoding="utf-8")
    return True


def ensure_proxy_run_support(*, repo_root: str | Path = ".") -> bool:
    """Patch ignored OSWorld run.py so DesktopEnv can enable proxy from env vars."""
    run_path = upstream_run_path(repo_root)
    if not run_path.exists():
        raise FileNotFoundError(f"OSWorld run.py not found: {run_path}")
    source = run_path.read_text(encoding="utf-8")
    if PROXY_RUN_PATCH_MARKER in source:
        return False

    replacement: tuple[str, str] | None = None
    for arg_indent in ("        ", "    "):
        close_indent = arg_indent[:-4]
        old = f"{arg_indent}vm_secret_mounts=args.vm_secret_mount,\n{close_indent})"
        new = (
            f"{arg_indent}vm_secret_mounts=args.vm_secret_mount,\n"
            f"{arg_indent}{PROXY_RUN_PATCH_MARKER}\n"
            f'{arg_indent}enable_proxy=os.environ.get("OSWORLD_ENABLE_PROXY") == "1",\n'
            f'{arg_indent}client_password=os.environ.get("OSWORLD_CLIENT_PASSWORD", "password"),\n'
            f"{close_indent})"
        )
        if old in source:
            replacement = (old, new)
            break
    if replacement is None:
        raise RuntimeError(
            "Could not apply OSWorld proxy patch to upstream run.py; "
            "the pinned upstream DesktopEnv construction changed."
        )
    run_path.write_text(source.replace(replacement[0], replacement[1], 1), encoding="utf-8")
    return True


def ensure_result_artifact_support(*, repo_root: str | Path = ".") -> bool:
    """Patch ignored OSWorld upstream to create artifact directories before writes."""
    lib_path = upstream_lib_run_single_path(repo_root)
    if not lib_path.exists():
        raise FileNotFoundError(f"OSWorld lib_run_single.py not found: {lib_path}")
    source = lib_path.read_text(encoding="utf-8")
    if RESULT_ARTIFACT_PATCH_MARKER in source:
        return False
    old = """def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)

    # Reset environment first to get fresh VM IP
"""
    new = """def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    os.makedirs(example_result_dir, exist_ok=True)
    runtime_logger = setup_logger(example, example_result_dir)

    # Reset environment first to get fresh VM IP
"""
    if old not in source:
        raise RuntimeError(
            "Could not apply OSWorld result artifact patch; upstream run_single_example changed."
        )
    patched = source.replace(old, new, 1)
    replacement_applied = False
    for indent in ("            ", "        "):
        write_target = (
            f"{indent}# Save screenshot and trajectory information\n"
            f"{indent}with open(os.path.join(example_result_dir, f\"step_{{step_idx + 1}}_{{action_timestamp}}.png\"),\n"
        )
        write_replacement = (
            f"{indent}# Save screenshot and trajectory information\n"
            f"{indent}os.makedirs(example_result_dir, exist_ok=True)  # {RESULT_ARTIFACT_PATCH_MARKER}\n"
            f"{indent}with open(os.path.join(example_result_dir, f\"step_{{step_idx + 1}}_{{action_timestamp}}.png\"),\n"
        )
        if write_target in patched:
            patched = patched.replace(write_target, write_replacement, 1)
            replacement_applied = True
            break
    if not replacement_applied:
        raise RuntimeError(
            "Could not apply OSWorld result artifact patch; upstream artifact write block changed."
        )
    lib_path.write_text(patched, encoding="utf-8")
    return True


def resolve_proxy_config_file(
    proxy_config_file: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> Path:
    """Resolve an OSWorld proxy config path for the upstream process."""
    if proxy_config_file is None:
        return upstream_path(repo_root) / DEFAULT_PROXY_CONFIG
    candidate = Path(proxy_config_file).expanduser()
    if candidate.is_absolute():
        return candidate
    repo_candidate = Path(repo_root) / candidate
    if repo_candidate.exists():
        return repo_candidate
    return upstream_path(repo_root) / candidate


def validate_proxy_config_file(
    proxy_config_file: str | Path | None = None,
    *,
    repo_root: str | Path = ".",
) -> Path:
    """Validate the proxy JSON shape expected by upstream OSWorld."""
    path = resolve_proxy_config_file(proxy_config_file, repo_root=repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"OSWorld proxy config not found: {path}. "
            "Create it or pass --proxy-config-file."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OSWorld proxy config is not valid JSON: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("OSWorld proxy config must be a non-empty JSON list.")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"OSWorld proxy config entry {index} must be an object.")
        if not isinstance(item.get("host"), str) or not item["host"]:
            raise ValueError(f"OSWorld proxy config entry {index} needs a host string.")
        if not isinstance(item.get("port"), int):
            raise ValueError(f"OSWorld proxy config entry {index} needs an integer port.")
        protocol = item.get("protocol", "http")
        if protocol not in {"http", "https", "socks5"}:
            raise ValueError(
                f"OSWorld proxy config entry {index} has unsupported protocol {protocol!r}."
            )
    return path


def proxy_environment(
    *,
    enable_proxy: bool,
    proxy_config_file: str | Path | None = None,
    client_password: str = DEFAULT_CLIENT_PASSWORD,
    repo_root: str | Path = ".",
) -> dict[str, str]:
    if not enable_proxy:
        return {"OSWORLD_ENABLE_PROXY": "0"}
    path = validate_proxy_config_file(proxy_config_file, repo_root=repo_root)
    return {
        "OSWORLD_ENABLE_PROXY": "1",
        "PROXY_CONFIG_FILE": str(path.resolve()),
        "OSWORLD_CLIENT_PASSWORD": client_password,
    }


def local_openai_environment(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if base_url:
        env["LOCAL_OPENAI_BASE_URL"] = base_url
    if api_key is not None:
        env["LOCAL_OPENAI_API_KEY"] = api_key
    return env


def a11y_compaction_environment(
    *,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
) -> dict[str, str]:
    if a11y_tree_max_items <= 0:
        raise ValueError("a11y_tree_max_items must be positive.")
    if a11y_iou_threshold < 0:
        raise ValueError("a11y_iou_threshold must be non-negative.")
    return {
        "OSWORLD_A11Y_TREE_MAX_ITEMS": str(a11y_tree_max_items),
        "OSWORLD_A11Y_IOU_THRESHOLD": str(a11y_iou_threshold),
    }


def parse_task(task: str) -> TaskRef:
    if "/" not in task:
        raise ValueError("OSWorld task must be formatted as <domain>/<task-id>.")
    domain, task_id = task.split("/", maxsplit=1)
    if not domain or not task_id:
        raise ValueError("OSWorld task must be formatted as <domain>/<task-id>.")
    return TaskRef(domain=domain, task_id=task_id)


def task_config_path(task: TaskRef, repo_root: str | Path = ".") -> Path:
    return (
        upstream_path(repo_root)
        / "evaluation_examples"
        / "examples"
        / task.domain
        / f"{task.task_id}.json"
    )


def task_examples_root(repo_root: str | Path = ".") -> Path:
    return upstream_path(repo_root) / "evaluation_examples" / "examples"


def _read_task_info(path: Path, domain: str) -> TaskInfo:
    task_id = path.stem
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    instruction = payload.get("instruction", "")
    if not isinstance(instruction, str):
        instruction = ""
    env_change = payload.get("possibility_of_env_change", "")
    if not isinstance(env_change, str):
        env_change = ""
    proxy = payload.get("proxy")
    fixed_ip = payload.get("fixed_ip")
    return TaskInfo(
        task=f"{domain}/{task_id}",
        domain=domain,
        task_id=task_id,
        instruction=instruction,
        proxy=proxy if isinstance(proxy, bool) else None,
        fixed_ip=fixed_ip if isinstance(fixed_ip, bool) else None,
        env_change=env_change,
        path=path,
    )


def list_tasks(
    *,
    domain: str | None = None,
    smoke_candidates: bool = False,
    repo_root: str | Path = ".",
) -> list[TaskInfo]:
    root = task_examples_root(repo_root)
    if not root.exists():
        return []
    domain_dirs = (
        [root / domain]
        if domain
        else sorted(path for path in root.iterdir() if path.is_dir())
    )
    tasks: list[TaskInfo] = []
    for domain_dir in domain_dirs:
        if not domain_dir.exists() or not domain_dir.is_dir():
            continue
        for task_file in sorted(domain_dir.glob("*.json")):
            task = _read_task_info(task_file, domain_dir.name)
            if smoke_candidates and not (
                task.proxy is False
                and task.fixed_ip is False
                and task.env_change == "low"
            ):
                continue
            tasks.append(task)
    return tasks


def list_domains(*, repo_root: str | Path = ".") -> tuple[str, ...]:
    root = task_examples_root(repo_root)
    if not root.exists():
        return ()
    return tuple(sorted(path.name for path in root.iterdir() if path.is_dir()))


def list_tasksets() -> tuple[str, ...]:
    return tuple(sorted(BUILTIN_TASKSETS))


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "-" for char in value)


def taskset_metadata_name(taskset: str) -> str:
    if taskset in BUILTIN_TASKSETS:
        return f"taskset-{taskset}"
    return f"taskset-{_safe_name(Path(taskset).stem or 'custom')}"


def run_selector_metadata_name(
    *,
    suite: str | None = None,
    domain: str | None = None,
    taskset: str | None = None,
) -> str:
    if suite:
        return f"run-{suite}"
    if domain:
        return f"run-{domain}"
    if taskset:
        return f"run-{taskset_metadata_name(taskset)}"
    return "run-tasks"


def _read_taskset_json(taskset: str, *, repo_root: str | Path = ".") -> dict:
    if taskset in BUILTIN_TASKSETS:
        filename = BUILTIN_TASKSETS[taskset]
        try:
            return json.loads(
                resources.files(TASKSET_PACKAGE).joinpath(filename).read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Bundled OSWorld taskset is missing: {filename}") from exc

    path = Path(taskset)
    if not path.is_absolute():
        path = Path(repo_root) / path
    if not path.exists():
        known = ", ".join(list_tasksets())
        raise ValueError(
            f"Unknown OSWorld taskset {taskset!r}. Use a JSON path or one of: {known}."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"OSWorld taskset is not valid JSON: {path}") from exc


def load_taskset(taskset: str, *, repo_root: str | Path = ".") -> dict[str, tuple[str, ...]]:
    payload = _read_taskset_json(taskset, repo_root=repo_root)
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), dict):
        payload = payload["tasks"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("OSWorld taskset must be a non-empty domain-to-task-list JSON object.")

    grouped: dict[str, tuple[str, ...]] = {}
    for domain, task_ids in payload.items():
        if not isinstance(domain, str) or not domain:
            raise ValueError("OSWorld taskset domain keys must be non-empty strings.")
        if not isinstance(task_ids, list) or not task_ids:
            raise ValueError(f"OSWorld taskset domain {domain!r} must contain a non-empty list.")
        normalized: list[str] = []
        for task_id in task_ids:
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"OSWorld taskset domain {domain!r} has a non-string task id.")
            normalized.append(task_id.split("/", maxsplit=1)[1] if "/" in task_id else task_id)
        grouped[domain] = tuple(dict.fromkeys(normalized))
    return grouped


def tasks_for_taskset(
    taskset: str,
    *,
    limit: int | None = None,
    repo_root: str | Path = ".",
) -> tuple[str, ...]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive.")
    grouped = load_taskset(taskset, repo_root=repo_root)
    task_keys: list[str] = []
    for domain, task_ids in grouped.items():
        for task_id in task_ids:
            task_keys.append(f"{domain}/{task_id}")
    return tuple(task_keys[:limit] if limit is not None else task_keys)


def benchmark_task_groups(
    *,
    domains: tuple[str, ...] = (),
    taskset: str | None = None,
    limit_per_domain: int | None = None,
    smoke_candidates: bool = False,
    repo_root: str | Path = ".",
) -> dict[str, tuple[str, ...]]:
    if limit_per_domain is not None and limit_per_domain <= 0:
        raise ValueError("limit_per_domain must be positive.")
    if taskset and domains:
        raise ValueError("Select either --taskset or repeated --domain, not both.")
    if taskset:
        grouped = load_taskset(taskset, repo_root=repo_root)
        result: dict[str, tuple[str, ...]] = {}
        for domain, task_ids in grouped.items():
            task_keys = [f"{domain}/{task_id}" for task_id in task_ids]
            if smoke_candidates:
                allowed = {
                    task.task
                    for task in list_tasks(
                        domain=domain, smoke_candidates=True, repo_root=repo_root
                    )
                }
                task_keys = [task_key for task_key in task_keys if task_key in allowed]
            if limit_per_domain is not None:
                task_keys = task_keys[:limit_per_domain]
            result[domain] = tuple(task_keys)
        return result

    selected_domains = select_benchmark_domains(domains, repo_root=repo_root)
    return {
        domain: select_benchmark_task_keys(
            domain=domain,
            limit_per_domain=limit_per_domain,
            smoke_candidates=smoke_candidates,
            repo_root=repo_root,
        )
        for domain in selected_domains
    }


def select_benchmark_domains(
    domains: tuple[str, ...] = (),
    *,
    repo_root: str | Path = ".",
) -> tuple[str, ...]:
    available = list_domains(repo_root=repo_root)
    if not available:
        raise ValueError("No OSWorld domains found in the upstream checkout.")
    if not domains:
        return available
    unknown = sorted(set(domains) - set(available))
    if unknown:
        raise ValueError(
            "Unknown OSWorld domain(s): "
            + ", ".join(unknown)
            + ". Known domains: "
            + ", ".join(available)
        )
    return tuple(dict.fromkeys(domains))


def select_benchmark_task_keys(
    *,
    domain: str,
    limit_per_domain: int | None = None,
    smoke_candidates: bool = False,
    repo_root: str | Path = ".",
) -> tuple[str, ...]:
    if limit_per_domain is not None and limit_per_domain <= 0:
        raise ValueError("limit_per_domain must be positive.")
    tasks = list_tasks(domain=domain, smoke_candidates=smoke_candidates, repo_root=repo_root)
    selected = tasks[:limit_per_domain] if limit_per_domain is not None else tasks
    return tuple(task.task for task in selected)


def one_task_metadata_path(task: TaskRef, repo_root: str | Path = ".") -> Path:
    return metadata_path(f"{task.domain}-{task.task_id}", repo_root=repo_root)


def metadata_path(name: str, repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / METADATA_DIR / f"{name}.json"


def results_path(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / RESULTS_DIR


def run_dir_path(run_id: str, repo_root: str | Path = ".") -> Path:
    return runs_path(repo_root) / run_id


def run_results_path(run_id: str, repo_root: str | Path = ".") -> Path:
    return run_dir_path(run_id, repo_root) / "results"


def run_metadata_path(run_id: str, name: str, repo_root: str | Path = ".") -> Path:
    return run_dir_path(run_id, repo_root) / "metadata" / f"{name}.json"


def cleanup_docker_provider_containers() -> tuple[str, ...]:
    """Stop OSWorld Docker-provider containers left behind by interrupted runs."""
    if shutil.which("docker") is None:
        return ()
    ps = subprocess.run(
        [
            "docker",
            "ps",
            "--filter",
            f"ancestor={DEFAULT_DOCKER_IMAGE}",
            "--format",
            "{{.ID}}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if ps.returncode != 0:
        return ()
    container_ids = tuple(line.strip() for line in ps.stdout.splitlines() if line.strip())
    if not container_ids:
        return ()
    subprocess.run(["docker", "stop", *container_ids], check=False)
    return container_ids


def runs_path(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / RUNS_DIR


def latest_run_path(repo_root: str | Path = ".") -> Path:
    return Path(repo_root) / LATEST_RUN


def make_run_id(prefix: str = "local") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    return f"{prefix}-{stamp}"


def run_record_path(run_id: str, repo_root: str | Path = ".") -> Path:
    return run_dir_path(run_id, repo_root) / "record.json"


def write_run_record(
    *,
    run_id: str,
    command: str,
    tasks: tuple[str, ...],
    metadata: Path,
    results: Path,
    status: str,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    action_space: str = DEFAULT_ACTION_SPACE,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_run_id: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    mlflow_enabled: bool | None = None,
    repo_root: str | Path = ".",
) -> Path:
    path = run_record_path(run_id, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "command": command,
        "status": status,
        "tasks": list(tasks),
        "metadata": str(metadata),
        "results": str(results),
        "action_space": action_space,
        "observation_type": observation_type,
        "model": model,
        "enable_proxy": enable_proxy,
        "proxy_config_file": str(proxy_config_file) if proxy_config_file is not None else None,
        "upstream": str(upstream_path(repo_root)),
        "mlflow_tracking_uri": mlflow_tracking_uri,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_run_name": mlflow_run_name,
        "mlflow_experiment_name": mlflow_experiment_name,
        "mlflow_enabled": mlflow_enabled,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    latest = latest_run_path(repo_root)
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(run_id + "\n", encoding="utf-8")
    return path


def read_run_record(run_id: str | None = None, repo_root: str | Path = ".") -> dict:
    if run_id is None:
        latest = latest_run_path(repo_root)
        if not latest.exists():
            raise FileNotFoundError("No OSWorld run record found yet.")
        run_id = latest.read_text(encoding="utf-8").strip()
    path = run_record_path(run_id, repo_root)
    legacy_path = runs_path(repo_root) / f"{run_id}.json"
    if not path.exists() and legacy_path.exists():
        path = legacy_path
    if not path.exists():
        raise FileNotFoundError(f"OSWorld run record not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_run_records(repo_root: str | Path = ".") -> list[dict]:
    root = runs_path(repo_root)
    if not root.exists():
        return []
    records = []
    paths = [*root.glob("*/record.json"), *root.glob("*.json")]
    for path in sorted(paths):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records


def task_result_path(
    task: TaskRef,
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    result_dir: str | Path | None = None,
    repo_root: str | Path = ".",
) -> Path:
    return (
        (Path(result_dir) if result_dir is not None else results_path(repo_root))
        / action_space
        / observation_type
        / model
        / task.domain
        / task.task_id
        / "result.txt"
    )


def task_artifact_dir(
    task: TaskRef,
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    result_dir: str | Path | None = None,
    repo_root: str | Path = ".",
) -> Path:
    return task_result_path(
        task,
        action_space=action_space,
        observation_type=observation_type,
        model=model,
        result_dir=result_dir,
        repo_root=repo_root,
    ).parent


def ensure_result_dir_writable(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".w8-write-test"
    try:
        probe.write_text("ok\n", encoding="utf-8")
    finally:
        if probe.exists():
            probe.unlink()


def prepare_task_artifact_dirs(
    task_keys: tuple[str, ...],
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    result_dir: str | Path | None = None,
    repo_root: str | Path = ".",
) -> None:
    for task_key in task_keys:
        task_ref = parse_task(task_key)
        task_dir = task_artifact_dir(
            task_ref,
            action_space=action_space,
            observation_type=observation_type,
            model=model,
            result_dir=result_dir,
            repo_root=repo_root,
        )
        ensure_result_dir_writable(task_dir)


def read_task_result(
    task: str,
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    result_dir: str | Path | None = None,
    repo_root: str | Path = ".",
) -> TaskResult:
    task_ref = parse_task(task)
    path = task_result_path(
        task_ref,
        action_space=action_space,
        observation_type=observation_type,
        model=model,
        result_dir=result_dir,
        repo_root=repo_root,
    )
    if not path.exists():
        return TaskResult(task=task_ref.key, status="missing", score=None, result_file=path)
    raw = path.read_text(encoding="utf-8").strip()
    try:
        score = float(raw)
    except ValueError:
        return TaskResult(task=task_ref.key, status="invalid", score=None, result_file=path)
    status = "success" if score >= 1.0 else "failure"
    return TaskResult(task=task_ref.key, status=status, score=score, result_file=path)


def summarize_task_results(
    tasks: tuple[str, ...],
    *,
    action_space: str = DEFAULT_ACTION_SPACE,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    result_dir: str | Path | None = None,
    repo_root: str | Path = ".",
) -> ResultSummary:
    return ResultSummary(
        tuple(
            read_task_result(
                task,
                action_space=action_space,
                observation_type=observation_type,
                model=model,
                result_dir=result_dir,
                repo_root=repo_root,
            )
            for task in tasks
        )
    )


def tasks_for_suite(suite: str) -> tuple[str, ...]:
    try:
        return SMOKE_SUITES[suite]
    except KeyError as exc:
        known = ", ".join(sorted(SMOKE_SUITES))
        raise ValueError(f"Unknown OSWorld smoke suite {suite!r}. Known suites: {known}.") from exc


def group_task_refs(tasks: list[TaskRef]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for task in tasks:
        grouped.setdefault(task.domain, []).append(task.task_id)
    return grouped


def write_task_metadata(tasks: list[TaskRef], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(group_task_refs(tasks), indent=2) + "\n", encoding="utf-8")
    return path


def write_one_task_metadata(task: TaskRef, repo_root: str | Path = ".") -> Path:
    return write_task_metadata([task], one_task_metadata_path(task, repo_root))


def estimate_time_left_seconds(*, completed_tasks: int, total_tasks: int, elapsed_seconds: float) -> float | None:
    if completed_tasks <= 0 or total_tasks <= completed_tasks or elapsed_seconds <= 0:
        return None
    average_seconds_per_task = elapsed_seconds / completed_tasks
    return average_seconds_per_task * (total_tasks - completed_tasks)


def validate(
    task: str = DEFAULT_OSWORLD_TASK,
    *,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    repo_root: str | Path = ".",
) -> list[ValidationRow]:
    rows: list[ValidationRow] = []
    root = upstream_path(repo_root)
    task_ref: TaskRef | None = None
    try:
        task_ref = parse_task(task)
        rows.append(ValidationRow("task format", "ok", task_ref.key, required=True))
    except ValueError as exc:
        rows.append(ValidationRow("task format", "error", str(exc), required=True))

    rows.append(
        ValidationRow(
            "OSWorld upstream",
            "ok" if root.exists() else "missing",
            str(root) if root.exists() else "run `uv run w8-biayn upstreams clone osworld`",
            required=True,
        )
    )
    rows.append(
        ValidationRow(
            "OSWorld run.py",
            "ok" if (root / "run.py").exists() else "missing",
            str(root / "run.py"),
            required=True,
        )
    )
    rows.append(
        ValidationRow(
            "OSWorld pyproject.toml",
            "ok" if (root / "pyproject.toml").exists() else "missing",
            str(root / "pyproject.toml"),
            required=True,
        )
    )
    rows.append(
        ValidationRow(
            "OSWorld .venv",
            "ok" if venv_path(repo_root).exists() else "missing",
            str(venv_path(repo_root)) + " or run `uv run w8-biayn osworld setup`",
            required=True,
        )
    )
    docker = shutil.which("docker")
    rows.append(
        ValidationRow(
            "docker",
            "ok" if docker else "missing",
            docker or "not on PATH",
            required=True,
        )
    )
    docker_image_status = "skipped"
    docker_image_detail = "docker is missing"
    if docker:
        proc = subprocess.run(
            ["docker", "image", "inspect", DEFAULT_DOCKER_IMAGE],
            check=False,
            capture_output=True,
            text=True,
        )
        docker_image_status = "ok" if proc.returncode == 0 else "missing"
        docker_image_detail = (
            DEFAULT_DOCKER_IMAGE
            if proc.returncode == 0
            else f"run `docker pull {DEFAULT_DOCKER_IMAGE}` before real smoke"
        )
    rows.append(
        ValidationRow(
            "OSWorld Docker image",
            docker_image_status,
            docker_image_detail,
            required=True,
        )
    )
    kvm = KVM_DEVICE
    rows.append(
        ValidationRow(
            "/dev/kvm",
            "ok" if kvm.exists() else "missing",
            "required for fast local OSWorld Docker provider",
            required=True,
        )
    )
    if task_ref is not None:
        config = task_config_path(task_ref, repo_root)
        rows.append(
            ValidationRow(
                "task config",
                "ok" if config.exists() else "missing",
                str(config),
                required=True,
            )
        )
    if enable_proxy:
        try:
            proxy_path = validate_proxy_config_file(proxy_config_file, repo_root=repo_root)
            rows.append(ValidationRow("proxy config", "ok", str(proxy_path), required=True))
        except (FileNotFoundError, ValueError) as exc:
            rows.append(ValidationRow("proxy config", "error", str(exc), required=True))
    return rows


def has_errors(rows: list[ValidationRow]) -> bool:
    return any(row.status == "error" or (row.required and row.status != "ok") for row in rows)


def setup(dry_run: bool = False, repo_root: str | Path = ".") -> None:
    root = upstream_path(repo_root)
    if not root.exists() and not dry_run:
        raise FileNotFoundError(
            "OSWorld upstream is missing. Run `uv run w8-biayn upstreams clone osworld` first."
        )
    run_command([*uv_command(), "sync"], cwd=root, dry_run=dry_run)


def command_for_task_keys(
    task_keys: tuple[str, ...],
    *,
    metadata_name: str,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    headless: bool = True,
    repo_root: str | Path = ".",
    write_metadata: bool = True,
    result_dir: str | Path | None = None,
) -> tuple[Path, list[str], tuple[str, ...]]:
    task_refs = [parse_task(task_key) for task_key in task_keys]
    metadata = metadata_path(metadata_name, repo_root=repo_root)
    if write_metadata:
        write_task_metadata(task_refs, metadata)
    result_dir_path = Path(result_dir) if result_dir is not None else results_path(repo_root)
    domains = {task_ref.domain for task_ref in task_refs}
    domain = next(iter(domains)) if len(domains) == 1 else "all"
    args = [
        *uv_command(),
        "run",
        "python",
        "run.py",
        "--action_space",
        DEFAULT_ACTION_SPACE,
        "--provider_name",
        provider,
        "--observation_type",
        observation_type,
        "--model",
        model,
        "--max_steps",
        str(max_steps),
        "--max_tokens",
        str(max_tokens),
        "--max_trajectory_length",
        str(max_trajectory_length),
        "--test_all_meta_path",
        str(metadata.resolve()),
        "--domain",
        domain,
        "--result_dir",
        str(result_dir_path.resolve()),
    ]
    if headless:
        args.insert(4, "--headless")
    return metadata, args, task_keys


def smoke_command(
    *,
    task: str = DEFAULT_OSWORLD_TASK,
    suite: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    headless: bool = True,
    repo_root: str | Path = ".",
    write_metadata: bool = True,
    result_dir: str | Path | None = None,
) -> tuple[Path, list[str], tuple[str, ...]]:
    task_keys = tasks_for_suite(suite) if suite else (task,)
    metadata_name = (
        f"suite-{suite}" if suite else f"{parse_task(task).domain}-{parse_task(task).task_id}"
    )
    return command_for_task_keys(
        task_keys,
        metadata_name=metadata_name,
        provider=provider,
        observation_type=observation_type,
        model=model,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_trajectory_length=max_trajectory_length,
        headless=headless,
        repo_root=repo_root,
        write_metadata=write_metadata,
        result_dir=result_dir,
    )


def select_task_keys(
    *,
    tasks: tuple[str, ...] = (),
    suite: str | None = None,
    domain: str | None = None,
    taskset: str | None = None,
    limit: int | None = None,
    repo_root: str | Path = ".",
) -> tuple[str, ...]:
    selectors = sum(bool(value) for value in (tasks, suite, domain, taskset))
    if selectors != 1:
        raise ValueError("Select exactly one of --task, --suite, --domain, or --taskset.")
    if tasks:
        return tasks[:limit] if limit is not None else tasks
    if suite:
        suite_tasks = tasks_for_suite(suite)
        return suite_tasks[:limit] if limit is not None else suite_tasks
    if taskset:
        return tasks_for_taskset(taskset, limit=limit, repo_root=repo_root)
    discovered = tuple(task.task for task in list_tasks(domain=domain, repo_root=repo_root))
    selected = discovered[:limit] if limit is not None else discovered
    if not selected:
        raise ValueError(f"No OSWorld tasks found for domain {domain!r}.")
    return selected


def run_upstream_command_with_cleanup(
    args: list[str],
    *,
    cwd: Path,
    provider: str,
    env: dict[str, str] | None = None,
) -> None:
    try:
        run_command(args, cwd=cwd, env=env)
    except KeyboardInterrupt:
        if provider == DEFAULT_PROVIDER:
            stopped = cleanup_docker_provider_containers()
            if stopped:
                print(
                    "Stopped interrupted OSWorld Docker container(s): "
                    + ", ".join(stopped)
                )
        raise


def run_plan(
    *,
    tasks: tuple[str, ...] = (),
    suite: str | None = None,
    domain: str | None = None,
    taskset: str | None = None,
    limit: int | None = None,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
    headless: bool = True,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    repo_root: str | Path = ".",
) -> str:
    task_keys = select_task_keys(
        tasks=tasks,
        suite=suite,
        domain=domain,
        taskset=taskset,
        limit=limit,
        repo_root=repo_root,
    )
    name = run_selector_metadata_name(suite=suite, domain=domain, taskset=taskset)
    metadata, args, _ = command_for_task_keys(
        task_keys,
        metadata_name=name,
        provider=provider,
        observation_type=observation_type,
        model=model,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_trajectory_length=max_trajectory_length,
        headless=headless,
        repo_root=repo_root,
        write_metadata=False,
    )
    proxy_detail = (
        str(resolve_proxy_config_file(proxy_config_file, repo_root=repo_root))
        if enable_proxy
        else ""
    )
    lines = [
        "# OSWorld run dry run",
        f"upstream={upstream_path(repo_root)}",
        f"tasks={', '.join(task_keys)}",
        f"metadata={metadata}",
        f"results={results_path(repo_root)}",
        f"enable_proxy={str(enable_proxy).lower()}",
        f"proxy_config_file={proxy_detail}",
        f"base_url={base_url or ''}",
        f"api_key={'set' if api_key is not None else ''}",
        f"cd {upstream_path(repo_root)}",
        format_command(args),
    ]
    return "\n".join(lines)


def run(
    *,
    tasks: tuple[str, ...] = (),
    suite: str | None = None,
    domain: str | None = None,
    taskset: str | None = None,
    limit: int | None = None,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
    headless: bool = True,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    client_password: str = DEFAULT_CLIENT_PASSWORD,
    mlflow_tracking_uri: str | None = None,
    mlflow_run_id: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_experiment_name: str | None = None,
    dry_run: bool = False,
    run_id: str | None = None,
    repo_root: str | Path = ".",
) -> tuple[Path, tuple[str, ...], str | None]:
    root = upstream_path(repo_root)
    if dry_run:
        print(
            run_plan(
                tasks=tasks,
                suite=suite,
                domain=domain,
                taskset=taskset,
                limit=limit,
                provider=provider,
                observation_type=observation_type,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_steps=max_steps,
                max_tokens=max_tokens,
                max_trajectory_length=max_trajectory_length,
                a11y_tree_max_items=a11y_tree_max_items,
                a11y_iou_threshold=a11y_iou_threshold,
                headless=headless,
                enable_proxy=enable_proxy,
                proxy_config_file=proxy_config_file,
                repo_root=repo_root,
            )
        )
        task_keys = select_task_keys(
            tasks=tasks,
            suite=suite,
            domain=domain,
            taskset=taskset,
            limit=limit,
            repo_root=repo_root,
        )
        name = run_selector_metadata_name(suite=suite, domain=domain, taskset=taskset)
        return metadata_path(name, repo_root=repo_root), task_keys, None
    if not root.exists():
        raise FileNotFoundError(
            "OSWorld upstream is missing. Run `uv run w8-biayn upstreams clone osworld` first."
        )
    task_keys = select_task_keys(
        tasks=tasks,
        suite=suite,
        domain=domain,
        taskset=taskset,
        limit=limit,
        repo_root=repo_root,
    )
    for task_key in task_keys:
        task_ref = parse_task(task_key)
        config = task_config_path(task_ref, repo_root)
        if not config.exists():
            raise FileNotFoundError(f"OSWorld task config not found: {config}")
    proxy_env = proxy_environment(
        enable_proxy=enable_proxy,
        proxy_config_file=proxy_config_file,
        client_password=client_password,
        repo_root=repo_root,
    )
    provider_env = {
        **proxy_env,
        **local_openai_environment(base_url=base_url, api_key=api_key),
        **a11y_compaction_environment(
            a11y_tree_max_items=a11y_tree_max_items,
            a11y_iou_threshold=a11y_iou_threshold,
        ),
    }
    effective_proxy_config_file = proxy_env.get("PROXY_CONFIG_FILE") if enable_proxy else None
    if enable_proxy:
        ensure_proxy_run_support(repo_root=repo_root)
    ensure_a11y_compaction_support(repo_root=repo_root)
    ensure_result_artifact_support(repo_root=repo_root)
    run_id = run_id or make_run_id("osworld")
    result_dir = run_results_path(run_id, repo_root)
    ensure_result_dir_writable(result_dir)
    name = run_selector_metadata_name(suite=suite, domain=domain, taskset=taskset)
    metadata, args, _ = command_for_task_keys(
        task_keys,
        metadata_name=name,
        provider=provider,
        observation_type=observation_type,
        model=model,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_trajectory_length=max_trajectory_length,
        headless=headless,
        repo_root=repo_root,
        write_metadata=True,
        result_dir=result_dir,
    )
    # Keep per-run metadata next to the per-run results and point upstream at it.
    per_run_metadata = run_metadata_path(run_id, name, repo_root)
    write_task_metadata([parse_task(task_key) for task_key in task_keys], per_run_metadata)
    metadata = per_run_metadata
    test_meta_idx = args.index("--test_all_meta_path") + 1
    args[test_meta_idx] = str(metadata.resolve())
    prepare_task_artifact_dirs(
        task_keys,
        observation_type=observation_type,
        model=model,
        result_dir=result_dir,
        repo_root=repo_root,
    )
    ensure_local_openai_provider(model=model, base_url=base_url, repo_root=repo_root)
    write_run_record(
        run_id=run_id,
        command="run",
        tasks=task_keys,
        metadata=metadata,
        results=result_dir,
        status="started",
        observation_type=observation_type,
        model=model,
        enable_proxy=enable_proxy,
        proxy_config_file=effective_proxy_config_file,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_run_id=mlflow_run_id,
        mlflow_run_name=mlflow_run_name,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_enabled=bool(mlflow_tracking_uri or mlflow_experiment_name or mlflow_run_name),
        repo_root=repo_root,
    )
    try:
        run_upstream_command_with_cleanup(args, cwd=root, provider=provider, env=provider_env)
    except subprocess.CalledProcessError:
        write_run_record(
            run_id=run_id,
            command="run",
            tasks=task_keys,
            metadata=metadata,
            results=result_dir,
            status="failed",
            observation_type=observation_type,
            model=model,
            enable_proxy=enable_proxy,
            proxy_config_file=effective_proxy_config_file,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_run_id=mlflow_run_id,
            mlflow_run_name=mlflow_run_name,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_enabled=bool(mlflow_tracking_uri or mlflow_experiment_name or mlflow_run_name),
            repo_root=repo_root,
        )
        raise
    except KeyboardInterrupt:
        write_run_record(
            run_id=run_id,
            command="run",
            tasks=task_keys,
            metadata=metadata,
            results=result_dir,
            status="interrupted",
            observation_type=observation_type,
            model=model,
            enable_proxy=enable_proxy,
            proxy_config_file=effective_proxy_config_file,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_run_id=mlflow_run_id,
            mlflow_run_name=mlflow_run_name,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_enabled=bool(mlflow_tracking_uri or mlflow_experiment_name or mlflow_run_name),
            repo_root=repo_root,
        )
        raise
    write_run_record(
        run_id=run_id,
        command="run",
        tasks=task_keys,
        metadata=metadata,
        results=result_dir,
        status="completed",
        observation_type=observation_type,
        model=model,
        enable_proxy=enable_proxy,
        proxy_config_file=effective_proxy_config_file,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_run_id=mlflow_run_id,
        mlflow_run_name=mlflow_run_name,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_enabled=bool(mlflow_tracking_uri or mlflow_experiment_name or mlflow_run_name),
        repo_root=repo_root,
    )
    return metadata, task_keys, run_id


def benchmark_plan(
    *,
    domains: tuple[str, ...] = (),
    taskset: str | None = None,
    limit_per_domain: int | None = None,
    smoke_candidates: bool = False,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
    headless: bool = True,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    repo_root: str | Path = ".",
) -> str:
    task_groups = benchmark_task_groups(
        domains=domains,
        taskset=taskset,
        limit_per_domain=limit_per_domain,
        smoke_candidates=smoke_candidates,
        repo_root=repo_root,
    )
    lines = [
        "# OSWorld benchmark dry run",
        f"upstream={upstream_path(repo_root)}",
        f"taskset={taskset or ''}",
        f"domains={', '.join(task_groups)}",
        f"limit_per_domain={limit_per_domain if limit_per_domain is not None else ''}",
        f"smoke_candidates={str(smoke_candidates).lower()}",
        f"enable_proxy={str(enable_proxy).lower()}",
        f"base_url={base_url or ''}",
        f"api_key={'set' if api_key is not None else ''}",
    ]
    if taskset:
        total = sum(len(task_keys) for task_keys in task_groups.values())
        lines.append(f"taskset_tasks={total}")
    if enable_proxy:
        lines.append(
            "proxy_config_file="
            + str(resolve_proxy_config_file(proxy_config_file, repo_root=repo_root))
        )
    for domain, task_keys in task_groups.items():
        lines.append("")
        lines.append(f"domain={domain} tasks={len(task_keys)}")
        if not task_keys:
            lines.append("skip=no selected tasks")
            continue
        args = [
            *uv_command(),
            "run",
            "w8-biayn",
            "osworld",
            "run",
        ]
        if taskset or smoke_candidates:
            for task_key in task_keys:
                args.extend(["--task", task_key])
        else:
            args.extend(["--domain", domain])
            if limit_per_domain is not None:
                args.extend(["--limit", str(limit_per_domain)])
        args.extend(
            [
                "--provider",
                provider,
                "--model",
                model,
                *(["--base-url", base_url] if base_url else []),
                *(["--api-key", api_key] if api_key is not None else []),
                "--observation-type",
                observation_type,
                "--max-steps",
                str(max_steps),
                "--max-tokens",
                str(max_tokens),
                "--max-trajectory-length",
                str(max_trajectory_length),
                "--a11y-tree-max-items",
                str(a11y_tree_max_items),
                "--a11y-iou-threshold",
                str(a11y_iou_threshold),
            ]
        )
        if not headless:
            args.append("--headed")
        if enable_proxy:
            args.append("--enable-proxy")
            if proxy_config_file is not None:
                args.extend(["--proxy-config-file", str(proxy_config_file)])
        lines.append(format_command(args))
        if smoke_candidates:
            lines.append("note=--smoke-candidates is applied by benchmark task selection")
    return "\n".join(lines)


def benchmark(
    *,
    domains: tuple[str, ...] = (),
    taskset: str | None = None,
    limit_per_domain: int | None = None,
    smoke_candidates: bool = False,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    base_url: str | None = None,
    api_key: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
    headless: bool = True,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    client_password: str = DEFAULT_CLIENT_PASSWORD,
    dry_run: bool = False,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
    progress_poll_seconds: float = 2.0,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str | None = None,
    mlflow_run_name: str | None = None,
    mlflow_tags: tuple[tuple[str, str], ...] = (),
    repo_root: str | Path = ".",
) -> BenchmarkResult | None:
    if dry_run:
        print(
            benchmark_plan(
                domains=domains,
                taskset=taskset,
                limit_per_domain=limit_per_domain,
                smoke_candidates=smoke_candidates,
                provider=provider,
                observation_type=observation_type,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_steps=max_steps,
                max_tokens=max_tokens,
                max_trajectory_length=max_trajectory_length,
                a11y_tree_max_items=a11y_tree_max_items,
                a11y_iou_threshold=a11y_iou_threshold,
                headless=headless,
                enable_proxy=enable_proxy,
                proxy_config_file=proxy_config_file,
                repo_root=repo_root,
            )
        )
        return None

    task_groups = benchmark_task_groups(
        domains=domains,
        taskset=taskset,
        limit_per_domain=limit_per_domain,
        smoke_candidates=smoke_candidates,
        repo_root=repo_root,
    )
    total_tasks = sum(len(task_keys) for task_keys in task_groups.values())
    start_time = time.monotonic()

    def emit_progress(*, completed_tasks: int, current_domain: str | None = None) -> None:
        if progress_callback is None:
            return
        elapsed_seconds = time.monotonic() - start_time
        progress_callback(
            BenchmarkProgress(
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                remaining_tasks=max(total_tasks - completed_tasks, 0),
                elapsed_seconds=elapsed_seconds,
                eta_seconds=estimate_time_left_seconds(
                    completed_tasks=completed_tasks,
                    total_tasks=total_tasks,
                    elapsed_seconds=elapsed_seconds,
                ),
                current_domain=current_domain,
            )
        )

    results: list[BenchmarkDomainResult] = []
    completed_so_far = 0
    emit_progress(completed_tasks=0)

    with ensure_tracker(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        mlflow_tags=mlflow_tags,
    ) as tracker:
        run_info = tracker.run_info
        tracker.log_param("model", model)
        tracker.log_param("observation_type", observation_type)
        tracker.log_param("max_steps", max_steps)
        tracker.log_param("max_tokens", max_tokens)
        tracker.log_param(
            "task_selection_domains",
            ",".join(domains) if domains else "all",
        )
        tracker.log_param("taskset", taskset or "")
        tracker.log_param("limit_per_domain", limit_per_domain if limit_per_domain is not None else "")
        tracker.log_param("smoke_candidates", smoke_candidates)
        tracker.log_param("provider", provider)
        tracker.log_param("headless", headless)
        tracker.log_param("enable_proxy", enable_proxy)
        tracker.log_param("base_url_mode", "provided" if base_url else "default")
        tracker.log_param("proxy_config_file", str(proxy_config_file or ""))
        tracker.log_metric("benchmark/domain_count", len(task_groups))

        for idx, (domain, task_keys) in enumerate(task_groups.items()):
            if not task_keys:
                continue
            run_id = make_run_id("osworld")
            result_dir = run_results_path(run_id, repo_root)
            run_state: dict[str, object] = {}

            def run_domain() -> None:
                try:
                    run_state["value"] = run(
                        tasks=task_keys,
                        provider=provider,
                        observation_type=observation_type,
                        model=model,
                        base_url=base_url,
                        api_key=api_key,
                        max_steps=max_steps,
                        max_tokens=max_tokens,
                        max_trajectory_length=max_trajectory_length,
                        a11y_tree_max_items=a11y_tree_max_items,
                        a11y_iou_threshold=a11y_iou_threshold,
                        headless=headless,
                        enable_proxy=enable_proxy,
                        proxy_config_file=proxy_config_file,
                        client_password=client_password,
                        mlflow_tracking_uri=mlflow_tracking_uri,
                        mlflow_run_id=run_info.run_id,
                        mlflow_run_name=run_info.run_name or mlflow_run_name,
                        mlflow_experiment_name=run_info.experiment_name or mlflow_experiment,
                        dry_run=False,
                        run_id=run_id,
                        repo_root=repo_root,
                    )
                except BaseException as exc:  # pragma: no cover - propagated after join
                    run_state["error"] = exc

            worker = threading.Thread(target=run_domain, daemon=True)
            worker.start()
            while worker.is_alive():
                summary = summarize_task_results(
                    task_keys,
                    observation_type=observation_type,
                    model=model,
                    result_dir=result_dir,
                    repo_root=repo_root,
                )
                emit_progress(completed_tasks=completed_so_far + summary.completed, current_domain=domain)
                worker.join(progress_poll_seconds)
            worker.join()
            error = run_state.get("error")
            if isinstance(error, BaseException):
                raise error
            _metadata, run_tasks, finished_run_id = run_state["value"]
            run_id = finished_run_id
            if run_id is None:
                continue
            record = read_run_record(run_id, repo_root=repo_root)
            result_dir = Path(record.get("results") or run_results_path(run_id, repo_root))
            summary = summarize_task_results(
                run_tasks,
                observation_type=observation_type,
                model=model,
                result_dir=result_dir,
                repo_root=repo_root,
            )
            completed_so_far += summary.completed
            emit_progress(completed_tasks=completed_so_far, current_domain=domain)
            tracker.log_metric(f"domain/{domain}/completed", summary.completed)
            tracker.log_metric(f"domain/{domain}/successes", summary.successes)
            tracker.log_metric(f"domain/{domain}/failures", summary.failures)
            if summary.average_score is not None:
                tracker.log_metric(f"domain/{domain}/avg_score", summary.average_score)
            tracker.log_metric(f"domain/{domain}/task_count", len(run_tasks))
            tracker.log_metric(f"domain/{domain}/step_index", idx)

            results.append(
                BenchmarkDomainResult(
                    domain=domain,
                    run_id=run_id,
                    tasks=run_tasks,
                    completed=summary.completed,
                    successes=summary.successes,
                    failures=summary.failures,
                    average_score=summary.average_score,
                    results=result_dir,
                )
            )
        emit_progress(completed_tasks=completed_so_far)
        final = BenchmarkResult(tuple(results))
        tracker.log_metric("benchmark/completed", final.completed)
        tracker.log_metric("benchmark/successes", final.successes)
        tracker.log_metric("benchmark/failures", final.failures)
        tracker.log_metric("benchmark/task_count", final.total_tasks)
        if final.average_score is not None:
            tracker.log_metric("benchmark/average_score", final.average_score)
        tracker.log_param("benchmark/run_id", run_info.run_id or "")
        tracker.log_param("benchmark/run_name", run_info.run_name or "")
        tracker.log_param("benchmark/experiment", run_info.experiment_name or "")
        return final

    return BenchmarkResult(tuple(results))


def smoke_plan(
    *,
    task: str = DEFAULT_OSWORLD_TASK,
    suite: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
    headless: bool = True,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    repo_root: str | Path = ".",
) -> str:
    metadata, args, task_keys = smoke_command(
        task=task,
        suite=suite,
        provider=provider,
        observation_type=observation_type,
        model=model,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_trajectory_length=max_trajectory_length,
        headless=headless,
        repo_root=repo_root,
        write_metadata=False,
    )
    proxy_detail = (
        str(resolve_proxy_config_file(proxy_config_file, repo_root=repo_root))
        if enable_proxy
        else ""
    )
    lines = [
        "# OSWorld smoke dry run",
        f"upstream={upstream_path(repo_root)}",
        f"suite={suite or ''}",
        f"tasks={', '.join(task_keys)}",
        f"metadata={metadata}",
        f"results={results_path(repo_root)}",
        f"enable_proxy={str(enable_proxy).lower()}",
        f"proxy_config_file={proxy_detail}",
        f"a11y_tree_max_items={a11y_tree_max_items}",
        f"a11y_iou_threshold={a11y_iou_threshold}",
        f"cd {upstream_path(repo_root)}",
        format_command(args),
    ]
    return "\n".join(lines)


def smoke(
    *,
    task: str = DEFAULT_OSWORLD_TASK,
    suite: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    observation_type: str = DEFAULT_OBSERVATION_TYPE,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_trajectory_length: int = 1,
    a11y_tree_max_items: int = DEFAULT_A11Y_TREE_MAX_ITEMS,
    a11y_iou_threshold: float = DEFAULT_A11Y_IOU_THRESHOLD,
    headless: bool = True,
    enable_proxy: bool = False,
    proxy_config_file: str | Path | None = None,
    client_password: str = DEFAULT_CLIENT_PASSWORD,
    dry_run: bool = False,
    repo_root: str | Path = ".",
) -> Path:
    root = upstream_path(repo_root)
    if dry_run:
        print(
            smoke_plan(
                task=task,
                suite=suite,
                provider=provider,
                observation_type=observation_type,
                model=model,
                max_steps=max_steps,
                max_tokens=max_tokens,
                max_trajectory_length=max_trajectory_length,
                a11y_tree_max_items=a11y_tree_max_items,
                a11y_iou_threshold=a11y_iou_threshold,
                headless=headless,
                enable_proxy=enable_proxy,
                proxy_config_file=proxy_config_file,
                repo_root=repo_root,
            )
        )
        return (
            metadata_path(f"suite-{suite}", repo_root=repo_root)
            if suite
            else one_task_metadata_path(parse_task(task), repo_root)
        )
    if not root.exists():
        raise FileNotFoundError(
            "OSWorld upstream is missing. Run `uv run w8-biayn upstreams clone osworld` first."
        )
    task_keys = tasks_for_suite(suite) if suite else (task,)
    for task_key in task_keys:
        task_ref = parse_task(task_key)
        config = task_config_path(task_ref, repo_root)
        if not config.exists():
            raise FileNotFoundError(f"OSWorld task config not found: {config}")
    proxy_env = proxy_environment(
        enable_proxy=enable_proxy,
        proxy_config_file=proxy_config_file,
        client_password=client_password,
        repo_root=repo_root,
    )
    provider_env = {
        **proxy_env,
        **a11y_compaction_environment(
            a11y_tree_max_items=a11y_tree_max_items,
            a11y_iou_threshold=a11y_iou_threshold,
        ),
    }
    if enable_proxy:
        ensure_proxy_run_support(repo_root=repo_root)
    ensure_a11y_compaction_support(repo_root=repo_root)
    results_path(repo_root).mkdir(parents=True, exist_ok=True)
    metadata, args, _ = smoke_command(
        task=task,
        suite=suite,
        provider=provider,
        observation_type=observation_type,
        model=model,
        max_steps=max_steps,
        max_tokens=max_tokens,
        max_trajectory_length=max_trajectory_length,
        headless=headless,
        repo_root=repo_root,
        write_metadata=True,
    )
    ensure_local_openai_provider(model=model, repo_root=repo_root)
    run_upstream_command_with_cleanup(args, cwd=root, provider=provider, env=provider_env)
    return metadata
