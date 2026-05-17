"""SkyRL-Gym environment for packaged Harbor DOMDiff tasks."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
except Exception:  # pragma: no cover - SkyRL-Gym is installed only in trainer runtime.
    BaseTextEnv = object  # type: ignore
    BaseTextEnvStepOutput = dict  # type: ignore
    ConversationType = list  # type: ignore

from w8_biayn.harbor.docker_runner import HarborDockerTaskRunner
from w8_biayn.harbor.domdiff_eval import evaluate_preview_url
from w8_biayn.harbor.tasks import load_task

SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.DOTALL | re.IGNORECASE)
FENCED_RE = re.compile(r"```(?:bash|sh|shell)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class HarborSkyRLEnv(BaseTextEnv):
    """One-turn SkyRL environment that scores a Harbor task through DOMDiff."""

    def __init__(self, env_config: Any | None = None, extras: dict[str, Any] | None = None):
        super().__init__()
        self.env_config = env_config or {}
        self.extras = extras or {}
        self.task_path = Path(str(_get_required(self.extras, "task_path"))).resolve()
        self.task_id = str(self.extras.get("task_id") or self.task_path.name)
        self.task = load_task(self.task_id, self.task_path.parent)
        self.chromiumrl_url = str(
            self.extras.get("chromiumrl_url")
            or os.environ.get("CHROMIUMRL_URL")
            or os.environ.get("CHROMIUMRL_API_URL")
            or ""
        )
        self.oracle = _as_bool(self.extras.get("oracle", os.environ.get("W8_HARBOR_ORACLE", "0")))
        self.keep_containers = _as_bool(
            self.extras.get("keep_containers", os.environ.get("W8_HARBOR_KEEP_CONTAINERS", "0"))
        )
        self.done = False
        self.last_reward = 0.0
        self.last_metadata: dict[str, Any] = {}

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        metadata = {
            "task_id": self.task_id,
            "task_path": str(self.task_path),
            "preview_path": self.task.preview_path,
            "oracle": self.oracle,
            "chromiumrl_enabled": bool(self.chromiumrl_url),
        }
        return list(prompt), metadata

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.done:
            return {"observations": [], "reward": self.last_reward, "done": True, "metadata": self.last_metadata}

        script = _extract_solution_script(action)
        runner = HarborDockerTaskRunner(
            chromiumrl_url=self.chromiumrl_url,
            oracle=self.oracle,
            keep_containers=self.keep_containers,
        )
        outcome = runner.run(
            self.task_id,
            task_root=self.task_path.parent,
            solution_script=script,
        )
        metadata: dict[str, Any] = {
            "task_id": self.task_id,
            "terminal_state": outcome.terminal_state,
            "preview_url": outcome.preview_url,
            "oracle": self.oracle,
            "chromiumrl_enabled": bool(self.chromiumrl_url),
            "docker_image": outcome.image,
            "docker_container": outcome.container,
        }

        reward = float(outcome.reward_doc.get("reward", 0.0) or 0.0)
        if outcome.terminal_state == "ready_for_chromiumrl" and self.chromiumrl_url:
            eval_doc, rubric_eval = evaluate_preview_url(
                chromiumrl_url=self.chromiumrl_url,
                task_dir=self.task_path,
                preview_url=outcome.preview_url,
                preview_headers=outcome.preview_headers,
            )
            reward = float(rubric_eval.score)
            metadata.update(
                {
                    "chromiumrl_status": "live_evaluated",
                    "request_id": eval_doc.get("request_id"),
                    "domdiff_total": rubric_eval.domdiff_total,
                    "rubric_passed": rubric_eval.passed,
                    "group_scores": rubric_eval.group_scores,
                }
            )

        self.done = True
        self.last_reward = reward
        self.last_metadata = metadata
        return {"observations": [], "reward": reward, "done": True, "metadata": metadata}

    def get_metrics(self) -> dict[str, Any]:
        return {
            "harbor_reward": self.last_reward,
            "harbor_task_id": self.task_id,
            "harbor_oracle": self.oracle,
            **self.last_metadata,
        }


def _extract_solution_script(action: str) -> str:
    text = action or ""
    match = SOLUTION_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"
    match = FENCED_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get_required(mapping: dict[str, Any], key: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"HarborSkyRLEnv requires extras.{key}")
    return value
