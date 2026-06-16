"""SkyRL-Gym environment for C++ performance rewards."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from w8_biayn.cpp_perf.reward import compute_reward, extract_code_block
from w8_biayn.cpp_perf.schema import CppTask
from w8_biayn.cpp_perf.skyrl_dataset import CPP_PERF_ENV_ID

try:  # pragma: no cover - exercised in real SkyRL runtime
    from skyrl_gym.envs import register
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:  # Local unit tests do not install SkyRL.
    register = None

    class BaseTextEnv:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.turns = 0
            self.max_turns = 1

    BaseTextEnvStepOutput = dict[str, Any]  # type: ignore[assignment]


class CppPerfEnv(BaseTextEnv):
    """One-turn environment that scores generated C++ with the local reward harness."""

    def __init__(self, env_config: dict[str, Any] | None = None, extras: dict[str, Any] | None = None):
        super().__init__()
        self.env_config = env_config or {}
        self.extras = extras or {}
        self.extra_info = self.extras.get("extra_info", self.extras)
        self.max_turns = int(self.extra_info.get("max_turns", self.extras.get("max_turns", 1)))
        self.task = CppTask.read_json(self._resolve_task_path(self.extra_info["task_path"]))
        self.last_metrics: dict[str, Any] = {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        runner = self.env_config.get("runner")
        code = None
        try:
            breakdown = compute_reward(self.task, action, runner=runner if callable(runner) else None)
            try:
                code = extract_code_block(action)
            except ValueError:
                pass
        except Exception as exc:  # pragma: no cover - guards real rollout workers
            self.last_metrics = {
                "reward": -1.0,
                "reason": "reward_exception",
                "compile_error": False,
                "sanitizer_error": False,
                "tests_passed": 0,
                "tests_total": 0,
                "runtime_cpu_ns": None,
                "runtime_wall_ns": None,
                "reference_runtime_cpu_ns": None,
                "reference_runtime_wall_ns": None,
                "runtime_speedup": None,
                "candidate_bytes": 0,
                "exception": str(exc),
            }
            return {
                "observations": [],
                "reward": -1.0,
                "done": True,
                "metadata": self.last_metrics | {"task_id": self.task.task_id},
                "postprocessed_action": None,
            }
        harness = breakdown.harness
        self.last_metrics = {
            "reward": breakdown.reward,
            "reason": breakdown.reason,
            "compile_error": bool(harness.compile_error) if harness else False,
            "sanitizer_error": bool(harness.sanitizer_error) if harness else False,
            "tests_passed": harness.tests_passed if harness else 0,
            "tests_total": harness.tests_total if harness else 0,
            "runtime_cpu_ns": harness.runtime_cpu_ns if harness else None,
            "runtime_wall_ns": harness.runtime_wall_ns if harness else None,
            "reference_runtime_cpu_ns": harness.reference_runtime_cpu_ns if harness else None,
            "reference_runtime_wall_ns": harness.reference_runtime_wall_ns if harness else None,
            "runtime_speedup": harness.runtime_speedup if harness else None,
            "candidate_bytes": len(code.encode("utf-8")) if code is not None else 0,
        }
        return {
            "observations": [],
            "reward": breakdown.reward,
            "done": True,
            "metadata": self.last_metrics | {"task_id": self.task.task_id},
            "postprocessed_action": code,
        }

    def get_metrics(self) -> dict[str, Any]:
        return dict(self.last_metrics)

    @staticmethod
    def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if not metrics:
            return {}
        numeric: dict[str, list[float]] = {}
        for item in metrics:
            for key, value in item.items():
                if isinstance(value, bool):
                    numeric.setdefault(key, []).append(float(value))
                elif isinstance(value, (int, float)) and value is not None:
                    numeric.setdefault(key, []).append(float(value))
        return {key: sum(values) / len(values) for key, values in numeric.items() if values}

    def _resolve_task_path(self, task_path: str) -> Path:
        path = Path(task_path)
        if path.is_absolute():
            return path
        data_dir = Path(os.environ.get("W8_BIAYN_DATA_DIR", "."))
        candidate = data_dir / path
        if candidate.exists():
            return candidate
        return Path.cwd() / path


def register_cpp_perf_env() -> None:
    """Register the C++ performance env inside the current SkyRL process."""

    if register is None:  # pragma: no cover - local tests assert this branch indirectly
        raise ImportError("skyrl_gym is required to register the cpp-perf environment")
    register(id=CPP_PERF_ENV_ID, entry_point="w8_biayn.integrations.cpp_perf_env:CppPerfEnv")
