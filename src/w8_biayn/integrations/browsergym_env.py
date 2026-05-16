"""SkyRL-Gym environment wrapper for BrowserGym."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
except Exception:  # pragma: no cover - used only when SkyRL-Gym is absent in lightweight tests.
    BaseTextEnv = object  # type: ignore
    BaseTextEnvStepOutput = dict  # type: ignore
    ConversationType = list  # type: ignore


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)


def serialize_observation(observation: Any, *, max_chars: int = 12000) -> str:
    """Serialize a BrowserGym observation into text for the next LLM turn."""
    if observation is None:
        text = "Observation: None"
    elif isinstance(observation, str):
        text = observation
    elif dataclasses.is_dataclass(observation):
        text = json.dumps(dataclasses.asdict(observation), default=str, indent=2)
    elif isinstance(observation, dict):
        text = json.dumps(observation, default=str, indent=2)
    elif hasattr(observation, "__dict__"):
        text = json.dumps(vars(observation), default=str, indent=2)
    else:
        text = str(observation)

    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def extract_action(text: str) -> str:
    """Extract a BrowserGym action from an LLM response."""
    matches = ACTION_RE.findall(text or "")
    if not matches:
        raise ValueError("No BrowserGym action found. Expected <action>...</action>.")
    action = matches[-1].strip()
    if not action:
        raise ValueError("BrowserGym action tag was empty.")
    return action


def import_browsergym_for_env(env_id: str) -> None:
    """Import the BrowserGym package that registers an env id."""
    if env_id.startswith("browsergym/miniwob"):
        import browsergym.miniwob  # noqa: F401
        return
    if "webarena" in env_id:
        try:
            import browsergym.async_webarena  # type: ignore # noqa: F401
        except Exception:
            import browsergym.webarena  # type: ignore # noqa: F401
        return
    import browsergym.core  # type: ignore # noqa: F401


class BrowserGymEnv(BaseTextEnv):
    """Text environment that exposes BrowserGym to SkyRL's generator."""

    def __init__(self, env_config: Any | None = None, extras: dict[str, Any] | None = None):
        super().__init__()
        self.env_config = env_config or {}
        self.extras = extras or {}
        self.env_id = self.extras.get("env_id") or self.extras.get("browsergym_env_id")
        if not self.env_id:
            raise ValueError("BrowserGymEnv requires extras.env_id")

        self.max_turns = int(self.extras.get("max_turns", 8))
        self.observation_max_chars = int(self.extras.get("observation_max_chars", 12000))
        self.turns = 0
        self.env = None
        self.last_reward = 0.0
        self.last_info: dict[str, Any] = {}

    def _make_env(self):
        import gymnasium as gym

        import_browsergym_for_env(str(self.env_id))
        kwargs = dict(self.extras.get("gym_kwargs", {}))
        return gym.make(str(self.env_id), **kwargs)

    def init(self, prompt: ConversationType) -> tuple[ConversationType, dict[str, Any]]:
        self.env = self._make_env()
        observation, info = self.env.reset()
        self.last_info = dict(info or {})
        obs_text = serialize_observation(observation, max_chars=self.observation_max_chars)
        return list(prompt) + [{"role": "user", "content": obs_text}], self.last_info

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.env is None:
            raise RuntimeError("BrowserGymEnv.step() called before init()")
        self.turns += 1

        try:
            parsed_action = extract_action(action)
            observation, reward, terminated, truncated, info = self.env.step(parsed_action)
            done = bool(terminated or truncated or self.turns >= self.max_turns)
            self.last_reward = float(reward or 0.0)
            self.last_info = dict(info or {})
            obs_text = serialize_observation(observation, max_chars=self.observation_max_chars)
            observations = [] if done else [{"role": "user", "content": obs_text}]
            metadata = {
                "env_id": self.env_id,
                "turn": self.turns,
                "action": parsed_action,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **self.last_info,
            }
            return {
                "observations": observations,
                "reward": self.last_reward,
                "done": done,
                "metadata": metadata,
            }
        except Exception as exc:
            done = self.turns >= self.max_turns
            return {
                "observations": [] if done else [{"role": "user", "content": f"Action error: {exc}"}],
                "reward": -1.0,
                "done": done,
                "metadata": {"env_id": self.env_id, "turn": self.turns, "error": str(exc)},
            }

    def get_metrics(self) -> dict[str, Any]:
        return {
            "browsergym_turns": self.turns,
            "browsergym_last_reward": self.last_reward,
            "browsergym_env_id": str(self.env_id),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

