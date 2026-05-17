"""rLLM AgentFlow for Harbor DOMDiff smoke rollouts."""

from __future__ import annotations

from typing import Any


class HarborPromptAgentFlow:
    """One-call agent flow that records model tokens for an infrastructure smoke."""

    name = "harbor_domdiff_prompt"

    async def arun(self, task: Any, config: Any):
        from openai import AsyncOpenAI
        from rllm.types import Episode, Trajectory

        metadata = getattr(task, "metadata", task)
        instruction = _task_instruction(metadata)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are solving a browser UI repository task. "
                    "Return a concise implementation plan before editing."
                ),
            },
            {"role": "user", "content": instruction},
        ]
        sampling_params = {
            k: v
            for k, v in dict(getattr(config, "sampling_params", {}) or {}).items()
            if k not in {"top_k"}
        }
        sampling_params.setdefault("max_tokens", 512)
        client = AsyncOpenAI(base_url=getattr(config, "base_url"), api_key="w8-biayn")
        response = await client.chat.completions.create(
            model=getattr(config, "model"),
            messages=messages,
            **sampling_params,
        )
        content = response.choices[0].message.content or ""
        return Episode(
            task=metadata,
            trajectories=[
                Trajectory(
                    name=self.name,
                    output=content,
                    metadata={"model_response": content},
                )
            ],
            artifacts={"model_response": content},
        )


def _task_instruction(task: dict[str, Any]) -> str:
    for key in ("task", "instruction", "question"):
        value = task.get(key)
        if value:
            return str(value)
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("task") or metadata.get("instruction")
        if value:
            return str(value)
    return "Solve the Harbor browser task."
