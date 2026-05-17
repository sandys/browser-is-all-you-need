"""rLLM evaluator that scores Harbor tasks with live DOMDiff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .docker_runner import HarborDockerTaskRunner
from .domdiff_eval import evaluate_preview_url


class HarborDomdiffEvaluator:
    def __init__(
        self,
        *,
        chromiumrl_url: str,
        oracle: bool = True,
        keep_containers: bool = False,
    ) -> None:
        self.chromiumrl_url = chromiumrl_url
        self.oracle = oracle
        self.keep_containers = keep_containers

    def evaluate(self, task: dict[str, Any], episode: Any):
        from rllm.eval.types import EvalOutput, Signal

        task_path = _task_path(task)
        task_id = task_path.name
        runner = HarborDockerTaskRunner(
            chromiumrl_url=self.chromiumrl_url,
            oracle=self.oracle,
            keep_containers=self.keep_containers,
        )
        outcome = runner.run(task_id, task_root=task_path.parent)
        metadata: dict[str, Any] = {
            "harbor_task": task_id,
            "terminal_state": outcome.terminal_state,
            "preview_url": outcome.preview_url,
            "oracle": self.oracle,
        }
        if outcome.terminal_state != "ready_for_chromiumrl":
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal("harbor_ready", 0.0)],
                metadata=metadata,
            )
        if not self.chromiumrl_url:
            return EvalOutput(
                reward=0.0,
                is_correct=False,
                signals=[Signal("harbor_ready", 1.0), Signal("chromiumrl_live", 0.0)],
                metadata=metadata,
            )

        eval_doc, rubric_eval = evaluate_preview_url(
            chromiumrl_url=self.chromiumrl_url,
            task_dir=task_path,
            preview_url=outcome.preview_url,
            preview_headers=outcome.preview_headers,
        )
        metadata.update(
            {
                "chromiumrl_status": "live_evaluated",
                "request_id": eval_doc.get("request_id"),
                "domdiff_total": rubric_eval.domdiff_total,
                "rubric_passed": rubric_eval.passed,
                "group_scores": rubric_eval.group_scores,
            }
        )
        return EvalOutput(
            reward=float(rubric_eval.score),
            is_correct=bool(rubric_eval.passed),
            signals=[
                Signal("harbor_ready", 1.0),
                Signal("chromiumrl_live", 1.0),
                Signal("domdiff_total", float(rubric_eval.domdiff_total)),
                Signal("rubric_passed", 1.0 if rubric_eval.passed else 0.0),
            ],
            metadata=metadata,
        )


def _task_path(task: dict[str, Any]) -> Path:
    for key in ("task_path",):
        value = task.get(key)
        if value:
            return Path(str(value)).resolve()
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("task_path")
        if value:
            return Path(str(value)).resolve()
    raise ValueError(f"Harbor task row is missing task_path: {task.keys()}")
