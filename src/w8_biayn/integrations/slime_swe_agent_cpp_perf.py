"""SLIME custom-generate hook for the file-state-scored agentic C++ lane.

Wired as ``--custom-generate-function-path
w8_biayn.integrations.slime_swe_agent_cpp_perf.generate``. Per rollout sample it:

1. serves an :class:`OpenAIAdapter` (once, as a process singleton) that captures
   the served model's sampled tokens/logprobs for GRPO;
2. runs one SWE-agent instance (host-side, via :mod:`swe_agent_driver`) that
   dials that adapter with the session id as its bearer key and edits
   ``candidate.cpp`` in its own Docker container;
3. grades the FINAL file with the hardened, CPU-pinned sandbox
   (:func:`run_in_directory_prewritten`) -- never model text;
4. delivers the correctness-gated child-CPU-ns reward through
   ``adapter.finish_session(reward=...)``, which drains the trajectory tree into
   trainable :class:`Sample` objects.

Reward flows via ``finish_session`` only; there is no ``--custom-rm-path`` for
this lane (SLIME's reward manager would score ``sample.response`` text, but the
trained tail is the agent's turns). The rich per-episode harness breakdown is
stashed on ``sample.metadata["cpp_reward_record"]`` so the existing
``aggregate-debug`` / ``compare`` eval pipeline keeps working unchanged.

All ``slime`` / ``sweagent`` imports are lazy so this module imports in the
control-plane venv (they exist only in the training image / Ray runtime_env).
This mirrors :mod:`w8_biayn.integrations.slime_cpp_perf`, which also duck-types
``args`` / ``sample``.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from w8_biayn import wandb_report
from w8_biayn.cpp_perf.reward import file_state_reward
from w8_biayn.cpp_perf.sandbox import DEFAULT_CPU, DEFAULT_DOCKER_IMAGE, run_in_directory_prewritten
from w8_biayn.cpp_perf.schema import CppTask, HarnessResult
from w8_biayn.integrations import swe_agent_driver
from w8_biayn.integrations.slime_cpp_perf import (
    SANDBOX_CPU_ENV,
    SANDBOX_IMAGE_ENV,
    _resolve_task_path,
    _sample_metadata,
    reward_record_from_breakdown,
)
from w8_biayn.integrations.swe_agent_driver import _int_env, _opt_int_env

ADAPTER_BIND_HOST_ENV = "W8_ADAPTER_BIND_HOST"
ADAPTER_PORT_ENV = "W8_ADAPTER_PORT"
ROLLOUT_GUARD_ENV = "W8_SWE_ROLLOUT_GUARD_SEC"
CONCURRENCY_ENV = "W8_SWE_AGENT_CONCURRENCY"
FORK_MERGE_THRESHOLD_ENV = "W8_SWE_FORK_MERGE_MAX_RESPONSE_TOKENS"

DEFAULT_ADAPTER_BIND_HOST = "127.0.0.1"
DEFAULT_ADAPTER_PORT = 18011
DEFAULT_ROLLOUT_GUARD_SEC = 1800
DEFAULT_CONCURRENCY = 4

_ADAPTER_SERVICE: "_AdapterService | None" = None
_SWE_SEM: "asyncio.Semaphore | None" = None


class _AdapterService:
    """Process singleton: an OpenAIAdapter served over HTTP on loopback.

    Model calls originate in the host-side SWE-agent process (its container only
    runs shell actions), so the adapter is reached at 127.0.0.1 -- no
    ``ADAPTER_PUBLIC_HOST`` reverse-connection routing is needed. ``adapter_url``
    already carries the ``/v1`` suffix that SWE-agent's LiteLLM ``openai/``
    provider posts under.
    """

    def __init__(self, args: Any) -> None:
        from slime.agent.adapters import OpenAIAdapter
        from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
        from slime.utils.processing_utils import load_tokenizer

        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        bind_host = os.environ.get(ADAPTER_BIND_HOST_ENV, DEFAULT_ADAPTER_BIND_HOST)
        port = _int_env(os.environ, ADAPTER_PORT_ENV, DEFAULT_ADAPTER_PORT)
        fork_threshold = _opt_int_env(os.environ, FORK_MERGE_THRESHOLD_ENV)

        self.adapter = OpenAIAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=fork_threshold,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=bind_host,
            port=port,
            thread_name="w8-swe-openai-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        self.adapter_url = f"http://{bind_host}:{self.app_handle.port}/v1"

        self.swe_image = os.environ.get(swe_agent_driver.SWE_AGENT_IMAGE_ENV, swe_agent_driver.DEFAULT_SWE_AGENT_IMAGE)
        self.model_label = os.environ.get(
            swe_agent_driver.SWE_AGENT_MODEL_LABEL_ENV, swe_agent_driver.DEFAULT_MODEL_LABEL
        )
        self.grader_image = os.environ.get(SANDBOX_IMAGE_ENV, DEFAULT_DOCKER_IMAGE)
        self.grader_cpu = os.environ.get(SANDBOX_CPU_ENV, DEFAULT_CPU)
        self.limits = swe_agent_driver.SweAgentLimits.from_env()
        self.rollout_guard_sec = _int_env(os.environ, ROLLOUT_GUARD_ENV, DEFAULT_ROLLOUT_GUARD_SEC)

        # Live rollout-health telemetry into the stage's W&B run: this process
        # (the SLIME rollout actor) is already attached in shared mode, so
        # RolloutHealth's wandb.log lands next to train/* and rollout/*. Window
        # = one rollout step so the panels line up with rollout/step.
        batch = int(getattr(args, "rollout_batch_size", 0) or 0)
        group = int(getattr(args, "n_samples_per_prompt", 1) or 1)
        wandb_report.define_health_metrics()
        self.health = wandb_report.RolloutHealth(window=(batch * group) or 16)


def _adapter_service(args: Any) -> _AdapterService:
    global _ADAPTER_SERVICE
    if _ADAPTER_SERVICE is None:
        _ADAPTER_SERVICE = _AdapterService(args)
    return _ADAPTER_SERVICE


def _get_semaphore() -> "asyncio.Semaphore":
    global _SWE_SEM
    if _SWE_SEM is None:
        _SWE_SEM = asyncio.Semaphore(_int_env(os.environ, CONCURRENCY_ENV, DEFAULT_CONCURRENCY))
    return _SWE_SEM


def _sampling_get(sampling_params: Any, key: str) -> Any:
    if isinstance(sampling_params, dict):
        return sampling_params.get(key)
    return None


def _grade_file_state(task: CppTask, code: str, *, image: str, cpu: str) -> HarnessResult:
    """Write the agent's final code to a fresh scratch dir and grade the FILE."""

    with TemporaryDirectory(prefix="w8-swe-grade-") as tmp:
        scratch = Path(tmp)
        (scratch / "candidate.cpp").write_text(code, encoding="utf-8")
        return run_in_directory_prewritten(task, scratch, image=image, cpu=cpu)


async def generate(args: Any, base_sample: Any, sampling_params: Any) -> list[Any]:
    """Per-sample agentic rollout: SWE-agent edits, we grade the file, reward flows."""

    state = _adapter_service(args)
    metadata = _sample_metadata(base_sample)
    started = time.monotonic()

    def _abort(reason: str, *, task_id: Any = None, error: str = "") -> list[Any]:
        _health_add(
            state,
            {
                "task_id": task_id or metadata.get("task_id"),
                "reward": 0.0,
                "abort_reason": reason,
                "wall_time_s": time.monotonic() - started,
            },
        )
        return _abort_result(base_sample, reason, error=error)

    task_path = metadata.get("task_path")
    if not task_path:
        return _abort("missing_task_path")
    try:
        task = CppTask.read_json(_resolve_task_path(str(task_path), metadata=metadata))
    except Exception as exc:  # pragma: no cover - guards real rollout data bugs
        return _abort(f"task_load_error:{type(exc).__name__}")

    sid = _session_id(base_sample, task.task_id)
    base_sample.session_id = sid
    state.adapter.open_session(
        sid,
        sampling_defaults=dict(sampling_params or {}),
        max_context_tokens=state.max_context_len,
    )
    try:
        async with _get_semaphore():
            async with asyncio.timeout(state.rollout_guard_sec):
                with TemporaryDirectory(prefix="w8-swe-") as work_root:
                    extract = await asyncio.to_thread(
                        swe_agent_driver.run_swe_agent_and_extract,
                        task,
                        task.prompt_code,
                        adapter_url=state.adapter_url,
                        sid=sid,
                        temperature=_sampling_get(sampling_params, "temperature"),
                        top_p=_sampling_get(sampling_params, "top_p"),
                        image=state.swe_image,
                        model_label=state.model_label,
                        limits=state.limits,
                        work_root=work_root,
                    )
                    harness = await asyncio.to_thread(
                        _grade_file_state,
                        task,
                        extract.candidate_code,
                        image=state.grader_image,
                        cpu=state.grader_cpu,
                    )
        breakdown = file_state_reward(harness, code=extract.candidate_code)
        samples = await state.adapter.finish_session(sid, base_sample=base_sample, reward=float(breakdown.reward))
        if not samples:
            return _abort("adapter_session_empty", task_id=task.task_id)

        record = reward_record_from_breakdown(base_sample, task, breakdown)
        _health_add(
            state,
            {
                "task_id": task.task_id,
                "reward": float(breakdown.reward),
                "format_valid": bool(breakdown.format_valid),
                "all_tests_pass": bool(harness.all_tests_pass),
                "compile_error": bool(harness.compile_error),
                "timeout": bool(harness.timeout),
                "agent_steps": extract.steps,
                "wall_time_s": time.monotonic() - started,
            },
        )
        for sample in samples:
            sample.metadata = {
                **(getattr(sample, "metadata", None) or {}),
                "cpp_reward_record": record,
                "swe_exit_status": extract.exit_status,
                "agent_steps": extract.steps,
                # SLIME --log-multi-turn reads round_number for multi_turn_metric/*.
                "round_number": int(extract.steps or 0),
            }
        return samples
    except asyncio.TimeoutError:
        return _abort("wall_clock_timeout", task_id=task.task_id)
    except Exception as exc:  # pragma: no cover - guards real rollout workers
        return _abort(f"exception:{type(exc).__name__}", task_id=task.task_id, error=str(exc))
    finally:
        await _drop_session_quietly(state, sid)


async def _drop_session_quietly(state: Any, sid: str) -> None:
    try:
        await state.adapter.drop_session(sid)
    except Exception:  # pragma: no cover - cleanup only, must never raise
        pass


def _health_add(state: Any, row: dict[str, Any]) -> None:
    """Feed the live rollout-health aggregator; telemetry must never break rollouts."""

    try:
        state.health.add(row)
    except Exception:  # pragma: no cover - best-effort telemetry
        pass


def _session_id(sample: Any, task_id: str) -> str:
    existing = getattr(sample, "session_id", None)
    if existing:
        return str(existing)
    index = getattr(sample, "index", None)
    group_index = getattr(sample, "group_index", None)
    if index is not None and group_index is not None:
        return f"w8swe-{task_id}-{index}-{group_index}"
    return f"w8swe-{task_id}-{secrets.token_hex(8)}"


def _aborted_status() -> Any:
    try:
        from slime.utils.types import Sample

        return Sample.Status.ABORTED
    except Exception:
        return "aborted"


def _abort_result(sample: Any, reason: str, *, error: str = "") -> list[Any]:
    """Mark ``sample`` aborted in place and return it in the fan-out list shape.

    ``error`` carries the exception MESSAGE (path, errno, ...) into metadata:
    the reason alone ("exception:FileExistsError") once cost a full paid smoke
    because the colliding path was invisible everywhere.
    """

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = _aborted_status()
    sample.metadata = {**(getattr(sample, "metadata", None) or {}), "abort_reason": reason}
    if error:
        sample.metadata["abort_error"] = error[:500]
    return [sample]
