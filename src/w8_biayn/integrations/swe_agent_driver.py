"""Drive one SWE-agent instance against a served model and extract candidate.cpp.

SWE-agent (OpenAI protocol via LiteLLM) dials our OpenAIAdapter at ``api_base``
with the rollout session id as the bearer ``api_key``; it edits ``candidate.cpp``
inside its own Docker container. We build the environment and agent objects
ourselves and read the file back **before** closing the env, because
``RunSingle.run()`` closes the container first and we would lose the file.

All ``sweagent`` imports are lazy so this module imports without the dependency
(``sweagent``/``swerex``/``litellm`` live only in the training image, not the
control-plane venv). The pure parts -- repo seeding, config assembly, limits --
are unit-tested locally; the ``SWEEnv``/agent run is validated on the M1 loop
proof against a served model.

Verified against the SWE-agent source (study clone):

* single instance: ``SWEEnv.from_config(cfg.env)`` + ``get_agent_from_config``,
  ``env.start()``, ``agent.run(problem_statement=..., env=env, output_dir=...)``,
  then ``env.read_file(f"/{repo_name}/candidate.cpp")`` before ``env.close()``.
* model (``GenericAPIModelConfig``): ``name="openai/<label>"`` +
  ``api_base``/``api_key``. Both ``per_instance_cost_limit`` and
  ``total_cost_limit`` must be ``0.0`` -- an unknown local model makes
  ``litellm.completion_cost`` raise, and any nonzero limit re-raises and kills
  the run on the first query. Bound the loop with ``per_instance_call_limit`` +
  ``total_execution_timeout`` + the adapter ``max_context_tokens`` instead.
  ``temperature``/``top_p`` MUST be taken from the rollout sampling params:
  SWE-agent defaults ``temperature=0.0`` which would collapse GRPO to greedy.
* action format ``thought_action`` (``config/default_backticks.yaml``): plain
  text turns, no ``tools=`` on the wire, the cleanest round-trip through the
  OpenAIAdapter so the trajectory manager keeps ~1 Sample per session.
* repo: SWE-agent requires a clean git repo; seed one per sample with
  ``candidate.cpp = v0`` committed and point ``LocalRepoConfig`` at it
  (``base_commit="HEAD"``). Never commit the hidden tests or the oracle.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from w8_biayn.cpp_perf.sandbox import DEFAULT_SWE_AGENT_IMAGE

DEFAULT_MODEL_LABEL = "slime-actor"
CANDIDATE_FILENAME = "candidate.cpp"

SWE_AGENT_IMAGE_ENV = "W8_SWE_AGENT_IMAGE"
SWE_AGENT_MODEL_LABEL_ENV = "W8_SWE_AGENT_MODEL_LABEL"
SWE_AGENT_PARSE_FUNCTION_ENV = "W8_SWE_AGENT_PARSE_FUNCTION"
SWE_AGENT_CALL_LIMIT_ENV = "W8_SWE_AGENT_PER_INSTANCE_CALL_LIMIT"
SWE_AGENT_EXEC_TIMEOUT_ENV = "W8_SWE_AGENT_TOTAL_EXECUTION_TIMEOUT"
SWE_AGENT_MAX_INPUT_TOKENS_ENV = "W8_SWE_AGENT_MAX_INPUT_TOKENS"
SWE_AGENT_MAX_OUTPUT_TOKENS_ENV = "W8_SWE_AGENT_MAX_OUTPUT_TOKENS"

# thought_action gives plain-text assistant turns (no function-calling), the
# cleanest token round-trip through the OpenAIAdapter. Shipped as the
# byte-identical-to-default config that only flips parse_function.
DEFAULT_PARSE_FUNCTION = "thought_action"
DEFAULT_CONFIG_FILENAME = "default_backticks.yaml"


@dataclass(frozen=True)
class SweAgentLimits:
    """Loop bounds for one SWE-agent run (cost limiting is always disabled)."""

    per_instance_call_limit: int = 40
    total_execution_timeout: int = 1200
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    parse_function: str = DEFAULT_PARSE_FUNCTION

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "SweAgentLimits":
        environ = os.environ if env is None else env
        return cls(
            per_instance_call_limit=_int_env(environ, SWE_AGENT_CALL_LIMIT_ENV, 40),
            total_execution_timeout=_int_env(environ, SWE_AGENT_EXEC_TIMEOUT_ENV, 1200),
            max_input_tokens=_opt_int_env(environ, SWE_AGENT_MAX_INPUT_TOKENS_ENV),
            max_output_tokens=_opt_int_env(environ, SWE_AGENT_MAX_OUTPUT_TOKENS_ENV),
            parse_function=environ.get(SWE_AGENT_PARSE_FUNCTION_ENV, DEFAULT_PARSE_FUNCTION) or DEFAULT_PARSE_FUNCTION,
        )


@dataclass(frozen=True)
class ExtractResult:
    """Outcome of one SWE-agent run: the final file plus run diagnostics."""

    candidate_code: str
    exit_status: str | None
    steps: int
    submission: str | None


def _int_env(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _opt_int_env(env: dict[str, str], key: str) -> int | None:
    raw = env.get(key)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


COMPILE_HELPER = "#!/usr/bin/env bash\nset -e\ng++ -O3 -std=c++20 candidate.cpp -o candidate\n"


def _safe_repo_name(sid: str) -> str:
    """A filesystem-safe, per-session repo basename.

    Under the local deployment swerex copies the repo to the FS root
    ``/{repo_name}`` (``LocalRepoConfig.copy``), so concurrent rollouts on one
    worker must not share a basename or they collide at ``/repo``. Key it off the
    session id, which is unique per sample.
    """

    safe = re.sub(r"[^A-Za-z0-9_-]", "-", sid).strip("-") or "session"
    return f"repo-{safe}"


def _materialize_repo(work_dir: str | Path, v0_code: str, *, task: Any | None = None) -> Path:
    """Create a clean git repo whose candidate.cpp is the slow v0 program.

    SWE-agent requires a valid, clean git repo (it runs ``git reset --hard`` /
    ``git checkout HEAD`` on the base commit). We seed candidate.cpp = v0 and,
    when a task is provided, the visible unit tests plus a compile helper so the
    agent can self-check. Hidden tests and the oracle are never written here.
    """

    repo = Path(work_dir)
    repo.mkdir(parents=True, exist_ok=True)
    (repo / CANDIDATE_FILENAME).write_text(v0_code, encoding="utf-8")
    if task is not None and getattr(task, "unit_tests", None):
        tests_dir = repo / "tests"
        tests_dir.mkdir(exist_ok=True)
        for index, case in enumerate(task.unit_tests):
            (tests_dir / f"{index}.in").write_text(case.input, encoding="utf-8")
            (tests_dir / f"{index}.out").write_text(case.expected, encoding="utf-8")
        (repo / "compile.sh").write_text(COMPILE_HELPER, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.email=agent@w8-biayn.local",
        "-c",
        "user.name=w8-biayn-agent",
        "commit",
        "-q",
        "-m",
        "seed v0 candidate",
    )
    return repo


def problem_statement_text(task: Any) -> str:
    """The task prompt handed to SWE-agent (no hidden tests, no oracle)."""

    visible = getattr(task, "unit_tests", []) or []
    return (
        "Optimize a C++ program for speed while preserving its exact behavior.\n\n"
        "The file `candidate.cpp` in this repository is a correct but slow solution. "
        "Edit `candidate.cpp` in place so it runs faster on large inputs while producing "
        "identical output for every input. Do not change the input/output format or the "
        "program's observable behavior.\n\n"
        f"There are {len(visible)} visible sample test(s) under `tests/` (`<n>.in` / `<n>.out`). "
        "Compile with `bash compile.sh` and check your work against them. Hidden tests and a "
        "reference solution are used to grade you but are not shown.\n\n"
        "Only edit `candidate.cpp`. When you are confident it is correct and faster, submit."
    )


def build_run_config(
    base_config: dict[str, Any],
    *,
    task: Any,
    repo_path: str | Path,
    adapter_url: str,
    sid: str,
    model_label: str = DEFAULT_MODEL_LABEL,
    temperature: float | None,
    top_p: float | None,
    image: str = DEFAULT_SWE_AGENT_IMAGE,
    limits: SweAgentLimits,
) -> dict[str, Any]:
    """Merge our per-sample overrides onto a loaded SWE-agent config dict.

    ``base_config`` is the parsed ``default_backticks.yaml`` (the tool bundle +
    templates + ``parse_function: thought_action``); we only override the model
    endpoint, the repo/deployment, and the problem statement. Returned as a plain
    dict so callers validate it with ``RunSingleConfig.model_validate`` and tests
    can assert the merged fields without importing ``sweagent``.
    """

    model_overrides: dict[str, Any] = {
        "name": f"openai/{model_label}",
        "api_base": adapter_url,
        "api_key": sid,
        # Cost limiting MUST be off: litellm cannot price an unknown local model,
        # and any nonzero limit kills the run on the first query.
        "per_instance_cost_limit": 0.0,
        "total_cost_limit": 0.0,
        "per_instance_call_limit": limits.per_instance_call_limit,
    }
    if temperature is not None:
        model_overrides["temperature"] = temperature
    if top_p is not None:
        model_overrides["top_p"] = top_p
    if limits.max_input_tokens is not None:
        model_overrides["max_input_tokens"] = limits.max_input_tokens
    if limits.max_output_tokens is not None:
        model_overrides["max_output_tokens"] = limits.max_output_tokens

    agent = dict(base_config.get("agent") or {})
    agent["model"] = {**dict(agent.get("model") or {}), **model_overrides}
    tools = dict(agent.get("tools") or {})
    tools["parse_function"] = {"type": limits.parse_function}
    tools["total_execution_timeout"] = limits.total_execution_timeout
    agent["tools"] = tools

    merged = dict(base_config)
    merged["agent"] = agent
    merged["env"] = {
        # LocalDeployment: SWE-agent runs the model's edits/bash in-process on the
        # local node (the rollout worker) -- no swerex sibling container, no port
        # publish, no localhost hop, so nothing needs --network host and Ray is
        # untouched. The final candidate.cpp is graded separately in our hardened
        # run_in_sandbox. (`image` is unused here; reserved for a future container
        # deployment via swerex RemoteDeployment.)
        "deployment": {"type": "local"},
        "repo": {"type": "local", "path": str(repo_path), "base_commit": "HEAD"},
    }
    merged["problem_statement"] = {
        "type": "text",
        "text": problem_statement_text(task),
        "id": str(getattr(task, "task_id", "cpp-task")),
    }
    return merged


def _load_base_config(config_filename: str = DEFAULT_CONFIG_FILENAME) -> dict[str, Any]:
    """Load the shipped SWE-agent thought_action config (lazy sweagent import)."""

    import yaml  # type: ignore

    from sweagent import CONFIG_DIR  # type: ignore

    path = Path(CONFIG_DIR) / config_filename
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run_exit_status(result: Any) -> str | None:
    info = getattr(result, "info", None)
    if isinstance(info, dict):
        return info.get("exit_status")
    return getattr(result, "exit_status", None)


def _run_submission(result: Any) -> str | None:
    info = getattr(result, "info", None)
    if isinstance(info, dict):
        submission = info.get("submission")
        return submission if isinstance(submission, str) else None
    return None


def _run_steps(result: Any) -> int:
    steps = getattr(result, "trajectory", None)
    if isinstance(steps, list):
        return len(steps)
    return 0


def run_swe_agent_and_extract(
    task: Any,
    v0_code: str,
    *,
    adapter_url: str,
    sid: str,
    temperature: float | None,
    top_p: float | None,
    image: str = DEFAULT_SWE_AGENT_IMAGE,
    model_label: str = DEFAULT_MODEL_LABEL,
    limits: SweAgentLimits,
    work_root: str | Path,
) -> ExtractResult:
    """Run one SWE-agent instance and return the final candidate.cpp + diagnostics.

    Uses swerex LocalDeployment: SWE-agent edits/runs the model's bash in-process
    on this node (no sibling container). Needs ``sweagent`` installed and a served
    model at ``adapter_url``. Falls back to the v0 program if extraction fails, so
    the grader always has a well-defined file to score rather than crashing the
    rollout.
    """

    from sweagent.agent.agents import get_agent_from_config  # type: ignore
    from sweagent.environment.swe_env import SWEEnv  # type: ignore
    from sweagent.run.run_single import RunSingleConfig  # type: ignore

    work = Path(work_root)
    repo = _materialize_repo(work / _safe_repo_name(sid), v0_code, task=task)
    base = _load_base_config()
    config_dict = build_run_config(
        base,
        task=task,
        repo_path=repo,
        adapter_url=adapter_url,
        sid=sid,
        model_label=model_label,
        temperature=temperature,
        top_p=top_p,
        image=image,
        limits=limits,
    )
    config = RunSingleConfig.model_validate(config_dict)

    env = SWEEnv.from_config(config.env)
    agent = get_agent_from_config(config.agent)
    candidate_code = ""
    exit_status: str | None = None
    submission: str | None = None
    steps = 0
    try:
        env.start()
        result = agent.run(
            problem_statement=config.problem_statement,
            env=env,
            output_dir=work / "out",
        )
        exit_status = _run_exit_status(result)
        submission = _run_submission(result)
        steps = _run_steps(result)
        # Read the edited file BEFORE the env (container) is torn down.
        candidate_code = env.read_file(f"/{repo.name}/{CANDIDATE_FILENAME}") or ""
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover - close must never mask the run outcome
            pass

    if not candidate_code.strip():
        # The read failed or came back empty; grade the v0 program rather than
        # crash. (A patch-apply fallback from ``submission`` can be added if the
        # empty-read path ever shows up in practice on M1.)
        candidate_code = v0_code
    return ExtractResult(
        candidate_code=candidate_code,
        exit_status=exit_status,
        steps=steps,
        submission=submission,
    )
