from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import w8_biayn.integrations.slime_swe_agent_cpp_perf as gen
from w8_biayn.integrations import slime_cpp_perf, swe_agent_driver
from w8_biayn.integrations.swe_agent_driver import (
    ExtractResult,
    SweAgentLimits,
    _materialize_repo,
    _safe_repo_name,
    build_run_config,
)
from w8_biayn.cpp_perf.schema import CppTask, HarnessResult, ReferencePerformance, TestCase, TestCoverage


def sample_task() -> CppTask:
    return CppTask(
        task_id="pie_cpp_000001",
        problem_id="p00001",
        prompt_code="int main(){return 0;}\n",
        unit_tests=[TestCase(input="1\n", expected="1\n")],
        hidden_tests=[TestCase(input="2\n", expected="2\n")],
        oracle_solution="int main(){return 0;}\n",
        test_coverage=TestCoverage(line=0.96, branch=0.86),
        reference=ReferencePerformance(value=1000, gem5_cycles=2000),
        split="train",
    )


class _FakeAdapter:
    def __init__(self) -> None:
        self.opened: dict | None = None
        self.finished: dict | None = None
        self.dropped: str | None = None

    def open_session(self, sid, *, sampling_defaults=None, max_context_tokens=0):
        self.opened = {"sid": sid, "sampling_defaults": sampling_defaults, "max_context_tokens": max_context_tokens}

    async def finish_session(self, sid, *, base_sample, reward=0.0, **_kw):
        self.finished = {"sid": sid, "reward": reward}
        # finish_session sets response/reward on the drained samples; emulate one.
        return [SimpleNamespace(metadata={}, reward=reward, index=0, group_index=0, rollout_id="r0", response="")]

    async def drop_session(self, sid, **_kw):
        self.dropped = sid


class _FakeHealth:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, row: dict) -> None:
        self.rows.append(row)


def _fake_state(adapter: _FakeAdapter, health: _FakeHealth | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        adapter=adapter,
        adapter_url="http://127.0.0.1:9/v1",
        max_context_len=4096,
        swe_image="w8-biayn-swe-agent:latest",
        model_label="slime-actor",
        limits=SweAgentLimits(),
        grader_image="gcc:13",
        grader_cpu="3",
        rollout_guard_sec=60,
        health=health or _FakeHealth(),
    )


def test_generate_runs_agent_grades_file_and_flows_reward(monkeypatch, tmp_path):
    task = sample_task()
    task_json = task.write_json(tmp_path / "task.json")

    adapter = _FakeAdapter()
    health = _FakeHealth()
    monkeypatch.setattr(gen, "_adapter_service", lambda args: _fake_state(adapter, health))

    driver_calls: dict = {}

    def fake_run(task_arg, v0_code, **kwargs):
        driver_calls.update(kwargs)
        driver_calls["v0_code"] = v0_code
        return ExtractResult(candidate_code="int main(){return 0;}\n", exit_status="submitted", steps=4, submission=None)

    monkeypatch.setattr(swe_agent_driver, "run_swe_agent_and_extract", fake_run)
    monkeypatch.setattr(
        gen,
        "_grade_file_state",
        lambda task_arg, code, **_kw: HarnessResult(
            tests_passed=2, tests_total=2, runtime_cpu_ns=500, reference_runtime_cpu_ns=1000, runtime_speedup=2.0
        ),
    )

    base = SimpleNamespace(metadata={"task_path": str(task_json)}, session_id=None, index=0, group_index=0)
    samples = asyncio.run(gen.generate(SimpleNamespace(), base, {"temperature": 0.8, "top_p": 0.95}))

    # one fan-out sample, faster-correct reward > 1.0, flowed through finish_session
    assert len(samples) == 1
    assert samples[0].reward > 1.0
    assert adapter.finished is not None and adapter.finished["reward"] > 1.0

    # session lifecycle: opened with context budget, dropped in finally
    assert adapter.opened is not None
    sid = adapter.opened["sid"]
    assert sid.startswith("w8swe-pie_cpp_000001-0-0")
    assert adapter.opened["max_context_tokens"] == 4096
    assert adapter.dropped == sid

    # rich harness record stashed for the eval pipeline
    record = samples[0].metadata["cpp_reward_record"]
    assert record["all_tests_pass"] is True
    assert record["runtime_speedup"] == pytest.approx(2.0)
    assert samples[0].metadata["swe_exit_status"] == "submitted"
    assert samples[0].metadata["agent_steps"] == 4
    # SLIME --log-multi-turn contract: round_number drives multi_turn_metric/*.
    assert samples[0].metadata["round_number"] == 4

    # live rollout-health row fed into the shared-mode W&B aggregator
    assert len(health.rows) == 1
    row = health.rows[0]
    assert row["task_id"] == "pie_cpp_000001"
    assert row["all_tests_pass"] is True and row["reward"] > 1.0
    assert row["agent_steps"] == 4 and row["wall_time_s"] >= 0.0

    # rollout sampling params (not SWE-agent's greedy defaults) reach the driver
    assert driver_calls["temperature"] == 0.8
    assert driver_calls["top_p"] == 0.95
    assert driver_calls["sid"] == sid
    assert driver_calls["adapter_url"] == "http://127.0.0.1:9/v1"
    assert driver_calls["v0_code"] == task.prompt_code


def test_generate_aborts_on_missing_task_path(monkeypatch):
    adapter = _FakeAdapter()
    health = _FakeHealth()
    monkeypatch.setattr(gen, "_adapter_service", lambda args: _fake_state(adapter, health))

    base = SimpleNamespace(metadata={}, session_id=None, index=0, group_index=0)
    samples = asyncio.run(gen.generate(SimpleNamespace(), base, {}))

    assert len(samples) == 1
    assert samples[0].remove_sample is True
    # aborts are health-visible too: reason + zero reward reach the aggregator
    assert health.rows and health.rows[0]["abort_reason"] == "missing_task_path"
    assert health.rows[0]["reward"] == 0.0
    assert samples[0].reward == 0.0
    assert samples[0].metadata["abort_reason"] == "missing_task_path"
    # aborted before any session was opened
    assert adapter.opened is None
    assert adapter.dropped is None


def test_grade_file_state_writes_candidate_and_delegates(monkeypatch):
    captured: dict = {}

    def fake_prewritten(task, scratch, *, image, cpu):
        captured["candidate"] = (Path(scratch) / "candidate.cpp").read_text(encoding="utf-8")
        captured["image"] = image
        captured["cpu"] = cpu
        return HarnessResult(tests_passed=1, tests_total=1)

    monkeypatch.setattr(gen, "run_in_directory_prewritten", fake_prewritten)

    result = gen._grade_file_state(sample_task(), "int main(){return 7;}\n", image="img", cpu="5")

    assert captured["candidate"] == "int main(){return 7;}\n"
    assert captured["image"] == "img"
    assert captured["cpu"] == "5"
    assert result.tests_passed == 1


def test_build_run_config_sets_endpoint_cost_limits_and_repo():
    base = {"agent": {"model": {"stop": ["</s>"]}, "tools": {"bundles": ["edit"]}}, "extra": 1}
    limits = SweAgentLimits(per_instance_call_limit=7, total_execution_timeout=900, max_input_tokens=1234)

    merged = build_run_config(
        base,
        task=sample_task(),
        repo_path="/work/repo",
        adapter_url="http://127.0.0.1:9/v1",
        sid="sess-123",
        model_label="slime-actor",
        temperature=0.7,
        top_p=0.9,
        image="w8-biayn-swe-agent:latest",
        limits=limits,
    )

    model = merged["agent"]["model"]
    assert model["name"] == "openai/slime-actor"
    assert model["api_base"] == "http://127.0.0.1:9/v1"
    assert model["api_key"] == "sess-123"
    # cost limiting disabled (both), loop bounded by call limit instead
    assert model["per_instance_cost_limit"] == 0.0
    assert model["total_cost_limit"] == 0.0
    assert model["per_instance_call_limit"] == 7
    assert model["max_input_tokens"] == 1234
    # rollout sampling params forwarded; base model fields preserved
    assert model["temperature"] == 0.7
    assert model["top_p"] == 0.9
    assert model["stop"] == ["</s>"]

    tools = merged["agent"]["tools"]
    assert tools["parse_function"] == {"type": "thought_action"}
    assert tools["total_execution_timeout"] == 900
    assert tools["bundles"] == ["edit"]

    # LocalDeployment: SWE-agent runs on the node in-process, no sibling container.
    assert merged["env"]["deployment"] == {"type": "local"}
    assert "image" not in merged["env"]["deployment"]
    assert merged["env"]["repo"] == {"type": "local", "path": "/work/repo", "base_commit": "HEAD"}
    assert merged["problem_statement"]["id"] == "pie_cpp_000001"
    assert "candidate.cpp" in merged["problem_statement"]["text"]
    assert merged["extra"] == 1  # untouched base keys survive


@pytest.mark.skipif(shutil.which("git") is None, reason="git required to seed the SWE-agent repo")
def test_materialize_repo_creates_clean_committed_repo(tmp_path):
    repo = _materialize_repo(tmp_path / "repo", "int main(){return 0;}\n", task=sample_task())

    assert (repo / "candidate.cpp").read_text(encoding="utf-8") == "int main(){return 0;}\n"
    # visible test seeded, compile helper present, hidden tests NEVER written
    assert (repo / "tests" / "0.in").read_text(encoding="utf-8") == "1\n"
    assert (repo / "compile.sh").exists()
    assert not (repo / "tests" / "1.in").exists()
    # a clean single commit at HEAD (SWE-agent's check_valid_repo rejects a dirty tree)
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True
    )
    assert status.stdout.strip() == ""
    count = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"], check=True, capture_output=True, text=True
    )
    assert count.stdout.strip() == "1"


def test_safe_repo_name_is_unique_per_session_and_fs_safe():
    # Under LocalDeployment the repo lands at FS root /{repo_name}; concurrent
    # rollouts must get distinct, filesystem-safe basenames keyed off the sid.
    a = _safe_repo_name("w8swe-pie_cpp_000001-0-0")
    b = _safe_repo_name("w8swe-pie_cpp_000001-0-1")
    assert a != b
    assert a.startswith("repo-")
    assert all(c.isalnum() or c in "_-" for c in a)
    assert _safe_repo_name("") == "repo-session"


def test_sweagent_limits_from_env_parses_and_defaults():
    limits = SweAgentLimits.from_env(
        {
            "W8_SWE_AGENT_PER_INSTANCE_CALL_LIMIT": "25",
            "W8_SWE_AGENT_MAX_INPUT_TOKENS": "8000",
            "W8_SWE_AGENT_PARSE_FUNCTION": "thought_action",
        }
    )
    assert limits.per_instance_call_limit == 25
    assert limits.max_input_tokens == 8000
    assert limits.max_output_tokens is None  # unset -> None
    assert limits.total_execution_timeout == 1200  # default


def test_record_from_debug_sample_prefers_stashed_agentic_record():
    # Agentic lane: reward is a float, the harness breakdown lives in metadata.
    stashed = {
        "score": 1.8,
        "reward": 1.8,
        "reason": "correct",
        "task_id": "pie_cpp_000001",
        "all_tests_pass": True,
        "runtime_speedup": 2.0,
    }
    sample = {"reward": 0.9, "metadata": {"cpp_reward_record": stashed}, "index": 0, "rollout_id": "r0"}

    record = slime_cpp_perf.record_from_debug_sample(sample, label="grpo")

    assert record["reason"] == "correct"
    assert record["all_tests_pass"] is True
    assert record["runtime_speedup"] == 2.0
    assert record["label"] == "grpo"
    # falls back to the float reward path when nothing is stashed
    plain = slime_cpp_perf.record_from_debug_sample(
        {"reward": -0.5, "metadata": {"reason": "compile_error", "task_id": "t"}}
    )
    assert plain["reward"] == -0.5
    assert plain["reason"] == "compile_error"
