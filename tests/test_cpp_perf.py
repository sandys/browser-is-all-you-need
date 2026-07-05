from __future__ import annotations

import json
import subprocess

import pytest

from w8_biayn.cpp_perf import sandbox
from w8_biayn.cpp_perf.coverage import coverage_passes, parse_lcov_summary
from w8_biayn.cpp_perf.judge import judge_output_matches, normalize_judge_output
from w8_biayn.cpp_perf.pie import build_tasks, read_pie_pairs
from w8_biayn.cpp_perf.reward import (
    MISSING_RUNTIME_REWARD,
    compute_reward,
    extract_code_block,
    extract_reward_code,
    file_state_reward,
    valid_model_output,
)
from w8_biayn.cpp_perf.sandbox import (
    DEFAULT_DOCKER_IMAGE,
    DEFAULT_SWE_AGENT_IMAGE,
    _runtime_benchmark_python,
    dry_run_plan,
    parse_runtime_benchmark_output,
    run_in_directory_prewritten,
    run_test_command,
    runtime_benchmark_command,
    swe_agent_image_build_plan,
    swe_agent_image_dockerfile,
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


def valid_output(code: str = "int main(){return 0;}") -> str:
    return f"<reasoning>use a closed form</reasoning>\n```cpp\n{code}\n```\n"


def test_task_requires_hidden_tests_and_coverage():
    with pytest.raises(ValueError, match="hidden test"):
        CppTask(
            task_id="bad",
            problem_id="p",
            prompt_code="x",
            unit_tests=[TestCase(input="", expected="")],
            hidden_tests=[],
            oracle_solution="x",
            test_coverage=TestCoverage(line=0.96, branch=0.86),
            reference=ReferencePerformance(value=1),
            split="train",
        )
    with pytest.raises(ValueError, match="coverage"):
        CppTask(
            task_id="bad",
            problem_id="p",
            prompt_code="x",
            unit_tests=[TestCase(input="", expected="")],
            hidden_tests=[TestCase(input="", expected="")],
            oracle_solution="x",
            test_coverage=TestCoverage(line=0.50, branch=0.86),
            reference=ReferencePerformance(value=1),
            split="train",
        )


def test_pie_parser_builds_valid_tasks_only_when_manifest_has_tests(tmp_path):
    tsv = tmp_path / "pie.tsv"
    tsv.write_text(
        "problem_id\tv0\tv1\tcpu_time_v1\tstatus_v0\n"
        "p1\tslow\tfast\t123\tAccepted\n"
        "p2\tslow2\tfast2\t456\tAccepted\n",
        encoding="utf-8",
    )
    manifest = {
        "p1": {
            "unit_tests": [{"input": "1\n", "expected": "1\n"}],
            "hidden_tests": [{"input": "2\n", "expected": "2\n"}],
            "coverage": {"line": 0.95, "branch": 0.85},
        }
    }

    pairs = read_pie_pairs(tsv)
    tasks = build_tasks(pairs, manifest, split="train")

    assert len(tasks) == 1
    assert tasks[0].problem_id == "p1"
    assert tasks[0].prompt_code == "slow"
    assert tasks[0].oracle_solution == "fast"
    assert tasks[0].reference.value == 123


def test_pie_parser_scales_fractional_positive_reference_values(tmp_path):
    jsonl = tmp_path / "pie.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "input": "slow",
                "target": "fast",
                "cpu_time_v1": 0.000123,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pair = read_pie_pairs(jsonl)[0]

    assert pair.reference_value == 123
    assert pair.gem5_cycles == 123


def test_pie_parser_can_skip_invalid_rows(tmp_path):
    jsonl = tmp_path / "pie.jsonl"
    jsonl.write_text(
        json.dumps({"problem_id": "bad", "input": "slow", "target": "fast", "cpu_time_v1": 0})
        + "\n"
        + json.dumps({"problem_id": "good", "input": "slow", "target": "fast", "cpu_time_v1": 2})
        + "\n",
        encoding="utf-8",
    )

    pairs = read_pie_pairs(jsonl, skip_invalid=True)

    assert [pair.problem_id for pair in pairs] == ["good"]


def test_coverage_summary_parses_and_gates():
    coverage = parse_lcov_summary("lines......: 96.2% (100 of 104)\nbranches...: 85.0% (17 of 20)\n")
    assert coverage.line == pytest.approx(0.962)
    assert coverage.branch == pytest.approx(0.85)
    assert coverage_passes(coverage)


def test_judge_output_comparison_allows_trailing_whitespace_only():
    assert normalize_judge_output("4 \n2\t\n3\n\n") == "4\n2\n3"
    assert judge_output_matches("4\n2\n3\n", "4 \n2\t\n3\n\n")
    assert not judge_output_matches("4\n2\n3\n", "4\n2\n5\n")


def test_judge_output_comparison_ignores_leading_blank_lines():
    # A reference oracle that prints a leading blank line (e.g. printf("\n")) must
    # still match the expected output a contest judge accepts.
    assert normalize_judge_output("\n\n4\n2\n3\n") == "4\n2\n3"
    assert judge_output_matches("\n4\n2\n3\n", "4\n2\n3\n")


def test_model_output_format_and_extraction():
    assert valid_model_output(valid_output())
    assert extract_code_block(valid_output()).startswith("int main")
    assert valid_model_output("<reasoning>x</reasoning>\n```cpp int main(){return 0;}```")
    assert extract_code_block("<reasoning>x</reasoning>\n```cpp int main(){return 0;}```") == "int main(){return 0;}\n"
    assert not valid_model_output("```cpp\nint main(){}\n```")
    assert not valid_model_output("<reasoning>x</reasoning>\n```python\nprint(1)\n```")
    assert not valid_model_output("<reasoning>x</reasoning>\n```cpp\nx\n```\n```cpp\ny\n```")
    assert not valid_model_output(
        "<reasoning>x</reasoning>\n<reasoning>y</reasoning>\n```cpp\nint main(){}\n```\n"
    )
    assert not valid_model_output("```cpp\nint main(){}\n```\n<reasoning>x</reasoning>\n")


def test_recoverable_code_extraction_accepts_bare_cpp_but_not_prose():
    code, strict = extract_reward_code("#include <bits/stdc++.h>\nint main(){return 0;}\n")

    assert strict is False
    assert code.startswith("#include")
    with pytest.raises(ValueError):
        extract_reward_code("not a formatted answer")


def test_reward_orders_compile_partial_correct_and_faster():
    task = sample_task()

    def compile_error(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(compile_error=True, tests_total=2)

    def partial(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(tests_passed=1, tests_total=2)

    def timeout(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(timeout=True, tests_passed=2, tests_total=2)

    def slower_correct(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(
            tests_passed=2,
            tests_total=2,
            runtime_cpu_ns=1200,
            reference_runtime_cpu_ns=1000,
        )

    def faster_correct(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(
            tests_passed=2,
            tests_total=2,
            runtime_cpu_ns=500,
            reference_runtime_cpu_ns=1000,
        )

    assert compute_reward(task, "bad", runner=faster_correct).reward == -1.0
    bare_correct = compute_reward(task, "#include <bits/stdc++.h>\nint main(){return 0;}\n", runner=faster_correct)
    assert bare_correct.reason == "recoverable_format_correct"
    assert bare_correct.format_valid is False
    assert 0 < bare_correct.reward < MISSING_RUNTIME_REWARD
    assert compute_reward(task, valid_output(), runner=compile_error).reward == -0.5
    bare_compile_error = compute_reward(
        task,
        "#include <bits/stdc++.h>\nint main(){return 0;}\n",
        runner=compile_error,
    )
    assert bare_compile_error.reason == "recoverable_format_compile_error"
    assert -1.0 < bare_compile_error.reward < -0.5
    assert compute_reward(task, valid_output(), runner=timeout).reason == "timeout"
    assert compute_reward(task, valid_output(), runner=timeout).reward == -0.5
    assert compute_reward(task, valid_output(), runner=partial).reward == pytest.approx(-0.1)
    assert compute_reward(task, valid_output(), runner=slower_correct).reward == pytest.approx(1.0)
    assert compute_reward(task, valid_output(), runner=faster_correct).reward > 1.0

    def missing_runtime(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(tests_passed=2, tests_total=2, runtime_cpu_ns=None, reference_runtime_cpu_ns=1000)

    missing = compute_reward(task, valid_output(), runner=missing_runtime)
    assert missing.reward == 0.5
    assert missing.reason == "missing_runtime"


def test_file_state_reward_orders_by_correctness_and_speed():
    # The agentic lane grades the final FILE, so the reward is the strict
    # correctness ladder over a HarnessResult with no response-format axis.
    def reward(harness: HarnessResult):
        return file_state_reward(harness, code="int main(){}\n")

    compile_err = reward(HarnessResult(compile_error=True, tests_total=2))
    sanitizer = reward(HarnessResult(sanitizer_error=True, tests_total=2))
    timed_out = reward(HarnessResult(timeout=True, tests_passed=2, tests_total=2))
    partial = reward(HarnessResult(tests_passed=1, tests_total=2))
    missing = reward(HarnessResult(tests_passed=2, tests_total=2, runtime_cpu_ns=None, reference_runtime_cpu_ns=1000))
    slower = reward(HarnessResult(tests_passed=2, tests_total=2, runtime_cpu_ns=1200, reference_runtime_cpu_ns=1000))
    faster = reward(HarnessResult(tests_passed=2, tests_total=2, runtime_cpu_ns=500, reference_runtime_cpu_ns=1000))

    assert (compile_err.reason, compile_err.reward) == ("compile_error", -0.5)
    assert (sanitizer.reason, sanitizer.reward) == ("sanitizer_error", -0.5)
    assert (timed_out.reason, timed_out.reward) == ("timeout", -0.5)
    assert partial.reason == "tests_failed" and partial.reward == pytest.approx(-0.1)
    assert missing.reason == "missing_runtime" and missing.reward == MISSING_RUNTIME_REWARD
    assert slower.reason == "correct" and slower.reward == pytest.approx(1.0)
    assert faster.reason == "correct" and faster.reward > 1.0
    # strict monotonic ordering across the tiers
    assert compile_err.reward < partial.reward < missing.reward < slower.reward < faster.reward


def test_run_in_directory_prewritten_grades_agent_file_without_overwriting(tmp_path, monkeypatch):
    # File-state grading: the agent's candidate.cpp is graded as-is; only
    # reference.cpp + tests/ are materialized from the task. Docker is stubbed
    # so this stays pure-Python and asserts the exact command sequence.
    scratch = tmp_path / "work"
    scratch.mkdir()
    sentinel = "// agent-authored\n#include <cstdio>\nint main(){return 0;}\n"
    (scratch / "candidate.cpp").write_text(sentinel, encoding="utf-8")

    calls: list[str] = []

    def fake_run(args: list[str]) -> subprocess.CompletedProcess[str]:
        joined = " ".join(args)
        calls.append(joined)
        if "w8_runtime_bench.py" in joined:
            cpu = 500 if "./candidate" in joined else 1000
            stdout = json.dumps({"ok": True, "runtime_cpu_ns": cpu, "runtime_wall_ns": cpu * 2})
        else:
            stdout = ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(sandbox, "_run", fake_run)

    task = sample_task()
    result = run_in_directory_prewritten(task, scratch, image="gcc:13", cpu="3")

    # the agent's file is graded verbatim, never rewritten
    assert (scratch / "candidate.cpp").read_text(encoding="utf-8") == sentinel
    # reference + tests come from the task
    assert (scratch / "reference.cpp").read_text(encoding="utf-8") == task.oracle_solution
    assert (scratch / "tests" / "0.in").read_text(encoding="utf-8") == task.unit_tests[0].input
    assert (scratch / "tests" / "1.in").read_text(encoding="utf-8") == task.hidden_tests[0].input

    # compile -> sanitize -> test0 -> test1 -> reference-compile -> bench candidate -> bench reference
    assert len(calls) == 7
    assert "candidate.cpp -o candidate" in calls[0]
    assert "candidate_san" in calls[1]
    assert "tests/0.in" in calls[2]
    assert "tests/1.in" in calls[3]
    assert "reference.cpp -o reference" in calls[4]
    assert "w8_runtime_bench.py" in calls[5] and "./candidate" in calls[5]
    assert "w8_runtime_bench.py" in calls[6] and "./reference" in calls[6]

    assert result.all_tests_pass
    assert result.runtime_speedup == pytest.approx(2.0)


def test_swe_agent_image_dockerfile_extends_cpp_image_with_git():
    # The SWE-agent edit-loop image is the C++ sandbox image plus git (for the
    # per-sample repo checkout); swerex installs its own runtime at deploy time.
    dockerfile = swe_agent_image_dockerfile()
    assert dockerfile.startswith(f"FROM {DEFAULT_DOCKER_IMAGE}")
    assert "git python3-pip" in dockerfile
    # swerex server must be pre-installed: the base image has no pip/network at
    # container-start, so swerex cannot self-install its runtime.
    assert "swe-rex==" in dockerfile

    plan = swe_agent_image_build_plan()
    assert " ".join(["docker", "build", "-t", DEFAULT_SWE_AGENT_IMAGE, "-"]) in plan
    assert f"FROM {DEFAULT_DOCKER_IMAGE}" in plan
    assert plan.rstrip().endswith("DOCKERFILE")


def test_sandbox_dry_run_and_runtime_parser():
    plan = dry_run_plan(sample_task(), image="gcc:13", cpu="3")
    assert "--network none" in plan
    assert "g++ -O3 -std=c++20 candidate.cpp -o candidate" in plan
    assert "python3 /tmp/w8_runtime_bench.py" in plan
    assert "--binary ./candidate" in plan
    assert "--binary ./reference" in plan
    assert "expected.norm" in " ".join(run_test_command(0, "/tmp/w8", image="gcc:13"))

    parsed = parse_runtime_benchmark_output('noise\n{"ok":true,"runtime_cpu_ns":12,"runtime_wall_ns":20}\n')
    assert parsed == {"ok": True, "runtime_cpu_ns": 12, "runtime_wall_ns": 20}
    assert parse_runtime_benchmark_output("garbage\n") is None
    assert parse_runtime_benchmark_output("") is None


def test_runtime_benchmark_skips_output_validation_for_reference_only():
    # The trusted reference oracle is timed without re-checking its stdout; the
    # candidate is still validated. This is what keeps a benignly-formatted oracle
    # (e.g. a leading blank line) from zeroing out the reference runtime.
    candidate = " ".join(runtime_benchmark_command("/tmp/w8", binary="candidate", test_count=2))
    reference = " ".join(runtime_benchmark_command("/tmp/w8", binary="reference", test_count=2, validate_output=False))
    assert "--validate-output 1" in candidate
    assert "--validate-output 0" in reference
    # the dry-run plan renders the reference benchmark with validation disabled too
    assert "--validate-output 0" in dry_run_plan(sample_task(), image="gcc:13", cpu="3")


def test_runtime_harness_normalize_ignores_leading_blank_lines():
    # Execute the inlined benchmark source and check its normalize + run_once leniency,
    # since that code runs verbatim inside Docker.
    ns: dict = {}
    exec(_runtime_benchmark_python(), ns)  # noqa: S102 - trusted in-repo harness source
    assert ns["normalize"]("\n\n4\n2\n3\n") == "4\n2\n3"
    # reference path (validate_output=False) accepts a leading-blank-line oracle and times it
    ok = ns["run_once"]("/bin/echo", "", "x", 5, 0, False)
    assert ok["ok"] is True and ok["cpu_ns"] >= 1
    # candidate path still rejects mismatched output
    bad = ns["run_once"]("/bin/echo", "", "definitely-not-this", 5, 0, True)
    assert bad["ok"] is False and bad["reason"] == "wrong_output"


def test_task_json_round_trip(tmp_path):
    task = sample_task()
    path = task.write_json(tmp_path / "task.json")
    assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == task.task_id
    assert CppTask.read_json(path) == task
