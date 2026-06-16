from __future__ import annotations

from w8_biayn.cpp_perf.schema import CppTask, HarnessResult, ReferencePerformance, TestCase, TestCoverage
from w8_biayn.integrations.cpp_perf_env import CppPerfEnv


def sample_task() -> CppTask:
    return CppTask(
        task_id="pie_cpp_000001",
        problem_id="p1",
        prompt_code="int main(){return 0;}\n",
        unit_tests=[TestCase(input="1\n", expected="1\n")],
        hidden_tests=[TestCase(input="2\n", expected="2\n")],
        oracle_solution="int main(){return 0;}\n",
        test_coverage=TestCoverage(line=0.96, branch=0.86),
        reference=ReferencePerformance(value=1000),
        split="train",
    )


def test_cpp_perf_env_scores_one_turn_with_existing_reward(tmp_path):
    task_path = sample_task().write_json(tmp_path / "task.json")

    def runner(_task: CppTask, _code: str) -> HarnessResult:
        return HarnessResult(tests_passed=2, tests_total=2, runtime_cpu_ns=500, reference_runtime_cpu_ns=1000)

    env = CppPerfEnv(
        env_config={"runner": runner},
        extras={"extra_info": {"task_path": str(task_path)}},
    )
    result = env.step("<reasoning>x</reasoning>\n```cpp\nint main(){return 0;}\n```\n")

    assert result["done"] is True
    assert result["reward"] > 1.0
    assert result["metadata"]["task_id"] == "pie_cpp_000001"
    assert result["metadata"]["tests_passed"] == 2
    assert env.get_metrics()["runtime_cpu_ns"] == 500
    assert env.get_metrics()["reference_runtime_cpu_ns"] == 1000


def test_cpp_perf_env_invalid_format_is_negative(tmp_path):
    task_path = sample_task().write_json(tmp_path / "task.json")
    env = CppPerfEnv(extras={"extra_info": {"task_path": str(task_path)}})

    result = env.step("not a formatted answer")

    assert result["reward"] == -1.0
    assert result["metadata"]["reason"] == "invalid_format"


def test_cpp_perf_env_turns_reward_exception_into_negative_result(tmp_path):
    task_path = sample_task().write_json(tmp_path / "task.json")

    def runner(_task: CppTask, _code: str) -> HarnessResult:
        raise RuntimeError("boom")

    env = CppPerfEnv(
        env_config={"runner": runner},
        extras={"extra_info": {"task_path": str(task_path)}},
    )
    result = env.step("<reasoning>x</reasoning>\n```cpp\nint main(){return 0;}\n```\n")

    assert result["done"] is True
    assert result["reward"] == -1.0
    assert result["metadata"]["reason"] == "reward_exception"
    assert "boom" in result["metadata"]["exception"]
