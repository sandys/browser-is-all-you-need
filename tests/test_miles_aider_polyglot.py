from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import glm47_posttraining.aider_polyglot.harness as harness_module
import glm47_posttraining.integrations.miles_aider_polyglot as integration_module
from glm47_posttraining.aider_polyglot.dataset import (
    DATASET_KIND,
    EXPECTED_SHADOW_TASKS,
    build_aider_polyglot_datasets,
)
from glm47_posttraining.aider_polyglot.harness import run_aider_tests, run_shadow_tests
from glm47_posttraining.aider_polyglot.parser import AiderResponseError, parse_whole_file_response
from glm47_posttraining.aider_polyglot.reward import compute_aider_reward
from glm47_posttraining.aider_polyglot.schema import AiderPolyglotTask, AiderTestResult


def _task() -> AiderPolyglotTask:
    return AiderPolyglotTask(
        task_id="aider-shadow-cpp/example",
        exercise="example",
        split="train",
        harness_kind="shadow_cpp17",
        exercise_dir="shadow/example",
        editable_files=["example.cpp", "example.h"],
        prompt=[{"role": "user", "content": "solve"}],
        source_revision="abc123",
        hidden_test_sha256="a" * 64,
        source_prompt_sha256="b" * 64,
        verification_gate="unit",
    )


def _response(label: str = "example.cpp", *, prefix: str = "") -> str:
    return f"{prefix}{label}\n```cpp\nint answer() {{ return 42; }}\n```\n"


def _make_shadow_tree(tmp_path: Path) -> Path:
    root = tmp_path / "rubrics"
    practice = root / "cpp" / "exercises" / "practice"
    for index in range(EXPECTED_SHADOW_TASKS):
        slug = f"exercise-{index:03d}"
        exercise = practice / slug
        (exercise / ".docs").mkdir(parents=True)
        (exercise / ".docs" / "instructions.md").write_text(
            f"# Introduction\n\n# {slug}\n\nImplement answer {index}.\n", encoding="utf-8"
        )
        header = f"{slug}.h"
        source = f"{slug}.cpp"
        test = f"{slug}_test.cpp"
        (exercise / header).write_text("#pragma once\nint answer();\n", encoding="utf-8")
        (exercise / source).write_text(
            f'#include "{header}"\nint answer() {{ return 0; }}\n', encoding="utf-8"
        )
        test_bytes = f'#include "{header}"\nint main() {{ return answer() == {index} ? 0 : 1; }}\n'
        (exercise / test).write_text(test_bytes, encoding="utf-8")
        (exercise / "CMakeLists.txt").write_text("project(example CXX)\n", encoding="utf-8")
        rubric = {
            "category": f"category-{index % 6}",
            "editable_files": [header, source],
            "family": f"family-{index % 32}",
            "hidden_test_file": test,
            "hidden_test_sha256": hashlib.sha256(test_bytes.encode()).hexdigest(),
            "language": "cpp",
            "reference_answer_packaged": False,
            "schema_version": 1,
            "source_prompt_sha256": hashlib.sha256(slug.encode()).hexdigest(),
            "tags": ["cpp", "aider-whole-edit"],
            "task_id": slug,
            "verification_gate": "unit",
            "verification_stage": "passed",
        }
        (exercise / ".rubric.json").write_text(json.dumps(rubric), encoding="utf-8")
    manifest = {
        "kind": "aider-polyglot-cpp-shadow-rubrics",
        "counts": {"tasks": EXPECTED_SHADOW_TASKS},
        "contract": {
            "official_task_id_overlap": [],
            "reference_answers_packaged": False,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_whole_file_parser_accepts_sft_and_public_environment_prefixes() -> None:
    direct = parse_whole_file_response(_response(), ["example.cpp", "example.h"])
    public = parse_whole_file_response(
        _response(prefix="///\n"), ["example.cpp", "example.h"]
    )

    assert direct.files == public.files == {"example.cpp": "int answer() { return 42; }\n"}
    assert direct.format_valid is public.format_valid is True


def test_whole_file_parser_marks_markdown_filename_as_recoverable() -> None:
    parsed = parse_whole_file_response(_response("### example.cpp"), ["example.cpp"])
    assert parsed.files["example.cpp"].startswith("int answer")
    assert parsed.format_valid is False


@pytest.mark.parametrize(
    "marker", ["<|endoftext|>", "<|user|>", "<|observation|>"]
)
@pytest.mark.parametrize("separator", ["", "\n", " \n"])
def test_whole_file_parser_removes_only_terminal_glm_stop_markers(
    marker: str, separator: str
) -> None:
    response = _response().rstrip("\n") + separator + marker
    parsed = parse_whole_file_response(response, ["example.cpp"])
    assert parsed.files == {"example.cpp": "int answer() { return 42; }\n"}
    assert parsed.format_valid is True


def test_whole_file_parser_preserves_stop_marker_inside_file() -> None:
    response = "example.cpp\n```cpp\nconst char *token = \"<|user|>\";\n```<|user|>"
    parsed = parse_whole_file_response(response, ["example.cpp"])
    assert parsed.files == {"example.cpp": 'const char *token = "<|user|>";\n'}
    assert parsed.format_valid is True


@pytest.mark.parametrize(
    "label", ["CMakeLists.txt", "example_test.cpp", "../example_test.cpp", "/tmp/CMakeLists.txt"]
)
def test_whole_file_parser_rejects_non_editable_targets(label: str) -> None:
    with pytest.raises(AiderResponseError) as exc:
        parse_whole_file_response(_response(label), ["example.cpp"])
    assert exc.value.reason == "forbidden_file"


@pytest.mark.parametrize("label", ["src/example.cpp", "../example.cpp", "/tmp/example.cpp"])
def test_whole_file_parser_maps_path_label_to_editable_basename(label: str) -> None:
    parsed = parse_whole_file_response(_response(label), ["example.cpp"])
    assert parsed.files == {"example.cpp": "int answer() { return 42; }\n"}
    assert parsed.format_valid is False


def test_whole_file_parser_skips_stray_fences_as_recoverable() -> None:
    response = (
        "Plan:\n```\npseudo code, not a file\n```\n\n"
        "Update example.cpp with this:\n```cpp\nint wrong() { return 0; }\n```\n\n"
        + _response()
    )
    parsed = parse_whole_file_response(response, ["example.cpp"])
    assert parsed.files == {"example.cpp": "int answer() { return 42; }\n"}
    assert parsed.format_valid is False


def test_whole_file_parser_requires_at_least_one_editable_file() -> None:
    with pytest.raises(AiderResponseError) as exc:
        parse_whole_file_response("Plan:\n```\njust prose\n```\n", ["example.cpp"])
    assert exc.value.reason == "invalid_format"


def test_whole_file_parser_rejects_duplicate_file() -> None:
    response = _response() + "\n" + _response()
    with pytest.raises(AiderResponseError) as exc:
        parse_whole_file_response(response, ["example.cpp"])
    assert exc.value.reason == "duplicate_file"


def test_aider_reward_prioritizes_tests_and_tiebreaks_format(tmp_path: Path) -> None:
    def passed(_path: Path, _files: dict[str, str]) -> AiderTestResult:
        return AiderTestResult(status="passed", tests_passed=1, tests_total=1)

    exact = compute_aider_reward(_task(), tmp_path, _response(), runner=passed)
    recoverable = compute_aider_reward(
        _task(), tmp_path, _response("### example.cpp"), runner=passed
    )
    assert (exact.reward, exact.reason) == (1.0, "passed")
    assert (recoverable.reward, recoverable.reason) == (0.9, "recoverable_format_passed")


def test_aider_reward_rejects_test_tampering_without_execution(tmp_path: Path) -> None:
    called = False

    def runner(_path: Path, _files: dict[str, str]) -> AiderTestResult:
        nonlocal called
        called = True
        return AiderTestResult(status="passed", tests_passed=1, tests_total=1)

    breakdown = compute_aider_reward(
        _task(), tmp_path, _response("example_test.cpp"), runner=runner
    )
    assert (breakdown.reward, breakdown.reason, called) == (-1.0, "forbidden_file", False)


def test_harness_parses_build_triggered_catch_success(tmp_path: Path, monkeypatch) -> None:
    exercise = tmp_path / "example"
    exercise.mkdir()
    (exercise / "example.cpp").write_text("int answer(){return 0;}\n", encoding="utf-8")
    results = iter(
        [
            subprocess.CompletedProcess(["cmake"], 0, stdout="configured\n", stderr=""),
            subprocess.CompletedProcess(
                ["cmake", "--build"],
                0,
                stdout="All tests passed (2004 assertions in 5 test cases)\n",
                stderr="",
            ),
        ]
    )
    monkeypatch.setattr(harness_module, "_run_stage", lambda *args, **kwargs: next(results))
    result = run_aider_tests(exercise, {"example.cpp": "int answer(){return 42;}\n"})
    assert (result.status, result.tests_passed, result.tests_total) == ("passed", 5, 5)


def test_shadow_harness_compiles_hidden_test_before_candidate(tmp_path: Path, monkeypatch) -> None:
    exercise = tmp_path / "shadow-example"
    (exercise / ".grader").mkdir(parents=True)
    (exercise / "example.cpp").write_text("int answer(){return 0;}\n", encoding="utf-8")
    hidden = "int answer(); int main(){return answer() == 42 ? 0 : 1;}\n"
    (exercise / ".grader" / "test.cpp").write_text(hidden, encoding="utf-8")
    results = iter(
        [
            subprocess.CompletedProcess(["c++"], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                ["candidate_test"], 0, stdout="GLM47_AIDER_PASS_abc\n", stderr=""
            ),
        ]
    )
    commands: list[str] = []

    def stage(_scratch, script, **_kwargs):
        commands.append(script)
        return next(results)

    monkeypatch.setattr(harness_module, "_run_stage", stage)
    monkeypatch.setattr(harness_module.secrets, "token_hex", lambda _size: "abc")
    result = run_shadow_tests(
        exercise,
        {"example.cpp": "int answer(){return 42;}\n"},
        expected_test_sha256=hashlib.sha256(hidden.encode()).hexdigest(),
    )
    assert result.all_tests_pass
    assert "-std=c++17" in commands[0]
    assert ".grader/test.cpp" in commands[0]
    assert "rm .grader/test.cpp" in commands[0]
    assert commands[1].endswith(".grader/candidate_test")


def _run_ordinal_shadow(tmp_path: Path, monkeypatch, candidate_returncode: int):
    exercise = tmp_path / "shadow-ordinal"
    (exercise / ".grader").mkdir(parents=True)
    (exercise / "example.cpp").write_text("int answer(){return 0;}\n", encoding="utf-8")
    hidden = (
        "int answer();\n"
        "int main(){\n"
        "  if (answer() < 1) return 1;\n"
        "  if (answer() < 2) return 2;\n"
        "  if (answer() < 3) return 3;\n"
        "  if (answer() < 4) return 4;\n"
        "  if (answer() != 5) return 5;\n"
        "  return 0;\n"
        "}\n"
    )
    (exercise / ".grader" / "test.cpp").write_text(hidden, encoding="utf-8")
    marker = "GLM47_AIDER_PASS_abc\n" if candidate_returncode == 0 else ""
    results = iter(
        [
            subprocess.CompletedProcess(["c++"], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(
                ["candidate_test"], candidate_returncode, stdout=marker, stderr=""
            ),
        ]
    )
    monkeypatch.setattr(harness_module, "_run_stage", lambda *a, **k: next(results))
    monkeypatch.setattr(harness_module.secrets, "token_hex", lambda _size: "abc")
    return run_shadow_tests(
        exercise,
        {"example.cpp": "int answer(){return 5;}\n"},
        expected_test_sha256=hashlib.sha256(hidden.encode()).hexdigest(),
    )


def test_shadow_harness_awards_partial_credit_for_ordinal_grader(tmp_path, monkeypatch) -> None:
    # Exit code 3 means checks 1 and 2 passed before check 3 failed: 2 of 5.
    result = _run_ordinal_shadow(tmp_path, monkeypatch, candidate_returncode=3)
    assert result.status == "tests_failed"
    assert result.tests_passed == 2
    assert result.tests_total == 5
    assert result.fraction_tests_passed == pytest.approx(0.4)
    assert not result.all_tests_pass


def test_shadow_harness_full_pass_uses_ordinal_total(tmp_path, monkeypatch) -> None:
    result = _run_ordinal_shadow(tmp_path, monkeypatch, candidate_returncode=0)
    assert result.all_tests_pass
    assert result.tests_passed == 5 and result.tests_total == 5


def test_shadow_harness_crash_exit_scores_zero(tmp_path, monkeypatch) -> None:
    # A crash signal (139) is outside [1, N]; award no partial credit but keep N.
    result = _run_ordinal_shadow(tmp_path, monkeypatch, candidate_returncode=139)
    assert result.status == "tests_failed"
    assert result.tests_passed == 0
    assert result.tests_total == 5


def test_shadow_harness_non_ordinal_grader_stays_binary(tmp_path, monkeypatch) -> None:
    exercise = tmp_path / "shadow-binary"
    (exercise / ".grader").mkdir(parents=True)
    (exercise / "example.cpp").write_text("int answer(){return 0;}\n", encoding="utf-8")
    # Macro-style grader always returns 1 on failure: no sequential ordinals to read.
    hidden = (
        "int answer();\n"
        "#define CHECK(c) do { if (!(c)) return 1; } while (0)\n"
        "int main(){ CHECK(answer()==1); CHECK(answer()==2); return 0; }\n"
    )
    (exercise / ".grader" / "test.cpp").write_text(hidden, encoding="utf-8")
    results = iter(
        [
            subprocess.CompletedProcess(["c++"], 0, stdout="", stderr=""),
            subprocess.CompletedProcess(["candidate_test"], 1, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(harness_module, "_run_stage", lambda *a, **k: next(results))
    monkeypatch.setattr(harness_module.secrets, "token_hex", lambda _size: "abc")
    result = run_shadow_tests(
        exercise,
        {"example.cpp": "int answer(){return 1;}\n"},
        expected_test_sha256=hashlib.sha256(hidden.encode()).hexdigest(),
    )
    assert result.status == "tests_failed"
    assert result.tests_passed == 0
    assert result.tests_total == 1


def test_shadow_harness_rejects_early_exit_bypass(tmp_path: Path) -> None:
    exercise = tmp_path / "shadow-example"
    (exercise / ".grader").mkdir(parents=True)
    (exercise / "example.cpp").write_text("int answer(){return 0;}\n", encoding="utf-8")
    hidden = "int answer(); int main(){return answer() == 42 ? 0 : 1;}\n"
    (exercise / ".grader" / "test.cpp").write_text(hidden, encoding="utf-8")
    with pytest.raises(harness_module.CandidatePolicyError):
        run_shadow_tests(
            exercise,
            {"example.cpp": "struct Escape { Escape(){ _Exit(0); } } escape;\n"},
            expected_test_sha256=hashlib.sha256(hidden.encode()).hexdigest(),
        )


def test_linux_local_stage_is_fail_closed_and_mount_isolated(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GLM47_CPP_SANDBOX_BACKEND", "local")
    monkeypatch.delenv("GLM47_CPP_SANDBOX_UNSHARE_NET", raising=False)
    monkeypatch.setattr(harness_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(harness_module.shutil, "which", lambda name: "/usr/bin/bwrap")
    command = harness_module._local_sandbox_command(tmp_path, "true")
    rendered = " ".join(command)
    assert "--unshare-all" in command
    assert "--clearenv" in command
    assert str(tmp_path.resolve()) in command
    assert "/workspace" not in rendered


def test_linux_local_stage_can_skip_only_the_net_unshare(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GLM47_CPP_SANDBOX_BACKEND", "local")
    monkeypatch.setenv("GLM47_CPP_SANDBOX_UNSHARE_NET", "0")
    monkeypatch.setattr(harness_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(harness_module.shutil, "which", lambda name: "/usr/bin/bwrap")
    command = harness_module._local_sandbox_command(tmp_path, "true")
    assert "--unshare-all" not in command
    assert "--unshare-net" not in command
    for flag in (
        "--unshare-user-try",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-cgroup-try",
    ):
        assert flag in command
    assert "--clearenv" in command


def test_dataset_builder_materializes_only_answer_blind_training_files(tmp_path: Path) -> None:
    source = _make_shadow_tree(tmp_path)
    paths = build_aider_polyglot_datasets(
        source, tmp_path / "prepared", profile="unit", train_limit=3, monitor_limit=2
    )
    train_rows = [json.loads(line) for line in paths["grpo_train"].read_text().splitlines()]
    monitor_rows = [json.loads(line) for line in paths["eval"].read_text().splitlines()]
    manifest = json.loads(paths["manifest"].read_text())

    assert len(train_rows) == 3
    assert len(monitor_rows) == 2
    assert manifest["kind"] == DATASET_KIND
    assert manifest["counts"] == {"available_shadow": 253, "monitor": 2, "train": 3}
    assert manifest["split_contract"]["official_26"] == "external fixed evaluation only"
    first = AiderPolyglotTask.read_json(
        paths["manifest"].parent / train_rows[0]["metadata"]["task_path"]
    )
    materialized = paths["manifest"].parent / first.exercise_dir
    assert (materialized / ".grader" / "test.cpp").is_file()
    assert not (materialized / "CMakeLists.txt").exists()
    assert not any(materialized.glob("*_test.cpp"))
    assert [message.role for message in first.prompt] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert "*file listing* format" in first.prompt[0].content
    assert "*added these files to the chat*" in first.prompt[5].content
    final = first.prompt[-1].content
    assert "Use the above instructions to modify the supplied files:" in final
    assert final.rstrip().endswith("including any appropriate path.")
    assert all("_test.cpp" not in message.content for message in first.prompt)


def test_dataset_builder_validates_source_before_replacing_output(tmp_path: Path) -> None:
    source = _make_shadow_tree(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    first = next((source / "cpp" / "exercises" / "practice").iterdir())
    (first / next(first.glob("*_test.cpp")).name).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_aider_polyglot_datasets(source, output, force=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_miles_reward_hook_uses_shadow_task_and_returns_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    data = tmp_path / "data"
    exercise = data / "shadow" / "example"
    (exercise / ".grader").mkdir(parents=True)
    task_path = _task().write_json(data / "tasks" / "train" / "example.json")
    monkeypatch.setenv("GLM47_DATA_DIR", str(data))
    monkeypatch.setattr(
        integration_module,
        "run_shadow_tests",
        lambda *args, **kwargs: AiderTestResult(
            status="passed", tests_passed=1, tests_total=1, candidate_returncode=0
        ),
    )
    sample = SimpleNamespace(
        index=2,
        rollout_id=3,
        response=_response(),
        metadata={"task_path": str(task_path.relative_to(data))},
    )
    record = asyncio.run(integration_module.reward_func(SimpleNamespace(), sample))
    assert record["score"] == 1.0
    assert record["all_tests_pass"] is True
    assert record["candidate_returncode"] == 0
    assert record["modified_files"] == ["example.cpp"]


def test_miles_reward_hook_rejects_missing_task_binding() -> None:
    sample = SimpleNamespace(response=_response(), metadata={})
    record = asyncio.run(integration_module.reward_func(SimpleNamespace(), sample))
    assert record["score"] == 0.0
    assert record["reason"] == "missing_task_path"
    assert record["infrastructure_error"] is True
