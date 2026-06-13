"""Docker sandbox command construction and execution for C++ candidates."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .schema import CppTask, HarnessResult, TestCase


DEFAULT_DOCKER_IMAGE = "gcc:13"
DEFAULT_CPU = "3"
DEFAULT_MEMORY = "2g"
DEFAULT_RUN_TIMEOUT_S = 5


def docker_base_args(
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    memory: str = DEFAULT_MEMORY,
) -> list[str]:
    """Return the locked-down Docker prefix used by compile/test/perf steps."""

    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        memory,
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "PERFMON",
        "--security-opt",
        "no-new-privileges",
        "--security-opt",
        "seccomp=unconfined",
        "-v",
        f"{Path(scratch).resolve()}:/work:rw",
        "-w",
        "/work",
        image,
    ]


def compile_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    script = f"timeout {task.build.timeout_s}s {task.build.cmd}"
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def sanitizer_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    script = (
        "timeout "
        f"{task.build.timeout_s}s g++ -O1 -g -std=c++20 -fsanitize=address,undefined "
        "candidate.cpp -o candidate_san"
    )
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def run_test_command(
    index: int,
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    binary: str = "candidate",
) -> list[str]:
    script = (
        f"timeout {timeout_s}s taskset -c {shlex.quote(cpu)} ./{binary} "
        f"< tests/{index}.in > tests/{index}.actual && "
        f"diff -u tests/{index}.out tests/{index}.actual"
    )
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def perf_command(
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    test_index: int = 0,
) -> list[str]:
    script = (
        f"timeout {timeout_s}s taskset -c {shlex.quote(cpu)} "
        "perf stat -e instructions:u -x, ./candidate "
        f"< tests/{test_index}.in > /tmp/perf.out"
    )
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def dry_run_plan(
    task: CppTask,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    scratch: str = "/tmp/w8-cpp-sandbox",
) -> str:
    """Render the commands the harness would run."""

    lines = [
        "# C++ performance harness dry run",
        "# scratch contains candidate.cpp and tests/<n>.in|out",
        shlex.join(compile_command(task, scratch, image=image)),
        shlex.join(sanitizer_command(task, scratch, image=image)),
        shlex.join(run_test_command(0, scratch, image=image, cpu=cpu)),
        shlex.join(perf_command(scratch, image=image, cpu=cpu)),
    ]
    return "\n".join(lines)


def run_in_sandbox(
    task: CppTask,
    candidate_code: str,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    work_dir: str | Path | None = None,
) -> HarnessResult:
    """Compile, test, sanitize, and measure one candidate in Docker."""

    if work_dir is None:
        with TemporaryDirectory(prefix="w8-cpp-") as temp:
            return _run_in_directory(task, candidate_code, Path(temp), image=image, cpu=cpu)
    return _run_in_directory(task, candidate_code, Path(work_dir), image=image, cpu=cpu)


def _run_in_directory(task: CppTask, candidate_code: str, scratch: Path, *, image: str, cpu: str) -> HarnessResult:
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "candidate.cpp").write_text(candidate_code, encoding="utf-8")
    tests = task.unit_tests + task.hidden_tests
    _write_tests(scratch, tests)

    compile_proc = _run(compile_command(task, scratch, image=image))
    if compile_proc.returncode != 0:
        return HarnessResult(compile_error=True, tests_total=len(tests), logs={"compile": compile_proc.stderr})

    sanitizer_proc = _run(sanitizer_command(task, scratch, image=image))
    if sanitizer_proc.returncode != 0:
        return HarnessResult(sanitizer_error=True, tests_total=len(tests), logs={"sanitizer": sanitizer_proc.stderr})

    tests_passed = 0
    logs: dict[str, str] = {}
    for index, _test in enumerate(tests):
        proc = _run(run_test_command(index, scratch, image=image, cpu=cpu))
        if proc.returncode == 0:
            tests_passed += 1
        else:
            logs[f"test_{index}"] = proc.stderr or proc.stdout

    if tests_passed != len(tests):
        return HarnessResult(tests_passed=tests_passed, tests_total=len(tests), logs=logs)

    perf_proc = _run(perf_command(scratch, image=image, cpu=cpu))
    if perf_proc.returncode != 0:
        logs["perf"] = perf_proc.stderr or perf_proc.stdout
        return HarnessResult(timeout=True, tests_passed=tests_passed, tests_total=len(tests), logs=logs)
    instr_count = parse_perf_instructions(perf_proc.stderr or perf_proc.stdout)
    return HarnessResult(
        tests_passed=tests_passed,
        tests_total=len(tests),
        instr_count=instr_count,
        logs=logs,
    )


def parse_perf_instructions(text: str) -> int:
    """Parse `perf stat -x, -e instructions:u` output."""

    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[2].startswith("instructions"):
            raw = parts[0].replace(",", "")
            return int(float(raw))
    raise ValueError("Could not parse instructions:u from perf output")


def _write_tests(scratch: Path, tests: list[TestCase]) -> None:
    tests_dir = scratch / "tests"
    tests_dir.mkdir(exist_ok=True)
    for index, test in enumerate(tests):
        (tests_dir / f"{index}.in").write_text(test.input, encoding="utf-8")
        (tests_dir / f"{index}.out").write_text(test.expected, encoding="utf-8")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)
