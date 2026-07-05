"""Docker sandbox command construction and execution for C++ candidates."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from w8_biayn.constants import DEFAULT_CPP_SANDBOX_IMAGE

from .schema import CppTask, HarnessResult, TestCase


BASE_DOCKER_IMAGE = "gcc:13"
DEFAULT_DOCKER_IMAGE = DEFAULT_CPP_SANDBOX_IMAGE
DEFAULT_CPU = "3"
DEFAULT_MEMORY = "2g"
DEFAULT_RUN_TIMEOUT_S = 5
DEFAULT_RUNTIME_WARMUPS = 1
DEFAULT_RUNTIME_REPEATS = 3


@dataclass(frozen=True)
class RuntimePreflightResult:
    """Result of checking that the runtime harness works inside the sandbox."""

    ok: bool
    runtime_cpu_ns: int | None
    runtime_wall_ns: int | None
    returncode: int | None
    logs: str
    reason: str
    command: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "runtime_cpu_ns": self.runtime_cpu_ns,
            "runtime_wall_ns": self.runtime_wall_ns,
            "returncode": self.returncode,
            "logs": self.logs,
            "reason": self.reason,
            "command": self.command,
        }


def docker_base_args(
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    memory: str = DEFAULT_MEMORY,
) -> list[str]:
    """Return the locked-down Docker prefix used by compile, test, and timing steps."""

    return [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
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
        "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{Path(scratch).resolve()}:/work:rw",
        "-w",
        "/work",
        image,
    ]


def sandbox_image_dockerfile() -> str:
    """Return the Dockerfile for the default C++ runtime sandbox image."""

    return f"""FROM {BASE_DOCKER_IMAGE}
RUN apt-get update \\
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3 \\
    && rm -rf /var/lib/apt/lists/*
"""


def build_sandbox_image_command(*, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    """Return the docker build command for the default sandbox image."""

    return ["docker", "build", "-t", image, "-"]


def sandbox_image_build_plan(*, image: str = DEFAULT_DOCKER_IMAGE) -> str:
    """Render the default sandbox-image build command and Dockerfile."""

    return "\n".join(
        [
            "# C++ runtime sandbox image build dry run",
            f"{shlex.join(build_sandbox_image_command(image=image))} <<'DOCKERFILE'",
            sandbox_image_dockerfile().rstrip(),
            "DOCKERFILE",
        ]
    )


def build_sandbox_image(*, image: str = DEFAULT_DOCKER_IMAGE) -> subprocess.CompletedProcess[str]:
    """Build the default sandbox image from stdin."""

    return subprocess.run(
        build_sandbox_image_command(image=image),
        input=sandbox_image_dockerfile(),
        check=False,
        capture_output=True,
        text=True,
    )


DEFAULT_SWE_AGENT_IMAGE = "w8-biayn-swe-agent:latest"
# SWE-ReX server baked into the edit-loop image. swerex boots the container with
# `swerex-remote ... || pip install pipx && pipx run swe-rex`, so without a
# pre-installed server (and with no pip in the base image) the container dies
# with "swerex-remote: not found". Pin it to the client version SWE-agent 1.1.0
# ships (SWE-ReX 1.4.0) so the client/server protocol matches.
SWEREX_PIN = "1.4.0"


def swe_agent_image_dockerfile(*, base_image: str = DEFAULT_DOCKER_IMAGE) -> str:
    """Return the Dockerfile for the SWE-agent edit-loop image.

    SWE-agent runs its bash/edit actions inside this container via the SWE-ReX
    server, so on top of the C++ sandbox image it needs ``git`` (for the repo
    checkout) and a **pre-installed ``swerex-remote``** (the base image has no
    pip and no network at container-start, so swerex cannot self-install its
    runtime). Grading stays in the separate hardened ``run_in_sandbox`` image and
    is never done here.
    """

    return f"""FROM {base_image}
RUN apt-get update \\
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git python3-pip \\
    && rm -rf /var/lib/apt/lists/* \\
    && pip3 install --no-cache-dir --break-system-packages swe-rex=={SWEREX_PIN}
"""


def build_swe_agent_image_command(*, image: str = DEFAULT_SWE_AGENT_IMAGE) -> list[str]:
    """Return the docker build command for the SWE-agent edit-loop image."""

    return ["docker", "build", "-t", image, "-"]


def swe_agent_image_build_plan(
    *,
    image: str = DEFAULT_SWE_AGENT_IMAGE,
    base_image: str = DEFAULT_DOCKER_IMAGE,
) -> str:
    """Render the SWE-agent edit-loop image build command and Dockerfile."""

    return "\n".join(
        [
            "# SWE-agent edit-loop image build dry run",
            f"{shlex.join(build_swe_agent_image_command(image=image))} <<'DOCKERFILE'",
            swe_agent_image_dockerfile(base_image=base_image).rstrip(),
            "DOCKERFILE",
        ]
    )


def build_swe_agent_image(
    *,
    image: str = DEFAULT_SWE_AGENT_IMAGE,
    base_image: str = DEFAULT_DOCKER_IMAGE,
) -> subprocess.CompletedProcess[str]:
    """Build the SWE-agent edit-loop image from stdin (base image must exist first)."""

    return subprocess.run(
        build_swe_agent_image_command(image=image),
        input=swe_agent_image_dockerfile(base_image=base_image),
        check=False,
        capture_output=True,
        text=True,
    )


def compile_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    script = f"timeout {task.build.timeout_s}s {task.build.cmd}"
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def reference_compile_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    script = f"timeout {task.build.timeout_s}s g++ {task.reference.compiler_flags} reference.cpp -o reference"
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
    normalize = (
        "normalize(){ awk '{ sub(/[[:space:]]+$/, \"\"); lines[NR]=$0 } "
        "END { n=NR; while (n>0 && lines[n]==\"\") n--; "
        "s=1; while (s<=n && lines[s]==\"\") s++; "
        "for (i=s; i<=n; i++) print lines[i] }' \"$1\"; }; "
    )
    script = (
        normalize
        + f"timeout {timeout_s}s taskset -c {shlex.quote(cpu)} ./{binary} "
        f"< tests/{index}.in > tests/{index}.actual && "
        f"normalize tests/{index}.out > tests/{index}.expected.norm && "
        f"normalize tests/{index}.actual > tests/{index}.actual.norm && "
        f"diff -u tests/{index}.expected.norm tests/{index}.actual.norm"
    )
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def runtime_benchmark_command(
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    binary: str = "candidate",
    test_count: int = 1,
    warmups: int = DEFAULT_RUNTIME_WARMUPS,
    repeats: int = DEFAULT_RUNTIME_REPEATS,
    validate_output: bool = True,
) -> list[str]:
    script = _runtime_benchmark_shell(
        binary=binary,
        cpu=cpu,
        timeout_s=timeout_s,
        test_count=test_count,
        warmups=warmups,
        repeats=repeats,
        validate_output=validate_output,
    )
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def runtime_preflight_command(
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
    warmups: int = DEFAULT_RUNTIME_WARMUPS,
    repeats: int = DEFAULT_RUNTIME_REPEATS,
) -> list[str]:
    script = (
        "cat > preflight.cpp <<'CPP'\n"
        "#include <iostream>\n"
        "int main(){volatile unsigned long long s=0; "
        "for(int i=0;i<100000;i++) s+=i; std::cout<<s<<\"\\n\"; return 0;}\n"
        "CPP\n"
        "g++ -O2 -std=c++20 preflight.cpp -o preflight && "
        "mkdir -p tests && printf '' > tests/0.in && printf '4999950000\\n' > tests/0.out && "
        + _runtime_benchmark_shell(
            binary="preflight",
            cpu=cpu,
            timeout_s=timeout_s,
            test_count=1,
            warmups=warmups,
            repeats=repeats,
        )
    )
    return docker_base_args(scratch, image=image) + ["bash", "-lc", script]


def dry_run_plan(
    task: CppTask,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    scratch: str = "/tmp/w8-cpp-sandbox",
    warmups: int = DEFAULT_RUNTIME_WARMUPS,
    repeats: int = DEFAULT_RUNTIME_REPEATS,
) -> str:
    """Render the commands the harness would run."""

    tests = task.unit_tests + task.hidden_tests
    lines = [
        "# C++ performance harness dry run",
        "# scratch contains candidate.cpp, reference.cpp, and tests/<n>.in|out",
        shlex.join(compile_command(task, scratch, image=image)),
        shlex.join(sanitizer_command(task, scratch, image=image)),
        shlex.join(run_test_command(0, scratch, image=image, cpu=cpu)),
        shlex.join(reference_compile_command(task, scratch, image=image)),
        shlex.join(
            runtime_benchmark_command(
                scratch,
                image=image,
                cpu=cpu,
                binary="candidate",
                test_count=len(tests),
                warmups=warmups,
                repeats=repeats,
            )
        ),
        shlex.join(
            runtime_benchmark_command(
                scratch,
                image=image,
                cpu=cpu,
                binary="reference",
                test_count=len(tests),
                warmups=warmups,
                repeats=repeats,
                validate_output=False,
            )
        ),
    ]
    return "\n".join(lines)


def runtime_preflight_plan(
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    scratch: str = "/tmp/w8-cpp-preflight",
    warmups: int = DEFAULT_RUNTIME_WARMUPS,
    repeats: int = DEFAULT_RUNTIME_REPEATS,
) -> str:
    """Render the command used to verify runtime measurement in the sandbox."""

    return "\n".join(
        [
            "# C++ runtime preflight dry run",
            "# succeeds when Docker can compile, run, and time a child process",
            shlex.join(runtime_preflight_command(scratch, image=image, cpu=cpu, warmups=warmups, repeats=repeats)),
        ]
    )


def run_runtime_preflight(
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    work_dir: str | Path | None = None,
) -> RuntimePreflightResult:
    """Check that CPU-time runtime measurement works in Docker."""

    if work_dir is None:
        with TemporaryDirectory(prefix="w8-cpp-preflight-") as temp:
            return _run_runtime_preflight_in_directory(Path(temp), image=image, cpu=cpu)
    return _run_runtime_preflight_in_directory(Path(work_dir), image=image, cpu=cpu)


def _run_runtime_preflight_in_directory(scratch: Path, *, image: str, cpu: str) -> RuntimePreflightResult:
    _prepare_scratch(scratch)
    command = runtime_preflight_command(scratch, image=image, cpu=cpu)
    try:
        proc = _run(command)
    except OSError as exc:
        return RuntimePreflightResult(
            ok=False,
            runtime_cpu_ns=None,
            runtime_wall_ns=None,
            returncode=None,
            logs=str(exc),
            reason="command_error",
            command=command,
        )
    logs = _combined_logs(proc)
    payload = parse_runtime_benchmark_output(proc.stdout)
    runtime_cpu_ns = _positive_int_from_payload(payload, "runtime_cpu_ns")
    runtime_wall_ns = _positive_int_from_payload(payload, "runtime_wall_ns")
    reason = str(payload.get("reason", "runtime_command_failed")) if payload else "runtime_command_failed"
    if proc.returncode != 0:
        return RuntimePreflightResult(
            ok=False,
            runtime_cpu_ns=runtime_cpu_ns,
            runtime_wall_ns=runtime_wall_ns,
            returncode=proc.returncode,
            logs=logs,
            reason=reason,
            command=command,
        )
    if runtime_cpu_ns is None or runtime_wall_ns is None:
        return RuntimePreflightResult(
            ok=False,
            runtime_cpu_ns=runtime_cpu_ns,
            runtime_wall_ns=runtime_wall_ns,
            returncode=proc.returncode,
            logs=logs,
            reason="missing_runtime",
            command=command,
        )
    return RuntimePreflightResult(
        ok=True,
        runtime_cpu_ns=runtime_cpu_ns,
        runtime_wall_ns=runtime_wall_ns,
        returncode=proc.returncode,
        logs=logs,
        reason="ok",
        command=command,
    )


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


def run_in_directory_prewritten(
    task: CppTask,
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
) -> HarnessResult:
    """Grade a candidate.cpp that already exists in ``scratch`` without touching it.

    Materializes ``reference.cpp`` and ``tests/`` from ``task`` but leaves the
    caller-written ``candidate.cpp`` byte-for-byte intact, so the literal file
    the agent produced is what gets compiled, tested, and benchmarked. This is
    the file-state entry point used by the agentic lane; ``run_in_sandbox`` is
    the equivalent that writes the candidate string itself.
    """

    scratch_path = Path(scratch)
    _prepare_scratch(scratch_path)
    (scratch_path / "reference.cpp").write_text(task.oracle_solution, encoding="utf-8")
    _write_tests(scratch_path, task.unit_tests + task.hidden_tests)
    return _grade_prepared_directory(task, scratch_path, image=image, cpu=cpu)


def _run_in_directory(task: CppTask, candidate_code: str, scratch: Path, *, image: str, cpu: str) -> HarnessResult:
    _prepare_scratch(scratch)
    (scratch / "candidate.cpp").write_text(candidate_code, encoding="utf-8")
    (scratch / "reference.cpp").write_text(task.oracle_solution, encoding="utf-8")
    _write_tests(scratch, task.unit_tests + task.hidden_tests)
    return _grade_prepared_directory(task, scratch, image=image, cpu=cpu)


def _grade_prepared_directory(task: CppTask, scratch: Path, *, image: str, cpu: str) -> HarnessResult:
    """Compile, sanitize, test, and benchmark the candidate/reference already in ``scratch``.

    ``candidate.cpp``, ``reference.cpp``, and ``tests/<n>.in|out`` must already
    exist in ``scratch``; how they got there is the caller's choice.
    """

    tests = task.unit_tests + task.hidden_tests

    compile_proc = _run(compile_command(task, scratch, image=image))
    if compile_proc.returncode != 0:
        return HarnessResult(compile_error=True, tests_total=len(tests), logs={"compile": _combined_logs(compile_proc)})

    sanitizer_proc = _run(sanitizer_command(task, scratch, image=image))
    if sanitizer_proc.returncode != 0:
        return HarnessResult(
            sanitizer_error=True,
            tests_total=len(tests),
            logs={"sanitizer": _combined_logs(sanitizer_proc)},
        )

    tests_passed = 0
    logs: dict[str, str] = {}
    for index, _test in enumerate(tests):
        proc = _run(run_test_command(index, scratch, image=image, cpu=cpu))
        if proc.returncode == 0:
            tests_passed += 1
        else:
            logs[f"test_{index}"] = _combined_logs(proc)

    if tests_passed != len(tests):
        return HarnessResult(tests_passed=tests_passed, tests_total=len(tests), logs=logs)

    reference_compile_proc = _run(reference_compile_command(task, scratch, image=image))
    if reference_compile_proc.returncode != 0:
        logs["reference_compile"] = _combined_logs(reference_compile_proc)
        return HarnessResult(tests_passed=tests_passed, tests_total=len(tests), logs=logs)

    candidate_payload, candidate_logs, candidate_returncode = _run_runtime_benchmark(
        scratch,
        image=image,
        cpu=cpu,
        binary="candidate",
        test_count=len(tests),
    )
    if candidate_returncode != 0 or not _runtime_payload_ok(candidate_payload):
        logs["runtime_candidate"] = candidate_logs
        return HarnessResult(
            timeout=_runtime_payload_reason(candidate_payload) == "timeout",
            tests_passed=tests_passed,
            tests_total=len(tests),
            logs=logs,
        )

    reference_payload, reference_logs, reference_returncode = _run_runtime_benchmark(
        scratch,
        image=image,
        cpu=cpu,
        binary="reference",
        test_count=len(tests),
        validate_output=False,
    )
    if reference_returncode != 0 or not _runtime_payload_ok(reference_payload):
        logs["runtime_reference"] = reference_logs
        return HarnessResult(
            timeout=_runtime_payload_reason(reference_payload) == "timeout",
            tests_passed=tests_passed,
            tests_total=len(tests),
            runtime_cpu_ns=_positive_int_from_payload(candidate_payload, "runtime_cpu_ns"),
            runtime_wall_ns=_positive_int_from_payload(candidate_payload, "runtime_wall_ns"),
            logs=logs,
        )

    runtime_cpu_ns = _positive_int_from_payload(candidate_payload, "runtime_cpu_ns")
    runtime_wall_ns = _positive_int_from_payload(candidate_payload, "runtime_wall_ns")
    reference_runtime_cpu_ns = _positive_int_from_payload(reference_payload, "runtime_cpu_ns")
    reference_runtime_wall_ns = _positive_int_from_payload(reference_payload, "runtime_wall_ns")
    return HarnessResult(
        tests_passed=tests_passed,
        tests_total=len(tests),
        runtime_cpu_ns=runtime_cpu_ns,
        runtime_wall_ns=runtime_wall_ns,
        reference_runtime_cpu_ns=reference_runtime_cpu_ns,
        reference_runtime_wall_ns=reference_runtime_wall_ns,
        runtime_speedup=_runtime_speedup(reference_runtime_cpu_ns, runtime_cpu_ns),
        logs=logs,
    )


def parse_runtime_benchmark_output(text: str) -> dict[str, Any] | None:
    """Parse the JSON line emitted by the runtime benchmark helper."""

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _run_runtime_benchmark(
    scratch: Path,
    *,
    image: str,
    cpu: str,
    binary: str,
    test_count: int,
    validate_output: bool = True,
) -> tuple[dict[str, Any] | None, str, int]:
    command = runtime_benchmark_command(
        scratch, image=image, cpu=cpu, binary=binary, test_count=test_count, validate_output=validate_output
    )
    proc = _run(command)
    return parse_runtime_benchmark_output(proc.stdout), _combined_logs(proc), proc.returncode


def _runtime_benchmark_shell(
    *,
    binary: str,
    cpu: str,
    timeout_s: int,
    test_count: int,
    warmups: int,
    repeats: int,
    validate_output: bool = True,
) -> str:
    return (
        "cat > /tmp/w8_runtime_bench.py <<'PY'\n"
        + _runtime_benchmark_python().rstrip()
        + "\nPY\n"
        + "taskset -c "
        + shlex.quote(cpu)
        + " python3 /tmp/w8_runtime_bench.py "
        + f"--binary {shlex.quote('./' + binary)} "
        + f"--test-count {test_count} "
        + f"--timeout-s {timeout_s} "
        + f"--warmups {warmups} "
        + f"--repeats {repeats} "
        + f"--validate-output {1 if validate_output else 0}"
    )


def _runtime_benchmark_python() -> str:
    return r"""
import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def normalize(text):
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and lines[-1] == "":
        lines.pop()
    while lines and lines[0] == "":
        lines.pop(0)
    return "\n".join(lines)


def child_cpu_ns():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return int((usage.ru_utime + usage.ru_stime) * 1_000_000_000)


def run_once(binary, input_text, expected_text, timeout_s, test_index, validate_output=True):
    before_cpu = child_cpu_ns()
    before_wall = time.perf_counter_ns()
    try:
        proc = subprocess.run(
            [binary],
            input=input_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "reason": "timeout",
            "test_index": test_index,
            "stdout": str(exc.stdout or "")[-2000:],
            "stderr": str(exc.stderr or "")[-2000:],
        }
    wall_ns = time.perf_counter_ns() - before_wall
    cpu_ns = child_cpu_ns() - before_cpu
    if proc.returncode != 0:
        return {
            "ok": False,
            "reason": "nonzero_exit",
            "test_index": test_index,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-2000:],
        }
    # The reference oracle is the pre-validated PIE v1 solution and correctness was
    # already established by the candidate test phase, so we only time it; re-checking
    # its stdout would spuriously fail benignly-formatted oracles (e.g. a leading blank
    # line) and zero out the reference runtime. Candidates are still re-validated.
    if validate_output and normalize(proc.stdout) != normalize(expected_text):
        return {
            "ok": False,
            "reason": "wrong_output",
            "test_index": test_index,
            "stdout": proc.stdout[-2000:],
            "expected": expected_text[-2000:],
        }
    return {"ok": True, "cpu_ns": max(1, cpu_ns), "wall_ns": max(1, wall_ns)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--timeout-s", type=int, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--validate-output", type=int, default=1)
    args = parser.parse_args()
    validate_output = bool(args.validate_output)

    per_test = []
    total_cpu_ns = 0
    total_wall_ns = 0
    for test_index in range(args.test_count):
        input_text = Path(f"tests/{test_index}.in").read_text()
        expected_text = Path(f"tests/{test_index}.out").read_text()
        for _ in range(args.warmups):
            result = run_once(args.binary, input_text, expected_text, args.timeout_s, test_index, validate_output)
            if not result["ok"]:
                print(json.dumps(result, sort_keys=True))
                sys.exit(2)
        cpu_samples = []
        wall_samples = []
        for _ in range(args.repeats):
            result = run_once(args.binary, input_text, expected_text, args.timeout_s, test_index, validate_output)
            if not result["ok"]:
                print(json.dumps(result, sort_keys=True))
                sys.exit(2)
            cpu_samples.append(int(result["cpu_ns"]))
            wall_samples.append(int(result["wall_ns"]))
        cpu_median = int(statistics.median(cpu_samples))
        wall_median = int(statistics.median(wall_samples))
        per_test.append(
            {
                "test_index": test_index,
                "cpu_ns": cpu_median,
                "wall_ns": wall_median,
                "cpu_samples": cpu_samples,
                "wall_samples": wall_samples,
            }
        )
        total_cpu_ns += cpu_median
        total_wall_ns += wall_median
    print(
        json.dumps(
            {
                "ok": True,
                "runtime_cpu_ns": total_cpu_ns,
                "runtime_wall_ns": total_wall_ns,
                "warmups": args.warmups,
                "repeats": args.repeats,
                "per_test": per_test,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
"""


def _runtime_payload_ok(payload: dict[str, Any] | None) -> bool:
    return (
        bool(payload and payload.get("ok") is True)
        and _positive_int_from_payload(payload, "runtime_cpu_ns") is not None
        and _positive_int_from_payload(payload, "runtime_wall_ns") is not None
    )


def _runtime_payload_reason(payload: dict[str, Any] | None) -> str:
    return str(payload.get("reason", "missing_runtime")) if payload else "missing_runtime"


def _positive_int_from_payload(payload: dict[str, Any] | None, key: str) -> int | None:
    if payload is None:
        return None
    value = payload.get(key)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _runtime_speedup(reference_runtime_cpu_ns: int | None, runtime_cpu_ns: int | None) -> float | None:
    if reference_runtime_cpu_ns is None or runtime_cpu_ns is None or runtime_cpu_ns <= 0:
        return None
    return reference_runtime_cpu_ns / runtime_cpu_ns


def _write_tests(scratch: Path, tests: list[TestCase]) -> None:
    tests_dir = scratch / "tests"
    tests_dir.mkdir(exist_ok=True)
    for index, test in enumerate(tests):
        (tests_dir / f"{index}.in").write_text(test.input, encoding="utf-8")
        (tests_dir / f"{index}.out").write_text(test.expected, encoding="utf-8")


def _prepare_scratch(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    scratch.chmod(0o777)


def _combined_logs(proc: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part for part in (proc.stdout, proc.stderr) if part)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)
