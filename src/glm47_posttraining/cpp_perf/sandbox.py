"""Docker sandbox command construction and execution for C++ candidates."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from glm47_posttraining.constants import DEFAULT_CPP_SANDBOX_IMAGE

from .schema import AiderPolyglotTask, Catch2HarnessResult, CppTask, HarnessResult, TestCase


BASE_DOCKER_IMAGE = "gcc:13"
DEFAULT_DOCKER_IMAGE = DEFAULT_CPP_SANDBOX_IMAGE
DEFAULT_CPU = "3"
DEFAULT_MEMORY = "2g"
DEFAULT_RUN_TIMEOUT_S = 5
DEFAULT_RUNTIME_WARMUPS = 1
DEFAULT_RUNTIME_REPEATS = 3
SANDBOX_BACKEND_ENV = "GLM47_CPP_SANDBOX_BACKEND"
DOCKER_INFRASTRUCTURE_ERROR_MARKERS = (
    "cannot connect to the docker daemon",
    "error response from daemon",
    "error creating overlay mount",
    "failed to create shim task",
)


class SandboxInfrastructureError(RuntimeError):
    """The sandbox runtime failed before candidate code could be evaluated."""


def sandbox_backend() -> str:
    """Selected sandbox backend: ``docker`` (default) or ``local``.

    ``local`` runs the same stage scripts directly in this process's container
    with the working directory set to the scratch dir — for hosts without a
    Docker daemon (local container runtime). ``timeout`` and ``taskset`` timing
    semantics are identical; the Docker cgroup memory/pids caps and read-only
    rootfs do not apply, so candidate and oracle still race in the same
    environment but without container isolation.
    """

    backend = os.environ.get(SANDBOX_BACKEND_ENV, "docker").strip().lower() or "docker"
    if backend not in ("docker", "local"):
        raise ValueError(f"{SANDBOX_BACKEND_ENV} must be docker|local, got: {backend}")
    return backend


class _LocalCorePool:
    """Leases one host-visible core per concurrent candidate.

    Both backends share the host CPU namespace. Docker's ``--cpus 1`` is a
    quota, not a private cpuset, so a fixed ``taskset -c N`` stampedes one core
    there just as it does in local mode. Leasing a distinct core per candidate
    keeps candidate and oracle on the same core for the whole measurement.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._available: list[str] | None = None

    def _ensure(self) -> None:
        if self._available is None:
            self._available = [str(c) for c in sorted(os.sched_getaffinity(0))]

    @contextmanager
    def lease(self, preferred: str) -> Any:
        with self._cv:
            self._ensure()
            while not self._available:
                self._cv.wait()
            core = preferred if preferred in self._available else self._available[-1]
            self._available.remove(core)
        try:
            yield core
        finally:
            with self._cv:
                self._available.append(core)
                self._cv.notify()


_LOCAL_CORE_POOL = _LocalCorePool()


@contextmanager
def _sandbox_cpu(cpu: str) -> Any:
    """Resolve the pin target for one candidate's full measurement."""

    with _LOCAL_CORE_POOL.lease(cpu) as core:
        yield core


def sandbox_command(
    scratch: str | Path,
    script: str,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    memory: str = DEFAULT_MEMORY,
) -> list[str]:
    """Wrap one stage script for the selected backend.

    Every harness stage is a bash script using paths relative to the scratch
    dir; Docker mode mounts scratch at /work, local mode cd's into it.
    """

    if sandbox_backend() == "local":
        wrapped = f"cd {shlex.quote(str(Path(scratch).resolve()))} && ulimit -c 0 && {script}"
        return ["bash", "-lc", wrapped]
    return docker_base_args(scratch, image=image, memory=memory) + ["bash", "-lc", script]


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
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
       python3 gawk util-linux libboost-date-time-dev libtbb-dev \\
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


def compile_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    script = f"timeout {task.build.timeout_s}s {task.build.cmd}"
    return sandbox_command(scratch, script, image=image)


def reference_compile_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    # Some accepted PIE references rely on INT32_MAX arriving transitively
    # under their original compiler. Define only that legacy constant here:
    # force-including a standard header before source-level _GLIBCXX_DEBUG
    # changes libstdc++ ABI mode and breaks otherwise-valid references.
    script = (
        f"timeout {task.build.timeout_s}s g++ {task.reference.compiler_flags} "
        "-DINT32_MAX=__INT32_MAX__ reference.cpp -o reference"
    )
    return sandbox_command(scratch, script, image=image)


def sanitizer_command(task: CppTask, scratch: str | Path, *, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    script = (
        "timeout "
        f"{task.build.timeout_s}s g++ -O1 -g -std=c++20 -fsanitize=address,undefined "
        "candidate.cpp -o candidate_san"
    )
    return sandbox_command(scratch, script, image=image)


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
    return sandbox_command(scratch, script, image=image)


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
    return sandbox_command(scratch, script, image=image)


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
    return sandbox_command(scratch, script, image=image)


def dry_run_plan(
    task: CppTask,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    scratch: str = "/tmp/glm47-cpp-sandbox",
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
    scratch: str = "/tmp/glm47-cpp-preflight",
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

    with _sandbox_cpu(cpu) as pinned:
        if work_dir is None:
            with TemporaryDirectory(prefix="glm47-cpp-preflight-") as temp:
                return _run_runtime_preflight_in_directory(Path(temp), image=image, cpu=pinned)
        return _run_runtime_preflight_in_directory(Path(work_dir), image=image, cpu=pinned)


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

    with _sandbox_cpu(cpu) as pinned:
        if work_dir is None:
            with TemporaryDirectory(prefix="glm47-cpp-") as temp:
                return _run_in_directory(task, candidate_code, Path(temp), image=image, cpu=pinned)
        return _run_in_directory(task, candidate_code, Path(work_dir), image=image, cpu=pinned)


def _run_in_directory(task: CppTask, candidate_code: str, scratch: Path, *, image: str, cpu: str) -> HarnessResult:
    _prepare_scratch(scratch)
    (scratch / "candidate.cpp").write_text(candidate_code, encoding="utf-8")
    (scratch / "reference.cpp").write_text(task.oracle_solution, encoding="utf-8")
    tests = task.unit_tests + task.hidden_tests
    _write_tests(scratch, tests)

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
        failure_reason = _runtime_payload_reason(candidate_payload)
        return HarnessResult(
            timeout=failure_reason == "timeout",
            tests_passed=_tests_passed_after_runtime_failure(tests_passed, failure_reason),
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


def _tests_passed_after_runtime_failure(tests_passed: int, reason: str) -> int:
    """Turn a timed correctness/reliability failure into one failed test."""

    if reason in {"wrong_output", "nonzero_exit"}:
        return max(0, tests_passed - 1)
    return tests_passed


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
        "cat > .glm47_runtime_bench.py <<'PY'\n"
        + _runtime_benchmark_python().rstrip()
        + "\nPY\n"
        + "taskset -c "
        + shlex.quote(cpu)
        + " python3 ./.glm47_runtime_bench.py "
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
    proc = subprocess.run(args, check=False, capture_output=True, text=True)
    _raise_for_docker_infrastructure(args, proc)
    return proc


AIDER_COMPILE_TIMEOUT_S = 180
AIDER_RUN_TIMEOUT_S = 10
SANITIZER_ERROR_MARKERS = (
    "addresssanitizer",
    "undefinedbehaviorsanitizer",
    "runtime error:",
    "leaksanitizer",
)
THREAD_SANITIZER_ERROR_MARKERS = (
    "threadsanitizer: data race",
    "threadsanitizer: reported",
    "threadsanitizer: lock-order-inversion",
    "threadsanitizer: heap-use-after-free",
    "threadsanitizer: signal-unsafe call",
    "fatal: threadsanitizer",
)


def parse_catch2_xml(xml_path: str | Path) -> tuple[int, int, int, int]:
    """Return passed/total cases and assertions from Catch2 v2/v3 XML.

    The vendored C++ exercises use Catch2 v2, whose aggregate nodes expose
    ``successes`` rather than the often-assumed ``passes`` attribute.
    """

    path = Path(xml_path)
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0, 0, 0
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return 0, 0, 0, 0

    assertion_nodes = list(root.iter("OverallResults"))
    case_nodes = list(root.iter("OverallResultsCases"))
    if assertion_nodes and case_nodes:
        assertions = assertion_nodes[-1]
        cases = case_nodes[-1]
        passed_assertions = _catch_count(assertions, "successes") + _catch_count(
            assertions, "expectedFailures"
        )
        failed_assertions = _catch_count(assertions, "failures")
        passed_cases = _catch_count(cases, "successes") + _catch_count(cases, "expectedFailures")
        failed_cases = _catch_count(cases, "failures")
        return (
            passed_cases,
            passed_cases + failed_cases,
            passed_assertions,
            passed_assertions + failed_assertions,
        )

    # Small synthetic reports and some Catch2 v3 reporters only carry
    # per-test-case OverallResult nodes. Assertion totals may be unavailable.
    outcomes = [node for node in root.iter("TestCase")]
    passed_cases = 0
    passed_assertions = 0
    total_assertions = 0
    for test_case in outcomes:
        overall = test_case.find("OverallResult")
        if overall is None:
            continue
        if overall.attrib.get("success", "false").lower() == "true":
            passed_cases += 1
        passed = _catch_count(overall, "passes") + _catch_count(overall, "successes")
        failed = _catch_count(overall, "failures")
        passed_assertions += passed
        total_assertions += passed + failed
    return passed_cases, len(outcomes), passed_assertions, total_assertions


def _catch_count(node: ET.Element, name: str) -> int:
    try:
        return int(node.attrib.get(name, "0"))
    except ValueError:
        return 0


def run_aider_in_sandbox(
    task: AiderPolyglotTask,
    candidate_files: dict[str, str],
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    memory: str = DEFAULT_MEMORY,
    compile_timeout_s: int = AIDER_COMPILE_TIMEOUT_S,
    run_timeout_s: int = AIDER_RUN_TIMEOUT_S,
) -> Catch2HarnessResult:
    """Compile and execute a complete Polyglot edit with vendored Catch2.

    This intentionally uses the exercise's own ``test/catch.hpp`` and
    ``test/tests-main.cpp``. It therefore has no Catch2 v2/v3 host dependency
    and enables the same full test set as Aider's benchmark harness.
    """

    expected = set(task.solution_files)
    if set(candidate_files) != expected:
        missing = sorted(expected - set(candidate_files))
        extra = sorted(set(candidate_files) - expected)
        return Catch2HarnessResult(
            compile_error=True,
            logs={"compile": f"candidate file mismatch; missing={missing}, extra={extra}"},
        )

    with TemporaryDirectory(prefix="aider-polyglot-cpp-") as temp:
        scratch = Path(temp)
        _prepare_scratch(scratch)
        _write_polyglot_files(scratch, task.support_files)
        _write_polyglot_files(scratch, task.test_files)
        _write_polyglot_files(scratch, candidate_files)

        normal_compile = _aider_compile_command(
            task,
            scratch,
            output="test_runner",
            sanitizer=False,
            image=image,
            memory=memory,
            timeout_s=compile_timeout_s,
        )
        compiled = _run(normal_compile)
        compile_logs = _combined_logs(compiled)
        if compiled.returncode != 0:
            return Catch2HarnessResult(
                compile_error=True,
                timeout=compiled.returncode == 124,
                logs={"compile": compile_logs[-8000:]},
            )

        normal_run = _run(
            _aider_test_command(
                scratch,
                binary="test_runner",
                report="results.xml",
                image=image,
                memory=memory,
                timeout_s=run_timeout_s,
            )
        )
        run_logs = _combined_logs(normal_run)
        if normal_run.returncode == 124:
            return Catch2HarnessResult(
                timeout=True,
                logs={"compile": compile_logs[-4000:], "run": run_logs[-8000:]},
            )

        passed_cases, total_cases, passed_assertions, total_assertions = parse_catch2_xml(
            scratch / "results.xml"
        )

        sanitizer_compile = _run(
            _aider_compile_command(
                task,
                scratch,
                output="test_runner_san",
                sanitizer=True,
                image=image,
                memory=memory,
                timeout_s=compile_timeout_s,
            )
        )
        sanitizer_compile_logs = _combined_logs(sanitizer_compile)
        if sanitizer_compile.returncode != 0:
            return Catch2HarnessResult(
                sanitizer_error=sanitizer_compile.returncode != 124,
                timeout=sanitizer_compile.returncode == 124,
                passed_test_cases=passed_cases,
                total_test_cases=total_cases,
                passed_assertions=passed_assertions,
                total_assertions=total_assertions,
                logs={
                    "compile": compile_logs[-4000:],
                    "run": run_logs[-4000:],
                    "sanitizer_compile": sanitizer_compile_logs[-8000:],
                },
            )

        sanitizer_run = _run(
            _aider_test_command(
                scratch,
                binary="test_runner_san",
                report="sanitizer-results.xml",
                image=image,
                memory=memory,
                timeout_s=run_timeout_s,
                sanitizer=True,
            )
        )
        sanitizer_logs = _combined_logs(sanitizer_run)
        sanitizer_error = any(
            marker in sanitizer_logs.lower() for marker in SANITIZER_ERROR_MARKERS
        )
        result = Catch2HarnessResult(
            sanitizer_error=sanitizer_error,
            timeout=sanitizer_run.returncode == 124,
            passed_test_cases=passed_cases,
            total_test_cases=total_cases,
            passed_assertions=passed_assertions,
            total_assertions=total_assertions,
            logs={
                "compile": compile_logs[-4000:],
                "run": run_logs[-8000:],
                "sanitizer": sanitizer_logs[-8000:],
            },
        )
        if (
            task.effective_rubric_category != "state_concurrency"
            or sanitizer_error
            or result.timeout
            or not result.all_tests_pass
        ):
            return result

        thread_compile = _run(
            _aider_compile_command(
                task,
                scratch,
                output="test_runner_tsan",
                sanitizer=False,
                thread_sanitizer=True,
                image=image,
                memory=memory,
                timeout_s=compile_timeout_s,
            )
        )
        thread_compile_logs = _combined_logs(thread_compile)
        if thread_compile.returncode != 0:
            return result.model_copy(
                update={
                    "logs": {
                        **result.logs,
                        "thread_sanitizer_compile": thread_compile_logs[-8000:],
                    },
                }
            )

        thread_run = _run(
            _aider_test_command(
                scratch,
                binary="test_runner_tsan",
                report="thread-sanitizer-results.xml",
                image=image,
                memory=memory,
                timeout_s=run_timeout_s,
                thread_sanitizer=True,
            )
        )
        thread_logs = _combined_logs(thread_run)
        tsan_passed, tsan_total, tsan_assertions, tsan_assertion_total = parse_catch2_xml(
            scratch / "thread-sanitizer-results.xml"
        )
        thread_error = any(
            marker in thread_logs.lower() for marker in THREAD_SANITIZER_ERROR_MARKERS
        )
        if thread_run.returncode not in {0, 124}:
            thread_error = True
        if tsan_total and (
            tsan_passed != tsan_total or tsan_assertions != tsan_assertion_total
        ):
            thread_error = True
        return result.model_copy(
            update={
                "thread_sanitizer_ran": True,
                "thread_sanitizer_error": thread_error,
                "thread_sanitizer_timeout": thread_run.returncode == 124,
                "logs": {**result.logs, "thread_sanitizer": thread_logs[-8000:]},
            }
        )


def _write_polyglot_files(scratch: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        destination = scratch / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def _aider_compile_command(
    task: AiderPolyglotTask,
    scratch: Path,
    *,
    output: str,
    sanitizer: bool,
    thread_sanitizer: bool = False,
    image: str,
    memory: str,
    timeout_s: int,
) -> list[str]:
    solution_sources = [
        name for name in task.solution_files if Path(name).suffix in {".cc", ".cpp", ".cxx"}
    ]
    test_sources = [
        name for name in task.test_files if Path(name).suffix in {".cc", ".cpp", ".cxx"}
    ]
    main_source = "test/tests-main.cpp"
    if main_source not in task.support_files:
        raise ValueError(f"{task.task_id} is missing {main_source}")
    instrumented = sanitizer or thread_sanitizer
    flags = [
        "g++",
        "-O1" if instrumented else "-O2",
        "-g" if instrumented else "-DNDEBUG",
        "-std=c++17",
        "-DEXERCISM_RUN_ALL_TESTS",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
    ]
    if sanitizer:
        # The normal build already enforces -Werror. GCC 13 emits
        # maybe-uninitialized false positives inside vendored Catch2 v2 when
        # optimization and sanitizer instrumentation are combined.
        flags.extend(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"])
    elif thread_sanitizer:
        flags.extend(["-fsanitize=thread", "-fno-omit-frame-pointer"])
    else:
        flags.append("-Werror")
    flags.extend(solution_sources + test_sources + [main_source, "-I.", "-pthread"])
    flags.extend(task.link_flags)
    flags.extend(["-o", output])
    script = f"timeout {int(timeout_s)}s {shlex.join(flags)}"
    return sandbox_command(scratch, script, image=image, memory=memory)


def _aider_test_command(
    scratch: Path,
    *,
    binary: str,
    report: str,
    image: str,
    memory: str,
    timeout_s: int,
    sanitizer: bool = False,
    thread_sanitizer: bool = False,
) -> list[str]:
    env = ""
    if sanitizer:
        # LeakSanitizer cannot attach in ptrace-restricted local containers;
        # Docker reward workers retain leak detection.
        detect_leaks = "0" if sandbox_backend() == "local" else "1"
        env = (
            f"ASAN_OPTIONS=detect_leaks={detect_leaks}:halt_on_error=1 "
            "UBSAN_OPTIONS=halt_on_error=1 "
        )
    elif thread_sanitizer:
        env = "TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1 "
    script = (
        f"{env}timeout {int(timeout_s)}s ./{shlex.quote(binary)} "
        f"--reporter xml --out {shlex.quote(report)}"
    )
    return sandbox_command(scratch, script, image=image, memory=memory)


def _raise_for_docker_infrastructure(
    args: list[str], proc: subprocess.CompletedProcess[str]
) -> None:
    if not args or args[0] != "docker" or proc.returncode == 0:
        return
    logs = _combined_logs(proc)
    normalized = logs.lower()
    if proc.returncode == 125 or any(marker in normalized for marker in DOCKER_INFRASTRUCTURE_ERROR_MARKERS):
        raise SandboxInfrastructureError(
            f"Docker sandbox infrastructure failed (exit {proc.returncode}): {logs[-2000:]}"
        )
