"""Docker sandbox command construction and execution for C++ candidates."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from w8_biayn.constants import DEFAULT_CPP_SANDBOX_IMAGE

from .schema import CppTask, HarnessResult, TestCase


BASE_DOCKER_IMAGE = "gcc:13"
DEFAULT_DOCKER_IMAGE = DEFAULT_CPP_SANDBOX_IMAGE
DEFAULT_CPU = "3"
DEFAULT_MEMORY = "2g"
DEFAULT_RUN_TIMEOUT_S = 5


@dataclass(frozen=True)
class PerfPreflightResult:
    """Result of checking that `instructions:u` is usable inside the sandbox."""

    ok: bool
    instr_count: int | None
    returncode: int | None
    logs: str
    reason: str
    command: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "instr_count": self.instr_count,
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


def sandbox_image_dockerfile() -> str:
    """Return the Dockerfile for the default C++ perf sandbox image."""

    return f"""FROM {BASE_DOCKER_IMAGE}
RUN apt-get update \\
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends linux-perf \\
    && rm -rf /var/lib/apt/lists/*
"""


def build_sandbox_image_command(*, image: str = DEFAULT_DOCKER_IMAGE) -> list[str]:
    """Return the docker build command for the default sandbox image."""

    return ["docker", "build", "-t", image, "-"]


def sandbox_image_build_plan(*, image: str = DEFAULT_DOCKER_IMAGE) -> str:
    """Render the default sandbox-image build command and Dockerfile."""

    return "\n".join(
        [
            "# C++ perf sandbox image build dry run",
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
        "for (i=1; i<=n; i++) print lines[i] }' \"$1\"; }; "
    )
    script = (
        normalize
        +
        f"timeout {timeout_s}s taskset -c {shlex.quote(cpu)} ./{binary} "
        f"< tests/{index}.in > tests/{index}.actual && "
        f"normalize tests/{index}.out > tests/{index}.expected.norm && "
        f"normalize tests/{index}.actual > tests/{index}.actual.norm && "
        f"diff -u tests/{index}.expected.norm tests/{index}.actual.norm"
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


def perf_preflight_command(
    scratch: str | Path,
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    timeout_s: int = DEFAULT_RUN_TIMEOUT_S,
) -> list[str]:
    script = (
        "cat > preflight.cpp <<'CPP'\n"
        "#include <iostream>\n"
        "int main(){volatile unsigned long long s=0; "
        "for(int i=0;i<100000;i++) s+=i; std::cout<<s<<\"\\n\"; return 0;}\n"
        "CPP\n"
        "g++ -O2 -std=c++20 preflight.cpp -o preflight && "
        f"timeout {timeout_s}s taskset -c {shlex.quote(cpu)} "
        "perf stat -e instructions:u -x, ./preflight > /tmp/w8-perf-preflight.out"
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


def perf_preflight_plan(
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    scratch: str = "/tmp/w8-cpp-preflight",
) -> str:
    """Render the command used to verify `instructions:u` in the sandbox."""

    return "\n".join(
        [
            "# C++ perf-counter preflight dry run",
            "# succeeds only when perf reports a numeric instructions:u count",
            shlex.join(perf_preflight_command(scratch, image=image, cpu=cpu)),
        ]
    )


def run_perf_preflight(
    *,
    image: str = DEFAULT_DOCKER_IMAGE,
    cpu: str = DEFAULT_CPU,
    work_dir: str | Path | None = None,
) -> PerfPreflightResult:
    """Check that `perf stat -e instructions:u` returns a numeric count in Docker."""

    if work_dir is None:
        with TemporaryDirectory(prefix="w8-cpp-preflight-") as temp:
            return _run_perf_preflight_in_directory(Path(temp), image=image, cpu=cpu)
    return _run_perf_preflight_in_directory(Path(work_dir), image=image, cpu=cpu)


def _run_perf_preflight_in_directory(scratch: Path, *, image: str, cpu: str) -> PerfPreflightResult:
    _prepare_scratch(scratch)
    command = perf_preflight_command(scratch, image=image, cpu=cpu)
    try:
        proc = _run(command)
    except OSError as exc:
        return PerfPreflightResult(
            ok=False,
            instr_count=None,
            returncode=None,
            logs=str(exc),
            reason="command_error",
            command=command,
        )
    logs = proc.stderr or proc.stdout
    instr_count = parse_perf_instructions(logs)
    if proc.returncode != 0:
        return PerfPreflightResult(
            ok=False,
            instr_count=instr_count,
            returncode=proc.returncode,
            logs=logs,
            reason="perf_command_failed",
            command=command,
        )
    if instr_count is None:
        return PerfPreflightResult(
            ok=False,
            instr_count=None,
            returncode=proc.returncode,
            logs=logs,
            reason="missing_instruction_count",
            command=command,
        )
    return PerfPreflightResult(
        ok=True,
        instr_count=instr_count,
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


def _run_in_directory(task: CppTask, candidate_code: str, scratch: Path, *, image: str, cpu: str) -> HarnessResult:
    _prepare_scratch(scratch)
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
    if instr_count is None:
        logs["perf"] = perf_proc.stderr or perf_proc.stdout
    return HarnessResult(
        tests_passed=tests_passed,
        tests_total=len(tests),
        instr_count=instr_count,
        logs=logs,
    )


def parse_perf_instructions(text: str) -> int | None:
    """Parse `perf stat -x, -e instructions:u` output."""

    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[2].startswith("instructions"):
            raw = parts[0].replace(",", "")
            try:
                return int(float(raw))
            except ValueError:
                return None
    return None


def _write_tests(scratch: Path, tests: list[TestCase]) -> None:
    tests_dir = scratch / "tests"
    tests_dir.mkdir(exist_ok=True)
    for index, test in enumerate(tests):
        (tests_dir / f"{index}.in").write_text(test.input, encoding="utf-8")
        (tests_dir / f"{index}.out").write_text(test.expected, encoding="utf-8")


def _prepare_scratch(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    scratch.chmod(0o777)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=False, capture_output=True, text=True)
