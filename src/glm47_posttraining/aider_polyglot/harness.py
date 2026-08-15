"""Sandboxed build-and-test harness for Aider Polyglot C++ exercises."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import secrets
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping

from glm47_posttraining.cpp_perf.sandbox import (
    SandboxInfrastructureError,
    docker_base_args,
    sandbox_backend,
)

from .schema import AiderTestResult


DEFAULT_AIDER_DOCKER_IMAGE = "glm47-aider-polyglot-cpp:latest"
DEFAULT_CONFIGURE_TIMEOUT_S = 30
DEFAULT_BUILD_TIMEOUT_S = 120
DEFAULT_TEST_TIMEOUT_S = 30
MAX_CANDIDATE_FILE_BYTES = 256 * 1024
TSAN_PREFLIGHT_REQUIRED_ENV = "GLM47_CPP_TSAN_PREFLIGHT_REQUIRED"
PASS_RE = re.compile(
    r"All tests passed \(\s*\d+ assertions? in\s*(\d+) test cases?\)", re.IGNORECASE
)
FAIL_RE = re.compile(
    r"test cases:\s*(\d+)\s*\|\s*(\d+) passed\s*\|\s*(\d+) failed",
    re.IGNORECASE,
)
ORDINAL_RETURN_RE = re.compile(r"return\s+(\d+)\s*;")
MIN_ORDINAL_CHECKS = 4
INFRASTRUCTURE_MARKERS = (
    "cannot connect to the docker daemon",
    "error response from daemon",
    "failed to create shim task",
    "no such image",
)
FORBIDDEN_CANDIDATE_PATTERNS = (
    re.compile(r"#\s*(?:define|undef)\s+(?:main|return|if|for|while|switch)\b"),
    re.compile(r"\b(?:std::)?(?:_Exit|_exit|exit|quick_exit|abort|terminate)\s*\("),
    re.compile(r"\b(?:system|popen|fork|vfork|exec[a-z]*|kill|raise)\s*\("),
    re.compile(r"\b(?:__asm__|__asm|asm)\b"),
    re.compile(r"(?:/proc/self|\.grader|CMakeLists\.txt|_test\.cpp)"),
)


class CandidatePolicyError(ValueError):
    """Generated source attempts to bypass or inspect the hidden verifier."""


def _shadow_ordinal_total(grader_source: str) -> int | None:
    """Return N when the grader short-circuits with clean sequential ordinals 1..N.

    Most shadow graders are a single ``main()`` of ``if (!check) return k;`` lines
    with a distinct 1-based ``k`` per check, so the renamed grader's return value
    (surfaced as the candidate process exit code) is the index of the first failing
    check. That lets us score partial progress without running checks against known-
    bad state, which would risk crashes that destroy the tally. Graders that do not
    follow this convention (abort-based ``assert``, constant ``return 1``) yield
    ``None`` and fall back to binary pass/fail.
    """
    values = [int(match) for match in ORDINAL_RETURN_RE.findall(grader_source)]
    values = [value for value in values if value != 0]
    if len(values) >= MIN_ORDINAL_CHECKS and values == list(range(1, len(values) + 1)):
        return len(values)
    return None


def aider_sandbox_image_dockerfile() -> str:
    return """FROM gcc:13
RUN apt-get update \\
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
      cmake make util-linux \\
    && rm -rf /var/lib/apt/lists/*
"""


def build_aider_sandbox_image(
    *, image: str = DEFAULT_AIDER_DOCKER_IMAGE
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "build", "-t", image, "-"],
        input=aider_sandbox_image_dockerfile(),
        check=False,
        capture_output=True,
        text=True,
    )


def run_aider_tests(
    exercise_dir: str | Path,
    files: Mapping[str, str],
    *,
    image: str = DEFAULT_AIDER_DOCKER_IMAGE,
    configure_timeout_s: int = DEFAULT_CONFIGURE_TIMEOUT_S,
    build_timeout_s: int = DEFAULT_BUILD_TIMEOUT_S,
) -> AiderTestResult:
    """Apply candidate files and run the benchmark's build-triggered Catch suite."""

    source = Path(exercise_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"exercise directory not found: {source}")

    with TemporaryDirectory(prefix=f"aider_{source.name}_") as scratch_value:
        scratch = Path(scratch_value)
        shutil.copytree(source, scratch, dirs_exist_ok=True)
        for name, contents in files.items():
            if len(contents.encode("utf-8")) > MAX_CANDIDATE_FILE_BYTES:
                raise ValueError(f"candidate file exceeds byte limit: {name}")
            target = scratch / name
            if target.parent != scratch:
                raise ValueError(f"candidate path escapes exercise root: {name}")
            target.write_text(contents, encoding="utf-8")

        configure = _run_stage(
            scratch,
            f"timeout {configure_timeout_s}s cmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=ON",
            image=image,
            timeout_s=configure_timeout_s + 10,
        )
        configure_logs = _combined_logs(configure)
        if _is_infrastructure_error(configure_logs):
            raise SandboxInfrastructureError(configure_logs)
        if configure.returncode != 0:
            return AiderTestResult(status="compile_failed", logs={"configure": configure_logs})

        build = _run_stage(
            scratch,
            f"timeout {build_timeout_s}s cmake --build build --parallel 2",
            image=image,
            timeout_s=build_timeout_s + 10,
        )
        build_logs = _combined_logs(build)
        logs = {"configure": configure_logs, "build_and_test": build_logs}
        if _is_infrastructure_error(build_logs):
            raise SandboxInfrastructureError(build_logs)

        passed = PASS_RE.search(build_logs)
        if passed:
            total = int(passed.group(1))
            return AiderTestResult(
                status="passed", tests_passed=total, tests_total=total, logs=logs
            )

        failed = FAIL_RE.search(build_logs)
        if failed:
            total, passed_count, _failed_count = (int(value) for value in failed.groups())
            return AiderTestResult(
                status="tests_failed",
                tests_passed=passed_count,
                tests_total=total,
                logs=logs,
            )

        if build.returncode == 124 or "timed out" in build_logs.lower():
            return AiderTestResult(status="candidate_timeout", logs=logs)
        return AiderTestResult(status="compile_failed", logs=logs)


def run_shadow_tests(
    exercise_dir: str | Path,
    files: Mapping[str, str],
    *,
    image: str = DEFAULT_AIDER_DOCKER_IMAGE,
    build_timeout_s: int = DEFAULT_BUILD_TIMEOUT_S,
    test_timeout_s: int = DEFAULT_TEST_TIMEOUT_S,
    expected_test_sha256: str | None = None,
) -> AiderTestResult:
    """Compile candidate sources against an answer-blind C++17 executable oracle."""

    source = Path(exercise_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"shadow task directory not found: {source}")
    grader_path = source / ".grader" / "test.cpp"
    if not grader_path.is_file():
        raise FileNotFoundError(f"shadow executable oracle not found: {source}")
    grader_bytes = grader_path.read_bytes()
    if expected_test_sha256:
        observed = hashlib.sha256(grader_bytes).hexdigest()
        if observed != expected_test_sha256:
            raise ValueError(f"shadow executable oracle hash mismatch: {source}")
    ordinal_total = _shadow_ordinal_total(grader_bytes.decode("utf-8", errors="replace"))

    with TemporaryDirectory(prefix=f"aider_shadow_{source.name}_") as scratch_value:
        scratch = Path(scratch_value)
        shutil.copytree(source, scratch, dirs_exist_ok=True)
        for name, contents in files.items():
            if len(contents.encode("utf-8")) > MAX_CANDIDATE_FILE_BYTES:
                raise ValueError(f"candidate file exceeds byte limit: {name}")
            _validate_candidate_source(name, contents)
            target = scratch / name
            if target.parent != scratch:
                raise ValueError(f"candidate path escapes shadow task root: {name}")
            target.write_text(contents, encoding="utf-8")

        sources = sorted(
            path.name for path in scratch.iterdir() if path.suffix in {".cpp", ".cc"}
        )
        quoted_sources = " ".join(shlex_quote(name) for name in sources)
        success_marker = f"GLM47_AIDER_PASS_{secrets.token_hex(16)}"
        driver = scratch / ".grader" / "driver.cpp"
        driver.write_text(
            "#include <cstdio>\n"
            "int glm47_hidden_main();\n"
            "int main() {\n"
            "  const int result = glm47_hidden_main();\n"
            f'  if (result == 0) {{ std::fputs("{success_marker}\\n", stdout); '
            "std::fflush(stdout); }\n"
            "  return result;\n"
            "}\n",
            encoding="utf-8",
        )
        compiler_flags = "-std=c++17 -Wall -Wextra -Werror -pedantic -pthread -I."
        compile_result = _run_stage(
            scratch,
            " ".join(
                [
                    f"timeout {build_timeout_s}s c++",
                    compiler_flags,
                    "-Dmain=glm47_hidden_main -c .grader/test.cpp -o .grader/test.o",
                    f"&& rm .grader/test.cpp && timeout {build_timeout_s}s c++",
                    compiler_flags,
                    quoted_sources,
                    ".grader/driver.cpp .grader/test.o -o .grader/candidate_test",
                ]
            ),
            image=image,
            timeout_s=build_timeout_s + 10,
        )
        compile_logs = _combined_logs(compile_result)
        if _is_infrastructure_error(compile_logs):
            raise SandboxInfrastructureError(compile_logs)
        if compile_result.returncode == 124 or "timed out" in compile_logs.lower():
            return AiderTestResult(
                status="candidate_timeout",
                candidate_returncode=compile_result.returncode,
                logs={"compile": compile_logs},
            )
        if compile_result.returncode != 0:
            return AiderTestResult(
                status="compile_failed",
                candidate_returncode=compile_result.returncode,
                logs={"compile": compile_logs},
            )
        # The candidate process sees neither hidden test source nor its linkable object.
        (scratch / ".grader" / "test.o").unlink(missing_ok=True)
        driver.unlink(missing_ok=True)

        test_result = _run_stage(
            scratch,
            f"timeout {test_timeout_s}s .grader/candidate_test",
            image=image,
            timeout_s=test_timeout_s + 10,
        )
        test_logs = _combined_logs(test_result)
        logs = {"compile": compile_logs, "test": test_logs}
        if _is_infrastructure_error(test_logs):
            raise SandboxInfrastructureError(test_logs)
        total = ordinal_total or 1
        if test_result.returncode == 124 or "timed out" in test_logs.lower():
            return AiderTestResult(
                status="candidate_timeout",
                tests_total=total,
                candidate_returncode=test_result.returncode,
                logs=logs,
            )
        if test_result.returncode == 0 and success_marker in test_logs:
            return AiderTestResult(
                status="passed",
                tests_passed=total,
                tests_total=total,
                candidate_returncode=0,
                logs=logs,
            )
        # For sequential-ordinal graders the exit code is the 1-based index of the
        # first failing check, so (returncode - 1) checks passed before it. Anything
        # outside [1, N] (a crash signal, or a non-ordinal grader) scores zero.
        passed_checks = 0
        if ordinal_total is not None and 1 <= test_result.returncode <= ordinal_total:
            passed_checks = test_result.returncode - 1
        return AiderTestResult(
            status="tests_failed",
            tests_passed=passed_checks,
            tests_total=total,
            candidate_returncode=test_result.returncode,
            logs=logs,
        )


def _validate_candidate_source(name: str, contents: str) -> None:
    for pattern in FORBIDDEN_CANDIDATE_PATTERNS:
        if pattern.search(contents):
            raise CandidatePolicyError(
                f"candidate file uses a forbidden verifier-bypass primitive: {name}"
            )


def _run_stage(
    scratch: Path, script: str, *, image: str, timeout_s: int
) -> subprocess.CompletedProcess[str]:
    if sandbox_backend() == "local":
        command = _local_sandbox_command(scratch, script)
    else:
        docker_args = docker_base_args(scratch, image=image, memory="4g")
        # TSan reserves a fixed shadow-memory range. High-entropy ASLR can place
        # the executable inside that range, while Docker's default seccomp profile
        # blocks the personality syscall TSan uses to recover. Scope the proven
        # workaround to executing an already-built TSan binary: compilation and
        # every non-TSan sandbox stage retain the default seccomp profile and ASLR.
        tsan_execution = (
            "candidate_test_tsan" in script
            and "-o .grader/candidate_test_tsan" not in script
        ) or ("probe_tsan" in script and "c++" not in script)
        if tsan_execution:
            docker_args[-1:-1] = ["--security-opt", "seccomp=unconfined"]
            script = f"setarch x86_64 -R env {script}"
        command = docker_args + ["bash", "-lc", script]
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _text(exc.stdout)
        stderr = _text(exc.stderr)
        return subprocess.CompletedProcess(command, 124, stdout=stdout, stderr=stderr)


def _local_sandbox_command(scratch: Path, script: str) -> list[str]:
    """Use bubblewrap on Linux and fail closed if it is unavailable.

    Modal cannot run Docker-in-Docker. Bubblewrap gives generated programs a
    private mount, PID, IPC, UTS, and (by default) network namespace and
    deliberately does not mount the repository, run volume, or inherited
    environment secrets. GLM47_CPP_SANDBOX_UNSHARE_NET=0 skips only the
    network unshare: gVisor-style runtimes (Modal) reject the RTM_NEWADDR
    loopback setup bwrap performs after unsharing the network namespace.
    """

    if platform.system() != "Linux":
        # Development-only path for macOS unit tests. Paid Linux runs never use it.
        local_script = re.sub(r"\btimeout\s+\d+s\s+", "", script)
        return [
            "bash",
            "-lc",
            f"cd {shlex_quote(str(scratch.resolve()))} && ulimit -c 0 && {local_script}",
        ]

    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise SandboxInfrastructureError(
            "bubblewrap is required for GLM47_CPP_SANDBOX_BACKEND=local on Linux"
        )
    if os.environ.get("GLM47_CPP_SANDBOX_UNSHARE_NET", "1") != "0":
        unshare_flags = ["--unshare-all"]
    else:
        unshare_flags = [
            "--unshare-user-try",
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-cgroup-try",
        ]
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        *unshare_flags,
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for host_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        if Path(host_path).exists():
            command.extend(["--ro-bind", host_path, host_path])
    command.extend(
        [
            "--bind",
            str(scratch.resolve()),
            "/work",
            "--chdir",
            "/work",
            "/bin/bash",
            "-lc",
            "ulimit -c 0 -f 262144 -n 64 -u 128; umask 077; " + script,
        ]
    )
    return command


def assert_local_sandbox_ready() -> None:
    """Fail before allocating a training run if secure local isolation is absent."""

    if sandbox_backend() == "local" and platform.system() == "Linux" and not shutil.which("bwrap"):
        raise SandboxInfrastructureError("bubblewrap is required for secure Aider reward execution")


def run_sandbox_preflight() -> None:
    """Compile and execute harmless probes through the selected isolation path."""

    assert_local_sandbox_ready()
    with TemporaryDirectory(prefix="aider_sandbox_preflight_") as scratch_value:
        scratch = Path(scratch_value)
        (scratch / "probe.cpp").write_text(
            "#include <thread>\n"
            "int main() { int value = 0; std::thread worker([&] { value = 1; }); "
            "worker.join(); return value == 1 ? 0 : 1; }\n",
            encoding="utf-8",
        )
        require_tsan = os.environ.get(TSAN_PREFLIGHT_REQUIRED_ENV, "0") == "1"
        compile_tsan = (
            " && c++ -std=c++17 -Wall -Wextra -Werror -pedantic -pthread "
            "-fsanitize=thread probe.cpp -o probe_tsan"
            if require_tsan
            else ""
        )
        result = _run_stage(
            scratch,
            "c++ -std=c++17 -Wall -Wextra -Werror -pedantic -pthread probe.cpp -o probe "
            "&& ./probe" + compile_tsan,
            image=DEFAULT_AIDER_DOCKER_IMAGE,
            timeout_s=90,
        )
        if result.returncode != 0:
            raise SandboxInfrastructureError(_combined_logs(result) or "sandbox preflight failed")
        if not require_tsan:
            return
        tsan_result = _run_stage(
            scratch,
            "TSAN_OPTIONS=halt_on_error=1 ./probe_tsan",
            image=DEFAULT_AIDER_DOCKER_IMAGE,
            timeout_s=90,
        )
        if tsan_result.returncode != 0:
            raise SandboxInfrastructureError(
                _combined_logs(tsan_result) or "TSan sandbox preflight failed"
            )


def shlex_quote(value: str) -> str:
    # Kept local so the candidate command construction has one tiny, auditable surface.
    import shlex

    return shlex.quote(value)


def _combined_logs(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stdout or "") + (result.stderr or "")


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _is_infrastructure_error(logs: str) -> bool:
    lowered = logs.lower()
    return any(marker in lowered for marker in INFRASTRUCTURE_MARKERS)
