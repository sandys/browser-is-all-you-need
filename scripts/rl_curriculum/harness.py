"""Compile/link/run primitive for the circular-state RL curriculum.

Flags are copied verbatim from the polyglot verifier
(src/glm47_posttraining/aider_polyglot/harness.py):

    -std=c++17 -Wall -Wextra -Werror -pedantic -pthread -I.

Compile and link run as separate steps, matching the benchmark's CMake build and
keeping randomly suffixed driver temporaries out of diagnostics. The compiler is
a parameter (REPAIR_CXX) so the same manifests can be re-verified under the
GCC 13.3 toolchain used inside the training image.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
TASKS = ROOT / "tasks"
REPO = ROOT.parent.parent
VENDOR = REPO / "data" / "rl-curriculum" / "vendor"

FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-pthread", "-I."]
CXX = os.environ.get("REPAIR_CXX", "c++")
COMPILE_TIMEOUT_S = 120
TEST_TIMEOUT_S = 60

PASS = "pass"
COMPILE = "compile"
LINK = "link-or-odr"
RUNTIME = "runtime-or-sanitizer"
SEMANTIC = "semantic-counterexample"

# Every non-pass outcome is an oracle rejection.
REJECTIONS = (COMPILE, LINK, RUNTIME, SEMANTIC)


def compiler_id() -> str:
    out = subprocess.run([CXX, "--version"], capture_output=True, text=True).stdout
    return out.splitlines()[0].strip() if out else CXX


def catch_main_object() -> pathlib.Path:
    header = VENDOR / "catch.hpp"
    if not header.exists():
        raise FileNotFoundError(
            f"missing {header}\n"
            "Copy Catch2 v2.13.6 from any polyglot exercise test/ directory."
        )
    obj = VENDOR / "catch_main.o"
    if obj.exists():
        return obj
    main_cpp = VENDOR / "catch_main.cpp"
    main_cpp.write_text('#define CATCH_CONFIG_MAIN\n#include "catch.hpp"\n')
    subprocess.run(
        [CXX, "-std=c++17", "-pthread", f"-I{VENDOR}", "-c", str(main_cpp), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    return obj


def load_tasks() -> dict:
    return json.loads((TASKS / "tasks.json").read_text())


def reference_files(task: str) -> dict[str, str]:
    meta = load_tasks()[task]
    ref = TASKS / task / "ref"
    return {name: (ref / name).read_text() for name in meta["files"]}


def evaluate(task: str, files: dict[str, str]) -> tuple[str, str]:
    """Run the task's hidden Catch2 oracle against `files`."""
    meta = load_tasks()[task]
    obj = catch_main_object()
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"rlc-{task}-"))
    try:
        for name, body in files.items():
            (work / name).write_text(body)
        shutil.copy(VENDOR / "catch.hpp", work / "catch.hpp")
        test_name = meta["test"]
        shutil.copy(TASKS / task / "test" / test_name, work / test_name)

        units = [test_name] + [f for f in files if f.endswith(".cpp")]
        objects = []
        for unit in units:
            proc = subprocess.run(
                [CXX, *FLAGS, "-c", unit, "-o", unit[: -len(".cpp")] + ".o"],
                cwd=work, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
            )
            if proc.returncode != 0:
                return COMPILE, _normalize((proc.stdout + proc.stderr).strip(), work, task)
            objects.append(unit[: -len(".cpp")] + ".o")

        shutil.copy(obj, work / "catch_main.o")
        link = subprocess.run(
            [CXX, *FLAGS, *objects, "catch_main.o", "-o", "runner"],
            cwd=work, capture_output=True, text=True, timeout=COMPILE_TIMEOUT_S,
        )
        if link.returncode != 0:
            return LINK, _normalize((link.stdout + link.stderr).strip(), work, task)

        run = subprocess.run(
            ["./runner"], cwd=work, capture_output=True, text=True, timeout=TEST_TIMEOUT_S
        )
        text = _normalize((run.stdout + run.stderr).strip(), work, task)
        if run.returncode == 0:
            return PASS, text
        if run.returncode < 0 or run.returncode > 128:
            return RUNTIME, text
        return SEMANTIC, text
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _normalize(text: str, work: pathlib.Path, task: str) -> str:
    """Canonicalise the scratch directory so diagnostics are run-stable."""
    root = f"/aider/{task}"
    # Longest first: on macOS the resolved path is /private/var/folders/... while
    # str(work) is /var/folders/..., so replacing the short form first would match
    # the tail of the long one and leave a /private/aider/... stub behind.
    for variant in sorted({str(work.resolve()), str(work)}, key=len, reverse=True):
        text = text.replace(variant, root)
    return text
