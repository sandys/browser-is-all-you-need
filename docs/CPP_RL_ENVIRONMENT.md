# C++ Rubric RL Environment

## Scope

This document describes the software environment presented to a policy and the
verification environment used to score its response. External task corpora are
not bundled with the repository.

## 1. Task contract

A serialized task contains:

- stable task identifier, split, category, and difficulty metadata;
- natural-language instructions;
- editable C++ header/source files containing starter content;
- privileged oracle files when available;
- privileged Catch2 tests and vendored test support files;
- build/link flags;
- analysis metadata, rubric category, and rubric risks.

All file paths must be relative, cannot contain `..`, and are validated before
use. At least one editable file and one test file are required.

## 2. Observation boundary

The model receives:

- task identifier and instructions;
- complete contents of every editable file;
- the response-format contract.

The model does not receive:

- test source;
- test support files;
- oracle implementations;
- sandbox logs from other candidates;
- evaluation-only metadata unless explicitly placed in the prompt.

Tests and oracle material may exist in the task JSON available to the reward
worker. They are withheld from model context, not cryptographically separated
from the scorer process.

## 3. Action contract

The policy returns one reasoning block followed by one C++ fence. Conceptually:

    <reasoning>...</reasoning>
    ```cpp
    // ===== FILE: relative/path/to/file.hpp =====
    ...complete file...

    // ===== FILE: relative/path/to/file.cpp =====
    ...complete file...
    ```

There must be exactly one reasoning block and one fenced C++ block. Every
expected editable file must appear once; missing, duplicate, or additional
files are rejected. Tests and build files cannot be edited through this action.

## 4. Episode sequence

```text
task load
  -> category/template selection
  -> prompt construction
  -> model response
  -> context/format validation
  -> isolated workspace creation
  -> normal compilation and Catch2 execution
  -> ASan/UBSan compilation and execution
  -> optional TSan compilation and execution
  -> optional trusted runtime evidence
  -> rubric scoring
  -> task-risk normalization
  -> weighted scalar reward
  -> structured record and cleanup
```

One response produces one terminal reward record. Retry prompts are a separate,
explicit multi-attempt mode.

## 5. Workspace construction

For each candidate, the scorer creates a fresh temporary directory and writes:

- only the candidate versions of editable files;
- privileged test and support files from the task record;
- generated binaries and XML reports.

Candidate paths are validated against the exact expected file set before any
compilation. Temporary workspaces are deleted when scoring completes. Logs are
truncated before being attached to reward records.

## 6. Build profiles

The task's own vendored Catch2 entry point is used, avoiding a host Catch2
version dependency. All configured tests are enabled.

### Normal verification

```text
g++ -O2 -DNDEBUG -std=c++17 -DEXERCISM_RUN_ALL_TESTS
    -Wall -Wextra -Wpedantic -Werror ... -pthread
```

### Memory/undefined-behavior verification

```text
g++ -O1 -g -std=c++17 -DEXERCISM_RUN_ALL_TESTS
    -Wall -Wextra -Wpedantic
    -fsanitize=address,undefined -fno-omit-frame-pointer ... -pthread
```

The instrumented binary is executed with sanitizer halt-on-error settings.
Leak detection remains enabled in the Docker backend. It may be disabled in the
local smoke backend when the enclosing container prevents leak-sanitizer
attachment.

### Thread verification

For `state_concurrency` tasks that first pass normal and memory-safety checks:

```text
g++ -O1 -g -std=c++17 -DEXERCISM_RUN_ALL_TESTS
    -Wall -Wextra -Wpedantic
    -fsanitize=thread -fno-omit-frame-pointer ... -pthread
```

The TSan binary is executed separately. Compilation failure is recorded as
missing TSan evidence, while an executed diagnostic or timeout is scored by the
thread-safety rubric. Production preflight fails closed if a state task exists
but clean TSan execution cannot be demonstrated.

## 7. Catch2 result handling

Catch2 XML is parsed for:

- passed and total test cases;
- passed and total assertions;
- aggregate Catch2 v2 results;
- compatible per-test result nodes used by synthetic or alternate reports.

Empty or malformed XML yields zero totals rather than an assumed pass. Full
correctness requires positive case and assertion totals with exact equality
between passed and total counts.

## 8. Isolation backends

### Docker backend

The Docker backend is the security boundary for untrusted generated code. Each
stage uses:

- no network;
- a read-only container root;
- a single writable mounted scratch directory;
- bounded CPU, memory, and process count;
- all Linux capabilities dropped;
- `no-new-privileges`;
- a small `noexec,nosuid` temporary filesystem;
- automatic container removal;
- the caller's numeric user/group instead of root.

Compile and execution stages have explicit timeouts.

### Local backend

The local backend executes the same commands and timeout semantics in the
current process environment. It does not provide Docker's filesystem, network,
capability, memory, or process isolation and must only be used inside an already
trusted isolation boundary. It is intended for smoke tests, not scoring
untrusted public submissions.

## 9. Runtime rubric boundary

Runtime is active only for performance-intensive tasks. The reward consumes
`runtime_cpu_ns` and `reference_runtime_cpu_ns` only when both were produced by
a trusted, controlled runner. The generic environment does not substitute
container launch time or wall-clock latency. Projects enabling runtime reward
should pin their benchmark protocol and preserve raw measurements separately.

## 10. Failure and terminal reasons

The environment records distinct reasons including:

- invalid format or invalid multi-file edit;
- context exhaustion;
- normal or sanitizer compilation failure;
- ordinary execution timeout;
- partial Catch2 failure;
- ASan/UBSan failure;
- TSan race or timeout;
- fully correct but TSan not executed;
- all active verification rubrics passed;
- reward infrastructure exception.

Infrastructure exceptions are never converted into successful verification.

## 11. Preflight

The preflight validates schema, prompt secrecy, format parsing, deterministic
reward arithmetic, a real ordinary oracle, and—when state tasks exist—a real
TSan oracle.

```bash
PYTHONPATH=src python3 scripts/check_aider_runtime.py \
  --tasks-dir path/to/private/tasks \
  --backend docker
```

The local smoke option may skip TSan explicitly, but that result is not a
production concurrency verification.

## 12. Dataset portability

The environment is benchmark-independent after adaptation. A private or
separately downloaded dataset must be converted into the validated task schema
while preserving:

- editable versus privileged file separation;
- complete Catch2 support files;
- deterministic build flags;
- task category and initial risk metadata;
- split boundaries and provenance outside model prompts.

No benchmark tasks need to be committed to use the environment.
