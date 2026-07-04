# SLIME C++ Performance-RL Roadmap

This roadmap describes the current training process from first principles. The
active stack is SLIME with Megatron training and SGLang rollouts, focused on
Moonlight and GLM models.

The project trains a model to rewrite C++ programs:

```text
correct but slower C++ program -> correct and faster C++ program
```

A faster wrong program is a failure. The reward must first prove behavior is
preserved, then measure speed.

## Names Used Here

- `w8-biayn`: this repository's CLI, usually run as `uv run w8-biayn ...`.
- PIE: the slower-to-faster C++ source dataset.
- SLIME: the active RL framework for SFT/GRPO experiments in this repo.
- Megatron: the training backend used by the SLIME lanes.
- SGLang: the rollout/inference backend used by the SLIME lanes.
- Moonlight: the active Moonlight-16B-A3B model lane.
- GLM: the active GLM-4.7-Flash model lane.
- Docker sandbox: the compile/test/runtime harness for C++ reward execution.

SkyRL/rLLM names may still appear in legacy files. Treat that stack as retired
for active training unless a task explicitly asks for compatibility work.

## Core Learning Problem

The model is not trained to solve arbitrary programming tasks from scratch. It
receives a complete C++ program that already solves the problem and must return
a complete C++ program that still solves the problem faster.

That shape changes the work:

- data must preserve slower/faster program pairs;
- prompts must show `v0` and may show visible tests;
- GRPO prompts must not show `v1` or hidden tests;
- reward must compile, sanitize, test, and benchmark generated code.

## What One Task Contains

Each admitted PIE task contains:

- `v0`: slower accepted C++ program, visible in the prompt;
- `v1`: faster accepted C++ program, used for SFT/reference only;
- visible tests: prompt-visible behavior examples;
- hidden tests: grading-only behavior checks;
- reference performance: timing material for speed comparison;
- split: train, validation, or test, split by problem.

The prompt may include `v0` and visible tests. It must not include `v1` or
hidden tests during GRPO.

## Why SFT Still Matters

SFT uses PIE `v0 -> v1` pairs:

```text
prompt: instruction + slower C++ v0
target: <reasoning>...</reasoning> + faster C++ v1 in one fenced cpp block
```

SFT teaches output format, complete compilable C++, and common optimization
patterns. It does not prove held-out optimization. GRPO is still needed to
learn from correctness-gated runtime reward.

## Required Output Format

Every model output must contain exactly one reasoning block followed by exactly
one fenced C++ block:

````text
<reasoning>
Brief optimization rationale.
</reasoning>
```cpp
// complete C++20 program
```
````

The parser may tolerate recoverable bare C++ for shaped reward, but strict
format remains the training target.

## Reward Ladder

Reward should enforce this order:

1. Invalid format is negative.
2. Compile or sanitizer failure is negative.
3. Timeout is negative.
4. Partial test pass remains below any fully correct answer.
5. All visible and hidden tests passing earns positive correctness reward.
6. Fully correct with missing non-timeout runtime gets a correctness-only
   fallback below measured correct outputs.
7. Fully correct and faster earns bounded runtime-efficiency bonus.

The active speed signal is child-process CPU time in nanoseconds. Wall-clock
time is diagnostic. Do not add PMU, Linux perf, PERFMON, or
`perf_event_paranoid` requirements.

## Stage 0: Prove The Local Runtime

Question:

> Can the machine prepare data, run SLIME, and execute the C++ harness?

Commands:

```bash
./scripts/bootstrap.sh
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
uv run w8-biayn slime setup
uv run w8-biayn cpp harness preflight --dry-run
```

Decision gate: do not start paid or long GPU work until SLIME setup and C++
harness preflight are clean.

## Stage 1: Build Admitted PIE Tasks

Question:

> Do we have tasks strong enough to teach and grade optimization?

Commands:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full \
  --source-root .w8-biayn/data/pie \
  --out .w8-biayn/data/pie-full \
  --force

uv run w8-biayn data pie measure-coverage \
  --prepared-root .w8-biayn/data/pie-full \
  --out .w8-biayn/data/pie-full/coverage.json \
  --report-out .w8-biayn/data/pie-full/coverage-report.json

uv run w8-biayn data pie build-full-tasks \
  --prepared-root .w8-biayn/data/pie-full \
  --coverage-json .w8-biayn/data/pie-full/coverage.json \
  --out .w8-biayn/data/tasks-full \
  --min-train 1000 \
  --min-validation 100 \
  --min-test 100 \
  --force
```

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- line coverage >= 95 percent;
- branch coverage >= 85 percent;
- visible and hidden tests exist;
- reference performance exists.

## Stage 2: Build SLIME JSONL

Question:

> Can admitted C++ tasks become SLIME SFT, GRPO, and eval rows?

Moonlight lane:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
```

Moonlight rank-16 LoRA lane:

```bash
bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh
```

GLM lane when present:

```bash
bash examples/slime/glm47_cpp_perf/prepare_data.sh
```

The underlying bridge is `w8_biayn.integrations.slime_cpp_perf`. It writes
SLIME JSONL and task copies while reusing the same prompt and reward contract.

Decision gate: inspect the generated manifest and sample rows before training.

## Stage 3: Baseline The Base Model

Question:

> What can the base Moonlight or GLM checkpoint already do?

Moonlight:

```bash
bash examples/slime/moonlight_cpp_perf/eval_base.sh
```

GLM:

```bash
bash examples/slime/glm47_cpp_perf/eval_base.sh
```

Inspect invalid-format rate, compile failures, hidden-test pass rate,
correct-and-faster rate, and missing runtime rows.

Decision gate: do not claim training progress until baseline evaluation has
complete records and no unexplained missing runtime rows.

## Stage 4: Run SFT

Question:

> Can supervised learning improve format, compilability, and basic rewrite
> quality?

Moonlight:

```bash
bash examples/slime/moonlight_cpp_perf/sft.sh
bash examples/slime/moonlight_cpp_perf/eval_sft.sh
```

GLM:

```bash
bash examples/slime/glm47_cpp_perf/sft.sh
bash examples/slime/glm47_cpp_perf/eval_sft.sh
```

Expected improvements:

- higher strict-format rate;
- fewer compile failures;
- more fully correct outputs;
- more useful starting point for GRPO.

Decision gate: verify the SFT checkpoint/export used for GRPO is complete and
loadable by the lane's SGLang path. The LoRA lane checks the active
SLIME/Megatron help surface before SFT/GRPO so rank-16 LoRA arguments cannot be
silently ignored.

## Stage 5: Run GRPO From SFT

Question:

> Can reward optimization beat base and SFT on correctness-gated speed?

Moonlight:

```bash
bash examples/slime/moonlight_cpp_perf/grpo.sh
bash examples/slime/moonlight_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_cpp_perf/compare.sh
```

Moonlight rank-16 LoRA variant:

```bash
bash examples/slime/moonlight_lora_cpp_perf/grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/compare.sh
```

GLM:

```bash
bash examples/slime/glm47_cpp_perf/grpo.sh
bash examples/slime/glm47_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_cpp_perf/compare.sh
```

Watch for:

- reward rising while held-out eval stalls;
- response-length drift;
- invalid-format regressions;
- hidden-test regressions;
- C++ reward workers bottlenecking GPU training;
- SGLang startup or memory failures.

Decision gate: a useful result beats base and SFT on held-out
`correct_and_faster_rate` and mean best reward with missing-runtime rate at
zero. For the paid GCP GLM path,
`examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py` provisions one
`H100:8` SkyPilot cluster, runs the full GLM sequence, downloads artifacts to
`.w8-biayn/slime/glm47-cpp-perf/`, and calls `sky.down` after completion.

## Stage 6: Compare Moonlight And GLM

Question:

> Which active model lane is the better next investment?

Compare the lane-local `comparison.json` files and run receipts:

- base/SFT/GRPO pass rate;
- correct-and-faster rate;
- mean best reward;
- runtime speedup among correct faster outputs;
- missing-runtime rows;
- GPU memory peak;
- rollout/training wall time;
- reward throughput.

Do not answer "is GLM faster?" or "is Moonlight better?" from model size or GPU
count alone. Use comparable runs on the same task split and reward harness.

## Stage 7: Archive Evidence Without Tracking Artifacts

Run receipts, debug rollouts, checkpoint exports, W&B links, raw reports, SVGs,
and CSVs are evidence artifacts. Keep them under `.w8-biayn/` or another
ignored artifact directory. Do not commit generated `RUN_REPORT*` files or
report asset directories.

When a result is worth preserving in git, write a concise markdown summary that
links to durable external artifacts without vendoring large/generated files.
