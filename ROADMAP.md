# C++ Performance-RL Roadmap

This roadmap teaches the training process from first principles. It assumes the
reader is seeing this repository for the first time.

The project is trying to train a model that rewrites C++ programs:

```text
correct but slower C++ program -> correct and faster C++ program
```

The important word is **correct**. A program that runs faster by changing the
answer is a failure. The training loop must therefore grade two things in this
order:

1. Does the generated C++ preserve behavior?
2. If it preserves behavior, does it run faster?

The active reward path is Docker-based C++ compile, test, and runtime
measurement. It uses child-process CPU time in nanoseconds for RL reward, with
wall-clock time as diagnostics. It does not depend on PMU counters, Linux
`perf`, PERFMON, or `perf_event_paranoid`.

## Names Used In This Roadmap

The roadmap uses a few project-specific names:

- `w8-biayn`: this repository's command-line interface. Commands are written as
  `uv run w8-biayn ...` so they run through the repo environment.
- PIE: the slower-to-faster C++ dataset used to build optimization tasks.
- SkyRL/rLLM: the training stack used for SFT and GRPO. This repo delegates
  training to that stack instead of implementing a custom trainer.
- SkyRL dataset: the converted training data format consumed by SkyRL.
- GCS: Google Cloud Storage, used for dataset caches, checkpoints, exports, and
  run artifacts.
- Docker sandbox: the isolated environment used to compile, test, and time C++
  candidates.

## The Core Learning Problem

The model is not being trained to chat about code, write explanations, or solve
arbitrary programming tasks from scratch. It is being trained to do one narrow
transformation:

- Input: a complete C++ program that already solves the problem.
- Output: a complete C++ program that still solves the problem.
- Goal: make the output faster than the input.

That shape changes how to think about data, reward, and evaluation.

In a normal coding benchmark, the model may receive only a problem statement and
must invent a solution. In this project, the model receives working code. The
question is whether it can recognize waste, preserve the algorithmic intent, and
rewrite the implementation without breaking hidden cases.

## What One Training Task Contains

The task source is PIE, a collection of competitive-programming C++ submissions
where two accepted programs solve the same problem but have different runtime.
This repository treats each useful pair as:

- `v0`: the slower accepted C++ program.
- `v1`: the faster accepted C++ program.
- Visible tests: tests that may be shown in the prompt to clarify behavior.
- Hidden tests: grading tests that the model must not see.
- Reference performance: runtime information used to compare candidate speed.
- Split: train, validation, or test, kept by problem so the model is evaluated
  on problems it did not train on.

Conceptually:

```text
PIE problem
  slower accepted solution: v0
  faster accepted solution: v1
  visible tests
  hidden tests
  coverage and runtime checks
```

The prompt can include `v0` and visible tests. The prompt must not include `v1`
or hidden tests during GRPO.

## Why `v1` Is Allowed In SFT But Not GRPO

This project uses two kinds of training:

- **SFT**, supervised fine-tuning, means imitation learning.
- **GRPO**, reinforcement learning, means sampling candidates and learning from
  reward.

During SFT, the model is supposed to imitate a known good answer. That is why
`v1` is allowed as the supervised target. The model sees examples like:

```text
prompt:  instruction + slower C++ v0
target:  <reasoning>...</reasoning> + faster C++ v1 in a fenced code block
```

During GRPO, the model is supposed to discover which sampled rewrites earn
reward. Showing `v1` during GRPO would leak the answer. It would turn the RL
task into copying the oracle rather than learning from correctness-gated speed.

The rule is:

- SFT may use `v1` as the target.
- GRPO may use `v0` and visible tests in the prompt.
- GRPO must not show `v1` or hidden tests to the model.
- The sandbox may use `v1` internally as oracle/reference material for coverage,
  comparison, and timing, but not as model-visible text.

## Where SFT Data Comes From

The key student question is:

> Where does SFT data come from for this use case?

It comes from PIE slower-to-faster pairs.

For each admitted task:

- The SFT prompt is built from the optimization instruction and `v0`, the slower
  correct C++ program.
- The SFT target is built from `v1`, the faster correct C++ program.
- The target is wrapped in the required response format:
  exactly one `<reasoning>...</reasoning>` block followed by exactly one fenced
  C++ code block.

SFT does not prove that the model can optimize new programs. It teaches the
model the format, the habit of producing complete compilable C++, and common
optimization patterns. That matters because early RL is often dominated by
boring failures: invalid output, broken code fences, missing includes, compile
errors, or rewrites that fail basic tests.

## Why SFT Before GRPO Helps

GRPO only learns from sampled trajectories. If almost every sampled program is
invalid, fails to compile, or fails hidden tests, then reward is mostly negative
and the training signal is weak.

SFT gives RL a better starting point:

- More outputs match the required format.
- More outputs are complete C++ programs.
- More outputs compile.
- More outputs preserve at least some behavior.
- GRPO starts seeing some fully correct trajectories, which makes the speed
  reward meaningful.

This is the cold-start reason for SFT. It does not replace GRPO. It prepares the
model so GRPO can optimize correctness-gated runtime instead of spending all of
its budget learning to stop making malformed answers.

## Required Output Format

Every model output must contain exactly:

1. One `<reasoning>...</reasoning>` block.
2. One fenced C++ code block.

For example:

````text
<reasoning>
Brief explanation of the optimization approach.
</reasoning>
```cpp
// complete C++ program
```
````

Invalid format receives negative reward. The format rule is strict because the
sandbox needs to parse generated code automatically at training scale. The code
may start on the next line after the opening fence or after whitespace on the
same C++ fence line, but there must still be exactly one reasoning block and
exactly one C++ code block.

## The Reward Ladder

The reward should teach the same priorities a human reviewer would use:

1. Invalid response format: negative.
2. Compile or sanitizer failure: negative.
3. Timeout: negative.
4. Some tests pass but not all: below any fully correct answer.
5. All visible and hidden tests pass: positive correctness reward.
6. Fully correct but missing non-timeout runtime measurement: a
   correctness-only fallback below any measured fully correct answer.
7. Fully correct and faster: correctness reward plus bounded speed bonus.

The sandbox compiles the candidate and the PIE `v1` reference, runs visible and
hidden tests, then benchmarks both binaries in the same Docker sandbox with the
same CPU pinning, compiler flags, and tests.

The active speed signal is child-process CPU time in nanoseconds. Wall-clock
time is recorded as a diagnostic because it can reveal noisy machines, slow
Docker execution, or environment bottlenecks.

## The Stage Mindset

The training process is a sequence of experiments. Each stage answers a specific
question. Do not skip straight to the biggest training run; large runs only make
confusion more expensive.

The recommended sequence is:

1. Prove the machine and harness.
2. Build admitted PIE tasks.
3. Baseline the base model.
4. Try GRPO-only and inspect failure modes.
5. Run SFT from PIE `v0 -> v1`.
6. Run GRPO initialized from SFT.
7. Evaluate held-out uplift against base and SFT.
8. Use machine-readable status to teach, debug, visualize, and clean up.

## Stage 0: Shared Run Identity

Use one run identity across data, training, evaluation, status, and cleanup:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
PROJECT_ID="<gcp-project-id>"
DATA_GCS="gs://${PROJECT_ID}-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl"
RUN_GCS="gs://${PROJECT_ID}-w8-biayn/runs/cpp-perf/${RUN_ID}"
```

This makes every artifact traceable:

- Which dataset was used?
- Which model was trained?
- Which checkpoints and exports were produced?
- Which evaluations correspond to the run?
- Which cloud resources should be cleaned up?

## Stage 1: Prove The Machine And Harness

Question:

> Can this machine run the project, access cloud resources, and measure C++
> runtime safely?

Useful commands:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
uv run w8-biayn cpp harness preflight --dry-run
```

What to learn:

- Missing tools should be reported with clear next actions.
- Dry-run launch should render without spending money.
- Credential contents must not be printed.
- The C++ harness must be able to compile, test, and time programs before GRPO
  depends on it.

Decision gate:

- Do not run paid training until doctor checks, dry-run rendering, and harness
  preflight are clean.

## Stage 2: Build Admitted PIE Tasks

Question:

> Do we have tasks that are strong enough to teach and grade optimization?

Raw slower-to-faster pairs are not enough. The repository must convert PIE into
admitted task JSON and then into SkyRL datasets.

Conceptual pipeline:

```text
PIE archives
  -> normalized v0/v1 C++ pairs
  -> coverage-checked task JSON
  -> SFT JSONL and GRPO parquet
  -> verified dataset manifest
  -> GCS dataset cache
```

Useful commands:

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

uv run w8-biayn data skyrl build \
  --tasks-dir .w8-biayn/data/tasks-full \
  --out .w8-biayn/data/skyrl-full \
  --profile full-official \
  --run-id "${RUN_ID}" \
  --min-train-tasks 1000 \
  --min-validation-tasks 100

uv run w8-biayn data cache upload \
  --path .w8-biayn/data/skyrl-full \
  --gcs-prefix "${DATA_GCS}" \
  --credentials .gcp-service-account.json
```

Admission gates:

- At least 1000 train tasks.
- At least 100 validation tasks.
- At least 100 test tasks.
- At least 95 percent line coverage.
- At least 85 percent branch coverage.
- Visible and hidden tests exist.
- Reference performance exists.
- Train, validation, and test split by problem.
- SkyRL bundle manifest verifies before upload.

What to learn:

- Data quality is part of the anti-cheating system.
- Weak hidden tests can make broken programs look fast.
- Low coverage means the reward may not protect behavior.
- A dataset is not ready just because PIE contains `v0/v1` pairs.

Decision gate:

- If task admission barely passes or rejection reasons look surprising, debug
  the data before training.

## Stage 3: Baseline The Base Model

Question:

> What can the untrained or base model already do?

This is the control group. Without it, later claims have no anchor.

Useful command:

```bash
uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --eval-label base \
  --n-samples-per-prompt 1
```

What to inspect:

- Invalid-format rate.
- Compile-failure rate.
- Visible-test pass rate.
- Hidden-test pass rate.
- Correct-and-faster rate.
- Missing-runtime rate.

What to learn:

- If format failures dominate, the output contract is not natural for the base
  model.
- If compile failures dominate, the model is not yet a reliable C++ rewriter.
- If hidden tests fail after visible tests pass, the model may be overfitting to
  examples or the task may need stronger tests.
- If `missing_runtime_rate` is nonzero, evaluation is not trustworthy yet.

Decision gate:

- Do not claim progress until baseline evaluation produces complete records with
  `missing_runtime_rate=0`.

## Stage 4: Try GRPO-Only

Question:

> What happens when the base model meets the real reward loop?

GRPO-only is not expected to be the best final model. It is an experiment that
reveals reward-loop and environment failures before SFT hides them.

Useful commands:

```bash
uv run w8-biayn config render cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}"

uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --n-samples-per-prompt 8 \
  --train-epochs 1 \
  --no-eval-before-train \
  --eval-interval 25 \
  --max-env-workers 128 \
  --ckpt-interval 50 \
  --hf-save-interval 100 \
  --ckpt-path "${RUN_GCS}/cpp-grpo-only/ckpts" \
  --export-path "${RUN_GCS}/cpp-grpo-only/exports" \
  --max-ckpts-to-keep 2 \
  --no-down-after
```

Likely failure modes:

- Invalid format: the parser cannot extract exactly one reasoning block and one
  C++ code block.
- Compile failures: the model edits C++ unsafely.
- Sanitizer failures: generated code has undefined behavior or memory errors.
- Visible tests pass but hidden tests fail: the rewrite does not preserve full
  behavior.
- Correct but not faster: the speed signal is weak or the model has not learned
  useful optimizations.
- Slow reward calls: Docker compile/test/runtime is bottlenecking training.
- Preflight failure: Docker-outside-Docker, `/tmp` mounts, or CPU pinning are
  not ready.

What to learn:

- GRPO needs at least some correct trajectories to compare.
- If every sample is negative, GRPO cannot learn much about speed.
- Throughput matters: GPUs can sit idle while C++ reward workers compile and
  run candidates.

Decision gate:

- If GRPO-only beats the base model, that is useful evidence.
- More commonly, use GRPO-only to decide what to fix before SFT and GRPO-from-
  SFT: prompt format, parser strictness, reward shape, sandbox reliability, or
  environment concurrency.

## Stage 5: Run SFT From PIE Pairs

Question:

> Can supervised learning teach the model to produce well-formed, compilable,
> plausible optimized C++?

SFT data comes from admitted PIE `v0 -> v1` pairs:

- Prompt: optimization instruction plus slower C++ `v0`.
- Target: required response format containing faster C++ `v1`.

Useful commands:

```bash
uv run w8-biayn config render cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}"

uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --train-epochs 2 \
  --eval-interval 50 \
  --ckpt-interval 100 \
  --hf-save-interval 100 \
  --ckpt-path "${RUN_GCS}/cpp-sft/ckpts" \
  --export-path "${RUN_GCS}/cpp-sft/exports" \
  --max-ckpts-to-keep 2 \
  --no-down-after
```

Expected improvements:

- Higher valid-format rate.
- Lower compile-failure rate.
- More complete C++ outputs.
- More correct trajectories for later GRPO.
- Some improvement in pass rate or speed, though SFT alone may mostly imitate
  PIE style.

What SFT does not prove:

- It does not prove the model can optimize held-out programs beyond imitation.
- It does not directly optimize the runtime reward.
- It can overfit to superficial properties of `v1`.

Decision gate:

- Before using the SFT model for GRPO, verify that the export exists and contains
  actual model weight files. A checkpoint alone is not enough for downstream
  training or evaluation.

## Stage 6: Run GRPO From SFT

Question:

> Can reward optimization improve correctness-gated speed beyond both the base
> model and the SFT model?

GRPO from SFT starts from a model that already knows the response format and can
often produce compilable C++. It then samples multiple candidates per prompt and
updates toward candidates that score better under the sandbox reward.

Useful command:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --model "${RUN_GCS}/cpp-sft/exports" \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --n-samples-per-prompt 8 \
  --train-epochs 3 \
  --no-eval-before-train \
  --eval-interval 25 \
  --max-env-workers 128 \
  --ckpt-interval 50 \
  --hf-save-interval 100 \
  --ckpt-path "${RUN_GCS}/cpp-grpo/ckpts" \
  --export-path "${RUN_GCS}/cpp-grpo/exports" \
  --max-ckpts-to-keep 2 \
  --no-down-after
```

What to inspect:

- Does valid-format rate stay high?
- Do compile and sanitizer failures fall compared with GRPO-only?
- Are more samples passing hidden tests?
- Among fully correct samples, is runtime improving?
- Is reward-worker throughput high enough to keep training efficient?
- Are checkpoints complete, and are exports usable by eval?

What to learn:

- GRPO is not copying `v1`; it is learning from sampled candidates.
- Speed only matters after correctness gates pass.
- A higher train reward without held-out improvement may indicate overfitting,
  task leakage, or reward noise.

Decision gate:

- Continue scaling only if validation or checkpoint evaluation improves
  `correct_and_faster_rate` and reward without increasing missing runtime.

## Stage 7: Evaluate Held-Out Uplift

Question:

> Did training produce a model that is measurably better on unseen tasks?

Run base, SFT, and GRPO on the same held-out data. Otherwise the comparison may
reflect different task difficulty rather than model quality.

Useful commands:

```bash
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "${DATA_GCS}" --run-id "${RUN_ID}" --model Qwen/Qwen2.5-Coder-7B-Instruct --eval-label base --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "${DATA_GCS}" --run-id "${RUN_ID}" --model "${RUN_GCS}/cpp-sft/exports" --eval-label sft --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "${DATA_GCS}" --run-id "${RUN_ID}" --model "${RUN_GCS}/cpp-grpo/exports" --eval-label grpo --n-samples-per-prompt 1

uv run w8-biayn eval cpp \
  --records base=base.records.jsonl \
  --records sft=sft.records.jsonl \
  --records grpo=grpo.records.jsonl \
  --out uplift-summary.json
```

Uplift criteria:

- GRPO beats base on `correct_and_faster_rate`.
- GRPO beats SFT on `correct_and_faster_rate`.
- GRPO beats base on `mean_best_reward`.
- GRPO beats SFT on `mean_best_reward`.
- `missing_runtime_rate=0`.

How to interpret outcomes:

- SFT improves correctness but not speed: SFT taught format and compilation, but
  reward optimization still needs work.
- GRPO improves speed but hurts correctness: the model is not preserving
  behavior well enough; this is not a win.
- Train reward rises but held-out uplift fails: suspect overfitting, split
  leakage, weak hidden tests, or reward noise.
- Missing runtime appears: fix measurement before making any model-quality
  claim.

Decision gate:

- If uplift fails, do not switch to a custom trainer. Study compatible lessons
  from PIE, LearningOpt PIE, or SuperCoder, then port only the useful data,
  prompt, reward, filtering, model, or hyperparameter changes into this
  SkyRL/rLLM pipeline.

## Stage 8: Status JSON As A Teaching Tool

Question:

> Can a student, operator, or UI understand what is happening without reading
> logs by hand?

The productized JSON status is part of the learning system. It should make the
training process visible and debuggable.

Useful command:

```bash
uv run w8-biayn ops run-status \
  --run-id "${RUN_ID}" \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --out ".w8-biayn/runs/${RUN_ID}/status.json"
```

The status JSON should be machine-readable enough for a dashboard or classroom
visualization. It should expose:

- Current stage: data, SFT, GRPO, eval, export, cleanup, or failed.
- Dataset manifest state and schema version.
- Cluster and job state.
- Labeled GCP instances tied to the run.
- Checkpoint marker and shard completeness.
- Highest checkpoint and any in-progress checkpoint upload.
- Export readiness and model weight counts.
- Recent stage, step, checkpoint, export, and error signals from logs.
- Log tail scan size and whether the tail may be truncated.
- GRPO trajectory throughput.
- Evaluation throughput.
- Training throughput.
- Reward metrics.
- ETA and timing metrics.
- Bottleneck hints, such as reward workers versus GPU utilization.
- Optional speed comparisons against a baseline status snapshot.
- Cleanup safety: what still exists and what can be removed.
- Backend/GCS/health check command, return code, timeout flag, and attempt count.

Why this matters:

- Students can see the pipeline as stages, not as a pile of logs.
- Operators can identify whether the bottleneck is data, reward workers, GPU
  training, export, or cloud cleanup.
- A UI can compare runs without scraping terminal output.
- Cleanup status prevents abandoned paid resources.

For reruns or cluster-size experiments, compare against a prior status snapshot
instead of guessing from GPU count alone:

```bash
uv run w8-biayn ops run-status \
  --run-id "${RUN_ID}" \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --baseline-status ".w8-biayn/runs/<baseline-run-id>/status.json" \
  --check-retries 1 \
  --out ".w8-biayn/runs/${RUN_ID}/status.json"
```

Interpret speedup factors directly: greater than `1.0` is faster, less than
`1.0` is slower. If the JSON says `tail_may_be_truncated`, increase
`--log-tail` before teaching or visualizing a stage conclusion from the log
parser.

## Stage 9: Cleanup And Cost Discipline

Question:

> Did the run leave behind expensive resources or ambiguous artifacts?

Use one active full-training cluster at a time on the shared GCP account. Label
paid resources with `project`, `phase`, `pipeline`, `run_id`, `owner`, and
`ttl`.

Cleanup commands:

```bash
uv run w8-biayn gcp cleanup \
  --run-id "${RUN_ID}" \
  --credentials .gcp-service-account.json \
  --dry-run

uv run w8-biayn gcp cleanup \
  --run-id "${RUN_ID}" \
  --credentials .gcp-service-account.json \
  --execute
```

What to learn:

- A failed run can still produce useful checkpoints, logs, and status records.
- A completed run can still leave cloud resources behind.
- Cleanup should be deliberate and inspectable, not a manual hunt through cloud
  consoles.

## Reading Common Failure Patterns

| Observation | Likely Meaning | Next Question |
|---|---|---|
| Invalid format dominates | The model has not learned the response contract | Does SFT or prompt wording teach exactly one reasoning block and one C++ block? |
| Compile failures dominate | The model is not yet a reliable C++ rewriter | Are SFT examples complete and representative? |
| Sanitizer failures dominate | The model writes unsafe C++ | Are unsafe rewrites being penalized clearly? |
| Visible tests pass but hidden tests fail | Behavior is not preserved generally | Are hidden tests and coverage strong enough? |
| Correct but not faster | The model preserves behavior but does not optimize | Is the reward speed bonus informative and stable? |
| Faster but wrong | The model found a shortcut | Are correctness gates ordered before speed reward? |
| Runtime missing | Measurement failed | Is Docker timing preflight clean? |
| GPUs idle | Reward workers may be bottlenecked | Is C++ compile/test/runtime slower than model generation? |
| Checkpoint exists but export is missing weights | Artifact readiness failed | Can the export be recovered before eval or GRPO? |
| Train reward rises but eval does not | Possible overfit or weak split | Are held-out problems truly unseen and well tested? |
| A bigger cluster is not faster | Scaling changed resources, not bottlenecks | What does status JSON say about throughput and speed comparison? |

## What Good Progress Looks Like

Good progress is a chain of falsifiable claims:

1. The machine can build, launch dry runs, and run C++ harness preflight.
2. PIE data has been converted into admitted tasks with tests, coverage, splits,
   reference performance, and a verified SkyRL bundle.
3. Base model evaluation has complete held-out records.
4. GRPO-only reveals the real reward-loop failure modes.
5. SFT from PIE `v0 -> v1` improves format validity and C++ compilation.
6. GRPO from SFT improves correctness-gated runtime reward.
7. Held-out evaluation shows GRPO beating both base and SFT on the required
   uplift metrics.
8. Status JSON explains stages, throughput, bottlenecks, speed comparisons, and
   cleanup state.
9. Paid resources are labeled, inspectable, and cleaned up.

The habit to learn is simple: define the claim, run the smallest experiment that
can falsify it, inspect the failure mode, and only then scale.
