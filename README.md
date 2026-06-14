# w8-biayn

`w8-biayn` is the command-and-control repository for Phase-1 C++ performance RL. It uses the inherited GCP/SkyPilot setup as infrastructure, but the active project is new: train an open-weight model to rewrite correct C++ programs so they run faster while preserving behavior.

BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, and Go are out of scope for Phase 1.

## Goal

Train and evaluate a C++ optimizer model on PIE-derived tasks:

- Input: a correct but slower C++20 program.
- Output: a behavior-preserving C++20 rewrite.
- Reward: valid format, compile/sanitizer success, visible and hidden tests, then bounded instruction-count efficiency.
- Training stack: SkyRL SFT for cold start, then SkyRL GRPO through rLLM/vLLM.
- Final claim: correctness-gated efficiency on held-out PIE problems versus base open models and Claude baselines.

We do not write a custom trainer. PIE, LearningOpt PIE, and SuperCoder can be studied and ported into this repo's data, reward, and eval surfaces, but SkyRL/rLLM remain the active trainer stack.

## Current Implementation

Implemented in this repo:

- Fresh-machine bootstrap and diagnostics.
- Pinned upstream clones for SkyRL, rLLM, PIE, and LearningOpt PIE.
- PIE and SuperCoder source downloads into local manifests.
- Test-manifest construction from normalized input/output directories plus coverage JSON.
- PIE row to validated C++ task JSON conversion.
- SkyRL GRPO parquet and SFT JSONL dataset bundle generation.
- GCS upload/restore for versioned dataset bundles.
- Docker-based C++ compile, sanitizer, hidden-test, and `perf stat -e instructions:u` reward harness.
- Docker-based `instructions:u` preflight so GRPO fails before training if the host PMU signal is unavailable.
- SkyRL `cpp-perf` environment registration and GRPO entrypoint glue.
- SkyPilot render/launch commands for GPU smoke, SFT, and GRPO jobs on GCP.
- Benchmark ladder metadata.

Still to build before a full benchmark run:

- Raw PIE archive normalization into the exact `pairs` and `cases` inputs used by the task builder, if the downloaded archive layout differs locally.
- Automated oracle-case generation from PIE `v1` solutions and coverage measurement instead of supplying a coverage JSON manually.
- gem5 calibration/final eval wiring through LearningOpt PIE.
- Scaled training configs, checkpoint storage policy, and full held-out `eff@k` reporting.

Any missing data conversion must be added as a `w8-biayn data ...` command. Do not do one-off munging in notebooks, shell history, or untracked scripts.

## Architecture

```mermaid
flowchart LR
  pie[PIE C++ v0 to v1 pairs] --> tasks[Validated C++ task JSON]
  cases[Official and oracle tests] --> tests[Test and coverage manifest]
  tests --> tasks
  tasks --> skyrl[SkyRL bundle: GRPO parquet and SFT JSONL]
  skyrl --> gcs[GCS versioned cache]
  gcs --> sft[SkyRL SFT]
  gcs --> grpo[SkyRL GRPO]
  grpo --> env[cpp-perf SkyRL env]
  env --> docker[Docker C++ sandbox]
  docker --> reward[format, compile, sanitizer, tests, perf instructions]
  reward --> grpo
  grpo --> lora[LoRA checkpoints]
  tasks --> eval[gem5 calibration and eff@k eval]
```

```mermaid
sequenceDiagram
  participant Dev
  participant CLI as w8-biayn
  participant GCS
  participant Sky as SkyPilot
  participant SkyRL
  participant Env as cpp-perf env
  participant Docker as Host Docker

  Dev->>CLI: data skyrl build
  CLI->>CLI: write parquet, JSONL, task copies, manifest
  Dev->>CLI: data cache upload
  CLI->>GCS: rsync verified bundle
  Dev->>CLI: launch cpp-grpo
  CLI->>Sky: render and launch YAML
  Sky->>GCS: restore data bundle
  Sky->>SkyRL: run skyrl_cpp_perf_main
  SkyRL->>Env: candidate output
  Env->>Docker: compile, sanitize, test, perf
  Docker-->>Env: harness result
  Env-->>SkyRL: reward and metrics
```

## Fresh-Machine Setup

Run from a clean clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn upstreams status
```

The service-account JSON stays local. The CLI uses scoped environment variables such as `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and `CLOUDSDK_CORE_PROJECT`; it must not run `gcloud auth activate-service-account` or mutate global `gcloud config`.

Generated data, upstream clones, rendered SkyPilot YAML, secrets, logs, and checkpoints are local state and are ignored by git.

## Data Pipeline

The data pipeline is a project deliverable. Every conversion, cleanup, cache write, and cache restore must be repeatable through the CLI.

Default local roots:

```text
.cache/upstreams/
.w8-biayn/data/
.w8-biayn/data/tasks/
.w8-biayn/data/skyrl/
```

Default schema and cache prefix:

```text
cpp-perf-v1
gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/skyrl
```

Download source artifacts:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data supercoder download --out .w8-biayn/data/supercoder
uv run w8-biayn data supercoder inspect --parquet path/to/resolved-supercoder.parquet
```

SuperCoder is reference material only. It can inform schema, correctness checks, and eval lessons; it must not replace SkyRL/rLLM training.

Build a tests manifest from normalized input/output files:

```bash
uv run w8-biayn data pie build-tests-manifest \
  --inputs-outputs-basepath .w8-biayn/data/pie/cases \
  --coverage-json .w8-biayn/data/pie/coverage.json \
  --out .w8-biayn/data/pie/tests-manifest.json \
  --visible-count 1 \
  --hidden-count 1
```

Expected case layout:

```text
.w8-biayn/data/pie/cases/<problem_id>/input.0.txt
.w8-biayn/data/pie/cases/<problem_id>/output.0.txt
.w8-biayn/data/pie/cases/<problem_id>/input.1.txt
.w8-biayn/data/pie/cases/<problem_id>/output.1.txt
```

Coverage JSON is keyed by problem id:

```json
{
  "p03050": {"line": 0.96, "branch": 0.87}
}
```

Build validated task JSON:

```bash
uv run w8-biayn data pie build-tasks \
  --pairs .w8-biayn/data/pie/train.jsonl \
  --tests-json .w8-biayn/data/pie/tests-manifest.json \
  --out .w8-biayn/data/tasks/train \
  --split train

uv run w8-biayn data pie build-tasks \
  --pairs .w8-biayn/data/pie/validation.jsonl \
  --tests-json .w8-biayn/data/pie/tests-manifest.json \
  --out .w8-biayn/data/tasks/validation \
  --split validation
```

`--pairs` may be TSV or JSONL. The parser accepts common PIE column aliases for problem id, slower code `v0`, faster code `v1`, and reference runtime.

Build SkyRL datasets:

```bash
uv run w8-biayn data skyrl build \
  --tasks-dir .w8-biayn/data/tasks \
  --out .w8-biayn/data/skyrl
```

This writes:

- `.w8-biayn/data/skyrl/grpo/train.parquet`
- `.w8-biayn/data/skyrl/grpo/validation.parquet`
- `.w8-biayn/data/skyrl/sft/train.jsonl`
- `.w8-biayn/data/skyrl/sft/validation.jsonl`
- `.w8-biayn/data/skyrl/tasks/...`
- `.w8-biayn/data/skyrl/_w8_data_manifest.json`

Upload and restore the bundle:

```bash
uv run w8-biayn data cache upload \
  --path .w8-biayn/data/skyrl \
  --credentials .gcp-service-account.json

uv run w8-biayn data cache restore \
  --path .w8-biayn/data/skyrl \
  --credentials .gcp-service-account.json
```

Upload verifies the local manifest before syncing. Restore verifies schema version, file sizes, and checksums after syncing.

## Task Contract

A task JSON is valid only if it has:

- PIE slower C++ `v0` as `prompt_code`.
- PIE faster C++ `v1` as `oracle_solution` only.
- At least one visible unit test and one hidden test.
- Coverage at or above 95 percent line and 85 percent branch.
- Positive reference instruction-count value.
- Split `train`, `validation`, or `test`.
- Source `PIE`.

The model never sees `v1` during GRPO rollout. Visible tests can be included in the prompt. Hidden tests are grading-only. SFT may use `v1` as the supervised target for cold start.

Model output format:

````text
<reasoning>...</reasoning>
```cpp
// complete optimized C++20 program
```
````

## Reward Harness

Local dry runs:

```bash
uv run w8-biayn cpp harness run \
  --task .w8-biayn/data/tasks/train/pie_cpp_000001.json \
  --candidate candidate.cpp \
  --dry-run

uv run w8-biayn cpp reward score \
  --task .w8-biayn/data/tasks/train/pie_cpp_000001.json \
  --model-output model-output.md \
  --dry-run
```

Real scoring requires Docker, Linux `perf`, taskset, and a sandbox image containing `g++`, `bash`, `taskset`, and `perf`. The CLI default sandbox image is `w8-biayn-cpp-perf:latest`, built locally from `gcc:13` with `linux-perf` installed. Use `--no-build-image` only when you provide a prebuilt image with `--image`.

Before any GRPO run, prove the host and sandbox can collect a numeric `instructions:u` count:

```bash
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn cpp harness preflight --cpu 3
uv run w8-biayn doctor --cpp-perf --credentials .gcp-service-account.json
```

Having the `perf` binary in the image is not enough. Some virtualized or cloud hosts do not expose the hardware counter; those hosts are invalid for training because fully correct candidates would receive `missing_instruction_count` instead of an efficiency reward.

Reward order:

- Invalid format: `-1.0`
- Compile or sanitizer failure: `-0.5`
- Partial tests: `-0.2 + 0.2 * fraction_passed`
- Fully correct: `1.0 + bounded instruction-count efficiency bonus`

The fast RL signal is `perf stat -e instructions:u`. gem5 from LearningOpt PIE is the deterministic calibration and final-eval reference, not the per-rollout reward path.

## Training On GCP

Training requires a verified SkyRL data bundle in GCS.

Render before launching:

```bash
uv run w8-biayn config render cpp-smoke --credentials .gcp-service-account.json
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json
```

GPU smoke:

```bash
uv run w8-biayn launch cpp-smoke \
  --credentials .gcp-service-account.json \
  --accelerators A100:8 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct
```

Cold-start SFT:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --accelerators A100:8 \
  --train-batch-size 16 \
  --no-down-after
```

Tiny GRPO smoke:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --accelerators A100:8 \
  --train-batch-size 16 \
  --n-samples-per-prompt 4 \
  --no-down-after
```

Full training should enable checkpoint/export retention explicitly:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --accelerators A100:8 \
  --train-batch-size 16 \
  --n-samples-per-prompt 8 \
  --train-epochs 3 \
  --eval-interval 25 \
  --ckpt-interval 50 \
  --hf-save-interval 100 \
  --ckpt-path gs://<project>-w8-biayn/checkpoints/cpp-grpo \
  --export-path gs://<project>-w8-biayn/exports/cpp-grpo \
  --max-ckpts-to-keep 2 \
  --no-down-after
```

Training defaults use `Qwen/Qwen2.5-Coder-7B-Instruct` on `A100:8` because the current GCP project has A100 quota. The smoke default remains `zai-org/GLM-5.1` on `H100:8`; override model and accelerators when quota or memory requires it.

The default GRPO launch is intentionally a small smoke path. It runs:

```bash
python -m w8_biayn.integrations.skyrl_cpp_perf_main
```

That entrypoint registers the `cpp-perf` SkyRL environment and delegates to SkyRL `BasePPOExp(cfg).run()`. It is glue, not a custom trainer. The rendered GRPO config disables checkpoint and HF saves by default with `trainer.ckpt_interval=-1` and `trainer.hf_save_interval=-1` so rollout, reward, update, and eval can be tested without filling a default GCP boot disk.

GRPO reward execution uses Docker-outside-Docker. The SkyPilot YAML mounts `/var/run/docker.sock` and host `/tmp` into the GPU training container so the SkyRL environment can create scratch directories and ask the host Docker daemon to run locked-down C++ sandboxes. Rendered `cpp-grpo` runs `w8-biayn cpp harness preflight` before `skyrl_cpp_perf_main`; a failed preflight stops the job before paid training starts.

## Operations

```bash
uv run w8-biayn status
uv run w8-biayn logs w8-biayn-cpp-grpo
uv run w8-biayn down w8-biayn-cpp-grpo
uv run w8-biayn benchmarks list
uv run w8-biayn benchmarks show grpo-tiny
```

`launch` passes `--down` to SkyPilot by default. Use `--no-down-after` for training runs you want to inspect after completion.

## Upstreams

Pinned upstream clones live under `.cache/upstreams/`:

- `SkyRL`
- `rllm`
- `pie-perf`
- `LearningOpt-pie`

Use:

```bash
uv run w8-biayn upstreams clone
uv run w8-biayn upstreams status
```

Do not vendor upstream repos, CodeNet, PIE archives, generated tests, gem5 outputs, datasets, checkpoints, or credentials. If you need an ad hoc clone for study, put it in `/tmp` and port only the required behavior into this repo's CLI, tests, and docs.

## Repository Map

```text
scripts/bootstrap.sh                       fresh-machine bootstrap
src/w8_biayn/cli.py                         CLI surface
src/w8_biayn/cpp_perf/schema.py             task and harness models
src/w8_biayn/cpp_perf/pie.py                PIE row parsing and task construction
src/w8_biayn/cpp_perf/data.py               downloads, manifests, GCS cache helpers
src/w8_biayn/cpp_perf/skyrl_dataset.py      SkyRL GRPO/SFT dataset builder
src/w8_biayn/cpp_perf/sandbox.py            Docker compile/test/perf harness
src/w8_biayn/cpp_perf/reward.py             correctness-gated efficiency reward
src/w8_biayn/integrations/cpp_perf_env.py   SkyRL environment adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                             SkyRL GRPO entrypoint glue
src/w8_biayn/sky_config.py                  SkyPilot YAML renderer
src/w8_biayn/gcp_auth.py                    scoped GCP auth
src/w8_biayn/secrets.py                     credential metadata only
src/w8_biayn/constants.py                   upstream pins and defaults
src/w8_biayn/upstreams.py                   upstream clone management
src/w8_biayn/benchmarks.py                  benchmark ladder
.agents/REPO_GUIDE.md                       shared AGENTS.md and CLAUDE.md target
.agents/skills/w8-biayn-framework/SKILL.md  AI coding-agent workflow skill
```

## Validation

Before handing off code or docs:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup, CLI, cloud, or data-pipeline changes, also run the relevant dry checks:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn data doctor
uv run w8-biayn benchmarks list
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json
```
