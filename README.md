# w8-biayn

`w8-biayn` is the command-and-control repository for C++ performance RL on PIE data, SkyRL, rLLM, SkyPilot, and Google Cloud.

The active project is C++ only. BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, and Go are out of scope for Phase 1.

## Goal

Train an open-weight model that rewrites correct C++ programs so they run faster while preserving behavior. The public claim is measured on held-out PIE problems with correctness-gated efficiency metrics against base open models and Claude baselines.

We do not write a custom trainer. Training runs through SkyRL/rLLM:

- SkyRL SFT for the cold-start v0-to-v1 format/capability pass.
- SkyRL GRPO for reward training.
- rLLM and vLLM for rollout/training infrastructure.
- `w8-biayn` owns data preparation, task schema, sandbox reward, SkyRL environment registration, SkyPilot rendering, GCP auth, cache management, and docs.

```mermaid
flowchart LR
  pie[PIE C++ v0 to v1 pairs] --> convert[Repeatable data conversion]
  codenet[PIE or CodeNet input-output cases] --> manifest[Tests and coverage manifest]
  manifest --> tasks[Validated task JSON]
  convert --> tasks
  tasks --> skyrlData[SkyRL SFT JSONL and GRPO parquet]
  skyrlData --> gcs[GCS versioned data cache]
  gcs --> sft[SkyRL SFT]
  gcs --> grpo[SkyRL GRPO]
  grpo --> env[cpp-perf SkyRL-Gym env]
  env --> reward[Compile, sanitize, hidden tests, perf instructions]
  reward --> grpo
  grpo --> model[LoRA/checkpoints]
  tasks --> eval[gem5 calibration and eff@k eval]
```

## Fresh-Machine Bootstrap

Run this from a clean clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
```

The bootstrap installs Python dependencies, including dataset tooling (`datasets`, `pyarrow`, `huggingface-hub`, `gdown`). It installs SkyPilot with GCP support unless `--no-sky` is passed.

The CLI uses `.gcp-service-account.json` directly through scoped environment variables. It must not run `gcloud auth activate-service-account` or mutate global `gcloud config`.

## Non-Negotiable Data Rule

Dataset conversion is part of the project. Do not perform one-off PIE or SuperCoder munging outside this CLI.

All source downloads, conversions, manifests, checksums, SkyRL files, GCS uploads, and restores must go through `w8-biayn data ...` commands. Derived data lives under `.w8-biayn/data/` locally and can be cached to GCS.

Default schema version:

```text
cpp-perf-v1
```

Default GCS cache prefix:

```text
gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/skyrl
```

## Dataset Setup

Download source artifacts:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data supercoder download --out .w8-biayn/data/supercoder
```

PIE is the training source. SuperCoder is a reference dataset for schema, evaluation lessons, and correctness-heavy examples. Do not use SuperCoder as a trainer or replace SkyRL/rLLM with SuperCoder code.

Build a tests manifest from input/output directories and a coverage JSON:

```bash
uv run w8-biayn data pie build-tests-manifest \
  --inputs-outputs-basepath .w8-biayn/data/pie/cases \
  --coverage-json .w8-biayn/data/pie/coverage.json \
  --out .w8-biayn/data/pie/tests-manifest.json
```

The coverage JSON is keyed by problem id:

```json
{
  "p03050": {"line": 0.96, "branch": 0.87}
}
```

Build validated task JSON from PIE TSV or JSONL rows:

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

Upload the repeatable bundle to GCS:

```bash
uv run w8-biayn data cache upload \
  --path .w8-biayn/data/skyrl \
  --credentials .gcp-service-account.json
```

The upload command verifies the local manifest and creates the default bucket if it is missing and the service account has permission.

Restore it on another machine:

```bash
uv run w8-biayn data cache restore \
  --path .w8-biayn/data/skyrl \
  --credentials .gcp-service-account.json
```

Restore verifies schema version and file checksums.

## Task Contract

A task JSON is admitted only if it has:

- PIE slower C++ `v0` as the prompt program.
- PIE faster `v1` as oracle/reference material only.
- At least one visible test and one hidden test.
- Coverage at or above 95 percent line and 85 percent branch.
- A positive reference instruction-count value.
- A split of `train`, `validation`, or `test`.

The model never sees `v1` during GRPO rollout. Visible tests are included in the prompt; hidden tests are used only by the reward environment.

Model outputs must have exactly:

````text
<reasoning>...</reasoning>
```cpp
// optimized C++20 program
```
````

Reward order:

- Invalid format: `-1.0`
- Compile or sanitizer failure: `-0.5`
- Partial tests: `-0.2 + 0.2 * fraction_passed`
- Fully correct: `1.0 + bounded instruction-count efficiency bonus`

`perf stat -e instructions:u` is the fast RL reward signal. gem5 from LearningOpt PIE is the calibration and final-eval reference.

## Training

First render before launching:

```bash
uv run w8-biayn config render cpp-sft \
  --credentials .gcp-service-account.json

uv run w8-biayn config render cpp-grpo \
  --credentials .gcp-service-account.json
```

Launch cold-start SFT:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --accelerators A100:8
```

Launch tiny GRPO:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --accelerators A100:8 \
  --train-batch-size 16 \
  --n-samples-per-prompt 4
```

Training defaults use `Qwen/Qwen2.5-Coder-7B-Instruct` on `A100:8` because the current GCP project has A100 quota. Override `--model` and `--accelerators` for GLM-5.1/H100 or larger runs after quota and memory are confirmed.

The GRPO launch calls:

```bash
python -m w8_biayn.integrations.skyrl_cpp_perf_main
```

That entrypoint registers `cpp-perf` inside the SkyRL Ray task and delegates to SkyRL `BasePPOExp(cfg).run()`. It is glue, not a custom trainer.

```mermaid
sequenceDiagram
  participant User
  participant CLI as w8-biayn
  participant GCS
  participant Sky as SkyPilot
  participant SkyRL
  participant Env as cpp-perf env
  participant Docker as C++ sandbox

  User->>CLI: data skyrl build
  CLI->>CLI: write parquet, JSONL, tasks, manifest
  User->>CLI: data cache upload
  CLI->>GCS: rsync verified bundle
  User->>CLI: launch cpp-grpo
  CLI->>Sky: render and launch YAML
  Sky->>GCS: restore data bundle
  Sky->>SkyRL: run skyrl_cpp_perf_main
  SkyRL->>Env: generate candidate
  Env->>Docker: compile, sanitize, test, perf
  Docker-->>Env: reward metadata
  Env-->>SkyRL: reward
```

## Smoke And Operations

Prove GPU/SkyPilot viability:

```bash
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-smoke --credentials .gcp-service-account.json
```

If the project has no H100 quota, override the smoke model/accelerator:

```bash
uv run w8-biayn launch cpp-smoke \
  --credentials .gcp-service-account.json \
  --accelerators A100:8 \
  --model Qwen/Qwen2.5-Coder-7B-Instruct
```

Inspect operations:

```bash
uv run w8-biayn status
uv run w8-biayn logs w8-biayn-cpp-grpo
uv run w8-biayn down w8-biayn-cpp-grpo
uv run w8-biayn benchmarks list
```

## Upstreams

Pinned upstream source clones live under `.cache/upstreams/`:

- SkyRL
- rLLM
- PIE eval/data scripts
- LearningOpt PIE gem5 environment

Use:

```bash
uv run w8-biayn upstreams clone
uv run w8-biayn upstreams status
```

Do not vendor upstream repos, CodeNet, gem5 outputs, generated datasets, checkpoints, or cloud credentials.

## Validation

Before handing off changes:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup or CLI-surface changes:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn data doctor
uv run w8-biayn benchmarks list
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json
```
