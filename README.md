# w8-biayn

`w8-biayn` is the command-and-control repository for Phase-1 C++ performance RL. The inherited repo is used only as baseline infrastructure for Google Cloud, SkyPilot, SkyRL, and rLLM.

The active project is C++ only: train an open-weight model that rewrites correct C++20 programs so they run faster while preserving behavior.

Out of scope for Phase 1: BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, and Go.

## Goal

- Data: official PIE C++ slower-to-faster pairs and official/merged/generated tests.
- Task: prompt with slower C++ `v0`; generate a complete optimized C++20 program.
- Reward: exact output format, compile and sanitizer success, visible and hidden tests, then bounded `perf stat -e instructions:u` efficiency.
- Training: SkyRL SFT cold start, then SkyRL GRPO through rLLM/vLLM. No custom trainer.
- Proof: compare base, SFT, and GRPO checkpoints on the same held-out PIE tasks with `pass_rate`, `correct_and_faster_rate`, mean reward, speedup, and missing-instruction rate.

PIE, LearningOpt PIE, and SuperCoder may be studied for data/eval lessons, but all repeatable work must be implemented as project CLI commands. Do not use PIE's old Trainer or any SuperCoder trainer as the active training stack.

## Architecture

```mermaid
flowchart LR
  raw[Official PIE archives] --> prep[prepare-full: splits and cases]
  prep --> cov[measure-coverage with gcov]
  cov --> tasks[coverage-gated C++ task JSON]
  tasks --> skyrl[SkyRL bundle: GRPO parquet and SFT JSONL]
  skyrl --> gcs[GCS versioned cache]
  gcs --> sft[SkyRL SFT]
  gcs --> grpo[SkyRL GRPO]
  grpo --> env[cpp-perf SkyRL env]
  env --> sandbox[Docker C++ sandbox]
  sandbox --> reward[format, compile, sanitizer, tests, instructions:u]
  reward --> grpo
  gcs --> eval[cpp-eval vLLM + reward harness]
  sft --> eval
  grpo --> eval
  eval --> report[uplift report]
```

```mermaid
sequenceDiagram
  participant Dev
  participant CLI as w8-biayn
  participant GCS
  participant Sky as SkyPilot
  participant SkyRL
  participant Docker as Host Docker

  Dev->>CLI: data pie download / prepare-full / measure-coverage
  CLI->>CLI: build-full-tasks with admission report
  CLI->>CLI: data skyrl build with full-run count gates
  Dev->>CLI: data cache upload --gcs-prefix full-official/RUN_ID
  CLI->>GCS: rsync verified bundle
  Dev->>CLI: launch cpp-sft/cpp-grpo --run-id RUN_ID
  CLI->>Sky: render YAML with GCP labels and scoped credentials
  Sky->>GCS: restore data bundle
  Sky->>SkyRL: SFT or GRPO
  SkyRL->>Docker: compile, sanitize, test, perf
  Docker-->>SkyRL: reward result
  Dev->>CLI: launch cpp-eval for base/SFT/GRPO
  CLI->>GCS: upload eval records and summaries
```

## Fresh Machine Setup

Run from a clean clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

The service-account JSON stays local. The CLI uses scoped environment variables such as `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and `CLOUDSDK_CORE_PROJECT`; it must not run `gcloud auth activate-service-account` or mutate global `gcloud config`.

Generated data, upstream clones, rendered SkyPilot YAML, secrets, logs, and checkpoints are local state and ignored by git.

## Full Official PIE Dataset

Dataset conversion is a deliverable. No one-off notebook, shell-history, or untracked-script munging is allowed.

Use one run ID for the dataset, training, eval, and cleanup:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
PROJECT_ID="$(python - <<'PY'
import json
print(json.load(open('.gcp-service-account.json'))['project_id'])
PY
)"
DATA_GCS="gs://${PROJECT_ID}-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl"
RUN_GCS="gs://${PROJECT_ID}-w8-biayn/runs/cpp-perf/${RUN_ID}"
```

Build the full official PIE task pool:

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

Admission gates are intentional. Paid training must not start if:

- full train tasks are fewer than 1000;
- validation or test tasks are fewer than 100;
- coverage is below 95 percent line or 85 percent branch;
- visible or hidden tests are missing;
- the SkyRL bundle manifest fails checksum verification.

The prepared data and task build write `_w8_*` manifests/reports with counts and rejection reasons.

## Task And Reward Contract

A valid task contains:

- `prompt_code`: slower correct PIE C++ `v0`;
- `oracle_solution`: faster PIE C++ `v1`, used for SFT/coverage/reference only;
- visible `unit_tests` and grading-only `hidden_tests`;
- `test_coverage` at or above 95 percent line and 85 percent branch;
- positive reference instruction-count value;
- split `train`, `validation`, or `test`.

The model must return exactly:

````text
<reasoning>...</reasoning>
```cpp
// complete optimized C++20 program
```
````

Reward order:

- invalid format: `-1.0`;
- compile or sanitizer failure: negative;
- partial tests: shaped but below fully correct;
- fully correct: `1.0 + bounded instruction-count efficiency bonus`.

Before GRPO, the host must expose a numeric instruction counter:

```bash
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn cpp harness preflight --cpu 3
uv run w8-biayn doctor --cpp-perf --credentials .gcp-service-account.json
```

Having the `perf` binary is not enough. Some virtualized hosts return `<not counted>` or `<not supported>` for `instructions:u`; those hosts are invalid for training. A GCP A100/A2 VM can fail this way even after provisioning succeeds.
Rendered `cpp-grpo` and `cpp-eval` SkyPilot jobs lower the host `kernel.perf_event_paranoid` setting to `0`, run a host-side C++ perf preflight before GCS restore, model staging, GPU image pulls, or framework installs, and then run the C++ perf preflight again inside the GPU container before `skyrl_cpp_perf_main` or eval scoring.

## GCP Training

Render before paid launches:

```bash
uv run w8-biayn config render cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}"

uv run w8-biayn config render cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}"
```

Launch SFT:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --accelerators A100:8 \
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

Launch GRPO after SFT produces a usable export. Use the SFT export as `--model` when available:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model "${RUN_GCS}/cpp-sft/exports" \
  --accelerators A100:8 \
  --train-batch-size 16 \
  --n-samples-per-prompt 8 \
  --train-epochs 3 \
  --eval-interval 25 \
  --ckpt-interval 50 \
  --hf-save-interval 100 \
  --ckpt-path "${RUN_GCS}/cpp-grpo/ckpts" \
  --export-path "${RUN_GCS}/cpp-grpo/exports" \
  --max-ckpts-to-keep 2 \
  --no-down-after
```

`launch` auto-adds a run ID if omitted, but full runs should pass the same `RUN_ID` everywhere. Rendered YAML includes GCP labels under `resources.labels`: `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`.

GRPO requires a GCP host where `perf stat -e instructions:u` returns a numeric count. Google Compute Engine PMU support must be enabled on a supported PMU machine series; do not keep retrying A2/A3/G2 GPU hosts that fail preflight with `<not supported>`.

The default `launch` includes SkyPilot `--down`; `--no-down-after` keeps a cluster for inspection. On a shared account, run cleanup when an attempt fails or finishes:

```bash
uv run w8-biayn gcp cleanup --run-id "${RUN_ID}" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "${RUN_ID}" --credentials .gcp-service-account.json --execute
```

Use the project ops commands for run inspection and control. Do not put raw `sky ...` commands in runbooks; `w8-biayn ops ...` is the stable DX boundary if the backend changes later.

```bash
uv run w8-biayn ops status --credentials .gcp-service-account.json --refresh
uv run w8-biayn ops queue "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json --tail 200
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" 1 --credentials .gcp-service-account.json --follow
uv run w8-biayn ops cancel "w8-biayn-cpp-grpo-${RUN_ID}" 1 --credentials .gcp-service-account.json
uv run w8-biayn ops down "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions
```

## Uplift Evaluation

Run the same held-out validation bundle against base, SFT, and GRPO models:

```bash
uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --eval-label base \
  --n-samples-per-prompt 1 \
  --eval-max-tasks 200

uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model "${RUN_GCS}/cpp-sft/exports" \
  --eval-label sft \
  --n-samples-per-prompt 1 \
  --eval-max-tasks 200

uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model "${RUN_GCS}/cpp-grpo/exports" \
  --eval-label grpo \
  --n-samples-per-prompt 1 \
  --eval-max-tasks 200
```

`cpp-eval` stages `gs://` model exports to local VM storage before loading vLLM. Eval artifacts are uploaded to:

```text
gs://<project>-w8-biayn/runs/cpp-perf/<RUN_ID>/cpp-eval/
```

Aggregate local or restored eval records:

```bash
uv run w8-biayn eval cpp \
  --records base=base.records.jsonl \
  --records sft=sft.records.jsonl \
  --records grpo=grpo.records.jsonl \
  --out uplift-summary.json
```

Success criterion for an uplift claim: GRPO beats base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_instr_rate=0`.

If uplift fails, clone/study SuperCoder and Microsoft/LearningOpt PIE into `/tmp`, compare filtering, prompts, reward shaping, model choice, and hyperparameters, then port only compatible fixes into this SkyRL/rLLM pipeline.

## Operations

```bash
uv run w8-biayn ops status --credentials .gcp-service-account.json
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json --tail 200
uv run w8-biayn ops down "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn benchmarks list
uv run w8-biayn benchmarks show grpo-tiny
```

The legacy top-level `w8-biayn status`, `w8-biayn logs`, and `w8-biayn down` commands remain aliases, but new docs and scripts should prefer `w8-biayn ops ...`.

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

Do not vendor upstream repos, CodeNet, PIE archives, generated tests, gem5 outputs, datasets, checkpoints, or credentials.

## Repository Map

```text
scripts/bootstrap.sh                         fresh-machine bootstrap
src/w8_biayn/cli.py                          CLI surface
src/w8_biayn/cpp_perf/data.py                downloads, full PIE prep, manifests, cache
src/w8_biayn/cpp_perf/coverage.py            gcov coverage measurement
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/skyrl_dataset.py       SkyRL GRPO/SFT dataset builder
src/w8_biayn/cpp_perf/eval.py                eval aggregation
src/w8_biayn/cpp_perf/sandbox.py             Docker compile/test/perf harness
src/w8_biayn/cpp_perf/reward.py              correctness-gated efficiency reward
src/w8_biayn/integrations/cpp_perf_env.py    SkyRL environment adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                               SkyRL GRPO entrypoint glue
src/w8_biayn/integrations/cpp_eval_main.py   vLLM eval generation and scoring
src/w8_biayn/sky_config.py                   SkyPilot YAML renderer
src/w8_biayn/gcp_auth.py                     scoped GCP auth
src/w8_biayn/secrets.py                      credential metadata only
src/w8_biayn/constants.py                    upstream pins and defaults
src/w8_biayn/upstreams.py                    upstream clone management
src/w8_biayn/benchmarks.py                   benchmark ladder
.agents/REPO_GUIDE.md                        shared AGENTS.md and CLAUDE.md target
.agents/skills/w8-biayn-framework/SKILL.md   AI coding-agent workflow skill
```

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup, CLI, cloud, or data-pipeline changes, also run relevant dry checks:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn data doctor
uv run w8-biayn benchmarks list
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn ops status --credentials .gcp-service-account.json --dry-run
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions --dry-run
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-eval --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn gcp cleanup --run-id rdoc --credentials .gcp-service-account.json --dry-run
```
