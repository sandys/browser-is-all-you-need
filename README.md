# w8-biayn

`w8-biayn` is the command-and-control repository for Phase-1 C++ performance RL. The inherited repo is used only as baseline infrastructure for Google Cloud, SkyPilot, SkyRL, and rLLM; custom GPU kernel labs and unrelated performance experiments are not part of the active surface.

The active project is C++ only: train an open-weight model that rewrites correct C++20 programs so they run faster while preserving behavior.

Out of scope for Phase 1: BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, and Go.

## Goal

- Data: official PIE C++ slower-to-faster pairs and official/merged/generated tests.
- Task: prompt with slower C++ `v0`; generate a complete optimized C++20 program.
- Reward: strict output format, compile and sanitizer success, visible and hidden tests, then bounded CPU-time runtime efficiency.
- Training: SkyRL SFT cold start, then SkyRL GRPO through rLLM/vLLM. No custom trainer.
- Proof: compare base, SFT, and GRPO checkpoints on the same held-out PIE tasks with `pass_rate`, `correct_and_faster_rate`, mean reward, speedup, and missing-runtime rate.

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
  sandbox --> reward[format, compile, sanitizer, tests, runtime CPU ns]
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
  SkyRL->>Docker: compile, sanitize, test, runtime benchmark
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
- positive reference performance metadata;
- split `train`, `validation`, or `test`.

The model must return exactly:

````text
<reasoning>...</reasoning>
```cpp
// complete optimized C++20 program
```
````

The parser accepts a standard fenced block with code on the next line, or C++ code after whitespace on the same fence line. It still requires exactly one reasoning block and exactly one C++ code block.

Reward order:

- invalid format: `-1.0`;
- compile or sanitizer failure: negative;
- timeout: negative;
- partial tests: shaped but below fully correct;
- fully correct with missing non-timeout runtime measurement: correctness-only fallback below any measured fully correct answer;
- fully correct: `1.0 + bounded runtime-efficiency bonus`.

Before GRPO, the host must pass the Docker runtime harness preflight:

```bash
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn cpp harness preflight --cpu 3
uv run w8-biayn doctor --cpp-perf --credentials .gcp-service-account.json
```

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and hidden tests, then benchmarks both binaries in the same Docker sandbox with the same CPU pinning, compiler flags, and tests. Runtime measurement uses child-process CPU time in nanoseconds, with wall-clock nanoseconds recorded as diagnostics. The default benchmark uses 1 warmup and 3 measured repeats, takes the median per test, and sums across tests.

Rendered `cpp-grpo` and `cpp-eval` SkyPilot jobs run a host-side C++ runtime preflight before GCS restore, model staging, GPU image pulls, or framework installs, and then run the same preflight again inside the GPU container before `skyrl_cpp_perf_main` or eval scoring.

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

Launch GRPO after SFT produces a usable export. Use the SFT export as `--model` when available:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
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

If a GRPO attempt is canceled or fails after a complete checkpoint, restart it with the same checkpoint path and `--resume-from latest`. For multi-node GRPO, pass `--num-nodes N`; the renderer keeps SkyRL colocated, starts a rank-gated Ray cluster inside the GPU containers, runs `skyrl_cpp_perf_main` only on rank 0, and sets rollout engines to `N * GPUs_PER_NODE` when tensor/data parallelism are `1`. Multi-node GRPO is rejected unless effective samples per step (`--train-batch-size * --n-samples-per-prompt`) are at least 16 per GPU and `--max-env-workers` can cover the effective samples. For 2x[A100:8], use a tuned starting point such as `--train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256`; pass `--allow-low-multinode-utilization` only for an intentional experiment. Full GRPO can skip the expensive initial validation pass with `--no-eval-before-train`; the renderer uses non-batched trajectory generation so SkyRL-Gym can overlap C++ reward calls, and `--max-env-workers` controls that Docker reward concurrency.

`launch` auto-adds a run ID if omitted, but full runs should pass the same `RUN_ID` everywhere. Rendered YAML includes GCP labels under `resources.labels`: `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`.
Full training and eval default to a 1024 GB boot disk; keep `--disk-size 1024` or larger on resumed/checkpointed runs because FSDP checkpoint restore and HF export staging need local scratch space.

If an SFT attempt fails after writing a checkpoint, resume it from the latest marker with the same checkpoint path:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
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
  --resume-from latest
```

If the SkyRL checkpoint is complete but the HF export is incomplete, recover the export without another SFT epoch. Use a local artifact export path so the launcher uploads the finished HF directory back under `${RUN_GCS}/cpp-sft/exports`:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --export-path "~/exports/" \
  --export-checkpoint "${RUN_GCS}/cpp-sft/ckpts/global_step_1074" \
  --no-down-after
```

Verify recovery with `ops run-status`: `artifacts.export.final_export_exists` must be `true` and `artifacts.export.final_export.weight_object_count` must be greater than zero before GRPO uses the SFT export.

When `--model` points at a `gs://` export for GRPO or eval, the launcher stages the model into a directory mounted inside the GPU container before invoking SkyRL or vLLM. Do not pass host-only staged paths into containerized training.

GRPO does not require PMU access or Linux perf counters. The required host capability is Docker-outside-Docker plus enough CPU stability for the runtime harness to compare candidate and oracle binaries consistently.

The default `launch` includes SkyPilot `--down`; `--no-down-after` keeps a cluster for inspection. On a shared account, run cleanup when an attempt fails or finishes:

```bash
uv run w8-biayn gcp cleanup --run-id "${RUN_ID}" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "${RUN_ID}" --credentials .gcp-service-account.json --execute
```

Use the project ops commands for run inspection and control. Do not put raw `sky ...` commands in runbooks; `w8-biayn ops ...` is the stable DX boundary if the backend changes later.

```bash
uv run w8-biayn ops status --credentials .gcp-service-account.json --refresh
uv run w8-biayn ops run-status \
  --run-id "${RUN_ID}" \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --expected-sft-final-step 1074 \
  --baseline-status ".w8-biayn/runs/<single-node-run-id>/status.json" \
  --check-retries 1 \
  --node-health \
  --out ".w8-biayn/runs/${RUN_ID}/status.json"
uv run w8-biayn ops queue "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json --tail 200
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" 1 --credentials .gcp-service-account.json --follow
uv run w8-biayn ops cancel "w8-biayn-cpp-grpo-${RUN_ID}" 1 --credentials .gcp-service-account.json
uv run w8-biayn ops down "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions
```

For a rerun or cluster-size experiment, pass one or more prior snapshots with `--baseline-status ".w8-biayn/runs/<baseline-run-id>/status.json"` so the JSON includes `speed_comparison` for training-step and rollout throughput. Interpret speedup factors directly: greater than `1.0` is faster than the baseline, less than `1.0` is slower, and `gpu_speedup_efficiency` is the speedup divided by the GPU scale factor. A `cost_verdict` of `cost_inefficient` means the current run used more GPUs without increasing the primary comparable throughput.

`ops run-status` emits `w8-run-status-v1` JSON for dashboards and polling loops. It includes dataset manifest state, per-pipeline cluster/job state, labeled GCP instances, checkpoint marker and shard completeness for the promoted `latest` checkpoint, highest checkpoint directory, active `in_progress` checkpoint upload, export readiness including final export object counts/bytes and model weight presence, recent log-derived stage/step/checkpoint/export/error signals, normalized phase/progress/resource/command fields, GRPO config (`effective_samples_per_step`, total GPUs, samples/GPU/step, reward workers), trajectory/evaluation/training throughput, GPU-normalized throughput, ETA/timing metrics, reward metrics, bottleneck verdicts from SkyRL timing, optional `speed_comparison` against prior `--baseline-status` snapshots, and cleanup safety. Pass `--node-health` for opt-in read-only SSH health with GPU utilization/memory, disk free space, top processes, a derived node activity, and an explicit `sample_scope` so dashboards do not mistake head-only probes for whole-cluster telemetry. Logs include `tail_lines_requested`, `tail_lines_scanned`, and `tail_may_be_truncated`; treat a truncated tail as a hint to increase `--log-tail` before drawing conclusions from stage parsing. Each backend/GCS/health check is listed with its command, return code, `timed_out`, and `attempt_count`; tune per-check timeout with `--check-timeout` and retry timed-out read-only checks with `--check-retries`.

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

Success criterion for an uplift claim: GRPO beats base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_runtime_rate=0`.

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
src/w8_biayn/cpp_perf/sandbox.py             Docker compile/test/runtime harness
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
uv run w8-biayn ops run-status --run-id rdoc --credentials .gcp-service-account.json --dry-run
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions --dry-run
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-eval --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn gcp cleanup --run-id rdoc --credentials .gcp-service-account.json --dry-run
```
