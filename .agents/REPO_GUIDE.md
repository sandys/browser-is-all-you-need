# Repository Development Guide

`AGENTS.md` and `CLAUDE.md` must stay symlinks to this file. Update this file once; do not fork the guidance.

## Active Project

This repository is a C++ performance-RL project. The inherited codebase is only baseline infrastructure for GCP, SkyPilot, SkyRL, and rLLM.

Phase 1 goal: train an open-weight model that rewrites correct C++ programs to run faster while preserving behavior, then prove uplift on held-out PIE tasks.

Out of scope unless a later phase is explicitly requested:

- BrowserGym
- DOMDiff
- Harbor
- WebArena
- MiniWoB
- AndroidWorld
- Go
- Custom GPU kernel labs or unrelated performance experiments

## Required Reading

Before changing behavior, read:

1. `README.md`
2. `/tmp/ENGINEERING_SPEC_v2_cpp_only.md`
3. `.agents/skills/w8-biayn-framework/SKILL.md`
4. Relevant implementation files under `src/w8_biayn/`

## Non-Negotiable Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder training path as the active trainer.

Do not replace SkyRL/rLLM with another framework.

Allowed upstream use:

- PIE: source C++ slower-to-faster pairs, official tests, and eval/data lessons.
- LearningOpt PIE: gem5 reference and calibration environment.
- SuperCoder: schema, correctness/eval lessons, and examples.
- SkyRL/rLLM: actual SFT and GRPO training stack.

Use `uv run w8-biayn upstreams clone` for pinned repo copies under `.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor upstream repos or data.

## Fresh-Machine Contract

A clean clone must be able to run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

If a change invalidates any command, update implementation, tests, README, this file, and `.agents/skills/w8-biayn-framework/SKILL.md` in the same logical change.

Do not rely on globally installed tools unless bootstrap installs them or `doctor` reports a clear missing prerequisite with the exact next action.

## Data Discipline

Dataset conversion is a deliverable. No one-off PIE or SuperCoder munging is allowed.

All source downloads, archive normalization, coverage measurement, task construction, SkyRL conversion, GCS upload, and GCS restore must be represented as `w8-biayn data ...` commands with tests and docs.

Full official PIE path:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full --out .w8-biayn/data/pie-full/coverage.json --report-out .w8-biayn/data/pie-full/coverage-report.json
uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full --coverage-json .w8-biayn/data/pie-full/coverage.json --out .w8-biayn/data/tasks-full --min-train 1000 --min-validation 100 --min-test 100 --force
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/skyrl-full --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl-full --gcs-prefix "gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl" --credentials .gcp-service-account.json
```

Admission gates are required:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- SkyRL bundle manifest verifies before upload.

Default schema version: `cpp-perf-v1`.

Do not overwrite a full-run dataset prefix. Use `full-official/<RUN_ID>/skyrl`; update a `latest` alias only with an explicit command or user request.

## Task Rules

Preserve PIE task discipline:

- `v0` slower C++ becomes the prompt.
- `v1` faster C++ is not shown during GRPO.
- `v1` may be used as SFT target, oracle/reference material, and coverage measurement input.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and coverage passing 95 percent line / 85 percent branch.

The prompt may include visible tests and `v0`. It must not include hidden tests or `v1`.

## Reward Rules

The reward is correctness gated:

- Invalid format is negative.
- Compile or sanitizer failure is negative.
- Timeout is negative.
- Partial tests remain below any fully correct answer.
- Fully correct answers with missing non-timeout runtime measurement get a correctness-only fallback below any measured fully correct answer.
- Fully correct answers get a base reward plus bounded runtime-efficiency.
- Child-process CPU time in nanoseconds is the fast RL reward metric.
- Wall-clock nanoseconds are recorded as diagnostics.
- gem5 is calibration/final-eval reference only.
- `w8-biayn cpp harness preflight` is required before GRPO to prove Docker runtime measurement works.

Model outputs must contain exactly one `<reasoning>...</reasoning>` block followed by exactly one fenced C++ code block. The code may start on the next line or after whitespace on the opening C++ fence line; any second code block is invalid.

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and hidden tests, then benchmarks both binaries in the same Docker sandbox with the same CPU pinning, compiler flags, and tests. Default timing uses 1 warmup and 3 measured repeats, median CPU time per test, summed across tests.

Do not add PMU, Linux perf, PERFMON, or `perf_event_paranoid` dependencies to the active reward path.

## Training Rules

Training runs through SkyPilot on GCP and delegates to SkyRL/rLLM.

Render before launch:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
```

Full training must pass explicit storage and retention:

```bash
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --accelerators A100:8 --disk-size 1024 --train-batch-size 16 --train-epochs 2 --eval-interval 50 --ckpt-interval 100 --hf-save-interval 100 --ckpt-path "$RUN_GCS/cpp-sft/ckpts" --export-path "$RUN_GCS/cpp-sft/exports" --max-ckpts-to-keep 2 --no-down-after
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports" --accelerators A100:8 --disk-size 1024 --train-batch-size 16 --n-samples-per-prompt 8 --train-epochs 3 --no-eval-before-train --eval-interval 25 --max-env-workers 128 --ckpt-interval 50 --hf-save-interval 100 --ckpt-path "$RUN_GCS/cpp-grpo/ckpts" --export-path "$RUN_GCS/cpp-grpo/exports" --max-ckpts-to-keep 2 --no-down-after
```

If an SFT attempt writes checkpoints and then fails, resume the same run with `--resume-from latest`, `--disk-size 1024` or larger, and the same `--ckpt-path`.
If a GRPO attempt writes a complete checkpoint and then fails or is canceled, resume the same run with `--resume-from latest`, `--disk-size 1024` or larger, and the same `--ckpt-path`. Multi-node GRPO uses `--num-nodes N`; rendered colocated SkyRL starts a rank-gated Ray cluster inside the GPU containers, runs `skyrl_cpp_perf_main` only on rank 0, and sets rollout engines to `N * GPUs_PER_NODE` for TP=1/DP=1 so policy and inference placement match. Multi-node GRPO must pass the CLI utilization gate: effective samples per step (`train_batch_size * n_samples_per_prompt`) must be at least 16 per GPU and `max_env_workers` must be at least the effective samples per step. For 2x[A100:8], use `--train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256`; use `--allow-low-multinode-utilization` only for a deliberate experiment. Use `--no-eval-before-train` to skip the expensive initial validation pass. Rendered GRPO uses non-batched trajectory generation so SkyRL-Gym can overlap C++ reward calls; raise `--max-env-workers` when Docker reward compilation/runtime is the bottleneck.
If the checkpoint is complete but the HF export has no model weight files, rerun `cpp-sft` as an export-only recovery with `--export-checkpoint "$RUN_GCS/cpp-sft/ckpts/global_step_N"`, `--export-path "~/exports/"`, `--disk-size 1024`, and `--no-down-after`; then verify `ops run-status` reports `artifacts.export.final_export_exists=true` and `artifacts.export.final_export.weight_object_count>0` before starting GRPO.
When GRPO or eval uses a `gs://` model export, the launcher must stage it into a path mounted inside the GPU container before invoking SkyRL or vLLM; host-only staged paths are invalid.

The GRPO entrypoint is `python -m w8_biayn.integrations.skyrl_cpp_perf_main`. It registers `cpp-perf` and delegates to SkyRL `BasePPOExp(cfg).run()`.

Rendered GRPO must run a host-side C++ runtime preflight before GCS restore, model staging, GPU image pulls, or framework installs, and must run the preflight again before `skyrl_cpp_perf_main`. GRPO rewards use Docker-outside-Docker, so rendered YAML must mount `/var/run/docker.sock` and host `/tmp` into the GPU training container.
Rendered GRPO and eval must not require PMU access or mutate `kernel.perf_event_paranoid`.

## Cloud Rules

Cloud commands must:

- support dry-run rendering before paid launches;
- use `.gcp-service-account.json` through scoped env vars;
- avoid `gcloud auth activate-service-account`;
- avoid mutating global `gcloud config`;
- avoid printing credential contents;
- render YAML into `.w8-biayn/rendered/`;
- label paid resources with `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`;
- provide operations through `w8-biayn ops status`, `w8-biayn ops run-status`, `w8-biayn ops logs`, `w8-biayn ops queue`, `w8-biayn ops cancel`, `w8-biayn ops down`, `w8-biayn ops gpus`, and `w8-biayn gcp cleanup`.

Use `w8-biayn ops run-status --run-id "$RUN_ID" --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --out ".w8-biayn/runs/$RUN_ID/status.json"` for dashboard/polling status. For reruns or cluster-size experiments, pass prior snapshots with `--baseline-status ".w8-biayn/runs/<baseline-run-id>/status.json"` so the output includes `speed_comparison`; do not answer "is it faster?" from GPU count alone. Speedup factors greater than `1.0` are faster, factors less than `1.0` are slower, and `gpu_speedup_efficiency` is speedup divided by GPU scale. A `cost_verdict` of `cost_inefficient` means more GPUs did not increase the primary comparable throughput. The JSON schema is `w8-run-status-v1` and includes dataset manifest state, per-pipeline cluster/job state, labeled GCP instances, checkpoint markers and shard completeness for the promoted `latest` checkpoint, highest checkpoint directory, active `in_progress` checkpoint upload, export readiness including final export object counts/bytes and model weight presence, recent log-derived stage/step/checkpoint/export/error signals, normalized phase/progress/resource/command fields, GRPO config (`effective_samples_per_step`, total GPUs, samples/GPU/step, reward workers), trajectory/evaluation/training throughput, GPU-normalized throughput, ETA/timing metrics, reward metrics, bottleneck verdicts from SkyRL timing, optional baseline speed comparison with GPU speedup efficiency/cost verdict, and cleanup safety. Logs include `tail_lines_requested`, `tail_lines_scanned`, and `tail_may_be_truncated`; increase `--log-tail` before relying on stage parsing when the tail may be truncated. Pass `--node-health` for opt-in read-only SSH health with GPU utilization/memory, disk free space, top processes, derived node activity, and explicit `sample_scope`; do not present head-only health as whole-cluster worker telemetry. Each backend/GCS/health check includes its command, return code, `timed_out`, and `attempt_count`; tune timeout with `--check-timeout` and retry timed-out read-only checks with `--check-retries`.

Do not put raw `sky ...` commands in docs, runbooks, or agent handoffs. Keep SkyPilot/SkyRL backend details behind the `w8-biayn` CLI so the operator DX stays stable if the backend changes.

`doctor --cloud` must check the full SkyPilot launch permission set before paid runs.

On the shared GCP account, use one active full-training cluster at a time. Run cleanup after failed or completed attempts:

```bash
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --execute
```

## Evaluation Rules

Use `cpp-eval` for base, SFT, and GRPO on the same held-out data:

```bash
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model Qwen/Qwen2.5-Coder-7B-Instruct --eval-label base --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports" --eval-label sft --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-grpo/exports" --eval-label grpo --n-samples-per-prompt 1
```

`cpp-eval` stages `gs://` model exports to local VM disk before loading vLLM.

Aggregate records with:

```bash
uv run w8-biayn eval cpp --records base=base.records.jsonl --records sft=sft.records.jsonl --records grpo=grpo.records.jsonl --out uplift-summary.json
```

Uplift claim requires GRPO to beat base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_runtime_rate=0`.

If uplift fails, clone/study SuperCoder and Microsoft/LearningOpt PIE into `/tmp`, compare filtering, prompts, reward shape, model choice, and hyperparameters, then port compatible fixes into this SkyRL/rLLM pipeline only.

## Repo Responsibilities

```text
scripts/bootstrap.sh                         bootstrap
src/w8_biayn/cli.py                          CLI
src/w8_biayn/cpp_perf/data.py                downloads, full PIE prep, manifests, cache
src/w8_biayn/cpp_perf/coverage.py            gcov coverage measurement
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/skyrl_dataset.py       SkyRL data conversion
src/w8_biayn/cpp_perf/eval.py                eval aggregation
src/w8_biayn/cpp_perf/schema.py              task and harness schema
src/w8_biayn/cpp_perf/sandbox.py             C++ Docker harness
src/w8_biayn/cpp_perf/reward.py              reward function
src/w8_biayn/integrations/cpp_perf_env.py    SkyRL env adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                             SkyRL entrypoint glue
src/w8_biayn/integrations/cpp_eval_main.py   vLLM eval and scoring
src/w8_biayn/sky_config.py                   SkyPilot rendering
src/w8_biayn/gcp_auth.py                     scoped GCP auth
src/w8_biayn/secrets.py                      credential metadata only
src/w8_biayn/constants.py                    upstream pins and defaults
src/w8_biayn/upstreams.py                    upstream clone management
src/w8_biayn/benchmarks.py                   benchmark ladder
README.md                                    user and operator docs
.agents/skills/w8-biayn-framework/SKILL.md   AI coding-agent skill
```

## Documentation Rules

Update docs in the same change when commands, setup, dataset shape, cache behavior, cloud behavior, task schema, reward logic, launch flow, benchmark protocol, or supported pipelines change.

Required doc targets:

1. `README.md`
2. Mermaid diagrams in `README.md`
3. `.agents/REPO_GUIDE.md`
4. `.agents/skills/w8-biayn-framework/SKILL.md`
5. Tests when command behavior changes

## Commit Discipline

Prefer focused commits. Each commit should include code, docs, tests, and skill updates needed for one logical change.

Never commit `.env`, `.gcp-service-account.json`, `.w8-biayn/`, `.cache/upstreams/`, PIE data, CodeNet data, SuperCoder data, gem5 outputs, logs, rendered configs, or checkpoints.

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup, CLI, cloud, or data changes, also run relevant dry checks:

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
