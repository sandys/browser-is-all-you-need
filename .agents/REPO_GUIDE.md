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
- Partial tests remain below any fully correct answer.
- Fully correct answers get a base reward plus bounded instruction-count efficiency.
- `perf stat -e instructions:u` is the fast RL reward metric.
- gem5 is calibration/final-eval reference only.
- `w8-biayn cpp harness preflight` is required before GRPO.

Model outputs must contain exactly one `<reasoning>...</reasoning>` block followed by exactly one fenced C++ code block.

A host without numeric `instructions:u` is invalid for GRPO. Having the `perf` binary in the image is not enough.
A GCP A100/A2 VM can return `<not supported>` for `instructions:u`; do not keep retrying GPU hosts that fail this preflight.

## Training Rules

Training runs through SkyPilot on GCP and delegates to SkyRL/rLLM.

Render before launch:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
```

Full training must pass explicit storage and retention:

```bash
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --accelerators A100:8 --train-batch-size 16 --train-epochs 2 --eval-interval 50 --ckpt-interval 100 --hf-save-interval 100 --ckpt-path "$RUN_GCS/cpp-sft/ckpts" --export-path "$RUN_GCS/cpp-sft/exports" --max-ckpts-to-keep 2 --no-down-after
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports" --accelerators A100:8 --train-batch-size 16 --n-samples-per-prompt 8 --train-epochs 3 --eval-interval 25 --ckpt-interval 50 --hf-save-interval 100 --ckpt-path "$RUN_GCS/cpp-grpo/ckpts" --export-path "$RUN_GCS/cpp-grpo/exports" --max-ckpts-to-keep 2 --no-down-after
```

The GRPO entrypoint is `python -m w8_biayn.integrations.skyrl_cpp_perf_main`. It registers `cpp-perf` and delegates to SkyRL `BasePPOExp(cfg).run()`.

Rendered GRPO must run a host-side C++ perf preflight before GCS restore, model staging, GPU image pulls, or framework installs, and must run the preflight again before `skyrl_cpp_perf_main`. GRPO rewards use Docker-outside-Docker, so rendered YAML must mount `/var/run/docker.sock` and host `/tmp` into the GPU training container.
Rendered GRPO and eval must also lower the host `kernel.perf_event_paranoid` setting to `0` before perf preflight; a default GCP value of `4` blocks `perf stat -e instructions:u` even when the sandbox image has `perf`.

## Cloud Rules

Cloud commands must:

- support dry-run rendering before paid launches;
- use `.gcp-service-account.json` through scoped env vars;
- avoid `gcloud auth activate-service-account`;
- avoid mutating global `gcloud config`;
- avoid printing credential contents;
- render YAML into `.w8-biayn/rendered/`;
- label paid resources with `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`;
- provide operations through `w8-biayn ops status`, `w8-biayn ops logs`, `w8-biayn ops queue`, `w8-biayn ops cancel`, `w8-biayn ops down`, `w8-biayn ops gpus`, and `w8-biayn gcp cleanup`.

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

Uplift claim requires GRPO to beat base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_instr_rate=0`.

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
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions --dry-run
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-eval --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn gcp cleanup --run-id rdoc --credentials .gcp-service-account.json --dry-run
```
