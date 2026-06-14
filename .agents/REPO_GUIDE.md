# Repository Development Guide

`AGENTS.md` and `CLAUDE.md` must stay symlinks to this file. Update this file once; do not fork the guidance.

## Active Project

This repository is now a C++ performance-RL project. The inherited codebase is only the baseline infrastructure for GCP/SkyPilot/SkyRL/rLLM.

Phase 1 goal: train an open-weight model that rewrites correct C++ programs to run faster while preserving behavior, then evaluate it on held-out PIE tasks against base open models and Claude baselines.

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
4. The relevant implementation files under `src/w8_biayn/`

## Non-Negotiable Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder training path as the active trainer.

Do not replace SkyRL/rLLM with another framework.

Allowed upstream use:

- PIE: source C++ slower-to-faster pairs and eval/data lessons.
- LearningOpt PIE: gem5 reference and calibration environment.
- SuperCoder: reference schema, correctness/eval lessons, and examples.
- SkyRL/rLLM: actual SFT and GRPO training stack.

If upstream code must be studied, use `uv run w8-biayn upstreams clone` for pinned repo copies under `.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor upstream repos or data into this repo.

## Fresh-Machine Contract

This repository is shipped to outside users and coding agents. A clean clone must be able to run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

If a change invalidates any command, update the implementation, tests, README, and this file in the same logical change.

Do not rely on globally installed tools unless bootstrap installs them or `doctor` reports a clear missing prerequisite with the exact next action.

## Data Discipline

Dataset conversion is a deliverable. No one-off PIE or SuperCoder munging is allowed.

All source downloads, archive normalization, test generation, coverage manifests, task construction, SkyRL conversion, GCS upload, and GCS restore must be represented as `w8-biayn data ...` commands with tests and docs.

Current repeatable path:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data supercoder download --out .w8-biayn/data/supercoder
uv run w8-biayn data pie build-tests-manifest --inputs-outputs-basepath .w8-biayn/data/pie/cases --coverage-json .w8-biayn/data/pie/coverage.json --out .w8-biayn/data/pie/tests-manifest.json
uv run w8-biayn data pie build-tasks --pairs .w8-biayn/data/pie/train.jsonl --tests-json .w8-biayn/data/pie/tests-manifest.json --out .w8-biayn/data/tasks/train --split train
uv run w8-biayn data pie build-tasks --pairs .w8-biayn/data/pie/validation.jsonl --tests-json .w8-biayn/data/pie/tests-manifest.json --out .w8-biayn/data/tasks/validation --split validation
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks --out .w8-biayn/data/skyrl
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl --credentials .gcp-service-account.json
```

Known remaining data work:

- Add CLI-backed raw PIE archive normalization if local archives do not already provide `train.jsonl`, `validation.jsonl`, and case directories.
- Add CLI-backed oracle-case generation from PIE `v1`.
- Add CLI-backed coverage measurement and gating instead of relying on a manually supplied coverage JSON.
- Add gem5 calibration/final eval commands through LearningOpt PIE.

Default schema version: `cpp-perf-v1`.

Default GCS prefix:

```text
gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/skyrl
```

SkyRL bundles must include GRPO parquet, SFT JSONL, copied task JSON, and `_w8_data_manifest.json` with schema version, sources, options, byte sizes, and checksums.

## Task Rules

Preserve PIE task discipline:

- `v0` slower C++ becomes the prompt program.
- `v1` faster C++ is not shown during GRPO.
- `v1` may be used as the SFT target, oracle generator, and reference material.
- Train/validation/test splits stay by problem.
- A task requires visible tests, hidden tests, reference performance, and at least 95 percent line / 85 percent branch coverage.

The prompt may include visible tests and `v0`. It must not include hidden tests or `v1`.

## Reward Rules

The reward is correctness gated:

- Invalid format is negative.
- Compile or sanitizer failure is negative.
- Partial tests remain below any fully correct answer.
- Fully correct answers get a base reward plus bounded instruction-count efficiency.
- `perf stat -e instructions:u` is the fast RL reward metric.
- gem5 is the calibration and final-eval reference only.
- Run `w8-biayn cpp harness preflight` before GRPO; a host without numeric `instructions:u` is invalid for training.

Model outputs must contain exactly one `<reasoning>...</reasoning>` block and exactly one fenced C++ code block.

For real scoring, the sandbox image must contain `g++`, `bash`, `taskset`, and `perf`. The CLI default is `w8-biayn-cpp-perf:latest`, built locally from `gcc:13` with `linux-perf`; pass `--image --no-build-image` only for a known-good prebuilt image.

## Training Rules

Training runs through SkyPilot on GCP and delegates to SkyRL/rLLM.

Render before launch:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json
```

Launch examples:

```bash
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --accelerators A100:8 --train-batch-size 16 --no-down-after
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --accelerators A100:8 --train-batch-size 16 --n-samples-per-prompt 4 --no-down-after
```

`launch` passes SkyPilot `--down` by default. Use `--no-down-after` for training runs that need inspection after completion.

The GRPO entrypoint is `python -m w8_biayn.integrations.skyrl_cpp_perf_main`. It registers `cpp-perf` and delegates to SkyRL `BasePPOExp(cfg).run()`.

Rendered GRPO must run the C++ perf preflight before `skyrl_cpp_perf_main`. The default GRPO launch is a one-step smoke path and must keep `trainer.ckpt_interval=-1` and `trainer.hf_save_interval=-1` unless checkpoint storage has been deliberately configured. Full runs should pass explicit `--train-epochs`, `--eval-interval`, `--ckpt-interval`, `--hf-save-interval`, `--ckpt-path`, `--export-path`, and `--max-ckpts-to-keep`.

GRPO rewards use Docker-outside-Docker. The rendered SkyPilot YAML must mount `/var/run/docker.sock` and host `/tmp` into the GPU training container.

## Cloud Rules

Cloud commands must:

- Support dry-run rendering before paid launches.
- Use `.gcp-service-account.json` through scoped env vars.
- Avoid `gcloud auth activate-service-account`.
- Avoid mutating global `gcloud config`.
- Avoid printing credential contents.
- Render YAML into `.w8-biayn/rendered/`.
- Provide operations through `w8-biayn status`, `w8-biayn logs`, and `w8-biayn down`.

`doctor --cloud` must check the full SkyPilot launch permission set before paid runs.

## Repo Responsibilities

```text
scripts/bootstrap.sh                         bootstrap
src/w8_biayn/cli.py                          CLI
src/w8_biayn/cpp_perf/data.py                data downloads/manifests/cache
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/skyrl_dataset.py       SkyRL data conversion
src/w8_biayn/cpp_perf/schema.py              task and harness schema
src/w8_biayn/cpp_perf/sandbox.py             C++ Docker harness
src/w8_biayn/cpp_perf/reward.py              reward function
src/w8_biayn/integrations/cpp_perf_env.py    SkyRL env adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                             SkyRL entrypoint glue
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
3. `.agents/REPO_GUIDE.md` when repo rules change
4. `.agents/skills/w8-biayn-framework/SKILL.md`
5. Tests, when command behavior changes

## Commit Discipline

Prefer focused commits. Each commit should include the code, docs, tests, and skill updates needed for that logical change.

Never commit `.env`, `.gcp-service-account.json`, `.w8-biayn/`, `.cache/upstreams/`, PIE data, CodeNet data, SuperCoder data, gem5 outputs, logs, rendered configs, or checkpoints.

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup, CLI, cloud, or data changes, also run the relevant dry checks:

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
