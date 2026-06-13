---
name: w8-biayn-framework
description: Maintain, extend, test, and operate the w8-biayn C++ performance-RL pipeline for PIE/SuperCoder data preparation, C++ task JSON, perf/gem5 rewards, SkyRL/rLLM training, SkyPilot rendering, GCS dataset caching, and Google Cloud.
---

# w8-biayn Framework

Use this skill for any work in this repository. The project is shipped to outside users and coding agents, so every feature must be reproducible from a fresh clone.

## Source Of Truth

Read these before changing behavior:

1. `AGENTS.md`
2. `README.md`
3. `/tmp/ENGINEERING_SPEC_v2_cpp_only.md`

The active goal is C++ performance RL:

- Build runnable C++ optimization tasks from PIE.
- Convert tasks into SkyRL/rLLM training datasets.
- Train with SkyRL SFT and SkyRL GRPO.
- Reward generated C++ by format, compile/sanitizer correctness, hidden tests, and `perf stat -e instructions:u`.
- Use gem5 from LearningOpt PIE only for calibration and final evaluation.

## Hard Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder training path as the active trainer.

Do not replace SkyRL/rLLM with another framework.

Allowed use of upstream projects:

- PIE: source C++ v0 to v1 pairs, prompts, sandbox/eval lessons, and gem5 reference workflow.
- LearningOpt PIE: gem5 reference/calibration environment.
- SuperCoder: schema/eval lessons and correctness-heavy examples.
- SkyRL/rLLM: the actual SFT/GRPO training stack.

Phase 1 is C++ only. Do not reintroduce BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, or Go workflows unless the user explicitly starts a later phase.

## Repository Responsibilities

`w8-biayn` owns:

- Fresh-machine bootstrap: `scripts/bootstrap.sh`
- CLI surface: `src/w8_biayn/cli.py`
- Dataset setup: `src/w8_biayn/cpp_perf/data.py`
- PIE parsing/task construction: `src/w8_biayn/cpp_perf/pie.py`
- SkyRL dataset conversion: `src/w8_biayn/cpp_perf/skyrl_dataset.py`
- C++ task schema: `src/w8_biayn/cpp_perf/schema.py`
- Sandbox/reward: `src/w8_biayn/cpp_perf/sandbox.py`, `src/w8_biayn/cpp_perf/reward.py`
- SkyRL env/entrypoint glue: `src/w8_biayn/integrations/cpp_perf_env.py`, `src/w8_biayn/integrations/skyrl_cpp_perf_main.py`
- SkyPilot rendering: `src/w8_biayn/sky_config.py`
- Scoped GCP auth: `src/w8_biayn/gcp_auth.py`, `src/w8_biayn/secrets.py`
- Upstream pins: `src/w8_biayn/constants.py`, `src/w8_biayn/upstreams.py`
- Benchmark ladder: `src/w8_biayn/benchmarks.py`
- User docs: `README.md`

## Required User Path

A new user must be able to run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
```

Dataset preparation is a deliverable and must be command-driven:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data supercoder download --out .w8-biayn/data/supercoder
uv run w8-biayn data pie build-tests-manifest --inputs-outputs-basepath .w8-biayn/data/pie/cases --coverage-json .w8-biayn/data/pie/coverage.json --out .w8-biayn/data/pie/tests-manifest.json
uv run w8-biayn data pie build-tasks --pairs .w8-biayn/data/pie/train.jsonl --tests-json .w8-biayn/data/pie/tests-manifest.json --out .w8-biayn/data/tasks/train --split train
uv run w8-biayn data pie build-tasks --pairs .w8-biayn/data/pie/validation.jsonl --tests-json .w8-biayn/data/pie/tests-manifest.json --out .w8-biayn/data/tasks/validation --split validation
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks --out .w8-biayn/data/skyrl
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl --credentials .gcp-service-account.json
```

Training renders and launches through SkyPilot:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --accelerators A100:8
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --accelerators A100:8 --train-batch-size 16 --n-samples-per-prompt 4
```

## Data Rules

No one-off dataset munging. Every conversion must be in CLI-backed project code and documented.

Local generated data belongs under `.w8-biayn/data/` and is ignored by git.

Derived SkyRL data bundles must include:

- `grpo/train.parquet`
- `grpo/validation.parquet`
- `sft/train.jsonl`
- `sft/validation.jsonl`
- copied task JSON under `tasks/`
- `_w8_data_manifest.json` with schema version, sources, options, checksums, and byte sizes.

Default schema version: `cpp-perf-v1`.

Default GCS cache prefix:

```text
gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/skyrl
```

Cache restore must verify manifest schema and checksums before use.

Cache upload must verify the local manifest first and may create the default bucket when missing.

## PIE Task Rules

Preserve PIE discipline:

- `v0` slower code becomes the prompt.
- `v1` fast code is not shown during GRPO.
- `v1` may be used as SFT target, oracle, and reference material.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and at least 95 percent line / 85 percent branch coverage.

The prompt may include visible tests and `v0`. It must not include hidden tests or `v1`.

## Reward Rules

The reward is correctness gated:

- Invalid format is negative.
- Compile and sanitizer failures are negative.
- Partial tests stay below any fully correct answer.
- Fully correct answers get base reward plus bounded instruction-count efficiency.
- `perf stat -e instructions:u` is the RL reward metric.
- gem5 is the calibration/final-eval reference only.

Model outputs must contain exactly one `<reasoning>...</reasoning>` block and exactly one fenced C++ code block.

## Cloud Rules

All GCP/SkyPilot commands must support dry-run rendering before paid launch.

Cloud commands must:

- Use `.gcp-service-account.json` through scoped env vars.
- Avoid `gcloud auth activate-service-account`.
- Avoid mutating global `gcloud config`.
- Avoid printing credential contents.
- Render YAML into `.w8-biayn/rendered/`.
- Use `w8-biayn status`, `w8-biayn logs`, and `w8-biayn down` for operations.

GRPO reward execution uses Docker-outside-Docker: the GPU training container must mount `/var/run/docker.sock` and host `/tmp`, install or expose a Docker CLI inside the container, and pre-pull the sandbox image (`gcc:13` by default) on the host. The shared `/tmp` mount is required because the reward harness creates temporary source/test directories that the host Docker daemon bind-mounts into sandbox containers.

Training defaults use `Qwen/Qwen2.5-Coder-7B-Instruct` and `A100:8` because the current project has A100 quota. GLM-5.1/H100 can be selected with `--model` and `--accelerators` after quota and memory are confirmed.

## Documentation Rules

When commands, setup, dataset shape, cache behavior, cloud behavior, task schema, reward logic, launch flow, or supported pipelines change, update:

1. `README.md`
2. Mermaid diagrams in `README.md`
3. `AGENTS.md` if development rules changed
4. This skill
5. Tests

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For bootstrap or CLI-surface changes:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn data doctor
uv run w8-biayn benchmarks list
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json
```
