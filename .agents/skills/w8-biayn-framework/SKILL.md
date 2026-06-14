---
name: w8-biayn-framework
description: "Maintain, extend, test, document, and operate the w8-biayn C++ performance-RL pipeline: full PIE data setup, C++ task JSON, perf/gem5 rewards, SkyRL/rLLM training, SkyPilot rendering, GCS dataset caching, labeled GCP operations, and uplift evaluation."
---

# w8-biayn Framework

Use this skill for work in this repository. The repo is intended for outside users and coding agents, so every workflow must be reproducible from a fresh clone.

## Source Of Truth

Read these before changing behavior:

1. `AGENTS.md`
2. `README.md`
3. `/tmp/ENGINEERING_SPEC_v2_cpp_only.md`
4. Relevant code under `src/w8_biayn/`

Active goal:

- Build runnable C++ optimization tasks from official PIE.
- Convert tasks into SkyRL/rLLM training datasets.
- Train with SkyRL SFT and SkyRL GRPO.
- Reward generated C++ by response format, compile/sanitizer correctness, visible and hidden tests, and `perf stat -e instructions:u`.
- Evaluate base, SFT, and GRPO on held-out PIE tasks and prove uplift.
- Use gem5 from LearningOpt PIE for calibration/final eval when that lane is implemented.

## Hard Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder training path as the active trainer.

Do not replace SkyRL/rLLM with another framework.

Allowed upstream use:

- PIE: C++ v0-to-v1 data, official tests, and eval/data lessons.
- LearningOpt PIE: gem5 reference/calibration environment.
- SuperCoder: schema, correctness, and eval lessons.
- SkyRL/rLLM: actual SFT and GRPO training stack.

Phase 1 is C++ only. Do not reintroduce BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, or Go workflows unless the user explicitly starts a later phase.

Use `uv run w8-biayn upstreams clone` for pinned upstream copies under `.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor upstream repos or data.

## Repository Map

`w8-biayn` owns:

- Bootstrap: `scripts/bootstrap.sh`
- CLI: `src/w8_biayn/cli.py`
- Dataset setup and manifests: `src/w8_biayn/cpp_perf/data.py`
- Coverage measurement: `src/w8_biayn/cpp_perf/coverage.py`
- PIE parsing/task construction: `src/w8_biayn/cpp_perf/pie.py`
- SkyRL dataset conversion: `src/w8_biayn/cpp_perf/skyrl_dataset.py`
- Eval aggregation: `src/w8_biayn/cpp_perf/eval.py`
- Task schema: `src/w8_biayn/cpp_perf/schema.py`
- Sandbox/reward: `src/w8_biayn/cpp_perf/sandbox.py`, `src/w8_biayn/cpp_perf/reward.py`
- SkyRL env and entrypoint glue: `src/w8_biayn/integrations/cpp_perf_env.py`, `src/w8_biayn/integrations/skyrl_cpp_perf_main.py`
- Eval entrypoint: `src/w8_biayn/integrations/cpp_eval_main.py`
- SkyPilot rendering: `src/w8_biayn/sky_config.py`
- Scoped GCP auth: `src/w8_biayn/gcp_auth.py`, `src/w8_biayn/secrets.py`
- Upstream pins: `src/w8_biayn/constants.py`, `src/w8_biayn/upstreams.py`
- Benchmark ladder: `src/w8_biayn/benchmarks.py`
- User docs: `README.md`
- Agent guide: `.agents/REPO_GUIDE.md`

## Required User Path

A new user must be able to run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

If a change breaks this path, update code, tests, README, AGENTS/CLAUDE, and this skill together.

## Data Workflow

No one-off dataset munging. Every conversion or cleanup must be a CLI-backed project command.

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

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- bundle manifest verifies before upload.

Generated local data belongs under `.w8-biayn/data/` and is ignored by git.

Derived SkyRL bundles must include:

- `grpo/train.parquet`
- `grpo/validation.parquet`
- `sft/train.jsonl`
- `sft/validation.jsonl`
- copied task JSON under `tasks/`
- `_w8_data_manifest.json` with schema version, sources, options, checksums, and byte sizes

Default schema version: `cpp-perf-v1`.

Use full-run GCS prefixes under:

```text
gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/<RUN_ID>/skyrl
```

Do not overwrite full-run data prefixes without an explicit user request.

## PIE Task Rules

Preserve PIE discipline:

- `v0` slower code becomes the prompt.
- `v1` fast code is not shown during GRPO.
- `v1` may be used as SFT target, coverage/reference input, and oracle material.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and at least 95 percent line / 85 percent branch coverage.

The prompt may include visible tests and `v0`. It must not include hidden tests or `v1`.

## Reward Rules

Model outputs must contain exactly one `<reasoning>...</reasoning>` block followed by exactly one fenced C++ code block.

The reward is correctness gated:

- Invalid format: negative.
- Compile or sanitizer failure: negative.
- Partial tests: shaped but below any fully correct answer.
- Fully correct: base reward plus bounded instruction-count efficiency.
- `perf stat -e instructions:u`: RL reward metric.
- gem5: calibration/final-eval reference only.
- `w8-biayn cpp harness preflight`: required before GRPO to prove the host exposes a numeric instruction counter.
- A GCP A100/A2 host can still return `<not supported>` for `instructions:u`; treat that as an invalid GRPO/eval host, not as a transient SkyRL error.

For real scoring, the sandbox image must contain `g++`, `bash`, `taskset`, and `perf`. The CLI default is `w8-biayn-cpp-perf:latest`, built locally from `gcc:13` with `linux-perf`; pass `--image --no-build-image` only for a known-good prebuilt image.

## Cloud, Training, And Cleanup

All GCP/SkyPilot commands must support dry-run rendering before paid launch.

Cloud commands must:

- use `.gcp-service-account.json` through scoped env vars;
- avoid `gcloud auth activate-service-account`;
- avoid mutating global `gcloud config`;
- avoid printing credential contents;
- render YAML into `.w8-biayn/rendered/`;
- label paid resources with `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`;
- use `w8-biayn ops status`, `w8-biayn ops logs`, `w8-biayn ops queue`, `w8-biayn ops cancel`, `w8-biayn ops down`, `w8-biayn ops gpus`, and `w8-biayn gcp cleanup`.

Do not put raw `sky ...` commands in docs, runbooks, or handoffs. Treat SkyPilot/SkyRL calls as backend details behind the `w8-biayn` CLI so the operator DX stays stable if the backend changes.

Training renders and launches through SkyPilot:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --accelerators A100:8 --train-batch-size 16 --no-down-after
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --accelerators A100:8 --train-batch-size 16 --n-samples-per-prompt 8 --no-down-after
```

`launch` passes SkyPilot `--down` by default. Use `--no-down-after` only for training runs that need post-run inspection.

On the shared GCP account, run one full-training cluster at a time and clean up every failed or completed attempt:

```bash
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --execute
```

GRPO reward execution uses Docker-outside-Docker: the GPU training container must mount `/var/run/docker.sock` and host `/tmp`.
Rendered GRPO and eval must lower the host `kernel.perf_event_paranoid` setting to `0`, then run a host-side C++ perf preflight before GCS restore, model staging, GPU image pulls, or framework installs. GRPO and eval still run the preflight again inside the GPU container before reward code starts.

The GRPO entrypoint is:

```bash
python -m w8_biayn.integrations.skyrl_cpp_perf_main
```

It registers `cpp-perf` inside SkyRL and delegates to SkyRL `BasePPOExp(cfg).run()`. It is glue, not a trainer.

Full runs should pass explicit `--train-epochs`, `--eval-interval`, `--ckpt-interval`, `--hf-save-interval`, `--ckpt-path`, `--export-path`, and `--max-ckpts-to-keep`.

Training defaults use `Qwen/Qwen2.5-Coder-7B-Instruct` and `A100:8`. Eval defaults to `A100:1`.

## Evaluation Workflow

Use `cpp-eval` on the same held-out data for base, SFT, and GRPO:

```bash
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model Qwen/Qwen2.5-Coder-7B-Instruct --eval-label base --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports" --eval-label sft --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-grpo/exports" --eval-label grpo --n-samples-per-prompt 1
```

`cpp-eval` stages `gs://` model exports to local VM disk before loading vLLM.

Aggregate records:

```bash
uv run w8-biayn eval cpp --records base=base.records.jsonl --records sft=sft.records.jsonl --records grpo=grpo.records.jsonl --out uplift-summary.json
```

Uplift claim requires GRPO to beat base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_instr_rate=0`.

If uplift fails, clone/study SuperCoder and Microsoft/LearningOpt PIE into `/tmp`, compare filtering, prompts, reward shape, model choice, and hyperparameters, then port compatible fixes into this SkyRL/rLLM pipeline only.

## Documentation Rules

When commands, setup, dataset shape, cache behavior, cloud behavior, task schema, reward logic, launch flow, benchmark protocol, or supported pipelines change, update:

1. `README.md`
2. Mermaid diagrams in `README.md`
3. `.agents/REPO_GUIDE.md`
4. This skill
5. Tests when command behavior changes

## Validation

Before handing off:

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
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn ops status --credentials .gcp-service-account.json --dry-run
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions --dry-run
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-eval --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn gcp cleanup --run-id rdoc --credentials .gcp-service-account.json --dry-run
```
