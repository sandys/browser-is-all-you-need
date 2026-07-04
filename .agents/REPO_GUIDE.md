# Repository Development Guide

`AGENTS.md` and `CLAUDE.md` must stay symlinks to this file. Update this file
once; do not fork the guidance.

## Active Project

This repository is a C++ performance-RL project. The active training direction
is SLIME-based Moonlight and GLM training on PIE C++ optimization tasks.

Current goal: train open-weight models that rewrite correct C++20 programs to
run faster while preserving behavior, then prove uplift on held-out PIE tasks.

SkyRL/rLLM, SkyPilot renderers, MLflow run-status parsing, and old GCP launch
helpers are legacy compatibility/reference surfaces. Do not use them for new
active training work unless the user explicitly asks for legacy maintenance.

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
2. `ROADMAP.md`
3. `.agents/skills/w8-biayn-framework/SKILL.md`
4. Relevant implementation files under `src/w8_biayn/`

The previous `/tmp/ENGINEERING_SPEC_v2_cpp_only.md` may not exist on every
machine. Treat checked-in guidance as the active source when that file is
missing.

## Non-Negotiable Boundaries

Use SLIME, Megatron, and SGLang for active SFT/GRPO work.

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder trainer as the
active trainer.

Do not reintroduce SkyRL/rLLM as the active stack unless the user explicitly
requests a rollback or legacy compatibility task.

Allowed upstream use:

- SLIME: active SFT/GRPO framework.
- PIE: source C++ slower-to-faster pairs, official tests, and data/eval lessons.
- LearningOpt PIE: gem5 reference/calibration lessons when relevant.
- SuperCoder: schema, correctness/eval lessons, and examples only.
- SkyRL/rLLM: legacy reference only.

Use `uv run w8-biayn upstreams clone` for pinned repo copies under
`.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not
vendor upstream repos or data.

## Fresh-Machine Contract

A clean clone should support:

```bash
./scripts/bootstrap.sh
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
uv run w8-biayn slime setup
uv run w8-biayn cpp harness preflight --dry-run
```

If a change invalidates this path, update implementation, tests, README, this
file, and `.agents/skills/w8-biayn-framework/SKILL.md` in the same logical
change.

Do not rely on globally installed tools unless bootstrap installs them or
`doctor` reports a clear missing prerequisite with the exact next action.

## Data Discipline

Dataset conversion is a deliverable. No one-off PIE or SuperCoder munging is
allowed.

All source downloads, archive normalization, coverage measurement, task construction, SkyRL conversion, SLIME conversion, GCS upload, and GCS restore must be represented as `w8-biayn data ...` commands with tests and docs.

Build admitted PIE task JSON:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full --out .w8-biayn/data/pie-full/coverage.json --report-out .w8-biayn/data/pie-full/coverage-report.json
uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full --coverage-json .w8-biayn/data/pie-full/coverage.json --out .w8-biayn/data/tasks-full --min-train 1000 --min-validation 100 --min-test 100 --force
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/skyrl-full --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
# Optional explicit SLIME C++ lane; keep prompt-only JSONL plus task metadata for the SLIME reward hook.
uv run w8-biayn data slime build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/slime-pie --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl-full --gcs-prefix "gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl" --credentials .gcp-service-account.json
```

Build active SLIME JSONL through lane wrappers:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
bash examples/slime/glm47_cpp_perf/prepare_data.sh  # when the GLM lane is present
```

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- reference performance exists.

Default schema version: `cpp-perf-v1`.

## Task Rules

Preserve PIE task discipline:

- `v0` slower C++ becomes the prompt.
- `v1` faster C++ is not shown during GRPO.
- `v1` may be used as SFT target, oracle/reference material, and coverage
  measurement input.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and
  coverage passing 95 percent line / 85 percent branch.

The prompt may include visible tests and `v0`. It must not include hidden tests
or `v1`.

## Reward Rules

The reward is correctness gated:

- Invalid format is negative.
- Recoverable C++ with missing wrapper/fence format is shaped below the
  correctness-only fallback.
- Compile or sanitizer failure is negative.
- Timeout is negative.
- Partial tests remain below any fully correct answer.
- Fully correct answers with missing non-timeout runtime measurement get a
  correctness-only fallback below any measured fully correct answer.
- Fully correct answers get a base reward plus bounded runtime-efficiency.
- child-process CPU time in nanoseconds is the fast RL reward metric.
- Wall-clock nanoseconds are diagnostics.

Model outputs must contain exactly one `<reasoning>...</reasoning>` block
followed by exactly one fenced C++ code block. The code may start on the next
line or after whitespace on the opening C++ fence line; any second code block is
invalid.

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and
hidden tests, then benchmarks both binaries in the same Docker sandbox with the
same CPU pinning, compiler flags, and tests.

Do not add PMU, Linux perf, PERFMON, or `perf_event_paranoid` dependencies to
the active reward path.

## Training Rules

Active training runs through the repo-owned SLIME lane wrappers.

Moonlight C++ lane:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_cpp_perf/eval_base.sh
bash examples/slime/moonlight_cpp_perf/sft.sh
bash examples/slime/moonlight_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_cpp_perf/grpo.sh
bash examples/slime/moonlight_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_cpp_perf/compare.sh
```

Moonlight rank-16 LoRA C++ lane:

```bash
bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_base.sh
bash examples/slime/moonlight_lora_cpp_perf/sft.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_lora_cpp_perf/grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/compare.sh
```

GLM C++ lane when present:

```bash
bash examples/slime/glm47_cpp_perf/prepare_data.sh
bash examples/slime/glm47_cpp_perf/eval_base.sh
bash examples/slime/glm47_cpp_perf/sft.sh
bash examples/slime/glm47_cpp_perf/eval_sft.sh
bash examples/slime/glm47_cpp_perf/grpo.sh
bash examples/slime/glm47_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_cpp_perf/compare.sh
```

Use W&B when configured by the lane; otherwise rely on local receipts,
`run.log`, `run_receipt.txt`, `vram_usage.csv`, `vram_peak.txt`, debug rollout
dumps, and eval summaries under `.w8-biayn/slime/...`. The paid GCP GLM full
launcher is `examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py`; keep it as
a provisioning wrapper around the repo-owned GLM SLIME lane, with dry-run
rendering, scoped secrets, downloaded local artifacts, labels, and automatic
`sky.down` teardown.

For Moonlight local attention/RMSNorm compatibility, keep the `src/local.py`
Megatron layer-spec shim in sync with lane defaults.

## Legacy Surface

The following files are legacy SkyRL/rLLM/GCP control-plane surfaces. Do not
extend them for new training work unless explicitly asked:

- `src/w8_biayn/cpp_perf/skyrl_dataset.py`
- `src/w8_biayn/sky_config.py`
- `src/w8_biayn/grpo_readiness.py`
- `src/w8_biayn/run_status.py`
- `src/w8_biayn/mlflow_metrics.py`
- `src/w8_biayn/integrations/skyrl_*.py`
- `src/w8_biayn/integrations/cpp_perf_env.py`
- `src/w8_biayn/integrations/cpp_eval_main.py`
- tests named `test_skyrl_*`, `test_sky_config.py`,
  `test_grpo_readiness.py`, `test_run_status.py`, and
  `test_mlflow_metrics.py`

Do not delete legacy files unless the user explicitly asks for removal. If a
file is generated evidence rather than source, untrack it with
`git rm --cached` and ignore future copies instead of deleting the working-tree
file.

## Cloud Rules

Cloud usage is not the default active path, but any cloud helper must:

- support dry-run rendering before paid launches;
- use `.gcp-service-account.json` through scoped env vars;
- avoid `gcloud auth activate-service-account`;
- avoid mutating global `gcloud config`;
- avoid printing credential contents;
- label paid resources with project, phase, pipeline, run id, owner, and TTL
  when resources are created.

Do not infer paid-resource count from local SkyPilot executor processes. Use
explicit provider/status commands for actual resource accounting.

## Repository Map

```text
scripts/bootstrap.sh                         fresh-machine bootstrap
scripts/prepare_dapo_math_dataset.py         optional SLIME text-smoke data prep
examples/slime/moonlight_cpp_perf/           active Moonlight C++ lane
examples/slime/moonlight_lora_cpp_perf/      rank-16 LoRA Moonlight C++ lane
examples/slime/glm47_cpp_perf/               active GLM C++ lane when present
examples/slime/retool/                       Moonlight ReTool lane
examples/slime/moonlight_moe_smoke/          light Moonlight MoE smoke
examples/slime/multi_agent/                  generic text-only SLIME smoke
src/local.py                                 Moonlight Megatron local-layer shim
src/w8_biayn/cpp_perf/                       PIE task, prompt, sandbox, reward, eval code
src/w8_biayn/slime_integration/              SLIME doctor/setup/sandbox helpers
src/w8_biayn/integrations/slime_cpp_perf.py  SLIME C++ data/reward/eval bridge
src/w8_biayn/integrations/slime_train_entry.py
                                             repo-owned SLIME train entry wrapper
src/w8_biayn/integrations/slime_moonlight_hf_export.py
                                             Moonlight Megatron-to-HF export shim
```

## Documentation Rules

When commands, setup, dataset shape, cache behavior, task schema, reward logic,
launch flow, benchmark protocol, or supported active pipelines change, update:

1. `README.md`
2. `ROADMAP.md`
3. this file
4. `.agents/skills/w8-biayn-framework/SKILL.md`
5. tests when command behavior changes

Do not commit generated `RUN_REPORT*` files, report asset directories, checkpoints,
model exports, PIE data, CodeNet data, SuperCoder data, gem5 outputs, logs, or
credentials.

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
uv run w8-biayn slime doctor
uv run w8-biayn slime setup --force
uv run w8-biayn cpp harness preflight --dry-run
```
