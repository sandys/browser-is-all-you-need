# w8-biayn

`w8-biayn` is the command-and-control repository for C++ performance RL. The
current active work is SLIME-based training for Moonlight and GLM models on PIE
C++ optimization tasks.

The training target is narrow:

```text
correct but slower C++20 program -> correct and faster C++20 program
```

Correctness is mandatory. Runtime improvement only matters after the generated
program preserves behavior on visible and hidden tests.

## Active Goal

- Data: official PIE C++ slower-to-faster pairs plus official, merged, and
  generated tests.
- Task: prompt with slower PIE `v0`; generate a complete optimized C++20
  program.
- Reward: strict response format, compile and sanitizer success, visible and
  hidden tests, then bounded child-process CPU-time efficiency.
- Training: SLIME with Megatron training and SGLang rollouts. Active model
  lanes are Moonlight and GLM.
- Proof: compare base, SFT, and GRPO outputs on the same held-out PIE tasks
  with pass rate, correct-and-faster rate, mean reward, speedup, and
  missing-runtime rate.

SkyRL/rLLM, SkyPilot renderers, MLflow run-status parsing, and older GCP
training commands remain in the tree only as legacy reference/compatibility
surface. Do not use them for new active training work unless a task explicitly
asks for legacy maintenance.

Out of scope unless a later phase explicitly asks for it: BrowserGym, DOMDiff,
Harbor, WebArena, MiniWoB, AndroidWorld, Go, custom GPU kernel labs, and
unrelated performance experiments.

## Boundaries And Artifact Hygiene

Do not write a custom trainer. Use SLIME, Megatron, and SGLang for active
training work. Do not reintroduce SkyRL/rLLM as the active training stack unless
the user explicitly asks for legacy maintenance or rollback.

Do not delete local evidence files just because they are no longer tracked. If a
generated artifact is in git, use `git rm --cached` so the working-tree file
remains available, then ignore future generated copies. Generated `RUN_REPORT*`
files and report asset directories should not be committed.

## Current SLIME Lanes

Use repo-owned wrappers rather than editing `.cache/upstreams/slime` directly.

- Moonlight C++ performance lane: `examples/slime/moonlight_cpp_perf/`
- Moonlight rank-16 LoRA C++ performance lane: `examples/slime/moonlight_lora_cpp_perf/`
- GLM C++ performance lane: `examples/slime/glm47_cpp_perf/`
- Moonlight ReTool lane: `examples/slime/retool/`
- Moonlight MoE smoke: `examples/slime/moonlight_moe_smoke/`
- Generic text-only SLIME smoke: `examples/slime/multi_agent/`

The Moonlight and GLM C++ lanes reuse the project PIE task schema, prompt
builder, Docker C++ sandbox, reward function, and eval aggregation through
`src/w8_biayn/integrations/slime_cpp_perf.py`.

## Fresh Machine Setup

Run from a clean clone:

```bash
./scripts/bootstrap.sh
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
uv run w8-biayn slime setup
uv run w8-biayn cpp harness preflight --dry-run
```

Generated data, upstream clones, rendered launchers, secrets, logs,
checkpoints, model exports, run reports, and evaluation artifacts are local
state and must stay out of git.

If cloud/GCS helpers are used for a specific run, keep credentials local at
`.gcp-service-account.json`; never print credential contents or mutate global
`gcloud` configuration.

## PIE Data Workflow

Dataset conversion is a deliverable. Do not use one-off notebooks, shell
history, or untracked munging for PIE data.

Build admitted PIE task JSON:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"

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
```

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- reference performance exists;
- train/validation/test split stays by problem.

Build SLIME-ready JSONL from the admitted task JSON with the lane wrapper:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
```

For the GLM lane, use the matching wrapper under
`examples/slime/glm47_cpp_perf/` when that lane is present in the workspace.

## Task And Reward Contract

A valid task contains:

- `prompt_code`: slower correct PIE C++ `v0`;
- `oracle_solution`: faster PIE C++ `v1`, used for SFT, coverage, reference
  timing, and oracle material only;
- visible `unit_tests` and grading-only `hidden_tests`;
- `test_coverage` at or above 95 percent line and 85 percent branch;
- positive reference performance metadata;
- split `train`, `validation`, or `test`.

During GRPO, the prompt may include visible tests and `v0`. It must not include
hidden tests or `v1`.

Model outputs must contain exactly:

````text
<reasoning>...</reasoning>
```cpp
// complete optimized C++20 program
```
````

Reward order:

- unrecoverable invalid format: negative;
- recoverable C++ with missing wrapper/fence format: shaped below the
  correctness-only fallback;
- compile or sanitizer failure: negative;
- timeout: negative;
- partial tests: below any fully correct answer;
- fully correct with missing non-timeout runtime measurement: correctness-only
  fallback below any measured fully correct answer;
- fully correct: base reward plus bounded runtime-efficiency bonus.

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and
hidden tests, then benchmarks both binaries in the same Docker sandbox with the
same CPU pinning, compiler flags, and tests. Runtime reward uses
child-process CPU time in nanoseconds. Wall-clock nanoseconds are diagnostics.

Do not add PMU, Linux perf, PERFMON, or `perf_event_paranoid` dependencies to
the active reward path.

## SLIME Setup

Clone or refresh the pinned SLIME checkout:

```bash
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
```

Generate the Docker-first launcher and in-container bootstrap helper:

```bash
uv run w8-biayn slime setup
.w8-biayn/slime/run-container.sh
```

The generated launcher starts the SLIME container with this repository mounted,
mounts `/var/run/docker.sock` for the Docker reward backend, and bootstraps
SLIME with `/root/Megatron-LM` on `PYTHONPATH`.

The generated Docker launcher keeps `--ulimit stack=67108864` enabled by
default. It leaves `--ulimit memlock=-1` off because some managed GPU hosts
reject that rlimit before the container starts. On hosts that allow locked
memory, opt in with:

```bash
SLIME_DOCKER_MEMLOCK_ULIMIT=1 .w8-biayn/slime/run-container.sh
```

It also raises the in-container open-file soft limit to
`SLIME_NOFILE_SOFT_LIMIT=65536` before bootstrapping SLIME.

## Moonlight C++ Performance

Run inside the SLIME container:

```bash
cd /workspace/<repo-name>

export SLIME_RUN_ID="moonlight_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_HF_CHECKPOINT=/root/models/Moonlight-16B-A3B-Instruct
export SLIME_REF_LOAD_DIR=/root/models/Moonlight-16B-A3B-Instruct_torch_dist

bash examples/slime/moonlight_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_cpp_perf/eval_base.sh
bash examples/slime/moonlight_cpp_perf/sft.sh
bash examples/slime/moonlight_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_cpp_perf/grpo.sh
bash examples/slime/moonlight_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_cpp_perf/compare.sh
```

The lane writes local state under
`.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/`.

For the lighter Moonlight MoE smoke, use:

```bash
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

## Moonlight Rank-16 LoRA C++ Performance

The rank-16 LoRA variant wraps the active Moonlight C++ lane and applies LoRA
arguments only to SFT, GRPO, and their eval stages. It checks the active
SLIME/Megatron `--help` surface before training so unsupported LoRA flags are
not silently ignored.

Run inside the SLIME container:

```bash
export SLIME_RUN_ID="moonlight_lora16_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-full
export SLIME_HF_CHECKPOINT=/root/models/Moonlight-16B-A3B-Instruct
export SLIME_REF_LOAD_DIR=/root/models/Moonlight-16B-A3B-Instruct_torch_dist
export SLIME_LORA_RANK=16
export SLIME_WANDB_PROJECT=slime-moonlight-lora-cpp-perf
export SLIME_WANDB_GROUP="${SLIME_RUN_ID}"

bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_base.sh
bash examples/slime/moonlight_lora_cpp_perf/sft.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_lora_cpp_perf/grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/compare.sh
```

The LoRA lane writes under
`.w8-biayn/slime/moonlight-lora-cpp-perf/runs/${SLIME_RUN_ID}/` and uses W&B
when `WANDB_API_KEY` or an existing W&B login is present.

### SLIME PIE C++ Path

For the SLIME version of the PIE C++ experiment, keep the normal PIE task build
and convert those tasks to SLIME JSONL:

```bash
uv run w8-biayn data slime build \
  --tasks-dir .w8-biayn/data/tasks-full \
  --out .w8-biayn/data/slime-pie \
  --profile full-official \
  --run-id "$RUN_ID" \
  --min-train-tasks 1000 \
  --min-validation-tasks 100
```

Then launch `examples/slime/cpp_perf/run_moonlight_cpp_perf_rl.sh` with the AIME
4-GPU resource profile and the C++ reward hook:

```bash
SLIME_PROMPT_DATA=.w8-biayn/data/slime-pie/train.jsonl \
W8_BIAYN_SLIME_TASK_ROOT=.w8-biayn/data/slime-pie \
SLIME_CUSTOM_GENERATE_FUNCTION_PATH= \
SLIME_CUSTOM_RM_PATH=generate_with_cpp_perf.reward_func \
SLIME_REWARD_KEY=score \
SLIME_NUM_GPUS=4 \
SLIME_ROLLOUT_BATCH_SIZE=4 \
SLIME_N_SAMPLES_PER_PROMPT=1 \
SLIME_GLOBAL_BATCH_SIZE=4 \
SLIME_MAX_RESPONSE_LEN=256 \
SLIME_MAX_TOKENS_PER_GPU=4096 \
bash examples/slime/cpp_perf/run_moonlight_cpp_perf_rl.sh
```

`generate_with_cpp_perf.reward_func` lives under `examples/slime/cpp_perf/` and
calls the repo C++ reward harness, so the SLIME path still compiles, tests, and
benchmarks candidates against the PIE `v1` oracle. `SLIME_CUSTOM_GENERATE_FUNCTION_PATH=`
disables the Python-tool ReTool trajectory for C++; SLIME's stock one-turn
generation is used instead.

### SLIME Moonlight MoE Smoke

When the GLM lane is present, run inside the SLIME container:

```bash
cd /workspace/<repo-name>

export SLIME_RUN_ID="glm47_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_HF_CHECKPOINT=/root/models/GLM-4.7-Flash
export SLIME_REF_LOAD_DIR=/root/models/GLM-4.7-Flash_torch_dist

bash examples/slime/glm47_cpp_perf/prepare_data.sh
bash examples/slime/glm47_cpp_perf/eval_base.sh
bash examples/slime/glm47_cpp_perf/sft.sh
bash examples/slime/glm47_cpp_perf/eval_sft.sh
bash examples/slime/glm47_cpp_perf/grpo.sh
bash examples/slime/glm47_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_cpp_perf/compare.sh
```

Keep GLM defaults conservative until SGLang startup, Megatron training, and C++
reward throughput are stable on the target GPU host. For the paid one-command
GCP `H100:8` full run, use
`examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py`; it loops for capacity
in `asia-southeast1` by default, runs base eval, SFT, SFT eval, GRPO, GRPO eval,
and compare, copies W&B/local artifacts under `.w8-biayn/slime/glm47-cpp-perf/`,
and tears the SkyPilot cluster down.

## Moonlight ReTool

The ReTool example uses local JSONL prompts, SGLang rollout,
`generate_with_retool.generate`, a local Python `code_interpreter` tool, and
`generate_with_retool.reward_func`.

Runbook and launcher:

- `examples/slime/retool/README.md`
- `examples/slime/retool/retool_moonlight_rl.sh`

This lane has no E2B dependency, no external browser sandbox, and no hosted tool
service.

## Evaluation

For SLIME C++ lanes, aggregate debug rollout dumps and compare summaries through
the project SLIME bridge:

```bash
python -m w8_biayn.integrations.slime_cpp_perf aggregate-debug \
  --label base \
  --debug-rollout <path-to-debug-rollout.pt-or-jsonl> \
  --out .w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/base

python -m w8_biayn.integrations.slime_cpp_perf compare \
  --summary base=.w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/base.summary.json \
  --summary sft=.w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/sft.summary.json \
  --summary grpo=.w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/grpo.summary.json \
  --out .w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/comparison.json
```

Formal uplift requires GRPO to beat base and SFT on
`correct_and_faster_rate` and mean best reward, with no missing runtime rows.

## Repository Map

```text
scripts/bootstrap.sh                         fresh-machine bootstrap
scripts/prepare_dapo_math_dataset.py         optional SLIME text-smoke data prep
examples/slime/moonlight_cpp_perf/           active Moonlight C++ lane
examples/slime/glm47_cpp_perf/               active GLM C++ lane when present
examples/slime/retool/                       Moonlight ReTool lane
examples/slime/moonlight_moe_smoke/          light Moonlight MoE smoke
examples/slime/multi_agent/                  generic text-only SLIME smoke
src/local.py                                 Moonlight Megatron local-layer shim
src/w8_biayn/cli.py                          CLI surface
src/w8_biayn/cpp_perf/data.py                downloads, full PIE prep, manifests, cache
src/w8_biayn/cpp_perf/coverage.py            gcov coverage measurement
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/skyrl_dataset.py       SkyRL GRPO/SFT dataset builder
src/w8_biayn/cpp_perf/slime_dataset.py       SLIME prompt/metadata JSONL builder
src/w8_biayn/cpp_perf/eval.py                eval aggregation
src/w8_biayn/cpp_perf/judge.py               contest-style stdout comparison
src/w8_biayn/cpp_perf/sandbox.py             Docker compile/test/runtime harness
src/w8_biayn/cpp_perf/reward.py              correctness-gated efficiency reward
src/w8_biayn/integrations/cpp_perf_env.py    SkyRL environment adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                               SkyRL GRPO entrypoint glue
src/w8_biayn/integrations/skyrl_sft_export_checkpoint_main.py
                                               SkyRL policy checkpoint HF export recovery
src/w8_biayn/integrations/skyrl_io_patch.py    SkyRL checkpoint download compatibility patch
src/w8_biayn/integrations/skyrl_vllm_logprob_patch.py
                                               SkyRL vLLM token/logprob alignment patch
src/w8_biayn/integrations/skyrl_grpo_health_patch.py
                                               SkyRL GRPO health metric logging patch
src/w8_biayn/integrations/skyrl_startup_patch.py
                                               SkyRL startup stage logging patch
src/w8_biayn/integrations/cpp_eval_main.py   vLLM eval generation and scoring
src/w8_biayn/grpo_readiness.py               GRPO readiness and live-status guardrails
src/w8_biayn/mlflow_metrics.py               MLflow Tracking Server API/SQLite metric reader
src/w8_biayn/reporting.py                    raw Markdown/CSV/SVG run evidence reports
src/w8_biayn/run_status.py                   ops run-status JSON snapshots
src/w8_biayn/shell.py                        dry-run-aware subprocess wrapper
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

Before handing off normal code/docs work:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For SLIME setup changes, also run:

```bash
uv run w8-biayn slime doctor
uv run w8-biayn slime setup --force
uv run w8-biayn cpp harness preflight --dry-run
```

For documentation-only changes, run the skill validator and the docs guardrail
tests when practical.
