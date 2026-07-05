# SLIME GLM-4.7-Flash Agentic PIE C++ Lane (SWE-agent, file-state scored)

This lane trains `zai-org/GLM-4.7-Flash` on the PIE C++ optimization task as a
multi-turn **agentic** loop: SWE-agent edits `candidate.cpp` over many turns and
the **final file** is graded (compile + visible/hidden tests + child-CPU-ns
benchmark), never the raw model text. This dissolves the thinking-model
truncation problem -- a `<think>` block that overruns one turn's budget no longer
zeroes the reward, because the file the agent leaves behind is what is scored.

It uses **SWE-agent** (not claude-code, not the codex CLI) and the repo's own
hardened Docker grader (not E2B). SLIME's OpenAIAdapter + TrajectoryManager
capture the served model's sampled tokens/logprobs; the correctness-gated reward
flows through `finish_session`, so there is no `--custom-rm-path`.

SFT stays single-turn (teacher-forced warm-start for C++ quality); base-eval,
sft-eval, GRPO, and grpo-eval are agentic.

## Files To Read In Order

1. `README.md`: this runbook.
2. `glm47_swe_agent_cpp_perf.sh`: shared stage runner and all defaults.
3. `prepare_data.sh`: builds SLIME JSONL files from validated PIE task JSON.
4. `eval_base.sh`: evaluates the base GLM-4.7-Flash checkpoint.
5. `sft.sh`: runs SLIME SFT and writes Megatron plus HuggingFace checkpoints.
6. `eval_sft.sh`: evaluates the SFT HuggingFace export on the same eval split.
7. `grpo.sh`: runs one SLIME GRPO rollout from the SFT checkpoint/export.
8. `eval_grpo.sh`: evaluates the GRPO HuggingFace export on the same eval split.
9. `compare.sh`: writes the final base/SFT/GRPO comparison JSON.

The Python pieces these scripts wire in:
`src/w8_biayn/integrations/slime_swe_agent_cpp_perf.py` (the `generate` hook that
serves the adapter, runs SWE-agent, grades the final file, and flows reward),
`src/w8_biayn/integrations/swe_agent_driver.py` (single-instance SWE-agent run +
`candidate.cpp` extraction), and `src/w8_biayn/integrations/slime_cpp_perf.py`
(dataset build + eval aggregation, reused unchanged). Rollout-worker runtime deps
and how SWE-agent is installed (editable git clone, not pip) are documented in
`requirements.txt`.

## Host Prerequisites

Run these commands from the repo root on the GPU host. The GPU stages must run
inside the SLIME container where `ray`, SLIME, Megatron-LM, CUDA, and Docker
access are available.

```bash
./scripts/bootstrap.sh
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn cpp harness preflight --cpu 3
```

Build the SWE-agent edit-loop image (the C++ sandbox image plus git and a baked
`swe-rex` server) and install SWE-agent as an editable clone (it is **not**
pip-installable — its `config/`+`tools/` dirs are unpackaged repo-root siblings).
The lane does both automatically before each agentic stage; this is only for a
manual/offline check:

```bash
uv run w8-biayn cpp harness swe-image --build-base
git clone https://github.com/SWE-agent/SWE-agent.git .cache/sweagent
git -C .cache/sweagent checkout 5f40e63360d654adcd91e30ed11473389bc4909b
python -m pip install -e .cache/sweagent
```

Create or verify PIE task JSON:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full \
  --source-root .w8-biayn/data/pie \
  --out .w8-biayn/data/pie-full \
  --force
uv run w8-biayn data pie measure-coverage \
  --prepared-root .w8-biayn/data/pie-full \
  --jobs 32
```

For the smallest infrastructure smoke, build a bounded task root:

```bash
uv run w8-biayn data pie build-full-tasks \
  --prepared-root .w8-biayn/data/pie-full \
  --coverage-json .w8-biayn/data/pie-full/coverage.json \
  --out .w8-biayn/data/tasks-small \
  --limit-per-split 50 \
  --min-train 2 \
  --min-validation 4 \
  --min-test 0 \
  --force
```

Then enter the SLIME runtime container:

```bash
uv run w8-biayn slime setup
bash .w8-biayn/slime/run-container.sh
```

## Cloud launch

A dedicated GCP launcher is not yet wired for this agentic lane. Run the stage
scripts on a GPU host inside the SLIME container (see the sequence below). The
`w8-biayn launch glm47-full` CLI launcher targets the single-turn
`glm47_cpp_perf` lane, not this one; wiring an agentic cloud launcher is a
follow-up once the M1 loop proof and a 1-rollout GRPO smoke are green.

## Smallest A100:8 80GB Sequence

Inside the SLIME container:

```bash
cd /workspace/browser-is-all-you-need

export SLIME_RUN_ID="glm47_swe_agent_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-small
export SLIME_HF_CHECKPOINT=/root/models/GLM-4.7-Flash
export SLIME_REF_LOAD_DIR=/root/models/GLM-4.7-Flash_torch_dist
export SLIME_WANDB_PROJECT=slime-glm47-swe-agent-cpp-perf
export SLIME_WANDB_GROUP="${SLIME_RUN_ID}"
```

Run the stages in this exact order:

```bash
bash examples/slime/glm47_swe_agent_cpp_perf/prepare_data.sh
bash examples/slime/glm47_swe_agent_cpp_perf/eval_base.sh
bash examples/slime/glm47_swe_agent_cpp_perf/sft.sh
bash examples/slime/glm47_swe_agent_cpp_perf/eval_sft.sh
bash examples/slime/glm47_swe_agent_cpp_perf/grpo.sh
bash examples/slime/glm47_swe_agent_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_swe_agent_cpp_perf/compare.sh
```

### Agentic knobs

Behavior is controlled by env vars read by the lane and forwarded to the Ray
rollout worker (defaults live in `glm47_swe_agent_cpp_perf.sh`):

- `SLIME_ROLLOUT_MAX_CONTEXT_LEN` (default 32768): multi-turn transcript budget
  the OpenAI adapter re-renders; bounds the agent's context window.
- `SLIME_GLM_REASONING_PARSER` (default `glm45`): sglang reasoning-parser that
  splits the GLM `<think>` block. Verify the exact name against the pinned sglang
  `ServerArgs` on the first smoke.
- `W8_SWE_AGENT_PER_INSTANCE_CALL_LIMIT` (default 40) and
  `W8_SWE_AGENT_TOTAL_EXECUTION_TIMEOUT` (default 1200): bound each agent run.
  Cost limiting is always disabled -- litellm cannot price a local model, so both
  cost limits are pinned to 0 in code and the loop is bounded by these instead.
- `W8_SWE_AGENT_CONCURRENCY` (default 4): concurrent SWE-agent + grader units.
- `W8_SWE_AGENT_IMAGE` (default `w8-biayn-swe-agent:latest`): the edit-loop image.

The final comparison is:

```text
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/eval/comparison.json
```

Per-stage logs and receipts are:

```text
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/run.log
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/run_receipt.txt
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/vram_usage.csv
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/vram_peak.txt
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/rollout_dumps/<label>_eval_0.pt
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/hf/sft/rollout_0/
.w8-biayn/slime/glm47-swe-agent-cpp-perf/runs/${SLIME_RUN_ID}/hf/grpo/rollout_0/
```

## GLM Defaults

This runner intentionally differs from the Moonlight example only where the
model stack requires it:

- model: `zai-org/GLM-4.7-Flash`
- local HF path: `/root/models/GLM-4.7-Flash`
- Megatron torch-dist path: `/root/models/GLM-4.7-Flash_torch_dist`
- SLIME model args: `scripts/models/glm4.7-30B-A3B.sh`
- GPUs: `8`
- tensor/pipeline/context/expert parallelism: TP `2`, PP `2`, CP `2`, EP `8`
- sequence length: `2048`
- dynamic batching: on, `SLIME_MAX_TOKENS_PER_GPU=8192`
- SGLang memory fraction: `0.70`
- SGLang DP attention and DP LM head: on
- SGLang speculative EAGLE mode: on
- GLM MTP training args: on
- optimizer CPU offload: on for SFT and GRPO
- DeepEP: off by default for first GCP A100 smoke
- standalone HF exporter: on, using SLIME's generic `convert_torch_dist_to_hf.py`
- inline SLIME `--save-hf` exporter: off

These are deliberately conservative for a first A100:8 80GB run. If SGLang
KV-cache allocation fails, reduce `SLIME_SGLANG_MEM_FRACTION` first. If the
host has enough RAM and the GLM smoke is stable, `SLIME_GLM_ENABLE_DEEPEP=1`
can be tested as a separate throughput experiment.

## Estimated Runtime

For the small smoke on a warm A100:8 80GB host:

- `prepare_data.sh`: seconds to 2 minutes
- `eval_base.sh`: 20 to 60 minutes
- `sft.sh`: 45 minutes to 2 hours
- `eval_sft.sh`: 20 to 60 minutes
- `grpo.sh`: 45 minutes to 2.5 hours
- `eval_grpo.sh`: 20 to 60 minutes
- `compare.sh`: seconds

Cold model download plus HF-to-Megatron conversion can add 1 to 4 hours. Full
PIE coverage/task construction can add several hours if not already present.

## Scaling Knobs

Increase these only after the small run is green and every stage receipt has
`status=0`:

```bash
export SLIME_CPP_TRAIN_LIMIT=128
export SLIME_CPP_EVAL_LIMIT=64
export SLIME_GRPO_NUM_ROLLOUT=8
export SLIME_GRPO_SKIP_WEIGHT_UPDATE=0
export SLIME_GRPO_ROLLOUT_BATCH_SIZE=4
export SLIME_GRPO_GLOBAL_BATCH_SIZE=4
export SLIME_GRPO_N_SAMPLES_PER_PROMPT=1
export SLIME_EVAL_N_SAMPLES_PER_PROMPT=2
```

For the scaled profile, start from a bigger task root:

```bash
uv run w8-biayn data pie build-full-tasks \
  --prepared-root .w8-biayn/data/pie-full \
  --coverage-json .w8-biayn/data/pie-full/coverage.json \
  --out .w8-biayn/data/tasks-medium \
  --limit-per-split 200 \
  --min-train 128 \
  --min-validation 64 \
  --min-test 0 \
  --force
```

Then set:

```bash
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-medium
```

The scaled run is expected to take roughly 10 to 30 hours on A100:8 80GB after
data and model prep are warm. Do not treat a completed launch as proof of
learning; the proof is the final `comparison.json` over the same eval split.

## Failure Checks

- Missing `GLM-4.7-Flash`: the runner downloads `zai-org/GLM-4.7-Flash`
  unless `SLIME_DOWNLOAD_HF_CHECKPOINT=0`.
- Missing Megatron torch-dist checkpoint: the runner converts the HF checkpoint
  to `${SLIME_REF_LOAD_DIR}` unless `SLIME_CONVERT_IF_MISSING=0`.
- SGLang or Transformers cannot load GLM-4.7-Flash: refresh the pinned SLIME
  container/upstream and verify the container has a recent GLM-capable SGLang
  and Transformers build.
- SGLang OOM during startup: reduce `SLIME_SGLANG_MEM_FRACTION` from `0.70` to
  `0.60`, then retry the failed stage with a fresh Ray cleanup.
- GRPO with multiple rollouts fails before/after weight sync: return to
  `SLIME_GRPO_NUM_ROLLOUT=1` and `SLIME_GRPO_SKIP_WEIGHT_UPDATE=1` until the
  one-rollout smoke is green.
- SFT or GRPO actor update finishes but dies before saving: keep
  `SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP=1` and
  `SLIME_GRPO_SKIP_FINAL_TRAIN_SLEEP=1`.
- Docker sandbox failures: run `uv run w8-biayn doctor --cpp-perf` on the host
  and verify the SLIME container has Docker socket access.
