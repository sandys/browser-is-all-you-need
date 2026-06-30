# SLIME Moonlight PIE C++ Measurement

This example measures `Moonlight-16B-A3B-Instruct` on the repo's PIE C++
performance task with the same held-out split at three points:

1. base checkpoint
2. after SLIME SFT on oracle C++ solutions
3. after SLIME GRPO from the SFT checkpoint

It does not use E2B or a hosted sandbox. C++ scoring uses the repo's local
Docker sandbox through `w8_biayn.cpp_perf.reward.compute_reward`; in other
words, the reward dependency is a local Docker sandbox, not an external
service.

## Files To Read In Order

1. `README.md`: this runbook.
2. `moonlight_cpp_perf.sh`: shared stage runner and all defaults.
3. `prepare_data.sh`: builds SLIME JSONL files from validated PIE task JSON.
4. `eval_base.sh`: runs base eval and writes `base.records.jsonl` /
   `base.summary.json`.
5. `sft.sh`: runs SLIME SFT and writes a Megatron torch-dist checkpoint.
6. `eval_sft.sh`: evaluates the SFT checkpoint on the same eval split.
7. `grpo.sh`: runs SLIME GRPO from the SFT checkpoint.
8. `eval_grpo.sh`: evaluates the GRPO checkpoint on the same eval split.
9. `compare.sh`: writes the final base/SFT/GRPO comparison JSON.

The Python bridge used by these scripts is
`src/w8_biayn/integrations/slime_cpp_perf.py`.

## Prerequisites

Run these commands from the repo root. The GPU stages must run inside the
SLIME container where `ray`, SLIME, Megatron-LM, CUDA, and Docker access are
available.

First create or verify the PIE task JSON:

```bash
./scripts/bootstrap.sh
uv run w8-biayn doctor --cpp-perf

uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full \
  --source-root .w8-biayn/data/pie \
  --out .w8-biayn/data/pie-full \
  --force
uv run w8-biayn data pie measure-coverage \
  --prepared-root .w8-biayn/data/pie-full \
  --jobs 8
uv run w8-biayn data pie build-full-tasks \
  --prepared-root .w8-biayn/data/pie-full \
  --coverage-json .w8-biayn/data/pie-full/coverage.json \
  --out .w8-biayn/data/tasks-full \
  --force
```

Then enter the SLIME runtime container using the repo's generated SLIME
container script. If the script is missing, run the repo SLIME setup first:

```bash
uv run w8-biayn slime setup
bash .w8-biayn/slime/run-container.sh
```

Inside the SLIME container, set the run identity and optional W&B credentials:

```bash
export SLIME_RUN_ID=moonlight_cpp_perf
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-full
export SLIME_HF_CHECKPOINT=/root/models/Moonlight-16B-A3B-Instruct
export SLIME_REF_LOAD_DIR=/root/models/Moonlight-16B-A3B-Instruct_torch_dist
export SLIME_WANDB_PROJECT=slime-moonlight-cpp-perf
export SLIME_WANDB_GROUP="${SLIME_RUN_ID}"
```

If `WANDB_API_KEY` or an existing W&B login is present, each stage gets a
stable W&B run id: `${SLIME_RUN_ID}-base-eval`, `${SLIME_RUN_ID}-sft`,
`${SLIME_RUN_ID}-sft-eval`, `${SLIME_RUN_ID}-grpo`, and
`${SLIME_RUN_ID}-grpo-eval`.

## Smallest Honest 4x A100 Sequence

The default run is intentionally small but task-real:

- train tasks: `16`
- eval tasks: `8`
- eval samples per prompt: `1`
- SFT epochs: `1`
- GRPO rollouts: `1`
- GRPO batch/global batch: `4`
- Moonlight parallelism: TP `2`, EP `4`
- SGLang memory fraction: `0.45`
- Ray memory threshold: `0.99`

Run the stages in this exact order:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_cpp_perf/eval_base.sh
bash examples/slime/moonlight_cpp_perf/sft.sh
bash examples/slime/moonlight_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_cpp_perf/grpo.sh
bash examples/slime/moonlight_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_cpp_perf/compare.sh
```

The final comparison is:

```text
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/comparison.json
```

Per-label records and summaries are:

```text
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/base.records.jsonl
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/base.summary.json
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/sft.records.jsonl
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/sft.summary.json
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/grpo.records.jsonl
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/eval/grpo.summary.json
```

Each GPU stage also writes:

```text
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/run.log
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/run_receipt.txt
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/vram_usage.csv
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/vram_peak.txt
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/rollout_dumps/<label>_eval_0.pt
```

## Scaling Knobs

Increase these only after the small run is green:

```bash
export SLIME_CPP_TRAIN_LIMIT=128
export SLIME_CPP_EVAL_LIMIT=64
export SLIME_GRPO_NUM_ROLLOUT=8
export SLIME_GRPO_ROLLOUT_BATCH_SIZE=4
export SLIME_GRPO_GLOBAL_BATCH_SIZE=4
export SLIME_GRPO_N_SAMPLES_PER_PROMPT=1
export SLIME_EVAL_N_SAMPLES_PER_PROMPT=2
```

For the profile gate on a paid or scarce node, record the first green setting
from `run_receipt.txt`, `vram_peak.txt`, W&B, and the summary JSON before
launching a longer run. Do not treat a completed launch as proof of learning;
the proof is the final `comparison.json` over the same eval split.

## Failure Checks

- Missing `Moonlight-16B-A3B-Instruct`: the runner downloads
  `moonshotai/Moonlight-16B-A3B-Instruct` unless
  `SLIME_DOWNLOAD_HF_CHECKPOINT=0`.
- Missing Megatron torch-dist checkpoint: the runner converts the HF checkpoint
  to `${SLIME_REF_LOAD_DIR}` unless `SLIME_CONVERT_IF_MISSING=0`.
- Missing SFT checkpoint: run `sft.sh` before `eval_sft.sh` or `grpo.sh`.
- Missing GRPO checkpoint: run `grpo.sh` before `eval_grpo.sh`.
- Docker sandbox failures: run `uv run w8-biayn doctor --cpp-perf` on the host
  and verify the SLIME container has Docker socket access.
