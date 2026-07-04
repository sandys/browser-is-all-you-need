# SLIME Multi-Agent Example

This repo-owned wrapper keeps SLIME's upstream `examples/multi_agent` flow reproducible without editing `.cache/upstreams/slime` directly.

## What It Does

- prepares `zhuzilin/dapo-math-17k` into `.w8-biayn/slime/dapo-math-17k/dapo-math-17k.jsonl`
- runs SLIME's multi-agent generator at `examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents`
- defaults to a small text-only `Qwen3-4B` smoke so the SLIME loop can be validated without a VLM memory path
- supports `Qwen3-30B-A3B` through environment overrides once the smoke is healthy

## Prepare Data

Tiny bring-up slice:

```bash
uv run python scripts/prepare_dapo_math_dataset.py \
  --out .w8-biayn/slime/dapo-math-17k \
  --limit 32
```

Full dataset:

```bash
uv run python scripts/prepare_dapo_math_dataset.py \
  --out .w8-biayn/slime/dapo-math-17k
```

## Run Training

Run inside the SLIME container started by `.w8-biayn/slime/run-container.sh`:

```bash
cd /workspace/<repo-name>

SLIME_NUM_ROLLOUT=16 \
SLIME_ROLLOUT_BATCH_SIZE=1 \
SLIME_N_SAMPLES_PER_PROMPT=2 \
SLIME_GLOBAL_BATCH_SIZE=2 \
SLIME_MAX_RESPONSE_LEN=512 \
SLIME_MAX_CONTEXT_LEN=2048 \
SLIME_MAX_TOKENS_PER_GPU=2048 \
bash examples/slime/multi_agent/run_multi_agent_text.sh
```

Optional W&B logging, assuming `wandb login` was already run inside the container:

```bash
SLIME_WANDB_PROJECT=slime-multi-agent \
SLIME_WANDB_GROUP=qwen3-4b-dapo-smoke \
SLIME_WANDB_RUN_ID=qwen3-4b-dapo-32 \
bash examples/slime/multi_agent/run_multi_agent_text.sh
```

`SLIME_MAX_TOKENS_PER_GPU` is required because the launcher uses SLIME dynamic batching. Keep `2048` for smoke runs; tune upward only after memory is stable.

Text MoE experiment after smoke:

```bash
SLIME_MODEL_NAME=Qwen3-30B-A3B \
SLIME_MODEL_HF_REPO=Qwen/Qwen3-30B-A3B \
SLIME_MODEL_ARGS_FILE=qwen3-30B-A3B.sh \
SLIME_CONVERT_NPROC=4 \
SLIME_TENSOR_MODEL_PARALLEL_SIZE=4 \
SLIME_EXPERT_MODEL_PARALLEL_SIZE=4 \
SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
bash examples/slime/multi_agent/run_multi_agent_text.sh
```
