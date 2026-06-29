# SLIME ReTool Moonlight Example

This is a repo-owned port of SLIME's upstream `examples/retool` example. It
keeps the upstream ReTool helper structure, but replaces the Qwen model wiring
with the same Moonlight-16B-A3B checkpoint pattern used by
`examples/slime/moonlight_moe_smoke`.

## Files

- `generate_with_retool.py`: custom generation function and reward hook.
- `tool_sandbox.py`: Python tool execution sandbox.
- `sft_data_processing.py`: converts `JoeYing/ReTool-SFT` to parquet.
- `rl_data_preprocess.py`: converts DAPO-Math prompts to JSONL.
- `retool_moonlight_sft.sh`: ReTool SFT launcher using Moonlight.
- `retool_moonlight_rl.sh`: ReTool GRPO launcher using Moonlight.

## Prerequisites

- Run inside the SLIME container started by `.w8-biayn/slime/run-container.sh`.
- Refresh the pinned SLIME sidecar with `uv run w8-biayn upstreams clone slime`.
- Put the Moonlight HF checkpoint at
  `/root/Moonlight-16B-A3B-Instruct`, or set `SLIME_HF_CHECKPOINT`.
- Put the converted Megatron torch_dist checkpoint at
  `/root/Moonlight-16B-A3B-Instruct_torch_dist`, or set
  `SLIME_REF_LOAD_DIR`.
- Install the example dependencies with `pip install -r examples/slime/retool/requirements.txt`.

The generated `.w8-biayn/slime/run-container.sh` mounts
`${HOST_MODELS_DIR:-$HOME/models}` at `/root/models` by default, so local model
files can be shared into the container.

## Data

SFT keeps the upstream parquet path by default:

```bash
python examples/slime/retool/sft_data_processing.py
```

RL keeps the upstream DAPO/AIME layout by default. You can override the paths
with `SLIME_PROMPT_DATA` and `SLIME_EVAL_PROMPT_DATA`.

## Run SFT

```bash
SLIME_NUM_GPUS=4 \
SLIME_HF_CHECKPOINT=/root/Moonlight-16B-A3B-Instruct \
SLIME_REF_LOAD_DIR=/root/Moonlight-16B-A3B-Instruct_torch_dist \
bash examples/slime/retool/retool_moonlight_sft.sh
```

## Run RL

```bash
SLIME_NUM_GPUS=4 \
SLIME_HF_CHECKPOINT=/root/Moonlight-16B-A3B-Instruct \
SLIME_REF_LOAD_DIR=/root/Moonlight-16B-A3B-Instruct_torch_dist \
bash examples/slime/retool/retool_moonlight_rl.sh
```

## W&B Diagnostics

The launchers enable W&B when `WANDB_API_KEY` or `WANDB_KEY` is set, when an
existing `wandb login` is detected in the container, or when
`SLIME_WANDB_PROJECT` is provided. They use deterministic run names with
`--disable-wandb-random-suffix` so related smoke runs are easy to compare.

SFT diagnostics:

```bash
SLIME_WANDB_PROJECT=slime-retool \
SLIME_WANDB_GROUP=moonlight-sft \
SLIME_WANDB_RUN_ID=moonlight-retool-sft-smoke \
bash examples/slime/retool/retool_moonlight_sft.sh
```

RL diagnostics:

```bash
SLIME_WANDB_PROJECT=slime-retool \
SLIME_WANDB_GROUP=moonlight-rl \
SLIME_WANDB_RUN_ID=moonlight-retool-rl-smoke \
bash examples/slime/retool/retool_moonlight_rl.sh
```

Optional one-time conversion:

```bash
SLIME_CONVERT_IF_MISSING=1 \
SLIME_CONVERT_NPROC=4 \
bash examples/slime/retool/retool_moonlight_rl.sh
```

Runtime state lands under `.w8-biayn/slime/retool/`.
