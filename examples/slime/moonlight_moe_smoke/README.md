# SLIME Moonlight MoE Smoke

This is the smallest repo-owned SLIME MoE bring-up path: Moonlight-16B-A3B
INT4, four local math prompts, one rollout, one sample per prompt, short
responses, and no E2B/browser/DAPO/W&B dependency by default.

## Prerequisites

- Run from a 4x A100 80 GB node, preferably inside the SLIME container started
  by `.w8-biayn/slime/run-container.sh`.
- Clone or refresh the pinned SLIME sidecar with `uv run w8-biayn upstreams clone slime`.
- Put the Moonlight INT4 HF checkpoint at
  `/root/Moonlight-16B-A3B-Instruct-INT4`, or set `SLIME_HF_CHECKPOINT`.
- Put the converted Megatron torch_dist checkpoint at
  `/root/Moonlight-16B-A3B-Instruct-INT4_torch_dist`, or set
  `SLIME_REF_LOAD_DIR`.

The script can convert the HF checkpoint when `SLIME_CONVERT_IF_MISSING=1`,
but keeping conversion explicit makes the smoke run easier to reason about.

## Run

```bash
cd /workspace/<repo-name>

SLIME_NUM_GPUS=4 \
SLIME_NUM_ROLLOUT=1 \
SLIME_ROLLOUT_BATCH_SIZE=4 \
SLIME_N_SAMPLES_PER_PROMPT=1 \
SLIME_MAX_RESPONSE_LEN=128 \
SLIME_MAX_TOKENS_PER_GPU=1024 \
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

Optional one-time conversion:

```bash
SLIME_CONVERT_IF_MISSING=1 \
SLIME_CONVERT_NPROC=4 \
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

DeepEP is off by default because it depends on the host/container communication
stack. Enable it only after the all-to-all smoke is healthy:

```bash
SLIME_ENABLE_DEEPEP=1 \
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

Runtime state lands under `.w8-biayn/slime/moonlight-16b-a3b-int4-smoke/`.
Each run writes `run.log`, `vram_usage.csv`, and `vram_peak.txt` under its
timestamped run directory. Use `vram_peak.txt` as the quick peak-VRAM receipt
for the smoke.
