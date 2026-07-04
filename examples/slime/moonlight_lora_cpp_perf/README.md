# SLIME Moonlight Rank-16 LoRA PIE C++ GRPO

This lane runs the PIE C++ performance task with Kimi Moonlight through the
active SLIME/Megatron/SGLang stack, but applies LoRA rank 16 to SFT, GRPO, and
their eval stages. It reuses the same data conversion, Docker reward sandbox,
debug rollout dumps, eval aggregation, and comparison artifacts as
`examples/slime/moonlight_cpp_perf/`.

The wrapper does not edit `.cache/upstreams/slime`. Before SFT/GRPO stages it
asks the active SLIME/Megatron runtime for `--help`, resolves known LoRA flags,
and fails early if no supported LoRA rank flag is present. This is intentional:
SLIME's Megatron parser can ignore unknown arguments, so the lane must not
pretend a full-model run is a LoRA run.

## Files

1. `moonlight_lora_cpp_perf.sh`: rank-16 LoRA wrapper over the base Moonlight
   C++ runner.
2. `prepare_data.sh`: builds SLIME JSONL from admitted PIE task JSON.
3. `eval_base.sh`: evaluates the base checkpoint without LoRA.
4. `sft.sh`: runs rank-16 LoRA SFT.
5. `eval_sft.sh`: evaluates the rank-16 LoRA SFT checkpoint/export.
6. `grpo.sh`: runs rank-16 LoRA GRPO from the SFT checkpoint.
7. `eval_grpo.sh`: evaluates the rank-16 LoRA GRPO checkpoint/export.
8. `compare.sh`: writes the final base/SFT/GRPO comparison JSON.

## Container Commands

Run these inside the SLIME GPU container started by `.w8-biayn/slime/run-container.sh`:

```bash
cd /workspace/browser-is-all-you-need

export SLIME_RUN_ID="moonlight_lora16_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-full
export SLIME_HF_CHECKPOINT=/root/models/Moonlight-16B-A3B-Instruct
export SLIME_REF_LOAD_DIR=/root/models/Moonlight-16B-A3B-Instruct_torch_dist

export WANDB_API_KEY="<your-wandb-key>"
export SLIME_WANDB_PROJECT=slime-moonlight-lora-cpp-perf
export SLIME_WANDB_GROUP="${SLIME_RUN_ID}"

export SLIME_LORA_RANK=16

bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_base.sh
bash examples/slime/moonlight_lora_cpp_perf/sft.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_lora_cpp_perf/grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/compare.sh
```

Optional LoRA knobs are passed only if the runtime exposes matching flags:

```bash
export SLIME_LORA_ALPHA=32
export SLIME_LORA_DROPOUT=0.0
export SLIME_LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj
```

If your SLIME/Megatron runtime uses different LoRA flag names, provide the
runtime-native arguments directly:

```bash
export SLIME_LORA_EXTRA_ARGS="--enable-lora --lora-rank 16"
```

## Outputs

The lane writes local state under:

```text
.w8-biayn/slime/moonlight-lora-cpp-perf/runs/${SLIME_RUN_ID}/
```

The main proof artifact is:

```text
.w8-biayn/slime/moonlight-lora-cpp-perf/runs/${SLIME_RUN_ID}/eval/comparison.json
```

W&B graphs use the configured `SLIME_WANDB_PROJECT` and stable stage run ids
`${SLIME_RUN_ID}-base-eval`, `${SLIME_RUN_ID}-sft`, `${SLIME_RUN_ID}-sft-eval`,
`${SLIME_RUN_ID}-grpo`, and `${SLIME_RUN_ID}-grpo-eval`.

Each stage also writes `run.log`, `run_receipt.txt`, `vram_usage.csv`, and
`vram_peak.txt` under `stages/<stage>/`. LoRA fields appear in the stage
receipt as `lora_enabled`, `lora_rank`, `lora_extra_args`, and
`resolved_lora_extra_args`.
