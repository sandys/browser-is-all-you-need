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
   `base.summary.json` using SLIME rollout-only mode.
5. `sft.sh`: runs SLIME SFT and writes Megatron plus HuggingFace checkpoints.
6. `eval_sft.sh`: evaluates the SFT HuggingFace export on the same eval split.
7. `grpo.sh`: runs one SLIME GRPO rollout from the SFT checkpoint/export.
8. `eval_grpo.sh`: evaluates the GRPO HuggingFace export on the same eval split.
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

For the smallest infrastructure smoke, build a bounded task root instead of
the full task tree. It is still the same PIE C++ task format and local reward
harness; it simply keeps the selected train/eval slice small enough to iterate
quickly:

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

For the smallest smoke, set
`SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-small`
instead. The runner's own defaults then select 2 train tasks and 4 eval tasks
from that root.

If `WANDB_API_KEY` or an existing W&B login is present, each stage gets a
stable W&B run id: `${SLIME_RUN_ID}-base-eval`, `${SLIME_RUN_ID}-sft`,
`${SLIME_RUN_ID}-sft-eval`, `${SLIME_RUN_ID}-grpo`, and
`${SLIME_RUN_ID}-grpo-eval`.

## Smallest Honest 4x A100 Sequence

The default run is intentionally small but task-real:

- train tasks: `2`
- eval tasks: `4`
- eval samples per prompt: `1`
- SFT epochs: `1`
- GRPO rollouts: `1`
- SFT batch/global batch: `2`
- GRPO batch/global batch: `2`
- Moonlight parallelism: TP `2`, EP `4`
- Megatron sequence length: `1024`
- SGLang memory fraction: `0.45`
- SGLang TP memory-balance check: `off`
- Ray memory threshold: `0.999`
- Megatron-to-HF export mode: `raw`
- standalone HuggingFace exporter: `on`
- inline SLIME `--save-hf` exporter: `off`
- optimizer CPU offload: `off`
- distributed timeout: `60` minutes

The eval stages deliberately avoid SLIME's live Megatron-to-SGLang weight
update path. Instead, they run SLIME rollout-only against HuggingFace
checkpoints:

- base eval loads `SLIME_HF_CHECKPOINT`
- SFT eval loads `${RUN_ROOT}/hf/sft/rollout_0`
- GRPO eval loads `${RUN_ROOT}/hf/grpo/rollout_0`

The launcher trains with SLIME's raw/local-attention Megatron path, then exports
the saved Megatron checkpoint with a repo-owned standalone HuggingFace exporter.
That exporter wraps SLIME's torch-dist converter and adds the Moonlight
`kv_layernorm` to `kv_a_layernorm` alias needed by the DeepSeek-v3 converter.
This avoids the inline SLIME `--save-hf` exporter, which timed out in MoE
parameter collection on the 4x A100 PCIe smoke. To force the old inline path for
debugging, set `SLIME_INLINE_SAVE_HF_EXPORTS=1` and
`SLIME_STANDALONE_HF_EXPORTS=0`.

The default GRPO smoke is one rollout. It starts SGLang from the SFT
HuggingFace export and marks that rollout server as frozen, so no live weight
sync is needed before the first rollout. For multi-rollout GRPO, set
`SLIME_GRPO_SKIP_WEIGHT_UPDATE=0` only after the live update path is separately
validated on the target node.

The launcher uses the repo-owned `w8_biayn.integrations.slime_train_entry`
module instead of calling SLIME's `train.py` directly. The wrapper preserves
the SLIME loop, but the small SFT and one-rollout GRPO measurements default
`SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP=1` and
`SLIME_GRPO_SKIP_FINAL_TRAIN_SLEEP=1`: after the final actor update, the wrapper
saves the checkpoint before SLIME enters the train-offload sleep path. This
avoids a known late worker-death failure on the 4x A100 PCIe node and still
produces a normal Megatron checkpoint for the standalone HuggingFace export.
For debugging the unwrapped upstream behavior, set the relevant skip flag to
`0`.

GRPO starts from the SFT Megatron weights with a fresh optimizer/RNG state by
default (`SLIME_GRPO_LOAD_WEIGHTS_ONLY=1`). This is intentional because the SFT
smoke saves model weights only (`--no-save-optim --no-save-rng`) to keep the
artifact small and easy to export. Set `SLIME_GRPO_LOAD_WEIGHTS_ONLY=0` only if
you deliberately saved optimizer and RNG state in the SFT checkpoint.

The smallest 4x A100 80G profile defaults `SLIME_OPTIMIZER_CPU_OFFLOAD=0`.
This keeps both SFT and GRPO away from the CPU optimizer offload path, which
caused host-RAM pressure and actor death on the PCIe A100 smoke node. Override
this knob only when you are deliberately profiling a different memory tradeoff
on a machine with enough host RAM.

The launcher also defaults `SLIME_SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1`.
SGLang's TP memory-balance check can see transient Ray/Megatron allocations on
GPU 0 during SLIME rollout startup and abort even when the actual model fits.

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
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/hf/sft/rollout_0/
.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/hf/grpo/rollout_0/
```

## Scaling Knobs

Increase these only after the small run is green:

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
- GRPO fails while loading `optimizer`: keep the default
  `SLIME_GRPO_LOAD_WEIGHTS_ONLY=1`, or rerun SFT with optimizer/RNG saving
  enabled and then set `SLIME_GRPO_LOAD_WEIGHTS_ONLY=0`.
- GRPO finishes the actor update but dies before saving: keep the default
  `SLIME_GRPO_SKIP_FINAL_TRAIN_SLEEP=1`. The default wrapper saves before the
  final train-offload sleep path, which is the fragile point on the 4x A100
  measurement profile.
- SFT finishes the actor update but dies before saving: keep the default
  `SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP=1`, which uses the same save-before-final-
  sleep wrapper as GRPO.
- Missing GRPO checkpoint: run `grpo.sh` before `eval_grpo.sh`.
- Docker sandbox failures: run `uv run w8-biayn doctor --cpp-perf` on the host
  and verify the SLIME container has Docker socket access.
- SGLang exits with `memory capacity is unbalanced`: keep the default
  `SLIME_SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1`, then verify the receipt
  and `vram_peak.txt` still stay below the 80G device limit.
