# SLIME GLM-4.7-Flash PIE C++ Measurement

This example mirrors `examples/slime/moonlight_cpp_perf/`, but swaps the
model to `zai-org/GLM-4.7-Flash` and uses the pinned SLIME GLM-4.7 30B-A3B
Megatron profile. It keeps the same PIE C++ data, local Docker reward harness,
stage order, evaluation aggregation, and comparison artifacts.

It does not use E2B or a hosted sandbox. C++ scoring uses the repo's local
Docker sandbox through `w8_biayn.cpp_perf.reward.compute_reward`.

## Files To Read In Order

1. `README.md`: this runbook.
2. `glm47_cpp_perf.sh`: shared stage runner and all defaults.
3. `prepare_data.sh`: builds SLIME JSONL files from validated PIE task JSON.
4. `eval_base.sh`: evaluates the base GLM-4.7-Flash checkpoint.
5. `sft.sh`: runs SLIME SFT and writes Megatron plus HuggingFace checkpoints.
6. `eval_sft.sh`: evaluates the SFT HuggingFace export on the same eval split.
7. `grpo.sh`: runs one SLIME GRPO rollout from the SFT checkpoint/export.
8. `eval_grpo.sh`: evaluates the GRPO HuggingFace export on the same eval split.
9. `compare.sh`: writes the final base/SFT/GRPO comparison JSON.

The Python bridge used by these scripts is
`src/w8_biayn/integrations/slime_cpp_perf.py`.

## Host Prerequisites

Run these commands from the repo root on the GPU host. The GPU stages must run
inside the SLIME container where `ray`, SLIME, Megatron-LM, CUDA, and Docker
access are available.

```bash
./scripts/bootstrap.sh
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn cpp harness preflight --cpu 3
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

## Smallest A100:8 80GB Sequence

Inside the SLIME container:

```bash
cd /workspace/browser-is-all-you-need

export SLIME_RUN_ID="glm47_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-small
export SLIME_HF_CHECKPOINT=/root/models/GLM-4.7-Flash
export SLIME_REF_LOAD_DIR=/root/models/GLM-4.7-Flash_torch_dist
export SLIME_WANDB_PROJECT=slime-glm47-cpp-perf
export SLIME_WANDB_GROUP="${SLIME_RUN_ID}"
```

Run the stages in this exact order:

```bash
bash examples/slime/glm47_cpp_perf/prepare_data.sh
bash examples/slime/glm47_cpp_perf/eval_base.sh
bash examples/slime/glm47_cpp_perf/sft.sh
bash examples/slime/glm47_cpp_perf/eval_sft.sh
bash examples/slime/glm47_cpp_perf/grpo.sh
bash examples/slime/glm47_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_cpp_perf/compare.sh
```

The final comparison is:

```text
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/eval/comparison.json
```

Per-stage logs and receipts are:

```text
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/run.log
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/run_receipt.txt
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/vram_usage.csv
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/stages/<stage>/vram_peak.txt
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/rollout_dumps/<label>_eval_0.pt
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/hf/sft/rollout_0/
.w8-biayn/slime/glm47-cpp-perf/runs/${SLIME_RUN_ID}/hf/grpo/rollout_0/
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
