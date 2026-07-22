# GLM-4.7-Flash Post-Training on 8x H100

A focused Miles pipeline for supervised fine-tuning and GRPO on the PIE C++
performance task.

The repository provides one configuration:

| Component | Configuration |
| --- | --- |
| Model | GLM-4.7-Flash |
| Hardware | 8x NVIDIA H100 80 GB with NVLink |
| Training | Miles, Megatron-Core, LoRA rank 16 |
| Parallelism | TP4 / PP1 / EP8 / ETP1 |
| Sequence length | 4,096 |
| Packed tokens per GPU | 16,384 |
| MoE dispatch | DeepEP flex |
| Rollout serving | SGLang DP8 with FlashInfer |
| Tracking | W&B scalars, samples, evaluation tables, and checkpoint manifests |

## Results

Measurements were collected on a dedicated 8x H100 80 GB node with the
configuration in this repository. Base and SFT use the same 1,259 held-out
tasks, greedy decoding, a 1,536-token response cap, and the same C++ sandbox
scorer.

| Stage | Model or adapter | Evaluation data | Pass rate | Valid format | Correct and faster | Mean successful speedup |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Base | [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67) | [`TokenBender/glm47-pie-cpp-posttraining-data`](https://huggingface.co/datasets/TokenBender/glm47-pie-cpp-posttraining-data/tree/09bc0276a0ff8ab84a8db81880ca7f739057e654) | 20.89% | 40.03% | 10.56% | 1.31x |
| SFT | [`TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100`](https://huggingface.co/TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100/tree/f1ac8df367080cc040f7cf769db219ee58f20f63) | [`TokenBender/glm47-pie-cpp-posttraining-data`](https://huggingface.co/datasets/TokenBender/glm47-pie-cpp-posttraining-data/tree/09bc0276a0ff8ab84a8db81880ca7f739057e654) | **90.79%** | **97.70%** | **28.36%** | **1.43x** |

The selected SFT profile completed four measured optimizer steps with finite
loss, a 14.88-second steady actor time, and 72,397 MiB peak memory per GPU.
The verified GRPO runtime completed rollout, reward scoring, policy update,
adapter synchronization, checkpointing, and evaluation at 6,735.5 actor
tokens/second across eight GPUs. Estimated active-MoE MFU was 2.0691%.

The W&B integration records training curves, rollout samples, evaluation
samples, reward outcomes, metric catalogs, per-rank adapter synchronization
fingerprints, and checkpoint manifests.

## Requirements

- One 8x H100 80 GB NVLink node
- Docker with NVIDIA Container Toolkit
- Access to the GLM-4.7-Flash base model
- A W&B API key for online experiment tracking

The Miles base image supplies Miles, Megatron-Core, SGLang, Ray, and the
GLM-4.7 model definition.

## Replication BOM

The published training and evaluation results were produced on one node with
eight H100 80 GB GPUs, full NVLink connectivity, 1 TiB of host memory, and
10 TB of local storage.

| Component | Exact experiment configuration | Measured size |
| --- | --- | ---: |
| Base model | [`zai-org/GLM-4.7-Flash`](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67), revision `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 62.5 GB |
| GPUs | 8x NVIDIA H100 80 GB with NVLink | 75,957 MiB peak per GPU |
| Host memory | 1 TiB installed on the experiment node | About 130 GiB run delta |
| Local storage | 10 TB installed on the experiment node | 250 GB practical clean-run footprint |
| Training image | `radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37` plus this repository's `Dockerfile`; Modal builds it directly through `examples/modal/modal_app.py` | 53.3 GB base image |
| Training and evaluation data | [`TokenBender/glm47-pie-cpp-posttraining-data`](https://huggingface.co/datasets/TokenBender/glm47-pie-cpp-posttraining-data/tree/09bc0276a0ff8ab84a8db81880ca7f739057e654) | 60 MB download; about 107 MB extracted |
| Aider shadow RL corpus v1 | [`TokenBender/glm47-aider-polyglot-cpp-shadow`](https://huggingface.co/datasets/TokenBender/glm47-aider-polyglot-cpp-shadow/tree/d8f86f752685d5ddc6cece2a08ea8851b395ee83), revision `d8f86f752685d5ddc6cece2a08ea8851b395ee83` | 253 tasks; 1,519 files; 294 KiB archive |
| SFT adapter | [`TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100`](https://huggingface.co/TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100/tree/f1ac8df367080cc040f7cf769db219ee58f20f63) | 772 MB |
| Converted TP4/PP1/EP8 base checkpoint | Created by `scripts/convert_checkpoint.sh` | Reserve 65 GB |
| LoRA checkpoint and run evidence | Adapter, native shards, logs, samples, and metrics | Reserve 2 GB per saved run |

Provision at least 250 GB of free local storage for a clean installation. This
covers the base model, converted checkpoint, unpacked training image, adapter,
run artifacts, and temporary image-download/build space. Use 500 GB or more
when retaining multiple checkpoints or evaluation generations.

## Assets

Download the exact base model, prepared PIE dataset, validated SFT adapter,
and versioned Aider RL corpus:

```bash
python3 scripts/download_assets.py model --output-root /root/models
python3 scripts/download_assets.py data
python3 scripts/download_assets.py sft
python3 scripts/download_assets.py aider-shadow
```

The base model is frozen to its Hugging Face commit. Dataset and adapter files
are additionally verified against the SHA-256 manifests published with their
repositories. The commands write:

```text
/root/models/GLM-4.7-Flash
.glm47-posttraining/assets/data
.glm47-posttraining/assets/adapters/sft
.glm47-posttraining/assets/aider-shadow/tasks/aider_polyglot_cpp_shadow
```

These revisions are pinned in `scripts/download_assets.py`; environment
variables can override them when intentionally testing a newer release.

## Modal 8x H100

The canonical Modal launcher is `examples/modal/modal_app.py`. It reproduces
the recorded machine and image configuration without requiring a separately
published project image:

| Modal setting | Value |
| --- | --- |
| GPU | `H100!:8` |
| CPU | 48 cores |
| Host memory | 256 GiB requested, 1 TiB limit |
| Timeout | 24 hours per stage |
| Base image | `radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37` |
| Runtime additions | Repository `Dockerfile`, GCC/G++ 13, `rsync`, `gawk`, `util-linux`, and `git` |
| Persistent storage | `glm47-models`, `glm47-assets`, and `glm47-runs` Modal Volumes |
| Tracking secret | Modal secret `wandb-glm47` containing `WANDB_API_KEY` |

Install the Modal client, authenticate to a workspace, and create the W&B
secret once:

```bash
python3 -m pip install "modal==1.2.6"
modal secret create wandb-glm47 WANDB_API_KEY="$WANDB_API_KEY"
```

Prepare the pinned model and assets, convert the checkpoint, and run either
training stage:

```bash
modal run examples/modal/modal_app.py::prepare
modal run examples/modal/modal_app.py::convert
modal run examples/modal/modal_app.py::sft
modal run examples/modal/modal_app.py::grpo
```

GRPO defaults to the published SFT adapter. To use a newly produced SFT
checkpoint, pass its path on the `glm47-runs` volume:

```bash
modal run examples/modal/modal_app.py::grpo \
  --adapter-path /workspace/runs/<sft-run>/checkpoints/sft_lora_r16/<adapter>
```

## Runtime

Build the aligned H100 image:

```bash
docker build -t glm47-h100-posttraining .
```

Start the container with the repository and model directory mounted:

```bash
docker run --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  -v "$PWD:/workspace/glm47-h100-posttraining" \
  -v /root/models:/root/models \
  -v "$PWD/.glm47-posttraining/assets:/workspace/assets:ro" \
  glm47-h100-posttraining \
  bash
```

Inside the container:

```bash
cd /workspace/glm47-h100-posttraining
python3 -m pip install -e .
export MILES_CPP_DATA_DIR=/workspace/assets/data
export WANDB_API_KEY=...
```

## Convert

Create the TP4/PP1/EP8 Megatron checkpoint:

```bash
bash scripts/convert_checkpoint.sh
```

The default output is:

```text
/root/models/GLM-4.7-Flash_torch_dist_tp4_pp1_ep8
```

## SFT

```bash
bash examples/sft.sh
```

Runs are written under:

```text
.glm47-posttraining/miles/glm47-h100-cpp-perf/runs/
```

Each run contains the prepared dataset, training log, VRAM trace, receipt,
LoRA checkpoints, and W&B artifact manifest.

## GRPO

Start GRPO from an SFT adapter:

```bash
export MILES_LORA_ADAPTER_PATH=/workspace/assets/adapters/sft
bash examples/grpo.sh
```

The launcher prepares a serving-compatible adapter, starts SGLang across all
eight H100s, performs C++ reward scoring, updates the LoRA policy, synchronizes
the adapter, evaluates the checkpoint, and publishes the run results.

The default GRPO schedule uses 32 prompts, 8 samples per prompt, 100 rollouts,
evaluation every 20 rollouts, and checkpointing every 10 rollouts.

## Evaluate

Run the base-model evaluation on the complete held-out set:

```bash
PYTHONPATH=src python3 scripts/evaluate.py \
  --data-dir .glm47-posttraining/assets/data \
  --model /root/models/GLM-4.7-Flash \
  --output-dir .glm47-posttraining/eval/base \
  --label base \
  --tp-size 4 \
  --batch-size 32 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 1536 \
  --attention-backend flashinfer \
  --apply-chat-template \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --score-workers 32
```

Run the same evaluation with the SFT adapter:

```bash
PYTHONPATH=src python3 scripts/evaluate.py \
  --data-dir .glm47-posttraining/assets/data \
  --model /root/models/GLM-4.7-Flash \
  --adapter .glm47-posttraining/assets/adapters/sft \
  --output-dir .glm47-posttraining/eval/sft \
  --label sft \
  --tp-size 4 \
  --batch-size 32 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 1536 \
  --attention-backend flashinfer \
  --apply-chat-template \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --lora-target-modules q_a_proj,kv_a_proj_with_mqa,o_proj,gate_proj,up_proj,down_proj \
  --experts-shared-outer-loras \
  --lora-use-virtual-experts \
  --score-workers 32
```

Evaluation writes generated samples, scored records, an aggregate summary,
quality metrics, and a run receipt under the selected output directory. Add
`--wandb-project glm47-pie-cpp-posttraining --wandb-timing-status verified`
to either command to publish the same metrics and sample tables to W&B.

### Aider Polyglot C++

This lane asks whether targeted supervised data and executable-reward training
improve GLM-4.7-Flash on repository edits. Every reported score uses the same
fixed 26-task C++ set, whole-file edit format, temperature `0.7`, top-p `1.0`,
and at most two attempts. Pass@1 measures the first answer. Pass@2 measures
whether the task passes after compiler or test feedback. The benchmark is
pinned to Aider commit `5dc9490bb35f9729ef2c95d00a19ccd30c26339c` and
Polyglot commit `7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f`.

The machine-readable ledger is
[`docs/aider_posttraining_runs.json`](docs/aider_posttraining_runs.json).

#### Progress ledger

| Stage | Training data | Pass@1 | Pass@2 | Well formed | Total tokens | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Base | No task-specific training | 0/26 | 4/26 | 26/26 | 2,212,419 | [Base model](https://huggingface.co/zai-org/GLM-4.7-Flash/tree/7dd20894a642a0aa287e9827cb1a1f7f91386b67) |
| SFT v1 | 401 source / 321 train / 320 consumed | 1/26 | 5/26 | 26/26 | 1,732,287 | [W&B](https://wandb.ai/ahm-rimer/glm47-aider-v1-sft/runs/glm47-aider-v1-sft-20260717T130336Z) |
| SFT v2 | 1,211 packaged / 1,184 consumed | 1/26 | 6/26 | 25/26 tasks | 1,582,781 | [W&B](https://wandb.ai/ahm-rimer/glm47-aider-v1-sft/runs/glm47-aider-1211-sft-20260718T192250Z) |
| SFT v3 | 530 packaged / 520 consumed | 0/26 | **7/26** | 26/26 | 1,611,304 | [W&B](https://wandb.ai/ahm-rimer/glm47-aider-v1-sft/runs/glm47-aider-complement-530-sft-20260721) |
| RL v2 | 169 train + 22 monitor; 11 updates, about 2.08 epochs | 1/26 | 6/26 | 26/26 | 1,650,420 | [Fixed-26 receipt](docs/receipts/glm47-aider-rl-v2-fixed26-run-receipt.json) |

SFT v3 is the strongest completed assisted result: 7/26 versus the base
model's 4/26, a 75% relative increase. The repaired RL run restores one
first-attempt pass and preserves valid formatting, but does not exceed SFT v3
on pass@2. This result is consistent with the working hypothesis that the lane
is limited by training duration and high-signal data coverage, but the
fixed-26 experiment does not prove that diagnosis by itself.

An earlier reward-parser trial is intentionally excluded from the table
because it produced no comparable fixed-26 receipt. It exposed terminal-token
handling that was corrected with skipped special tokens and explicit stop IDs
before RL v2.

#### Dataset and checkpoint identity

Large datasets and checkpoints are not stored in Git. A reproduction must
provide the payloads at the documented paths and match these identities before
launching:

The 253-task RL shadow corpus is package `v1` at
[`d8f86f752685d5ddc6cece2a08ea8851b395ee83`](https://huggingface.co/datasets/TokenBender/glm47-aider-polyglot-cpp-shadow/tree/d8f86f752685d5ddc6cece2a08ea8851b395ee83).
Its deterministic archive SHA-256 is
`e89d0f6b5ee78796a5519f1ab6130cc119837bedca456c1fa4b835a19e026714`,
its source-manifest SHA-256 is
`002993b94ddf85e23863e22484459df4b724d91204e5e48c37904a1f34748f00`,
and its canonical source-tree SHA-256 is
`a8bb8030f7ec287eee4f5c19146374d6722ec5af310cad632ed5938e7280f686`.
`scripts/download_assets.py aider-shadow` verifies the published checksums,
rejects unsafe archive members, and requires all 253 tasks before extraction is
accepted. The Git image explicitly excludes `rubrics/`; Modal and Lium consume
only this extracted asset path.

| Stage | Dataset identity | Checkpoint or adapter |
| --- | --- | --- |
| SFT v1 | Source archive SHA-256 `2efe714c454de7ba1c5fd523f5849b3c6c9af65e8e5ca5bf4cf438017fb1e03a`; inner 401-row JSONL SHA-256 `2ddfe6966c828007f7d6c439e51bfaf07c8959dddc4423ceadb602dc3d49517b` | `glm47-runs:/glm47-aider-v1-sft-20260717T130336Z/checkpoints/sft_lora_r16/iter_0000009/adapter` |
| SFT v2 | Train JSONL SHA-256 `13219cae85551714d4280b60600bb7ef5336dffda54698340ba40f3405ccd51b` | `glm47-runs:/glm47-aider-1211-sft-20260718T192250Z/checkpoints/sft_lora_r16/iter_0000036/adapter` |
| SFT v3 | Train JSONL SHA-256 `805aa59bbc936ee20687a293ef47d2fb9bcaee12419c6539ecd6180dfab02089` | `glm47-runs:/glm47-aider-complement-530-sft-20260721/checkpoints/sft_lora_r16/iter_0000025/adapter` |
| RL v2 | Manifest SHA-256 `a7e54c0245b97ae78f9b2fa57ff5278844585cf03004254137b6cfc8e91ef157`; train JSONL SHA-256 `b72394ab603b4b6faf22370ea70605446f112ab50c883eb61e308e2dd9ab4dd2` | Merged start adapter SHA-256 `dbea7d3e2d6603f278b94c6be134bca83bb5f0ebdc4840eb53898ec5b3affb91`; final adapter SHA-256 `046a1018b605aa29f8b8c4f2677f47ce55489105f6766155f4c009798f48abe2` |

The data root contract is:

```text
<data-root>/manifest.json
<data-root>/sft/train.jsonl                 # SFT
<data-root>/grpo/train.jsonl                # RL
<data-root>/eval/train_monitor.jsonl        # RL monitor
```

#### Reproduce SFT v1-v3

Use one epoch and disable automatic PIE data rebuilding. Set the batch size to
`32` for SFT v1 and v2, or `20` for SFT v3. Use a fresh run ID for every
attempt.

```bash
export MILES_CPP_DATA_DIR=<data-root>
export MILES_CPP_AUTO_PREPARE_DATA=0
export MILES_SFT_NUM_EPOCH=1
export MILES_ROLLOUT_BATCH_SIZE=<32-or-20>
export MILES_GLOBAL_BATCH_SIZE=<32-or-20>
export MILES_RUN_ID=<fresh-run-id>
export MILES_WANDB_PROJECT=glm47-aider-v1-sft
export MILES_WANDB_GROUP="${MILES_RUN_ID}"
export MILES_WANDB_RUN_ID="${MILES_RUN_ID}"
bash examples/sft.sh
```

The loader consumes complete batches. That is why the measured counts are 320
of 321 rows for SFT v1, 1,184 of 1,211 for SFT v2, and 520 of 530 for SFT v3.
Prepare a serving adapter with `scripts/prepare_grpo_adapter.py` before SGLang
evaluation; it preserves the source adapter and removes the auxiliary
next-token-prediction layer from the serving copy.

#### Reproduce repaired RL v2

The recorded run used source commit
`c5cb63f15166ee3fdcf52dc2a882504758a594cd`, eight H100 GPUs, rank-32 LoRA,
11 updates, rollout batch 32, eight samples per prompt, learning rate `5e-7`,
KL coefficient `0.02`, temperature `0.7`, skipped special tokens, and stop IDs
`154820 154827 154829`. W&B was deliberately offline for this run.

Place the verified inputs at:

```text
/workspace/assets/prepared-aider-169
/workspace/assets/merged-1211-530-r32
/workspace/assets/aider-shadow/tasks/aider_polyglot_cpp_shadow
/workspace/models/GLM-4.7-Flash
/workspace/models/GLM-4.7-Flash_torch_dist_tp4_pp1_ep8
```

Then run the checked-in, hash-gated specification with a fresh ID:

```bash
export GLM47_REPRO_RUN_ID=glm47-aider-grpo169-r32-2ep-repro-$(date -u +%Y%m%dT%H%M%SZ)
bash examples/lium/aider_grpo_2ep.sh
```

The script refuses to start when the RL manifest, train JSONL, or merged source
adapter does not match the recorded SHA-256 values.

#### Reproduce the fixed-26 evaluation

Build the pinned runtime image from the recorded training commit, prepare the
exact benchmark checkouts, and run two TP4 shards on one 8x H100 node:

```bash
git worktree add /workspace/glm47-c5 \
  c5cb63f15166ee3fdcf52dc2a882504758a594cd
docker build -t glm47-fixed:c5cb63f /workspace/glm47-c5

export GLM47_EVAL_ROOT=/workspace/eval-final-iter10
git clone https://github.com/Aider-AI/aider.git "${GLM47_EVAL_ROOT}/aider"
git -C "${GLM47_EVAL_ROOT}/aider" checkout \
  5dc9490bb35f9729ef2c95d00a19ccd30c26339c
git clone https://github.com/Aider-AI/polyglot-benchmark.git \
  "${GLM47_EVAL_ROOT}/aider/tmp.benchmarks/polyglot-benchmark"
git -C "${GLM47_EVAL_ROOT}/aider/tmp.benchmarks/polyglot-benchmark" checkout \
  7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
python3 -m venv "${GLM47_EVAL_ROOT}/aider-venv"
"${GLM47_EVAL_ROOT}/aider-venv/bin/pip" install -e \
  "${GLM47_EVAL_ROOT}/aider[dev]"

export GLM47_EVAL_TRAIN_RUN_ID=<training-run-id>
export GLM47_EVAL_ADAPTER_PATH=<iter-10-adapter-path>
export GLM47_EVAL_ADAPTER_SHA256=<adapter-model-sha256>
python3 examples/lium/aider_fixed26_eval.py
```

The evaluator refuses stale result directories, verifies the adapter and both
benchmark commits, runs the two 13-task shards independently, and writes a
merged receipt only after all 26 unique task identities are present.

#### Reproducibility boundary

The repository versions the code, exact configuration, hashes, progress
ledger, and final RL receipt. The answer-free 253-task shadow corpus is public
at the pinned Hugging Face revision above and is no longer stored in Git. The
base model is public at its pinned revision. A same-organization rerun can use
the recorded volume paths for all SFT datasets, the filtered 169-row RL data,
and the adapters. A fully independent rerun still needs public releases of
those three SFT datasets, the filtered RL data, and the LoRA checkpoints. The
recorded RL v2 W&B directory was offline and must be synced separately if a web
run is required.

#### Modal shadow-data alternative

The Modal path trains on up to 253 answer-free shadow tasks. The official fixed
26 remain external and evaluation-only. Prepare the pinned corpus once per
`glm47-assets` volume:

```bash
modal run examples/modal/modal_app.py::prepare_aider_shadow_asset
```

That command verifies the immutable Hugging Face revision and published
checksums before extraction. It does not need to be repeated for every run.
Routine continuations should use an already prepared, hash-gated dataset. This
one-update command starts from SFT v3, the strongest measured checkpoint:

```bash
modal run examples/modal/modal_app.py::aider_grpo \
  --run-id glm47-aider-sftv3-r16-one-update-YYYYMMDD \
  --adapter-path /workspace/runs/glm47-aider-complement-530-sft-20260721/checkpoints/sft_lora_r16/iter_0000025/adapter \
  --adapter-sha256 f1ea45bc327dc6e28d0287aea75c6b691e99d2ec2f7fdb7f07bbbf5ccd6cf36a \
  --data-dir /workspace/assets/prepared-aider-169 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --num-rollout 1
```

Rebuild from all 253 source tasks only when intentionally creating a new
prepared dataset. In that case omit `--data-dir`; the builder performs the
full per-task rubric and hidden-test validation once while materializing it.

For the fixed-26 Modal evaluator, pass the expected task count and LoRA rank
when they differ from the shadow defaults:

```bash
GLM47_EXPECTED_TRAINING_TASK_COUNT=169 \
GLM47_EVAL_LORA_RANK=32 \
modal run examples/modal/aider_eval_app.py --parallel \
  --adapter-path /runs/<run>/checkpoints/grpo_lora_r16/iter_<N>/adapter \
  --expected-adapter-sha256 <checkpoint-sha256> \
  --expected-data-manifest-sha256 <manifest-sha256> \
  --run-id <fresh-fixed26-run-id>
```

## Repository

```text
Dockerfile                         H100 runtime
examples/sft.sh                    canonical SFT configuration
examples/grpo.sh                   canonical GRPO configuration
examples/lium/aider_grpo_2ep.sh    hash-gated repaired-RL reproduction
examples/lium/aider_fixed26_eval.py  two-shard fixed-26 Lium evaluator
examples/modal/modal_app.py        Modal 8x H100 reproduction
examples/modal/aider_eval_app.py   provenance-gated fixed-26 Aider evaluation
docs/aider_posttraining_runs.json  machine-readable Aider progress ledger
docs/receipts/                     immutable measured evaluation receipts
scripts/convert_checkpoint.sh      TP4/PP1/EP8 conversion
scripts/download_assets.py         verified Hugging Face asset download
scripts/package_aider_shadow.py    deterministic Aider shadow archive builder
scripts/evaluate.py                held-out generation and scoring
scripts/prepare_grpo_adapter.py    serving adapter preparation
scripts/create_grpo_training_gate.py  Aider GRPO checkpoint provenance gate
scripts/publish_results.py         W&B results publishing
src/glm47_posttraining/            GLM-4.7 Miles integrations and rewards
```
