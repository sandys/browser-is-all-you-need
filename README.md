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
| SFT adapter | [`TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100`](https://huggingface.co/TokenBender/glm47-flash-pie-cpp-lora-r16-sft-h100/tree/f1ac8df367080cc040f7cf769db219ee58f20f63) | 772 MB |
| Converted TP4/PP1/EP8 base checkpoint | Created by `scripts/convert_checkpoint.sh` | Reserve 65 GB |
| LoRA checkpoint and run evidence | Adapter, native shards, logs, samples, and metrics | Reserve 2 GB per saved run |

Provision at least 250 GB of free local storage for a clean installation. This
covers the base model, converted checkpoint, unpacked training image, adapter,
run artifacts, and temporary image-download/build space. Use 500 GB or more
when retaining multiple checkpoints or evaluation generations.

## Assets

Download the exact base model, prepared dataset, and validated SFT adapter:

```bash
python3 scripts/download_assets.py model --output-root /root/models
python3 scripts/download_assets.py data
python3 scripts/download_assets.py sft
```

The base model is frozen to its Hugging Face commit. Dataset and adapter files
are additionally verified against the SHA-256 manifests published with their
repositories. The commands write:

```text
/root/models/GLM-4.7-Flash
.glm47-posttraining/assets/data
.glm47-posttraining/assets/adapters/sft
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

## Repository

```text
Dockerfile                         H100 runtime
examples/sft.sh                    canonical SFT configuration
examples/grpo.sh                   canonical GRPO configuration
examples/modal/modal_app.py        Modal 8x H100 reproduction
scripts/convert_checkpoint.sh      TP4/PP1/EP8 conversion
scripts/download_assets.py         verified Hugging Face asset download
scripts/evaluate.py                held-out generation and scoring
scripts/prepare_grpo_adapter.py    serving adapter preparation
scripts/publish_results.py         W&B results publishing
src/glm47_posttraining/            GLM-4.7 Miles integration and PIE reward
```
