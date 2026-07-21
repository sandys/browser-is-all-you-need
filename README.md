# GLM-4.7-Flash Post-Training on 8x H100

A focused Miles pipeline for supervised fine-tuning and GRPO on the PIE C++
performance task.

## Aider Polyglot C++ mode

The repository also supports the 26 C++ exercises from
`Aider-AI/polyglot-benchmark`. This is a functional-correctness, multi-file
editing mode; it does not replace or reinterpret the PIE speedup experiment.
Headers and sources are both editable, tests remain hidden from model prompts,
and reward workers compile every test guarded by `EXERCISM_RUN_ALL_TESTS` using
the benchmark's vendored Catch2 harness.

Prepare and validate the local benchmark clone:

```bash
PYTHONPATH=src python3 scripts/ingest_exercism_aider.py \
  polyglot-benchmark/cpp/exercises/practice \
  .glm47-posttraining/data/aider_tasks

PYTHONPATH=src python3 scripts/build_task_taxonomy.py \
  .glm47-posttraining/data/aider_tasks \
  data/task_taxonomy.json \
  --tokenizer /root/models/GLM-4.7-Flash

PYTHONPATH=src python3 scripts/check_aider_runtime.py \
  --tasks-dir .glm47-posttraining/data/aider_tasks \
  --backend local \
  --skip-thread-sanitizer
```

Omit `--tokenizer` for the documented lexical fallback. It is useful for
preflight checks but is not presented as a GLM token count. The default
curriculum split is 20 train, 3 validation, and 3 test exercises. Use
`--split-mode all-test` to construct an evaluation-only copy of all 26 tasks;
training on these exercises contaminates comparisons with Aider's public
leaderboard.

The local command above is a functional/ASan smoke check only. Before training,
run the same preflight with `--backend docker` and without
`--skip-thread-sanitizer`; when state tasks are present, it executes a real
state-task oracle and fails closed unless TSan actually runs cleanly.

Build the Miles JSONL data and start the Polyglot GRPO profile:

```bash
PYTHONPATH=src python3 -m glm47_posttraining.integrations.miles_polyglot_cpp build-data \
  --tasks-dir .glm47-posttraining/data/aider_tasks \
  --taxonomy-file data/task_taxonomy.json \
  --out .glm47-posttraining/assets/aider_data \
  --sampling-strategy wire \
  --wire-batch-size 32 \
  --force

bash examples/polyglot_grpo.sh
```

For one-command data preparation plus launch, use
`scripts/run_full_aider_pipeline.sh`. Docker reward workers are the security
boundary for untrusted model code. The `local` backend is intended only inside
an already isolated training container; leak detection is disabled there when
the host disallows LeakSanitizer's ptrace operations.

The cross-group calibrated-advantage formula is implemented and unit-tested in
`miles_polyglot_cpp.py`, but is deliberately not monkey-patched into an unknown
Miles release. Wire it to the exact installed Miles advantage-estimator API and
validate the loss numerics before enabling that experimental objective. The
default launch continues to use Miles' standard GRPO estimator.

Polyglot reward uses three static templates. Standard tasks activate
correctness, reasoning, memory-safety, and C++-quality rubrics. State tasks add
thread safety; performance-intensive tasks add runtime instead. `allergies`,
`bank-account`, and `knapsack` are assigned to those three templates,
respectively. The benchmark remains C++17 because that is its actual build
contract.

Each task carries a non-negative risk value for its active rubrics. The scorer
normalizes those risks to weights and computes only
`reward = sum(weight[r] * score[r])`. Empty risk metadata produces uniform
weights; omitted active risks receive zero weight when at least one risk is
specified. Invalid multi-file structure remains a hard -1 gate. Runtime is
neutral unless the harness provides measured candidate and reference CPU time;
the scorer never fabricates performance credit from wall time.

After an evaluation epoch, update risks with the documented EMA (`alpha=0.8`):

```bash
PYTHONPATH=src python3 scripts/update_rubric_risks.py \
  data/task_taxonomy.json path/to/eval.records.jsonl data/task_taxonomy.json \
  --alpha 0.8
```

The Miles reward adapter watches `MILES_CPP_RUBRIC_RISK_FILE`, falling back to
`MILES_CPP_TAXONOMY_FILE`, and reloads the sidecar after its modification time
changes. This makes weights adaptive between evaluation epochs without adding
an optimizer or modifying GRPO.

The rollout adapter forwards truncation/finish reasons to the scorer and emits
each rubric component, active template, normalized weights, the reasoning-token
estimate, sanitizer status, and diagnostic observations in reward records and
W&B tables. If the rollout API
does not supply a tokenizer-derived reasoning count, the scorer uses a stable
lexical approximation; production integrations should pass the exact count
when Miles exposes it.

Detailed references:

- [Adaptive rubric reward](docs/ADAPTIVE_CPP_RUBRIC_REWARD.md)
- [C++ RL environment](docs/CPP_RL_ENVIRONMENT.md)
- [Public release checklist](docs/PUBLIC_RELEASE_CHECKLIST.md)

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
