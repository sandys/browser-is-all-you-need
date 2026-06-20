# w8-biayn

`w8-biayn` is the command-and-control repository for Phase-1 C++ performance RL. The inherited repo is used only as baseline infrastructure for Google Cloud, SkyPilot, SkyRL, and rLLM; custom GPU kernel labs and unrelated performance experiments are not part of the active surface.

The active project is C++ only: train an open-weight model that rewrites correct C++20 programs so they run faster while preserving behavior.

Out of scope for Phase 1: BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, and Go.

## Goal

- Data: official PIE C++ slower-to-faster pairs and official/merged/generated tests.
- Task: prompt with slower C++ `v0`; generate a complete optimized C++20 program.
- Reward: strict output format, compile and sanitizer success, visible and hidden tests, then bounded CPU-time runtime efficiency.
- Training: SkyRL SFT cold start, then SkyRL GRPO through rLLM/vLLM. No custom trainer.
- Proof: compare base, SFT, and GRPO checkpoints on the same held-out PIE tasks with `pass_rate`, `correct_and_faster_rate`, mean reward, speedup, and missing-runtime rate.

PIE, LearningOpt PIE, and SuperCoder may be studied for data/eval lessons, but all repeatable work must be implemented as project CLI commands. Do not use PIE's old Trainer or any SuperCoder trainer as the active training stack.

## Architecture

```mermaid
flowchart LR
  raw[Official PIE archives] --> prep[prepare-full: splits and cases]
  prep --> cov[measure-coverage with gcov]
  cov --> tasks[coverage-gated C++ task JSON]
  tasks --> skyrl[SkyRL bundle: GRPO parquet and SFT JSONL]
  skyrl --> gcs[GCS versioned cache]
  gcs --> sft[SkyRL SFT]
  gcs --> grpo[SkyRL GRPO]
  grpo --> env[cpp-perf SkyRL env]
  env --> sandbox[Docker C++ sandbox]
  sandbox --> reward[format, compile, sanitizer, tests, runtime CPU ns]
  reward --> grpo
  gcs --> eval[cpp-eval vLLM + reward harness]
  sft --> eval
  grpo --> eval
  eval --> report[uplift report]
```

```mermaid
sequenceDiagram
  participant Dev
  participant CLI as w8-biayn
  participant GCS
  participant Sky as SkyPilot
  participant SkyRL
  participant Docker as Host Docker

  Dev->>CLI: data pie download / prepare-full / measure-coverage
  CLI->>CLI: build-full-tasks with admission report
  CLI->>CLI: data skyrl build with full-run count gates
  Dev->>CLI: data cache upload --gcs-prefix full-official/RUN_ID
  CLI->>GCS: rsync verified bundle
  Dev->>CLI: launch cpp-sft/cpp-grpo --run-id RUN_ID
  CLI->>Sky: render YAML with GCP labels and scoped credentials
  Sky->>GCS: restore data bundle
  Sky->>SkyRL: SFT or GRPO
  SkyRL->>Docker: compile, sanitize, test, runtime benchmark
  Docker-->>SkyRL: reward result
  Dev->>CLI: launch cpp-eval for base/SFT/GRPO
  CLI->>GCS: upload eval records and summaries
```

## Fresh Machine Setup

Run from a clean clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

The service-account JSON stays local. The CLI uses scoped environment variables such as `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and `CLOUDSDK_CORE_PROJECT`; it must not run `gcloud auth activate-service-account` or mutate global `gcloud config`.

Generated data, upstream clones, rendered SkyPilot YAML, secrets, logs, and checkpoints are local state and ignored by git.

## Full Official PIE Dataset

Dataset conversion is a deliverable. No one-off notebook, shell-history, or untracked-script munging is allowed.

Use one run ID for the dataset, training, eval, and cleanup:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
PROJECT_ID="$(python - <<'PY'
import json
print(json.load(open('.gcp-service-account.json'))['project_id'])
PY
)"
DATA_GCS="gs://${PROJECT_ID}-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl"
RUN_GCS="gs://${PROJECT_ID}-w8-biayn/runs/cpp-perf/${RUN_ID}"
```

Build the full official PIE task pool:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full \
  --source-root .w8-biayn/data/pie \
  --out .w8-biayn/data/pie-full \
  --force

uv run w8-biayn data pie measure-coverage \
  --prepared-root .w8-biayn/data/pie-full \
  --out .w8-biayn/data/pie-full/coverage.json \
  --report-out .w8-biayn/data/pie-full/coverage-report.json

uv run w8-biayn data pie build-full-tasks \
  --prepared-root .w8-biayn/data/pie-full \
  --coverage-json .w8-biayn/data/pie-full/coverage.json \
  --out .w8-biayn/data/tasks-full \
  --min-train 1000 \
  --min-validation 100 \
  --min-test 100 \
  --force

uv run w8-biayn data skyrl build \
  --tasks-dir .w8-biayn/data/tasks-full \
  --out .w8-biayn/data/skyrl-full \
  --profile full-official \
  --run-id "${RUN_ID}" \
  --min-train-tasks 1000 \
  --min-validation-tasks 100

uv run w8-biayn data cache upload \
  --path .w8-biayn/data/skyrl-full \
  --gcs-prefix "${DATA_GCS}" \
  --credentials .gcp-service-account.json
```

Admission gates are intentional. Paid training must not start if:

- full train tasks are fewer than 1000;
- validation or test tasks are fewer than 100;
- coverage is below 95 percent line or 85 percent branch;
- visible or hidden tests are missing;
- the SkyRL bundle manifest fails checksum verification.

The prepared data and task build write `_w8_*` manifests/reports with counts and rejection reasons.

## Task And Reward Contract

A valid task contains:

- `prompt_code`: slower correct PIE C++ `v0`;
- `oracle_solution`: faster PIE C++ `v1`, used for SFT/coverage/reference only;
- visible `unit_tests` and grading-only `hidden_tests`;
- `test_coverage` at or above 95 percent line and 85 percent branch;
- positive reference performance metadata;
- split `train`, `validation`, or `test`.

The model must return exactly:

````text
<reasoning>...</reasoning>
```cpp
// complete optimized C++20 program
```
````

The parser accepts a standard fenced block with code on the next line, or C++ code after whitespace on the same fence line. It still requires exactly one reasoning block and exactly one C++ code block.

Reward order:

- unrecoverable invalid format: `-1.0`;
- recoverable C++ with missing wrapper/fence format: shaped below the correctness-only fallback, so GRPO keeps a gradient without making bare code attractive;
- compile or sanitizer failure: negative;
- timeout: negative;
- partial tests: shaped but below fully correct;
- fully correct with missing non-timeout runtime measurement: correctness-only fallback below any measured fully correct answer;
- fully correct: `1.0 + bounded runtime-efficiency bonus`.

Before GRPO, the host must pass the Docker runtime harness preflight:

```bash
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn cpp harness preflight --cpu 3
uv run w8-biayn doctor --cpp-perf --credentials .gcp-service-account.json
```

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and hidden tests, then benchmarks both binaries in the same Docker sandbox with the same CPU pinning, compiler flags, and tests. Runtime measurement uses child-process CPU time in nanoseconds, with wall-clock nanoseconds recorded as diagnostics. The default benchmark uses 1 warmup and 3 measured repeats, takes the median per test, and sums across tests.

Rendered `cpp-grpo` and `cpp-eval` SkyPilot jobs run a host-side C++ runtime preflight before GCS restore, model staging, GPU image pulls, or framework installs, and then run the same preflight again inside the GPU container before `skyrl_cpp_perf_main` or eval scoring.

## GCP Training

Render before paid launches:

```bash
uv run w8-biayn config render cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}"

uv run w8-biayn config render cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}"
```

Launch SFT:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --train-epochs 2 \
  --eval-interval 50 \
  --ckpt-interval 100 \
  --hf-save-interval 1074 \
  --ckpt-path "${RUN_GCS}/cpp-sft/ckpts" \
  --export-path "${RUN_GCS}/cpp-sft/exports" \
  --max-ckpts-to-keep 2 \
  --no-down-after
```

For the full official SFT setup above, `1074` is the expected final step for two epochs at batch size 16 on the current PIE bundle. Keep `--ckpt-interval 100` for recovery, but use a final-step `--hf-save-interval` unless you explicitly need intermediate HF model exports; `save_hf_model` is CPU/GCS-heavy and can leave the A100s idle for tens of minutes per export. If a resumed run changes the expected final step, update `--hf-save-interval` and the `ops run-status --expected-sft-final-step` value together.
Pipeline defaults are per purpose: `cpp-smoke` uses `H100:8` and `zai-org/GLM-5.1`, `cpp-sft`/`cpp-grpo` use `A100:8` and `Qwen/Qwen2.5-Coder-7B-Instruct`, and `cpp-eval` uses `A100:1` and `Qwen/Qwen2.5-Coder-7B-Instruct`. Rendered training jobs request `128+` GB host memory; rendered eval jobs request `80+` GB so GCP single-A100 shapes are not filtered out before provisioning. Eval pins `vllm==0.6.6.post1`, `transformers==4.57.6`, and CUDA 12.4 PyTorch wheels because unpinned latest vLLM can select a CUDA 13 stack that the A2 driver cannot load. Override `--model` and `--accelerators` explicitly when a run is meant to test another model or GPU shape.

Launch GRPO after SFT produces a usable export. Use the SFT export as `--model` when available:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model "${RUN_GCS}/cpp-sft/exports/global_step_1074/policy" \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --n-samples-per-prompt 8 \
  --train-epochs 3 \
  --no-eval-before-train \
  --eval-interval 25 \
  --max-env-workers 128 \
  --ckpt-interval 50 \
  --hf-save-interval 10000 \
  --ckpt-path "${RUN_GCS}/cpp-grpo/ckpts" \
  --export-path "${RUN_GCS}/cpp-grpo/exports" \
  --max-ckpts-to-keep 8 \
  --grpo-vllm-gpu-memory-utilization 0.7 \
  --no-down-after
```

SkyRL also exports HF models at epoch boundaries and after the loop, so a large `--hf-save-interval` is not final-only; it suppresses interval exports while preserving epoch-boundary/final exports. Use a large interval such as `10000` for GRPO unless intermediate HF exports are part of the experiment, and keep checkpoint saves (`--ckpt-interval`) for resume safety.
Rendered GRPO enables a small KL anchor and entropy bonus by default (`--grpo-use-kl-loss`, `--grpo-kl-loss-coef 0.001`, `--grpo-use-entropy-loss`, `--grpo-entropy-loss-coef 0.001`) to reduce drift from the SFT/reference policy and avoid deterministic format collapse. It also sets `--grpo-vllm-gpu-memory-utilization 0.7` by default so colocated vLLM leaves enough headroom for the FSDP policy/ref workers on A100-40GB; if vLLM fails while waking the KV cache, lower this before relaunching. Keep at least several GRPO checkpoints (`--max-ckpts-to-keep 8` in the full-run example) because the most recent checkpoint may be worse than an earlier healthy checkpoint after reward over-optimization.

If a GRPO attempt is canceled or fails after a complete checkpoint, restart it with the same checkpoint path and `--resume-from latest`. Use `--disk-size 2048` or larger for multi-node GRPO resume; a 2x[A100:8] resume from `global_step_150` filled a 1024 GB boot disk during FSDP checkpoint restore and failed with `[Errno 28] No space left on device`, so the CLI rejects smaller explicit disks and defaults resumed multi-node GRPO to 2048 GB. When the SFT final step is known, pass the concrete HF policy export (`.../exports/global_step_N/policy`) as `--model`; if a `gs://.../exports` root is passed, rendered GRPO/eval jobs resolve it to the highest complete `global_step_N/policy` directory with model weights before staging it locally. For multi-node GRPO, pass `--num-nodes N`; the renderer keeps SkyRL colocated, starts a rank-gated Ray cluster inside the GPU containers, runs `skyrl_cpp_perf_main` only on rank 0, and sets rollout engines to `N * GPUs_PER_NODE` when tensor/data parallelism are `1`. It also sets `trainer.policy.fsdp_config.fsdp_size` and `trainer.ref.fsdp_config.fsdp_size` to GPUs-per-node so multi-node FSDP shards within a node (NVLink) and replicates across nodes (HSDP) instead of full-sharding the model across the slow inter-node link; without this, a model that fits on one node can train slower on two nodes than on one, so reserve multi-node for models that do not fit on a single node or generation/reward-bound steps. Multi-node GRPO is rejected unless effective samples per step (`--train-batch-size * --n-samples-per-prompt`) are at least 16 per GPU and `--max-env-workers` can cover the effective samples. For 2x[A100:8], use a tuned starting point such as `--train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256`; pass `--allow-low-multinode-utilization` only for an intentional experiment. `--micro-train-batch-size-per-gpu` defaults to `1`; try `2` only as a memory-checked policy-update tuning experiment on A100-40GB and fall back to `1` on OOM or no throughput gain. Full GRPO can skip the expensive initial validation pass with `--no-eval-before-train`; the renderer uses non-batched trajectory generation so SkyRL-Gym can overlap C++ reward calls, and `--max-env-workers` controls that Docker reward concurrency.

### Multi-Node GRPO Readiness

Before a paid multi-node GRPO launch, the CLI runs `ops grpo-readiness` against the rendered YAML and blocks critical failures. Run it directly when debugging or handing work to another operator:

```bash
uv run w8-biayn ops grpo-readiness \
  --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml \
  --out ".w8-biayn/runs/${RUN_ID}/grpo-readiness.json"
```

`ops grpo-readiness` emits `w8-grpo-readiness-v1` JSON. Static checks verify the Docker reward mounts, host and container preflights, `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`, `NCCL_DEBUG`, concrete default-route `GLOO_SOCKET_IFNAME`, `skyrl_io_patch.py`, `skyrl_vllm_logprob_patch.py`, HSDP `fsdp_size`, rollout engine count, utilization gate, resume disk, checkpoint retention, and KL/entropy settings. Gloo must receive a real interface name derived from `ip route`; do not pass NCCL's `^lo,docker,veth` exclusion syntax to `GLOO_SOCKET_IFNAME`.

For a live multi-node run, pass a status snapshot produced with `--node-health`:

```bash
uv run w8-biayn ops grpo-readiness \
  --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml \
  --status-json ".w8-biayn/runs/${RUN_ID}/status.json"
```

The live check requires `node_health.sample_scope=all_active`, sampled nodes equal active nodes, failed node probes equal zero, and sampled GPUs equal the GRPO GPU count before anyone claims the worker node is participating. If `training_health.should_stop=true`, readiness returns `overall=action_required`; follow `training_health.recommended_action` instead of continuing to poll or train.

`launch` auto-adds a run ID if omitted, but full runs should pass the same `RUN_ID` everywhere. Rendered YAML includes GCP labels under `resources.labels`: `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`.
Full training and eval default to a 1024 GB boot disk, except resumed multi-node GRPO, which defaults to and requires 2048 GB or larger because FSDP checkpoint restore needs substantial local scratch space.
Rendered SFT and GRPO containers set longer SkyRL/Ray distributed timeouts (`SKYRL_RAY_PG_TIMEOUT_IN_S=1800`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600`) so slow FSDP checkpoint restore and HF export barriers do not fail at SkyRL's 10-minute worker default. Override those environment variables only for a deliberate debugging run.
Rendered training containers also pass `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=^lo,docker,veth`, a concrete default-route `GLOO_SOCKET_IFNAME`, and `NCCL_DEBUG=WARN` into Docker so SkyRL, Ray, and vLLM use the VM network interface instead of loopback or transient container interfaces. Missing this propagation is a known multi-node reproducibility mismatch: the host shell can have valid networking while the training container still fails during Ray/NCCL/vLLM startup. Do not reuse NCCL's `^lo,docker,veth` exclusion syntax for Gloo; PyTorch Gloo expects a real interface name and will fail process-group initialization if given `^lo`.
The training container also applies idempotent SkyRL compatibility patches: `skyrl_io_patch.py` flattens cloud checkpoint directory downloads before FSDP rank-shard validation, and `skyrl_vllm_logprob_patch.py` aligns vLLM response token IDs with returned rollout logprobs when vLLM emits an unpaired trailing token at the generation cap. Without these patches, GCS resume can falsely report missing `model_world_size_*_rank_*.pt` shards, or GRPO can abort on SkyRL's response/logprob length assertion.

Operational lessons from the full run:

- If a job is canceled or fails during distributed checkpoint restore or HF export, treat the warm cluster as suspect even when the GCS checkpoint is valid. `ops run-status` surfaces this as `recovery.recommended_action=fresh_cluster_resume`; down the cluster, relaunch with `--resume-from latest`, the same `--ckpt-path`, and at least 1024 GB on single-node GRPO or 2048 GB on multi-node GRPO.
- If the mistake happens before Ray/SkyRL training starts, such as bad dataset/model staging, cancel the bad job and reuse the allocated cluster when possible. This preserves scarce GPU capacity and avoids another provisioning round; the next launch syncs the corrected workdir and starts a new job on the same cluster.
- Do not infer paid-resource count from local `SkyPilot:executor:*` processes. SkyPilot keeps a local API server and executor pool alive; those are not GPU jobs. Use `w8-biayn ops status`, `w8-biayn ops queue`, `w8-biayn ops run-status`, and labeled GCP instance checks to decide what is actually running and whether cleanup is safe.

If an SFT attempt fails after writing a checkpoint, resume it from the latest marker with the same checkpoint path:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --train-epochs 2 \
  --eval-interval 50 \
  --ckpt-interval 100 \
  --hf-save-interval 1074 \
  --ckpt-path "${RUN_GCS}/cpp-sft/ckpts" \
  --export-path "${RUN_GCS}/cpp-sft/exports" \
  --max-ckpts-to-keep 2 \
  --resume-from latest
```

If a SkyRL SFT checkpoint is complete but the HF export is incomplete, recover the export without another SFT epoch. Use a local artifact export path so the launcher uploads the finished HF directory back under `${RUN_GCS}/cpp-sft/exports`:

```bash
uv run w8-biayn launch cpp-sft \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --export-path "~/exports/" \
  --export-checkpoint "${RUN_GCS}/cpp-sft/ckpts/global_step_1074" \
  --no-down-after
```

Verify recovery with `ops run-status`: `artifacts.export.final_export_exists` must be `true` and `artifacts.export.final_export.weight_object_count` must be greater than zero before GRPO uses the SFT export.

The same export-only path can export a GRPO policy checkpoint for evaluation after an early stop. Pass the SFT HF export used to initialize GRPO as `--model`, and pass the GRPO checkpoint as `--export-checkpoint`; the launcher uploads the result under `${RUN_GCS}/cpp-grpo/exports`:

```bash
uv run w8-biayn launch cpp-grpo \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --model "${RUN_GCS}/cpp-sft/exports/global_step_1074/policy" \
  --accelerators A100:8 \
  --disk-size 1024 \
  --train-batch-size 16 \
  --export-path "~/exports/" \
  --export-checkpoint "${RUN_GCS}/cpp-grpo/ckpts/global_step_250" \
  --no-down-after
```

When `--model` points at a `gs://` export for GRPO or eval, the launcher stages the model into a directory mounted inside the GPU container before invoking SkyRL or vLLM. During staging it normalizes legacy exported tokenizer configs by moving an `extra_special_tokens` list to `additional_special_tokens`, which current Transformers can load. Eval uses the mounted staged path directly; it must not copy the same HF export a second time inside the container. Do not pass host-only staged paths into containerized training. Warm-cluster reruns skip dataset restore only when the local marker matches the requested `--dataset-gcs-prefix` and the manifest/tasks are still present.

GRPO does not require PMU access or Linux perf counters. The required host capability is Docker-outside-Docker plus enough CPU stability for the runtime harness to compare candidate and oracle binaries consistently.

The default `launch` includes SkyPilot `--down`; `--no-down-after` keeps a cluster for inspection. On a shared account, run cleanup when an attempt fails or finishes:

```bash
uv run w8-biayn gcp cleanup --run-id "${RUN_ID}" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "${RUN_ID}" --credentials .gcp-service-account.json --execute
```

Use the project ops commands for run inspection and control. Do not put raw `sky ...` commands in runbooks; `w8-biayn ops ...` is the stable DX boundary if the backend changes later.

```bash
uv run w8-biayn ops status --credentials .gcp-service-account.json --refresh
uv run w8-biayn ops run-status \
  --run-id "${RUN_ID}" \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --expected-sft-final-step 1074 \
  --baseline-status ".w8-biayn/runs/<single-node-run-id>/status.json" \
  --check-retries 1 \
  --node-health \
  --out ".w8-biayn/runs/${RUN_ID}/status.json"
uv run w8-biayn ops grpo-readiness \
  --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml \
  --status-json ".w8-biayn/runs/${RUN_ID}/status.json" \
  --out ".w8-biayn/runs/${RUN_ID}/grpo-readiness.json"
uv run w8-biayn ops queue "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json --tail 200
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" 1 --credentials .gcp-service-account.json --follow
uv run w8-biayn ops cancel "w8-biayn-cpp-grpo-${RUN_ID}" 1 --credentials .gcp-service-account.json
uv run w8-biayn ops down "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions
```

For a rerun or cluster-size experiment, pass one or more prior snapshots with `--baseline-status ".w8-biayn/runs/<baseline-run-id>/status.json"` so the JSON includes `speed_comparison` for training-step and rollout throughput. Interpret speedup factors directly: greater than `1.0` is faster than the baseline, less than `1.0` is slower, and `gpu_speedup_efficiency` is the speedup divided by the GPU scale factor. A `cost_verdict` of `cost_inefficient` means the current run used more GPUs without increasing the primary comparable throughput.

`ops run-status` emits `w8-run-status-v1` JSON for dashboards and polling loops. It includes dataset manifest state, per-pipeline cluster/job state, labeled GCP instances, checkpoint marker and shard completeness for the promoted `latest` checkpoint, highest checkpoint directory, active `in_progress` checkpoint upload, export readiness including final export object counts/bytes and model weight presence, recent log-derived stage/step/checkpoint/export/error signals, normalized phase/progress/resource/command fields, SFT config/last-step progress including micro train batch and SkyRL timeout settings, GRPO config (`effective_samples_per_step`, total GPUs, samples/GPU/step, micro train batch, KL/entropy settings, vLLM GPU memory utilization, checkpoint retention, reward workers, FSDP sizes, HSDP mesh/activity, SkyRL timeout settings), cpp-eval generation/scoring progress, trajectory/evaluation/training throughput, GPU-normalized throughput, ETA/timing metrics, reward metrics, GRPO `training_health` collapse verdicts and `recommended_action`, bottleneck verdicts from SkyRL timing, optional `speed_comparison` against prior `--baseline-status` snapshots, and cleanup safety. Config fields are merged from logs and the local rendered YAML, with `logs.config_sources` and `logs.rendered_config_path`, so long-running dashboards keep stable batch/checkpoint/FSDP settings even after the launch command scrolls out of the log tail. Pass `--node-health` for opt-in read-only SSH health with GPU utilization/memory, disk free space, top processes, a derived node activity, and an explicit `sample_scope`. For multi-node clusters, `--node-health` probes every active labeled VM for the pipeline, so dashboards can verify worker-node participation instead of inferring it from head-node activity. Logs include `tail_lines_requested`, `tail_lines_scanned`, and `tail_may_be_truncated`; treat a truncated tail as a hint to increase `--log-tail` before drawing conclusions from stage parsing. Each backend/GCS/health check is listed with its command, return code, `timed_out`, and `attempt_count`; tune per-check timeout with `--check-timeout` and retry timed-out read-only checks with `--check-retries`.

### Run Status JSON Structure

Dashboards should treat `schema_version` as the compatibility key and prefer the normalized `summary`, `pipelines[].phase`, `pipelines[].progress`, `pipelines[].recovery`, `pipelines[].resources`, and `cleanup` blocks over direct log scraping. The current `w8-run-status-v1` structure is:

```text
root
  schema_version: "w8-run-status-v1"
  generated_at_utc, run_id, project_id, artifact_bucket
  summary
    state, dataset_state, current_pipeline, current_cluster
    current_phase: {current, group, source, log_stage, node_activity, artifact_activity, failed, message}
    current_progress: {primary, training, trajectory, throughput, training_health}
    training_health, speed_comparison, recovery, resources, cleanup_safe
  dataset
    state, gcs_prefix, manifest_uri, schema_version, file_count
    task_counts: {train, validation, test}
    split_files: {sft_train, sft_validation, grpo_train, grpo_validation}
    checks[]
  pipelines[]
    pipeline, cluster, state, run_gcs_prefix
    active_job: {job_id, status, resources, log_path, start_at, end_at, duration_s, job_name}
    backend: {instances[], queue}
    artifacts
      run_gcs_prefix, checks[]
      checkpoint: {prefix, latest_marker, steps[], latest, highest, in_progress}
      export: {prefix, steps[], expected_final_step, final_export_prefix, final_export_exists, final_export}
      eval_outputs: {prefix, objects[], records[], summaries[], labels[], complete_labels[]}  # cpp-eval only
    logs
      stage, stage_events[], last_step, last_loss, last_eval_loss
      trajectory_progress, evaluation_progress, training_progress
      checkpoint_events[], export_events[], timings, metrics, errors[]
      config, config_sources[], rendered_config_path
      tail_lines_requested, tail_lines_scanned, tail_may_be_truncated
      available, error
    phase: {current, group, source, log_stage, node_activity, artifact_activity, failed, message}
    progress
      pipeline, primary, training, trajectory, evaluation, checkpoint
      timings, config, sft_config, grpo_config, metrics, training_health, bottleneck, throughput
    speed_comparison: {available, reason, primary, baselines[]}
    recovery: {available, recommended_action, fresh_cluster_recommended, requires_down_before_resume, reason, signals[], resume_from, resume_checkpoint_step, commands}
    resources: {total_instance_count, active_instance_count, zones[], machine_types[], accelerators, gpu_count, sampled_node_count, failed_node_count, sampled_gpu_count, gpu utilization/memory summaries}
    commands, checks[]
    node_health: {available, skipped, sample_scope, sample_note, expected_node_count, sampled_node_count, failed_node_count, activity, gpus[], filesystems[], processes[], nodes[], error}
                                           # only with --node-health
  gcp: {instances[], active_instance_count, checks[]}
  cleanup: {safe_to_cleanup, active_job_count, active_instance_count, provisioning_instance_count, working_pipeline_count, commands}
```

Each `checks[]` entry has `name`, `command[]`, `ok`, `returncode`, `skipped`, `timed_out`, `attempt_count`, and optional stdout/stderr tails. With `--node-health`, `sample_scope=all_active` means every active labeled VM for the pipeline was probed, `partial` means at least one active VM probe failed, and `none` means no live node telemetry was collected. Top-level `node_health.gpus[]`, `filesystems[]`, and `processes[]` are aggregates annotated with `node_name` and `role`; `node_health.nodes[]` keeps the per-node detail, including failed worker probes. The checkpoint detail objects (`latest`, `highest`, and `in_progress`) contain `prefix`, `step`, `promoted`, `trainer_state_present`, `expected_world_size`, rank counts, world sizes, ranks, object counts, byte totals, and `resumable`. When an active job has an incomplete higher checkpoint than the promoted `latest`, `phase.source` becomes `artifacts` and `phase.artifact_activity` reports `checkpoint_stage`, which protects dashboards from stale or truncated log tails. The export detail object contains object counts, byte totals, config/tokenizer/weight presence, `weight_object_count`, sampled object URIs, and `complete`. `eval_outputs.records[]` and `eval_outputs.summaries[]` expose uploaded eval artifacts by label; `complete_labels[]` contains labels that have both a records JSONL and summary JSON. `training_health.verdict=collapsed` with `should_stop=true` means GRPO metrics match the terminal entropy-collapse/zero-advantage pattern and the operator should stop the run rather than waiting for recovery; `training_health.recommended_action=stop_and_relaunch_from_sft` is the machine-readable action. `training_health.verdict=deterministic_low_gradient` with `should_stop=true` means recent raw trainer metrics show very low entropy and tiny gradients while a resumable checkpoint exists; `training_health.recommended_action=stop_and_evaluate_checkpoint` and `training_health.checkpoint_step` identify the next action. `recovery.available=true` means a failed/canceled checkpointed training job has a machine-readable next action; `recommended_action=fresh_cluster_resume` means the latest checkpoint is resumable but the failed job has distributed-state or interrupted-export signals, so tear down the cluster before relaunching with `--resume-from latest`. `speed_comparison.primary` is present only when `--baseline-status` yields comparable throughput; factors greater than `1.0` are faster, factors less than `1.0` are slower, and `cost_verdict=cost_inefficient` means extra GPUs did not improve the primary comparable throughput. `cleanup.safe_to_cleanup=false` means do not tear the run down automatically.

## Uplift Evaluation

Run the same held-out validation bundle against base, SFT, and GRPO models:

```bash
uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --eval-label base \
  --n-samples-per-prompt 1 \
  --eval-max-tasks 200

uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model "${RUN_GCS}/cpp-sft/exports/global_step_1074/policy" \
  --eval-label sft \
  --n-samples-per-prompt 1 \
  --eval-max-tasks 200

uv run w8-biayn launch cpp-eval \
  --credentials .gcp-service-account.json \
  --dataset-gcs-prefix "${DATA_GCS}" \
  --run-id "${RUN_ID}" \
  --owner sss \
  --model "${RUN_GCS}/cpp-grpo/exports" \
  --eval-label grpo \
  --n-samples-per-prompt 1 \
  --eval-max-tasks 200
```

`cpp-eval` stages `gs://` model exports to local VM storage before loading vLLM, normalizes legacy tokenizer config shape if needed, and mounts that staged directory into the eval container. If the model is an export root with multiple `global_step_*` directories, the launcher resolves the highest complete `global_step_N/policy` export and validates that the staged local directory has `config.json` and model weights before vLLM starts. After generation, `cpp_eval_main` releases the vLLM object and CUDA cache before CPU/Docker scoring, then emits `W8 eval generation ...` and `W8 eval scoring ...` markers that `ops run-status` maps to `eval_generation`, `eval_scoring`, and `progress.evaluation`. On a warm cluster, dataset restore is skipped only when the stored prefix marker matches the requested dataset prefix and the local manifest/tasks are present. Eval artifacts are uploaded to:

```text
gs://<project>-w8-biayn/runs/cpp-perf/<RUN_ID>/cpp-eval/
```

Aggregate local or restored eval records:

```bash
uv run w8-biayn eval cpp \
  --records base=base.records.jsonl \
  --records sft=sft.records.jsonl \
  --records grpo=grpo.records.jsonl \
  --out uplift-summary.json
```

Success criterion for an uplift claim: GRPO beats base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_runtime_rate=0`.

If uplift fails, clone/study SuperCoder and Microsoft/LearningOpt PIE into `/tmp`, compare filtering, prompts, reward shaping, model choice, and hyperparameters, then port only compatible fixes into this SkyRL/rLLM pipeline.

## Operations

```bash
uv run w8-biayn ops status --credentials .gcp-service-account.json
uv run w8-biayn ops logs "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json --tail 200
uv run w8-biayn ops down "w8-biayn-cpp-grpo-${RUN_ID}" --credentials .gcp-service-account.json
uv run w8-biayn benchmarks list
uv run w8-biayn benchmarks show grpo-tiny
```

The legacy top-level `w8-biayn status`, `w8-biayn logs`, and `w8-biayn down` commands remain aliases, but new docs and scripts should prefer `w8-biayn ops ...`.

Pinned upstream clones live under `.cache/upstreams/`:

- `SkyRL`
- `rllm`
- `pie-perf`
- `LearningOpt-pie`

Use:

```bash
uv run w8-biayn upstreams clone
uv run w8-biayn upstreams status
```

Do not vendor upstream repos, CodeNet, PIE archives, generated tests, gem5 outputs, datasets, checkpoints, or credentials.

## Repository Map

```text
scripts/bootstrap.sh                         fresh-machine bootstrap
src/w8_biayn/cli.py                          CLI surface
src/w8_biayn/cpp_perf/data.py                downloads, full PIE prep, manifests, cache
src/w8_biayn/cpp_perf/coverage.py            gcov coverage measurement
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/skyrl_dataset.py       SkyRL GRPO/SFT dataset builder
src/w8_biayn/cpp_perf/eval.py                eval aggregation
src/w8_biayn/cpp_perf/judge.py               contest-style stdout comparison
src/w8_biayn/cpp_perf/sandbox.py             Docker compile/test/runtime harness
src/w8_biayn/cpp_perf/reward.py              correctness-gated efficiency reward
src/w8_biayn/integrations/cpp_perf_env.py    SkyRL environment adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                               SkyRL GRPO entrypoint glue
src/w8_biayn/integrations/skyrl_sft_export_checkpoint_main.py
                                               SkyRL policy checkpoint HF export recovery
src/w8_biayn/integrations/skyrl_io_patch.py    SkyRL checkpoint download compatibility patch
src/w8_biayn/integrations/skyrl_vllm_logprob_patch.py
                                               SkyRL vLLM token/logprob alignment patch
src/w8_biayn/integrations/cpp_eval_main.py   vLLM eval generation and scoring
src/w8_biayn/grpo_readiness.py               GRPO readiness and live-status guardrails
src/w8_biayn/run_status.py                   ops run-status JSON snapshots
src/w8_biayn/shell.py                        dry-run-aware subprocess wrapper
src/w8_biayn/sky_config.py                   SkyPilot YAML renderer
src/w8_biayn/gcp_auth.py                     scoped GCP auth
src/w8_biayn/secrets.py                      credential metadata only
src/w8_biayn/constants.py                    upstream pins and defaults
src/w8_biayn/upstreams.py                    upstream clone management
src/w8_biayn/benchmarks.py                   benchmark ladder
.agents/REPO_GUIDE.md                        shared AGENTS.md and CLAUDE.md target
.agents/skills/w8-biayn-framework/SKILL.md   AI coding-agent workflow skill
```

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup, CLI, cloud, or data-pipeline changes, also run relevant dry checks:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn data doctor
uv run w8-biayn benchmarks list
uv run w8-biayn cpp harness preflight --dry-run
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn ops status --credentials .gcp-service-account.json --dry-run
uv run w8-biayn ops run-status --run-id rdoc --credentials .gcp-service-account.json --dry-run
uv run w8-biayn ops gpus A100 --credentials .gcp-service-account.json --all-regions --dry-run
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn ops grpo-readiness --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml
uv run w8-biayn launch cpp-eval --dry-run --credentials .gcp-service-account.json --run-id rdoc
uv run w8-biayn gcp cleanup --run-id rdoc --credentials .gcp-service-account.json --dry-run
```
