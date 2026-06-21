# Repository Development Guide

`AGENTS.md` and `CLAUDE.md` must stay symlinks to this file. Update this file once; do not fork the guidance.

## Active Project

This repository is a C++ performance-RL project. The inherited codebase is only baseline infrastructure for GCP, SkyPilot, SkyRL, and rLLM.

Phase 1 goal: train an open-weight model that rewrites correct C++ programs to run faster while preserving behavior, then prove uplift on held-out PIE tasks.

Out of scope unless a later phase is explicitly requested:

- BrowserGym
- DOMDiff
- Harbor
- WebArena
- MiniWoB
- AndroidWorld
- Go
- Custom GPU kernel labs or unrelated performance experiments

## Required Reading

Before changing behavior, read:

1. `README.md`
2. `/tmp/ENGINEERING_SPEC_v2_cpp_only.md`
3. `.agents/skills/w8-biayn-framework/SKILL.md`
4. Relevant implementation files under `src/w8_biayn/`

## Non-Negotiable Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder training path as the active trainer.

Do not replace SkyRL/rLLM with another framework.

Allowed upstream use:

- PIE: source C++ slower-to-faster pairs, official tests, and eval/data lessons.
- LearningOpt PIE: gem5 reference and calibration environment.
- SuperCoder: schema, correctness/eval lessons, and examples.
- SkyRL/rLLM: actual SFT and GRPO training stack.

Use `uv run w8-biayn upstreams clone` for pinned repo copies under `.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor upstream repos or data.

## Fresh-Machine Contract

A clean clone must be able to run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

If a change invalidates any command, update implementation, tests, README, this file, and `.agents/skills/w8-biayn-framework/SKILL.md` in the same logical change.

Do not rely on globally installed tools unless bootstrap installs them or `doctor` reports a clear missing prerequisite with the exact next action.

## Data Discipline

Dataset conversion is a deliverable. No one-off PIE or SuperCoder munging is allowed.

All source downloads, archive normalization, coverage measurement, task construction, SkyRL conversion, GCS upload, and GCS restore must be represented as `w8-biayn data ...` commands with tests and docs.

Full official PIE path:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full --out .w8-biayn/data/pie-full/coverage.json --report-out .w8-biayn/data/pie-full/coverage-report.json
uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full --coverage-json .w8-biayn/data/pie-full/coverage.json --out .w8-biayn/data/tasks-full --min-train 1000 --min-validation 100 --min-test 100 --force
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/skyrl-full --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl-full --gcs-prefix "gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl" --credentials .gcp-service-account.json
```

Admission gates are required:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- SkyRL bundle manifest verifies before upload.

Default schema version: `cpp-perf-v1`.

Do not overwrite a full-run dataset prefix. Use `full-official/<RUN_ID>/skyrl`; update a `latest` alias only with an explicit command or user request.

## Task Rules

Preserve PIE task discipline:

- `v0` slower C++ becomes the prompt.
- `v1` faster C++ is not shown during GRPO.
- `v1` may be used as SFT target, oracle/reference material, and coverage measurement input.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and coverage passing 95 percent line / 85 percent branch.

The prompt may include visible tests and `v0`. It must not include hidden tests or `v1`.

## Reward Rules

The reward is correctness gated:

- Invalid format is negative.
- Recoverable C++ with missing wrapper/fence format is shaped below the correctness-only fallback, so GRPO has a gradient out of format collapse without making bare code a target.
- Compile or sanitizer failure is negative.
- Timeout is negative.
- Partial tests remain below any fully correct answer.
- Fully correct answers with missing non-timeout runtime measurement get a correctness-only fallback below any measured fully correct answer.
- Fully correct answers get a base reward plus bounded runtime-efficiency.
- Child-process CPU time in nanoseconds is the fast RL reward metric.
- Wall-clock nanoseconds are recorded as diagnostics.
- gem5 is calibration/final-eval reference only.
- `w8-biayn cpp harness preflight` is required before GRPO to prove Docker runtime measurement works.

Model outputs must contain exactly one `<reasoning>...</reasoning>` block followed by exactly one fenced C++ code block. The code may start on the next line or after whitespace on the opening C++ fence line; any second code block is invalid.

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and hidden tests, then benchmarks both binaries in the same Docker sandbox with the same CPU pinning, compiler flags, and tests. Default timing uses 1 warmup and 3 measured repeats, median CPU time per test, summed across tests.

Do not add PMU, Linux perf, PERFMON, or `perf_event_paranoid` dependencies to the active reward path.

## Training Rules

Training runs through SkyPilot on GCP and delegates to SkyRL/rLLM.

Render before launch:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
```

Full training must pass explicit storage and retention:

```bash
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --accelerators A100:8 --disk-size 1024 --train-batch-size 16 --train-epochs 2 --eval-interval 50 --ckpt-interval 100 --hf-save-interval 1074 --ckpt-path "$RUN_GCS/cpp-sft/ckpts" --export-path "$RUN_GCS/cpp-sft/exports" --max-ckpts-to-keep 2 --no-down-after
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports/global_step_1074/policy" --accelerators A100:8 --disk-size 1024 --train-batch-size 16 --n-samples-per-prompt 8 --train-epochs 3 --no-eval-before-train --eval-interval 25 --max-env-workers 128 --ckpt-interval 50 --hf-save-interval 10000 --ckpt-path "$RUN_GCS/cpp-grpo/ckpts" --export-path "$RUN_GCS/cpp-grpo/exports" --max-ckpts-to-keep 8 --grpo-vllm-gpu-memory-utilization 0.7 --no-down-after
```

Use `--detach-run` for long full-training jobs so the local launch command returns after backend submission and operators poll with `ops run-status` / `ops metrics`; do not use raw backend commands for detaching.

For the full official SFT setup, keep `--ckpt-interval 100` for recovery but set `--hf-save-interval` to the expected final step (`1074` for two epochs at batch size 16 on the current full PIE bundle) unless a run intentionally needs intermediate HF exports. `save_hf_model` is CPU/GCS-heavy and can idle the GPUs for tens of minutes per export; `pipelines[].progress.bottleneck` should surface that as `save_hf_model` when it happens. SkyRL also exports HF models at epoch boundaries and after the loop, so a large interval is not final-only; it suppresses interval exports while preserving epoch-boundary/final exports. Pipeline defaults are per purpose: `cpp-smoke` uses `H100:8` and `zai-org/GLM-5.1`, `cpp-sft`/`cpp-grpo` use `A100:8` and `Qwen/Qwen2.5-Coder-7B-Instruct`, and `cpp-eval` uses `A100:1` and `Qwen/Qwen2.5-Coder-7B-Instruct`. Rendered training jobs request `128+` GB host memory; rendered eval jobs request `80+` GB so GCP single-A100 shapes are not filtered out before provisioning. Eval pins `vllm==0.6.6.post1`, `transformers==4.57.6`, and CUDA 12.4 PyTorch wheels because unpinned latest vLLM can select a CUDA 13 stack that the A2 driver cannot load. If an SFT attempt writes checkpoints and then fails, resume the same run with `--resume-from latest`, `--disk-size 1024` or larger, the same `--ckpt-path`, and the same final-step `--hf-save-interval`.
Rendered GRPO enables a small KL anchor and entropy bonus by default (`--grpo-use-kl-loss`, `--grpo-kl-loss-coef 0.001`, `--grpo-use-entropy-loss`, `--grpo-entropy-loss-coef 0.001`) to reduce drift from the SFT/reference policy and avoid deterministic format collapse. It also sets `--grpo-vllm-gpu-memory-utilization 0.7` by default so colocated vLLM leaves enough headroom for FSDP policy/ref workers on A100-40GB; if vLLM fails while waking the KV cache, lower this before relaunching. Keep at least several GRPO checkpoints (`--max-ckpts-to-keep 8` in the full-run command) because the latest checkpoint can be worse than an earlier healthy checkpoint after reward over-optimization.
Rendered SFT and GRPO containers must export longer SkyRL/Ray distributed timeouts (`SKYRL_RAY_PG_TIMEOUT_IN_S=1800`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600`) so slow FSDP checkpoint restore and HF export barriers do not fail at SkyRL's 10-minute worker default. Override them only for a deliberate debugging run, and keep the values visible in `ops run-status`.
Rendered training containers must also pass `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=^lo,docker,veth`, a concrete default-route `GLOO_SOCKET_IFNAME`, and `NCCL_DEBUG=WARN` into Docker so SkyRL, Ray, and vLLM use the VM network interface instead of loopback or transient container interfaces. Missing this propagation is a known multi-node reproducibility mismatch: the host shell can have valid networking while the training container still fails during Ray/NCCL/vLLM startup. Do not reuse NCCL's `^lo,docker,veth` exclusion syntax for Gloo; PyTorch Gloo expects a real interface name and will fail process-group initialization if given `^lo`.
Rendered training containers must also apply idempotent SkyRL compatibility patches: `skyrl_io_patch.py` flattens cloud checkpoint directory downloads before FSDP rank-shard validation, `skyrl_vllm_logprob_patch.py` aligns vLLM response token IDs with returned rollout logprobs when vLLM emits an unpaired trailing token at the generation cap, `skyrl_grpo_health_patch.py` emits `W8_GRPO_HEALTH` JSON lines with train/eval reward, KL, entropy, advantage, reward-variance, and phase-timing metrics for `ops run-status`, and `skyrl_startup_patch.py` emits `W8_SETUP_STAGE` JSON lines around FSDP policy/reference model initialization. Without these patches, GCS resume can falsely report missing `model_world_size_*_rank_*.pt` shards, GRPO can abort on SkyRL's response/logprob length assertion, dashboards can miss deterministic-convergence and overfit signals, or a long pre-metric startup can look like an untracked hang.
Rendered full SFT and GRPO containers must keep SkyRL `console` logging and enable `mlflow` logging through an MLflow Tracking Server, not through direct SQLite writes. Rank 0 starts `mlflow server` backed by `/artifacts/tracking/mlflow/mlflow.db`, exports `MLFLOW_TRACKING_URI` for SkyRL/Ray, and the SkyPilot host snapshots the backend store to `$RUN_GCS/<pipeline>/tracking/mlflow/mlflow.db`. Use `w8-biayn ops metrics` for headless `w8-mlflow-metrics-v1` series and `ops run-status` for the merged `tracking.mlflow` and learning-signal view. If MLflow reports `run_active_no_metrics`, poll with `ops run-status --node-health` while backend resources are live and inspect `progress.startup` / `node_health.startup`; if no live backend resources remain, `run-status` treats the no-scalar tracking state as a failed pre-metric startup and reports `progress.startup.recommended_action=inspect_failed_startup_or_relaunch`. KL-enabled GRPO initializes a full FSDP reference model before first rollout, so `ref_model_init` can be the real pre-metric bottleneck.
If a GRPO attempt writes a complete checkpoint and then fails or is canceled, resume the same run with `--resume-from latest` and the same `--ckpt-path`. Use `--disk-size 1024` or larger for single-node GRPO resume and `--disk-size 2048` or larger for multi-node GRPO resume; a 2x[A100:8] resume from `global_step_150` filled a 1024 GB boot disk during FSDP checkpoint restore and failed with `[Errno 28] No space left on device`, so the CLI rejects smaller explicit disks and defaults resumed multi-node GRPO to 2048 GB. When the final SFT step is known, pass the concrete HF policy export (`.../exports/global_step_N/policy`) as `--model`; if a `gs://.../exports` root is passed, rendered GRPO/eval jobs resolve it to the highest complete `global_step_N/policy` directory with model weights before staging it locally. Multi-node GRPO uses `--num-nodes N`; rendered colocated SkyRL starts a rank-gated Ray cluster inside the GPU containers, runs `skyrl_cpp_perf_main` only on rank 0, and sets rollout engines to `N * GPUs_PER_NODE` for TP=1/DP=1 so policy and inference placement match. It also sets `trainer.policy.fsdp_config.fsdp_size` and `trainer.ref.fsdp_config.fsdp_size` to GPUs-per-node so multi-node FSDP shards within a node (NVLink) and replicates across nodes (HSDP) instead of full-sharding the model across the slow inter-node link; without this, a model that fits on one node can train slower on two nodes than on one. Reserve multi-node for models that do not fit on a single node or generation/reward-bound steps, and verify the inter-node interconnect before scaling out. Multi-node GRPO must pass the CLI utilization gate: effective samples per step (`train_batch_size * n_samples_per_prompt`) must be at least 16 per GPU and `max_env_workers` must be at least the effective samples per step. For 2x[A100:8], use `--train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256`; use `--allow-low-multinode-utilization` only for a deliberate experiment. `--micro-train-batch-size-per-gpu` defaults to `1`; try `2` only as a memory-checked policy-update tuning experiment on A100-40GB and fall back to `1` on OOM or no throughput gain. Use `--no-eval-before-train` to skip the expensive initial validation pass. Rendered GRPO uses non-batched trajectory generation so SkyRL-Gym can overlap C++ reward calls; raise `--max-env-workers` when Docker reward compilation/runtime is the bottleneck.
Before any paid multi-node GRPO training launch, `launch cpp-grpo --num-nodes N` runs `ops grpo-readiness` against the rendered YAML and blocks critical failures. Operators and agents may also run it directly:

```bash
uv run w8-biayn ops grpo-readiness --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml --out ".w8-biayn/runs/$RUN_ID/grpo-readiness.json"
uv run w8-biayn ops grpo-readiness --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml --status-json ".w8-biayn/runs/$RUN_ID/status.json"
```

The readiness JSON schema is `w8-grpo-readiness-v1`. Static checks must validate Docker reward mounts, host/container preflights, `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`, `NCCL_DEBUG`, concrete default-route `GLOO_SOCKET_IFNAME`, `skyrl_io_patch.py`, `skyrl_vllm_logprob_patch.py`, `skyrl_grpo_health_patch.py`, `skyrl_startup_patch.py`, MLflow Tracking Server setup, MLflow GCS persistence, HSDP `fsdp_size`, rollout engine count, utilization gate, resume disk, checkpoint retention, and KL/entropy settings. Gloo must receive a real interface name; do not pass NCCL's `^lo,docker,veth` exclusion syntax to `GLOO_SOCKET_IFNAME`. For live multi-node claims, readiness must be run with a `--node-health` status snapshot and must show `node_health.sample_scope=all_active`, sampled nodes equal active nodes, failed node probes equal zero, and sampled GPUs equal the GRPO GPU count. If `training_health.should_stop=true` or `learning_signal.recommended_action=evaluate_checkpoint`, treat readiness `overall=action_required` as a stop/evaluate or relaunch instruction, not as a warning to keep polling.
If an SFT checkpoint is complete but the HF export has no model weight files, rerun `cpp-sft` as an export-only recovery with `--export-checkpoint "$RUN_GCS/cpp-sft/ckpts/global_step_N"`, `--export-path "~/exports/"`, `--disk-size 1024`, and `--no-down-after`; then verify `ops run-status` reports `artifacts.export.final_export_exists=true` and `artifacts.export.final_export.weight_object_count>0` before starting GRPO. If a GRPO checkpoint should be evaluated after an early stop, rerun `cpp-grpo` as an export-only recovery with the SFT HF export as `--model`, the GRPO checkpoint as `--export-checkpoint "$RUN_GCS/cpp-grpo/ckpts/global_step_N"`, and a local `--export-path`; the launcher uploads the policy export under `$RUN_GCS/cpp-grpo/exports`.
When GRPO or eval uses a `gs://` model export, the launcher must stage it into a path mounted inside the GPU container before invoking SkyRL or vLLM; host-only staged paths are invalid. During staging, normalize legacy tokenizer configs by moving an `extra_special_tokens` list to `additional_special_tokens` so current Transformers can load SkyRL exports. Eval must use that mounted staged path directly rather than copying the same HF export again inside the container. If the model points at an export root with multiple `global_step_*` directories, the rendered job must resolve the highest complete `global_step_N/policy` export and validate local `config.json` plus model weight files before starting SkyRL or vLLM. Warm-cluster reruns may skip dataset restore only when the local prefix marker matches the requested `--dataset-gcs-prefix` and the manifest/tasks are present.
If a job is canceled or fails during distributed checkpoint restore or HF export, treat the warm cluster as suspect even when the GCS checkpoint is valid. `ops run-status` should expose `recovery.recommended_action=fresh_cluster_resume`; down the cluster, relaunch with `--resume-from latest`, the same `--ckpt-path`, and at least 1024 GB on single-node GRPO or 2048 GB on multi-node GRPO. If the mistake happens before Ray/SkyRL training starts, such as bad dataset/model staging, cancel the bad job and reuse the allocated cluster when possible so the corrected workdir can start as a new job without another scarce-GPU provisioning round. Do not infer paid-resource count from local `SkyPilot:executor:*` processes; SkyPilot keeps a local API server and executor pool alive, and actual paid resources must be checked through `w8-biayn ops status`, `w8-biayn ops queue`, `w8-biayn ops run-status`, and labeled GCP instance checks.

The GRPO entrypoint is `python -m w8_biayn.integrations.skyrl_cpp_perf_main`. It registers `cpp-perf` and delegates to SkyRL `BasePPOExp(cfg).run()`.

Rendered GRPO must run a host-side C++ runtime preflight before GCS restore, model staging, GPU image pulls, or framework installs, and must run the preflight again before `skyrl_cpp_perf_main`. GRPO rewards use Docker-outside-Docker, so rendered YAML must mount `/var/run/docker.sock` and host `/tmp` into the GPU training container.
Rendered GRPO and eval must not require PMU access or mutate `kernel.perf_event_paranoid`.

## Cloud Rules

Cloud commands must:

- support dry-run rendering before paid launches;
- use `.gcp-service-account.json` through scoped env vars;
- avoid `gcloud auth activate-service-account`;
- avoid mutating global `gcloud config`;
- avoid printing credential contents;
- render YAML into `.w8-biayn/rendered/`;
- label paid resources with `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`;
- provide operations through `w8-biayn ops status`, `w8-biayn ops run-status`, `w8-biayn ops metrics`, `w8-biayn ops grpo-readiness`, `w8-biayn ops logs`, `w8-biayn ops queue`, `w8-biayn ops cancel`, `w8-biayn ops down`, `w8-biayn ops gpus`, and `w8-biayn gcp cleanup`.

Use `w8-biayn ops run-status --run-id "$RUN_ID" --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --out ".w8-biayn/runs/$RUN_ID/status.json"` for dashboard/polling status, and `w8-biayn ops metrics --run-id "$RUN_ID" --pipeline cpp-grpo --credentials .gcp-service-account.json --out ".w8-biayn/runs/$RUN_ID/cpp-grpo-mlflow-metrics.json"` for direct MLflow time-series JSON. The MLflow Tracking Server is private to the training cluster. `ops metrics --source auto` opens an SSH tunnel to the cluster head and queries the live MLflow API first, then falls back to the GCS-synced SQLite backend at `$RUN_GCS/<pipeline>/tracking/mlflow/mlflow.db`; `--source api` requires the tunnel/API path and `--source sqlite` reads only the durable snapshot. If `metrics.available=true`, `metrics.tracking_state=run_active_no_metrics`, and `metrics.metric_count=0`, the server and SkyRL run are registered but SkyRL has not logged its first scalar yet. Treat that as a startup state, not proof of progress: use `ops run-status --node-health` while the cluster is live and inspect `summary.current_progress.startup.active_stage`, `groups[].process_count`, `max_elapsed_s`, `nodes`, and `roles` to find dependency setup, Ray startup, `ref_model_init`, `policy_model_init`, or another pre-rollout stage. After teardown, the same no-scalar state means no further node-health probe is possible, so `run-status` marks the pipeline failed and sets `progress.startup.recommended_action=inspect_failed_startup_or_relaunch`. Once scalar rows exist, `tracking_state=metrics_available`.

For GRPO analysis, watch the standard post-training groups from MLflow: held-out quality (`eval/all/avg_score`, `eval/all/pass_at_1`, `eval/all/pass_at_4`, `eval/all/pass_at_8`, `eval/all/mean_positive_reward`), train reward (`loss/avg_final_rewards`, `reward/avg_pass_at_8`, `reward/avg_raw_reward`, `reward/mean_positive_reward`), policy sanity (`policy/policy_kl`, `loss/avg_kl`, `loss/avg_kl_max`, `policy/policy_entropy`, `policy/response_length`, `generate/{avg,std,min,max}_assistant_tokens`), optimizer health (`policy/grad_norm`, `policy/policy_loss`, `policy/pg_loss:sum`, `policy/total_loss:sum`, `policy/loss_metrics/clip_ratio`, `loss/avg_raw_advantages_abs`, `policy/policy_lr`), RL correctness (`policy/rollout_train_logprobs_abs_diff_mean`, `policy/rollout_train_logprobs_abs_diff_std`, `generate/avg_tokens_zero_rewards`, `generate/avg_tokens_non_zero_rewards`), W8 variance (`w8/reward_group_variance_mean`, `w8/zero_variance_group_fraction`, `w8/zero_advantage_token_fraction`), and phase timings (`timing/step`, `timing/generate`, `timing/policy_train`, `timing/fwd_logprobs_values_reward`, `timing/sync_weights`, `timing/save_checkpoint`, `timing/save_hf_model`). Interpret failure modes from combinations, not one metric: entropy collapse is entropy near zero plus advantage and grad norm near zero; reward hacking is train reward rising while held-out eval stalls or falls, often with length drift; KL blowup is unbounded KL; instability is grad-norm or clip-ratio spikes; rollout/train logprob divergence points at vLLM/FSDP alignment or numerical issues.

For reruns or cluster-size experiments, pass prior snapshots with `--baseline-status ".w8-biayn/runs/<baseline-run-id>/status.json"` so the output includes `speed_comparison`; do not answer "is it faster?" from GPU count alone. Speedup factors greater than `1.0` are faster, factors less than `1.0` are slower, and `gpu_speedup_efficiency` is speedup divided by GPU scale. A `cost_verdict` of `cost_inefficient` means more GPUs did not increase the primary comparable throughput. The JSON schemas are `w8-run-status-v1` and `w8-mlflow-metrics-v1`; run-status includes dataset manifest state, per-pipeline cluster/job state, labeled GCP instances, checkpoint markers and shard completeness for the promoted `latest` checkpoint, highest checkpoint directory, active `in_progress` checkpoint upload, export readiness including final export object counts/bytes and model weight presence, MLflow tracking-server snapshot state, recent log-derived setup/stage/step/checkpoint/export/error signals, normalized phase/progress/resource/command fields, SFT config/last-step progress including micro train batch and SkyRL timeout settings, GRPO config (`effective_samples_per_step`, total GPUs, samples/GPU/step, micro train batch, KL/entropy settings, vLLM GPU memory utilization, checkpoint retention, reward workers, FSDP sizes, HSDP mesh/activity, SkyRL timeout settings), cpp-eval generation/scoring progress, startup substage summaries, trajectory/evaluation/training throughput, GPU-normalized throughput, ETA/timing metrics, reward metrics, GRPO `training_health`, `learning_signal`, and `phase_timing` verdicts, bottleneck verdicts from SkyRL timing, optional baseline speed comparison with GPU speedup efficiency/cost verdict, and cleanup safety. Config fields are merged from logs and the local rendered YAML, with `logs.config_sources` and `logs.rendered_config_path`, so long-running dashboards keep stable batch/checkpoint/FSDP settings even after the launch command scrolls out of the log tail. Logs include `tail_lines_requested`, `tail_lines_scanned`, and `tail_may_be_truncated`; increase `--log-tail` before relying on stage parsing when the tail may be truncated. Pass `--node-health` for opt-in read-only SSH health with GPU utilization/memory, disk free space, top processes, startup process groups, derived node activity, and explicit `sample_scope`. On multi-node clusters, `--node-health` must probe every active labeled pipeline VM so dashboards can verify worker-node participation instead of inferring it from the head node. Instantaneous GPU utilization is phase-dependent: `resources.gpu_sample_interpretation` and `progress.phase_timing` must be read together before concluding a run is GPU-bound, reward-bound, idle, or healthy. Each backend/GCS/health/tracking check includes its command, return code, `timed_out`, and `attempt_count`; tune timeout with `--check-timeout` and retry timed-out or transiently failed read-only checks with `--check-retries`.

`w8-run-status-v1` uses this dashboard-facing structure:

```text
root
  schema_version: "w8-run-status-v1"
  generated_at_utc, run_id, project_id, artifact_bucket
  summary
    state, dataset_state, current_pipeline, current_cluster
    current_phase: {current, group, source, log_stage, node_activity, artifact_activity, failed, message}
    current_progress: {primary, training, trajectory, throughput, training_health, learning_signal, phase_timing, tracking, startup}
    training_health, learning_signal, speed_comparison, recovery, resources, cleanup_safe
  dataset
    state, gcs_prefix, manifest_uri, schema_version, file_count
    task_counts: {train, validation, test}
    split_files: {sft_train, sft_validation, grpo_train, grpo_validation}
    checks[]
  pipelines[]
    pipeline, cluster, state, run_gcs_prefix, active_job
    backend: {instances[], queue}
    artifacts
      run_gcs_prefix, checks[]
      checkpoint: {prefix, latest_marker, steps[], latest, highest, in_progress}
      export: {prefix, steps[], expected_final_step, final_export_prefix, final_export_exists, final_export}
      eval_outputs: {prefix, objects[], records[], summaries[], labels[], complete_labels[]}  # cpp-eval only
    tracking
      available, reason, backend
      mlflow: {available, tracking_state, latest_step, metric_count, metric_row_count, run, params, tags, available_keys[], selected_keys[], latest, series, source}
    logs
      stage, stage_events[], setup_events[], last_step, last_loss, last_eval_loss
      trajectory_progress, evaluation_progress, training_progress
      checkpoint_events[], export_events[], timings, metrics, policy_health_events[], grpo_health_events[], errors[]
      config, config_sources[], rendered_config_path
      tail_lines_requested, tail_lines_scanned, tail_may_be_truncated
      available, error
    phase: {current, group, source, log_stage, node_activity, artifact_activity, failed, message}
    progress
      pipeline, primary, training, trajectory, evaluation, checkpoint
      timings, config, sft_config, grpo_config, metrics, tracking, startup, training_health, learning_signal, phase_timing, bottleneck, throughput
    speed_comparison: {available, reason, primary, baselines[]}
    recovery: {available, recommended_action, fresh_cluster_recommended, requires_down_before_resume, reason, signals[], resume_from, resume_checkpoint_step, commands}
    resources: {total_instance_count, active_instance_count, zones[], machine_types[], accelerators, gpu_count, sampled_node_count, failed_node_count, sampled_gpu_count, gpu utilization/memory summaries, gpu_sample_interpretation}
    commands, checks[]
    node_health: {available, skipped, sample_scope, sample_note, expected_node_count, sampled_node_count, failed_node_count, activity, startup, gpus[], filesystems[], processes[], nodes[], error}
                                           # only with --node-health
  gcp: {instances[], active_instance_count, checks[]}
  cleanup: {safe_to_cleanup, active_job_count, active_instance_count, provisioning_instance_count, working_pipeline_count, commands}
```

For UI bindings, prefer `summary.current_phase`, `summary.current_progress`, `summary.current_progress.startup`, `summary.training_health`, `summary.learning_signal`, `summary.recovery`, `pipelines[].progress.throughput`, `pipelines[].progress.startup`, `pipelines[].progress.training_health`, `pipelines[].progress.learning_signal`, `pipelines[].progress.phase_timing`, `pipelines[].progress.sft_config`, `pipelines[].progress.grpo_config`, `pipelines[].speed_comparison.primary`, `pipelines[].recovery`, `pipelines[].resources`, `pipelines[].node_health.startup`, `pipelines[].artifacts.checkpoint.latest.resumable`, `pipelines[].artifacts.export.final_export.complete`, `pipelines[].artifacts.eval_outputs.complete_labels`, and `cleanup.safe_to_cleanup`. Each `checks[]` entry has `name`, `command[]`, `ok`, `returncode`, `skipped`, `timed_out`, `attempt_count`, and optional stdout/stderr tails. With `--node-health`, `sample_scope=all_active` means every active labeled VM for the pipeline was probed, `partial` means at least one active VM probe failed, and `none` means no live node telemetry was collected. Top-level `node_health.gpus[]`, `filesystems[]`, and `processes[]` are aggregates annotated with `node_name` and `role`; `node_health.nodes[]` keeps per-node detail. `node_health.startup` and `progress.startup` summarize pre-metric process groups such as `dependency_setup`, `ray_cluster`, `ray_worker_join`, `skyrl_entrypoint`, `ref_model_init`, and `policy_model_init`, including process counts, elapsed seconds, CPU use, sampled nodes, roles, and a warning when the MLflow run is active but no scalar metrics have been logged. When that no-scalar tracking state remains after all backend resources are gone, `progress.startup.severity=error` and `recommended_action=inspect_failed_startup_or_relaunch` because the startup can no longer be observed live. Checkpoint detail objects expose `trainer_state_present`, `expected_world_size`, shard rank/world-size counts, bytes, and `resumable`; when an active job has an incomplete higher checkpoint than the promoted `latest`, `phase.source=artifacts` and `phase.artifact_activity=checkpoint_stage` protect dashboards from stale or truncated log tails. Export detail objects expose config/tokenizer/weight presence, `weight_object_count`, bytes, sampled objects, and `complete`. Eval output records and summaries expose uploaded artifacts by label; `complete_labels` contains labels with both records and summary files. `training_health.verdict=collapsed` with `should_stop=true` sets `training_health.recommended_action=stop_and_relaunch_from_sft`; `training_health.verdict=deterministic_low_gradient` with `should_stop=true` sets `training_health.recommended_action=stop_and_evaluate_checkpoint` and `training_health.checkpoint_step`; `training_health.verdict=deterministic_convergence_risk` keeps `should_stop=false` but sets `recommended_action=evaluate_checkpoint` when train reward/pass are high while entropy, gradients, advantages, or reward variance suggest learning may be stalled. `learning_signal` is built from `W8_GRPO_HEALTH` and SkyRL metrics; it exposes train reward/pass, held-out eval metrics when available, policy KL, entropy, estimated KL/entropy loss terms, reward-group variance, mean absolute advantage, zero-advantage fraction, recent trends, and a dashboard verdict. `phase_timing` groups SkyRL timing into rollout, reward, policy-update, synchronization, checkpoint, and export work; use it with `resources.gpu_sample_interpretation` before deciding whether a 0 percent or 100 percent GPU sample means anything about throughput. `recovery.available=true` means a failed/canceled checkpointed training job has a machine-readable next action; `recommended_action=fresh_cluster_resume` means the latest checkpoint is resumable but distributed-state or interrupted-export signals make same-cluster reuse risky, so down the cluster before relaunching with `--resume-from latest`. `speed_comparison.primary` exists only with comparable `--baseline-status` inputs; factors greater than `1.0` are faster, factors less than `1.0` are slower. `cleanup.safe_to_cleanup=false` means the shared account still has active/provisioning work for the run.

If `progress.startup.recommended_action=inspect_failed_startup_or_relaunch`, dashboards should also expect `phase.source=tracking`: the run registered in MLflow, never produced scalar metrics, and no live backend remains to inspect.

Do not put raw `sky ...` commands in docs, runbooks, or agent handoffs. Keep SkyPilot/SkyRL backend details behind the `w8-biayn` CLI so the operator DX stays stable if the backend changes.

`doctor --cloud` must check the full SkyPilot launch permission set before paid runs.

On the shared GCP account, use one active full-training cluster at a time. Run cleanup after failed or completed attempts:

```bash
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --execute
```

## Evaluation Rules

Use `cpp-eval` for base, SFT, and GRPO on the same held-out data:

```bash
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model Qwen/Qwen2.5-Coder-7B-Instruct --eval-label base --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports/global_step_1074/policy" --eval-label sft --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-grpo/exports" --eval-label grpo --n-samples-per-prompt 1
```

`cpp-eval` stages `gs://` model exports to local VM disk, normalizes legacy tokenizer config shape if needed, mounts the staged directory into the eval container, and then loads vLLM from that mounted path. After generation, `cpp_eval_main` releases the vLLM object and CUDA cache before CPU/Docker scoring, then emits `W8 eval generation ...` and `W8 eval scoring ...` markers that `ops run-status` maps to `eval_generation`, `eval_scoring`, and `progress.evaluation`. On warm clusters, dataset restore is skipped only when the local prefix marker matches the requested dataset prefix and the manifest/tasks are present.

Aggregate records with:

```bash
uv run w8-biayn eval cpp --records base=base.records.jsonl --records sft=sft.records.jsonl --records grpo=grpo.records.jsonl --out uplift-summary.json
```

Uplift claim requires GRPO to beat base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_runtime_rate=0`.

If uplift fails, clone/study SuperCoder and Microsoft/LearningOpt PIE into `/tmp`, compare filtering, prompts, reward shape, model choice, and hyperparameters, then port compatible fixes into this SkyRL/rLLM pipeline only.

## Repo Responsibilities

```text
scripts/bootstrap.sh                         bootstrap
src/w8_biayn/cli.py                          CLI
src/w8_biayn/cpp_perf/data.py                downloads, full PIE prep, manifests, cache
src/w8_biayn/cpp_perf/coverage.py            gcov coverage measurement
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/skyrl_dataset.py       SkyRL data conversion
src/w8_biayn/cpp_perf/eval.py                eval aggregation
src/w8_biayn/cpp_perf/judge.py               contest-style stdout comparison
src/w8_biayn/cpp_perf/schema.py              task and harness schema
src/w8_biayn/cpp_perf/sandbox.py             C++ Docker harness
src/w8_biayn/cpp_perf/reward.py              reward function
src/w8_biayn/integrations/cpp_perf_env.py    SkyRL env adapter
src/w8_biayn/integrations/skyrl_cpp_perf_main.py
                                             SkyRL entrypoint glue
src/w8_biayn/integrations/skyrl_sft_export_checkpoint_main.py
                                             SkyRL policy checkpoint HF export recovery
src/w8_biayn/integrations/skyrl_io_patch.py
                                             SkyRL checkpoint download compatibility patch
src/w8_biayn/integrations/skyrl_vllm_logprob_patch.py
                                             SkyRL vLLM token/logprob alignment patch
src/w8_biayn/integrations/skyrl_grpo_health_patch.py
                                             SkyRL GRPO health metric logging patch
src/w8_biayn/integrations/skyrl_startup_patch.py
                                             SkyRL startup stage logging patch
src/w8_biayn/integrations/cpp_eval_main.py   vLLM eval and scoring
src/w8_biayn/grpo_readiness.py               GRPO readiness and live-status guardrails
src/w8_biayn/mlflow_metrics.py               MLflow tracking-server SQLite metric reader
src/w8_biayn/run_status.py                   ops run-status JSON snapshot builder
src/w8_biayn/shell.py                        dry-run-aware subprocess wrapper
src/w8_biayn/sky_config.py                   SkyPilot rendering
src/w8_biayn/gcp_auth.py                     scoped GCP auth
src/w8_biayn/secrets.py                      credential metadata only
src/w8_biayn/constants.py                    upstream pins and defaults
src/w8_biayn/upstreams.py                    upstream clone management
src/w8_biayn/benchmarks.py                   benchmark ladder
README.md                                    user and operator docs
.agents/skills/w8-biayn-framework/SKILL.md   AI coding-agent skill
```

## Documentation Rules

Update docs in the same change when commands, setup, dataset shape, cache behavior, cloud behavior, task schema, reward logic, launch flow, benchmark protocol, or supported pipelines change.

Required doc targets:

1. `README.md`
2. Mermaid diagrams in `README.md`
3. `.agents/REPO_GUIDE.md`
4. `.agents/skills/w8-biayn-framework/SKILL.md`
5. Tests when command behavior changes

## Commit Discipline

Prefer focused commits. Each commit should include code, docs, tests, and skill updates needed for one logical change.

Never commit `.env`, `.gcp-service-account.json`, `.w8-biayn/`, `.cache/upstreams/`, PIE data, CodeNet data, SuperCoder data, gem5 outputs, logs, rendered configs, or checkpoints.

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup, CLI, cloud, or data changes, also run relevant dry checks:

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
