---
name: w8-biayn-framework
description: "Maintain, extend, test, document, and operate the w8-biayn C++ performance-RL pipeline: full PIE data setup, C++ task JSON, runtime/gem5 rewards, SkyRL/rLLM training, SkyPilot rendering, GCS dataset caching, labeled GCP operations, and uplift evaluation."
---

# w8-biayn Framework

Use this skill for work in this repository. The repo is intended for outside users and coding agents, so every workflow must be reproducible from a fresh clone.

## Source Of Truth

Read these before changing behavior:

1. `AGENTS.md`
2. `README.md`
3. `/tmp/ENGINEERING_SPEC_v2_cpp_only.md`
4. Relevant code under `src/w8_biayn/`

Active goal:

- Build runnable C++ optimization tasks from official PIE.
- Convert tasks into SkyRL/rLLM training datasets.
- Train with SkyRL SFT and SkyRL GRPO.
- Reward generated C++ by response format, compile/sanitizer correctness, visible and hidden tests, and CPU-time runtime efficiency.
- Evaluate base, SFT, and GRPO on held-out PIE tasks and prove uplift.
- Use gem5 from LearningOpt PIE for calibration/final eval when that lane is implemented.

## Hard Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder training path as the active trainer.

Do not replace SkyRL/rLLM with another framework.

Allowed upstream use:

- PIE: C++ v0-to-v1 data, official tests, and eval/data lessons.
- LearningOpt PIE: gem5 reference/calibration environment.
- SuperCoder: schema, correctness, and eval lessons.
- SkyRL/rLLM: actual SFT and GRPO training stack.

Phase 1 is C++ only. Do not reintroduce BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, AndroidWorld, Go workflows, custom GPU kernel labs, or unrelated performance experiments unless the user explicitly starts a later phase.

Use `uv run w8-biayn upstreams clone` for pinned upstream copies under `.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor upstream repos or data.

## Repository Map

`w8-biayn` owns:

- Bootstrap: `scripts/bootstrap.sh`
- CLI: `src/w8_biayn/cli.py`
- Dataset setup and manifests: `src/w8_biayn/cpp_perf/data.py`
- Coverage measurement: `src/w8_biayn/cpp_perf/coverage.py`
- PIE parsing/task construction: `src/w8_biayn/cpp_perf/pie.py`
- SkyRL dataset conversion: `src/w8_biayn/cpp_perf/skyrl_dataset.py`
- Eval aggregation: `src/w8_biayn/cpp_perf/eval.py`
- Contest-style output judging: `src/w8_biayn/cpp_perf/judge.py`
- Task schema: `src/w8_biayn/cpp_perf/schema.py`
- Sandbox/reward: `src/w8_biayn/cpp_perf/sandbox.py`, `src/w8_biayn/cpp_perf/reward.py`
- SkyRL env and entrypoint glue: `src/w8_biayn/integrations/cpp_perf_env.py`, `src/w8_biayn/integrations/skyrl_cpp_perf_main.py`
- SkyRL policy checkpoint HF export recovery: `src/w8_biayn/integrations/skyrl_sft_export_checkpoint_main.py`
- SkyRL checkpoint download compatibility patch: `src/w8_biayn/integrations/skyrl_io_patch.py`
- SkyRL vLLM token/logprob alignment patch: `src/w8_biayn/integrations/skyrl_vllm_logprob_patch.py`
- SkyRL GRPO health metric logging patch: `src/w8_biayn/integrations/skyrl_grpo_health_patch.py`
- Eval entrypoint: `src/w8_biayn/integrations/cpp_eval_main.py`
- GRPO readiness guardrails: `src/w8_biayn/grpo_readiness.py`
- Run status snapshots: `src/w8_biayn/run_status.py`
- Dry-run subprocess wrapper: `src/w8_biayn/shell.py`
- SkyPilot rendering: `src/w8_biayn/sky_config.py`
- Scoped GCP auth: `src/w8_biayn/gcp_auth.py`, `src/w8_biayn/secrets.py`
- Upstream pins: `src/w8_biayn/constants.py`, `src/w8_biayn/upstreams.py`
- Benchmark ladder: `src/w8_biayn/benchmarks.py`
- User docs: `README.md`
- Agent guide: `.agents/REPO_GUIDE.md`

## Required User Path

A new user must be able to run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
```

If a change breaks this path, update code, tests, README, AGENTS/CLAUDE, and this skill together.

## Data Workflow

No one-off dataset munging. Every conversion or cleanup must be a CLI-backed project command.

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

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- bundle manifest verifies before upload.

Generated local data belongs under `.w8-biayn/data/` and is ignored by git.

Derived SkyRL bundles must include:

- `grpo/train.parquet`
- `grpo/validation.parquet`
- `sft/train.jsonl`
- `sft/validation.jsonl`
- copied task JSON under `tasks/`
- `_w8_data_manifest.json` with schema version, sources, options, checksums, and byte sizes

Default schema version: `cpp-perf-v1`.

Use full-run GCS prefixes under:

```text
gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/<RUN_ID>/skyrl
```

Do not overwrite full-run data prefixes without an explicit user request.

## PIE Task Rules

Preserve PIE discipline:

- `v0` slower code becomes the prompt.
- `v1` fast code is not shown during GRPO.
- `v1` may be used as SFT target, coverage/reference input, and oracle material.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and at least 95 percent line / 85 percent branch coverage.

The prompt may include visible tests and `v0`. It must not include hidden tests or `v1`.

## Reward Rules

Model outputs must contain exactly one `<reasoning>...</reasoning>` block followed by exactly one fenced C++ code block. The code may start on the next line or after whitespace on the opening C++ fence line; any second code block is invalid.

The reward is correctness gated:

- Invalid format: negative.
- Recoverable C++ with missing wrapper/fence format: shaped below the correctness-only fallback, so GRPO has a gradient out of format collapse without making bare code a target.
- Compile or sanitizer failure: negative.
- Timeout: negative.
- Partial tests: shaped but below any fully correct answer.
- Fully correct with missing non-timeout runtime measurement: correctness-only fallback below any measured fully correct answer.
- Fully correct: base reward plus bounded runtime-efficiency.
- Child-process CPU time in nanoseconds: RL reward metric.
- Wall-clock nanoseconds: diagnostic metric.
- gem5: calibration/final-eval reference only.
- `w8-biayn cpp harness preflight`: required before GRPO to prove Docker runtime measurement works.

For real scoring, the sandbox image must contain `g++`, `bash`, `python3`, and `taskset`. The CLI default is `w8-biayn-cpp-perf:latest`, built locally from `gcc:13` with `python3`; pass `--image --no-build-image` only for a known-good prebuilt image.

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and hidden tests, then benchmarks both binaries in the same Docker sandbox with the same CPU pinning, compiler flags, and tests. Default timing uses 1 warmup and 3 measured repeats, median CPU time per test, summed across tests. Do not add PMU, Linux perf, PERFMON, or `perf_event_paranoid` dependencies to the active reward path.

## Cloud, Training, And Cleanup

All GCP/SkyPilot commands must support dry-run rendering before paid launch.

Cloud commands must:

- use `.gcp-service-account.json` through scoped env vars;
- avoid `gcloud auth activate-service-account`;
- avoid mutating global `gcloud config`;
- avoid printing credential contents;
- render YAML into `.w8-biayn/rendered/`;
- label paid resources with `project`, `phase`, `pipeline`, `run_id`, `owner`, and `ttl`;
- use `w8-biayn ops status`, `w8-biayn ops run-status`, `w8-biayn ops grpo-readiness`, `w8-biayn ops logs`, `w8-biayn ops queue`, `w8-biayn ops cancel`, `w8-biayn ops down`, `w8-biayn ops gpus`, and `w8-biayn gcp cleanup`.

Use `w8-biayn ops run-status --run-id "$RUN_ID" --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --out ".w8-biayn/runs/$RUN_ID/status.json"` for machine-readable run polling. For reruns or cluster-size experiments, pass prior snapshots with `--baseline-status ".w8-biayn/runs/<baseline-run-id>/status.json"` so the output includes `speed_comparison`; do not answer "is it faster?" from GPU count alone. Speedup factors greater than `1.0` are faster, factors less than `1.0` are slower, and `gpu_speedup_efficiency` is speedup divided by GPU scale. A `cost_verdict` of `cost_inefficient` means more GPUs did not increase the primary comparable throughput. The JSON schema is `w8-run-status-v1` and includes dataset manifest state, per-pipeline cluster/job state, labeled GCP instances, checkpoint marker and shard completeness for the promoted `latest` checkpoint, highest checkpoint directory, active `in_progress` checkpoint upload, export readiness including final export object counts/bytes and model weight presence, recent log-derived stage/step/checkpoint/export/error signals, normalized phase/progress/resource/command fields, SFT config/last-step progress including micro train batch and SkyRL timeout settings, GRPO config (`effective_samples_per_step`, total GPUs, samples/GPU/step, micro train batch, KL/entropy settings, vLLM GPU memory utilization, checkpoint retention, reward workers, FSDP sizes, HSDP mesh/activity, SkyRL timeout settings), cpp-eval generation/scoring progress, trajectory/evaluation/training throughput, GPU-normalized throughput, ETA/timing metrics, reward metrics, GRPO `training_health`, `learning_signal`, and `phase_timing` verdicts, bottleneck verdicts from SkyRL timing, optional baseline speed comparison with GPU speedup efficiency/cost verdict, and cleanup safety. Config fields are merged from logs and the local rendered YAML, with `logs.config_sources` and `logs.rendered_config_path`, so long-running dashboards keep stable batch/checkpoint/FSDP settings even after the launch command scrolls out of the log tail. Logs include `tail_lines_requested`, `tail_lines_scanned`, and `tail_may_be_truncated`; increase `--log-tail` before relying on stage parsing when the tail may be truncated. Pass `--node-health` for opt-in read-only SSH health with GPU utilization/memory, disk free space, top processes, derived node activity, and explicit `sample_scope`. On multi-node clusters, `--node-health` must probe every active labeled pipeline VM so dashboards can verify worker-node participation instead of inferring it from the head node. Instantaneous GPU utilization is phase-dependent: `resources.gpu_sample_interpretation` and `progress.phase_timing` must be read together before concluding a run is GPU-bound, reward-bound, idle, or healthy. Each backend/GCS/health check includes its command, return code, `timed_out`, and `attempt_count`; tune timeout with `--check-timeout` and retry timed-out read-only checks with `--check-retries`.

`w8-run-status-v1` structure for dashboards and agents:

```text
root
  schema_version: "w8-run-status-v1"
  generated_at_utc, run_id, project_id, artifact_bucket
  summary
    state, dataset_state, current_pipeline, current_cluster
    current_phase: {current, group, source, log_stage, node_activity, artifact_activity, failed, message}
    current_progress: {primary, training, trajectory, throughput, training_health, learning_signal, phase_timing}
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
    logs
      stage, stage_events[], last_step, last_loss, last_eval_loss
      trajectory_progress, evaluation_progress, training_progress
      checkpoint_events[], export_events[], timings, metrics, policy_health_events[], grpo_health_events[], errors[]
      config, config_sources[], rendered_config_path
      tail_lines_requested, tail_lines_scanned, tail_may_be_truncated
      available, error
    phase: {current, group, source, log_stage, node_activity, artifact_activity, failed, message}
    progress
      pipeline, primary, training, trajectory, evaluation, checkpoint
      timings, config, sft_config, grpo_config, metrics, training_health, learning_signal, phase_timing, bottleneck, throughput
    speed_comparison: {available, reason, primary, baselines[]}
    recovery: {available, recommended_action, fresh_cluster_recommended, requires_down_before_resume, reason, signals[], resume_from, resume_checkpoint_step, commands}
    resources: {total_instance_count, active_instance_count, zones[], machine_types[], accelerators, gpu_count, sampled_node_count, failed_node_count, sampled_gpu_count, gpu utilization/memory summaries, gpu_sample_interpretation}
    commands, checks[]
    node_health: {available, skipped, sample_scope, sample_note, expected_node_count, sampled_node_count, failed_node_count, activity, gpus[], filesystems[], processes[], nodes[], error}
                                           # only with --node-health
  gcp: {instances[], active_instance_count, checks[]}
  cleanup: {safe_to_cleanup, active_job_count, active_instance_count, provisioning_instance_count, working_pipeline_count, commands}
```

For UI work, bind to normalized fields first: `summary.current_phase`, `summary.current_progress`, `summary.training_health`, `summary.learning_signal`, `summary.recovery`, `pipelines[].progress.throughput`, `pipelines[].progress.training_health`, `pipelines[].progress.learning_signal`, `pipelines[].progress.phase_timing`, `pipelines[].progress.sft_config`, `pipelines[].progress.grpo_config`, `pipelines[].speed_comparison.primary`, `pipelines[].recovery`, `pipelines[].resources`, `pipelines[].node_health`, `pipelines[].artifacts.checkpoint.latest.resumable`, `pipelines[].artifacts.export.final_export.complete`, `pipelines[].artifacts.eval_outputs.complete_labels`, and `cleanup.safe_to_cleanup`. Each `checks[]` entry has `name`, `command[]`, `ok`, `returncode`, `skipped`, `timed_out`, `attempt_count`, and optional stdout/stderr tails. With `--node-health`, `sample_scope=all_active` means every active labeled VM for the pipeline was probed, `partial` means at least one active VM probe failed, and `none` means no live node telemetry was collected. Top-level `node_health.gpus[]`, `filesystems[]`, and `processes[]` are aggregates annotated with `node_name` and `role`; `node_health.nodes[]` keeps per-node detail. Checkpoint detail objects expose shard completeness, bytes, and `resumable`; when an active job has an incomplete higher checkpoint than the promoted `latest`, `phase.source=artifacts` and `phase.artifact_activity=checkpoint_stage` protect dashboards from stale or truncated log tails. Export detail objects expose config/tokenizer/weight presence, bytes, sampled objects, and `complete`. Eval output records and summaries expose uploaded artifacts by label; `complete_labels` contains labels with both records and summary files. `training_health.verdict=collapsed` with `should_stop=true` sets `training_health.recommended_action=stop_and_relaunch_from_sft`; `training_health.verdict=deterministic_low_gradient` with `should_stop=true` sets `training_health.recommended_action=stop_and_evaluate_checkpoint` and `training_health.checkpoint_step`; `training_health.verdict=deterministic_convergence_risk` keeps `should_stop=false` but sets `recommended_action=evaluate_checkpoint` when train reward/pass are high while entropy, gradients, advantages, or reward variance suggest learning may be stalled. `learning_signal` is built from `W8_GRPO_HEALTH` and SkyRL metrics; it exposes train reward/pass, held-out eval metrics when available, policy KL, entropy, estimated KL/entropy loss terms, reward-group variance, mean absolute advantage, zero-advantage fraction, recent trends, and a dashboard verdict. `phase_timing` groups SkyRL timing into rollout, reward, policy-update, synchronization, checkpoint, and export work; use it with `resources.gpu_sample_interpretation` before deciding whether a 0 percent or 100 percent GPU sample means anything about throughput. `recovery.available=true` means a failed/canceled checkpointed training job has a machine-readable next action; `recommended_action=fresh_cluster_resume` means the latest checkpoint is resumable but distributed-state or interrupted-export signals make same-cluster reuse risky, so down the cluster before relaunching with `--resume-from latest`. `speed_comparison.primary` exists only with comparable `--baseline-status` inputs; factors greater than `1.0` are faster, factors less than `1.0` are slower. `cleanup.safe_to_cleanup=false` means the shared account still has active/provisioning work for the run.

Do not put raw `sky ...` commands in docs, runbooks, or handoffs. Treat SkyPilot/SkyRL calls as backend details behind the `w8-biayn` CLI so the operator DX stays stable if the backend changes.

Training renders and launches through SkyPilot:

```bash
uv run w8-biayn config render cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn config render cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID"
uv run w8-biayn launch cpp-sft --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --accelerators A100:8 --disk-size 1024 --train-batch-size 16 --no-down-after
uv run w8-biayn launch cpp-grpo --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports/global_step_1074/policy" --accelerators A100:8 --disk-size 1024 --train-batch-size 16 --n-samples-per-prompt 8 --train-epochs 3 --no-eval-before-train --eval-interval 25 --max-env-workers 128 --ckpt-interval 50 --hf-save-interval 10000 --ckpt-path "$RUN_GCS/cpp-grpo/ckpts" --export-path "$RUN_GCS/cpp-grpo/exports" --max-ckpts-to-keep 8 --grpo-vllm-gpu-memory-utilization 0.7 --no-down-after
```

`launch` passes SkyPilot `--down` by default. Use `--no-down-after` only for training runs that need post-run inspection.
Full training and eval default to a 1024 GB boot disk, except resumed multi-node GRPO, which defaults to and requires 2048 GB or larger because FSDP checkpoint restore needs substantial local scratch space.
If an SFT attempt writes checkpoints and then fails, resume the same run with `--resume-from latest`, `--disk-size 1024` or larger, and the same `--ckpt-path`.
Rendered SFT and GRPO containers must export longer SkyRL/Ray distributed timeouts (`SKYRL_RAY_PG_TIMEOUT_IN_S=1800`, `SKYRL_WORKER_NCCL_TIMEOUT_IN_S=3600`) so slow FSDP checkpoint restore and HF export barriers do not fail at SkyRL's 10-minute worker default. Override them only for a deliberate debugging run, and keep the values visible in `ops run-status`.
Rendered training containers must also pass `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=^lo,docker,veth`, a concrete default-route `GLOO_SOCKET_IFNAME`, and `NCCL_DEBUG=WARN` into Docker so SkyRL, Ray, and vLLM use the VM network interface instead of loopback or transient container interfaces. Missing this propagation is a known multi-node reproducibility mismatch: the host shell can have valid networking while the training container still fails during Ray/NCCL/vLLM startup. Do not reuse NCCL's `^lo,docker,veth` exclusion syntax for Gloo; PyTorch Gloo expects a real interface name and will fail process-group initialization if given `^lo`.
Rendered training containers must also apply idempotent SkyRL compatibility patches: `skyrl_io_patch.py` flattens cloud checkpoint directory downloads before FSDP rank-shard validation, `skyrl_vllm_logprob_patch.py` aligns vLLM response token IDs with returned rollout logprobs when vLLM emits an unpaired trailing token at the generation cap, and `skyrl_grpo_health_patch.py` emits `W8_GRPO_HEALTH` JSON lines for run-status learning-signal dashboards. Without these patches, GCS resume can falsely report missing `model_world_size_*_rank_*.pt` shards, GRPO can abort on SkyRL's response/logprob length assertion, or dashboards can miss deterministic-convergence and overfit signals.
Rendered GRPO enables a small KL anchor and entropy bonus by default (`--grpo-use-kl-loss`, `--grpo-kl-loss-coef 0.001`, `--grpo-use-entropy-loss`, `--grpo-entropy-loss-coef 0.001`) to reduce drift from the SFT/reference policy and avoid deterministic format collapse. It also sets `--grpo-vllm-gpu-memory-utilization 0.7` by default so colocated vLLM leaves enough headroom for FSDP policy/ref workers on A100-40GB; if vLLM fails while waking the KV cache, lower this before relaunching. Keep at least several GRPO checkpoints (`--max-ckpts-to-keep 8` in the full-run command) because the latest checkpoint can be worse than an earlier healthy checkpoint after reward over-optimization.
If a GRPO attempt writes a complete checkpoint and then fails or is canceled, resume the same run with `--resume-from latest` and the same `--ckpt-path`. Use `--disk-size 1024` or larger for single-node GRPO resume and `--disk-size 2048` or larger for multi-node GRPO resume; a 2x[A100:8] resume from `global_step_150` filled a 1024 GB boot disk during FSDP checkpoint restore and failed with `[Errno 28] No space left on device`, so the CLI rejects smaller explicit disks and defaults resumed multi-node GRPO to 2048 GB. When the final SFT step is known, pass the concrete HF policy export (`.../exports/global_step_N/policy`) as `--model`; if a `gs://.../exports` root is passed, rendered GRPO/eval jobs resolve it to the highest complete `global_step_N/policy` directory with model weights before staging it locally. For multi-node GRPO, pass `--num-nodes N`; rendered colocated SkyRL starts a rank-gated Ray cluster inside the GPU containers, runs `skyrl_cpp_perf_main` only on rank 0, and sets rollout engines to total policy GPUs (`N * GPUs_PER_NODE`) when TP=1/DP=1, and sets `trainer.policy.fsdp_config.fsdp_size`/`trainer.ref.fsdp_config.fsdp_size` to GPUs-per-node so multi-node FSDP shards within a node and replicates across nodes (HSDP) rather than full-sharding the model across the slow inter-node link (otherwise a model that fits on one node can be slower on two). Reserve multi-node for models that do not fit on one node or generation/reward-bound steps. Multi-node GRPO must pass the CLI utilization gate: effective samples per step (`train_batch_size * n_samples_per_prompt`) must be at least 16 per GPU and `max_env_workers` must be at least the effective samples per step. For 2x[A100:8], use `--train-batch-size 32 --n-samples-per-prompt 8 --max-env-workers 256`; use `--allow-low-multinode-utilization` only for a deliberate experiment. `--micro-train-batch-size-per-gpu` defaults to `1`; try `2` only as a memory-checked policy-update tuning experiment on A100-40GB and fall back to `1` on OOM or no throughput gain. Use `--no-eval-before-train` to skip the expensive initial validation pass. Rendered GRPO uses non-batched trajectory generation so SkyRL-Gym can overlap C++ reward calls; raise `--max-env-workers` when Docker reward compilation/runtime is the bottleneck.
Before any paid multi-node GRPO training launch, `launch cpp-grpo --num-nodes N` runs `ops grpo-readiness` against the rendered YAML and blocks critical failures. Agents should run it directly when debugging multi-node setup:

```bash
uv run w8-biayn ops grpo-readiness --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml --out ".w8-biayn/runs/$RUN_ID/grpo-readiness.json"
uv run w8-biayn ops grpo-readiness --rendered-config .w8-biayn/rendered/cpp-grpo.sky.yaml --status-json ".w8-biayn/runs/$RUN_ID/status.json"
```

The readiness JSON schema is `w8-grpo-readiness-v1`. Static checks must validate Docker reward mounts, host/container preflights, `NCCL_IB_DISABLE`, `NCCL_SOCKET_IFNAME`, `NCCL_DEBUG`, concrete default-route `GLOO_SOCKET_IFNAME`, `skyrl_io_patch.py`, `skyrl_vllm_logprob_patch.py`, `skyrl_grpo_health_patch.py`, HSDP `fsdp_size`, rollout engine count, utilization gate, resume disk, checkpoint retention, and KL/entropy settings. Gloo must receive a real interface name; do not pass NCCL's `^lo,docker,veth` exclusion syntax to `GLOO_SOCKET_IFNAME`. For live multi-node claims, readiness must be run with a `--node-health` status snapshot and must show `node_health.sample_scope=all_active`, sampled nodes equal active nodes, failed node probes equal zero, and sampled GPUs equal the GRPO GPU count. If `training_health.should_stop=true` or `learning_signal.recommended_action=evaluate_checkpoint`, treat readiness `overall=action_required` as a stop/evaluate or relaunch instruction, not as a warning to keep polling.
If an SFT checkpoint is complete but the HF export has no model weight files, rerun `cpp-sft` as an export-only recovery with `--export-checkpoint "$RUN_GCS/cpp-sft/ckpts/global_step_N"`, `--export-path "~/exports/"`, `--disk-size 1024`, and `--no-down-after`; then verify `ops run-status` reports `artifacts.export.final_export_exists=true` and `artifacts.export.final_export.weight_object_count>0` before starting GRPO. If a GRPO checkpoint should be evaluated after an early stop, rerun `cpp-grpo` as an export-only recovery with the SFT HF export as `--model`, the GRPO checkpoint as `--export-checkpoint "$RUN_GCS/cpp-grpo/ckpts/global_step_N"`, and a local `--export-path`; the launcher uploads the policy export under `$RUN_GCS/cpp-grpo/exports`.
When GRPO or eval uses a `gs://` model export, stage it into a path mounted inside the GPU container before invoking SkyRL or vLLM; host-only staged paths are invalid. During staging, normalize legacy tokenizer configs by moving an `extra_special_tokens` list to `additional_special_tokens` so current Transformers can load SkyRL exports. Eval must use that mounted staged path directly rather than copying the same HF export again inside the container. If the model points at an export root with multiple `global_step_*` directories, resolve the highest complete `global_step_N/policy` export and validate local `config.json` plus model weight files before starting SkyRL or vLLM. Warm-cluster reruns may skip dataset restore only when the local prefix marker matches the requested `--dataset-gcs-prefix` and the manifest/tasks are present.
If a job is canceled or fails during distributed checkpoint restore or HF export, treat the warm cluster as suspect even when the GCS checkpoint is valid. `ops run-status` should expose `recovery.recommended_action=fresh_cluster_resume`; down the cluster, relaunch with `--resume-from latest`, the same `--ckpt-path`, and at least 1024 GB on single-node GRPO or 2048 GB on multi-node GRPO. If the mistake happens before Ray/SkyRL training starts, such as bad dataset/model staging, cancel the bad job and reuse the allocated cluster when possible so the corrected workdir can start as a new job without another scarce-GPU provisioning round. Do not infer paid-resource count from local `SkyPilot:executor:*` processes; SkyPilot keeps a local API server and executor pool alive, and actual paid resources must be checked through `w8-biayn ops status`, `w8-biayn ops queue`, `w8-biayn ops run-status`, and labeled GCP instance checks.

On the shared GCP account, run one full-training cluster at a time and clean up every failed or completed attempt:

```bash
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --dry-run
uv run w8-biayn gcp cleanup --run-id "$RUN_ID" --credentials .gcp-service-account.json --execute
```

GRPO reward execution uses Docker-outside-Docker: the GPU training container must mount `/var/run/docker.sock` and host `/tmp`.
Rendered GRPO and eval must run a host-side C++ runtime preflight before GCS restore, model staging, GPU image pulls, or framework installs. GRPO and eval still run the preflight again inside the GPU container before reward code starts. Rendered jobs must not require PMU access or mutate `kernel.perf_event_paranoid`.

The GRPO entrypoint is:

```bash
python -m w8_biayn.integrations.skyrl_cpp_perf_main
```

It registers `cpp-perf` inside SkyRL and delegates to SkyRL `BasePPOExp(cfg).run()`. It is glue, not a trainer.

Full runs should pass explicit `--train-epochs`, `--eval-interval`, `--ckpt-interval`, `--hf-save-interval`, `--ckpt-path`, `--export-path`, and `--max-ckpts-to-keep`. For full official SFT, keep `--ckpt-interval 100` for recovery but set `--hf-save-interval` to the expected final step (`1074` for two epochs at batch size 16 on the current full PIE bundle) unless a run intentionally needs intermediate HF exports; `save_hf_model` is CPU/GCS-heavy and should show up in `pipelines[].progress.bottleneck` when it dominates. SkyRL still saves HF models at epoch boundaries and after the loop, so a large interval suppresses interval exports but is not final-only. Full GRPO runs should also set `--no-eval-before-train`, a deliberate `--max-env-workers` value when reward execution is CPU/Docker-bound, a large `--hf-save-interval` such as `10000` unless intermediate HF exports are part of the experiment, and enough checkpoint retention (`--max-ckpts-to-keep 8` as the current full-run default). Resume SFT with `--resume-from latest` only after verifying the checkpoint prefix has a valid `latest_ckpt_global_step.txt`, and keep the same final-step `--hf-save-interval` on the resumed launch.

Training defaults use `Qwen/Qwen2.5-Coder-7B-Instruct` and `A100:8`. Smoke defaults use `zai-org/GLM-5.1` and `H100:8`. Eval defaults to `Qwen/Qwen2.5-Coder-7B-Instruct` and `A100:1`. Rendered training jobs request `128+` GB host memory; rendered eval jobs request `80+` GB so GCP single-A100 shapes are not filtered out before provisioning. Eval pins `vllm==0.6.6.post1`, `transformers==4.57.6`, and CUDA 12.4 PyTorch wheels because unpinned latest vLLM can select a CUDA 13 stack that the A2 driver cannot load.

## Evaluation Workflow

Use `cpp-eval` on the same held-out data for base, SFT, and GRPO:

```bash
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model Qwen/Qwen2.5-Coder-7B-Instruct --eval-label base --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-sft/exports/global_step_1074/policy" --eval-label sft --n-samples-per-prompt 1
uv run w8-biayn launch cpp-eval --credentials .gcp-service-account.json --dataset-gcs-prefix "$DATA_GCS" --run-id "$RUN_ID" --model "$RUN_GCS/cpp-grpo/exports" --eval-label grpo --n-samples-per-prompt 1
```

`cpp-eval` stages `gs://` model exports to local VM disk, normalizes legacy tokenizer config shape if needed, mounts the staged directory into the eval container, and then loads vLLM from that mounted path. After generation, `cpp_eval_main` releases the vLLM object and CUDA cache before CPU/Docker scoring, then emits `W8 eval generation ...` and `W8 eval scoring ...` markers that `ops run-status` maps to `eval_generation`, `eval_scoring`, and `progress.evaluation`. On warm clusters, dataset restore is skipped only when the local prefix marker matches the requested dataset prefix and the manifest/tasks are present.

Aggregate records:

```bash
uv run w8-biayn eval cpp --records base=base.records.jsonl --records sft=sft.records.jsonl --records grpo=grpo.records.jsonl --out uplift-summary.json
```

Uplift claim requires GRPO to beat base and SFT on `correct_and_faster_rate` and `mean_best_reward`, with `missing_runtime_rate=0`.

If uplift fails, clone/study SuperCoder and Microsoft/LearningOpt PIE into `/tmp`, compare filtering, prompts, reward shape, model choice, and hyperparameters, then port compatible fixes into this SkyRL/rLLM pipeline only.

## Documentation Rules

When commands, setup, dataset shape, cache behavior, cloud behavior, task schema, reward logic, launch flow, benchmark protocol, or supported pipelines change, update:

1. `README.md`
2. Mermaid diagrams in `README.md`
3. `.agents/REPO_GUIDE.md`
4. This skill
5. Tests when command behavior changes

## Validation

Before handing off:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For setup or CLI-surface changes:

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
