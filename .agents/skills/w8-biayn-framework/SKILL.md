---
name: w8-biayn-framework
description: "Maintain, extend, test, document, and operate the w8-biayn C++ performance-RL pipeline now focused on SLIME-based Moonlight and GLM training: PIE task setup, SLIME JSONL conversion, Megatron/SGLang launch wrappers, Docker C++ rewards, local run receipts, and held-out uplift evaluation. Use for work in this repo, especially when touching SLIME, Moonlight, GLM, PIE data, C++ reward/eval, repo guidance, or legacy SkyRL/rLLM migration boundaries."
---

# w8-biayn Framework

Use this skill for work in this repository. Keep workflows reproducible from a
fresh clone and prefer repo-owned commands/wrappers over one-off shell history.

## Source Of Truth

Read these before changing behavior:

1. `AGENTS.md`
2. `README.md`
3. `ROADMAP.md`
4. Relevant code under `src/w8_biayn/`

If `/tmp/ENGINEERING_SPEC_v2_cpp_only.md` exists, it may provide historical
context, but checked-in guidance is the active source of truth.

## Active Goal

- Build runnable C++ optimization tasks from official PIE.
- Convert admitted tasks into SLIME SFT, GRPO, and eval JSONL.
- Train Moonlight and GLM lanes with SLIME, Megatron, and SGLang.
- Reward generated C++ by response format, compile/sanitizer correctness,
  visible and hidden tests, and child-process CPU-time runtime efficiency.
- Evaluate base, SFT, and GRPO outputs on held-out PIE tasks and prove uplift.

## Hard Boundaries

Do not write a custom trainer.

Do not use PIE's old Hugging Face Trainer path or any SuperCoder trainer as the
active trainer.

Do not reintroduce SkyRL/rLLM as the active training stack unless the user
explicitly asks for legacy maintenance or rollback.

Allowed upstream use:

- SLIME: active SFT/GRPO framework.
- PIE: C++ `v0 -> v1` data, official tests, and eval/data lessons.
- LearningOpt PIE: gem5 reference/calibration lessons when relevant.
- SuperCoder: schema, correctness, and eval lessons only.
- SkyRL/rLLM: legacy reference only.

Use `uv run w8-biayn upstreams clone` for pinned upstream copies under `.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor upstream repos or data. Experimental sidecar frameworks such as SLIME may be pinned for exploration only when explicitly requested; they must not replace the active SkyRL/rLLM C++ training path without an explicit project-phase change. When working on the SLIME sidecar lane, prefer the repo-owned `w8-biayn slime setup` Docker-first flow instead of trying to force SLIME runtime dependencies into the main project virtualenv. For an explicit SLIME C++ PIE run, build prompt JSONL with `w8-biayn data slime build` and use the separate `examples/slime/cpp_perf/` launcher and `generate_with_cpp_perf.py` reward hook. For the text-only SLIME bring-up path, prefer the repo-owned DAPO-Math prep script plus `examples/slime/multi_agent/run_multi_agent_text.sh` wrapper instead of editing the upstream example directly.

Use `uv run w8-biayn upstreams clone` for pinned upstream copies under
`.cache/upstreams/`. Temporary study clones may live under `/tmp`; do not vendor
upstream repos or data.

## Active Repository Map

- Bootstrap: `scripts/bootstrap.sh`
- CLI: `src/w8_biayn/cli.py`
- Dataset setup and manifests: `src/w8_biayn/cpp_perf/data.py`
- Coverage measurement: `src/w8_biayn/cpp_perf/coverage.py`
- PIE parsing/task construction: `src/w8_biayn/cpp_perf/pie.py`
- SkyRL dataset conversion: `src/w8_biayn/cpp_perf/skyrl_dataset.py`
- SLIME dataset conversion: `src/w8_biayn/cpp_perf/slime_dataset.py`
- Eval aggregation: `src/w8_biayn/cpp_perf/eval.py`
- Contest-style output judging: `src/w8_biayn/cpp_perf/judge.py`
- Task schema: `src/w8_biayn/cpp_perf/schema.py`
- Prompt/SFT helpers: `src/w8_biayn/cpp_perf/prompts.py`
- Reward and sandbox: `src/w8_biayn/cpp_perf/reward.py`,
  `src/w8_biayn/cpp_perf/sandbox.py`
- Eval aggregation: `src/w8_biayn/cpp_perf/eval.py`
- SLIME setup/doctor/sandbox helpers: `src/w8_biayn/slime_integration/`
- SLIME C++ bridge: `src/w8_biayn/integrations/slime_cpp_perf.py`
- W&B reporting layer (metrics/tables/artifacts/alerts/workspace):
  `src/w8_biayn/wandb_report.py`
- Pipeline milestone logger (standalone, host+container):
  `scripts/wandb_milestone.py`
- SLIME train entry wrapper:
  `src/w8_biayn/integrations/slime_train_entry.py`
- Moonlight HF export shim:
  `src/w8_biayn/integrations/slime_moonlight_hf_export.py`
- Moonlight Megatron local layer spec: `src/local.py`
- Moonlight C++ lane: `examples/slime/moonlight_cpp_perf/`
- Moonlight rank-16 LoRA C++ lane: `examples/slime/moonlight_lora_cpp_perf/`
- GLM C++ lane: `examples/slime/glm47_cpp_perf/` when present
- GLM agentic SWE-agent file-state C++ lane: `examples/slime/glm47_swe_agent_cpp_perf/`
- ReTool lane: `examples/slime/retool/`
- Moonlight MoE smoke: `examples/slime/moonlight_moe_smoke/`

Legacy SkyRL/rLLM/GCP control-plane files include
`src/w8_biayn/cpp_perf/skyrl_dataset.py`, `src/w8_biayn/sky_config.py`,
`src/w8_biayn/run_status.py`, `src/w8_biayn/mlflow_metrics.py`,
`src/w8_biayn/grpo_readiness.py`, `src/w8_biayn/integrations/skyrl_*.py`,
`src/w8_biayn/integrations/cpp_perf_env.py`, and
`src/w8_biayn/integrations/cpp_eval_main.py`. Leave them alone unless the task
explicitly asks for legacy work.

## Required User Path

A clean clone should support:

```bash
./scripts/bootstrap.sh
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
uv run w8-biayn slime setup
uv run w8-biayn cpp harness preflight --dry-run
```

If a change breaks this path, update code, tests, README, AGENTS/CLAUDE, and
this skill together.

## Data Workflow

No one-off dataset munging. Every conversion or cleanup must be a CLI-backed
project command or repo-owned script.

Build official PIE task JSON:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full --out .w8-biayn/data/pie-full/coverage.json --report-out .w8-biayn/data/pie-full/coverage-report.json
uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full --coverage-json .w8-biayn/data/pie-full/coverage.json --out .w8-biayn/data/tasks-full --min-train 1000 --min-validation 100 --min-test 100 --force
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/skyrl-full --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
# Optional explicit SLIME C++ lane; keep prompt-only JSONL plus task metadata for the SLIME reward hook.
uv run w8-biayn data slime build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/slime-pie --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl-full --gcs-prefix "gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/full-official/${RUN_ID}/skyrl" --credentials .gcp-service-account.json
```

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- reference performance exists.

Generated local data belongs under `.w8-biayn/` and is ignored by git.

## PIE Task Rules

- `v0` slower code becomes the prompt.
- `v1` fast code is not shown during GRPO.
- `v1` may be used as SFT target, coverage/reference input, and oracle
  material.
- Train/validation/test split stays by problem.
- A task requires visible tests, hidden tests, reference performance, and at
  least 95 percent line / 85 percent branch coverage.

The prompt may include visible tests and `v0`. It must not include hidden tests
or `v1`.

## Reward Rules

Model outputs must contain exactly one `<reasoning>...</reasoning>` block
followed by exactly one fenced C++ code block. The code may start on the next
line or after whitespace on the opening C++ fence line; any second code block is
invalid.

The reward is correctness gated:

- invalid format: negative;
- recoverable C++ with missing wrapper/fence format: shaped below the
  correctness-only fallback;
- compile or sanitizer failure: negative;
- timeout: negative;
- partial tests: below any fully correct answer;
- fully correct with missing non-timeout runtime measurement:
  correctness-only fallback below any measured fully correct answer;
- fully correct: base reward plus bounded runtime-efficiency;
- child-process CPU time in nanoseconds: RL reward metric;
- wall-clock nanoseconds: diagnostic metric.

The sandbox compiles candidate and PIE `v1` oracle, runs visible and hidden
tests, then benchmarks both binaries in the same Docker sandbox with the same
CPU pinning, compiler flags, and tests. Do not add PMU, Linux perf, PERFMON, or
`perf_event_paranoid` dependencies.

## SLIME Training Workflow

Start inside the generated SLIME container:

```bash
uv run w8-biayn slime setup
.w8-biayn/slime/run-container.sh
```

Moonlight C++ lane:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_cpp_perf/eval_base.sh
bash examples/slime/moonlight_cpp_perf/sft.sh
bash examples/slime/moonlight_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_cpp_perf/grpo.sh
bash examples/slime/moonlight_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_cpp_perf/compare.sh
```

Moonlight rank-16 LoRA C++ lane:

```bash
bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_base.sh
bash examples/slime/moonlight_lora_cpp_perf/sft.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_lora_cpp_perf/grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/compare.sh
```

The LoRA lane resolves rank-16 LoRA flags from the active SLIME/Megatron help
surface and fails before training if the runtime does not advertise a supported
LoRA rank flag.

GLM C++ lane when present:

```bash
bash examples/slime/glm47_cpp_perf/prepare_data.sh
bash examples/slime/glm47_cpp_perf/eval_base.sh
bash examples/slime/glm47_cpp_perf/sft.sh
bash examples/slime/glm47_cpp_perf/eval_sft.sh
bash examples/slime/glm47_cpp_perf/grpo.sh
bash examples/slime/glm47_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_cpp_perf/compare.sh
```

GLM agentic SWE-agent file-state lane. SWE-agent (not claude-code) edits
`candidate.cpp`; the final FILE is graded in the repo's Docker sandbox (not E2B)
and the correctness-gated reward flows via the OpenAI adapter `finish_session`.
The `generate` hook is `w8_biayn.integrations.slime_swe_agent_cpp_perf.generate`;
the driver is `swe_agent_driver.py`; the grader seam is
`w8_biayn.cpp_perf.sandbox.run_in_directory_prewritten`. SFT stays single-turn;
base/sft/grpo evals and GRPO are agentic. SWE-agent runs its edit/bash loop
in-process on the rollout worker via swerex `LocalDeployment` (`{"type":
"local"}`) — there is no sibling execution container and no `--network host`
(which collided with Ray's `127.0.0.1` node-ip), and each concurrent rollout
copies the repo to a unique basename so drivers do not share one working tree.
Only final-file grading re-enters the hardened `w8-biayn-cpp-perf` Docker image;
the `swe-image` CLI/dockerfile stays for reference but is off the hot path.

```bash
bash examples/slime/glm47_swe_agent_cpp_perf/prepare_data.sh
bash examples/slime/glm47_swe_agent_cpp_perf/eval_base.sh
bash examples/slime/glm47_swe_agent_cpp_perf/sft.sh
bash examples/slime/glm47_swe_agent_cpp_perf/eval_sft.sh
bash examples/slime/glm47_swe_agent_cpp_perf/grpo.sh
bash examples/slime/glm47_swe_agent_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_swe_agent_cpp_perf/compare.sh
```

Status (after four bounded A100-80GB:8 spot smokes): the agentic loop is
GPU-proven end-to-end short of one clean GRPO train step — base-eval runs 48
SWE-agent episodes at 0% aborts / 87.5% pass, SFT (re)trains and exports,
sft-eval grades the SFT model, and GRPO completes its rollout into the trainer.

Hard-won invariants for this lane (each one cost a paid smoke; keep them):

- **swerex LocalDeployment shares one filesystem across episodes AND stages.**
  Its `upload` is bare `shutil.copytree` (no `dirs_exist_ok`); SWE-agent
  uploads tool bundles to the fixed `/root/tools/{bundle}`. The driver
  monkeypatches `LocalRuntime.upload` to be idempotent
  (`_patch_swerex_local_upload`, a vendored-surface shim — re-check on swe-rex
  pin bumps), uses a unique repo basename per attempt, and rmtrees the root-FS
  copy afterward. Also note `/root/state.json` is shared by concurrent
  episodes (tool-state bleed; accepted for now).
- **HF-export gates require real weight shards** (`hf_export_ready`), restores
  prune weightless exports, and the GCS persist is loud with one retry — an
  index-without-shards restore once hung SGLang ~85 min.
- **Every trainer-reaching sample needs `metadata.round_number`** (slime's
  `--log-multi-turn` KeyErrors otherwise); abort husks reach the trainer, so
  `_abort_result` stamps it too, plus `abort_reason`/`abort_error` (exception
  message — never strip it to just the type name).
- Group size `--grpo-n-samples-per-prompt` (default 8; 1 zeroes every
  group-relative advantage, `kl=nan`); global batch derives as
  rollout_batch × n_samples. The sid rides in the request body via SWE-agent's
  `completion_kwargs` (`user` + `extra_body.metadata.session_id`) besides the
  bearer.

W&B is the primary observability surface (module `src/w8_biayn/wandb_report.py`;
one launch = one group = run id; per-stage runs `<run-id>-<stage>` with
deterministic ids via `WANDB_RUN_ID` + a train-entry rename shim because pinned
slime ignores `--wandb-run-id`). Where data goes: SLIME `train/*`/`rollout/*`
natively; live `rollout_health/*` (abort reasons, zero-variance group fraction,
correctness-gate rates, agent steps) from the generate hook via shared mode;
`eval/*` + per-task tables + `eval/abort/<reason>` from the offline scorer onto
the stage's own run (same keys overlay base/sft/grpo); uplift table + summary
and launch outcome/config/checkpoint-reference-artifact on `<run-id>-pipeline`;
`pipeline/elapsed_seconds` + a timeline table for milestones (never raw unix
scalars); `wandb.alert` on all-abort evals, >30% abort rate, and failed
launches. Push the curated saved view with
`uv run --extra cloud w8-biayn wandb workspace` (wandb + wandb-workspaces live
in the cloud extra).

For a GPU box you SSH into yourself (no GCP/SkyPilot/Google keys), follow the
README "SSH Manual Run" section: host setup + local PIE build, then
`.w8-biayn/slime/run-container.sh` and the lane stages with `SLIME_*` env
knobs; only `WANDB_KEY` from `.env` is needed, and checkpoints stay local.

Use local receipts (`run.log`, `run_receipt.txt`, `vram_usage.csv`,
`vram_peak.txt`), debug rollout dumps, W&B links when configured, and eval
summaries for evidence. The paid GCP GLM full launch is
`uv run --extra cloud w8-biayn launch glm47-full` (dry-run first with
`--dry-run`; implementation `src/w8_biayn/cloud_launch.py`, with the old
`examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py` kept as a thin shim).
Keep it a provisioning wrapper around the repo-owned GLM SLIME lane, with
dry-run rendering, scoped secrets, downloaded local artifacts, labels, spot
support via `--use-spot`, and automatic teardown. The W&B key resolves from
`--wandb-api-key-file`, `WANDB_API_KEY`, or a `WANDB_KEY` entry in `.env`
(`.env.sample` documents the only key read from `.env`).

The launch restores `tasks-full` from a project-scoped, gate-keyed GCS cache
(`gs://<project>-w8-biayn/cache/<version>/tasks-full/mintrain..-minval..-mintest..`)
before building, builds only on a miss, and repopulates the cache after — so
the PIE build runs once per (cache version, admission gates) and is shared
across users. It provisions a 1024GB boot disk by default (`--disk-size`): the
30B model is staged ~4x (HF download, torch_dist conversion, SFT checkpoint,
HF export) and 256GB overflows mid-export. Each launch also writes a
`<run-id>-pipeline` W&B run with timestamped preamble milestones.

Checkpoints are ephemeral unless persisted: on exit the launch rsyncs the run's
Megatron checkpoints and HF exports to `gs://<project>-w8-biayn/runs/glm47/<run-id>/`
(even for a partially-completed run), and `--resume-from-run <old-id>` restores
them into a new run and sets `SLIME_RESUME_SKIP_COMPLETED=1` so the lane skips
training stages whose Megatron checkpoint and HF export already exist
(`stage_already_complete` in `glm47_cpp_perf.sh`). Never assume a torn-down run
is resumable unless its checkpoints reached GCS — the launch teardown or
`ops down-run` deletes the node's disk.

## Tool Interlinkage

`w8-biayn` is the single management CLI. It wraps deeper tools; when a cloud
or training behavior looks wrong at the CLI level, drop down to the wrapped
tool and its documentation:

- Cloud hardware: `w8-biayn launch glm47-full` and `w8-biayn ops
  status|logs|down|queue` wrap SkyPilot, pinned in
  `w8_biayn.constants.SKYPILOT_PIN` and installed through the `cloud` extra
  (`uv run --extra cloud ...`). Debug directly with `uv run --extra cloud sky
  status|logs|down` and https://docs.skypilot.co. Semantics that matter:
  `sky.launch` on API-server builds resolves at job submission, so the CLI
  tracks the job to a terminal state before declaring success; the client
  version must match any locally running sky API server.
- Teardown safety (three layers): (1) the launch downloads artifacts and downs
  the cluster on any exit — but a killed/hung launch process cannot run its
  `finally` block; (2) so the launch arms SkyPilot `idle_minutes_to_autostop`
  (default 20, `--idle-autostop-minutes`, `0` disables) with `down=True`, and
  the on-cluster autostop *terminates* the cluster once the job ends and it
  idles, independent of the launcher process (this closes the gap that once
  orphaned a paid A100 for ~6 hours when a background launcher was SIGKILLed
  mid-teardown); (3) as a last resort every instance is tagged
  `labels.run_id=<run>` (the W&B group id) so
  `w8-biayn ops down-run <run-id> --execute` is the launcher-independent reaper
  that downs the cluster by name and deletes any GCE instance still carrying the
  label. After any launch, confirm no orphan with
  `gcloud compute instances list --filter=labels.project=w8-biayn`.
- Training: lane scripts wrap SLIME (pinned checkout at
  `.cache/upstreams/slime`, docs under `.cache/upstreams/slime/docs`).
  Megatron model args come from `scripts/models/*.sh` inside that checkout;
  the SLIME train loop is vendored in
  `src/w8_biayn/integrations/slime_train_entry.py` and must be re-diffed on
  pin bumps. The GLM container run also fetches/checks out `SLIME_PIN` (no
  drift to HEAD) and applies an in-place Megatron dist-checkpoint patch (in
  `cloud_launch.build_container_script`) so converted checkpoints load: TE
  `_extra_state` objects that HF conversion writes in a format the trainer
  cannot read are nulled, and `slime_train_entry` disables
  `ckpt_fully_parallel_load`. Re-verify both on a SLIME/Megatron/TE bump.
- GLM lane on non-Hopper GPUs: the default `alltoall` MoE dispatcher and
  `EP=4` (ETP*EP*PP must divide the 8-GPU world) are required on A100; DeepEP
  `flex` and `EP=8` are Hopper/16-rank only. GLM-4.7-Flash is a thinking model,
  so response budgets are 2048 and seq-length 4096 (512 truncated 100% of
  generations). These live in `examples/slime/glm47_cpp_perf/glm47_cpp_perf.sh`
  with `SLIME_GLM_*`/`SLIME_*` env overrides.
- GCP accounting and quotas: use `gcloud compute instances list` and the Cloud
  Quotas API with the scoped service-account env (never `gcloud auth
  activate-service-account`). Newer GPU quota limits (H100 class) are visible
  only through the Cloud Quotas API, and on-demand H100 is not self-service —
  spot/preemptible quota is.

## Git And Artifact Hygiene

Do not delete user files unless explicitly asked. If a generated artifact is
tracked but should not be in git, use `git rm --cached` so the working-tree file
remains available.

Never commit `.env`, `.gcp-service-account.json`, `.w8-biayn/`,
`.cache/upstreams/`, PIE data, CodeNet data, SuperCoder data, gem5 outputs,
logs, rendered configs, checkpoints, model exports, generated `RUN_REPORT*`
files, or generated report asset directories.

## Documentation Rules

When commands, setup, dataset shape, cache behavior, task schema, reward logic,
launch flow, benchmark protocol, or supported active pipelines change, update:

1. `README.md`
2. `ROADMAP.md`
3. `.agents/REPO_GUIDE.md`
4. this skill
5. tests when command behavior changes

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
uv run w8-biayn slime doctor
uv run w8-biayn slime setup --force
uv run w8-biayn cpp harness preflight --dry-run
```
