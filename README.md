# w8-biayn

`w8-biayn` is the command-and-control repository for C++ performance RL. The
current active work is SLIME-based training for Moonlight and GLM models on PIE
C++ optimization tasks.

The training target is narrow:

```text
correct but slower C++20 program -> correct and faster C++20 program
```

Correctness is mandatory. Runtime improvement only matters after the generated
program preserves behavior on visible and hidden tests.

## Active Goal

- Data: official PIE C++ slower-to-faster pairs plus official, merged, and
  generated tests.
- Task: prompt with slower PIE `v0`; generate a complete optimized C++20
  program.
- Reward: strict response format, compile and sanitizer success, visible and
  hidden tests, then bounded child-process CPU-time efficiency.
- Training: SLIME with Megatron training and SGLang rollouts. Active model
  lanes are Moonlight and GLM.
- Proof: compare base, SFT, and GRPO outputs on the same held-out PIE tasks
  with pass rate, correct-and-faster rate, mean reward, speedup, and
  missing-runtime rate.

SkyRL/rLLM, SkyPilot renderers, MLflow run-status parsing, and older GCP
training commands were removed from this branch. They remain available in git
history, and the CLI keeps only thin legacy shims (for example
`data skyrl build`) that fail with a clear legacy-unavailable message. Do not
use them for new active training work unless a task explicitly asks for legacy
maintenance.

Out of scope unless a later phase explicitly asks for it: BrowserGym, DOMDiff,
Harbor, WebArena, MiniWoB, AndroidWorld, Go, custom GPU kernel labs, and
unrelated performance experiments.

## Boundaries And Artifact Hygiene

Do not write a custom trainer. Use SLIME, Megatron, and SGLang for active
training work. Do not reintroduce SkyRL/rLLM as the active training stack unless
the user explicitly asks for legacy maintenance or rollback.

Do not delete local evidence files just because they are no longer tracked. If a
generated artifact is in git, use `git rm --cached` so the working-tree file
remains available, then ignore future generated copies. Generated `RUN_REPORT*`
files and report asset directories should not be committed.

## Current SLIME Lanes

Use repo-owned wrappers rather than editing `.cache/upstreams/slime` directly.

- Moonlight C++ performance lane: `examples/slime/moonlight_cpp_perf/`
- Moonlight rank-16 LoRA C++ performance lane: `examples/slime/moonlight_lora_cpp_perf/`
- GLM C++ performance lane: `examples/slime/glm47_cpp_perf/`
- GLM agentic SWE-agent C++ lane (multi-turn, file-state scored): `examples/slime/glm47_swe_agent_cpp_perf/`
- Moonlight ReTool lane: `examples/slime/retool/`
- Moonlight MoE smoke: `examples/slime/moonlight_moe_smoke/`
- Generic text-only SLIME smoke: `examples/slime/multi_agent/`

## GLM-4.7 Miles Lane On 8x H100

The repository also includes the measured Miles configuration for
GLM-4.7-Flash LoRA SFT and GRPO on one 8x H100 80 GB NVLink node:

- canonical SFT launcher: `examples/sft.sh`;
- canonical GRPO launcher: `examples/grpo.sh`;
- TP4 / PP1 / EP8 checkpoint conversion: `scripts/convert_checkpoint.sh`;
- pinned Hugging Face dataset and SFT adapter download:
  `scripts/download_assets.py`;
- W&B scalars, samples, evaluation tables, synchronization fingerprints, and
  checkpoint manifests.

The complete configuration, replication BOM, published artifacts, measured
base-to-SFT evaluation, and evaluation commands are documented in
[`docs/GLM47_H100_MILES.md`](docs/GLM47_H100_MILES.md).

The Moonlight and GLM C++ lanes reuse the project PIE task schema, prompt
builder, Docker C++ sandbox, reward function, and eval aggregation through
`src/w8_biayn/integrations/slime_cpp_perf.py`.

The GLM agentic SWE-agent lane grades the final edited FILE instead of model
text: SWE-agent edits `candidate.cpp` over many turns, the hardened Docker
grader scores the file it leaves behind (compile + visible and hidden tests +
child-process CPU time), and reward flows through SLIME's OpenAI adapter
`finish_session`. This dissolves the GLM thinking-mode truncation that made
single-turn responses unscoreable. It uses SWE-agent (not claude-code) and the
repo's Docker grader (not E2B), via
`src/w8_biayn/integrations/slime_swe_agent_cpp_perf.py` (the `generate` hook)
and `src/w8_biayn/integrations/swe_agent_driver.py`. SWE-agent runs its
edit/bash loop in-process on the rollout worker through swerex `LocalDeployment`
(`{"type": "local"}`) — no sibling execution container and no `--network host`
(which collided with Ray) — and each concurrent rollout copies the repo to a
unique basename so the drivers do not fight over one working tree. Only the
final-file grading crosses back into the hardened Docker sandbox.

**Status: the pipeline is proven END-TO-END.** Run `w8swe-20260707091714`
completed all seven stages in 78 minutes on an A100-80GB:8 spot box —
base-eval → SFT → sft-eval → **GRPO with a clean weight update** → grpo-eval →
compare — with the verdict on the live panels: `train/ppo_kl` finite,
`grad_norm` 5.6, `zero_variance_group_fraction` 0, `trained_tokens_mean` 161,
1/16 aborts, 87.5% eval pass across all three model stages, and the run's
checkpoints + HF exports fully persisted to GCS (resumable). The
eight-smoke campaign that got here converted every failure into a committed
fix and a regression lint (`tests/test_regression_lints.py`, one guard per
paid incident); the load-bearing lessons:

- **swerex LocalDeployment shares one filesystem across every episode and
  stage.** Its `upload` is a bare `shutil.copytree` (no `dirs_exist_ok`), and
  SWE-agent uploads tool bundles to the fixed path `/root/tools/{bundle}` —
  so episode 2+ died with `FileExistsError` before the first model call. The
  driver monkeypatches the upload to be idempotent, gives each attempt a
  unique repo basename, and removes the root-FS copy afterward.
- **HF-export gates must demand real weight shards.** A GCS persist that fails
  partway leaves `config.json` + `model.safetensors.index.json` without
  shards; the old existence gate accepted that and SGLang hung ~85 min on a
  weightless model. Gates now require `*.safetensors`/`*.bin`, restores prune
  weightless exports, and the persist rsync is loud with a retry.
- **Every sample that reaches the trainer must carry
  `metadata.round_number`** — slime's `--log-multi-turn` does a direct dict
  access, and abort husks do reach the trainer (also the mechanism behind the
  original all-abort NaN). Success and abort paths both stamp it.
- **GRPO needs exactly one trainable sample per episode.** GLM's chat template
  strips `<think>` from history, so turns re-tokenize past the adapter's fork
  threshold and a 3-turn episode drained 3 samples (48 rewards where the
  `(prompts × n_samples)` reshape expected 16). The lane raises the merge
  threshold and the hook keeps the fork with the most trained tokens,
  reporting drops as `rollout_health/fork_samples_dropped_mean`.
- **Spot preemption is a delay, not a hang.** A vanished cluster returns
  `CLUSTER_LOST` into the provisioning retry loop (it once ghost-polled for
  two hours); network reachability is preflighted before any spend and
  watchdogged throughout (`net_degraded`/`net_recovered` launch events,
  `w8-biayn ops net-check` for the manual probe).
- Earlier fixes hold: Ray vs `--network host` (dissolved by LocalDeployment),
  `tokenizers` pinned via frozen-env `--constraint`, group size
  `--grpo-n-samples-per-prompt` (default 8; 1 zeroes every group-relative
  advantage), sid carried in the request body besides the bearer, pinned
  upstream fetches skipped when the commit is already local.

Open before a full run: the fork-merge threshold did not actually prevent
3-way episode forks (the keep-best guard preserved group math but discards
~2/3 of captured tokens — investigate adapter REALIGN vs GLM think-stripping),
plus the pre-existing full-run gates (thinking budget, PIE admission
coverage, model/torch_dist GCS cache).

## Fresh Machine Setup

Run from a clean clone:

```bash
./scripts/bootstrap.sh
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
uv run w8-biayn slime setup
uv run w8-biayn cpp harness preflight --dry-run
```

Generated data, upstream clones, rendered launchers, secrets, logs,
checkpoints, model exports, run reports, and evaluation artifacts are local
state and must stay out of git.

If cloud/GCS helpers are used for a specific run, keep credentials local at
`.gcp-service-account.json`; never print credential contents or mutate global
`gcloud` configuration.

## PIE Data Workflow

Dataset conversion is a deliverable. Do not use one-off notebooks, shell
history, or untracked munging for PIE data.

Build admitted PIE task JSON:

```bash
RUN_ID="r$(date -u +%Y%m%d%H%M%S)"

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
```

Admission gates:

- train tasks >= 1000;
- validation/test tasks >= 100;
- coverage >= 95 percent line and 85 percent branch;
- visible and hidden tests exist;
- reference performance exists;
- train/validation/test split stays by problem.

Build SLIME-ready JSONL from the admitted task JSON with the lane wrapper:

```bash
bash examples/slime/moonlight_cpp_perf/prepare_data.sh
```

For the GLM lane, use the matching wrapper under
`examples/slime/glm47_cpp_perf/` when that lane is present in the workspace.

## Task And Reward Contract

A valid task contains:

- `prompt_code`: slower correct PIE C++ `v0`;
- `oracle_solution`: faster PIE C++ `v1`, used for SFT, coverage, reference
  timing, and oracle material only;
- visible `unit_tests` and grading-only `hidden_tests`;
- `test_coverage` at or above 95 percent line and 85 percent branch;
- positive reference performance metadata;
- split `train`, `validation`, or `test`.

During GRPO, the prompt may include visible tests and `v0`. It must not include
hidden tests or `v1`.

Model outputs must contain exactly:

````text
<reasoning>...</reasoning>
```cpp
// complete optimized C++20 program
```
````

Reward order:

- unrecoverable invalid format: negative;
- recoverable C++ with missing wrapper/fence format: shaped below the
  correctness-only fallback;
- compile or sanitizer failure: negative;
- timeout: negative;
- partial tests: below any fully correct answer;
- fully correct with missing non-timeout runtime measurement: correctness-only
  fallback below any measured fully correct answer;
- fully correct: base reward plus bounded runtime-efficiency bonus.

The sandbox compiles the candidate and PIE `v1` oracle, runs all visible and
hidden tests, then benchmarks both binaries in the same Docker sandbox with the
same CPU pinning, compiler flags, and tests. Runtime reward uses
child-process CPU time in nanoseconds. Wall-clock nanoseconds are diagnostics.

Do not add PMU, Linux perf, PERFMON, or `perf_event_paranoid` dependencies to
the active reward path.

## SLIME Setup

Clone or refresh the pinned SLIME checkout:

```bash
uv run w8-biayn upstreams clone slime
uv run w8-biayn slime doctor
```

Generate the Docker-first launcher and in-container bootstrap helper:

```bash
uv run w8-biayn slime setup
.w8-biayn/slime/run-container.sh
```

The generated launcher starts the SLIME container with this repository mounted,
mounts `/var/run/docker.sock` for the Docker reward backend, and bootstraps
SLIME with `/root/Megatron-LM` on `PYTHONPATH`.

The generated Docker launcher keeps `--ulimit stack=67108864` enabled by
default. It leaves `--ulimit memlock=-1` off because some managed GPU hosts
reject that rlimit before the container starts. On hosts that allow locked
memory, opt in with:

```bash
SLIME_DOCKER_MEMLOCK_ULIMIT=1 .w8-biayn/slime/run-container.sh
```

It also raises the in-container open-file soft limit to
`SLIME_NOFILE_SOFT_LIMIT=65536` before bootstrapping SLIME.

## Moonlight C++ Performance

Run inside the SLIME container:

```bash
cd /workspace/<repo-name>

export SLIME_RUN_ID="moonlight_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_HF_CHECKPOINT=/root/models/Moonlight-16B-A3B-Instruct
export SLIME_REF_LOAD_DIR=/root/models/Moonlight-16B-A3B-Instruct_torch_dist

bash examples/slime/moonlight_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_cpp_perf/eval_base.sh
bash examples/slime/moonlight_cpp_perf/sft.sh
bash examples/slime/moonlight_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_cpp_perf/grpo.sh
bash examples/slime/moonlight_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_cpp_perf/compare.sh
```

The lane writes local state under
`.w8-biayn/slime/moonlight-cpp-perf/runs/${SLIME_RUN_ID}/`.

For the lighter Moonlight MoE smoke, use:

```bash
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

## Moonlight Rank-16 LoRA C++ Performance

The rank-16 LoRA variant wraps the active Moonlight C++ lane and applies LoRA
arguments only to SFT, GRPO, and their eval stages. It checks the active
SLIME/Megatron `--help` surface before training so unsupported LoRA flags are
not silently ignored.

Run inside the SLIME container:

```bash
export SLIME_RUN_ID="moonlight_lora16_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_CPP_TASKS_DIR=/workspace/browser-is-all-you-need/.w8-biayn/data/tasks-full
export SLIME_HF_CHECKPOINT=/root/models/Moonlight-16B-A3B-Instruct
export SLIME_REF_LOAD_DIR=/root/models/Moonlight-16B-A3B-Instruct_torch_dist
export SLIME_LORA_RANK=16
export SLIME_WANDB_PROJECT=slime-moonlight-lora-cpp-perf
export SLIME_WANDB_GROUP="${SLIME_RUN_ID}"

bash examples/slime/moonlight_lora_cpp_perf/prepare_data.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_base.sh
bash examples/slime/moonlight_lora_cpp_perf/sft.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_sft.sh
bash examples/slime/moonlight_lora_cpp_perf/grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/eval_grpo.sh
bash examples/slime/moonlight_lora_cpp_perf/compare.sh
```

The LoRA lane writes under
`.w8-biayn/slime/moonlight-lora-cpp-perf/runs/${SLIME_RUN_ID}/` and uses W&B
when `WANDB_API_KEY` or an existing W&B login is present.

### SLIME PIE C++ Path

For the SLIME version of the PIE C++ experiment, keep the normal PIE task build
and convert those tasks to SLIME JSONL:

```bash
uv run w8-biayn data slime build \
  --tasks-dir .w8-biayn/data/tasks-full \
  --out .w8-biayn/data/slime-pie \
  --profile full-official \
  --run-id "$RUN_ID" \
  --min-train-tasks 1000 \
  --min-validation-tasks 100
```

Then launch `examples/slime/cpp_perf/run_moonlight_cpp_perf_rl.sh` with the AIME
4-GPU resource profile and the C++ reward hook:

```bash
SLIME_PROMPT_DATA=.w8-biayn/data/slime-pie/train.jsonl \
W8_BIAYN_SLIME_TASK_ROOT=.w8-biayn/data/slime-pie \
SLIME_CUSTOM_GENERATE_FUNCTION_PATH= \
SLIME_CUSTOM_RM_PATH=generate_with_cpp_perf.reward_func \
SLIME_REWARD_KEY=score \
SLIME_NUM_GPUS=4 \
SLIME_ROLLOUT_BATCH_SIZE=4 \
SLIME_N_SAMPLES_PER_PROMPT=1 \
SLIME_GLOBAL_BATCH_SIZE=4 \
SLIME_MAX_RESPONSE_LEN=256 \
SLIME_MAX_TOKENS_PER_GPU=4096 \
bash examples/slime/cpp_perf/run_moonlight_cpp_perf_rl.sh
```

`generate_with_cpp_perf.reward_func` lives under `examples/slime/cpp_perf/` and
calls the repo C++ reward harness, so the SLIME path still compiles, tests, and
benchmarks candidates against the PIE `v1` oracle. `SLIME_CUSTOM_GENERATE_FUNCTION_PATH=`
disables the Python-tool ReTool trajectory for C++; SLIME's stock one-turn
generation is used instead.

## GLM C++ Performance

When the GLM lane is present, run inside the SLIME container:

```bash
cd /workspace/<repo-name>

export SLIME_RUN_ID="glm47_cpp_perf_$(date -u +%Y%m%d%H%M%S)"
export SLIME_HF_CHECKPOINT=/root/models/GLM-4.7-Flash
export SLIME_REF_LOAD_DIR=/root/models/GLM-4.7-Flash_torch_dist

bash examples/slime/glm47_cpp_perf/prepare_data.sh
bash examples/slime/glm47_cpp_perf/eval_base.sh
bash examples/slime/glm47_cpp_perf/sft.sh
bash examples/slime/glm47_cpp_perf/eval_sft.sh
bash examples/slime/glm47_cpp_perf/grpo.sh
bash examples/slime/glm47_cpp_perf/eval_grpo.sh
bash examples/slime/glm47_cpp_perf/compare.sh
```

Keep GLM defaults conservative until SGLang startup, Megatron training, and C++
reward throughput are stable on the target GPU host. For the paid one-command
GCP full run, use the CLI (dry-run first, always):

```bash
uv run --extra cloud w8-biayn launch glm47-full --dry-run
uv run --extra cloud w8-biayn launch glm47-full \
  --accelerators H100:8 --use-spot --max-attempts 12
```

It loops for capacity in `asia-southeast1` by default (repeat `--region` for
the other allowed regions; pass `--use-spot` when the project only holds
preemptible GPU quota; `--accelerators A100-80GB:8` is the drop-in
80GB-class alternative), runs base eval, SFT, SFT eval, GRPO, GRPO eval, and
compare, copies W&B/local artifacts under `.w8-biayn/slime/glm47-cpp-perf/`,
and tears the provisioned node down. The W&B key resolves from
`--wandb-api-key-file`, `WANDB_API_KEY`, or a `WANDB_KEY` entry in `.env`
(copy `.env.sample` — it documents the only key the tooling reads from
`.env`; GCP credentials live in `.gcp-service-account.json`, not `.env`).
Cloud hardware is managed by SkyPilot behind the CLI (pinned via the `cloud`
extra; see `w8_biayn.constants.SKYPILOT_PIN`), the training itself is pure
SLIME inside the lane container, and the launch tracks the submitted job to a
terminal state before declaring success or tearing down. Implementation:
`src/w8_biayn/cloud_launch.py`
(`examples/slime/glm47_cpp_perf/launch_gcp_h100_full.py` is a thin
compatibility shim).

Key launch flags:

- `--accelerators` — `H100:8` (default) or `A100-80GB:8`. On non-Hopper GPUs
  the lane must use the `alltoall` MoE dispatcher (now the default); DeepEP's
  `flex` dispatcher is Hopper-only and opt-in via
  `SLIME_GLM_MOE_TOKEN_DISPATCHER_TYPE=flex`.
- `--use-spot` — request preemptible capacity (on-demand H100 quota is often 0;
  spot/preemptible quota is self-service).
- `--disk-size` — boot disk GB (default 1024). The 30B model is staged ~4x
  (HF download, torch_dist conversion, SFT checkpoint, HF export); the 256GB
  default overflows mid-export.
- `--min-train-tasks` / `--min-validation-tasks` / `--min-test-tasks` — PIE
  admission gates (defaults 1000/100/100). Lower them for a bounded pilot.
- `--train-limit` / `--eval-limit` — task caps (default: all admitted).
- `--resume-from-run <run-id>` — restore that run's persisted checkpoints and
  skip the training stages it already finished (see below).

**Checkpoint persistence and resume.** On exit — including a partial run where,
say, SFT finished but GRPO failed — the launch rsyncs the run's Megatron
checkpoints and HF exports to `gs://<project>-w8-biayn/runs/glm47/<run-id>/`.
Because the node is ephemeral and gets torn down, this is what makes a run
recoverable. To continue a run, relaunch with a new run id and
`--resume-from-run <old-run-id>`: it restores the old checkpoints into the new
run and the lane skips any training stage whose Megatron checkpoint *and* HF
export are already present, re-running only what's left (evals always re-run).
Persist/restore run on the node's ambient credentials and are best-effort.

**Dataset cache.** The launch restores `tasks-full` from a project-scoped,
gate-keyed GCS path (`gs://<project>-w8-biayn/cache/<version>/tasks-full/...`)
before building, only builds on a cache miss, and repopulates the cache after.
The path is a pure function of the cache version and admission gates, so every
user in the project shares the same cache and different inputs never collide.
The first run for a given key pays the ~20-minute PIE build; later runs and
other users restore in seconds. Cache ops use the node's ambient credentials
and are best-effort — a node without bucket write still trains.

**Teardown.** Three layers, so a dead launcher process can never orphan a paid
box:

1. *Launcher teardown (fast path).* When the launch process is alive it
   downloads artifacts and downs the cluster on exit.
2. *Cluster-side autostop (automatic backstop).* The launch arms SkyPilot
   `idle_minutes_to_autostop` (default 20, `--idle-autostop-minutes`) with
   `down=True`, so once the job ends and the node goes idle SkyPilot
   *terminates* the cluster on its own — independent of the launcher process.
   This closes the gap that once left a cluster UP for ~6 hours when a
   background launcher was killed mid-teardown and its `finally` block never
   ran. Set `0` to disable and rely on launcher-only teardown.
3. *Manual reaper (last resort).* Every instance is tagged
   `labels.run_id=<run-id>` (the same id as the W&B group), so
   `uv run --extra cloud w8-biayn ops down-run <run-id> --execute` downs the
   cluster and deletes any instance still carrying the tag.

After a run, verify with
`gcloud compute instances list --filter=labels.project=w8-biayn`.

**Network honesty.** Every launch probes its dependencies
(compute/storage.googleapis.com, GitHub, W&B) BEFORE spending — unreachable
GCP endpoints fail the launch fast — and a watchdog thread reports
`net_degraded` / `net_still_degraded` / `net_recovered` transitions into the
console log and the `pipeline/launch_events` table for the whole run (a local
DNS blip once stalled provisioning in silent client retries). A vanished spot
cluster is terminal (`CLUSTER_LOST`) and re-enters the provisioning retry
loop instead of being polled forever. Manual probe:
`uv run w8-biayn ops net-check`.

**Observability (W&B).** One launch = one W&B group (= the run id) containing
per-stage runs with deterministic ids and distinct names (`<run-id>-<stage>`;
the lanes pin `WANDB_RUN_ID` and the train-entry shim renames the live run,
because pinned slime ignores `--wandb-run-id` and would name every run after
the group). Each kind of data has one home, following the drill path
*curve → distribution → sample → artifact*:

| Data | Where |
|---|---|
| Training dynamics (kl, clipfrac, grad_norm, logprob drift) | SLIME `train/*`, `rollout/*` (native) |
| Live rollout health (reward mean/std, **zero-variance group fraction** — the GRPO signal heartbeat, abort/format/compile/test/timeout rates, agent steps, wall time) | `rollout_health/*` in the same stage run, logged by the generate hook via SLIME's shared mode |
| Distributions (reward, speedup, agent steps) | `wandb.Histogram` panels |
| Eval outcomes per stage | `eval/*` on the stage's own run (same keys across base/sft/grpo → one overlay panel) + per-task `wandb.Table` + `eval/abort/<reason>` counts, from the offline scorer |
| Uplift verdict | comparison `wandb.Table` + `uplift/*` summary on `<run-id>-pipeline` |
| Token capture (trained tokens/episode, zero-trained-token rate, captured response length) | `rollout_health/trained_tokens_*` — drained directly from the samples in the generate hook |
| Dataset composition (per-task table, counts, gates, GCS link) | `dataset/tasks` table + `dataset_*` config on the pipeline run (`publish-dataset` after prepare-data) |
| Cloud lifecycle (provision attempts, job ids, terminal states, teardown) | `pipeline/launch_events` table + provenance config (git SHA, pins, checkpoint GCS link) |
| GPU memory per stage | `vram/<stage>_usage` table + peak summary (`publish-vram` from each stage's nvidia-smi trace) |
| Setup timeline | `pipeline/elapsed_seconds` curve + one `pipeline/timeline` table (never raw unix scalars) |
| Launch knobs / outcome / lineage | `wandb.config` (redacted `LaunchOptions`), `pipeline/outcome` summary, GCS checkpoint reference artifact |
| Catastrophes | `wandb.alert`: all-abort evals, >30% abort rate with top reason, failed/interrupted launches |

`uv run --extra cloud w8-biayn wandb workspace` pushes the curated saved view
(sections: Uplift & Eval Comparison, Rollout Health, Training Dynamics,
Pipeline) so the project does not render as an unordered metric dump. The
agentic lane also enables SLIME's `--log-passrate`/`--log-multi-turn`
(`passrate/*`, `multi_turn_metric/*`). GRPO group size is
`--grpo-n-samples-per-prompt` (default 8; it must be >= 2 — one sample per
prompt collapses every group-relative advantage to zero, the kl-NaN failure)
and the global batch derives as `rollout_batch_size * n_samples_per_prompt`
unless overridden.

## SSH Manual Run

Use this when you SSH into a GPU machine you already own (no GCP, no SkyPilot,
no Google keys) and want to run the GLM lanes by hand. Everything the paid
launcher automates is reproducible from the repo checkout; the GCS dataset
cache, checkpoint persist, autostop, and reaper simply do not apply — your
box, your disk, your checkpoints. Assumptions: the repo source is on the node,
Docker + the NVIDIA container toolkit work (`docker run --gpus all` succeeds),
and `.env` at the repo root carries `WANDB_KEY` (see `.env.sample`; W&B is
optional — without a key the lanes skip it). 8× 80GB GPUs match the default
GLM parallelism (TP2 · PP2 · CP2 · EP4). Before a long run,
`uv run w8-biayn ops net-check` (keyless) probes connectivity — on a manual
box GitHub (SWE-agent clone, model download) and `api.wandb.ai` are the ones
that matter; the googleapis probes are only relevant with GCP.

1. Host setup (installs `uv`, checks tools, pulls the SLIME image, builds the
   C++ grader image on the HOST daemon — the reward sandbox runs through the
   mounted docker socket):

   ```bash
   ./scripts/bootstrap.sh --no-sky
   uv run w8-biayn data doctor
   uv run w8-biayn upstreams clone slime
   uv run w8-biayn slime doctor
   uv run w8-biayn slime setup          # writes .w8-biayn/slime/run-container.sh
   uv run w8-biayn cpp harness preflight --cpu 3
   ```

2. Build the dataset locally on the node (downloads PIE, prepares, measures
   coverage, admits tasks into `.w8-biayn/data/tasks-full`):

   ```bash
   uv run w8-biayn data pie download --out .w8-biayn/data/pie
   uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
   uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full \
     --out .w8-biayn/data/pie-full/coverage.json --report-out .w8-biayn/data/pie-full/coverage-report.json
   uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full \
     --coverage-json .w8-biayn/data/pie-full/coverage.json --out .w8-biayn/data/tasks-full \
     --min-train 1000 --min-validation 100 --min-test 100 --force
   ```

3. Enter the training container. It mounts the repo at
   `/workspace/<repo-name>`, `$HOME/models` at `/root/models` (override with
   `HOST_MODELS_DIR`), and the host docker socket (for the grader):

   ```bash
   bash .w8-biayn/slime/run-container.sh
   ```

4. Inside the container, set the run knobs. The env is NOT inherited from the
   host shell, so export what you need here (the GLM checkpoint auto-downloads
   to `/root/models/GLM-4.7-Flash` and torch_dist-converts on first use):

   ```bash
   cd /workspace/<repo-name>
   # W&B (optional): the only key the tooling reads from .env
   export WANDB_API_KEY="$(grep -m1 '^WANDB_KEY=' .env | cut -d= -f2-)"
   export SLIME_WANDB_PROJECT=slime-glm47-cpp-perf   # enables W&B logging

   export SLIME_RUN_ID="manual-$(date -u +%Y%m%d%H%M%S)"   # = W&B group
   export SLIME_CPP_TRAIN_LIMIT=8 SLIME_CPP_EVAL_LIMIT=8   # bound the smoke; unset for full
   export SLIME_GRPO_NUM_ROLLOUT=1                          # bump for real training
   # Group size must stay >= 2 (defaults to 8); global batch derives from it.
   # export SLIME_GRPO_N_SAMPLES_PER_PROMPT=8
   # export SLIME_NUM_GPUS=8   # with fewer GPUs also override the parallelism
   #                           # envs (SLIME_EXPERT_MODEL_PARALLEL_SIZE etc.)
   # Adapter fork/merge threshold (defaults 4096; keep > the response budget).
   # Watch rollout_health/fork_samples_dropped_mean in W&B: persistently high
   # means trained tokens are being discarded to keep GRPO's group shape.
   # export SLIME_SWE_FORK_MERGE_MAX_RESPONSE_TOKENS=4096
   ```

5. Run the agentic lane stage by stage (single-turn lane: swap the directory
   for `glm47_cpp_perf`):

   ```bash
   bash examples/slime/glm47_swe_agent_cpp_perf/prepare_data.sh
   bash examples/slime/glm47_swe_agent_cpp_perf/eval_base.sh
   bash examples/slime/glm47_swe_agent_cpp_perf/sft.sh
   bash examples/slime/glm47_swe_agent_cpp_perf/eval_sft.sh
   bash examples/slime/glm47_swe_agent_cpp_perf/grpo.sh
   bash examples/slime/glm47_swe_agent_cpp_perf/eval_grpo.sh
   bash examples/slime/glm47_swe_agent_cpp_perf/compare.sh
   ```

6. Results land under
   `.w8-biayn/slime/glm47-cpp-perf/runs/<run-id>/` — per-stage
   `run_receipt.txt`, `run.log`, VRAM traces, eval records/summaries, and
   `eval/comparison.json` (the uplift verdict). With a W&B key the same run id
   is the W&B group: per-stage runs, `rollout_health/*` live panels, eval
   tables, and the pipeline timeline.

Notes for manual boxes: checkpoints stay on the node (nothing persists to GCS
— copy `runs/<run-id>/checkpoints` + `hf/` yourself if you need durability);
there is no autostop or reaper, so nothing tears the machine down; the lane
re-clones SWE-agent into `.cache/sweagent` at a pinned commit on first agentic
stage; keep the swerex LocalDeployment invariants in mind if you hack on the
driver (see the lane status section above).

## Moonlight MoE Smoke

For the lightest MoE smoke, start with the repo-owned Moonlight wrapper under `examples/slime/moonlight_moe_smoke/`. It uses a Moonlight-16B-A3B Instruct checkpoint, a four-row local math JSONL, one rollout, one sample per prompt, short responses, and the real colocated Megatron + SGLang training path. It does not require E2B, browser sandboxes, DAPO-Math downloads, or W&B by default.

Prerequisites are intentionally narrow: a 4x A100 80 GB node, the pinned SLIME sidecar, `/root/Megatron-LM`, a local Moonlight HF checkpoint, and its converted Megatron torch_dist checkpoint. The launcher defaults are `/root/Moonlight-16B-A3B-Instruct` and `/root/Moonlight-16B-A3B-Instruct_torch_dist`; override with `SLIME_HF_CHECKPOINT` and `SLIME_REF_LOAD_DIR`. The current Moonlight smoke also depends on the generated `.w8-biayn/slime/run-container.sh` including `-v "${HOST_MODELS_DIR:-$HOME/models}":/root/models \`; add that mount before starting the GPU container so the model files are visible inside the SLIME runtime.

Start the SLIME container:

```bash
.w8-biayn/slime/run-container.sh
```

Then launch the smoke inside the container:

```bash
cd /workspace/<repo-name>

SLIME_NUM_GPUS=4 \
SLIME_NUM_ROLLOUT=1 \
SLIME_ROLLOUT_BATCH_SIZE=4 \
SLIME_N_SAMPLES_PER_PROMPT=1 \
SLIME_MAX_RESPONSE_LEN=128 \
SLIME_MAX_TOKENS_PER_GPU=1024 \
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

If the torch_dist checkpoint is not already present, the same script can do the conversion explicitly:

```bash
SLIME_CONVERT_IF_MISSING=1 \
SLIME_CONVERT_NPROC=4 \
bash examples/slime/moonlight_moe_smoke/run_moonlight_16b_a3b_int4_smoke.sh
```

Keep `SLIME_ENABLE_DEEPEP=0` for the first pass. Set `SLIME_ENABLE_DEEPEP=1` only after the default all-to-all smoke is healthy on that host/container stack.

The launcher samples `nvidia-smi` during the run and writes `vram_usage.csv` plus `vram_peak.txt` under `.w8-biayn/slime/moonlight-16b-a3b-int4-smoke/runs/<timestamp>/`; use `vram_peak.txt` as the peak-VRAM receipt.

## Moonlight ReTool

Runbook and launchers:

- `examples/slime/retool/README.md`
- `examples/slime/retool/retool_moonlight_sft.sh`
- `examples/slime/retool/retool_moonlight_rl.sh`

This lane runs a local Python tool sandbox only: no E2B dependency, no external
browser sandbox, and no hosted tool service.

## SLIME Multi-Agent Text Example

Runbook and launcher:

- `examples/slime/multi_agent/README.md`
- `examples/slime/multi_agent/run_multi_agent_text.sh`

This is the generic text-only SLIME smoke. It prepares the DAPO-Math-17k JSONL
and runs upstream SLIME's multi-agent generate function with a small Qwen3-4B
default model.

## Evaluation

For SLIME C++ lanes, aggregate debug rollout dumps and compare summaries through
the project SLIME bridge:

```bash
python -m w8_biayn.integrations.slime_cpp_perf aggregate-debug \
  --label base \
  --debug-rollout <path-to-debug-rollout.pt-or-jsonl> \
  --out .w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/base

python -m w8_biayn.integrations.slime_cpp_perf compare \
  --summary base=.w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/base.summary.json \
  --summary sft=.w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/sft.summary.json \
  --summary grpo=.w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/grpo.summary.json \
  --out .w8-biayn/slime/<lane>/runs/${SLIME_RUN_ID}/eval/comparison.json
```

Formal uplift requires GRPO to beat base and SFT on
`correct_and_faster_rate` and mean best reward, with no missing runtime rows.

## Repository Map

```text
scripts/bootstrap.sh                         fresh-machine bootstrap
scripts/prepare_dapo_math_dataset.py         optional SLIME text-smoke data prep
scripts/wandb_milestone.py                   standalone pipeline-milestone logger (elapsed curve + timeline table)
examples/slime/moonlight_cpp_perf/           active Moonlight C++ lane
examples/slime/moonlight_lora_cpp_perf/      rank-16 LoRA Moonlight C++ lane
examples/slime/glm47_cpp_perf/               active GLM C++ lane when present
examples/slime/glm47_swe_agent_cpp_perf/     agentic SWE-agent file-state C++ lane
examples/slime/cpp_perf/                     single-launcher Moonlight C++ RL profile
examples/slime/retool/                       Moonlight ReTool lane
examples/slime/moonlight_moe_smoke/          light Moonlight MoE smoke
examples/slime/multi_agent/                  generic text-only SLIME smoke
src/local.py                                 Moonlight Megatron local-layer shim
src/w8_biayn/cli.py                          CLI surface
src/w8_biayn/cloud_launch.py                 SkyPilot-backed paid GLM launch (w8-biayn launch glm47-full)
src/w8_biayn/cpp_perf/data.py                downloads, full PIE prep, manifests, cache
src/w8_biayn/cpp_perf/coverage.py            gcov coverage measurement
src/w8_biayn/cpp_perf/pie.py                 PIE parsing and task construction
src/w8_biayn/cpp_perf/schema.py              task and harness schema
src/w8_biayn/cpp_perf/prompts.py             prompt builder and task loading
src/w8_biayn/cpp_perf/slime_dataset.py       SLIME prompt/metadata JSONL builder
src/w8_biayn/cpp_perf/eval.py                eval aggregation
src/w8_biayn/cpp_perf/judge.py               contest-style stdout comparison
src/w8_biayn/cpp_perf/sandbox.py             Docker compile/test/runtime harness
src/w8_biayn/cpp_perf/reward.py              correctness-gated efficiency reward
src/w8_biayn/integrations/slime_cpp_perf.py  SLIME C++ data/reward/eval bridge
src/w8_biayn/integrations/slime_swe_agent_cpp_perf.py
                                             agentic SWE-agent generate() hook (file-state reward + rollout health)
src/w8_biayn/integrations/swe_agent_driver.py
                                             single-instance SWE-agent run + candidate.cpp extraction
src/w8_biayn/integrations/slime_train_entry.py
                                             repo-owned SLIME train entry wrapper
src/w8_biayn/integrations/slime_moonlight_hf_export.py
                                             Moonlight Megatron-to-HF export shim
src/w8_biayn/slime_integration/doctor.py     pinned SLIME clone doctor
src/w8_biayn/slime_integration/setup.py      SLIME container launcher/bootstrap writer
src/w8_biayn/slime_integration/sandbox.py    SLIME agent sandbox backends
src/w8_biayn/slime_integration/lora.py       runtime-native LoRA flag resolution
src/w8_biayn/wandb_report.py                 W&B data->surface contract (eval/health metrics, tables, artifacts, alerts, workspace template)
src/w8_biayn/net_health.py                   reachability probes, launch preflight, net watchdog (ops net-check)
src/w8_biayn/reporting.py                    raw Markdown/CSV/SVG run evidence reports
src/w8_biayn/shell.py                        dry-run-aware subprocess wrapper
src/w8_biayn/gcp_auth.py                     scoped GCP auth
src/w8_biayn/secrets.py                      credential metadata only
src/w8_biayn/constants.py                    upstream pins and defaults
src/w8_biayn/upstreams.py                    upstream clone management
src/w8_biayn/benchmarks.py                   benchmark ladder
.agents/REPO_GUIDE.md                        shared AGENTS.md and CLAUDE.md target
.agents/skills/w8-biayn-framework/SKILL.md   AI coding-agent workflow skill
```

## Validation

Before handing off normal code/docs work:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For SLIME setup changes, also run:

```bash
uv run w8-biayn slime doctor
uv run w8-biayn slime setup --force
uv run w8-biayn cpp harness preflight --dry-run
```

For documentation-only changes, run the skill validator and the docs guardrail
tests when practical.
