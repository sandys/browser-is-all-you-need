---
name: w8-biayn-framework
description: Maintain, extend, test, and operate the w8-biayn BrowserGym RL pipeline for rLLM, SkyRL, SkyPilot, and Google Cloud. Use when working in this repository on bootstrap, CLI commands, SkyPilot YAML rendering, BrowserGym/SkyRL integration, GCP smoke runs, README diagrams, AGENTS.md rules, or repo-specific AI skill guidance.
---

# w8-biayn Framework

Use this skill when changing this repository. The project is shipped to third parties, so every change must preserve a fresh-machine path.

## Required Context

Read [../../../AGENTS.md](../../../AGENTS.md) before editing. It is the repository-level development contract.

Read [../../../README.md](../../../README.md) when changing setup, CLI usage, architecture, workflow, cloud behavior, or supported pipelines. Keep its Mermaid diagrams current.

## Core Model

`w8-biayn` is a Typer CLI that owns command/control, config rendering, dataset prep, and BrowserGym integration. It delegates training to SkyRL and studies rLLM/SkyRL from pinned ignored clones.

Main responsibilities:

- Bootstrap: `scripts/bootstrap.sh`
- CLI: `src/w8_biayn/cli.py`
- SkyPilot rendering: `src/w8_biayn/sky_config.py`
- DOMDiff local and GCP lifecycle: `src/w8_biayn/domdiff.py`
- DOMDiff reward service: `src/w8_biayn/rewards/chromiumrl_service.py`
- Benchmark scorecard: `src/w8_biayn/benchmarks.py`
- OSWorld upstream smoke and benchmark path: `src/w8_biayn/osworld.py`
- ScaleCUA-to-OSWorld SFT conversion: `scripts/convert_scalecua_to_osworld_toolcalls.py`
- ScaleCUA Qwen2.5-VL LoRA SFT: `src/w8_biayn/scalecua_sft.py`
- ScaleCUA real-image subset prep: `scripts/prepare_scalecua_real_subset.py`
- ScaleCUA held-out JSONL splitting: `scripts/split_jsonl_train_eval.py`
- Harbor task discovery and SkyRL dataset prep: `src/w8_biayn/harbor/tasks.py`, `src/w8_biayn/harbor/skyrl_dataset.py`
- Harbor Docker task runner and SkyRL-Gym env: `src/w8_biayn/harbor/docker_runner.py`, `src/w8_biayn/integrations/harbor_env.py`
- BrowserGym dataset shape: `src/w8_biayn/datasets.py`
- SkyRL-Gym adapter: `src/w8_biayn/integrations/browsergym_env.py`
- SkyRL registration entrypoint: `src/w8_biayn/integrations/skyrl_browsergym_main.py`
- Harbor SkyRL registration entrypoint: `src/w8_biayn/integrations/skyrl_harbor_main.py`
- Tests: `tests/`

## Development Rules

Do not assume globally installed tools. If a command needs a tool, bootstrap must install it or `doctor` must diagnose it clearly.

Commit after each logical change. Prefer several focused commits over one large commit, and make each commit include the code, docs, tests, and skill updates required for that specific change before moving to the next one.

Do not commit `.env`, `.gcp-service-account.json`, `.cache/`, `.w8-biayn/`, `.venv/`, `dist/`, generated datasets, or upstream clones.

Do not vendor rLLM or SkyRL. Use:

```bash
uv run w8-biayn upstreams clone
```

Keep cloud commands dry-run capable. Real launches must go through `w8-biayn launch`, not raw `sky launch`, unless debugging a failing rendered command.

Cloud commands must use `.gcp-service-account.json` directly through scoped environment variables (`GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, `CLOUDSDK_CORE_PROJECT`). Do not run `gcloud auth activate-service-account`, do not call `gcloud config set project`, and do not depend on a preconfigured global gcloud account. Python code that needs an OAuth token should mint it from the service-account JSON, not from `gcloud auth print-access-token`.

Before paid SkyPilot launches, `doctor --cloud` must verify launch-time IAM too. Use the explicit project-level SkyPilot GCP launch permission set from SkyPilot's GCP backend: compute, storage, service usage, firewall, reservation, IAM service-account creation/get, and project IAM policy get/set permissions. SkyPilot may call project `setIamPolicy` while preparing worker service-account bindings, so `resourcemanager.projects.setIamPolicy` must remain in the checked set.

Do not vendor AndroidWorld, WootzApp, APKs, or browser source. DOMDiff uses a prebuilt image, defaulting to:

```text
ghcr.io/wootzapp/android-world-domdiff:daytona-92000b7
```

For development and smoke runs, prefer local DOMDiff mode: run `android-world-domdiff:local` on the workstation, run the ChromiumRL reward service locally, and expose reward HTTP through a Cloudflare quick tunnel. Do not expose CDP unless explicitly debugging with `--publish-cdp` or `--local-publish-cdp`. Do not push local DOMDiff images unless the user explicitly asks for a GCP-hosted reward VM. If the DOMDiff image is local-only and must run on GCP, push it to Google Artifact Registry with `w8-biayn domdiff push-image` or use `--local-reward-image`. Do not copy image tarballs or source trees into this repo. The GCP reward VM must authenticate to Artifact Registry only for `*.pkg.dev` images and must not print service-account contents.

Keep Harbor DOMDiff submit/poll calls retryable with the trycloudflare curl fallback. A transient quick-tunnel submit timeout should be recovered before failing a paid R3 smoke.

The only code copied into that container should be the small `w8_biayn.rewards` adapter.

In local DOMDiff mode, the ChromiumRL reward service runs outside the container with `CDP_URL=ws://localhost:9224`; keep state/logs under `.w8-biayn/domdiff-local/`.

Never render local/private DOMDiff URLs into SkyPilot configs. `localhost`, `127.0.0.1`, `host.docker.internal`, `.local` names, private RFC1918 addresses, link-local addresses, and unspecified addresses are invalid for remote GCP/SkyPilot trainers. Use Cloudflare tunnel URLs.

Keep the benchmark ladder runnable and documented. If R3, DOMDiff, Harbor, BrowserGym, WebArena, OSWorld, or AndroidWorld behavior changes, update `src/w8_biayn/benchmarks.py`, README benchmark guidance, and rendered benchmark metadata.

OSWorld benchmarking is supported through `w8-biayn osworld` for upstream clone/setup, validation, smoke runs, local run summaries, and domain benchmarks. Clone the pinned upstream with `uv run w8-biayn upstreams clone osworld`, build the upstream env with `uv run w8-biayn osworld setup`, validate with `uv run w8-biayn osworld validate`, use `uv run w8-biayn osworld smoke --dry-run` before real runs, and use `uv run w8-biayn osworld benchmark --dry-run` before domain sweeps. If a user wants a subset, prefer a custom grouped JSON file passed via `--taskset`.

ScaleCUA OSWorld SFT data must be converted into the upstream Qwen2.5-VL `computer_use` tool-call format before training for OSWorld. Do not train the first pass on raw ScaleCUA `<action>click(x=..., y=...)</action>` targets. Use `scripts/convert_scalecua_to_osworld_toolcalls.py` to convert safe desktop/web actions into `<tool_call>{"name": "computer_use", ...}</tool_call>` assistant targets and write unsupported actions to a reject JSONL. Keep generated ScaleCUA raw data, prepared JSONL files, reports, and LoRA outputs under ignored `.w8-biayn/scalecua/`.

For local SFT smoke runs, use `uv run --extra sft w8-biayn scalecua sft --model Qwen/Qwen2.5-VL-7B-Instruct --train .w8-biayn/scalecua/prepared/smoke-toolcall.jsonl --output .w8-biayn/scalecua/lora/qwen25vl7b-smoke --max-steps 100 --batch-size 1 --grad-accum 8`. The trainer in `src/w8_biayn/scalecua_sft.py` uses Transformers plus PEFT, applies LoRA to Qwen projection/MLP modules, builds image+text chat inputs, masks prompt tokens with `labels=-100`, supports W&B through `--wandb-project` and `--wandb-run-name`, and saves only the adapter plus processor. Keep this first path single-node; add distributed support only after the single-GPU smoke is validated.

For real-image ScaleCUA LoRA runs, use `uv run --extra sft python scripts/prepare_scalecua_real_subset.py --limit 10000 --dry-run` first to rank archives and preview compressed GB, then rerun without `--dry-run` to download selected archive parts and extract only selected image members. The prep script deletes the temporary concatenated `.tar.gz` by default after extraction; pass `--cleanup-archive-parts` when the run should keep only extracted images and redownload parts later if needed. Train only after the generated summary report shows `missing_images: 0`. Use `scripts/split_jsonl_train_eval.py` to create deterministic train/eval JSONL splits, then pass `--eval` and `--eval-steps` to `w8-biayn scalecua sft` so W&B logs held-out `eval_loss`.

For Harbor DOMDiff R3, do not use Tinker or Thinking Machines as the backend. Use self-hosted SkyRL on the SkyPilot/GCP side. The rendered path should install Docker and Cloudflare on the GCP host, start the Google GPU container with `--shm-size=32g` and `/var/run/docker.sock` mounted, reuse an existing Harbor virtualenv when complete and recreate it when incomplete, prepare Harbor parquet data with `w8-biayn harbor prepare-data`, and run `w8_biayn.integrations.skyrl_harbor_main`. That entrypoint must register `harbor-domdiff` inside SkyRL's Ray entrypoint task before constructing `BasePPOExp`; registering only in the local driver does not reach the trainer process. The default R3 smoke model is `moonshotai/Moonlight-16B-A3B-Instruct`; its `DeepseekV3ForCausalLM` architecture is supported by Megatron Bridge, unlike `Qwen2MoeForCausalLM` models. Harbor DOMDiff R3 defaults to `H100:8`, matching SkyRL's Moonlight router replay recipe. A100 40GB overrides can reach Harbor rollout and DOMDiff reward scoring, but need CPU optimizer offload for the Megatron optimizer step; keep the CLI warning and rendered offload before paid A100 40GB Harbor R3 launches. Render the SkyRL GPU count from the requested accelerator string; do not depend on SkyPilot host-only shell variables inside the Google GPU container. R3 routing replay must run `uv sync --active --extra megatron --extra gcp` from the SkyRL checkout so upstream `tool.uv` overrides resolve Megatron dependencies, set `trainer.strategy=megatron`, set Megatron TP/PP/CP/EP values, set vLLM MoE expert parallelism, set `generator.inference_engine.distributed_executor_backend=mp`, disable flash attention for Moonlight's MLA attention, and disable KL loss for the smoke. SkyRL rejects FSDP or Ray for router replay. Do not require `TINKER_API_KEY`, Daytona, GitHub tokens, or copied browser source. Harbor task containers need collision-resistant names because SkyRL may evaluate multiple samples from the same task at the same time. Copy Harbor verifier assets file-by-file into `/tests`; Docker directory-copy behavior differs by Docker version. Harbor task previews must publish Cloudflare quick tunnels so the laptop-local ChromiumRL reward service can score them.

## Documentation Rule

Whenever implementation changes any setup step, CLI command, rendered config, cloud/IAM expectation, data flow, launch flow, teardown behavior, or supported pipeline, update all relevant docs in the same change:

1. `README.md`
2. Mermaid architecture/workflow diagrams in `README.md`
3. `AGENTS.md` if it changes development rules
4. This skill if it changes AI-agent operating instructions

## Smoke Workflow

Use this order for real GCP smoke work:

```bash
./scripts/bootstrap.sh
uv run w8-biayn doctor --cloud --domdiff
uv run w8-biayn benchmarks list
uv run w8-biayn launch miniwob --dry-run
uv run w8-biayn launch miniwob
```

If `doctor --cloud` reports GCP disabled, stop. The blocker is IAM or cloud setup, not the training code.

Use this order for real DOMDiff reward-host smoke work:

```bash
./scripts/bootstrap.sh
uv run w8-biayn doctor --cloud --domdiff
uv run w8-biayn domdiff local up --image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff local smoke --image android-world-domdiff:local
```

When the local DOMDiff image should be used by GCP training, keep the local smoke stack running and pass its printed URLs:

```bash
uv run w8-biayn domdiff local up --image android-world-domdiff:local
uv run w8-biayn launch r3 \
  --chromiumrl-url https://<local-domdiff-reward-tunnel> \
  --benchmark webvoyager-domdiff-heldout
```

For the packaged Harbor DOMDiff R3 smoke, keep local DOMDiff running and launch SkyRL on GCP:

```bash
uv run w8-biayn harbor validate
uv run w8-biayn launch r3 \
  --with-local-domdiff \
  --benchmark harbor-domdiff-browser-swe
```

Use `--no-harbor-oracle` only when testing model-generated fixes instead of deterministic infrastructure smoke patches.

When the local DOMDiff image must be hosted on a GCP reward VM:

```bash
uv run w8-biayn domdiff push-image --source-image android-world-domdiff:local
uv run w8-biayn domdiff smoke --local-reward-image android-world-domdiff:local
```

The default smoke must tear down the nested-virtualization VM. Use `--keep` only when intentionally debugging and follow with `uv run w8-biayn domdiff down --run-id <run-id>`.

## Validation

Before finishing repo changes, run:

```bash
uv run --extra dev pytest
uv run python -m compileall src tests
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

For bootstrap changes, also run:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn benchmarks list
uv run w8-biayn domdiff local up --image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff push-image --source-image android-world-domdiff:local --tag smoke --dry-run
uv run w8-biayn launch r3 --with-local-domdiff --benchmark webvoyager-domdiff-heldout --dry-run
uv run w8-biayn launch r3 --with-local-domdiff --benchmark harbor-domdiff-browser-swe --dry-run
uv run w8-biayn upstreams clone osworld --dry-run
uv run w8-biayn osworld setup --dry-run
uv run w8-biayn osworld validate
uv run w8-biayn osworld list --domain os --smoke-candidates
uv run w8-biayn osworld smoke --dry-run
uv run w8-biayn osworld run --suite tiny --dry-run
uv run w8-biayn osworld benchmark --dry-run
```
