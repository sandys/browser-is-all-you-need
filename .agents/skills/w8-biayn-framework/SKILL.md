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

Do not vendor AndroidWorld, WootzApp, APKs, or browser source. DOMDiff uses a prebuilt image, defaulting to:

```text
ghcr.io/wootzapp/android-world-domdiff:daytona-92000b7
```

For development and smoke runs, prefer local DOMDiff mode: run `android-world-domdiff:local` on the workstation, run the ChromiumRL reward service locally, and expose reward HTTP through a Cloudflare quick tunnel. Do not expose CDP unless explicitly debugging with `--publish-cdp` or `--local-publish-cdp`. Do not push local DOMDiff images unless the user explicitly asks for a GCP-hosted reward VM. If the DOMDiff image is local-only and must run on GCP, push it to Google Artifact Registry with `w8-biayn domdiff push-image` or use `--local-reward-image`. Do not copy image tarballs or source trees into this repo. The GCP reward VM must authenticate to Artifact Registry only for `*.pkg.dev` images and must not print service-account contents.

The only code copied into that container should be the small `w8_biayn.rewards` adapter.

In local DOMDiff mode, the ChromiumRL reward service runs outside the container with `CDP_URL=ws://localhost:9224`; keep state/logs under `.w8-biayn/domdiff-local/`.

Never render local/private DOMDiff URLs into SkyPilot configs. `localhost`, `127.0.0.1`, `host.docker.internal`, `.local` names, private RFC1918 addresses, link-local addresses, and unspecified addresses are invalid for remote GCP/SkyPilot trainers. Use Cloudflare tunnel URLs.

Keep the benchmark ladder runnable and documented. If R3, DOMDiff, Harbor, BrowserGym, WebArena, or AndroidWorld behavior changes, update `src/w8_biayn/benchmarks.py`, README benchmark guidance, and rendered benchmark metadata.

For Harbor DOMDiff R3, do not use Tinker or Thinking Machines as the backend. Use self-hosted SkyRL on the SkyPilot/GCP side. The rendered path should install Docker and Cloudflare on the GCP host, start the Google GPU container with `/var/run/docker.sock` mounted, prepare Harbor parquet data with `w8-biayn harbor prepare-data`, and run `w8_biayn.integrations.skyrl_harbor_main`. Do not require `TINKER_API_KEY`, Daytona, GitHub tokens, or copied browser source. Harbor task previews must publish Cloudflare quick tunnels so the laptop-local ChromiumRL reward service can score them.

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
```
