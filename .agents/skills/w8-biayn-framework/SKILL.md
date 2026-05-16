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
- BrowserGym dataset shape: `src/w8_biayn/datasets.py`
- SkyRL-Gym adapter: `src/w8_biayn/integrations/browsergym_env.py`
- SkyRL registration entrypoint: `src/w8_biayn/integrations/skyrl_browsergym_main.py`
- Tests: `tests/`

## Development Rules

Do not assume globally installed tools. If a command needs a tool, bootstrap must install it or `doctor` must diagnose it clearly.

Do not commit `.env`, `.gcp-service-account.json`, `.cache/`, `.w8-biayn/`, `.venv/`, `dist/`, generated datasets, or upstream clones.

Do not vendor rLLM or SkyRL. Use:

```bash
uv run w8-biayn upstreams clone
```

Keep cloud commands dry-run capable. Real launches must go through `w8-biayn launch`, not raw `sky launch`, unless debugging a failing rendered command.

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
uv run w8-biayn doctor --cloud
uv run w8-biayn launch miniwob --dry-run
uv run w8-biayn launch miniwob
```

If `doctor --cloud` reports GCP disabled, stop. The blocker is IAM or cloud setup, not the training code.

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
```

