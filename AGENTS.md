# Repository Development Guide

## Fresh-Machine Bootstrap Is Required

This repository is shipped to other people. Development must assume a fresh third-party computer, not this local workstation.

Every feature that requires a tool, cloud CLI, Python extra, environment variable, generated config, upstream clone, or local setup step must be represented in one of these places:

- `pyproject.toml` / `uv.lock` for Python package dependencies.
- `scripts/bootstrap.sh` for machine bootstrap and external CLI setup.
- `w8-biayn doctor` for validation and actionable diagnostics.
- `README.md` for user-facing setup, command usage, architecture, and workflow diagrams.
- `.agents/skills/` for AI-facing repo workflow guidance.
- `.gitignore` for local-only generated files, secrets, caches, and upstream clones.

Do not rely on globally installed tools unless bootstrap installs them or `doctor` reports a clear missing prerequisite with the exact next action.

## Required User Path

A new user must be able to start from a clean clone and run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud
uv run w8-biayn launch miniwob --dry-run
```

The real GCP smoke path is:

```bash
uv run w8-biayn launch miniwob
```

If any development change invalidates those commands, update bootstrap, CLI help, tests, and this file in the same change.

## Documentation And Diagrams Must Stay Current

`README.md` is the canonical user-facing entrypoint. Keep it current whenever commands, setup steps, cloud behavior, supported pipelines, IAM requirements, or operational workflows change.

The README must include Mermaid architecture and workflow diagrams. When implementation changes affect data flow, launch flow, cloud boundaries, upstream responsibilities, or teardown behavior, update the diagrams in the same change.

AI-facing skills must also stay current. When repo-specific workflows change, update `.agents/skills/w8-biayn-framework/` and preserve the symlinked skill discovery paths. If the change is generic to skill authoring, update `.agents/skills/agent-skills-framework/` as well.

## Secrets And Local State

`.env` and `.gcp-service-account.json` are local-only and must never be committed.

Upstream source clones belong in `.cache/upstreams/` through:

```bash
uv run w8-biayn upstreams clone
```

Do not vendor rLLM or SkyRL into this repository. Keep the pinned SHAs in code/config and clone into ignored cache paths.

## Cloud Development Rules

All GCP/SkyPilot workflows must support dry-run rendering before launching resources.

Any command that provisions cloud resources must:

- Use `.gcp-service-account.json` through `GOOGLE_APPLICATION_CREDENTIALS`.
- Avoid printing credential contents.
- Render the SkyPilot YAML into an ignored path.
- Provide status, logs, and teardown commands through `w8-biayn`.

For WebArena, do not assume the official archives are bundled. Require an explicit GCS prefix with `--webarena-archives-gcs` or external `WA_*` URLs.

## Tests And Validation

Before handing off changes, run:

```bash
uv run --extra dev pytest
uv run python -m compileall src tests
```

For bootstrap-affecting changes, also run:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn config render miniwob --credentials .gcp-service-account.json
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

Use `--no-sky` only for local validation when SkyPilot install is intentionally skipped; the default bootstrap path must install SkyPilot.
