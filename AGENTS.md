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

The real DOMDiff reward-host smoke path is:

```bash
uv run w8-biayn doctor --cloud --domdiff
uv run w8-biayn domdiff local smoke --image android-world-domdiff:local
```

If any development change invalidates those commands, update bootstrap, CLI help, tests, and this file in the same change.

## Documentation And Diagrams Must Stay Current

`README.md` is the canonical user-facing entrypoint. Keep it current whenever commands, setup steps, cloud behavior, supported pipelines, IAM requirements, or operational workflows change.

The README must include Mermaid architecture and workflow diagrams. When implementation changes affect data flow, launch flow, cloud boundaries, upstream responsibilities, or teardown behavior, update the diagrams in the same change.

AI-facing skills must also stay current. When repo-specific workflows change, update `.agents/skills/w8-biayn-framework/` and preserve the symlinked skill discovery paths. If the change is generic to skill authoring, update `.agents/skills/agent-skills-framework/` as well.

## Commit Discipline

Prefer many focused commits over one large commit. Commit after each logical change, such as a development-rule update, a CLI behavior change, a docs/diagram update, a dependency/bootstrap change, or a validation/test fix.

Keep each commit internally coherent: include the code, docs, tests, and skill updates needed for that one change, then continue with the next logical change in a new commit.

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

- Use `.gcp-service-account.json` through scoped environment variables such as `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and `CLOUDSDK_CORE_PROJECT`.
- Never run `gcloud auth activate-service-account` or mutate the user's global `gcloud config`.
- Avoid printing credential contents.
- Render the SkyPilot YAML into an ignored path.
- Provide status, logs, and teardown commands through `w8-biayn`.

DOMDiff reward-host workflows must also:

- Use the prebuilt reward image by default: `ghcr.io/wootzapp/android-world-domdiff:daytona-92000b7`.
- For development and smoke runs, prefer `w8-biayn domdiff local ...` with the local `android-world-domdiff:local` image and Cloudflare quick tunnels. Do not push local DOMDiff images unless the user explicitly asks for a GCP-hosted reward VM.
- If the reward image must run on a GCP reward VM and exists only in local Docker, upload it with `w8-biayn domdiff push-image` or pass `--local-reward-image`; never copy image tarballs or source into this repo.
- Use Google Artifact Registry (`<location>-docker.pkg.dev/<project>/w8-biayn/android-world-domdiff:<tag>`) for local DOMDiff image promotion to GCP.
- Never vendor AndroidWorld, WootzApp, APKs, or browser source into this repository.
- Copy only the small `w8_biayn.rewards` adapter into the running container.
- In local DOMDiff mode, run the ChromiumRL reward service outside the container with `CDP_URL=ws://localhost:9224`; expose the reward HTTP service through a Cloudflare quick tunnel for GCP trainers. Expose CDP only with an explicit `--publish-cdp` or `--local-publish-cdp` debug request.
- Never render local/private DOMDiff URLs into SkyPilot configs. `localhost`, `127.0.0.1`, `host.docker.internal`, `.local` names, private RFC1918 addresses, link-local addresses, and unspecified addresses are invalid for remote GCP/SkyPilot trainers. Use Cloudflare tunnel URLs.
- Create GCP nested-virtualization hosts with explicit dry-run support and state under `.w8-biayn/domdiff/`.
- Keep local DOMDiff state under `.w8-biayn/domdiff-local/`.
- Authenticate the remote VM to Artifact Registry with `.gcp-service-account.json` only for `*.pkg.dev` reward images, and never print credential contents.
- Expose CDP and reward HTTP through tunnels, not broad direct GCP ingress.
- Tear down the VM, firewall, tunnels, local SSH key, and container by default. Use `--keep` only for intentional debugging.

For WebArena, do not assume the official archives are bundled. Require an explicit GCS prefix with `--webarena-archives-gcs` or external `WA_*` URLs.

For Harbor DOMDiff R3, do not use Tinker or Thinking Machines as the training backend. Render and launch self-hosted SkyRL through `w8-biayn launch r3 --benchmark harbor-domdiff-browser-swe`, normally with `--with-local-domdiff` so the local DOMDiff image stays on the workstation. The SkyPilot job must install Docker and Cloudflare on the GCP host, run the Google GPU container with the host Docker socket mounted, prepare Harbor SkyRL parquet data with `w8-biayn harbor prepare-data`, and start `w8_biayn.integrations.skyrl_harbor_main`. Harbor task previews must use Cloudflare quick tunnels back to the laptop-local ChromiumRL reward service. This path must not require `TINKER_API_KEY`, Daytona, a GitHub token, or copied browser source.

## Benchmark Rules

Infrastructure changes are not sufficient unless they preserve a runnable benchmark ladder. Keep `w8-biayn benchmarks list`, README benchmark guidance, and rendered benchmark metadata current when changing R3, DOMDiff, Harbor, BrowserGym, WebArena, or AndroidWorld paths.

The default scorecard is MiniWoB smoke, DOMDiff local live smoke, WebVoyager DOMDiff held-out, Harbor DOMDiff browser/SWE tasks, WebArena, and AndroidWorld transfer.

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
uv run w8-biayn domdiff local up --image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff push-image --source-image android-world-domdiff:local --tag smoke --dry-run
uv run w8-biayn launch r3 --with-local-domdiff --benchmark webvoyager-domdiff-heldout --dry-run
uv run w8-biayn launch r3 --with-local-domdiff --benchmark harbor-domdiff-browser-swe --dry-run
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

Use `--no-sky` only for local validation when SkyPilot install is intentionally skipped; the default bootstrap path must install SkyPilot.
