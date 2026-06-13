# Repository Development Guide

## Fresh-Machine Bootstrap Is Required

This repository is shipped to other people. Development must assume a fresh third-party computer, not this local workstation.

Every feature that requires a tool, cloud CLI, Python extra, environment variable, generated config, upstream clone, or local setup step must be represented in one of these places:

- `pyproject.toml` / `uv.lock` for Python package dependencies.
- `scripts/bootstrap.sh` for machine bootstrap and external CLI setup.
- `w8-biayn doctor` for validation and actionable diagnostics.
- `README.md` for user-facing setup, command usage, architecture, and workflow diagrams.
- `.agents/skills/` for AI-facing repo workflow guidance.
- `.gitignore` for local-only generated files, secrets, caches, upstream clones, task pools, logs, and checkpoints.

Do not rely on globally installed tools unless bootstrap installs them or `doctor` reports a clear missing prerequisite with the exact next action.

## Required User Path

A new user must be able to start from a clean clone and run:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --cpp-perf
uv run w8-biayn data doctor
uv run w8-biayn upstreams clone
uv run w8-biayn launch cpp-smoke --dry-run
```

The real GCP smoke path is:

```bash
uv run w8-biayn launch cpp-smoke
```

The first local reward-harness path is:

```bash
uv run w8-biayn cpp harness run --task path/to/task.json --candidate path/to/candidate.cpp --dry-run
uv run w8-biayn cpp reward score --task path/to/task.json --model-output path/to/output.md --dry-run
```

If any development change invalidates those commands, update bootstrap, CLI help, tests, README, and this file in the same change.

The repeatable dataset setup path is:

```bash
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data supercoder download --out .w8-biayn/data/supercoder
uv run w8-biayn data pie build-tests-manifest --inputs-outputs-basepath .w8-biayn/data/pie/cases --coverage-json .w8-biayn/data/pie/coverage.json --out .w8-biayn/data/pie/tests-manifest.json
uv run w8-biayn data pie build-tasks --pairs .w8-biayn/data/pie/train.jsonl --tests-json .w8-biayn/data/pie/tests-manifest.json --out .w8-biayn/data/tasks/train --split train
uv run w8-biayn data pie build-tasks --pairs .w8-biayn/data/pie/validation.jsonl --tests-json .w8-biayn/data/pie/tests-manifest.json --out .w8-biayn/data/tasks/validation --split validation
uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks --out .w8-biayn/data/skyrl
uv run w8-biayn data cache upload --path .w8-biayn/data/skyrl --credentials .gcp-service-account.json
```

No PIE or SuperCoder conversion may be performed as a one-off. If a conversion or data cleanup is needed, implement it in `w8-biayn data ...`, record schema/checksum metadata, update docs, and add tests.

## Documentation And Diagrams Must Stay Current

`README.md` is the canonical user-facing entrypoint. Keep it current whenever commands, setup steps, cloud behavior, supported pipelines, IAM requirements, data flow, task schema, measurement flow, or operational workflows change.

The README must include Mermaid architecture and workflow diagrams. When implementation changes affect task construction, sandbox measurement, launch flow, cloud boundaries, upstream responsibilities, or teardown behavior, update the diagrams in the same change.

AI-facing skills must also stay current. When repo-specific workflows change, update `.agents/skills/w8-biayn-framework/` and preserve the symlinked skill discovery paths. If the change is generic to skill authoring, update `.agents/skills/agent-skills-framework/` as well.

## Commit Discipline

Prefer many focused commits over one large commit. Keep each commit internally coherent: include the code, docs, tests, and skill updates needed for that one logical change.

## Secrets And Local State

`.env` and `.gcp-service-account.json` are local-only and must never be committed.

Upstream source clones belong in `.cache/upstreams/` through:

```bash
uv run w8-biayn upstreams clone
```

Do not vendor rLLM, SkyRL, PIE, CodeNet, SuperCoder data, gem5 outputs, model checkpoints, generated task pools, or upstream repositories into this repository.

## C++ Performance-RL Rules

Phase 1 is C++ only. Go, BrowserGym, DOMDiff, Harbor, WebArena, MiniWoB, and AndroidWorld are not part of the active project surface.

Use `/tmp/ENGINEERING_SPEC_v2_cpp_only.md` as the source spec until the equivalent content is moved into repo docs. The project goal is to train an open-weight model to rewrite correct C++ programs so they run faster while preserving correctness.

Training must use SkyRL/rLLM. Do not write a custom trainer, do not use PIE's old Hugging Face Trainer path as the active trainer, and do not use SuperCoder as a replacement trainer. PIE, LearningOpt PIE, and SuperCoder may be studied and ported into this repo's data/reward/eval surfaces only.

Task construction must follow the PIE data discipline:

- Use PIE C++ slower-to-faster pairs.
- Never expose the fast `v1` solution in the model prompt.
- Use `v1` only as an oracle for test generation and reference performance.
- Preserve train/test splits by problem.
- Do not admit a task without visible tests, hidden tests, reference performance, and coverage of at least 95% line / 85% branch.

Dataset bundles for SkyRL must include GRPO parquet, SFT JSONL, copied task JSON, and `_w8_data_manifest.json` with schema version, sources, options, byte sizes, and checksums. Default schema version is `cpp-perf-v1`; default GCS prefix is `gs://<project>-w8-biayn/datasets/cpp-perf/cpp-perf-v1/skyrl`.

The reward path must be correctness gated:

- Invalid response format: negative reward.
- Compile or sanitizer failure: negative reward.
- Partial tests: shaped but still below a fully correct answer.
- Correct answers: base reward of 1.0 plus bounded instruction-count efficiency bonus.
- Instruction count from `perf stat -e instructions:u` is the fast RL reward metric.
- gem5 remains the deterministic gold reference for calibration and final reporting.

## Cloud Development Rules

All GCP/SkyPilot workflows must support dry-run rendering before launching resources.

Any command that provisions cloud resources must:

- Use `.gcp-service-account.json` through scoped environment variables such as `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and `CLOUDSDK_CORE_PROJECT`.
- Never run `gcloud auth activate-service-account` or mutate the user's global `gcloud config`.
- Check the full project-level SkyPilot GCP launch IAM set before paid launches, including compute, storage, service usage, service-account creation, and project IAM policy update permissions.
- Avoid printing credential contents.
- Render the SkyPilot YAML into an ignored path.
- Provide status, logs, and teardown commands through `w8-biayn`.

SkyPilot is the Phase-1 bridge because the inherited Google training path works. dstack migration is deferred until the one-example gem5/perf/reward loop is stable.

`cpp-sft` and `cpp-grpo` launch through SkyPilot and restore their dataset bundle from GCS before invoking SkyRL. The GRPO entrypoint must register `cpp-perf` inside the SkyRL Ray task and then delegate to SkyRL `BasePPOExp(cfg).run()`.

## Benchmark Rules

Infrastructure changes are not sufficient unless they preserve a runnable benchmark ladder. Keep `w8-biayn benchmarks list`, README benchmark guidance, and rendered benchmark metadata current when changing the C++ task builder, harness, reward, SkyRL/rLLM integration, or eval protocol.

The current scorecard is `pie-one-smoke`, `pie-10-task`, `pie-heldout-effk`, `sft-cold-start`, and `grpo-tiny`.

## Custom-Kernel R&D Rules

The custom Triton-kernel lane remains opt-in and off by default. It is secondary to the C++ reward harness and must not change default training numerics or rendered output unless explicitly selected.

Do not vendor or edit SkyRL/Megatron. Kernel numerics need a CUDA GPU; local CPU-only validation can cover imports and command construction only.

## Tests And Validation

Before handing off changes, run:

```bash
uv run --extra dev pytest
uv run --extra dev ruff check src tests scripts
uv run python -m compileall src tests
```

For bootstrap-affecting changes, also run:

```bash
./scripts/bootstrap.sh --no-sky
uv run w8-biayn --help
uv run w8-biayn data doctor
uv run w8-biayn doctor --cpp-perf
uv run w8-biayn config render cpp-smoke --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-smoke --dry-run --credentials .gcp-service-account.json
uv run w8-biayn launch cpp-grpo --dry-run --credentials .gcp-service-account.json
python3 .agents/skills/agent-skills-framework/scripts/validate_skill.py .agents/skills/w8-biayn-framework
```

Use `--no-sky` only for local validation when SkyPilot install is intentionally skipped; the default bootstrap path must install SkyPilot.
