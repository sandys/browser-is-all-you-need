# browser-is-all-you-need

`browser-is-all-you-need` provides `w8-biayn`, a command-and-control CLI for BrowserGym reinforcement-learning smoke runs on rLLM, SkyRL, SkyPilot, and Google Cloud.

The current implementation focuses on the MiniWoB smoke path first, then WebArena, then SkyRL routing-replay R3 for MoE models.

## Bootstrap

Start from a fresh clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud
uv run w8-biayn launch miniwob --dry-run
```

Run the real MiniWoB smoke:

```bash
uv run w8-biayn launch miniwob
```

The launch command renders a SkyPilot YAML into `.w8-biayn/rendered/`, activates the local service account, launches with `sky launch -y --down`, and tears down the cluster after a successful job.

## GCP Requirements

`.gcp-service-account.json` is local-only and ignored by git. The service account must be able to pass `sky check gcp`.

At minimum, SkyPilot needs permissions to inspect and use GCP services, create/delete compute instances, networks/firewalls/disks, use service accounts, and create/delete storage buckets. If `doctor --cloud` reports GCP disabled, fix IAM before launching.

Useful commands:

```bash
uv run w8-biayn doctor --cloud
uv run w8-biayn status
uv run w8-biayn logs w8-biayn-miniwob
uv run w8-biayn down w8-biayn-miniwob
```

## CLI

```bash
uv run w8-biayn --help
uv run w8-biayn upstreams clone
uv run w8-biayn upstreams status
uv run w8-biayn data prepare miniwob --out ./data/miniwob
uv run w8-biayn config render miniwob --credentials .gcp-service-account.json
uv run w8-biayn launch miniwob --dry-run
```

Pinned upstreams are cloned into ignored cache paths:

- rLLM: `.cache/upstreams/rllm`
- SkyRL: `.cache/upstreams/SkyRL`

Do not vendor either upstream repository into this repo.

## WebArena

WebArena requires official service archives and runtime services. Provide a GCS prefix containing the archives:

```bash
uv run w8-biayn launch webarena --webarena-archives-gcs gs://<bucket>/webarena
```

Without `--webarena-archives-gcs` or external `WA_*` URLs, MiniWoB is the supported smoke path.

## R3 Pipeline

The first R3 target is SkyRL routing replay for `Qwen/Qwen1.5-MoE-A2.7B-Chat`.

```bash
uv run w8-biayn config render r3 --credentials .gcp-service-account.json
```

The rendered config enables routed expert capture and MoE routing replay:

- `generator.inference_engine.enable_return_routed_experts=true`
- `trainer.policy.megatron_config.moe_enable_routing_replay=true`
- `trainer.ref.megatron_config.moe_enable_routing_replay=true`

## Architecture

```mermaid
flowchart LR
  user[User / Operator] --> cli[w8-biayn CLI]
  cli --> doctor[doctor / auth checks]
  cli --> render[SkyPilot YAML renderer]
  cli --> data[BrowserGym dataset prep]
  cli --> upstreams[Ignored upstream clones]

  upstreams --> rllm[rLLM pinned source]
  upstreams --> skyrl[SkyRL pinned source]

  render --> sky[SkyPilot]
  sky --> gcp[GCP VM]
  gcp --> setup[Remote setup]
  setup --> skyrl_remote[SkyRL trainer]
  setup --> browsergym[BrowserGym envs]
  skyrl_remote --> adapter[w8_biayn BrowserGymEnv]
  adapter --> browsergym
  skyrl_remote --> checkpoints[Checkpoints / exports]
  checkpoints --> gcs[GCS artifact bucket]
```

## Smoke Workflow

```mermaid
sequenceDiagram
  participant U as User
  participant CLI as w8-biayn
  participant GC as gcloud
  participant SKY as SkyPilot
  participant VM as GCP VM
  participant SRL as SkyRL
  participant BG as BrowserGym

  U->>CLI: doctor --cloud
  CLI->>GC: activate service account
  CLI->>SKY: sky check gcp
  SKY-->>CLI: enabled or IAM blocker
  U->>CLI: launch miniwob
  CLI->>CLI: render .w8-biayn/rendered/miniwob.sky.yaml
  CLI->>SKY: sky launch -y --down
  SKY->>VM: provision A100:4 VM
  VM->>VM: install uv, clone SkyRL, install package
  VM->>CLI: run w8-biayn data prepare miniwob
  VM->>SRL: start SkyRL trainer entrypoint
  SRL->>BG: rollout through BrowserGymEnv
  BG-->>SRL: observations and rewards
  SRL-->>VM: logs, checkpoints, exports
  SKY-->>U: stream logs
  SKY->>VM: tear down after successful job
```

## Development

Follow [AGENTS.md](AGENTS.md). Any change that affects setup, cloud behavior, CLI UX, or pipeline flow must update this README, the Mermaid diagrams, and the relevant skills.

Run before handoff:

```bash
uv run --extra dev pytest
uv run python -m compileall src tests
```

