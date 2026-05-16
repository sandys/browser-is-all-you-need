# browser-is-all-you-need

`browser-is-all-you-need` provides `w8-biayn`, a command-and-control CLI for BrowserGym reinforcement-learning smoke runs on rLLM, SkyRL, SkyPilot, and Google Cloud.

The current implementation focuses on the MiniWoB smoke path first, then WebArena, then SkyRL routing-replay R3 for MoE models.

## Bootstrap

Start from a fresh clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --domdiff
uv run w8-biayn launch miniwob --dry-run
```

Run the real MiniWoB smoke:

```bash
uv run w8-biayn launch miniwob
```

The launch command renders a SkyPilot YAML into `.w8-biayn/rendered/`, activates the local service account, launches with `sky launch -y --down`, and tears down the cluster after a successful job.

Run the real DOMDiff reward-host smoke:

```bash
uv run w8-biayn domdiff local smoke --image android-world-domdiff:local
```

For R3 development, keep the DOMDiff image local and expose the reward service to the GCP trainer through Cloudflare quick tunnels:

```bash
uv run w8-biayn domdiff local up --image android-world-domdiff:local

uv run w8-biayn launch r3 \
  --chromiumrl-url https://<local-domdiff-reward-tunnel> \
  --benchmark webvoyager-domdiff-heldout \
  --credentials .gcp-service-account.json
```

`domdiff local up` prints the reward tunnel URL. Keep that terminal and machine running while the GCP trainer is active. CDP stays bound to the workstation by default; add `--publish-cdp` only for explicit CDP debugging.

If you explicitly want a GCP-hosted DOMDiff reward VM instead, push a local-only image to Google Artifact Registry first and use that registry URI for GCP:

```bash
uv run w8-biayn domdiff push-image \
  --source-image android-world-domdiff:local \
  --credentials .gcp-service-account.json

uv run w8-biayn domdiff smoke \
  --credentials .gcp-service-account.json \
  --local-reward-image android-world-domdiff:local
```

## GCP Requirements

`.gcp-service-account.json` is local-only and ignored by git. The service account must be able to pass `sky check gcp`.

At minimum, SkyPilot needs permissions to inspect and use GCP services, create/delete compute instances, networks/firewalls/disks, use service accounts, and create/delete storage buckets. If `doctor --cloud` reports GCP disabled, fix IAM before launching.

Useful commands:

```bash
uv run w8-biayn doctor --cloud
uv run w8-biayn doctor --cloud --domdiff
uv run w8-biayn status
uv run w8-biayn logs w8-biayn-miniwob
uv run w8-biayn down w8-biayn-miniwob
```

## CLI

```bash
uv run w8-biayn --help
uv run w8-biayn upstreams clone
uv run w8-biayn upstreams status
uv run w8-biayn benchmarks list
uv run w8-biayn data prepare miniwob --out ./data/miniwob
uv run w8-biayn config render miniwob --credentials .gcp-service-account.json
uv run w8-biayn launch miniwob --dry-run
uv run w8-biayn domdiff push-image --source-image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff local up --image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff local smoke --image android-world-domdiff:local
uv run w8-biayn domdiff smoke --dry-run
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

Run R3 against the local DOMDiff reward container through quick tunnels:

```bash
uv run w8-biayn domdiff local up --image android-world-domdiff:local

uv run w8-biayn launch r3 \
  --chromiumrl-url https://<local-domdiff-reward-tunnel> \
  --benchmark webvoyager-domdiff-heldout \
  --credentials .gcp-service-account.json
```

Or let `launch` start and tear down the local DOMDiff stack around the SkyPilot run:

```bash
uv run w8-biayn launch r3 \
  --with-local-domdiff \
  --local-domdiff-image android-world-domdiff:local \
  --benchmark webvoyager-domdiff-heldout \
  --credentials .gcp-service-account.json
```

Run R3 with a GCP-hosted DOMDiff reward VM only when you want the reward host to live in GCP:

```bash
uv run w8-biayn launch r3 \
  --with-domdiff \
  --benchmark webvoyager-domdiff-heldout \
  --credentials .gcp-service-account.json
```

The rendered config enables routed expert capture and MoE routing replay:

- `generator.inference_engine.enable_return_routed_experts=true`
- `trainer.policy.megatron_config.moe_enable_routing_replay=true`
- `trainer.ref.megatron_config.moe_enable_routing_replay=true`

## DOMDiff Rewards

`w8-biayn` uses the prebuilt image `ghcr.io/wootzapp/android-world-domdiff:daytona-92000b7` by default. It does not vendor AndroidWorld, WootzApp, or browser source code into this repository.

The fastest development path uses a local Docker image and quick tunnels:

```bash
uv run w8-biayn doctor --domdiff
uv run w8-biayn domdiff local up --image android-world-domdiff:local
uv run w8-biayn domdiff local verify
uv run w8-biayn domdiff local logs
uv run w8-biayn domdiff local down
```

`domdiff local up` starts the local Android/WootzApp container with KVM, starts the ChromiumRL reward service locally on `127.0.0.1:8080`, publishes a Cloudflare quick tunnel for reward HTTP, and writes state/logs under `.w8-biayn/domdiff-local/<run-id>/`. This path does not push Docker layers or copy browser source. Use `--publish-cdp` only when you need a temporary CDP tunnel for debugging.

The GCP-hosted DOMDiff lifecycle remains available for remote reward hosting. It creates one temporary GCP Compute VM with nested virtualization, starts the prebuilt Android/WootzApp container, copies in only the small `w8_biayn.rewards` adapter, publishes Cloudflare quick tunnels for reward HTTP and CDP, and writes state/logs under `.w8-biayn/domdiff/<run-id>/`.

When the reward image is local-only, use Artifact Registry instead of copying source or saving image tarballs:

```bash
uv run w8-biayn domdiff push-image \
  --source-image android-world-domdiff:local \
  --credentials .gcp-service-account.json
```

The command creates `us-central1-docker.pkg.dev/<project>/w8-biayn/android-world-domdiff:<tag>` by default, where `<tag>` is `local-<image-id>` unless `--tag` is supplied. `domdiff smoke`, `domdiff up`, and `launch r3 --with-domdiff` also accept `--local-reward-image android-world-domdiff:local`; they push the local image and then pass the Artifact Registry URI to the GCP reward VM. The remote VM logs in to Artifact Registry with `.gcp-service-account.json` before `docker pull`.

Useful commands:

```bash
uv run w8-biayn domdiff local up --image android-world-domdiff:local
uv run w8-biayn domdiff local smoke --image android-world-domdiff:local
uv run w8-biayn launch r3 --with-local-domdiff --benchmark webvoyager-domdiff-heldout
uv run w8-biayn domdiff push-image --source-image android-world-domdiff:local
uv run w8-biayn domdiff up
uv run w8-biayn domdiff verify
uv run w8-biayn domdiff logs
uv run w8-biayn domdiff down
```

The local smoke tears local processes down by default. Use `--keep-running` only when debugging or when the following GCP training run needs the tunnels to stay alive. The GCP-hosted smoke tears the VM down by default; use `--keep` only when debugging and then run `uv run w8-biayn domdiff down --run-id <run-id>`.

## Benchmarks

The pitch needs a scorecard, not just infrastructure. List the current benchmark ladder with:

```bash
uv run w8-biayn benchmarks list
```

Recommended order:

- `miniwob-smoke`: cheapest SkyPilot/SkyRL end-to-end check.
- `domdiff-local-live`: proves local KVM, WootzApp CDP, the reward quick tunnel, and reward service health without pushing the image.
- `webvoyager-domdiff-heldout`: primary browser-use DOMDiff benchmark for live no-anti-bot web tasks.
- `harbor-domdiff-browser-swe`: browser/SWE preview tasks where DOMDiff is the verifier.
- `webarena-browsergym`: reproducible self-hosted web benchmark through BrowserGym.
- `androidworld-transfer`: mobile transfer check for the claim that browser-use RL generalizes to app UI.

## Architecture

```mermaid
flowchart LR
  user[User / Operator] --> cli[w8-biayn CLI]
  cli --> doctor[doctor / auth checks]
  cli --> render[SkyPilot YAML renderer]
  cli --> data[BrowserGym dataset prep]
  cli --> bench[benchmark scorecard]
  cli --> domdiff_local[Local DOMDiff lifecycle]
  cli --> domdiff_gcp[GCP DOMDiff lifecycle]
  cli --> gar[Artifact Registry image push]
  cli --> upstreams[Ignored upstream clones]

  upstreams --> rllm[rLLM pinned source]
  upstreams --> skyrl[SkyRL pinned source]

  render --> sky[SkyPilot]
  sky --> gcp[GCP trainer VM]
  gcp --> setup[Remote trainer setup]
  setup --> skyrl_remote[SkyRL trainer]
  setup --> browsergym[BrowserGym envs]
  skyrl_remote --> adapter[w8_biayn BrowserGymEnv]
  adapter --> browsergym
  domdiff_local --> local_image[Local android-world-domdiff image]
  local_image --> local_container[Local Android/WootzApp container]
  local_container --> local_reward[w8_biayn ChromiumRL service on localhost]
  local_reward --> tunnels[Cloudflare quick tunnels]
  domdiff_gcp --> reward_vm[GCP nested-virt reward VM]
  gar --> artifact_image[Artifact Registry DOMDiff image]
  reward_vm --> artifact_image
  reward_vm --> gcp_reward[w8_biayn reward adapter on VM]
  gcp_reward --> tunnels
  tunnels --> skyrl_remote
  skyrl_remote --> checkpoints[Checkpoints / exports]
  checkpoints --> gcs[GCS artifact bucket]
```

## Smoke Workflow

```mermaid
sequenceDiagram
  participant U as User
  participant CLI as w8-biayn
  participant GC as gcloud
  participant DO as Local Docker
  participant CF as Cloudflare Quick Tunnels
  participant SKY as SkyPilot
  participant VM as GCP Trainer VM
  participant RW as Reward Adapter
  participant SRL as SkyRL
  participant BG as BrowserGym

  U->>CLI: doctor --cloud --domdiff
  CLI->>GC: activate service account
  CLI->>SKY: sky check gcp
  CLI->>DO: inspect android-world-domdiff:local and /dev/kvm
  SKY-->>CLI: GCP enabled or IAM blocker
  U->>CLI: domdiff local up --image android-world-domdiff:local
  CLI->>DO: run local Android/WootzApp container
  CLI->>RW: start local reward service with CDP_URL=ws://localhost:9224
  CLI->>CF: publish reward quick tunnel
  CF-->>CLI: chromiumrl_url
  U->>CLI: launch r3 --chromiumrl-url ...
  CLI->>CLI: render .w8-biayn/rendered/r3.sky.yaml
  CLI->>SKY: sky launch -y --down
  SKY->>VM: provision A100:4 VM
  VM->>VM: install uv, clone SkyRL, install package
  VM->>CLI: run w8-biayn data prepare / benchmark setup
  VM->>SRL: start SkyRL trainer entrypoint
  SRL->>BG: rollout through BrowserGymEnv
  SRL->>RW: call DOMDiff reward service when configured
  BG-->>SRL: observations and rewards
  RW-->>SRL: DOMDiff reward metrics
  SRL-->>VM: logs, checkpoints, exports
  SKY-->>U: stream logs
  SKY->>VM: tear down after successful job
  U->>CLI: domdiff local down
  CLI->>CF: stop quick tunnels
  CLI->>DO: stop local container unless --keep-container
```

## Development

Follow [AGENTS.md](AGENTS.md). Any change that affects setup, cloud behavior, CLI UX, or pipeline flow must update this README, the Mermaid diagrams, and the relevant skills.

Run before handoff:

```bash
uv run --extra dev pytest
uv run python -m compileall src tests
```
