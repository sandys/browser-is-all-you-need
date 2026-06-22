# browser-is-all-you-need

## Latest Smoke Stats

Completed on May 17, 2026 against `harbor-domdiff-browser-swe` with SkyRL R3 on Google Cloud (`asia-southeast1-a`, `a2-highgpu-8g`, `A100:8`) and the laptop-local `android-world-domdiff:local` reward image exposed through Cloudflare quick tunnel.

| Metric | Value |
| --- | ---: |
| Job status | `SUCCEEDED` |
| Tasks | 2 |
| Samples per task | 2 |
| Generated trajectories | 4 |
| `reward/avg_pass_at_2` | `1.0000` |
| `reward/avg_raw_reward` | `0.3354` |
| `environment/domdiff_total` | `0.6708` |
| `environment/harbor_reward` | `0.3354` |
| `environment/chromiumrl_enabled` | `1.0000` |
| `environment/harbor_oracle` | `1.0000` |
| `environment/rubric_passed` | `0.0000` |
| Avg response length | `177.0` tokens |
| Generation time | `670.2853s` |
| Policy train time | `40.5535s` |
| End-to-end R3 step time | `749.0343s` |
| Policy grad norm | `19.4491` |

Task IDs: `radix-ui__primitives-3548`, `chakra-ui__chakra-ui-8905`. This is an oracle-mode paid infrastructure smoke: it verifies the GCP/SkyPilot/SkyRL/Harbor/DOMDiff pipeline end to end, not a held-out competition leaderboard score.

`browser-is-all-you-need` provides `w8-biayn`, a command-and-control CLI for BrowserGym reinforcement-learning smoke runs on rLLM, SkyRL, SkyPilot, and Google Cloud.

The current implementation supports MiniWoB smoke runs, WebArena config rendering, DOMDiff reward hosting, a Harbor DOMDiff browser/SWE R3 smoke that runs task containers on GCP while using the local DOMDiff image through a Cloudflare reward tunnel, and a local OSWorld benchmark path through the pinned upstream Docker provider.

OSWorld benchmarking is supported through upstream clone/setup, validation, smoke runs, per-run result summaries, and domain benchmarks.

## Bootstrap

Start from a fresh clone:

```bash
./scripts/bootstrap.sh
cp /secure/path/service-account.json .gcp-service-account.json
uv run w8-biayn doctor --cloud --domdiff
uv run w8-biayn launch miniwob --dry-run
uv run w8-biayn harbor validate
uv run w8-biayn upstreams clone osworld
uv run w8-biayn osworld setup
uv run w8-biayn osworld validate
```

Run the real MiniWoB smoke:

```bash
uv run w8-biayn launch miniwob
```

The launch command renders a SkyPilot YAML into `.w8-biayn/rendered/`, runs SkyPilot with scoped environment variables from `.gcp-service-account.json`, launches with `sky launch -y --down`, and tears down the cluster after a successful job. It does not run `gcloud auth activate-service-account` or mutate global `gcloud config`.

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

`domdiff local up` prints the reward tunnel URL. Keep that terminal and machine running while the GCP trainer is active. CDP stays bound to the workstation by default; add `--publish-cdp` only for explicit CDP debugging. SkyPilot configs reject local/private DOMDiff URLs such as `localhost`, `127.0.0.1`, `192.168.x.x`, and `.local` names because the remote trainer cannot reach them.

Run the packaged Harbor DOMDiff R3 smoke with SkyRL on a GCP GPU container:

```bash
uv run w8-biayn launch r3 \
  --with-local-domdiff \
  --benchmark harbor-domdiff-browser-swe \
  --credentials .gcp-service-account.json
```

This path does not use Daytona, Tinker, Thinking Machines, or a GitHub token. SkyPilot provisions the GCP GPU VM, pulls the Google PyTorch GPU container, mounts the host Docker socket, and runs SkyRL plus the two packaged Harbor task containers on that VM. Each task publishes its preview through a Cloudflare quick tunnel so the laptop-local DOMDiff reward service can evaluate it.

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

`w8-biayn` reads that JSON directly and passes it to SkyPilot/GCP tooling through `GOOGLE_APPLICATION_CREDENTIALS`, `CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE`, and `CLOUDSDK_CORE_PROJECT`. Do not pre-authenticate with `gcloud auth`; the CLI is designed to work from a fresh machine with only the service-account JSON present.

At minimum, SkyPilot needs permissions to inspect and use GCP services, create/delete compute instances, networks/firewalls/disks, use service accounts, and create/delete storage buckets. If `doctor --cloud` reports GCP disabled, fix IAM before launching.

SkyPilot launch also prepares the `skypilot-v1` worker service account and its project bindings. `doctor --cloud` explicitly preflights the project-level SkyPilot launch permissions from SkyPilot's GCP backend, including Compute, Storage, Service Usage, IAM service-account creation, `resourcemanager.projects.getIamPolicy`, and `resourcemanager.projects.setIamPolicy`.

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
uv run w8-biayn harbor list
uv run w8-biayn harbor validate
uv run w8-biayn harbor oracle-smoke --task radix-ui__primitives-3548 --dry-run
uv run w8-biayn harbor prepare-data --out /tmp/w8-harbor-data --task radix-ui__primitives-3548 --oracle
uv run w8-biayn data prepare miniwob --out ./data/miniwob
uv run w8-biayn config render miniwob --credentials .gcp-service-account.json
uv run w8-biayn config render r3 --benchmark harbor-domdiff-browser-swe
uv run w8-biayn launch miniwob --dry-run
uv run w8-biayn domdiff push-image --source-image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff local up --image android-world-domdiff:local --dry-run
uv run w8-biayn domdiff local smoke --image android-world-domdiff:local
uv run w8-biayn domdiff smoke --dry-run
uv run w8-biayn osworld list --domain os --smoke-candidates
uv run w8-biayn osworld smoke --dry-run
uv run w8-biayn osworld run --suite tiny --dry-run
uv run w8-biayn osworld benchmark --dry-run
uv run w8-biayn scalecua sft --help
```

Pinned upstreams are cloned into ignored cache paths:

- rLLM: `.cache/upstreams/rllm`
- SkyRL: `.cache/upstreams/SkyRL`
- OSWorld: `.cache/upstreams/OSWorld`

Do not vendor upstream repositories into this repo.

## WebArena

WebArena requires official service archives and runtime services. Provide a GCS prefix containing the archives:

```bash
uv run w8-biayn launch webarena --webarena-archives-gcs gs://<bucket>/webarena
```

Without `--webarena-archives-gcs` or external `WA_*` URLs, MiniWoB is the supported smoke path.

## OSWorld

OSWorld runs through the pinned upstream clone in `.cache/upstreams/OSWorld` and keeps the upstream Python environment inside `.cache/upstreams/OSWorld/.venv`.

Start with the local benchmark path:

```bash
uv run w8-biayn upstreams clone osworld
uv run w8-biayn osworld setup
uv run w8-biayn osworld validate
uv run w8-biayn osworld list --domain os --smoke-candidates
uv run w8-biayn osworld smoke --dry-run
uv run w8-biayn osworld run --suite tiny --dry-run
uv run w8-biayn osworld benchmark --dry-run
```

Run a real one-task smoke after validation passes:

```bash
uv run w8-biayn osworld smoke   --task os/e0df059f-28a6-4169-924f-b9623e7184cc
```

Run the local tiny suite:

```bash
uv run w8-biayn osworld run --suite tiny
uv run w8-biayn osworld results
```

Run a domain benchmark against a local OpenAI-compatible model server:

```bash
uv run w8-biayn osworld benchmark   --domain os   --model qwen3-vl-8b   --base-url http://127.0.0.1:8000/v1   --api-key EMPTY
```

If you want custom subsets, pass your own grouped JSON path with `--taskset /path/to/tasks.json`. Use a JSON object where each top-level key is an OSWorld domain and each value is a list of task IDs from that domain. For example:

```json
{
  "os": [
    "e0df059f-28a6-4169-924f-b9623e7184cc",
    "28cc3b7e-b194-4bc9-8353-d04c0f4d56d2"
  ],
  "chrome": [
    "06fe7178-4491-4589-810f-2e2bc9502122"
  ]
}
```

List available tasks first with commands such as `uv run w8-biayn osworld list --domain os` and `uv run w8-biayn osworld list --domain chrome`, then dry-run the custom subset with `uv run w8-biayn osworld benchmark --taskset /path/to/tasks.json --dry-run`, and finally run it with `uv run w8-biayn osworld benchmark --taskset /path/to/tasks.json --model qwen3-vl-8b --base-url http://127.0.0.1:8000/v1 --api-key EMPTY`.

## ScaleCUA Conversion

ScaleCUA action annotations use normalized actions such as `<action>click(x=0.5, y=0.25)</action>`. The OSWorld Qwen2.5-VL agent expects `computer_use` tool calls and then converts those tool calls into executable PyAutoGUI actions. Convert ScaleCUA actions into the OSWorld Qwen tool-call format before using them for OSWorld-oriented SFT.

Download only lightweight annotation files first:

```bash
huggingface-cli download OpenGVLab/ScaleCUA-Data \
  --repo-type dataset \
  --include 'meta.json' \
  --include '*.jsonl' \
  --include '**/*.jsonl' \
  --local-dir .w8-biayn/scalecua/raw
```

Create a 1k desktop/web smoke conversion:

```bash
python scripts/convert_scalecua_to_osworld_toolcalls.py \
  --annotations .w8-biayn/scalecua/raw/annotations \
  --limit 1000 \
  --platforms windows ubuntu mac web \
  --out .w8-biayn/scalecua/prepared/smoke-toolcall.jsonl \
  --rejects .w8-biayn/scalecua/prepared/smoke-rejected.jsonl
```

Validate the smoke output before LoRA training:

```bash
wc -l .w8-biayn/scalecua/prepared/smoke-toolcall.jsonl
wc -l .w8-biayn/scalecua/prepared/smoke-rejected.jsonl
head -1 .w8-biayn/scalecua/prepared/smoke-toolcall.jsonl | python -m json.tool
```

The converted assistant target should look like:

```xml
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [960, 270]}}
</tool_call>
```

The first conversion pass keeps safe desktop/web actions (`click`, `rightclick`, `doubleclick`, `moveto`, `dragto`, `write`, `press`, `hotkey`, `scroll`, `swipe`, `wait`, `terminate`, `success`, `failure`) and writes unsupported actions such as `response`, `open_app`, `long_press`, and `tripleclick` to the reject JSONL.

Run a local Qwen2.5-VL LoRA SFT smoke with Transformers and PEFT:

```bash
uv run --extra sft w8-biayn scalecua sft \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train .w8-biayn/scalecua/prepared/smoke-toolcall.jsonl \
  --output .w8-biayn/scalecua/lora/qwen25vl7b-smoke \
  --max-steps 100 \
  --batch-size 1 \
  --grad-accum 8
```

The SFT command loads `Qwen2_5_VLForConditionalGeneration`, applies PEFT LoRA to the language-model projection/MLP modules, builds Qwen image+text chat inputs from the converted JSONL, and masks prompt tokens with `labels=-100` so loss is computed only on the assistant `<tool_call>` target. It saves only the LoRA adapter and processor files under the ignored `.w8-biayn/scalecua/lora/` tree.

For real-image LoRA training, build the subset before downloading large image archives. The planner uses `meta.json` to map annotations to image roots, ranks archives by convertible rows per compressed GB, downloads only selected archive parts, extracts only selected image members, and writes a training JSONL whose `image` fields point at extracted files. Preview the archive cost first:

```bash
uv run --extra sft python scripts/prepare_scalecua_real_subset.py \
  --limit 10000 \
  --dry-run \
  --out .w8-biayn/scalecua/prepared/train-10k-toolcall.jsonl \
  --rejects .w8-biayn/scalecua/prepared/train-10k-rejected.jsonl \
  --report .w8-biayn/scalecua/reports/train-10k-summary.md
```

Download archive parts and extract only selected images by removing `--dry-run`:

```bash
uv run --extra sft python scripts/prepare_scalecua_real_subset.py \
  --limit 10000 \
  --out .w8-biayn/scalecua/prepared/train-10k-toolcall.jsonl \
  --rejects .w8-biayn/scalecua/prepared/train-10k-rejected.jsonl \
  --report .w8-biayn/scalecua/reports/train-10k-summary.md
```

Run a W&B-tracked LoRA after the summary shows `missing_images: 0`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra sft w8-biayn scalecua sft \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --train .w8-biayn/scalecua/prepared/train-10k-toolcall.jsonl \
  --output .w8-biayn/scalecua/lora/qwen25vl7b-scalecua-10k \
  --max-steps 1000 \
  --batch-size 1 \
  --grad-accum 8 \
  --wandb-project scalecua-osworld \
  --wandb-run-name qwen25vl7b-scalecua-10k
```

## R3 Pipeline

The first R3 target is SkyRL routing replay for `moonshotai/Moonlight-16B-A3B-Instruct`.
SkyRL still owns the trainer and rollout lifecycle; the rendered job enables SkyRL's internal vLLM
engine in `mp` mode because router replay needs routed-expert metadata from rollout.

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

Run the Harbor DOMDiff browser/SWE R3 smoke with self-hosted SkyRL:

```bash
uv run w8-biayn harbor list
uv run w8-biayn harbor validate

uv run w8-biayn launch r3 \
  --with-local-domdiff \
  --local-domdiff-image android-world-domdiff:local \
  --benchmark harbor-domdiff-browser-swe \
  --credentials .gcp-service-account.json
```

The Harbor smoke uses the two packaged tasks `radix-ui__primitives-3548` and `chakra-ui__chakra-ui-8905` by default. Select a subset with repeated `--harbor-task <task-id>` flags. The default uses packaged oracle patches so infrastructure can be smoked deterministically; pass `--no-harbor-oracle` when you want the model-generated `<solution>...</solution>` script to determine the reward.

The rendered Harbor config is different from the MiniWoB/WebArena SkyRL path. It installs Docker and Cloudflare on the SkyPilot host, clones the pinned SkyRL repository into `$HOME/.cache/w8-biayn/upstreams`, then starts `us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310` with GPU access, `--shm-size=32g`, and the host Docker socket mounted. R3 reuses the cached Harbor virtualenv when present and recreates it if incomplete, then runs `uv sync --active --extra megatron --extra gcp` from the SkyRL checkout so SkyRL's own `tool.uv` dependency overrides are honored and installs this repo into the same environment. It renders `trainer.strategy=megatron`, Megatron TP/PP/CP/EP settings, vLLM's `mp` distributed executor backend, vLLM MoE expert parallelism, and `trainer.algorithm.use_kl_loss=false`, matching SkyRL's router replay requirements. Harbor DOMDiff R3 defaults to `H100:8`, matching SkyRL's Moonlight router replay recipe. A100 40GB overrides can reach Harbor rollout and DOMDiff reward scoring, but need CPU optimizer offload for the Megatron optimizer step; the CLI prints a warning and renders offload before paid A100 40GB Harbor R3 launches. The GPU count is rendered from the accelerator request, so the container does not depend on SkyPilot host-only shell variables. Inside that Google GPU container, `w8-biayn harbor prepare-data` writes SkyRL parquet files and `w8_biayn.integrations.skyrl_harbor_main` registers the `harbor-domdiff` SkyRL-Gym environment inside SkyRL's Ray entrypoint before training begins.

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

`domdiff local up` starts the local Android/WootzApp container with KVM, starts the ChromiumRL reward service locally on `127.0.0.1:8080`, publishes a Cloudflare quick tunnel for reward HTTP, and writes state/logs under `.w8-biayn/domdiff-local/<run-id>/`. This path does not push Docker layers or copy browser source. Use `--publish-cdp` only when you need a temporary CDP tunnel for debugging. Pass only Cloudflare or otherwise publicly reachable tunnel URLs to remote SkyPilot runs.

For Harbor tasks, the browser preview runs inside a Docker task container on the GCP trainer VM. Each SkyRL trajectory gets a unique task container name so parallel samples cannot remove or overwrite each other's verifier files. The task verifier starts its own Cloudflare quick tunnel for that preview URL and sends the preview URL to the local ChromiumRL reward service through `CHROMIUMRL_API_URL`. This keeps the DOMDiff image local while still allowing GCP task containers and SkyRL to evaluate the same browser state.

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
- `harbor-domdiff-browser-swe`: two packaged Harbor browser/SWE preview tasks with definitive DOMDiff rubrics; task containers run on the GCP trainer VM and publish previews back to the laptop-local reward service.
- `webarena-browsergym`: reproducible self-hosted web benchmark through BrowserGym.
- `osworld-smoke`: one low-risk OSWorld desktop task through the pinned upstream Docker provider.
- `osworld-tiny-local`: curated local OSWorld tiny suite with run records and parsed result summaries.
- `androidworld-transfer`: mobile transfer check for the claim that browser-use RL generalizes to app UI.

## Architecture

```mermaid
flowchart LR
  user[User / Operator] --> cli[w8-biayn CLI]
  cli --> doctor[doctor / service-account checks]
  cli --> sa[.gcp-service-account.json scoped env]
  cli --> render[SkyPilot YAML renderer]
  cli --> data[BrowserGym dataset prep]
  cli --> bench[benchmark scorecard]
  cli --> harbor[Harbor task commands]
  cli --> domdiff_local[Local DOMDiff lifecycle]
  cli --> domdiff_gcp[GCP DOMDiff lifecycle]
  cli --> gar[Artifact Registry image push]
  cli --> osworld_cli[OSWorld benchmark commands]
  cli --> upstreams[Ignored upstream clones]

  upstreams --> rllm[rLLM pinned source]
  upstreams --> skyrl[SkyRL pinned source]
  upstreams --> osworld_src[OSWorld pinned source]

  sa --> sky[SkyPilot]
  render --> sky
  sky --> gcp[GCP trainer VM]
  gcp --> setup[Remote trainer setup]
  setup --> skyrl_remote[SkyRL trainer]
  setup --> browsergym[BrowserGym envs]
  setup --> gpu_container[Google GPU Docker container]
  gpu_container --> skyrl_harbor[w8_biayn SkyRL Harbor Ray entrypoint]
  skyrl_harbor --> harbor_env[harbor-domdiff SkyRL-Gym env]
  harbor_env --> task_docker[GCP Harbor task containers]
  task_docker --> preview_tunnels[Task preview quick tunnels]
  osworld_cli --> osworld_env[OSWorld upstream .venv]
  osworld_env --> osworld_docker[OSWorld Docker provider]
  osworld_docker --> osworld_runs[.w8-biayn/osworld run records]
  skyrl_remote --> adapter[w8_biayn BrowserGymEnv]
  adapter --> browsergym
  domdiff_local --> local_image[Local android-world-domdiff image]
  local_image --> local_container[Local Android/WootzApp container]
  local_container --> local_reward[w8_biayn ChromiumRL service on localhost]
  local_reward --> tunnels[Cloudflare quick tunnels]
  preview_tunnels --> local_reward
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
  participant SA as Service-account JSON
  participant DO as Local Docker
  participant CF as Cloudflare Quick Tunnels
  participant SKY as SkyPilot
  participant VM as GCP Trainer VM
  participant RW as Reward Adapter
  participant SRL as SkyRL
  participant BG as BrowserGym
  participant GPU as Google GPU Container
  participant HT as Harbor Task Container

  U->>CLI: doctor --cloud --domdiff
  CLI->>SA: read project_id and build scoped credential env
  CLI->>SKY: sky check gcp with service-account env
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
  SKY->>VM: provision H100:8 VM for Harbor R3, A100:4 for lighter paths
  VM->>VM: install uv, clone SkyRL, install package
  VM->>CLI: run w8-biayn data prepare / benchmark setup
  VM->>SRL: start SkyRL trainer entrypoint
  SRL->>BG: rollout through BrowserGymEnv
  SRL->>RW: call DOMDiff reward service when configured
  BG-->>SRL: observations and rewards
  RW-->>SRL: DOMDiff reward metrics
  U->>CLI: launch r3 --benchmark harbor-domdiff-browser-swe --with-local-domdiff
  CLI->>SKY: sky launch Harbor R3 YAML
  SKY->>VM: provision GPU VM
  VM->>GPU: run Google PyTorch GPU container with host Docker socket
  GPU->>SRL: start skyrl_harbor_main
  SRL->>HT: run harbor-domdiff env and task Docker container
  HT->>CF: publish task preview quick tunnel
  HT->>RW: request DOMDiff score through CHROMIUMRL_API_URL
  RW-->>SRL: rubric reward
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
