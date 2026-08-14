# SkyPilot on GCP

This directory owns GCP provisioning and job lifecycle through SkyPilot. Training,
dataset, reward, and evaluation behavior remains in the provider-neutral `examples/`,
`scripts/`, and `src/` layers.

## Shared project contract

The canonical project is `transformer-tales` in `us-central1`. Compute and shared
buckets are private to explicitly authorized project principals. The runtime image
is distributed separately as a public GitHub Container Registry (GHCR) package.

One administrator runs the bootstrap with a Google Group as the runner principal:

```bash
export GLM47_GCP_RUNNER_PRINCIPAL=group:glm47-runners@example.com
export GLM47_GCP_ASSET_WRITER_PRINCIPAL=group:glm47-asset-writers@example.com  # optional
bash infra/gcp/bootstrap-project.sh
```

The bootstrap creates or reconciles:

- the project-level `glm47SkyPilotRunner` custom role;
- the shared `skypilot-v1@transformer-tales.iam.gserviceaccount.com` runtime identity;
- the private `transformer-tales-glm47-models` read-only runtime bucket;
- the private `transformer-tales-glm47-assets` read-only runtime bucket;
- the private `transformer-tales-glm47-runs` writable output bucket;
- runner, runtime, and optional asset-writer bindings scoped to those resources.

The bootstrap is additive: it does not revoke any broader role that a principal
or pre-existing service account may already hold. Audit existing IAM separately
before treating the resulting project as least privilege.

The runner role deliberately excludes project-wide `iam.serviceAccounts.actAs`.
Runners receive `roles/iam.serviceAccountUser` only on `skypilot-v1`. No
service-account key is created or stored.

The same split protects experiment inputs: ordinary runners and VMs can read the
models and assets buckets but cannot replace model, dataset, or adapter objects.
They can write checkpoints and logs only to the runs bucket. The optional
asset-writer group can publish models and other inputs. GHCR publication is
controlled by GitHub repository and package permissions rather than GCP IAM.

## Runtime image publication

The pinned Miles base image is approximately 25 GB compressed, which exceeds a
standard GitHub-hosted runner's 14 GB SSD before the derived layers are built.
The image therefore lives in GHCR but is built on an ephemeral, CPU-only GCP VM
with a 500 GB disk. No H100 is used for image publication.

Create a classic GitHub personal access token with `write:packages`, expose it
only in the launching shell, and launch the publisher from the repository root:

```bash
export GHCR_TOKEN='<write:packages token>'
sky launch -c glm47-runtime-publisher \
  infra/gcp/skypilot-publish-ghcr.yaml \
  --env GHCR_USERNAME=tokenbender \
  --env GHCR_IMAGE=ghcr.io/tokenbender/glm47-runtime \
  --env GLM47_SOURCE_REVISION="$(git rev-parse HEAD)" \
  --env GLM47_IMAGE_SOURCE=https://github.com/tokenbender/browser-is-all-you-need-upstream \
  --secret GHCR_TOKEN
```

The task builds the root `Dockerfile` for `linux/amd64` and publishes one
immutable source-revision tag:

```text
ghcr.io/tokenbender/glm47-runtime:git-<full-commit-sha>
```

The pinned SkyPilot 0.13 GCP image supplies Docker Engine and buildx. The
publisher starts and verifies that existing stack before authenticating, then
removes the GHCR credential from the builder when the job exits. It deliberately
does not mix Ubuntu's `docker.io` packages with SkyPilot's Docker CE packages.
The published manifest includes maximal BuildKit provenance. Inline SPDX SBOM
attestation is deliberately disabled: Syft's inventory of this CUDA training
image exceeds BuildKit's 40 MiB attestation-file ceiling, which otherwise makes
the exporter fail closed before pushing the manifest.

The publisher prints the content-addressed `sha256` reference and automatically
deletes its GCP VM after ten idle minutes. A newly created GHCR package is private
by default; a package owner must make
`glm47-runtime` public once in its package settings before GCP runners can pull it
without GitHub credentials. Public visibility should be enabled only after the
image contents and third-party notices have been reviewed. Miles itself is
Apache-2.0 licensed.

The currently published digest and the source revision that produced it are
recorded in `runtime-image.env`. Source that file before launch instead of
copying a mutable tag:

```bash
set -a
source infra/gcp/runtime-image.env
set +a
```

Verify the public boundary without reusing local Docker credentials:

```bash
: "${GLM47_RUNTIME_IMAGE:?Set the docker:ghcr.io/...@sha256:... value printed by the publisher}"
clean_docker_config="$(mktemp -d)"
DOCKER_CONFIG="${clean_docker_config}" docker pull \
  "${GLM47_RUNTIME_IMAGE#docker:}"
```

Do not use a mutable tag for a run. Supply the verified digest to SkyPilot:

```bash
sky launch infra/gcp/skypilot-h100-8.yaml \
  --image-id "${GLM47_RUNTIME_IMAGE}"
```

## Source payload

Run SkyPilot from the repository root. The H100 task declares `workdir: .`, so
SkyPilot syncs the local checkout into `~/sky_workdir` and executes later `setup`
and `run` commands there. The public GHCR image contains the shared toolchain and
dependencies, not this repository's source tree.

The root `.skyignore` is the fail-closed upload boundary for generated artifacts,
local environments, caches, credentials, checkpoints, and previous results. Keep
large models, datasets, adapters, checkpoints, and logs out of `workdir`; the next
task mounts those through private GCS buckets.

## Storage mounts

The H100 task mounts three existing private GCS buckets at the same paths used by
the Modal launcher:

| Container path | Bucket | Access and cache profile |
| --- | --- | --- |
| `/root/models` | `transformer-tales-glm47-models` | Read-only model checkpoints, 100 GB cache ceiling |
| `/workspace/assets` | `transformer-tales-glm47-assets` | Read-only datasets and adapters, 50 GB cache ceiling |
| `/workspace/runs` | `transformer-tales-glm47-runs` | Read-write checkpoints and logs, 200 GB cache ceiling |

SkyPilot does not currently support mounting independent subpaths from one cloud
bucket, so models and other assets are deliberately separate buckets. The cached
mount profiles leave 150 GB of the 500 GB boot disk outside configured cache
ceilings for the container image, source checkout, and runtime scratch space.

Bucket names are ordinary SkyPilot environment values and can be overridden with
`--env` when targeting another GCP project. IAM remains the enforcement boundary:
setting a different bucket name does not grant access to it.

## Runtime secrets

The H100 task declares `WANDB_API_KEY: null`, making the W&B credential required
without storing its value in YAML. Export it locally and pass only its name:

```bash
export WANDB_API_KEY='<personal W&B API key>'
sky launch infra/gcp/skypilot-h100-8.yaml \
  --secret WANDB_API_KEY \
  --image-id "${GLM47_RUNTIME_IMAGE}"
```

SkyPilot reads the value from the launching shell and redacts it from dashboard
and YAML output. The task disables API-server credential injection because the
training process does not launch nested SkyPilot operations.

Secret scopes remain separate:

- `WANDB_API_KEY` is the only secret required by the H100 SFT/RL task.
- `HF_TOKEN` is needed only when an asset-publishing process downloads gated
  model files; it is not needed to read the pre-populated private GCS mounts.
- `GHCR_TOKEN` is needed only by `skypilot-publish-ghcr.yaml` and never reaches
  an H100 training task.
- GCP access uses each runner's login plus the attached runtime service account;
  no service-account key is generated, uploaded, or passed as a task secret.

## Runtime setup and preflight

SkyPilot runs the H100 task's idempotent `setup` block after the image, source,
mounts, environment, and secret are available. It installs this checkout in
editable mode without resolving dependencies, then fails before training unless:

- the pinned Miles, SGLang, and FlashInfer runtime packages are compatible;
- `nvidia-smi` reports exactly eight H100 GPUs;
- the compiler and Bubblewrap sandbox executables are present;
- the mounted Hugging Face model matches revision
  `7dd20894a642a0aa287e9827cb1a1f7f91386b67`;
- the converted TP4/PP1/EP8 reference checkpoint and C++ task directory exist;
- the runs mount is writable and the required W&B secret is non-empty; and
- a harmless C++ program compiles and executes inside the network-isolated
  Bubblewrap reward sandbox.

The setup block is read-only apart from installing the synced checkout into the
container environment. It does not create a run directory or begin SFT/RL. A
successful setup ends with `GLM47_SKYPILOT_PREFLIGHT_READY`; any missing or
incompatible prerequisite stops the task first.

After the training job exits, SkyPilot waits 15 idle minutes, flushes mounted
filesystem writes, and tears the H100 VM down. Idleness is tied to jobs rather
than SSH sessions so an abandoned shell cannot keep the paid node alive.

## SFT and GRPO execution

The task has no default training stage. Four environment values are deliberately
`null`, so SkyPilot refuses a launch unless the caller supplies an explicit
`sft` or `grpo` stage, a unique run ID, the full source commit, and the exact
GHCR image digest. From a clean repository root:

```bash
export GLM47_STAGE=sft  # or: grpo
export MILES_RUN_ID="glm47-${GLM47_STAGE}-$(date -u +%Y%m%d-%H%M%S)"
export GLM47_SOURCE_COMMIT="$(git rev-parse HEAD)"
: "${GLM47_RUNTIME_IMAGE:?Set the immutable image value printed by the publisher}"

sky launch -c "${MILES_RUN_ID}" infra/gcp/skypilot-h100-8.yaml \
  --env GLM47_STAGE \
  --env MILES_RUN_ID \
  --env GLM47_SOURCE_COMMIT \
  --env GLM47_RUNTIME_IMAGE \
  --secret WANDB_API_KEY \
  --image-id "${GLM47_RUNTIME_IMAGE}"
```

The dispatcher accepts only `sft` and `grpo`, validates the identifiers and
immutable image reference, and refuses to reuse `/workspace/runs/<run-id>`.
Each run gets its own prepared dataset, checkpoints, logs, and W&B directory
beneath that GCS-backed prefix. The dispatcher always writes a top-level
SkyPilot launch receipt; an invoked training script also writes its stage receipt.
SFT uses the Modal-parity 3072-token, batch-20 defaults. GRPO warm-starts from
`/workspace/assets/adapters/sft` by default; override `MILES_LORA_ADAPTER_PATH`
with `--env MILES_LORA_ADAPTER_PATH=/workspace/assets/adapters/<name>` only when
intentionally selecting a different mounted adapter.

The source commit records provenance but cannot prove a dirty local tree was not
synced because `.git` is excluded from the payload. Check `git status --short`
before launching; benchmark-grade runs require an empty result.

## Runner onboarding

Each authorized user uses their own Google identity:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project transformer-tales
uv tool install --with pip "skypilot[gcp]==0.13.0"
sky check gcp
```

The public runtime image requires no registry credential. Required task secrets
must be passed with SkyPilot `--secret`; they do not belong in this repository or
the shared buckets.

The runtime image is not considered available until its published GHCR digest is
recorded, anonymously pullable, and supplied through `sky launch --image-id`.
Likewise, the 8xH100 task is not launchable until project quota and live A3
capacity are verified.

This is a single-node task, so it intentionally uses SkyPilot's standard network
tier. `network_tier: best` enables GCP GPUDirect and requires broad VPC/subnet
management permissions intended for multi-node traffic.

## Trust boundary

Authorized runners can manage SkyPilot VMs created in this project and can read
and write the shared runs bucket. Use unique user-prefixed run IDs. If collaborators
must be isolated from one another, use separate projects or a centrally managed
SkyPilot API server/workspace instead of this trusted-project setup.
