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
assets bucket but cannot replace its model, dataset, or adapter objects. They can
write checkpoints and logs only to the runs bucket. The optional asset-writer
group can publish inputs. GHCR publication is controlled by GitHub repository and
package permissions rather than GCP IAM.

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

The publisher prints the content-addressed `sha256` reference and automatically
deletes its GCP VM after ten idle minutes. A newly created GHCR package is private
by default; a package owner must make
`glm47-runtime` public once in its package settings before GCP runners can pull it
without GitHub credentials. Public visibility should be enabled only after the
image contents and third-party notices have been reviewed. Miles itself is
Apache-2.0 licensed.

Verify the public boundary without reusing local Docker credentials:

```bash
clean_docker_config="$(mktemp -d)"
DOCKER_CONFIG="${clean_docker_config}" docker pull \
  ghcr.io/<owner>/glm47-runtime@sha256:<digest>
```

Do not use a mutable tag for a run. Supply the verified digest to SkyPilot:

```bash
sky launch infra/gcp/skypilot-h100-8.yaml \
  --image-id docker:ghcr.io/<owner>/glm47-runtime@sha256:<digest>
```

## Runner onboarding

Each authorized user uses their own Google identity:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project transformer-tales
uv tool install --with pip "skypilot[gcp]==0.13.0"
sky check gcp
```

The public runtime image requires no registry credential. User secrets such as
`HF_TOKEN` and `WANDB_API_KEY` must be passed with SkyPilot `--secret`; they do
not belong in this repository or the shared bucket.

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
