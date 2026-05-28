"""Project constants and upstream pins."""

from __future__ import annotations

from dataclasses import dataclass

RLLM_REPO = "https://github.com/rllm-org/rllm.git"
SKYRL_REPO = "https://github.com/NovaSky-AI/SkyRL.git"

RLLM_PIN = "9b9e3582b0d73708cea5d1782571c3cc184d248c"
SKYRL_PIN = "bfa4dc338b0db629f6222364bd1d8645f43baaa2"

DEFAULT_CREDENTIALS_PATH = ".gcp-service-account.json"
DEFAULT_RENDER_DIR = ".w8-biayn/rendered"
DEFAULT_CLUSTER_PREFIX = "w8-biayn"
DEFAULT_ACCELERATORS = "A100:4"
DEFAULT_HARBOR_R3_ACCELERATORS = "H100:8"
DEFAULT_DOMDIFF_IMAGE = "ghcr.io/wootzapp/android-world-domdiff:daytona-92000b7"
DEFAULT_DOMDIFF_LOCAL_IMAGE = "android-world-domdiff:local"
DEFAULT_DOMDIFF_ARTIFACT_LOCATION = "us-central1"
DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY = "w8-biayn"
DEFAULT_DOMDIFF_ARTIFACT_IMAGE = "android-world-domdiff"
DEFAULT_DOMDIFF_ZONE = "us-central1-a"
DEFAULT_DOMDIFF_MACHINE_TYPE = "n2-standard-4"
DEFAULT_DOMDIFF_MIN_CPU_PLATFORM = "Intel Cascade Lake"
DEFAULT_DOMDIFF_BOOT_DISK_TYPE = "pd-balanced"
DEFAULT_DOMDIFF_VOLUME_GB = 100
DEFAULT_GPU_CONTAINER_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310"

# Custom-kernel R&D lane. `baseline` renders byte-for-byte identically to the
# stock path; `a100-kernel-lab` activates gated Triton patches; `a100-safe` is
# observability-only (numerics unchanged).
OPTIMIZATION_PROFILES = ("baseline", "a100-kernel-lab", "a100-safe")
DEFAULT_OPTIMIZATION_PROFILE = "baseline"
# Tier-A SkyRL-numerics kernels safe to activate inside the full Harbor R3 trainer
# (correctness gate only). Tier-B MLA/MoE kernels run in the single-device lab.
DEFAULT_KERNEL_LAB_SPEC = "logprob,entropy,ppo"


@dataclass(frozen=True)
class Upstream:
    name: str
    repo: str
    pin: str


UPSTREAMS = {
    "rllm": Upstream("rllm", RLLM_REPO, RLLM_PIN),
    "skyrl": Upstream("SkyRL", SKYRL_REPO, SKYRL_PIN),
}
