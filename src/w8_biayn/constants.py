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


@dataclass(frozen=True)
class Upstream:
    name: str
    repo: str
    pin: str


UPSTREAMS = {
    "rllm": Upstream("rllm", RLLM_REPO, RLLM_PIN),
    "skyrl": Upstream("SkyRL", SKYRL_REPO, SKYRL_PIN),
}

