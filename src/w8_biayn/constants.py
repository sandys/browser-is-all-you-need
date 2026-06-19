"""Project constants and upstream pins."""

from __future__ import annotations

from dataclasses import dataclass

RLLM_REPO = "https://github.com/rllm-org/rllm.git"
SKYRL_REPO = "https://github.com/NovaSky-AI/SkyRL.git"
PIE_PERF_REPO = "https://github.com/madaan/pie-perf.git"
LEARNINGOPT_PIE_REPO = "https://github.com/LearningOpt/pie.git"
SUPERCODER_DATASET = "LLM4Code/llm_superoptimizer_ds"

RLLM_PIN = "9b9e3582b0d73708cea5d1782571c3cc184d248c"
SKYRL_PIN = "bfa4dc338b0db629f6222364bd1d8645f43baaa2"
PIE_PERF_PIN = "8f74491065bcd5b66d3913bf31cfba0ecf4ea5f0"
LEARNINGOPT_PIE_PIN = "eb3d72bc4f3c9095d0f62506340bc8ca0ef27b09"

DEFAULT_CREDENTIALS_PATH = ".gcp-service-account.json"
DEFAULT_RENDER_DIR = ".w8-biayn/rendered"
DEFAULT_CLUSTER_PREFIX = "w8-biayn"
DEFAULT_CPP_SMOKE_ACCELERATORS = "H100:8"
DEFAULT_CPP_TRAIN_ACCELERATORS = "A100:8"
DEFAULT_CPP_EVAL_ACCELERATORS = "A100:1"
DEFAULT_CPP_MODEL = "zai-org/GLM-5.1"
DEFAULT_CPP_TRAIN_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_CPP_CONTAINER_IMAGE = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310"
DEFAULT_CPP_EVAL_VLLM_VERSION = "0.6.6.post1"
DEFAULT_CPP_EVAL_TRANSFORMERS_VERSION = "4.57.6"
DEFAULT_CPP_SANDBOX_IMAGE = "w8-biayn-cpp-perf:latest"
CPP_DATA_SCHEMA_VERSION = "cpp-perf-v1"
DEFAULT_DATA_ROOT = ".w8-biayn/data"
DEFAULT_SKYRL_DATA_DIR = ".w8-biayn/data/skyrl"

PIE_TRAJECTORIES_URL = "https://drive.google.com/file/d/19IL3VETwVI9rdibB979Xm4gEWYwn0CkV/view?usp=sharing"
PIE_CXX_SPLITS_URL = "https://drive.google.com/file/d/1NqMT7kqCwk99hj4BjpUcsxLIzPFv_DtT/view?usp=sharing"
PIE_PUBLIC_TEST_CASES_URL = "https://drive.google.com/file/d/1RcUpZMOR8L2xYYWDZx7I0tHFzFgg7COO/view?usp=share_link"
PIE_GENERATED_TEST_CASES_URL = "https://drive.google.com/file/d/1migwX4wpED0gDDxn7gS6q55vWeXIDgId/view?usp=drive_link"
PIE_MERGED_TEST_CASES_URL = "https://drive.google.com/file/d/1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr/view?usp=sharing"
PIE_PROBLEM_STATEMENTS_URL = "https://github.com/madaan/pie-perf/raw/main/data/problem_statements_translated.zip"


@dataclass(frozen=True)
class Upstream:
    name: str
    repo: str
    pin: str


UPSTREAMS = {
    "rllm": Upstream("rllm", RLLM_REPO, RLLM_PIN),
    "skyrl": Upstream("SkyRL", SKYRL_REPO, SKYRL_PIN),
    "pie-perf": Upstream("pie-perf", PIE_PERF_REPO, PIE_PERF_PIN),
    "learningopt-pie": Upstream("LearningOpt-pie", LEARNINGOPT_PIE_REPO, LEARNINGOPT_PIE_PIN),
}
