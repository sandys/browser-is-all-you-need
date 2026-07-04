"""Generic helpers for the pinned SLIME upstream integration."""

from .doctor import run_slime_doctor, slime_root
from .lora import LoraArgPlan, available_flags, build_lora_args
from .sandbox import DockerSandbox, SandboxError, create_sandbox, sandbox_backend_from_env
from .setup import (
    DEFAULT_SLIME_IMAGE,
    SlimeSetupPlan,
    build_slime_setup_plan,
    write_slime_setup_files,
)

__all__ = [
    "DEFAULT_SLIME_IMAGE",
    "DockerSandbox",
    "LoraArgPlan",
    "SandboxError",
    "SlimeSetupPlan",
    "available_flags",
    "build_slime_setup_plan",
    "build_lora_args",
    "create_sandbox",
    "run_slime_doctor",
    "sandbox_backend_from_env",
    "slime_root",
    "write_slime_setup_files",
]
