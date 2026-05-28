"""Register Harbor DOMDiff in SkyRL's Ray entrypoint and run the trainer."""

from __future__ import annotations

from typing import Any


def _skyrl_entrypoint(cfg: Any) -> None:
    from skyrl.train.entrypoints.main_base import BasePPOExp

    from w8_biayn.integrations.harbor_env import register_harbor_env
    from w8_biayn.kernels.patch import apply_kernel_patches_from_env

    register_harbor_env()
    # Gated custom-kernel activation. No-op unless W8_BIAYN_KERNELS is set. Applied here,
    # inside the Ray entrypoint, so patches reach the trainer worker process (not just the
    # local driver) — mirroring how register_harbor_env must run here.
    apply_kernel_patches_from_env()
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    import sys

    import ray
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.utils import initialize_ray, validate_cfg

    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    initialize_ray(cfg)
    skyrl_entrypoint = ray.remote(num_cpus=1)(_skyrl_entrypoint)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
