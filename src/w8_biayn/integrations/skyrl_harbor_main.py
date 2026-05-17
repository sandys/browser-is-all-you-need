"""Register Harbor DOMDiff in SkyRL's Ray entrypoint and run the trainer."""

from __future__ import annotations

from typing import Any


def _skyrl_entrypoint(cfg: Any) -> None:
    from skyrl.train.entrypoints.main_base import BasePPOExp

    from w8_biayn.integrations.harbor_env import register_harbor_env

    register_harbor_env()
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
