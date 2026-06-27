"""SkyRL GRPO entrypoint for C++ performance RL.

This is glue code only: SkyRL owns the trainer. We register the C++ reward
environment inside the Ray task, then delegate to ``BasePPOExp``.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

try:
    import ray
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
    from skyrl.train.utils import initialize_ray
except ImportError:  # pragma: no cover - local tests can import helpers without SkyRL
    ray = None
    SkyRLTrainConfig = None
    BasePPOExp = None
    validate_cfg = None
    initialize_ray = None

from w8_biayn.integrations.cpp_perf_env import register_cpp_perf_env


def _configure_multiprocessing() -> None:
    """Match SkyRL's spawn behavior when imported outside its main module."""

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


if ray is not None:  # pragma: no branch - import-time runtime split

    @ray.remote(num_cpus=1)
    def skyrl_entrypoint(cfg: SkyRLTrainConfig):  # type: ignore[valid-type]
        register_cpp_perf_env()
        exp = BasePPOExp(cfg)
        exp.run()


def main() -> None:
    """Run C++ GRPO through SkyRL/rLLM."""

    if ray is None or SkyRLTrainConfig is None or validate_cfg is None or initialize_ray is None:
        raise ImportError("SkyRL is required to run C++ GRPO training")
    _configure_multiprocessing()
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
