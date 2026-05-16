"""Register BrowserGym with SkyRL-Gym and delegate to SkyRL's trainer."""

from __future__ import annotations


def main() -> None:
    from skyrl_gym.envs import register

    try:
        register(
            id="browsergym",
            entry_point="w8_biayn.integrations.browsergym_env:BrowserGymEnv",
        )
    except Exception as exc:
        if "already registered" not in str(exc):
            raise

    from skyrl.train.entrypoints.main_base import main as skyrl_main

    skyrl_main()


if __name__ == "__main__":
    main()

