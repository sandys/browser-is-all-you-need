from __future__ import annotations

import sys
import types


def test_register_harbor_env_registers_current_process(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    envs_module = types.ModuleType("skyrl_gym.envs")

    def register(**kwargs: object) -> None:
        calls.append(kwargs)

    envs_module.register = register  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skyrl_gym", types.ModuleType("skyrl_gym"))
    monkeypatch.setitem(sys.modules, "skyrl_gym.envs", envs_module)

    from w8_biayn.integrations.harbor_env import HARBOR_SKYRL_ENV_ID, register_harbor_env

    register_harbor_env()

    assert calls == [
        {
            "id": HARBOR_SKYRL_ENV_ID,
            "entry_point": "w8_biayn.integrations.harbor_env:HarborSkyRLEnv",
        }
    ]


def test_skyrl_entrypoint_registers_env_inside_ray_worker(monkeypatch) -> None:
    events: list[str] = []
    main_base_module = types.ModuleType("skyrl.train.entrypoints.main_base")

    class FakeBasePPOExp:
        def __init__(self, cfg: object) -> None:
            events.append(f"exp:{cfg}")

        def run(self) -> None:
            events.append("run")

    main_base_module.BasePPOExp = FakeBasePPOExp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "skyrl", types.ModuleType("skyrl"))
    monkeypatch.setitem(sys.modules, "skyrl.train", types.ModuleType("skyrl.train"))
    monkeypatch.setitem(sys.modules, "skyrl.train.entrypoints", types.ModuleType("skyrl.train.entrypoints"))
    monkeypatch.setitem(sys.modules, "skyrl.train.entrypoints.main_base", main_base_module)

    import w8_biayn.integrations.harbor_env as harbor_env
    from w8_biayn.integrations.skyrl_harbor_main import _skyrl_entrypoint

    monkeypatch.setattr(harbor_env, "register_harbor_env", lambda: events.append("register"))

    _skyrl_entrypoint("cfg")

    assert events == ["register", "exp:cfg", "run"]
