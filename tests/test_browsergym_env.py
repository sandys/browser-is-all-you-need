from __future__ import annotations

import sys
import types

import pytest

from w8_biayn.integrations.browsergym_env import BrowserGymEnv, extract_action, serialize_observation


class FakeGymEnv:
    def __init__(self):
        self.closed = False

    def reset(self):
        return {"goal": "click the button"}, {"reset": True}

    def step(self, action):
        assert action == "click('button')"
        return {"state": "done"}, 1.0, True, False, {"ok": True}

    def close(self):
        self.closed = True


def install_fake_gym(monkeypatch):
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.make = lambda _env_id, **_kwargs: FakeGymEnv()
    browsergym = types.ModuleType("browsergym")
    miniwob = types.ModuleType("browsergym.miniwob")
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "browsergym", browsergym)
    monkeypatch.setitem(sys.modules, "browsergym.miniwob", miniwob)


def test_extract_action_uses_last_action():
    text = "<action>noop()</action>\n<action>click('button')</action>"
    assert extract_action(text) == "click('button')"


def test_serialize_observation_truncates():
    assert serialize_observation("abcdef", max_chars=3) == "abc\n...[truncated]"


def test_browsergym_env_init_step_close(monkeypatch):
    install_fake_gym(monkeypatch)
    env = BrowserGymEnv(extras={"env_id": "browsergym/miniwob.click-button", "max_turns": 3})

    prompt, info = env.init([{"role": "system", "content": "sys"}])
    assert info == {"reset": True}
    assert prompt[-1]["role"] == "user"
    assert "click the button" in prompt[-1]["content"]

    output = env.step("<action>click('button')</action>")
    assert output["reward"] == 1.0
    assert output["done"] is True
    assert output["observations"] == []
    assert env.get_metrics()["browsergym_turns"] == 1
    env.close()


def test_browsergym_env_bad_action_returns_observation(monkeypatch):
    install_fake_gym(monkeypatch)
    env = BrowserGymEnv(extras={"env_id": "browsergym/miniwob.click-button", "max_turns": 2})
    env.init([])

    output = env.step("not tagged")

    assert output["reward"] == -1.0
    assert output["done"] is False
    assert "Action error" in output["observations"][0]["content"]

