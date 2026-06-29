from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from w8_biayn.integrations.skyrl_startup_patch import PATCH_MARKER, patch_skyrl_startup


SAMPLE_FSDP_WORKER = '''import io
import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


class FSDPPolicyWorkerBase:
    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy == "fsdp"
        strategy = object()
        wrapped_model = object()
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )


class FSDPRefWorkerBase:
    def init_model(self, model_path):
        assert self.cfg.strategy == "fsdp"
        strategy = object()
        wrapped_model = object()
        self.model = strategy.prepare(wrapped_model)
'''


def test_patch_skyrl_startup_emits_policy_and_ref_markers(tmp_path: Path) -> None:
    worker_path = tmp_path / "fsdp_worker.py"
    worker_path.write_text(SAMPLE_FSDP_WORKER, encoding="utf-8")

    assert patch_skyrl_startup(worker_path) == "patched"

    patched = worker_path.read_text(encoding="utf-8")
    compile(patched, str(worker_path), "exec")
    assert PATCH_MARKER in patched
    assert "W8_SETUP_STAGE " in patched
    assert '_w8_startup_marker("policy_model_init", "start", model_path)' in patched
    assert '_w8_startup_marker("policy_model_init", "end", model_path)' in patched
    assert '_w8_startup_marker("ref_model_init", "start", model_path)' in patched
    assert '_w8_startup_marker("ref_model_init", "end", model_path)' in patched


def test_patch_skyrl_startup_is_idempotent(tmp_path: Path) -> None:
    worker_path = tmp_path / "fsdp_worker.py"
    worker_path.write_text(SAMPLE_FSDP_WORKER, encoding="utf-8")

    assert patch_skyrl_startup(worker_path) == "patched"
    once = worker_path.read_text(encoding="utf-8")
    assert patch_skyrl_startup(worker_path) == "already_patched"
    assert worker_path.read_text(encoding="utf-8") == once


def test_patch_skyrl_startup_fails_closed_when_target_moves(tmp_path: Path) -> None:
    worker_path = tmp_path / "fsdp_worker.py"
    worker_path.write_text("import io\n\nclass Worker:\n    pass\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="patch .*target"):
        patch_skyrl_startup(worker_path)


def test_patch_skyrl_startup_applies_to_pinned_skyrl_source(tmp_path: Path) -> None:
    upstream = Path(".cache/upstreams/SkyRL/skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py")
    if not upstream.exists():
        pytest.skip("pinned SkyRL checkout is not available")

    worker_path = tmp_path / "fsdp_worker.py"
    shutil.copyfile(upstream, worker_path)

    assert patch_skyrl_startup(worker_path) == "patched"
    patched = worker_path.read_text(encoding="utf-8")
    compile(patched, str(worker_path), "exec")
    assert PATCH_MARKER in patched
