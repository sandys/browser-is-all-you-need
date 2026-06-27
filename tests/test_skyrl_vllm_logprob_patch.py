from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from w8_biayn.integrations.skyrl_vllm_logprob_patch import PATCH_MARKER, patch_skyrl_vllm_logprobs


SAMPLE_VLLM_ENGINE = """from typing import Dict, List, Optional

from loguru import logger


class Engine:
    def _postprocess_outputs(self, outputs):
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []
        for output in outputs:
            resp = output.outputs[0]
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)
        return response_ids, response_logprobs
"""


def test_patch_skyrl_vllm_logprobs_aligns_response_ids_with_logprobs(tmp_path: Path) -> None:
    engine_path = tmp_path / "vllm_engine.py"
    engine_path.write_text(SAMPLE_VLLM_ENGINE, encoding="utf-8")

    assert patch_skyrl_vllm_logprobs(engine_path) == "patched"

    patched = engine_path.read_text(encoding="utf-8")
    compile(patched, str(engine_path), "exec")
    assert PATCH_MARKER in patched
    assert "response_token_ids = list(resp.token_ids)" in patched
    assert "min(len(response_token_ids), len(_logprobs))" in patched
    assert "response_ids.append(response_token_ids)" in patched
    assert "response_ids.append(resp.token_ids)" not in patched


def test_patch_skyrl_vllm_logprobs_is_idempotent(tmp_path: Path) -> None:
    engine_path = tmp_path / "vllm_engine.py"
    engine_path.write_text(SAMPLE_VLLM_ENGINE, encoding="utf-8")

    assert patch_skyrl_vllm_logprobs(engine_path) == "patched"
    once = engine_path.read_text(encoding="utf-8")
    assert patch_skyrl_vllm_logprobs(engine_path) == "already_patched"
    assert engine_path.read_text(encoding="utf-8") == once


def test_patch_skyrl_vllm_logprobs_fails_closed_when_target_moves(tmp_path: Path) -> None:
    engine_path = tmp_path / "vllm_engine.py"
    engine_path.write_text("def _postprocess_outputs(outputs):\n    return outputs\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="patch target not found"):
        patch_skyrl_vllm_logprobs(engine_path)


def test_patch_skyrl_vllm_logprobs_applies_to_pinned_skyrl_source(tmp_path: Path) -> None:
    upstream = Path(".cache/upstreams/SkyRL/skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py")
    if not upstream.exists():
        pytest.skip("pinned SkyRL checkout is not available")

    engine_path = tmp_path / "vllm_engine.py"
    shutil.copyfile(upstream, engine_path)

    assert patch_skyrl_vllm_logprobs(engine_path) == "patched"
    patched = engine_path.read_text(encoding="utf-8")
    compile(patched, str(engine_path), "exec")
    assert PATCH_MARKER in patched
