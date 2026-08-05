from __future__ import annotations

import ast
import json
import re
import runpy
from pathlib import Path


EVAL_APP = Path("examples/modal/aider_eval_app.py")
OVERLAY = Path("examples/modal/aider_fixed26_contract_overlay.py")
ORIGINALS = Path("examples/modal/aider_fixed26_originals.json")


def test_contract_overlay_covers_exactly_the_pinned_fixed26() -> None:
    module = runpy.run_path(str(OVERLAY))
    contracts = module["CONTRACTS"]
    marker = module["MARKER"]
    originals = json.loads(ORIGINALS.read_text(encoding="utf-8"))
    hashes = originals["instructions_sha256"]
    tests = originals["tests"]

    assert module["OVERLAY_VERSION"] == "fixed26-contract-v2"
    assert originals["polyglot_commit"] == "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
    assert len(contracts) == len(hashes) == len(tests) == 26
    assert set(contracts) == set(hashes) == set(tests)
    assert all(record["file"].endswith("_test.cpp") for record in tests.values())
    assert all(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) for record in tests.values())
    assert all(marker in contract for contract in contracts.values())
    assert all("## Build environment" in contract for contract in contracts.values())


def test_contract_regressions_are_explicit_and_audited() -> None:
    module = runpy.run_path(str(OVERLAY))
    contracts = module["CONTRACTS"]
    requirements = module["_REGRESSION_REQUIREMENTS"]

    assert "newly opened account has a zero balance" in contracts["bank-account"]
    assert "previous balance is not retained" in contracts["bank-account"]
    assert len(requirements["all-your-base"]["test"]) == 8
    assert "std::invalid_argument" in contracts["all-your-base"]
    assert "empty vector" in contracts["all-your-base"]
    for task in ("gigasecond", "meetup"):
        assert "Boost (1.58 or newer) is installed" in contracts[task]


def test_eval_receipts_bind_the_prompt_test_audit() -> None:
    text = EVAL_APP.read_text(encoding="utf-8")
    assert 'audit_manifest = overlay.audit(destination)' in text
    assert '"prompt_test_audit_sha256": overlay_manifest["prompt_test_audit_sha256"]' in text
    assert '"prompt_test_audit": overlay_manifest["prompt_test_audit"]' in text
    assert '"contract_overlay_shard_sha256": {' in text
    assert '"prompt_test_audit_shard_sha256": {' in text
    assert '"thinking_disabled": DISABLE_THINKING' in text
    assert '"lora_activation_verified": (' in text
    assert "parallel evaluation shard identity mismatch" in text


def test_eval_app_has_no_duplicate_literal_receipt_keys() -> None:
    tree = ast.parse(EVAL_APP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        assert len(keys) == len(set(keys))


def test_pass_at_k_prepares_once_before_launching_any_gpu_shard() -> None:
    text = EVAL_APP.read_text(encoding="utf-8")
    start = text.index("def _run_pass_at_k(")
    end = text.index("\ndef ", text.index("\n", start))
    body = text[start:end]

    assert body.count("preparation = prepare_adapter.remote(") == 1
    assert body.index("preparation = prepare_adapter.remote(") < body.index("for index in range(")
    assert "if tries not in (1, 2):" in body
    assert '"serving_preflight": preparation' in body


def test_base_pass_at_k_is_explicitly_adapter_free() -> None:
    text = EVAL_APP.read_text(encoding="utf-8")
    start = text.index("def base_pass_at_k(")
    end = text.index("\n@app.function(", start)
    body = text[start:end]

    assert 'adapter_path=""' in body
    assert 'expected_adapter_sha256=""' in body
    assert 'expected_data_manifest_sha256=""' in body
    assert "base_model=True" in body
    assert '"adapter_loaded": False' in body


def test_base_shard_skips_lora_startup_and_records_base_identity() -> None:
    text = EVAL_APP.read_text(encoding="utf-8")
    start = text.index("def evaluate_shard(")
    end = text.index("\ndef _merge_shard_receipts(", start)
    body = text[start:end]

    assert "if base_model:" in body
    assert (
        'raise ValueError("base-model evaluation must not receive adapter binding arguments")'
        in body
    )
    guard = body.index("if not base_model:")
    assert "--enable-lora" not in body[:guard]
    assert "--enable-lora" in body[guard:]
    assert 'adapter_load = {"mode": "base", "loaded_adapters": []}' in body
    assert '"model_kind": "base" if base_model else "adapter"' in body


def test_adapter_requests_select_the_lora_and_base_requests_do_not() -> None:
    """SGLang applies a loaded LoRA only when the request names it (colon syntax or an
    explicit lora_path body field); a plain model name silently serves base weights."""
    text = EVAL_APP.read_text(encoding="utf-8")
    assert 'LORA_NAME = "glm-4.7-flash-grpo"' in text
    assert "lora_path: {lora_name}" in text
    assert "_model_settings_yaml(lora_name=None if base_model else LORA_NAME)" in text
    assert "_model_settings_yaml(lora_name=LORA_NAME)" in text


def test_adapter_shard_verifies_lora_activation_before_benchmarking() -> None:
    text = EVAL_APP.read_text(encoding="utf-8")
    start = text.index("def evaluate_shard(")
    end = text.index("\n@app.function(", start)
    body = text[start:end]
    assert body.count("_verify_lora_activation()") == 1
    assert '"lora_activation_probe": activation_probe' in body
    load_pos = body.index("_load_adapter(serving_path)")
    probe_pos = body.index("_verify_lora_activation()")
    bench_pos = body.index("full = _benchmark(")
    assert load_pos < probe_pos < bench_pos
