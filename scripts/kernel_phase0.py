"""Phase-0 discovery probe for the custom-kernel lab (runs on a GPU VM).

Introspection only — prints what the pinned Megatron-Bridge / Megatron-Core actually
expose so we can write the ModuleSpec injection (and later the Triton kernels) against
real APIs instead of guessing. Defensive: every step is wrapped so the probe always
reaches DONE and reports what it found.
"""

from __future__ import annotations

import os
import traceback


def show(title: str) -> None:
    print(f"\n===== {title} =====", flush=True)


show("versions")
try:
    import torch

    print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))
except Exception:
    traceback.print_exc()

for mod in ("transformer_engine", "megatron.core", "megatron.bridge"):
    try:
        m = __import__(mod, fromlist=["__version__"])
        print(f"{mod} OK", getattr(m, "__version__", "(no __version__)"))
    except Exception as exc:  # noqa: BLE001
        print(f"{mod} IMPORT FAILED: {exc!r}")

show("locate AutoBridge")
AutoBridge = None
for path in ("megatron.bridge", "megatron.bridge.models", "megatron.bridge.api"):
    try:
        module = __import__(path, fromlist=["AutoBridge"])
        if hasattr(module, "AutoBridge"):
            AutoBridge = module.AutoBridge
            print("AutoBridge from", path)
            break
    except Exception as exc:  # noqa: BLE001
        print(path, "->", repr(exc))

MODEL = os.environ.get("W8_TINY_MODEL", "eatang/qwen3-moe-tiny-random")
show(f"build provider for {MODEL}")
if AutoBridge is None:
    print("AutoBridge unavailable; skipping provider build")
else:
    try:
        bridge = AutoBridge.from_hf_pretrained(MODEL, trust_remote_code=True)
        provider = bridge.to_megatron_provider()
        cls = type(provider)
        print("provider type:", f"{cls.__module__}.{cls.__name__}")
        attrs = [a for a in dir(provider) if not a.startswith("_")]
        spec_attrs = [a for a in attrs if "spec" in a.lower() or a.lower().endswith("layer")]
        print("spec/layer attrs:", spec_attrs)
        for a in spec_attrs:
            try:
                print("   ", a, "=", repr(getattr(provider, a)))
            except Exception as exc:  # noqa: BLE001
                print("   ", a, "getattr ->", repr(exc))
        moe_attrs = [
            a
            for a in attrs
            if any(k in a.lower() for k in ("moe", "expert", "mla", "latent", "attention", "q_lora", "kv_lora"))
        ]
        print("moe/mla/attn attrs:", moe_attrs)
        for a in moe_attrs:
            try:
                print("   ", a, "=", repr(getattr(provider, a)))
            except Exception as exc:  # noqa: BLE001
                print("   ", a, "getattr ->", repr(exc))
    except Exception:
        print("provider build FAILED:")
        traceback.print_exc()

show("megatron-core layer spec builders")
for path, names in (
    ("megatron.core.models.gpt.gpt_layer_specs", ("get_gpt_layer_with_transformer_engine_spec", "get_gpt_layer_local_spec")),
    ("megatron.core.transformer.spec_utils", ("ModuleSpec",)),
):
    try:
        module = __import__(path, fromlist=list(names))
        for name in names:
            print(f"{path}.{name}:", "OK" if hasattr(module, name) else "MISSING")
    except Exception as exc:  # noqa: BLE001
        print(path, "->", repr(exc))

show("DONE")
