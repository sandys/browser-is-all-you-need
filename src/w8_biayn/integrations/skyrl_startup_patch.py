"""Patch pinned SkyRL startup logging for model-init visibility."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_SKYRL_FSDP_WORKER_PATH = Path("skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py")
PATCH_MARKER = "Emit W8 startup stage markers for run-status dashboards."
IMPORT_IO_RE = re.compile(r"(?m)^import io\s*$")
IMPORT_JSON_RE = re.compile(r"(?m)^import json\s*$")
IMPORT_TIME_RE = re.compile(r"(?m)^import time\s*$")
FSDP_POLICY_INIT_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)def init_model\(self, model_path, num_training_steps: int = None\):\n"
    r"(?P=indent)    assert self\.cfg\.strategy == \"fsdp\""
)
FSDP_REF_INIT_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)def init_model\(self, model_path\):\n"
    r"(?P=indent)    assert self\.cfg\.strategy == \"fsdp\""
)
POLICY_PREPARE_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)self\.model, self\.optimizer, self\.scheduler = strategy\.prepare\(\n"
    r"(?P=indent)    \(wrapped_model, None, None\),\n"
    r"(?P=indent)\)"
)
REF_PREPARE_RE = re.compile(r"(?m)^(?P<indent>[ \t]*)self\.model = strategy\.prepare\(wrapped_model\)\s*$")


def patch_skyrl_startup(path: str | Path = DEFAULT_SKYRL_FSDP_WORKER_PATH) -> str:
    fsdp_worker_path = Path(path)
    text = fsdp_worker_path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return "already_patched"

    text = _ensure_imports(text, fsdp_worker_path)
    text = _insert_marker_helper(text, fsdp_worker_path)

    policy_match = FSDP_POLICY_INIT_RE.search(text)
    ref_match = FSDP_REF_INIT_RE.search(text)
    policy_prepare_match = POLICY_PREPARE_RE.search(text)
    ref_prepare_match = REF_PREPARE_RE.search(text)
    if not all([policy_match, ref_match, policy_prepare_match, ref_prepare_match]):
        raise SystemExit(
            "SkyRL startup patch target not found in "
            f"{fsdp_worker_path}; pinned SkyRL source may have changed"
        )

    policy_indent = policy_match.group("indent") + "    "
    text = (
        text[: policy_match.end()]
        + "\n"
        + f'{policy_indent}_w8_startup_marker("policy_model_init", "start", model_path)'
        + text[policy_match.end() :]
    )

    ref_match = FSDP_REF_INIT_RE.search(text)
    if ref_match is None:
        raise SystemExit(
            "SkyRL startup patch ref init target not found in "
            f"{fsdp_worker_path}; pinned SkyRL source may have changed"
        )
    ref_indent = ref_match.group("indent") + "    "
    text = (
        text[: ref_match.end()]
        + "\n"
        + f'{ref_indent}_w8_startup_marker("ref_model_init", "start", model_path)'
        + text[ref_match.end() :]
    )

    policy_prepare_match = POLICY_PREPARE_RE.search(text)
    if policy_prepare_match is None:
        raise SystemExit(
            "SkyRL startup patch policy prepare target not found in "
            f"{fsdp_worker_path}; pinned SkyRL source may have changed"
        )
    policy_indent = policy_prepare_match.group("indent")
    text = (
        text[: policy_prepare_match.end()]
        + "\n"
        + f'{policy_indent}_w8_startup_marker("policy_model_init", "end", model_path)'
        + text[policy_prepare_match.end() :]
    )

    ref_prepare_match = REF_PREPARE_RE.search(text)
    if ref_prepare_match is None:
        raise SystemExit(
            "SkyRL startup patch ref prepare target not found in "
            f"{fsdp_worker_path}; pinned SkyRL source may have changed"
        )
    ref_indent = ref_prepare_match.group("indent")
    text = (
        text[: ref_prepare_match.end()]
        + "\n"
        + f'{ref_indent}_w8_startup_marker("ref_model_init", "end", model_path)'
        + text[ref_prepare_match.end() :]
    )

    fsdp_worker_path.write_text(text, encoding="utf-8")
    return "patched"


def _ensure_imports(text: str, fsdp_worker_path: Path) -> str:
    if IMPORT_JSON_RE.search(text) and IMPORT_TIME_RE.search(text):
        return text
    match = IMPORT_IO_RE.search(text)
    if match is None:
        raise SystemExit(
            "SkyRL startup patch target has no top-level import io in "
            f"{fsdp_worker_path}; pinned SkyRL source may have changed"
        )
    additions = []
    if not IMPORT_JSON_RE.search(text):
        additions.append("import json")
    if not IMPORT_TIME_RE.search(text):
        additions.append("import time")
    return text[: match.end()] + "\n" + "\n".join(additions) + text[match.end() :]


def _insert_marker_helper(text: str, fsdp_worker_path: Path) -> str:
    marker = f"# {PATCH_MARKER}"
    if marker in text:
        return text
    type_check_match = re.search(r"(?m)^if TYPE_CHECKING:\n", text)
    if type_check_match is None:
        raise SystemExit(
            "SkyRL startup patch helper target not found in "
            f"{fsdp_worker_path}; pinned SkyRL source may have changed"
        )
    helper = f'''

def _w8_startup_marker(stage, event, model_path=None):
    # {PATCH_MARKER}
    payload = {{
        "schema_version": "w8-setup-stage-v1",
        "stage": stage,
        "event": event,
        "ts_unix": time.time(),
        "model_path": str(model_path) if model_path is not None else None,
    }}
    try:
        payload["rank"] = torch.distributed.get_rank() if torch.distributed.is_initialized() else None
        payload["world_size"] = torch.distributed.get_world_size() if torch.distributed.is_initialized() else None
    except Exception:
        payload["rank"] = None
        payload["world_size"] = None
    print("W8_SETUP_STAGE " + json.dumps(payload, sort_keys=True), flush=True)

'''
    return text[: type_check_match.start()] + helper + text[type_check_match.start() :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch SkyRL FSDP worker startup logging.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_SKYRL_FSDP_WORKER_PATH), help="Path to fsdp_worker.py")
    args = parser.parse_args()

    result = patch_skyrl_startup(args.path)
    print(f"SkyRL startup logging patch: {result} ({args.path})", flush=True)


if __name__ == "__main__":
    main()
