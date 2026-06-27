"""Patch pinned SkyRL vLLM output postprocessing to keep ids/logprobs aligned."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_SKYRL_VLLM_ENGINE_PATH = Path("skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py")
PATCH_MARKER = "Align vLLM response token IDs with returned rollout logprobs."
RESPONSE_IDS_APPEND_RE = re.compile(r"(?m)^(?P<indent>[ \t]*)response_ids\.append\(\s*resp\.token_ids\s*\)\s*$")
LOGPROBS_APPEND_RE = re.compile(r"(?m)^(?P<indent>[ \t]*)response_logprobs\.append\(\s*_logprobs\s*\)\s*$")


def patch_skyrl_vllm_logprobs(path: str | Path = DEFAULT_SKYRL_VLLM_ENGINE_PATH) -> str:
    engine_path = Path(path)
    text = engine_path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return "already_patched"

    response_match = RESPONSE_IDS_APPEND_RE.search(text)
    logprobs_match = LOGPROBS_APPEND_RE.search(text)
    if response_match is None or logprobs_match is None or response_match.start() > logprobs_match.start():
        raise SystemExit(
            "SkyRL vLLM logprob alignment patch target not found in "
            f"{engine_path}; pinned SkyRL source may have changed"
        )

    response_indent = response_match.group("indent")
    response_patch = "\n".join(
        [
            f"{response_indent}# {PATCH_MARKER}",
            f"{response_indent}response_token_ids = list(resp.token_ids)",
        ]
    )

    logprobs_indent = logprobs_match.group("indent")
    logprobs_patch = "\n".join(
        [
            f"{logprobs_indent}if _logprobs is not None and len(response_token_ids) != len(_logprobs):",
            f'{logprobs_indent}    logger.warning("Aligning mismatched vLLM response token/logprob lengths: ids={{}}, logprobs={{}}", len(response_token_ids), len(_logprobs))',
            f"{logprobs_indent}    _paired_len = min(len(response_token_ids), len(_logprobs))",
            f"{logprobs_indent}    response_token_ids = response_token_ids[:_paired_len]",
            f"{logprobs_indent}    _logprobs = _logprobs[:_paired_len]",
            f"{logprobs_indent}response_ids.append(response_token_ids)",
            f"{logprobs_indent}response_logprobs.append(_logprobs)",
        ]
    )

    text = text[: response_match.start()] + response_patch + text[response_match.end() :]
    shift = len(response_patch) - (response_match.end() - response_match.start())
    shifted_logprobs_start = logprobs_match.start() + shift
    shifted_logprobs_end = logprobs_match.end() + shift
    text = text[:shifted_logprobs_start] + logprobs_patch + text[shifted_logprobs_end:]
    engine_path.write_text(text, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch SkyRL vLLM token/logprob alignment.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_SKYRL_VLLM_ENGINE_PATH), help="Path to SkyRL vLLM engine")
    args = parser.parse_args()

    result = patch_skyrl_vllm_logprobs(args.path)
    print(f"SkyRL vLLM logprob alignment patch: {result} ({args.path})", flush=True)


if __name__ == "__main__":
    main()
