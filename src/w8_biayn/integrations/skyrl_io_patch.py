"""Patch pinned SkyRL cloud directory downloads for checkpoint restore."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_SKYRL_IO_PATH = Path("skyrl/backends/skyrl_train/utils/io/io.py")
PATCH_MARKER = "Flatten cloud directory downloads whose fsspec backend preserves the source basename."
LOGGER_RE = re.compile(
    r"(?m)^(?P<indent>[ \t]*)logger\.info\(\s*f(?P<quote>[\"'])"
    r"Downloaded \{cloud_path\} to \{local_path\}(?P=quote)\s*\)\s*$"
)
IMPORT_OS_RE = re.compile(r"(?m)^import os\s*$")
IMPORT_SHUTIL_RE = re.compile(r"(?m)^import shutil\s*$")


def patch_skyrl_io(path: str | Path = DEFAULT_SKYRL_IO_PATH) -> str:
    skyrl_io_path = Path(path)
    text = skyrl_io_path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return "already_patched"

    text = _ensure_shutil_import(text, skyrl_io_path)
    match = LOGGER_RE.search(text)
    if match is None:
        raise SystemExit(
            "SkyRL download_directory patch target not found in "
            f"{skyrl_io_path}; pinned SkyRL source may have changed"
        )

    indent = match.group("indent")
    patch = "\n".join(
        [
            f"{indent}# {PATCH_MARKER}",
            f'{indent}nested_dir = os.path.join(local_path, os.path.basename(cloud_path.rstrip("/")))',
            f"{indent}if os.path.isdir(nested_dir):",
            f"{indent}    for name in os.listdir(nested_dir):",
            f"{indent}        source = os.path.join(nested_dir, name)",
            f"{indent}        target = os.path.join(local_path, name)",
            f"{indent}        if os.path.exists(target):",
            f"{indent}            if os.path.isdir(target):",
            f"{indent}                shutil.rmtree(target)",
            f"{indent}            else:",
            f"{indent}                os.remove(target)",
            f"{indent}        shutil.move(source, target)",
            f"{indent}    os.rmdir(nested_dir)",
            f'{indent}logger.info(f"Downloaded {{cloud_path}} to {{local_path}}")',
        ]
    )
    text = text[: match.start()] + patch + text[match.end() :]
    skyrl_io_path.write_text(text, encoding="utf-8")
    return "patched"


def _ensure_shutil_import(text: str, skyrl_io_path: Path) -> str:
    if IMPORT_SHUTIL_RE.search(text):
        return text
    match = IMPORT_OS_RE.search(text)
    if match is None:
        raise SystemExit(
            "SkyRL download_directory patch target has no top-level import os in "
            f"{skyrl_io_path}; pinned SkyRL source may have changed"
        )
    return text[: match.end()] + "\nimport shutil" + text[match.end() :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch SkyRL io.py checkpoint download staging.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT_SKYRL_IO_PATH), help="Path to SkyRL io.py")
    args = parser.parse_args()

    result = patch_skyrl_io(args.path)
    print(f"SkyRL checkpoint download staging patch: {result} ({args.path})", flush=True)


if __name__ == "__main__":
    main()
