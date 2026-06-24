from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from w8_biayn.integrations.skyrl_io_patch import PATCH_MARKER, patch_skyrl_io


SAMPLE_IO = '''import os
import tempfile

from loguru import logger


def download_directory(cloud_path: str, local_path: str) -> None:
    fs.get(cloud_path, local_path, recursive=True)
    logger.info(f"Downloaded {cloud_path} to {local_path}")
'''


def test_patch_skyrl_io_flattens_nested_cloud_downloads(tmp_path: Path) -> None:
    io_path = tmp_path / "io.py"
    io_path.write_text(SAMPLE_IO, encoding="utf-8")

    assert patch_skyrl_io(io_path) == "patched"

    patched = io_path.read_text(encoding="utf-8")
    compile(patched, str(io_path), "exec")
    assert "import shutil" in patched
    assert PATCH_MARKER in patched
    assert "nested_dir = os.path.join(local_path, os.path.basename(cloud_path.rstrip(\"/\")))" in patched
    assert "shutil.move(source, target)" in patched
    assert 'logger.info(f"Downloaded {cloud_path} to {local_path}")' in patched


def test_patch_skyrl_io_is_idempotent(tmp_path: Path) -> None:
    io_path = tmp_path / "io.py"
    io_path.write_text(SAMPLE_IO, encoding="utf-8")

    assert patch_skyrl_io(io_path) == "patched"
    once = io_path.read_text(encoding="utf-8")
    assert patch_skyrl_io(io_path) == "already_patched"
    assert io_path.read_text(encoding="utf-8") == once


def test_patch_skyrl_io_matches_logger_format_tolerantly(tmp_path: Path) -> None:
    io_path = tmp_path / "io.py"
    io_path.write_text(
        """import os

def download_directory(cloud_path, local_path):
        logger.info(  f'Downloaded {cloud_path} to {local_path}'  )
""",
        encoding="utf-8",
    )

    assert patch_skyrl_io(io_path) == "patched"
    patched = io_path.read_text(encoding="utf-8")
    compile(patched, str(io_path), "exec")
    assert "import shutil" in patched
    assert "        # " + PATCH_MARKER in patched


def test_patch_skyrl_io_fails_closed_when_target_moves(tmp_path: Path) -> None:
    io_path = tmp_path / "io.py"
    io_path.write_text("import os\n\ndef download_directory():\n    pass\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="patch target not found"):
        patch_skyrl_io(io_path)


def test_patch_skyrl_io_applies_to_pinned_skyrl_source(tmp_path: Path) -> None:
    upstream = Path(".cache/upstreams/SkyRL/skyrl/backends/skyrl_train/utils/io/io.py")
    if not upstream.exists():
        pytest.skip("pinned SkyRL checkout is not available")

    io_path = tmp_path / "io.py"
    shutil.copyfile(upstream, io_path)

    assert patch_skyrl_io(io_path) == "patched"
    patched = io_path.read_text(encoding="utf-8")
    compile(patched, str(io_path), "exec")
    assert PATCH_MARKER in patched
