from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_scalecua_real_subset.py"
spec = importlib.util.spec_from_file_location("scalecua_real_subset", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
subset = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = subset
spec.loader.exec_module(subset)


def test_archive_base_from_root_handles_images_suffix():
    assert subset.archive_base_from_root("data/data_20250310/windows/images") == "data/data_20250310/windows"
    assert subset.archive_base_from_root("data/data_20250310/ubuntu/") == "data/data_20250310/ubuntu"


def test_extract_member_path_is_relative_to_archive_parent():
    assert subset.extract_member_path(
        "data/data_20250310/windows/images",
        "slack/slack_1/images/step_3.png",
    ) == "windows/images/slack/slack_1/images/step_3.png"
    assert subset.extract_member_path(
        "data/data_20250310/ubuntu/",
        "files/files_1/images/step_2.png",
    ) == "ubuntu/files/files_1/images/step_2.png"


def test_output_image_path_keeps_scalecua_data_prefix():
    assert subset.output_image_path(
        Path(".w8-biayn/scalecua/images"),
        "data/data_20250310/windows/images",
        "slack/x.png",
    ) == Path(".w8-biayn/scalecua/images/data/data_20250310/windows/images/slack/x.png")


def test_choose_candidates_prefers_archive_density_and_action_quota():
    archive_small = subset.ArchiveInfo("data/data_a/web", ["a"], 1024)
    archive_big = subset.ArchiveInfo("data/data_b/windows", ["b"], 1024 * 1024)
    dense = subset.SourceBucket(Path("a.jsonl"), "a", "web", "data/data_a/web/images", archive_small)
    sparse = subset.SourceBucket(Path("b.jsonl"), "b", "windows", "data/data_b/windows/images", archive_big)
    dense.candidates = [
        subset.Candidate("a", i, archive_small.base, f"m{i}", f"web/m{i}", f"out{i}", "left_click", {"i": i})
        for i in range(3)
    ]
    sparse.candidates = [
        subset.Candidate("b", i, archive_big.base, f"n{i}", f"windows/n{i}", f"other{i}", "key", {"i": i})
        for i in range(3)
    ]

    selected = subset.choose_candidates([sparse, dense], 4, ("left_click", "key"))

    assert [item.archive_base for item in selected[:2]] == [archive_small.base, archive_small.base]
    assert len(selected) == 4


def test_cleanup_materialized_archive_removes_concat_tar_and_optionally_parts(tmp_path):
    archive = subset.ArchiveInfo(
        "data/data_a/web",
        ["data/data_a/web.tar.gz.part-000", "data/data_a/web.tar.gz.part-001"],
        6,
    )
    concat = subset.materialized_archive_path(tmp_path, archive)
    concat.parent.mkdir(parents=True)
    concat.write_bytes(b"concat")
    for part in archive.parts:
        part_path = tmp_path / part
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_bytes(b"part")

    subset.cleanup_materialized_archive(tmp_path, archive, keep_concat_tar=False, cleanup_parts=True)

    assert not concat.exists()
    assert all(not (tmp_path / part).exists() for part in archive.parts)


def test_cleanup_materialized_archive_can_keep_downloaded_files(tmp_path):
    archive = subset.ArchiveInfo("data/data_a/web", ["data/data_a/web.tar.gz.part-000"], 4)
    concat = subset.materialized_archive_path(tmp_path, archive)
    concat.parent.mkdir(parents=True)
    concat.write_bytes(b"keep")
    part_path = tmp_path / archive.parts[0]
    part_path.write_bytes(b"part")

    subset.cleanup_materialized_archive(tmp_path, archive, keep_concat_tar=True, cleanup_parts=False)

    assert concat.exists()
    assert part_path.exists()
