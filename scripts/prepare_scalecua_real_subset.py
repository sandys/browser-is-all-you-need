#!/usr/bin/env python3
"""Prepare a real-image ScaleCUA subset for OSWorld-oriented Qwen LoRA SFT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
CONVERTER_PATH = SCRIPT_DIR / "convert_scalecua_to_osworld_toolcalls.py"
spec = importlib.util.spec_from_file_location("scalecua_converter", CONVERTER_PATH)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

ARCHIVE_PART_RE = re.compile(r"^(data/data_[^/]+/[^/]+)\.tar\.gz\.part-\d+$")
DEFAULT_ACTIONS = (
    "left_click",
    "right_click",
    "double_click",
    "type",
    "key",
    "scroll",
    "left_click_drag",
    "wait",
    "terminate",
    "mouse_move",
)


@dataclass
class ArchiveInfo:
    base: str
    parts: list[str]
    size: int


@dataclass
class Candidate:
    source: str
    line_no: int
    archive_base: str
    member_path: str
    extract_member: str
    output_image: str
    action: str
    row: dict[str, Any]


@dataclass
class SourceBucket:
    annotation: Path
    meta_key: str
    platform: str
    root: str
    archive: ArchiveInfo
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def density(self) -> float:
        gb = max(self.archive.size / (1024**3), 1e-9)
        return len(self.candidates) / gb


def iter_annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def normalized_root(root: str) -> str:
    return root.strip().strip("/")


def archive_base_from_root(root: str) -> str:
    parts = Path(normalized_root(root)).parts
    if len(parts) < 3 or parts[0] != "data":
        raise ValueError(f"unsupported ScaleCUA root: {root}")
    if parts[-1] == "images":
        parts = parts[:-1]
    return str(Path(*parts[:3]))


def platform_from_meta(meta: dict[str, Any], annotation: Path) -> str:
    task = str(meta.get("task") or "").lower()
    name = f"{annotation.name} {task}".lower()
    for platform in ("windows", "ubuntu", "web", "mac", "android", "iphone"):
        if platform in name:
            return platform
    return "unknown"


def ensure_meta(meta_path: Path, repo_id: str) -> Path:
    if meta_path.exists():
        return meta_path
    from huggingface_hub import hf_hub_download

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(repo_id, "meta.json", repo_type="dataset"))
    meta_path.write_text(downloaded.read_text(encoding="utf-8"), encoding="utf-8")
    return meta_path


def load_meta_by_annotation(meta_path: Path, repo_id: str) -> dict[str, tuple[str, dict[str, Any]]]:
    meta_path = ensure_meta(meta_path, repo_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    by_annotation: dict[str, tuple[str, dict[str, Any]]] = {}
    for key, value in meta.items():
        by_annotation[Path(value["annotation"]).name] = (key, value)
    return by_annotation


def repo_archives(repo_id: str) -> dict[str, ArchiveInfo]:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id, files_metadata=True)
    parts_by_base: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for sibling in info.siblings:
        path = sibling.rfilename
        match = ARCHIVE_PART_RE.match(path)
        if match:
            parts_by_base[match.group(1)].append((path, int(sibling.size or 0)))
    archives: dict[str, ArchiveInfo] = {}
    for base, parts in parts_by_base.items():
        ordered = sorted(parts)
        archives[base] = ArchiveInfo(
            base=base,
            parts=[path for path, _ in ordered],
            size=sum(size for _, size in ordered),
        )
    return archives


def output_image_path(images_dir: Path, root: str, image: str) -> Path:
    return images_dir / normalized_root(root) / image.lstrip("/")


def extract_member_path(root: str, image: str) -> str:
    full = Path(normalized_root(root)) / image.lstrip("/")
    parts = full.parts
    if len(parts) < 4 or parts[0] != "data":
        raise ValueError(f"unsupported image member path: {full}")
    return str(Path(*parts[2:]))


def scan_sources(args: argparse.Namespace, archives: dict[str, ArchiveInfo]) -> tuple[list[SourceBucket], list[dict[str, Any]]]:
    meta_by_annotation = load_meta_by_annotation(args.meta, args.repo_id)
    platforms = {item.lower() for item in args.platforms}
    buckets: list[SourceBucket] = []
    rejects: list[dict[str, Any]] = []
    for annotation in iter_annotation_files(args.annotations):
        meta_item = meta_by_annotation.get(annotation.name)
        if not meta_item:
            continue
        meta_key, meta = meta_item
        platform = platform_from_meta(meta, annotation)
        if platforms and platform not in platforms:
            continue
        root = normalized_root(meta["root"])
        archive_base = archive_base_from_root(root)
        archive = archives.get(archive_base)
        if not archive:
            rejects.append({"source": annotation.name, "reason": f"missing archive for {archive_base}"})
            continue
        bucket = SourceBucket(annotation=annotation, meta_key=meta_key, platform=platform, root=root, archive=archive)
        with annotation.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if args.max_rows_per_source and len(bucket.candidates) >= args.max_rows_per_source:
                    break
                try:
                    record = json.loads(line)
                    actions = converter.ACTION_RE.findall(converter.assistant_text(record))
                    if len(actions) != 1:
                        continue
                    name, parsed_args, kwargs = converter.parse_action_call(actions[0].strip())
                    arguments = converter.convert_action(name, parsed_args, kwargs, int(record["width"]), int(record["height"]))
                    row = converter.output_row(record, annotation, arguments)
                    image = str(record.get("image") or "")
                    row["image"] = str(output_image_path(args.images_dir, root, image))
                    row["scale_cua_root"] = root
                    row["scale_cua_archive"] = archive.base
                    candidate = Candidate(
                        source=annotation.name,
                        line_no=line_no,
                        archive_base=archive.base,
                        member_path=str(Path(root) / image),
                        extract_member=extract_member_path(root, image),
                        output_image=row["image"],
                        action=str(arguments.get("action", "unknown")),
                        row=row,
                    )
                    bucket.candidates.append(candidate)
                except Exception as exc:
                    rejects.append({"source": annotation.name, "line": line_no, "reason": str(exc)})
        if bucket.candidates:
            buckets.append(bucket)
    return buckets, rejects


def choose_candidates(buckets: list[SourceBucket], limit: int, actions: tuple[str, ...]) -> list[Candidate]:
    selected: list[Candidate] = []
    seen_images: set[str] = set()
    action_counts: Counter[str] = Counter()
    quota = max(1, limit // max(len(actions), 1))
    action_set = set(actions)
    ordered_buckets = sorted(buckets, key=lambda item: (item.density, len(item.candidates)), reverse=True)

    for bucket in ordered_buckets:
        for candidate in bucket.candidates:
            if len(selected) >= limit:
                return selected
            if candidate.output_image in seen_images:
                continue
            if candidate.action in action_set and action_counts[candidate.action] >= quota:
                continue
            selected.append(candidate)
            seen_images.add(candidate.output_image)
            action_counts[candidate.action] += 1

    for bucket in ordered_buckets:
        for candidate in bucket.candidates:
            if len(selected) >= limit:
                return selected
            if candidate.output_image in seen_images:
                continue
            selected.append(candidate)
            seen_images.add(candidate.output_image)
            action_counts[candidate.action] += 1
    return selected


def download_archive_parts(repo_id: str, archives: dict[str, ArchiveInfo], data_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    allow_patterns = [part for archive in archives.values() for part in archive.parts]
    if not allow_patterns:
        return
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=data_dir,
    )


def materialized_archive_path(data_dir: Path, archive: ArchiveInfo) -> Path:
    return data_dir / f"{archive.base}.tar.gz"


def concatenate_parts(data_dir: Path, archive: ArchiveInfo) -> Path:
    target = materialized_archive_path(data_dir, archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == archive.size:
        return target
    with target.open("wb") as out_handle:
        for part in archive.parts:
            part_path = data_dir / part
            with part_path.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=16 * 1024 * 1024)
    return target


def cleanup_materialized_archive(data_dir: Path, archive: ArchiveInfo, *, keep_concat_tar: bool, cleanup_parts: bool) -> None:
    if not keep_concat_tar:
        materialized_archive_path(data_dir, archive).unlink(missing_ok=True)
    if cleanup_parts:
        for part in archive.parts:
            (data_dir / part).unlink(missing_ok=True)


def safe_extract_selected(archive_path: Path, members: set[str], output_root: Path) -> int:
    extracted = 0
    output_root.mkdir(parents=True, exist_ok=True)
    wanted = {member.strip("/") for member in members}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            normalized = member.name.strip("./")
            if normalized not in wanted:
                continue
            if member.isdir():
                continue
            target = output_root / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            extracted += 1
    return extracted


def write_outputs(args: argparse.Namespace, selected: list[Candidate], rejects: list[dict[str, Any]], buckets: list[SourceBucket]) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.rejects.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(item.row, ensure_ascii=True) + "\n" for item in selected), encoding="utf-8")
    args.rejects.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in rejects), encoding="utf-8")

    action_counts = Counter(item.action for item in selected)
    archive_counts = Counter(item.archive_base for item in selected)
    source_counts = Counter(item.source for item in selected)
    selected_archives = {bucket.archive.base: bucket.archive for bucket in buckets if bucket.archive.base in archive_counts}
    missing_images = [item.output_image for item in selected if not Path(item.output_image).exists()]
    total_archive_gb = sum(archive.size for archive in selected_archives.values()) / (1024**3)
    lines = [
        "# ScaleCUA Real Subset Summary",
        "",
        f"rows: {len(selected)}",
        f"missing_images: {len(missing_images)}",
        f"selected_archives: {len(selected_archives)}",
        f"compressed_archive_gb: {total_archive_gb:.2f}",
        "",
        "## Actions",
    ]
    for action, count in action_counts.most_common():
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## Archives"])
    for archive, count in archive_counts.most_common():
        size_gb = selected_archives[archive].size / (1024**3)
        lines.append(f"- {archive}: rows={count}, compressed_gb={size_gb:.2f}")
    lines.extend(["", "## Sources"])
    for source, count in source_counts.most_common(30):
        lines.append(f"- {source}: {count}")
    if missing_images:
        lines.extend(["", "## Missing Images", *[f"- {path}" for path in missing_images[:20]]])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    archives = repo_archives(args.repo_id)
    buckets, rejects = scan_sources(args, archives)
    selected = choose_candidates(buckets, args.limit, tuple(args.actions))
    selected_archive_map = {item.archive_base: archives[item.archive_base] for item in selected}

    if not args.dry_run:
        if args.download:
            download_archive_parts(args.repo_id, selected_archive_map, args.data_dir)
        if args.extract:
            by_archive: dict[str, set[str]] = defaultdict(set)
            for item in selected:
                by_archive[item.archive_base].add(item.extract_member)
            for archive_base, members in by_archive.items():
                archive = selected_archive_map[archive_base]
                archive_path = concatenate_parts(args.data_dir, archive)
                output_root = args.images_dir / Path(archive_base).parent
                safe_extract_selected(archive_path, members, output_root)
                cleanup_materialized_archive(
                    args.data_dir,
                    archive,
                    keep_concat_tar=args.keep_concat_tar,
                    cleanup_parts=args.cleanup_archive_parts,
                )
    write_outputs(args, selected, rejects, buckets)
    return {
        "rows": len(selected),
        "archives": len(selected_archive_map),
        "compressed_gb": round(sum(item.size for item in selected_archive_map.values()) / (1024**3), 3),
        "out": str(args.out),
        "report": str(args.report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="OpenGVLab/ScaleCUA-Data", help="Hugging Face dataset repo id.")
    parser.add_argument("--annotations", type=Path, default=Path(".w8-biayn/scalecua/raw/annotations"))
    parser.add_argument("--meta", type=Path, default=Path(".w8-biayn/scalecua/raw/meta.json"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--platforms", nargs="*", default=["ubuntu", "windows", "web", "mac"])
    parser.add_argument("--actions", nargs="*", default=list(DEFAULT_ACTIONS))
    parser.add_argument("--max-rows-per-source", type=int, default=0, help="Optional scan cap per annotation source.")
    parser.add_argument("--data-dir", type=Path, default=Path(".w8-biayn/scalecua/raw"))
    parser.add_argument("--images-dir", type=Path, default=Path(".w8-biayn/scalecua/images"))
    parser.add_argument("--out", type=Path, default=Path(".w8-biayn/scalecua/prepared/train-10k-toolcall.jsonl"))
    parser.add_argument("--rejects", type=Path, default=Path(".w8-biayn/scalecua/prepared/train-10k-rejected.jsonl"))
    parser.add_argument("--report", type=Path, default=Path(".w8-biayn/scalecua/reports/train-10k-summary.md"))
    parser.add_argument("--dry-run", action="store_true", help="Plan/write JSONL/report without downloading or extracting archives.")
    parser.add_argument("--no-download", dest="download", action="store_false", help="Skip archive part download.")
    parser.add_argument("--no-extract", dest="extract", action="store_false", help="Skip selected-image extraction.")
    parser.add_argument(
        "--keep-concat-tar",
        action="store_true",
        help="Keep the temporary concatenated .tar.gz after extracting selected images. By default it is deleted.",
    )
    parser.add_argument(
        "--cleanup-archive-parts",
        action="store_true",
        help="Delete downloaded .part-* files after extraction so only extracted selected images remain.",
    )
    parser.set_defaults(download=True, extract=True)
    return parser.parse_args()


def main() -> None:
    result = prepare(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
