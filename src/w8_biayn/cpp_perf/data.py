"""Repeatable dataset setup helpers for C++ performance RL."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from w8_biayn.constants import (
    CPP_DATA_SCHEMA_VERSION,
    PIE_CXX_SPLITS_URL,
    PIE_GENERATED_TEST_CASES_URL,
    PIE_MERGED_TEST_CASES_URL,
    PIE_PROBLEM_STATEMENTS_URL,
    PIE_PUBLIC_TEST_CASES_URL,
    PIE_TRAJECTORIES_URL,
    SUPERCODER_DATASET,
)


DATA_MANIFEST = "_w8_data_manifest.json"
SUPERCODER_ALLOW_PATTERNS = (
    "README.md",
    "data/fewshot-*.parquet",
    "data/val-*.parquet",
)


@dataclass(frozen=True)
class DownloadTarget:
    name: str
    url: str
    relative_path: str


PIE_DOWNLOADS: tuple[DownloadTarget, ...] = (
    DownloadTarget("trajectories", PIE_TRAJECTORIES_URL, "raw/pie-trajectories.zip"),
    DownloadTarget("cxx-splits", PIE_CXX_SPLITS_URL, "raw/pie-cxx-splits.zip"),
    DownloadTarget("merged-tests", PIE_MERGED_TEST_CASES_URL, "raw/merged-test-cases.zip"),
    DownloadTarget("public-tests", PIE_PUBLIC_TEST_CASES_URL, "raw/public-test-cases.zip"),
    DownloadTarget("generated-tests", PIE_GENERATED_TEST_CASES_URL, "raw/generated-test-cases.zip"),
    DownloadTarget("problem-statements", PIE_PROBLEM_STATEMENTS_URL, "raw/problem-statements-translated.zip"),
)

SPLIT_FILE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "train": ("train.jsonl", "train.tsv"),
    "validation": ("validation.jsonl", "valid.jsonl", "val.jsonl", "validation.tsv", "valid.tsv", "val.tsv"),
    "test": ("test.jsonl", "test.tsv"),
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_records(root: str | Path) -> list[dict[str, Any]]:
    """Return checksum records for files under a data tree."""

    base = Path(root)
    records: list[dict[str, Any]] = []
    if not base.exists():
        return records
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        if path.name == DATA_MANIFEST:
            continue
        records.append(
            {
                "path": path.relative_to(base).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def write_data_manifest(
    root: str | Path,
    *,
    kind: str,
    sources: Iterable[str] = (),
    options: dict[str, Any] | None = None,
    include_files: bool = True,
) -> Path:
    """Write a conversion/download manifest with checksums."""

    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CPP_DATA_SCHEMA_VERSION,
        "kind": kind,
        "sources": list(sources),
        "options": options or {},
        "files": tree_records(base) if include_files else [],
    }
    path = base / DATA_MANIFEST
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_data_manifest(root: str | Path, manifest_path: str | Path | None = None) -> list[str]:
    """Return manifest verification errors for a data tree."""

    base = Path(root)
    manifest = Path(manifest_path) if manifest_path is not None else base / DATA_MANIFEST
    if not manifest.exists():
        return [f"missing manifest: {manifest}"]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CPP_DATA_SCHEMA_VERSION:
        return [
            f"schema mismatch: expected {CPP_DATA_SCHEMA_VERSION}, got {payload.get('schema_version')}"
        ]
    errors: list[str] = []
    for record in payload.get("files", []):
        rel = record.get("path")
        path = base / rel
        if not path.exists():
            errors.append(f"missing file: {rel}")
            continue
        if path.stat().st_size != record.get("bytes"):
            errors.append(f"size mismatch: {rel}")
            continue
        if sha256_file(path) != record.get("sha256"):
            errors.append(f"sha256 mismatch: {rel}")
    return errors


def download_url(url: str, output: str | Path, *, force: bool = False) -> Path:
    """Download one URL, using gdown for Google Drive links."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination
    if "drive.google.com" in url:
        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - dependency check covers this
            raise RuntimeError("gdown is required for Google Drive PIE downloads") from exc
        downloaded = _download_with_gdown(gdown, url=url, destination=destination)
        if not downloaded:
            raise RuntimeError(f"gdown did not return an output path for {url}")
    else:
        with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return destination


def _download_with_gdown(gdown_module: Any, *, url: str, destination: Path) -> Any:
    """Download a Google Drive URL across gdown versions."""

    try:
        return gdown_module.download(url=url, output=str(destination), quiet=False, fuzzy=True)
    except TypeError as exc:
        if "fuzzy" not in str(exc):
            raise
        return gdown_module.download(url=url, output=str(destination), quiet=False)


def download_pie(root: str | Path, *, force: bool = False) -> list[Path]:
    """Download PIE source archives into the project data root."""

    base = Path(root)
    written = [download_url(target.url, base / target.relative_path, force=force) for target in PIE_DOWNLOADS]
    write_data_manifest(
        base,
        kind="pie-download",
        sources=[target.url for target in PIE_DOWNLOADS],
        options={"force": force},
    )
    return written


def prepare_full_pie(
    source_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Unpack official PIE archives and normalize full-run split/test inputs."""

    source = Path(source_root)
    output = Path(output_root)
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    extracted = output / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)

    archives: dict[str, Path] = {}
    for target in PIE_DOWNLOADS:
        path = source / target.relative_path
        if path.exists() and zipfile.is_zipfile(path):
            destination = extracted / target.name
            if force and destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True, exist_ok=True)
            _safe_extract_zip(path, destination)
            archives[target.name] = destination

    if "cxx-splits" not in archives:
        raise FileNotFoundError(
            f"missing C++ split archive: expected {source / 'raw/pie-cxx-splits.zip'}; "
            "run `w8-biayn data pie download` first"
        )

    split_paths = _normalize_split_files(archives["cxx-splits"], output)
    cases = _normalize_test_cases(
        [
            archives[name]
            for name in ("merged-tests", "public-tests", "generated-tests")
            if name in archives
        ],
        output / "cases",
        force=force,
    )
    manifest = write_data_manifest(
        output,
        kind="pie-full-prepared",
        sources=[str(source / target.relative_path) for target in PIE_DOWNLOADS],
        options={
            "force": force,
            "split_files": {key: str(path.relative_to(output)) for key, path in split_paths.items()},
            "problem_count_with_tests": len(cases),
            "test_count": sum(cases.values()),
        },
        include_files=False,
    )
    return {
        "root": output,
        "manifest": manifest,
        "splits": split_paths,
        "problem_count_with_tests": len(cases),
        "test_count": sum(cases.values()),
    }


def download_supercoder(
    root: str | Path,
    *,
    dataset: str = SUPERCODER_DATASET,
    force: bool = False,
) -> Path:
    """Download the small SuperCoder reference splits used for schema/eval study."""

    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        local_dir=str(output),
        local_dir_use_symlinks=False,
        allow_patterns=list(SUPERCODER_ALLOW_PATTERNS),
        force_download=force,
    )
    write_data_manifest(
        output,
        kind="supercoder-download",
        sources=[dataset],
        options={"allow_patterns": list(SUPERCODER_ALLOW_PATTERNS), "force": force},
    )
    return Path(snapshot)


def inspect_supercoder_parquet(path: str | Path) -> dict[str, Any]:
    """Inspect a SuperCoder parquet file without loading the full dataset."""

    parquet_path = Path(path)
    if _is_git_lfs_pointer(parquet_path):
        raise ValueError(f"{parquet_path} is a Git LFS pointer, not resolved parquet data")
    metadata = pq.ParquetFile(parquet_path)
    schema = metadata.schema_arrow
    return {
        "path": str(parquet_path),
        "rows": metadata.metadata.num_rows if metadata.metadata else 0,
        "columns": schema.names,
        "schema": str(schema),
    }


def build_tests_manifest_from_io(
    problem_ids: Iterable[str],
    inputs_outputs_basepath: str | Path,
    coverage_by_problem: dict[str, dict[str, float]],
    *,
    visible_count: int = 1,
    hidden_count: int = 1,
) -> dict[str, dict[str, Any]]:
    """Build a tests manifest from PIE/CodeNet-style input/output files."""

    if visible_count <= 0 or hidden_count <= 0:
        raise ValueError("visible_count and hidden_count must both be positive")
    base = Path(inputs_outputs_basepath)
    manifest: dict[str, dict[str, Any]] = {}
    for problem_id in problem_ids:
        tests = discover_problem_tests(base / problem_id)
        if len(tests) < visible_count + hidden_count:
            continue
        coverage = coverage_by_problem.get(problem_id)
        if coverage is None:
            continue
        manifest[problem_id] = {
            "unit_tests": tests[:visible_count],
            "hidden_tests": tests[visible_count : visible_count + hidden_count],
            "coverage": coverage,
        }
    return manifest


def build_tests_manifest_with_report(
    problem_ids: Iterable[str],
    inputs_outputs_basepath: str | Path,
    coverage_by_problem: dict[str, dict[str, float]],
    *,
    visible_count: int = 1,
    hidden_count: int = 1,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build a tests manifest and report skipped problem reasons."""

    if visible_count <= 0 or hidden_count <= 0:
        raise ValueError("visible_count and hidden_count must both be positive")
    base = Path(inputs_outputs_basepath)
    manifest: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "problems_seen": 0,
        "problems_with_manifest": 0,
        "missing_tests": [],
        "missing_coverage": [],
        "insufficient_tests": [],
    }
    for problem_id in problem_ids:
        report["problems_seen"] += 1
        tests = discover_problem_tests(base / problem_id)
        if not tests:
            report["missing_tests"].append(problem_id)
            continue
        if len(tests) < visible_count + hidden_count:
            report["insufficient_tests"].append(problem_id)
            continue
        coverage = coverage_by_problem.get(problem_id)
        if coverage is None:
            report["missing_coverage"].append(problem_id)
            continue
        manifest[problem_id] = {
            "unit_tests": tests[:visible_count],
            "hidden_tests": tests[visible_count : visible_count + hidden_count],
            "coverage": coverage,
        }
    report["problems_with_manifest"] = len(manifest)
    return manifest, report


def discover_problem_tests(problem_dir: str | Path) -> list[dict[str, str]]:
    """Read input/output test pairs from one problem directory."""

    base = Path(problem_dir)
    if not base.exists():
        return []
    indexed_inputs = sorted(base.glob("input.*.txt"))
    tests: list[dict[str, str]] = []
    for input_path in indexed_inputs:
        suffix = input_path.name.removeprefix("input.").removesuffix(".txt")
        output_path = base / f"output.{suffix}.txt"
        if output_path.exists():
            tests.append(
                {
                    "input": input_path.read_text(encoding="utf-8"),
                    "expected": output_path.read_text(encoding="utf-8"),
                }
            )
    if tests:
        return tests

    single_input = base / "inputs.txt"
    single_output = base / "outputs.txt"
    if single_input.exists() and single_output.exists():
        return [
            {
                "input": single_input.read_text(encoding="utf-8"),
                "expected": single_output.read_text(encoding="utf-8"),
            }
        ]
    return []


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"unsafe zip member path in {zip_path}: {member.filename}")
        archive.extractall(destination)


def _normalize_split_files(split_root: Path, output: Path) -> dict[str, Path]:
    written: dict[str, Path] = {}
    for split, names in SPLIT_FILE_CANDIDATES.items():
        source = _find_named_file(split_root, names)
        if source is None:
            raise FileNotFoundError(
                f"could not find {split} split under {split_root}; tried {', '.join(names)}"
            )
        destination = output / f"{split}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        written[split] = destination
    return written


def _find_named_file(root: Path, names: tuple[str, ...]) -> Path | None:
    by_name = {name.lower(): name for name in names}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name.lower() in by_name:
            return path
    return None


def _normalize_test_cases(roots: list[Path], output: Path, *, force: bool) -> dict[str, int]:
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    seen: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for root in roots:
        for problem_dir in _iter_problem_test_dirs(root):
            problem_id = problem_dir.name
            problem_out = output / problem_id
            problem_out.mkdir(parents=True, exist_ok=True)
            for test in discover_problem_tests(problem_dir):
                digest = hashlib.sha256(
                    (test["input"] + "\0" + test["expected"]).encode("utf-8")
                ).hexdigest()
                seen.setdefault(problem_id, set())
                if digest in seen[problem_id]:
                    continue
                seen[problem_id].add(digest)
                index = counts.get(problem_id, 0)
                (problem_out / f"input.{index:04d}.txt").write_text(
                    test["input"],
                    encoding="utf-8",
                )
                (problem_out / f"output.{index:04d}.txt").write_text(
                    test["expected"],
                    encoding="utf-8",
                )
                counts[problem_id] = index + 1
    return counts


def _iter_problem_test_dirs(root: Path) -> Iterable[Path]:
    for directory in sorted(item for item in root.rglob("*") if item.is_dir()):
        if discover_problem_tests(directory):
            yield directory


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    prefix = path.read_bytes()[:128]
    return prefix.startswith(b"version https://git-lfs.github.com/spec/v1")
