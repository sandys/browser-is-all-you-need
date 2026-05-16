#!/usr/bin/env python3
"""Validate a portable Agent Skill folder."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$|^[a-z0-9]$")
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ALLOWED_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}


def parse_simple_frontmatter(text: str) -> dict[str, object]:
    """Parse simple YAML frontmatter when PyYAML is unavailable."""
    parsed: dict[str, object] = {}
    current_map: str | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith((" ", "\t")):
            if current_map is not None and ":" in raw_line:
                key, value = raw_line.strip().split(":", 1)
                nested = parsed.setdefault(current_map, {})
                if isinstance(nested, dict):
                    nested[key.strip()] = value.strip().strip("\"'")
            continue

        current_map = None
        if ":" not in raw_line:
            raise ValueError(f"Invalid frontmatter line: {raw_line}")
        key, value = raw_line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            parsed[key] = {}
            current_map = key
        else:
            parsed[key] = value.strip("\"'")

    return parsed


def load_frontmatter(skill_md: Path) -> tuple[dict[str, object], str]:
    content = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(content)
    if not match:
        raise ValueError("SKILL.md must start with YAML frontmatter between --- markers")

    frontmatter_text = match.group(1)
    body = content[match.end() :]

    try:
        import yaml  # type: ignore
    except Exception:
        frontmatter = parse_simple_frontmatter(frontmatter_text)
    else:
        loaded = yaml.safe_load(frontmatter_text)
        if not isinstance(loaded, dict):
            raise ValueError("Frontmatter must be a YAML mapping")
        frontmatter = loaded

    return frontmatter, body


def is_external_link(target: str) -> bool:
    parsed = urlparse(target)
    return bool(parsed.scheme or parsed.netloc or target.startswith("#"))


def referenced_paths(body: str) -> list[str]:
    paths: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(body):
        target = match.group(1).strip()
        if not target or is_external_link(target):
            continue
        path = target.split("#", 1)[0].strip()
        if path:
            paths.append(path)
    return paths


def validate(skill_dir: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not skill_dir.exists():
        return [f"Skill directory does not exist: {skill_dir}"], warnings
    if not skill_dir.is_dir():
        return [f"Skill path is not a directory: {skill_dir}"], warnings

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [f"Missing required file: {skill_md}"], warnings

    try:
        frontmatter, body = load_frontmatter(skill_md)
    except Exception as exc:
        return [str(exc)], warnings

    unknown = sorted(set(frontmatter) - ALLOWED_FIELDS)
    if unknown:
        warnings.append(
            "Non-standard frontmatter fields may reduce portability: "
            + ", ".join(unknown)
        )

    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("Frontmatter requires non-empty string field: name")
    else:
        name = name.strip()
        if len(name) > 64:
            errors.append("name must be 64 characters or fewer")
        if not NAME_RE.match(name) or "--" in name:
            errors.append(
                "name must use lowercase letters, digits, single hyphens, and no leading/trailing hyphen"
            )
        if skill_dir.name != name:
            errors.append(f"name must match folder name: {name!r} != {skill_dir.name!r}")

    description = frontmatter.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("Frontmatter requires non-empty string field: description")
    else:
        if len(description.strip()) > 1024:
            errors.append("description must be 1024 characters or fewer")

    body_lines = body.splitlines()
    if len(body_lines) > 500:
        warnings.append(f"SKILL.md body is {len(body_lines)} lines; consider references/")

    for relative in referenced_paths(body):
        if (skill_dir / relative).exists():
            continue
        errors.append(f"Referenced file does not exist: {relative}")

    for dirname in ("references", "scripts", "assets"):
        path = skill_dir / dirname
        if path.exists() and not path.is_dir():
            errors.append(f"{dirname}/ exists but is not a directory")

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill_dir", type=Path, help="Path to a skill directory")
    args = parser.parse_args()

    errors, warnings = validate(args.skill_dir)

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        return 1

    print(f"OK: {args.skill_dir} is a valid Agent Skill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
