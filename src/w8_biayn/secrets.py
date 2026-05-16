"""Credential helpers that avoid printing secret material."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CredentialError(RuntimeError):
    """Raised when a local credential file is missing or malformed."""


def load_service_account_metadata(path: str | Path) -> dict[str, Any]:
    """Load non-secret metadata from a GCP service-account JSON file."""
    credential_path = Path(path)
    if not credential_path.exists():
        raise CredentialError(f"Credential file not found: {credential_path}")
    if not credential_path.is_file():
        raise CredentialError(f"Credential path is not a file: {credential_path}")

    try:
        data = json.loads(credential_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CredentialError(f"Credential file is not valid JSON: {credential_path}") from exc

    if not isinstance(data, dict):
        raise CredentialError("Credential JSON must be an object")
    if data.get("type") != "service_account":
        raise CredentialError("Credential JSON must be a GCP service-account key")
    return data


def get_project_id(path: str | Path) -> str:
    """Return the GCP project id from a service-account JSON file."""
    data = load_service_account_metadata(path)
    project_id = data.get("project_id") or data.get("quota_project_id")
    if not isinstance(project_id, str) or not project_id:
        raise CredentialError("Credential JSON does not contain project_id")
    return project_id


def default_bucket_for_project(project_id: str) -> str:
    """Return the default artifact bucket URI for a project."""
    safe_project = project_id.lower().replace("_", "-")
    return f"gs://{safe_project}-w8-biayn"

