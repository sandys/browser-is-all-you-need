"""Service-account credential helpers for scoped GCP command execution."""

from __future__ import annotations

import os
import json
import shlex
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Mapping

from .secrets import CredentialError, get_project_id

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GcpAuthError(RuntimeError):
    """Expected service-account credential failure."""


def service_account_env(
    credentials_path: str,
    *,
    project_id: str | None = None,
    cloud_sdk_config_dir: str | None = None,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return env vars that make tools use the JSON key without mutating gcloud state."""
    try:
        resolved_project = project_id or get_project_id(credentials_path)
    except CredentialError as exc:
        raise GcpAuthError(str(exc)) from exc
    resolved_credentials = str(Path(credentials_path).expanduser().resolve())
    env = dict(base_env or os.environ)
    env.update(
        {
            "GOOGLE_APPLICATION_CREDENTIALS": resolved_credentials,
            "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE": resolved_credentials,
            "CLOUDSDK_CORE_PROJECT": resolved_project,
            "GOOGLE_CLOUD_PROJECT": resolved_project,
            "GCLOUD_PROJECT": resolved_project,
            "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
        }
    )
    if cloud_sdk_config_dir:
        env["CLOUDSDK_CONFIG"] = cloud_sdk_config_dir
    return env


def service_account_env_exports(credentials_path: str, *, project_id: str) -> list[str]:
    """Return dry-run export lines for the scoped service-account environment."""
    resolved_credentials = str(Path(credentials_path).expanduser().resolve())
    return [
        f"export GOOGLE_APPLICATION_CREDENTIALS={shlex.quote(resolved_credentials)}",
        f"export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE={shlex.quote(resolved_credentials)}",
        f"export CLOUDSDK_CORE_PROJECT={shlex.quote(project_id)}",
    ]


def service_account_access_token(
    credentials_path: str,
    *,
    scopes: Iterable[str] = (CLOUD_PLATFORM_SCOPE,),
) -> str:
    """Mint an OAuth token from the JSON key without calling `gcloud auth`."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as exc:
        raise GcpAuthError("google-auth is required; run `./scripts/bootstrap.sh` or `uv sync`.") from exc
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=list(scopes),
        )
        credentials.refresh(Request())
    except Exception as exc:  # pragma: no cover - exercised only with real credentials/network
        raise GcpAuthError(f"Could not mint access token from service-account JSON: {exc}") from exc
    if not credentials.token:
        raise GcpAuthError("Service-account token refresh returned no token.")
    return credentials.token


def check_project_permissions(
    credentials_path: str,
    *,
    project_id: str,
    permissions: Iterable[str],
) -> list[str]:
    """Return project-level IAM permissions missing from the service account."""
    requested = list(permissions)
    token = service_account_access_token(credentials_path)
    body = json.dumps({"permissions": requested}).encode("utf-8")
    request = urllib.request.Request(
        f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}:testIamPermissions",
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            doc = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GcpAuthError(f"GCP permission preflight failed: {exc.code} {detail}") from exc
    allowed = set(doc.get("permissions", []))
    return [permission for permission in requested if permission not in allowed]
