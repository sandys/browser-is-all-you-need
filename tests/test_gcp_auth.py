from __future__ import annotations

import json

from w8_biayn.gcp_auth import check_project_permissions, service_account_env, service_account_env_exports


def test_service_account_env_uses_json_without_gcloud_state(tmp_path):
    credentials = tmp_path / "sa.json"
    credentials.write_text(json.dumps({"type": "service_account", "project_id": "proj"}), encoding="utf-8")

    env = service_account_env(str(credentials), base_env={"PATH": "/bin"})

    assert env["PATH"] == "/bin"
    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == str(credentials.resolve())
    assert env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] == str(credentials.resolve())
    assert env["CLOUDSDK_CORE_PROJECT"] == "proj"
    assert env["GOOGLE_CLOUD_PROJECT"] == "proj"
    assert env["GCLOUD_PROJECT"] == "proj"
    assert env["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "1"
    assert "CLOUDSDK_CONFIG" not in env


def test_service_account_env_exports_for_dry_run(tmp_path):
    credentials = tmp_path / "sa.json"
    credentials.write_text(json.dumps({"type": "service_account", "project_id": "proj"}), encoding="utf-8")

    exports = service_account_env_exports(str(credentials), project_id="proj")

    assert exports == [
        f"export GOOGLE_APPLICATION_CREDENTIALS={credentials.resolve()}",
        f"export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE={credentials.resolve()}",
        "export CLOUDSDK_CORE_PROJECT=proj",
    ]


def test_project_permissions_returns_missing(monkeypatch, tmp_path):
    credentials = tmp_path / "sa.json"
    credentials.write_text(json.dumps({"type": "service_account", "project_id": "proj"}), encoding="utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps({"permissions": ["compute.instances.create"]}).encode("utf-8")

    monkeypatch.setattr("w8_biayn.gcp_auth.service_account_access_token", lambda _path: "token")
    monkeypatch.setattr("w8_biayn.gcp_auth.urllib.request.urlopen", lambda *_args, **_kwargs: Response())

    missing = check_project_permissions(
        str(credentials),
        project_id="proj",
        permissions=["compute.instances.create", "resourcemanager.projects.setIamPolicy"],
    )

    assert missing == ["resourcemanager.projects.setIamPolicy"]
