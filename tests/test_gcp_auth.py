from __future__ import annotations

import json

from w8_biayn.gcp_auth import service_account_env, service_account_env_exports


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
