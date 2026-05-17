from __future__ import annotations

import pytest

from w8_biayn.domdiff import (
    ArtifactImageConfig,
    DomdiffConfig,
    LocalDomdiffConfig,
    artifact_image_uri,
    artifact_push_plan,
    dry_run_plan,
    extract_quick_url,
    image_registry,
    is_artifact_registry_image,
    local_dry_run_plan,
    safe_name,
    trycloudflare_connect_to_args,
    wait_http_json,
)


def test_safe_name_is_gcp_compatible():
    assert safe_name("W8 DOMDiff Run_123!") == "w8-domdiff-run-123"
    assert safe_name("123") == "r-123"


def test_extract_quick_url():
    text = "INFO https://abc-123.trycloudflare.com is ready"

    assert extract_quick_url(text) == "https://abc-123.trycloudflare.com"


def test_trycloudflare_connect_to_args():
    args = trycloudflare_connect_to_args("https://abc.trycloudflare.com/health")

    assert args == ["--connect-to", "abc.trycloudflare.com:443:trycloudflare.com:443"]


def test_wait_http_json_retries_post_payload(monkeypatch):
    calls = []

    def fake_request(url, *, method="GET", payload=None, timeout=30):
        calls.append((url, method, payload, timeout))
        if len(calls) == 1:
            raise RuntimeError("not ready")
        return {"job_id": "abc"}

    monkeypatch.setattr("w8_biayn.domdiff._json_http_request", fake_request)
    monkeypatch.setattr("w8_biayn.domdiff.time.sleep", lambda _seconds: None)

    assert wait_http_json(
        "https://reward.trycloudflare.com/evaluate_async",
        method="POST",
        payload={"target_url": "https://example.com"},
        expected_key="job_id",
        timeout_s=5,
    ) == {"job_id": "abc"}
    assert calls[0][1] == "POST"
    assert calls[0][2] == {"target_url": "https://example.com"}


def test_dry_run_plan_contains_nested_virtualization_and_image():
    plan = dry_run_plan(DomdiffConfig(run_id="test-run", project_id="proj"))

    assert "ghcr.io/wootzapp/android-world-domdiff:daytona-92000b7" in plan
    assert "--enable-nested-virtualization" in plan
    assert "gcloud compute instances create" in plan


def test_artifact_image_uri():
    uri = artifact_image_uri(
        project_id="proj",
        location="us-central1",
        repository="w8-biayn",
        image_name="android-world-domdiff",
        tag="local-test",
    )

    assert uri == "us-central1-docker.pkg.dev/proj/w8-biayn/android-world-domdiff:local-test"


def test_image_registry_detection():
    assert image_registry("us-central1-docker.pkg.dev/proj/repo/image:tag") == "us-central1-docker.pkg.dev"
    assert image_registry("android-world-domdiff:local") == ""
    assert is_artifact_registry_image("us-central1-docker.pkg.dev/proj/repo/image:tag")
    assert not is_artifact_registry_image("ghcr.io/wootzapp/android-world-domdiff:tag")


def test_artifact_push_plan_contains_gar_commands(tmp_path):
    credentials = tmp_path / "sa.json"
    credentials.write_text('{"type":"service_account","project_id":"proj"}', encoding="utf-8")
    plan = artifact_push_plan(
        ArtifactImageConfig(
            source_image="android-world-domdiff:local",
            credentials_path=str(credentials),
            repository="w8-biayn",
            tag="test",
        )
    )

    assert "gcloud artifacts repositories create w8-biayn" in plan
    assert "export GOOGLE_APPLICATION_CREDENTIALS=" in plan
    assert "export CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE=" in plan
    assert "gcloud auth activate-service-account" not in plan
    assert "gcloud auth configure-docker" not in plan
    assert "docker login https://us-central1-docker.pkg.dev -u _json_key --password-stdin" in plan
    assert "docker push us-central1-docker.pkg.dev/proj/w8-biayn/android-world-domdiff:test" in plan


def test_local_dry_run_plan_contains_container_adapter_and_tunnels():
    plan = local_dry_run_plan(LocalDomdiffConfig(run_id="local-test"))

    assert "Local DOMDiff dry run" in plan
    assert "docker run -d --name android-world-domdiff" in plan
    assert "-p 5080:5000" in plan
    assert "-p 9224:9224" in plan
    assert "CDP_URL=ws://localhost:9224 CHROMIUMRL_ALLOW_CDP_FALLBACK=0 uv run --extra domdiff uvicorn" in plan
    assert "w8_biayn.rewards.chromiumrl_service:app" in plan
    assert "cloudflared" in plan
    assert "wss://<local-cdp-tunnel>" not in plan


def test_local_quick_tunnel_uses_http2_protocol(tmp_path, monkeypatch):
    from w8_biayn.domdiff import DomdiffError, LocalDomdiffHost

    config = LocalDomdiffConfig(run_id="local-test")
    manager = LocalDomdiffHost(config, tmp_path)
    manager.state.cloudflared_path = "/bin/echo"
    launched = {}

    class FakeProcess:
        pid = 123

        def poll(self):
            return None

    class FakeLogHandle:
        def close(self):
            return None

    def fake_open(*_args, **_kwargs):
        return FakeLogHandle()

    def fake_popen(command, **_kwargs):
        launched["command"] = command
        raise DomdiffError("stop after command capture")

    monkeypatch.setattr("w8_biayn.domdiff.Path.open", fake_open)
    monkeypatch.setattr("w8_biayn.domdiff.subprocess.Popen", fake_popen)

    with pytest.raises(DomdiffError):
        manager._start_local_tunnel(
            label="chromiumrl",
            url="http://127.0.0.1:8080",
            log_path=tmp_path / "cloudflared.log",
            expected_path="/health",
            expected_key="status",
        )

    assert "--protocol" in launched["command"]
    assert "http2" in launched["command"]


def test_local_dry_run_plan_publishes_cdp_only_when_requested():
    plan = local_dry_run_plan(LocalDomdiffConfig(run_id="local-test", publish_cdp=True))

    assert "wss://<local-cdp-tunnel>" in plan
    assert "--http-host-header localhost" in plan
