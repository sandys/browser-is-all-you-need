from __future__ import annotations

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
    assert "gcloud auth configure-docker us-central1-docker.pkg.dev --quiet" in plan
    assert "docker push us-central1-docker.pkg.dev/proj/w8-biayn/android-world-domdiff:test" in plan


def test_local_dry_run_plan_contains_container_adapter_and_tunnels():
    plan = local_dry_run_plan(LocalDomdiffConfig(run_id="local-test"))

    assert "Local DOMDiff dry run" in plan
    assert "docker run -d --name android-world-domdiff" in plan
    assert "-p 5080:5000" in plan
    assert "-p 9224:9224" in plan
    assert "CDP_URL=ws://127.0.0.1:9224 uv run --extra domdiff uvicorn" in plan
    assert "cloudflared" in plan
    assert "wss://<local-cdp-tunnel>" in plan
