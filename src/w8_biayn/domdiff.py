"""GCP lifecycle for the prebuilt Android DOMDiff reward image."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import (
    DEFAULT_DOMDIFF_ARTIFACT_IMAGE,
    DEFAULT_DOMDIFF_ARTIFACT_LOCATION,
    DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_DOMDIFF_BOOT_DISK_TYPE,
    DEFAULT_DOMDIFF_IMAGE,
    DEFAULT_DOMDIFF_LOCAL_IMAGE,
    DEFAULT_DOMDIFF_MACHINE_TYPE,
    DEFAULT_DOMDIFF_MIN_CPU_PLATFORM,
    DEFAULT_DOMDIFF_VOLUME_GB,
    DEFAULT_DOMDIFF_ZONE,
)
from .gcp_auth import GcpAuthError, check_project_permissions, service_account_env, service_account_env_exports
from .secrets import CredentialError, get_project_id
from .shell import format_command

DOMDIFF_STATE_ROOT = Path(".w8-biayn/domdiff")
DOMDIFF_LOCAL_STATE_ROOT = Path(".w8-biayn/domdiff-local")
REQUIRED_GCP_PERMISSIONS = (
    "compute.instances.create",
    "compute.instances.delete",
    "compute.instances.get",
    "compute.instances.getSerialPortOutput",
    "compute.disks.create",
    "compute.disks.delete",
    "compute.firewalls.create",
    "compute.firewalls.delete",
    "compute.networks.get",
)


class DomdiffError(RuntimeError):
    """Expected DOMDiff lifecycle failure with an actionable message."""


@dataclass(frozen=True)
class ArtifactImageConfig:
    source_image: str = DEFAULT_DOMDIFF_LOCAL_IMAGE
    credentials_path: str = DEFAULT_CREDENTIALS_PATH
    project_id: str = ""
    location: str = DEFAULT_DOMDIFF_ARTIFACT_LOCATION
    repository: str = DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY
    image_name: str = DEFAULT_DOMDIFF_ARTIFACT_IMAGE
    tag: str = ""
    create_repository: bool = True
    gcloud_config_dir: str = ".w8-biayn/domdiff/image-push/gcloud-config"

    @property
    def resolved_project_id(self) -> str:
        if self.project_id:
            return self.project_id
        try:
            return get_project_id(self.credentials_path)
        except CredentialError as exc:
            raise DomdiffError(str(exc)) from exc


@dataclass(frozen=True)
class DomdiffConfig:
    run_id: str
    credentials_path: str = DEFAULT_CREDENTIALS_PATH
    project_id: str = ""
    zone: str = DEFAULT_DOMDIFF_ZONE
    machine_type: str = DEFAULT_DOMDIFF_MACHINE_TYPE
    min_cpu_platform: str = DEFAULT_DOMDIFF_MIN_CPU_PLATFORM
    image_family: str = "ubuntu-2204-lts"
    image_project: str = "ubuntu-os-cloud"
    network: str = "default"
    boot_disk_type: str = DEFAULT_DOMDIFF_BOOT_DISK_TYPE
    volume_gb: int = DEFAULT_DOMDIFF_VOLUME_GB
    reward_image: str = DEFAULT_DOMDIFF_IMAGE
    keep: bool = False

    @property
    def resolved_project_id(self) -> str:
        if self.project_id:
            return self.project_id
        try:
            return get_project_id(self.credentials_path)
        except CredentialError as exc:
            raise DomdiffError(str(exc)) from exc


@dataclass(frozen=True)
class LocalDomdiffConfig:
    run_id: str
    image: str = DEFAULT_DOMDIFF_LOCAL_IMAGE
    container_name: str = "android-world-domdiff"
    host_health_port: int = 5080
    container_health_port: int = 5000
    host_cdp_port: int = 9224
    container_cdp_port: int = 9224
    reward_host: str = "127.0.0.1"
    reward_port: int = 8080
    keep_container: bool = False
    reuse_container: bool = True
    publish_cdp: bool = False
    verify_timeout_s: int = 420


@dataclass
class DomdiffState:
    run_id: str
    project_id: str
    zone: str
    reward_image: str
    instance_name: str = ""
    public_ip: str = ""
    firewall_rule: str = ""
    network_tag: str = ""
    key_path: str = ""
    gcloud_config_dir: str = ""
    chromiumrl_url: str = ""
    cdp_quick_url: str = ""
    cdp_url: str = ""
    status: str = "created"

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "state.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, run_dir: Path) -> "DomdiffState":
        path = run_dir / "state.json"
        if not path.exists():
            raise DomdiffError(f"DOMDiff state file not found: {path}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class LocalDomdiffState:
    run_id: str
    image: str
    container_name: str
    health_url: str
    cdp_http_url: str
    local_cdp_url: str
    reward_url: str
    cloudflared_path: str = ""
    started_container: bool = False
    reward_pid: int = 0
    reward_log_path: str = ""
    chromiumrl_tunnel_pid: int = 0
    chromiumrl_tunnel_log_path: str = ""
    cdp_tunnel_pid: int = 0
    cdp_tunnel_log_path: str = ""
    chromiumrl_url: str = ""
    cdp_quick_url: str = ""
    cdp_url: str = ""
    status: str = "created"

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "state.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, run_dir: Path) -> "LocalDomdiffState":
        path = run_dir / "state.json"
        if not path.exists():
            raise DomdiffError(f"Local DOMDiff state file not found: {path}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def default_run_id(prefix: str = "domdiff") -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def run_dir_for(run_id: str) -> Path:
    return DOMDIFF_STATE_ROOT / run_id


def resolve_run_dir(run_id: str | None) -> Path:
    if run_id:
        return run_dir_for(run_id)
    candidates = [path for path in DOMDIFF_STATE_ROOT.glob("*") if (path / "state.json").exists()]
    if not candidates:
        raise DomdiffError("No DOMDiff run state found. Pass --run-id or run `w8-biayn domdiff up`.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def local_run_dir_for(run_id: str) -> Path:
    return DOMDIFF_LOCAL_STATE_ROOT / run_id


def resolve_local_run_dir(run_id: str | None) -> Path:
    if run_id:
        return local_run_dir_for(run_id)
    candidates = [path for path in DOMDIFF_LOCAL_STATE_ROOT.glob("*") if (path / "state.json").exists()]
    if not candidates:
        raise DomdiffError("No local DOMDiff run state found. Pass --run-id or run `w8-biayn domdiff local up`.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def safe_name(value: str, *, max_len: int = 63) -> str:
    safe = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    safe = re.sub(r"-+", "-", safe) or "run"
    if not safe[0].isalpha():
        safe = f"r-{safe}"
    safe = safe[:max_len].rstrip("-")
    if not safe[-1].isalnum():
        safe = f"{safe}0"
    return safe


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def find_cloudflared() -> Path | None:
    configured = os.environ.get("CLOUDFLARED_BIN", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(".local/cloudflared/cloudflared"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    import shutil

    found = shutil.which("cloudflared")
    return Path(found) if found else None


def extract_quick_url(text: str) -> str:
    match = re.search(r"https://[-A-Za-z0-9]+[.]trycloudflare[.]com", text or "")
    return match.group(0) if match else ""


def trycloudflare_connect_to_args(url: str) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host.lower().endswith(".trycloudflare.com"):
        return []
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target_port = 443 if parsed.scheme == "https" else 80
    return ["--connect-to", f"{host}:{port}:trycloudflare.com:{target_port}"]


def _tail(text: str, limit: int = 2000) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[-limit:]


def _run_local(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        env=env,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise DomdiffError(f"{cmd[0]} failed exit={result.returncode}: {_tail(result.stdout)}")
    return result


def _public_ip() -> str:
    with urllib.request.urlopen("https://api.ipify.org", timeout=20) as response:
        ip = response.read().decode("utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
        raise DomdiffError(f"Could not determine a valid public IPv4 address: {ip!r}")
    return ip


def _json_http_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            doc = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        connect_to = trycloudflare_connect_to_args(url)
        if not connect_to:
            raise
        args = [
            "curl",
            "-fsSL",
            "--retry",
            "6",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--max-time",
            str(timeout),
            *connect_to,
        ]
        if body is not None:
            args.extend(["-H", "Content-Type: application/json", "-X", method, "--data-binary", "@-"])
        args.append(url)
        result = _run_local(
            args,
            input_text=body.decode("utf-8") if body is not None else None,
            timeout=timeout + 10,
            check=False,
        )
        if result.returncode != 0:
            raise DomdiffError(f"trycloudflare curl fallback failed for {url}: {_tail(result.stdout)}") from exc
        doc = json.loads(result.stdout)
    if not isinstance(doc, dict):
        raise DomdiffError(f"Expected object JSON from {url}")
    return doc


def wait_http_json(
    url: str,
    *,
    timeout_s: int = 300,
    expected_key: str | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            doc = _json_http_request(url, method=method, payload=payload, timeout=20)
            if expected_key is None or expected_key in doc:
                return doc
            last_error = f"missing key {expected_key}"
        except Exception as exc:  # noqa: BLE001 - retry tunnel startup and DNS propagation
            last_error = str(exc)
        time.sleep(2)
    raise DomdiffError(f"Timed out waiting for {url}: {_tail(last_error)}")


def image_registry(image: str) -> str:
    if "/" not in image:
        return ""
    first = image.split("/", 1)[0]
    if "." in first or ":" in first or first == "localhost":
        return first
    return ""


def is_artifact_registry_image(image: str) -> bool:
    return image_registry(image).endswith("-docker.pkg.dev")


def artifact_registry_host(location: str) -> str:
    return f"{location}-docker.pkg.dev"


def artifact_image_uri(
    *,
    project_id: str,
    location: str = DEFAULT_DOMDIFF_ARTIFACT_LOCATION,
    repository: str = DEFAULT_DOMDIFF_ARTIFACT_REPOSITORY,
    image_name: str = DEFAULT_DOMDIFF_ARTIFACT_IMAGE,
    tag: str,
) -> str:
    return f"{artifact_registry_host(location)}/{project_id}/{repository}/{image_name}:{tag}"


def local_docker_image_id(source_image: str) -> str:
    result = _run_local(
        ["docker", "image", "inspect", "--format", "{{.Id}}", source_image],
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise DomdiffError(
            f"Local Docker image not found: {source_image}. "
            "Run `docker images`, build/tag the DOMDiff image locally, or pass --source-image/--local-reward-image."
        )
    image_id = result.stdout.strip()
    if not image_id:
        raise DomdiffError(f"Docker did not return an image id for {source_image}")
    return image_id.removeprefix("sha256:")


def local_docker_image_exists(source_image: str) -> bool:
    if not shutil.which("docker"):
        return False
    result = _run_local(
        ["docker", "image", "inspect", source_image],
        timeout=120,
        check=False,
    )
    return result.returncode == 0


def default_artifact_tag(source_image: str) -> str:
    return f"local-{local_docker_image_id(source_image)[:12]}"


def resolved_artifact_tag(config: ArtifactImageConfig, *, dry_run: bool = False) -> str:
    if config.tag:
        return config.tag
    if dry_run:
        return "local-<image-id>"
    return default_artifact_tag(config.source_image)


def artifact_push_plan(config: ArtifactImageConfig) -> str:
    project = config.resolved_project_id
    tag = resolved_artifact_tag(config, dry_run=True)
    target = artifact_image_uri(
        project_id=project,
        location=config.location,
        repository=config.repository,
        image_name=config.image_name,
        tag=tag,
    )
    host = artifact_registry_host(config.location)
    lines = [
        "# DOMDiff Artifact Registry image push dry run",
        f"source_image={config.source_image}",
        f"target_image={target}",
        *service_account_env_exports(config.credentials_path, project_id=project),
        format_command(["docker", "image", "inspect", config.source_image]),
        format_command(["gcloud", "artifacts", "repositories", "describe", config.repository, "--location", config.location, "--project", project]),
    ]
    if config.create_repository:
        lines.append(
            format_command(
                [
                    "gcloud",
                    "artifacts",
                    "repositories",
                    "create",
                    config.repository,
                    "--repository-format=docker",
                    "--location",
                    config.location,
                    "--project",
                    project,
                    "--description",
                    "w8-biayn DOMDiff reward images",
                ]
            )
        )
    lines.extend(
        [
            f"{format_command(['docker', 'login', f'https://{host}', '-u', '_json_key', '--password-stdin'])} < {shell_quote(config.credentials_path)}",
            format_command(["docker", "tag", config.source_image, target]),
            format_command(["docker", "push", target]),
        ]
    )
    return "\n".join(lines)


def push_local_image_to_artifact_registry(config: ArtifactImageConfig, *, dry_run: bool = False) -> str:
    project = config.resolved_project_id
    tag = resolved_artifact_tag(config, dry_run=dry_run)
    target = artifact_image_uri(
        project_id=project,
        location=config.location,
        repository=config.repository,
        image_name=config.image_name,
        tag=tag,
    )
    if dry_run:
        return target
    local_docker_image_id(config.source_image)
    env = service_account_env(
        config.credentials_path,
        project_id=project,
        cloud_sdk_config_dir=config.gcloud_config_dir,
    )
    Path(config.gcloud_config_dir).mkdir(parents=True, exist_ok=True)
    describe = _run_local(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "describe",
            config.repository,
            "--location",
            config.location,
            "--project",
            project,
        ],
        env=env,
        timeout=120,
        check=False,
    )
    if describe.returncode != 0:
        if not config.create_repository:
            raise DomdiffError(
                f"Artifact Registry repository {config.repository!r} was not found in {config.location}, "
                "and --no-create-repository was set."
            )
        _run_local(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                config.repository,
                "--repository-format=docker",
                "--location",
                config.location,
                "--project",
                project,
                "--description",
                "w8-biayn DOMDiff reward images",
            ],
            env=env,
            timeout=180,
        )
    credentials_path = Path(config.credentials_path)
    try:
        credential_text = credentials_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DomdiffError(f"Could not read GCP credentials for Artifact Registry login: {credentials_path}") from exc
    _run_local(
        ["docker", "login", f"https://{artifact_registry_host(config.location)}", "-u", "_json_key", "--password-stdin"],
        input_text=credential_text,
        timeout=120,
    )
    _run_local(["docker", "tag", config.source_image, target], timeout=600)
    _run_local(["docker", "push", target], timeout=3600)
    return target


class LocalDomdiffHost:
    def __init__(self, config: LocalDomdiffConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.state = LocalDomdiffState(
            run_id=config.run_id,
            image=config.image,
            container_name=config.container_name,
            health_url=f"http://127.0.0.1:{config.host_health_port}/health",
            cdp_http_url=f"http://127.0.0.1:{config.host_cdp_port}",
            local_cdp_url=f"ws://localhost:{config.host_cdp_port}",
            reward_url=f"http://{config.reward_host}:{config.reward_port}",
            reward_log_path=str(run_dir / "local-chromiumrl-service.log"),
            chromiumrl_tunnel_log_path=str(run_dir / "local-cloudflared-chromiumrl.log"),
            cdp_tunnel_log_path=str(run_dir / "local-cloudflared-cdp.log"),
        )

    @classmethod
    def from_state(cls, config: LocalDomdiffConfig, state: LocalDomdiffState, run_dir: Path) -> "LocalDomdiffHost":
        host = cls(config, run_dir)
        host.state = state
        return host

    def _state_path(self) -> Path:
        return self.state.write(self.run_dir)

    def preflight(self) -> None:
        missing = [tool for tool in ("docker", "uv") if not shutil.which(tool)]
        if missing:
            raise DomdiffError(
                "Missing local DOMDiff prerequisites: "
                + ", ".join(missing)
                + ". Run `./scripts/bootstrap.sh` and rerun `w8-biayn doctor --domdiff`."
            )
        cloudflared = find_cloudflared()
        if not cloudflared:
            raise DomdiffError("cloudflared was not found. Run `./scripts/bootstrap.sh` or set CLOUDFLARED_BIN.")
        self.state.cloudflared_path = str(cloudflared)
        if not Path("/dev/kvm").exists():
            raise DomdiffError("Local DOMDiff requires /dev/kvm. Enable KVM/nested virtualization on this host.")
        local_docker_image_id(self.config.image)
        _run_local(["docker", "version"], timeout=120)
        self._state_path()

    def start_container(self) -> None:
        running = self._container_running()
        if running:
            if not self.config.reuse_container:
                _run_local(["docker", "rm", "-f", self.config.container_name], timeout=180, check=False)
            else:
                self.state.started_container = False
                self.state.status = "container-reused"
                self._state_path()
                return
        else:
            _run_local(["docker", "rm", "-f", self.config.container_name], timeout=180, check=False)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.config.container_name,
            "--privileged",
            "--user",
            "root",
            "--device=/dev/kvm:/dev/kvm:rwm",
            "--device-cgroup-rule=c 10:232 rwm",
            "--cap-add=ALL",
            "--security-opt",
            "seccomp=unconfined",
            "--security-opt",
            "apparmor=unconfined",
            "-p",
            f"{self.config.host_health_port}:{self.config.container_health_port}",
            "-p",
            f"{self.config.host_cdp_port}:{self.config.container_cdp_port}",
            "-e",
            "BROWSER=wootzapp",
            "-e",
            "CDP_SOCKET_NAME=wootzapp_world_devtools",
            "-e",
            "WOOTZAPP_PACKAGE=com.wootzapp.web",
            "-e",
            "EMULATOR_SNAPSHOT_MODE=off",
            "-e",
            "EMULATOR_SNAPSHOT_NAME=w8rl_golden",
            self.config.image,
        ]
        result = _run_local(command, timeout=240)
        (self.run_dir / "local-docker-run.log").write_text(result.stdout, encoding="utf-8", errors="replace")
        self.state.started_container = True
        self.state.status = "container-running"
        self._state_path()

    def wait_container(self) -> None:
        wait_http_json(self.state.health_url, timeout_s=self.config.verify_timeout_s, expected_key="status")
        wait_http_json(
            self.state.cdp_http_url.rstrip("/") + "/json/version",
            timeout_s=self.config.verify_timeout_s,
            expected_key="webSocketDebuggerUrl",
        )
        self.state.status = "cdp-ready"
        self._state_path()

    def start_reward_adapter(self) -> None:
        if self._local_reward_ready(timeout_s=4):
            self.state.status = "reward-reused"
            self._state_path()
            return
        log_path = Path(self.state.reward_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        env = os.environ.copy()
        env["CDP_URL"] = self.state.local_cdp_url
        env["CHROMIUMRL_ALLOW_CDP_FALLBACK"] = "0"
        command = [
            "uv",
            "run",
            "--extra",
            "domdiff",
            "uvicorn",
            "w8_biayn.rewards.chromiumrl_service:app",
            "--host",
            self.config.reward_host,
            "--port",
            str(self.config.reward_port),
        ]
        proc = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        self.state.reward_pid = proc.pid
        self.state.status = "reward-starting"
        self._state_path()
        try:
            wait_http_json(
                self.state.reward_url.rstrip("/") + "/health",
                timeout_s=180,
                expected_key="wootzapp_verified",
            )
        except Exception as exc:
            self._stop_pid(self.state.reward_pid)
            raise DomdiffError(f"Local ChromiumRL reward service did not become healthy. See {log_path}") from exc
        self.state.status = "reward-ready"
        self._state_path()

    def start_tunnels(self) -> LocalDomdiffState:
        self.state.chromiumrl_url = self._start_local_tunnel(
            label="chromiumrl",
            url=self.state.reward_url,
            log_path=Path(self.state.chromiumrl_tunnel_log_path),
            expected_path="/health",
            expected_key="status",
        )
        if self.config.publish_cdp:
            self.state.cdp_quick_url = self._start_local_tunnel(
                label="cdp",
                url=self.state.cdp_http_url,
                log_path=Path(self.state.cdp_tunnel_log_path),
                expected_path="/json/version",
                expected_key="webSocketDebuggerUrl",
                extra_args=["--http-host-header", "localhost"],
            )
            self.state.cdp_url = "wss://" + urllib.parse.urlparse(self.state.cdp_quick_url).netloc
        else:
            self.state.cdp_quick_url = ""
            self.state.cdp_url = ""
        self.state.status = "tunnels-ready"
        self._state_path()
        return self.state

    def verify(self) -> dict[str, Any]:
        self.verify_local_kvm()
        wait_http_json(self.state.health_url, timeout_s=120, expected_key="status")
        wait_http_json(self.state.cdp_http_url.rstrip("/") + "/json/version", timeout_s=120, expected_key="webSocketDebuggerUrl")
        if self.state.cdp_quick_url:
            wait_http_json(
                self.state.cdp_quick_url.rstrip("/") + "/json/version",
                timeout_s=120,
                expected_key="webSocketDebuggerUrl",
            )
        health_url = (self.state.chromiumrl_url or self.state.reward_url).rstrip("/") + "/health"
        health = wait_http_json(health_url, timeout_s=120, expected_key="status")
        reward_base = (self.state.chromiumrl_url or self.state.reward_url).rstrip("/")
        submit = wait_http_json(
            reward_base + "/evaluate_async",
            method="POST",
            payload={"target_url": "https://example.com", "timeout_sec": 120},
            timeout_s=120,
            expected_key="job_id",
        )
        job_id = str(submit.get("job_id") or "")
        if not job_id:
            raise DomdiffError("Reward adapter did not return a job_id")
        deadline = time.time() + self.config.verify_timeout_s
        result: dict[str, Any] = {}
        last_error = ""
        while time.time() < deadline:
            try:
                status = _json_http_request(reward_base + f"/evaluate_jobs/{job_id}", timeout=60)
            except Exception as exc:  # noqa: BLE001 - quick-tunnel DNS can be briefly inconsistent
                last_error = str(exc)
                time.sleep(3)
                continue
            if status.get("status") == "done":
                result = status.get("result") if isinstance(status.get("result"), dict) else {}
                break
            if status.get("status") == "error":
                raise DomdiffError(f"Reward adapter job failed: {status.get('error')}")
            time.sleep(3)
        if not result:
            raise DomdiffError(f"Reward adapter job {job_id} did not finish: {_tail(last_error)}")
        if result.get("error"):
            raise DomdiffError(f"Reward adapter returned error: {result['error']}")
        report = {
            "success": True,
            "mode": "local",
            "health": health,
            "evaluate": result,
            "chromiumrl_url": self.state.chromiumrl_url,
            "cdp_url": self.state.cdp_url,
            "cdp_published": bool(self.state.cdp_url),
        }
        (self.run_dir / "verify.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def verify_local_kvm(self) -> None:
        _run_local(
            [
                "docker",
                "exec",
                self.state.container_name,
                "bash",
                "-lc",
                "set -euo pipefail; test -e /dev/kvm; test -r /dev/kvm; test -w /dev/kvm",
            ],
            timeout=120,
        )

    def collect_logs(self) -> None:
        commands = {
            "local-domdiff-docker.log": ["docker", "logs", "--tail", "400", self.state.container_name],
            "local-docker-ps.txt": ["docker", "ps", "-a", "--format", "{{.Names}} {{.Image}} {{.Status}} {{.Ports}}"],
        }
        for filename, command in commands.items():
            try:
                result = _run_local(command, check=False, timeout=120)
                output = result.stdout
            except Exception as exc:  # noqa: BLE001 - best-effort local log collection
                output = f"failed to collect {filename}: {exc}\n"
            (self.run_dir / filename).write_text(output, encoding="utf-8", errors="replace")

    def cleanup(self, *, keep_container: bool | None = None) -> None:
        self.collect_logs()
        self._stop_pid(self.state.chromiumrl_tunnel_pid)
        self._stop_pid(self.state.cdp_tunnel_pid)
        self._stop_pid(self.state.reward_pid)
        should_keep_container = self.config.keep_container if keep_container is None else keep_container
        if not should_keep_container:
            _run_local(["docker", "rm", "-f", self.state.container_name], timeout=180, check=False)
        self.state.status = "deleted" if not should_keep_container else "local-processes-stopped"
        self._state_path()

    def _container_running(self) -> bool:
        result = _run_local(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.config.container_name],
            timeout=60,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def _local_reward_ready(self, *, timeout_s: int) -> bool:
        try:
            wait_http_json(
                self.state.reward_url.rstrip("/") + "/health",
                timeout_s=timeout_s,
                expected_key="wootzapp_verified",
            )
            return True
        except Exception:
            return False

    def _start_local_tunnel(
        self,
        *,
        label: str,
        url: str,
        log_path: Path,
        expected_path: str,
        expected_key: str,
        extra_args: list[str] | None = None,
    ) -> str:
        cloudflared = self.state.cloudflared_path or str(find_cloudflared() or "cloudflared")
        for attempt in range(1, 4):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("wb")
            command = [
                cloudflared,
                "tunnel",
                "--no-autoupdate",
                "--url",
                url,
                *(extra_args or []),
            ]
            proc = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            log_handle.close()
            if label == "chromiumrl":
                self.state.chromiumrl_tunnel_pid = proc.pid
            else:
                self.state.cdp_tunnel_pid = proc.pid
            self._state_path()
            quick_url = ""
            for _ in range(90):
                quick_url = extract_quick_url(log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else "")
                if quick_url:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(1)
            if quick_url:
                try:
                    wait_http_json(quick_url.rstrip("/") + expected_path, timeout_s=300, expected_key=expected_key)
                    return quick_url
                except DomdiffError:
                    self._stop_pid(proc.pid)
                    if attempt == 3:
                        raise
            else:
                self._stop_pid(proc.pid)
        raise DomdiffError(f"Timed out waiting for local {label} quick tunnel URL")

    @staticmethod
    def _stop_pid(pid: int) -> None:
        if not pid:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                return
        time.sleep(1)
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


class GcpDomdiffHost:
    def __init__(self, config: DomdiffConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = run_dir
        self.project_id = config.resolved_project_id
        self.state = DomdiffState(
            run_id=config.run_id,
            project_id=self.project_id,
            zone=config.zone,
            reward_image=config.reward_image,
            gcloud_config_dir=str(run_dir / "gcloud-config"),
        )

    @classmethod
    def from_state(cls, config: DomdiffConfig, state: DomdiffState, run_dir: Path) -> "GcpDomdiffHost":
        host = cls(config, run_dir)
        host.state = state
        host.project_id = state.project_id
        return host

    def _state_path(self) -> Path:
        return self.state.write(self.run_dir)

    def _gcloud_env(self) -> dict[str, str]:
        return service_account_env(
            self.config.credentials_path,
            project_id=self.project_id,
            cloud_sdk_config_dir=self.state.gcloud_config_dir,
        )

    def _gcloud(
        self,
        name: str,
        args: list[str],
        *,
        timeout: int = 300,
        check: bool = True,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        Path(self.state.gcloud_config_dir).mkdir(parents=True, exist_ok=True)
        result = _run_local(
            ["gcloud", *args],
            env=self._gcloud_env(),
            timeout=timeout,
            check=False,
            input_text=input_text,
        )
        (self.run_dir / f"{name}.log").write_text(result.stdout, encoding="utf-8", errors="replace")
        if check and result.returncode != 0:
            raise DomdiffError(f"{name} failed exit={result.returncode}: {_tail(result.stdout)}")
        return result

    def authenticate(self) -> None:
        Path(self.state.gcloud_config_dir).mkdir(parents=True, exist_ok=True)

    def preflight_permissions(self) -> list[str]:
        self.authenticate()
        try:
            missing = check_project_permissions(
                self.config.credentials_path,
                project_id=self.project_id,
                permissions=REQUIRED_GCP_PERMISSIONS,
            )
        except GcpAuthError as exc:
            raise DomdiffError(str(exc)) from exc
        (self.run_dir / "gcp-permission-preflight.json").write_text(
            json.dumps(
                {
                    "required": REQUIRED_GCP_PERMISSIONS,
                    "missing": missing,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if missing:
            raise DomdiffError("GCP service account lacks Compute permissions: " + ", ".join(missing))
        return missing

    def create(self) -> DomdiffState:
        self.authenticate()
        base_name = safe_name(f"w8-domdiff-{self.config.run_id}", max_len=55)
        self.state.instance_name = base_name
        self.state.firewall_rule = safe_name(f"{base_name}-ssh", max_len=63)
        self.state.network_tag = safe_name(f"{base_name}-ssh", max_len=63)
        self.state.key_path = str(self.run_dir / f"{base_name}.ssh")
        self._create_ssh_key()
        self._create_firewall_rule()
        pubkey = Path(self.state.key_path).with_suffix(".ssh.pub").read_text(encoding="utf-8").strip()
        labels = ",".join(
            [
                "project=w8-biayn",
                "purpose=domdiff-reward-host",
                "managed_by=w8-biayn",
                f"run_id={safe_name(self.config.run_id, max_len=63)}",
            ]
        )
        self._gcloud(
            "gcp_create_instance",
            [
                "compute",
                "instances",
                "create",
                self.state.instance_name,
                "--project",
                self.project_id,
                "--zone",
                self.config.zone,
                "--machine-type",
                self.config.machine_type,
                "--image-family",
                self.config.image_family,
                "--image-project",
                self.config.image_project,
                "--boot-disk-size",
                f"{self.config.volume_gb}GB",
                "--boot-disk-type",
                self.config.boot_disk_type,
                "--enable-nested-virtualization",
                "--min-cpu-platform",
                self.config.min_cpu_platform,
                "--tags",
                self.state.network_tag,
                "--labels",
                labels,
                "--metadata",
                f"ssh-keys=w8:{pubkey}",
                "--quiet",
            ],
            timeout=900,
        )
        self.state.public_ip = self._describe_public_ip()
        self.state.status = "instance-running"
        self._state_path()
        self._wait_ssh()
        return self.state

    def provision(self) -> DomdiffState:
        self.ssh(
            "install_base",
            r"""
set -euxo pipefail
if command -v cloud-init >/dev/null 2>&1; then
  sudo cloud-init status --wait >/tmp/w8-cloud-init.log 2>&1 || cat /tmp/w8-cloud-init.log
fi
for attempt in 1 2 3 4 5; do
  if sudo apt-get update >/tmp/w8-apt.log 2>&1; then break; fi
  cat /tmp/w8-apt.log
  if [ "$attempt" = "5" ]; then exit 1; fi
  sleep $((attempt * 10))
done
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io curl tar gzip util-linux python3 ca-certificates
if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cpu-checker; then
  sudo tee /usr/local/bin/kvm-ok >/dev/null <<'KVM_OK'
#!/usr/bin/env bash
set -euo pipefail
test -e /dev/kvm
test -r /dev/kvm
test -w /dev/kvm
grep -Eq 'vmx|svm' /proc/cpuinfo
echo "KVM acceleration can be used"
KVM_OK
  sudo chmod 0755 /usr/local/bin/kvm-ok
fi
sudo systemctl enable --now docker
sudo chmod 666 /var/run/docker.sock
if ! command -v cloudflared >/dev/null 2>&1; then
  curl -fsSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  sudo install -m 0755 /tmp/cloudflared /usr/local/bin/cloudflared
fi
docker version
cloudflared --version
ls -l /dev/kvm
kvm-ok
""",
            timeout=900,
        )
        token = os.environ.get("GITHUB_PAT_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
        if token and self.config.reward_image.startswith("ghcr.io/"):
            self.ssh(
                "ghcr_login",
                "cat > /tmp/ghcr-token && docker login ghcr.io -u wootzapp --password-stdin < /tmp/ghcr-token && rm -f /tmp/ghcr-token",
                input_text=token,
                timeout=120,
            )
        if is_artifact_registry_image(self.config.reward_image):
            self._remote_artifact_registry_login()
        self.ssh("docker_pull_domdiff", f"docker pull {shell_quote(self.config.reward_image)}", timeout=1200)
        self.ssh(
            "start_domdiff_container",
            f"""
set -euxo pipefail
docker rm -f android-world-domdiff >/dev/null 2>&1 || true
docker run -d --name android-world-domdiff \\
  --privileged \\
  --user root \\
  --network=host \\
  --device=/dev/kvm:/dev/kvm:rwm \\
  --device-cgroup-rule='c 10:232 rwm' \\
  --cap-add=ALL \\
  --security-opt seccomp=unconfined \\
  --security-opt apparmor=unconfined \\
  -e BROWSER=wootzapp \\
  -e CDP_SOCKET_NAME=wootzapp_world_devtools \\
  -e WOOTZAPP_PACKAGE=com.wootzapp.web \\
  -e EMULATOR_SNAPSHOT_MODE=off \\
  {shell_quote(self.config.reward_image)}
""",
            timeout=180,
        )
        self.verify_remote_kvm()
        self.wait_remote_cdp()
        self.copy_reward_adapter()
        self.start_reward_adapter()
        self.state.status = "reward-ready"
        self._state_path()
        return self.state

    def verify_remote_kvm(self) -> None:
        self.ssh(
            "verify_host_kvm",
            "set -euxo pipefail; ls -l /dev/kvm; grep -Eq 'vmx|svm' /proc/cpuinfo; kvm-ok",
            timeout=120,
        )
        self.ssh(
            "verify_container_kvm",
            r"""
set -euxo pipefail
docker exec android-world-domdiff bash -lc '
  ls -l /dev/kvm
  if command -v kvm-ok >/dev/null 2>&1; then kvm-ok; fi
  /opt/py311/bin/python - <<'"'"'PY'"'"'
import os, fcntl
fd = os.open("/dev/kvm", os.O_RDWR)
try:
    print("KVM API version", fcntl.ioctl(fd, 0xAE00, 0))
finally:
    os.close(fd)
PY
'
""",
            timeout=300,
        )

    def _remote_artifact_registry_login(self) -> None:
        registry = image_registry(self.config.reward_image)
        if not registry:
            return
        credentials = Path(self.config.credentials_path)
        try:
            credential_text = credentials.read_text(encoding="utf-8")
        except OSError as exc:
            raise DomdiffError(f"Could not read GCP credentials for remote Artifact Registry login: {credentials}") from exc
        self.ssh(
            "artifact_registry_login",
            f"""
set -euo pipefail
tmp="$(mktemp /tmp/w8-gcp-key.XXXXXX.json)"
trap 'rm -f "$tmp"' EXIT
cat > "$tmp"
chmod 0600 "$tmp"
docker login {shell_quote("https://" + registry)} -u _json_key --password-stdin < "$tmp"
rm -f "$tmp"
""",
            input_text=credential_text,
            timeout=180,
        )

    def wait_remote_cdp(self) -> None:
        output = self.ssh(
            "wait_cdp",
            r"""
set -euo pipefail
for i in $(seq 1 600); do
  body="$(curl -sS --max-time 10 http://127.0.0.1:9224/json/version 2>/tmp/w8-cdp.err || true)"
  if [[ "$body" == *'"Browser"'* ]]; then
    printf '%s\n' "$body"
    exit 0
  fi
  cat /tmp/w8-cdp.err || true
  sleep 2
done
docker logs --tail 200 android-world-domdiff || true
exit 1
""",
            timeout=1320,
            check=False,
        )
        if '"Browser"' not in output:
            raise DomdiffError(f"CDP did not become ready: {_tail(output)}")

    def copy_reward_adapter(self) -> None:
        archive = self.run_dir / "w8_biayn_reward_adapter.tar.gz"
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "w8_biayn" / "__init__.py",
            root / "w8_biayn" / "rewards" / "__init__.py",
            root / "w8_biayn" / "rewards" / "chromiumrl_adapter.py",
        ]
        with tarfile.open(archive, "w:gz") as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(root)))
        self.scp_to(archive, "/tmp/w8_biayn_reward_adapter.tar.gz")
        self.ssh(
            "copy_reward_adapter",
            "docker cp /tmp/w8_biayn_reward_adapter.tar.gz android-world-domdiff:/tmp/w8_biayn_reward_adapter.tar.gz && "
            "docker exec android-world-domdiff bash -lc 'cd /app && tar xzf /tmp/w8_biayn_reward_adapter.tar.gz'",
            timeout=120,
        )

    def start_reward_adapter(self) -> None:
        self.ssh(
            "start_reward_adapter",
            r"""
set -euxo pipefail
docker exec android-world-domdiff bash -lc '
  set -euxo pipefail
  /opt/py311/bin/pip install -q "fastapi>=0.110" "uvicorn>=0.29" "websocket-client>=1.7" "pydantic>=2.7"
  if [ -f /tmp/w8-biayn-reward.pid ]; then
    kill "$(cat /tmp/w8-biayn-reward.pid)" >/dev/null 2>&1 || true
  fi
  rm -f /tmp/w8-biayn-reward.log /tmp/w8-biayn-reward.pid
  CDP_URL=ws://127.0.0.1:9224 PYTHONPATH=/app \
    nohup /opt/py311/bin/python -m uvicorn w8_biayn.rewards.chromiumrl_adapter:app \
      --host 127.0.0.1 --port 8080 >/tmp/w8-biayn-reward.log 2>&1 &
  echo $! >/tmp/w8-biayn-reward.pid
'
for i in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8080/health; then
    exit 0
  fi
  sleep 2
done
docker exec android-world-domdiff bash -lc 'cat /tmp/w8-biayn-reward.log || true'
exit 1
""",
            timeout=480,
        )

    def start_tunnels(self) -> DomdiffState:
        self.state.chromiumrl_url = self._start_remote_tunnel(
            label="chromiumrl",
            url="http://127.0.0.1:8080",
            pid_file="/tmp/w8-cloudflared-chromiumrl.pid",
            log_file="/tmp/w8-cloudflared-chromiumrl.log",
            expected_path="/health",
            expected_key="status",
        )
        cdp_quick_url = self._start_remote_tunnel(
            label="cdp",
            url="http://127.0.0.1:9224",
            pid_file="/tmp/w8-cloudflared-cdp.pid",
            log_file="/tmp/w8-cloudflared-cdp.log",
            extra_args="--http-host-header localhost",
            expected_path="/json/version",
            expected_key="webSocketDebuggerUrl",
        )
        self.state.cdp_quick_url = cdp_quick_url
        self.state.cdp_url = "wss://" + urllib.parse.urlparse(cdp_quick_url).netloc
        self.state.status = "tunnels-ready"
        self._state_path()
        return self.state

    def _start_remote_tunnel(
        self,
        *,
        label: str,
        url: str,
        pid_file: str,
        log_file: str,
        expected_path: str,
        expected_key: str,
        extra_args: str = "",
    ) -> str:
        for attempt in range(1, 4):
            self.ssh(
                f"start_{label}_quick_tunnel",
                f"""
set -euxo pipefail
if [ -f {shell_quote(pid_file)} ]; then
  kill "$(cat {shell_quote(pid_file)})" >/dev/null 2>&1 || true
fi
rm -f {shell_quote(log_file)}
nohup cloudflared tunnel --no-autoupdate --url {shell_quote(url)} {extra_args} >{shell_quote(log_file)} 2>&1 &
echo $! >{shell_quote(pid_file)}
""",
                timeout=60,
            )
            quick_url = ""
            for _ in range(90):
                text = self.ssh(
                    f"read_{label}_quick_tunnel_log",
                    f"cat {shell_quote(log_file)} 2>/dev/null || true",
                    check=False,
                    timeout=30,
                )
                quick_url = extract_quick_url(text)
                if quick_url:
                    break
                time.sleep(1)
            if not quick_url:
                continue
            try:
                wait_http_json(quick_url.rstrip("/") + expected_path, timeout_s=300, expected_key=expected_key)
                return quick_url
            except DomdiffError:
                self._stop_remote_process(pid_file)
                if attempt == 3:
                    raise
        raise DomdiffError(f"Timed out waiting for {label} quick tunnel URL")

    def verify(self) -> dict[str, Any]:
        self.verify_remote_kvm()
        self.wait_remote_cdp()
        health = wait_http_json(self.state.chromiumrl_url.rstrip("/") + "/health", timeout_s=120, expected_key="status")
        submit = wait_http_json(
            self.state.chromiumrl_url.rstrip("/") + "/evaluate_async",
            method="POST",
            payload={"target_url": "https://example.com", "timeout_sec": 120},
            timeout_s=120,
            expected_key="job_id",
        )
        job_id = str(submit.get("job_id") or "")
        if not job_id:
            raise DomdiffError("Reward adapter did not return a job_id")
        deadline = time.time() + 420
        result: dict[str, Any] = {}
        last_error = ""
        while time.time() < deadline:
            try:
                status = _json_http_request(
                    self.state.chromiumrl_url.rstrip("/") + f"/evaluate_jobs/{job_id}",
                    timeout=60,
                )
            except Exception as exc:  # noqa: BLE001 - quick-tunnel DNS can be briefly inconsistent
                last_error = str(exc)
                time.sleep(3)
                continue
            if status.get("status") == "done":
                result = status.get("result") if isinstance(status.get("result"), dict) else {}
                break
            if status.get("status") == "error":
                raise DomdiffError(f"Reward adapter job failed: {status.get('error')}")
            time.sleep(3)
        if not result:
            raise DomdiffError(f"Reward adapter job {job_id} did not finish: {_tail(last_error)}")
        if result.get("error"):
            raise DomdiffError(f"Reward adapter returned error: {result['error']}")
        report = {"success": True, "health": health, "evaluate": result}
        (self.run_dir / "verify.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def collect_logs(self) -> None:
        if not self.state.public_ip:
            return
        commands = {
            "gcp-domdiff-docker.log": "docker logs --tail 400 android-world-domdiff 2>&1 || true",
            "gcp-reward-adapter.log": "docker exec android-world-domdiff bash -lc 'cat /tmp/w8-biayn-reward.log 2>/dev/null || true'",
            "gcp-cloudflared-chromiumrl.log": "cat /tmp/w8-cloudflared-chromiumrl.log 2>/dev/null || true",
            "gcp-cloudflared-cdp.log": "cat /tmp/w8-cloudflared-cdp.log 2>/dev/null || true",
            "gcp-docker-ps.txt": "docker ps -a --format '{{.Names}} {{.Status}}' || true",
        }
        for filename, command in commands.items():
            try:
                output = self.ssh(f"collect_{filename}", command, check=False, timeout=60)
            except Exception as exc:  # noqa: BLE001
                output = f"failed to collect {filename}: {exc}\n"
            (self.run_dir / filename).write_text(output, encoding="utf-8", errors="replace")
        if self.state.instance_name:
            serial = self._gcloud(
                "gcp_serial_output",
                [
                    "compute",
                    "instances",
                    "get-serial-port-output",
                    self.state.instance_name,
                    "--project",
                    self.project_id,
                    "--zone",
                    self.config.zone,
                    "--port",
                    "1",
                    "--quiet",
                ],
                check=False,
                timeout=120,
            ).stdout
            (self.run_dir / "gcp-serial-output.log").write_text(serial, encoding="utf-8", errors="replace")

    def cleanup(self) -> None:
        self.collect_logs()
        if self.config.keep:
            return
        if self.state.public_ip:
            self.ssh(
                "remote_cleanup",
                r"""
set +e
for pid in /tmp/w8-cloudflared-chromiumrl.pid /tmp/w8-cloudflared-cdp.pid; do
  if [ -f "$pid" ]; then kill "$(cat "$pid")" >/dev/null 2>&1 || true; fi
done
docker rm -f android-world-domdiff >/dev/null 2>&1 || true
""",
                check=False,
                timeout=120,
            )
        if self.state.instance_name:
            self._gcloud(
                "gcp_delete_instance",
                [
                    "compute",
                    "instances",
                    "delete",
                    self.state.instance_name,
                    "--project",
                    self.project_id,
                    "--zone",
                    self.config.zone,
                    "--quiet",
                ],
                check=False,
                timeout=900,
            )
        if self.state.firewall_rule:
            self._gcloud(
                "gcp_delete_firewall",
                [
                    "compute",
                    "firewall-rules",
                    "delete",
                    self.state.firewall_rule,
                    "--project",
                    self.project_id,
                    "--quiet",
                ],
                check=False,
                timeout=300,
            )
        if self.state.key_path:
            key = Path(self.state.key_path)
            for path in (key, key.with_suffix(".ssh.pub")):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.state.status = "deleted"
        self._state_path()

    def ssh(
        self,
        name: str,
        command: str,
        *,
        timeout: int = 120,
        check: bool = True,
        input_text: str | None = None,
    ) -> str:
        if not self.state.key_path or not self.state.public_ip:
            raise DomdiffError("DOMDiff SSH state is incomplete")
        cmd = [
            "ssh",
            "-i",
            self.state.key_path,
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            f"w8@{self.state.public_ip}",
            f"bash -c {shell_quote(command)}",
        ]
        result = _run_local(cmd, timeout=timeout, check=False, input_text=input_text)
        (self.run_dir / f"{name}.log").write_text(result.stdout, encoding="utf-8", errors="replace")
        if check and result.returncode != 0:
            raise DomdiffError(f"{name} failed exit={result.returncode}: {_tail(result.stdout)}")
        return result.stdout

    def scp_to(self, local_path: Path, remote_path: str) -> None:
        _run_local(
            [
                "scp",
                "-i",
                self.state.key_path,
                "-o",
                "BatchMode=yes",
                "-o",
                "IdentitiesOnly=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                str(local_path),
                f"w8@{self.state.public_ip}:{remote_path}",
            ],
            timeout=180,
        )

    def _create_ssh_key(self) -> None:
        key_path = Path(self.state.key_path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        _run_local(
            [
                "ssh-keygen",
                "-t",
                "rsa",
                "-b",
                "4096",
                "-N",
                "",
                "-C",
                f"w8@{self.state.instance_name}",
                "-f",
                str(key_path),
            ],
            timeout=120,
        )
        key_path.chmod(0o600)
        key_path.with_suffix(".ssh.pub").chmod(0o644)

    def _create_firewall_rule(self) -> None:
        self._gcloud(
            "gcp_create_firewall",
            [
                "compute",
                "firewall-rules",
                "create",
                self.state.firewall_rule,
                "--project",
                self.project_id,
                "--network",
                self.config.network,
                "--direction",
                "INGRESS",
                "--priority",
                "1000",
                "--action",
                "ALLOW",
                "--rules",
                "tcp:22",
                "--source-ranges",
                f"{_public_ip()}/32",
                "--target-tags",
                self.state.network_tag,
                "--description",
                f"w8-biayn DOMDiff SSH for {self.config.run_id}",
                "--quiet",
            ],
            timeout=300,
        )

    def _describe_public_ip(self) -> str:
        result = self._gcloud(
            "gcp_describe_instance_ip",
            [
                "compute",
                "instances",
                "describe",
                self.state.instance_name,
                "--project",
                self.project_id,
                "--zone",
                self.config.zone,
                "--format",
                "get(networkInterfaces[0].accessConfigs[0].natIP)",
            ],
            timeout=120,
        )
        ip = result.stdout.strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip):
            raise DomdiffError(f"GCP instance did not get a valid external IPv4 address: {ip!r}")
        return ip

    def _wait_ssh(self) -> None:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if "ok" in self.ssh("wait_ssh", "echo ok", timeout=20, check=False):
                    return
            except Exception:
                pass
            time.sleep(5)
        raise DomdiffError("Timed out waiting for SSH")

    def _stop_remote_process(self, pid_file: str) -> None:
        self.ssh(
            f"stop_{Path(pid_file).stem}",
            f"set +e; if [ -f {shell_quote(pid_file)} ]; then kill \"$(cat {shell_quote(pid_file)})\" >/dev/null 2>&1 || true; fi",
            check=False,
            timeout=30,
        )


def create_host(config: DomdiffConfig) -> GcpDomdiffHost:
    return GcpDomdiffHost(config, run_dir_for(config.run_id))


def create_host_from_state(config: DomdiffConfig, run_dir: Path) -> GcpDomdiffHost:
    state = DomdiffState.read(run_dir)
    return GcpDomdiffHost.from_state(config, state, run_dir)


def create_local_host(config: LocalDomdiffConfig) -> LocalDomdiffHost:
    return LocalDomdiffHost(config, local_run_dir_for(config.run_id))


def create_local_host_from_state(config: LocalDomdiffConfig, run_dir: Path) -> LocalDomdiffHost:
    state = LocalDomdiffState.read(run_dir)
    return LocalDomdiffHost.from_state(config, state, run_dir)


def up(config: DomdiffConfig) -> DomdiffState:
    run_dir = run_dir_for(config.run_id)
    host = GcpDomdiffHost(config, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        host.preflight_permissions()
        host.create()
        host.provision()
        return host.start_tunnels()
    except Exception:
        if host.state.instance_name and not config.keep:
            host.cleanup()
        raise


def local_up(config: LocalDomdiffConfig) -> LocalDomdiffState:
    run_dir = local_run_dir_for(config.run_id)
    host = LocalDomdiffHost(config, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        host.preflight()
        host.start_container()
        host.wait_container()
        host.start_reward_adapter()
        return host.start_tunnels()
    except Exception:
        if not config.keep_container:
            try:
                host.cleanup()
            except Exception:
                pass
        raise


def dry_run_plan(config: DomdiffConfig) -> str:
    project = config.project_id or "<project-from-service-account>"
    instance = safe_name(f"w8-domdiff-{config.run_id}", max_len=55)
    firewall = safe_name(f"{instance}-ssh", max_len=63)
    lines = [
        "# DOMDiff dry run",
        f"run_id={config.run_id}",
        f"reward_image={config.reward_image}",
        *service_account_env_exports(config.credentials_path, project_id=project),
        format_command(
            [
                "gcloud",
                "compute",
                "firewall-rules",
                "create",
                firewall,
                "--project",
                project,
                "--network",
                config.network,
                "--rules",
                "tcp:22",
                "--source-ranges",
                "<current-public-ip>/32",
            ]
        ),
        format_command(
            [
                "gcloud",
                "compute",
                "instances",
                "create",
                instance,
                "--project",
                project,
                "--zone",
                config.zone,
                "--machine-type",
                config.machine_type,
                "--enable-nested-virtualization",
                "--min-cpu-platform",
                config.min_cpu_platform,
            ]
        ),
        "# remote: install Docker/cloudflared",
        "# remote: if reward_image is Artifact Registry, docker login with the service-account JSON",
        "# remote: pull image, run privileged android-world-domdiff, start reward adapter, publish tunnels",
    ]
    return "\n".join(lines)


def local_dry_run_plan(config: LocalDomdiffConfig) -> str:
    cloudflared = find_cloudflared()
    cloudflared_bin = str(cloudflared) if cloudflared else "<cloudflared>"
    health_url = f"http://127.0.0.1:{config.host_health_port}/health"
    cdp_http_url = f"http://127.0.0.1:{config.host_cdp_port}"
    reward_url = f"http://{config.reward_host}:{config.reward_port}"
    lines = [
        "# Local DOMDiff dry run",
        f"run_id={config.run_id}",
        f"image={config.image}",
        f"container_name={config.container_name}",
        format_command(["docker", "image", "inspect", config.image]),
        "test -e /dev/kvm && test -r /dev/kvm && test -w /dev/kvm",
        format_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                config.container_name,
                "--privileged",
                "--user",
                "root",
                "--device=/dev/kvm:/dev/kvm:rwm",
                "--device-cgroup-rule=c 10:232 rwm",
                "--cap-add=ALL",
                "--security-opt",
                "seccomp=unconfined",
                "--security-opt",
                "apparmor=unconfined",
                "-p",
                f"{config.host_health_port}:{config.container_health_port}",
                "-p",
                f"{config.host_cdp_port}:{config.container_cdp_port}",
                "-e",
                "BROWSER=wootzapp",
                "-e",
                "CDP_SOCKET_NAME=wootzapp_world_devtools",
                "-e",
                "WOOTZAPP_PACKAGE=com.wootzapp.web",
                "-e",
                "EMULATOR_SNAPSHOT_MODE=off",
                "-e",
                "EMULATOR_SNAPSHOT_NAME=w8rl_golden",
                config.image,
            ]
        ),
        f"curl -fsS {health_url}",
        f"curl -fsS {cdp_http_url}/json/version",
        (
            "CDP_URL="
            f"ws://localhost:{config.host_cdp_port} "
            "CHROMIUMRL_ALLOW_CDP_FALLBACK=0 "
            + format_command(
                [
                    "uv",
                    "run",
                    "--extra",
                    "domdiff",
                    "uvicorn",
                    "w8_biayn.rewards.chromiumrl_service:app",
                    "--host",
                    config.reward_host,
                    "--port",
                    str(config.reward_port),
                ]
            )
        ),
        format_command([cloudflared_bin, "tunnel", "--no-autoupdate", "--url", reward_url]),
        "# launch: pass chromiumrl_url=https://<local-reward-tunnel>; CDP stays local by default",
    ]
    if config.publish_cdp:
        lines.extend(
            [
                format_command(
                    [
                        cloudflared_bin,
                        "tunnel",
                        "--no-autoupdate",
                        "--url",
                        cdp_http_url,
                        "--http-host-header",
                        "localhost",
                    ]
                ),
                "# launch: pass cdp_url=wss://<local-cdp-tunnel> only for explicit CDP debugging",
            ]
        )
    return "\n".join(lines)
