"""Docker-first setup helpers for the pinned SLIME upstream sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from .doctor import slime_root

DEFAULT_SLIME_IMAGE = "slimerl/slime:latest"
DEFAULT_CONTAINER_NAME = "w8-slime"
DEFAULT_LAUNCHER_RELATIVE = Path(".w8-biayn/slime/run-container.sh")
DEFAULT_BOOTSTRAP_RELATIVE = Path(".w8-biayn/slime/bootstrap-inside-container.sh")


@dataclass(frozen=True)
class SlimeSetupPlan:
    repo_root: Path
    upstream_root: Path
    repo_mount: str
    image: str
    container_name: str
    launcher_path: Path
    bootstrap_path: Path

    @property
    def launcher_command(self) -> str:
        return str(self.launcher_path)

    @property
    def bootstrap_command(self) -> str:
        return f"bash {shlex.quote(self.repo_mount + '/' + self.bootstrap_path.relative_to(self.repo_root).as_posix())}"


def build_slime_setup_plan(
    repo_root: str | Path = ".",
    *,
    image: str = DEFAULT_SLIME_IMAGE,
    container_name: str = DEFAULT_CONTAINER_NAME,
    launcher_relative: Path = DEFAULT_LAUNCHER_RELATIVE,
    bootstrap_relative: Path = DEFAULT_BOOTSTRAP_RELATIVE,
) -> SlimeSetupPlan:
    repo_path = Path(repo_root).resolve()
    repo_mount = f"/workspace/{repo_path.name}"
    return SlimeSetupPlan(
        repo_root=repo_path,
        upstream_root=slime_root(repo_path),
        repo_mount=repo_mount,
        image=image,
        container_name=container_name,
        launcher_path=repo_path / launcher_relative,
        bootstrap_path=repo_path / bootstrap_relative,
    )


def _launcher_contents(plan: SlimeSetupPlan) -> str:
    repo_root = shlex.quote(str(plan.repo_root))
    image = shlex.quote(plan.image)
    container_name = shlex.quote(plan.container_name)
    repo_mount = shlex.quote(plan.repo_mount)
    bootstrap_rel = plan.bootstrap_path.relative_to(plan.repo_root).as_posix()
    bootstrap_target = shlex.quote(f"{plan.repo_mount}/{bootstrap_rel}")
    return f"""#!/usr/bin/env bash
set -euo pipefail

docker pull {image}

exec docker run --rm --gpus all --ipc=host --shm-size=16g \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  --name {container_name} \\
  -v {repo_root}:{repo_mount} \\
  -v "${{HOST_MODELS_DIR:-$HOME/models}}":/root/models \\
  -v /var/run/docker.sock:/var/run/docker.sock \\
  -e HOST_REPO_ROOT={repo_mount} \\
  -it {image} /bin/bash -lc "echo repo_mount={repo_mount}; echo bootstrap={bootstrap_target}; bash {bootstrap_target}; exec /bin/bash"
"""


def _bootstrap_contents(plan: SlimeSetupPlan) -> str:
    return """#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/root/Megatron-LM${PYTHONPATH:+:${PYTHONPATH}}

cd /root/slime
git pull
pip install -e . --no-deps
python train.py --help >/tmp/slime-train-help.txt
sed -n '1,80p' /tmp/slime-train-help.txt
"""


def write_slime_setup_files(plan: SlimeSetupPlan, *, force: bool = False) -> tuple[Path, Path]:
    launcher_path = plan.launcher_path
    bootstrap_path = plan.bootstrap_path
    for path, contents in (
        (launcher_path, _launcher_contents(plan)),
        (bootstrap_path, _bootstrap_contents(plan)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            continue
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o755)
    return launcher_path, bootstrap_path
