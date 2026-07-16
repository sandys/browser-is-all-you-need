from __future__ import annotations

import os
import runpy
import subprocess
import tarfile
from pathlib import Path


SCRIPTS = (
    Path("examples/sft.sh"),
    Path("examples/grpo.sh"),
    Path("scripts/train_sft.sh"),
    Path("scripts/train_grpo.sh"),
    Path("scripts/convert_checkpoint.sh"),
)


def test_miles_h100_launchers_are_executable_and_syntax_valid() -> None:
    for script in SCRIPTS:
        assert script.is_file(), script
        assert os.access(script, os.X_OK), script
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_miles_h100_launchers_pin_the_measured_parallelism() -> None:
    for script in (Path("examples/sft.sh"), Path("examples/grpo.sh")):
        text = script.read_text(encoding="utf-8")
        assert 'MILES_GPUS_PER_NODE="${MILES_GPUS_PER_NODE:-8}"' in text
        assert 'MILES_TENSOR_MODEL_PARALLEL_SIZE="${MILES_TENSOR_MODEL_PARALLEL_SIZE:-4}"' in text
        assert 'MILES_PIPELINE_MODEL_PARALLEL_SIZE="${MILES_PIPELINE_MODEL_PARALLEL_SIZE:-1}"' in text
        assert 'MILES_EXPERT_MODEL_PARALLEL_SIZE="${MILES_EXPERT_MODEL_PARALLEL_SIZE:-8}"' in text
        assert 'MILES_MAX_TOKENS_PER_GPU="${MILES_MAX_TOKENS_PER_GPU:-16384}"' in text
        assert 'MILES_MOE_TOKEN_DISPATCHER_TYPE="${MILES_MOE_TOKEN_DISPATCHER_TYPE:-flex}"' in text
        assert 'MILES_MOE_ENABLE_DEEPEP="${MILES_MOE_ENABLE_DEEPEP:-1}"' in text
        assert 'MILES_SGLANG_ENABLE_DP_ATTENTION="${MILES_SGLANG_ENABLE_DP_ATTENTION:-1}"' in text
        assert 'MILES_SGLANG_DP_SIZE="${MILES_SGLANG_DP_SIZE:-8}"' in text

    grpo = Path("examples/grpo.sh").read_text(encoding="utf-8")
    assert 'MILES_N_SAMPLES_PER_PROMPT="${MILES_N_SAMPLES_PER_PROMPT:-8}"' in grpo
    assert 'MILES_GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-256}"' in grpo
    assert 'MILES_SGLANG_ATTENTION_BACKEND="${MILES_SGLANG_ATTENTION_BACKEND:-flashinfer}"' in grpo


def test_miles_h100_package_is_in_the_wheel() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["src/w8_biayn", "src/glm47_posttraining"]' in pyproject


def test_data_asset_extracts_the_verified_task_bundle(tmp_path: Path) -> None:
    module = runpy.run_path("scripts/download_assets.py")
    extract_task_archive = module["_extract_task_archive"]

    root = tmp_path / "data"
    source = tmp_path / "source"
    (source / "train").mkdir(parents=True)
    (source / "validation").mkdir()
    (source / "train" / "one.json").write_text("{}", encoding="utf-8")
    (source / "validation" / "two.json").write_text("{}", encoding="utf-8")
    root.mkdir()
    (root / "manifest.json").write_text(
        '{"counts": {"copied_tasks": 2}}',
        encoding="utf-8",
    )
    with tarfile.open(root / "tasks.tar.gz", "w:gz") as handle:
        handle.add(source / "train", arcname="train")
        handle.add(source / "validation", arcname="validation")

    destination = extract_task_archive(root)

    assert (destination / "train" / "one.json").is_file()
    assert (destination / "validation" / "two.json").is_file()
