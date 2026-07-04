from __future__ import annotations

import json
from pathlib import Path


EXAMPLE_ROOT = Path("examples/slime/retool")
LAUNCHER = EXAMPLE_ROOT / "retool_moonlight_rl.sh"
README = EXAMPLE_ROOT / "README.md"
DATA = EXAMPLE_ROOT / "moonlight_math_tool_smoke.jsonl"
AIME_STYLE_DATA = EXAMPLE_ROOT / "moonlight_math_tool_aime_style.jsonl"


def _executable_shell_text(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.lstrip().startswith("#")
    ).lower()


def test_retool_example_files_are_repo_owned() -> None:
    expected_files = {
        "README.md",
        "generate_with_retool.py",
        "tool_sandbox.py",
        "requirements.txt",
        "retool_moonlight_rl.sh",
        "moonlight_math_tool_smoke.jsonl",
        "moonlight_math_tool_aime_style.jsonl",
    }

    assert expected_files.issubset({path.name for path in EXAMPLE_ROOT.iterdir()})


def test_retool_launcher_uses_moonlight_and_local_smoke_defaults() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'source "${SLIME_ROOT}/scripts/models/moonlight.sh"' in text
    assert "Moonlight-16B-A3B-Instruct" in text
    assert 'PROMPT_DATA="${SLIME_PROMPT_DATA:-${SCRIPT_DIR}/moonlight_math_tool_smoke.jsonl}"' in text
    assert 'HF_CHECKPOINT="${SLIME_HF_CHECKPOINT:-/root/models/Moonlight-16B-A3B-Instruct}"' in text
    assert 'HF_MODEL_ID="${SLIME_HF_MODEL_ID:-moonshotai/Moonlight-16B-A3B-Instruct}"' in text
    assert 'DOWNLOAD_HF_CHECKPOINT="${SLIME_DOWNLOAD_HF_CHECKPOINT:-1}"' in text
    assert 'ACTOR_LOAD_DIR="${SLIME_ACTOR_LOAD_DIR:-${REF_LOAD_DIR}}"' in text
    assert "checkpoints/${RUN_ID}" in text
    assert 'NUM_GPUS="${SLIME_NUM_GPUS:-4}"' in text
    assert 'TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"' in text
    assert 'EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"' in text
    assert 'MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-4096}"' in text
    assert 'SGLANG_MEM_FRACTION_STATIC="${SLIME_SGLANG_MEM_FRACTION:-0.45}"' in text
    assert 'SGLANG_CUDA_GRAPH_MAX_BS="${SLIME_SGLANG_CUDA_GRAPH_MAX_BS:-16}"' in text
    assert 'DISABLE_EVAL="${SLIME_DISABLE_EVAL:-1}"' in text
    assert 'CONVERT_IF_MISSING="${SLIME_CONVERT_IF_MISSING:-1}"' in text
    assert 'PROMPT_DATA="$(absolute_path "${PROMPT_DATA}")"' in text
    assert 'EVAL_PROMPT_DATA="$(absolute_path "${EVAL_PROMPT_DATA}")"' in text


def test_retool_launcher_downloads_missing_hf_checkpoint() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    requirements = (EXAMPLE_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "download_hf_checkpoint_if_missing" in text
    assert "snapshot_download(repo_id=repo_id, local_dir=local_dir)" in text
    assert 'HF_CHECKPOINT_WAS_DOWNLOADED=1' in text
    assert "hf_checkpoint_was_downloaded=${HF_CHECKPOINT_WAS_DOWNLOADED}" in text
    assert "huggingface-hub>=0.24" in requirements


def test_retool_launcher_keeps_custom_retool_hooks_and_pythonpath() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "--custom-generate-function-path generate_with_retool.generate" in text
    assert "--custom-rm-path generate_with_retool.reward_func" in text
    assert 'PYTHONPATH": ":".join(' in text
    assert '"${SCRIPT_DIR}"' in text
    assert '"${REPO_ROOT}/src"' not in text
    assert "--reward-key score" in text


def test_retool_launcher_has_4gpu_memory_controls() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "--optimizer-cpu-offload" in text
    assert "--overlap-cpu-optimizer-d2h-h2d" in text
    assert "--use-precision-aware-optimizer" in text
    assert "--sglang-cuda-graph-max-bs" in text
    assert "--sglang-disable-custom-all-reduce" in text
    assert "--train-memory-margin-bytes" in text
    assert 'RAY_MEMORY_USAGE_THRESHOLD="${SLIME_RAY_MEMORY_USAGE_THRESHOLD-0.99}"' in text
    assert 'export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"' in text
    assert "--no-save-optim" in text
    assert "--no-save-rng" in text
    assert 'SLIME_SAVE_OPTIM_RNG:-0' in text


def test_retool_launcher_writes_vram_and_run_receipts() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "vram_usage.csv" in text
    assert "vram_peak.txt" in text
    assert "run_receipt.txt" in text
    assert "--query-gpu=timestamp,index,name,memory.used,memory.total" in text
    assert "max_peak_vram_mib:" in text
    assert "status=${RAY_STATUS}" in text
    assert "--submission-id" in text
    assert "ray job status" in text
    assert "ray_job_terminal_status=${RAY_JOB_TERMINAL_STATUS}" in text
    assert "ray_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD:-}" in text
    assert "SLIME_NOFILE_SOFT_LIMIT" in text
    assert "SLIME_RAY_STATUS_TIMEOUT_SECONDS" in text
    assert "Job '.*' succeeded" in text


def test_retool_launcher_configures_wandb_without_forcing_it() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'WANDB_KEY="${WANDB_API_KEY:-${WANDB_KEY:-}}"' in text
    assert "WANDB_ALREADY_LOGGED_IN" in text
    assert '${HOME}/.netrc' in text
    assert '${HOME}/.config/wandb/settings' in text
    assert "--wandb-project" in text
    assert "--wandb-group" in text
    assert "--disable-wandb-random-suffix" in text
    assert "--wandb-run-id" in text


def test_retool_launcher_excludes_external_sandbox_dependencies() -> None:
    executable_text = _executable_shell_text(LAUNCHER)
    helper_text = (EXAMPLE_ROOT / "tool_sandbox.py").read_text(encoding="utf-8").lower()

    assert "e2b" not in executable_text
    assert "browsergym" not in executable_text
    assert "webarena" not in executable_text
    assert "harbor" not in executable_text
    assert "e2b" not in helper_text


def test_retool_local_data_is_tiny_prompt_label_jsonl() -> None:
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 4
    assert all(sorted(row) == ["label", "prompt"] for row in rows)
    assert {row["label"] for row in rows} == {"7", "30", "42", "45"}
    assert all("Answer: \\boxed{integer}" in row["prompt"] for row in rows)


def test_retool_aime_style_data_is_repo_owned_prompt_label_jsonl() -> None:
    rows = [json.loads(line) for line in AIME_STYLE_DATA.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 16
    assert all(sorted(row) == ["label", "prompt"] for row in rows)
    assert all(row["label"].isdigit() for row in rows)
    assert all("code_interpreter" in row["prompt"] for row in rows)
    assert all("Answer: \\boxed{integer}" in row["prompt"] for row in rows)


def test_retool_readme_is_reproduction_oriented() -> None:
    text = README.read_text(encoding="utf-8")

    assert "SLIME ReTool Moonlight RL" in text
    assert "JSONL prompts -> SGLang rollout -> generate_with_retool.generate" in text
    assert "no E2B" in text
    assert "moonlight_math_tool_aime_style.jsonl" in text
    assert "no external dataset" in text
    assert "SLIME_HF_MODEL_ID=moonshotai/Moonlight-16B-A3B-Instruct" in text
    assert "SLIME_DOWNLOAD_HF_CHECKPOINT=0" in text
    assert "SLIME_CONVERT_NPROC=4" in text
    assert "SLIME_ROLLOUT_BATCH_SIZE=4" in text
    assert "SLIME_GLOBAL_BATCH_SIZE=4" in text
    assert "runs from the SLIME checkout" in text
    assert "SLIME_SGLANG_MEM_FRACTION=0.45" in text
    assert "SLIME_SGLANG_CUDA_GRAPH_MAX_BS=16" in text
    assert "SLIME_OPTIMIZER_CPU_OFFLOAD=1" in text
    assert "SLIME_RAY_MEMORY_USAGE_THRESHOLD=0.99" in text
    assert "SLIME_DISABLE_EVAL=1" in text
    assert "run_receipt.txt" in text
    assert "39367 MiB" in text
    assert "runs/ptrj71uh" in text
    assert "4x NVIDIA A100 80 GB PCIe" in text
    assert "Ray job terminal status `SUCCEEDED`" in text
    assert "does not claim learning improvement" in text
    assert "PIE C++" not in text
