from __future__ import annotations

from pathlib import Path


EXAMPLE_ROOT = Path("examples/slime/cpp_perf")
LAUNCHER = EXAMPLE_ROOT / "run_moonlight_cpp_perf_rl.sh"
README = EXAMPLE_ROOT / "README.md"


def test_cpp_perf_example_files_are_separate_from_retool() -> None:
    expected_files = {
        "README.md",
        "generate_with_cpp_perf.py",
        "requirements.txt",
        "run_moonlight_cpp_perf_rl.sh",
    }

    assert expected_files.issubset({path.name for path in EXAMPLE_ROOT.iterdir()})
    assert not Path("examples/slime/retool/generate_with_cpp_perf.py").exists()


def test_cpp_perf_launcher_uses_moonlight_and_slime_bundle_defaults() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'source "${SLIME_ROOT}/scripts/models/moonlight.sh"' in text
    assert "Moonlight-16B-A3B-Instruct" in text
    assert (
        'PROMPT_DATA="${SLIME_PROMPT_DATA:-${REPO_ROOT}/.w8-biayn/data/slime-pie/train.jsonl}"'
        in text
    )
    assert (
        'ACTOR_SAVE_DIR="${SLIME_ACTOR_SAVE_DIR:-${REPO_ROOT}/.w8-biayn/slime/cpp-perf'
        in text
    )
    assert 'CUSTOM_GENERATE_FUNCTION_PATH="${SLIME_CUSTOM_GENERATE_FUNCTION_PATH-}"' in text
    assert (
        'CUSTOM_RM_PATH="${SLIME_CUSTOM_RM_PATH-generate_with_cpp_perf.reward_func}"'
        in text
    )
    assert '--metadata-key "${METADATA_KEY}"' in text
    assert '--reward-key "${REWARD_KEY}"' in text
    assert 'CUSTOM_ARGS+=(--custom-rm-path "${CUSTOM_RM_PATH}")' in text
    assert '"${REPO_ROOT}/src"' in text
    assert '"W8_BIAYN_SLIME_TASK_ROOT"' in text
    assert '"SLIME_CPP_SANDBOX_IMAGE"' in text


def test_cpp_perf_launcher_keeps_aime_4gpu_resource_profile() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'NUM_GPUS="${SLIME_NUM_GPUS:-4}"' in text
    assert 'TP_SIZE="${SLIME_TENSOR_MODEL_PARALLEL_SIZE:-2}"' in text
    assert 'EP_SIZE="${SLIME_EXPERT_MODEL_PARALLEL_SIZE:-4}"' in text
    assert 'ROLLOUT_BATCH_SIZE="${SLIME_ROLLOUT_BATCH_SIZE:-4}"' in text
    assert 'GLOBAL_BATCH_SIZE="${SLIME_GLOBAL_BATCH_SIZE:-4}"' in text
    assert 'MAX_TOKENS_PER_GPU="${SLIME_MAX_TOKENS_PER_GPU:-4096}"' in text
    assert 'SGLANG_MEM_FRACTION_STATIC="${SLIME_SGLANG_MEM_FRACTION:-0.45}"' in text
    assert 'SGLANG_CUDA_GRAPH_MAX_BS="${SLIME_SGLANG_CUDA_GRAPH_MAX_BS:-16}"' in text
    assert (
        'RAY_MEMORY_USAGE_THRESHOLD="${SLIME_RAY_MEMORY_USAGE_THRESHOLD-0.99}"' in text
    )
    assert "--optimizer-cpu-offload" in text
    assert "--no-save-optim" in text
    assert "--no-save-rng" in text


def test_cpp_perf_reward_hook_uses_repo_reward_harness() -> None:
    text = (EXAMPLE_ROOT / "generate_with_cpp_perf.py").read_text(encoding="utf-8")

    assert "from w8_biayn.cpp_perf.reward import compute_reward" in text
    assert "from w8_biayn.cpp_perf.sandbox" in text
    assert 'metadata.get("task_path")' in text
    assert "asyncio.to_thread" in text
    assert '"score": breakdown.reward' in text


def test_cpp_perf_readme_points_to_cpp_folder_and_data_builder() -> None:
    text = README.read_text(encoding="utf-8")

    assert "SLIME PIE C++ Moonlight RL" in text
    assert "uv run w8-biayn data slime build" in text
    assert "uv run w8-biayn cpp harness preflight --cpu 3" in text
    assert "pip install -r examples/slime/cpp_perf/requirements.txt" in text
    assert "bash examples/slime/cpp_perf/run_moonlight_cpp_perf_rl.sh" in text
    assert "SLIME_CUSTOM_RM_PATH=generate_with_cpp_perf.reward_func" in text
    assert "SLIME_CUSTOM_GENERATE_FUNCTION_PATH=" in text
    assert "examples/slime/retool/retool_moonlight_rl.sh" not in text
