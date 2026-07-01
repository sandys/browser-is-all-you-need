from __future__ import annotations

import os
import subprocess
from pathlib import Path


EXAMPLE_ROOT = Path("examples/slime/moonlight_cpp_perf")
RUNNER = EXAMPLE_ROOT / "moonlight_cpp_perf.sh"
README = EXAMPLE_ROOT / "README.md"
LOCAL_SPEC = Path("src/local.py")
TRAIN_ENTRY = Path("src/w8_biayn/integrations/slime_train_entry.py")


def test_moonlight_cpp_perf_example_files_are_present_and_executable() -> None:
    expected = {
        "README.md",
        "moonlight_cpp_perf.sh",
        "prepare_data.sh",
        "eval_base.sh",
        "sft.sh",
        "eval_sft.sh",
        "grpo.sh",
        "eval_grpo.sh",
        "compare.sh",
    }

    assert expected.issubset({path.name for path in EXAMPLE_ROOT.iterdir()})
    for script in expected - {"README.md"}:
        assert os.access(EXAMPLE_ROOT / script, os.X_OK), script


def test_moonlight_cpp_perf_scripts_are_bash_syntax_valid() -> None:
    for script in sorted(EXAMPLE_ROOT.glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_moonlight_cpp_perf_runner_pins_stage_sequence_and_artifacts() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "prepare-data|base-eval|sft|sft-eval|grpo|grpo-eval|compare" in text
    assert 'RUN_ID="${SLIME_RUN_ID:-moonlight_cpp_perf}"' in text
    assert 'TASKS_DIR="${SLIME_CPP_TASKS_DIR:-${REPO_ROOT}/.w8-biayn/data/tasks-full}"' in text
    assert 'TRAIN_LIMIT="${SLIME_CPP_TRAIN_LIMIT:-2}"' in text
    assert 'SORT_BY_SIZE="${SLIME_CPP_SORT_BY_SIZE:-1}"' in text
    assert "w8_biayn.integrations.slime_cpp_perf build-data" in text
    assert "--train-limit" in text
    assert "--eval-limit" in text
    assert "--sort-by-size" in text
    assert 'EVAL_LIMIT="${SLIME_CPP_EVAL_LIMIT:-4}"' in text
    assert "base.records.jsonl" not in text
    assert "--save-debug-rollout-data" in text
    assert "rollout_dumps/${STAGE_LABEL}_{rollout_id}.pt" in text
    assert "vram_usage.csv" in text
    assert "vram_peak.txt" in text
    assert "run_receipt.txt" in text
    assert 'DISTRIBUTED_TIMEOUT_MINUTES="${SLIME_DISTRIBUTED_TIMEOUT_MINUTES:-60}"' in text
    assert "--distributed-timeout-minutes" in text
    assert "distributed_timeout_minutes=${DISTRIBUTED_TIMEOUT_MINUTES}" in text
    assert 'sft) OPTIMIZER_CPU_OFFLOAD=0 ;;' in text
    assert 'grpo) OPTIMIZER_CPU_OFFLOAD=0 ;;' in text
    assert 'timeout "${SLIME_RAY_STOP_TIMEOUT_SECONDS:-60}" ray stop --force' in text
    assert "append_optimizer_offload_args" in text
    assert "stage_args\n  append_optimizer_offload_args" in text
    assert "optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD}" in text
    assert "hf_export_status=${HF_EXPORT_STATUS}" in text
    assert "hf_export_log=${HF_EXPORT_LOG}" in text
    assert 'SLIME_TRAIN_MODULE="${SLIME_TRAIN_MODULE:-w8_biayn.integrations.slime_train_entry}"' in text
    assert '-- "${PYTHON_BIN}" -u -m "${SLIME_TRAIN_MODULE}"' in text
    assert "-u train.py" not in text
    assert text.count('--save-interval "${SAVE_INTERVAL}"') == 6
    assert 'EVAL_LR_DECAY_ITERS="${SLIME_EVAL_LR_DECAY_ITERS:-1}"' in text
    assert text.count('--lr-decay-iters "${EVAL_LR_DECAY_ITERS}"') == 3


def test_moonlight_cpp_perf_runner_has_base_sft_grpo_slime_modes() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert "Moonlight-16B-A3B-Instruct" in text
    assert "moonshotai/Moonlight-16B-A3B-Instruct" in text
    assert 'source "${SLIME_ROOT}/scripts/models/moonlight.sh"' in text
    assert "--num-rollout 0" in text
    assert "--eval-prompt-data cpp_perf" in text
    assert "--rollout-function-path slime.rollout.sft_rollout.generate_rollout" in text
    assert "--loss-type sft_loss" in text
    assert "--disable-compute-advantages-and-returns" in text
    assert 'SFT_START_ROLLOUT_ID="${SLIME_SFT_START_ROLLOUT_ID:-0}"' in text
    assert 'GRPO_START_ROLLOUT_ID="${SLIME_GRPO_START_ROLLOUT_ID:-0}"' in text
    assert text.count('--start-rollout-id "${SFT_START_ROLLOUT_ID}"') == 1
    assert text.count('--start-rollout-id "${GRPO_START_ROLLOUT_ID}"') == 1
    assert 'SAVE_HF_EXPORTS="${SLIME_SAVE_HF_EXPORTS:-1}"' in text
    assert 'INLINE_SAVE_HF_EXPORTS="${SLIME_INLINE_SAVE_HF_EXPORTS:-0}"' in text
    assert 'STANDALONE_HF_EXPORTS="${SLIME_STANDALONE_HF_EXPORTS:-1}"' in text
    assert 'SFT_HF_SAVE_TEMPLATE="${SLIME_SFT_HF_SAVE_TEMPLATE:-${RUN_ROOT}/hf/sft/rollout_{rollout_id}}"' in text
    assert 'GRPO_HF_SAVE_TEMPLATE="${SLIME_GRPO_HF_SAVE_TEMPLATE:-${RUN_ROOT}/hf/grpo/rollout_{rollout_id}}"' in text
    assert 'export_hf_checkpoint "${SFT_SAVE_DIR}" "${SFT_HF_CHECKPOINT}" "SFT"' in text
    assert 'export_hf_checkpoint "${GRPO_SAVE_DIR}" "${GRPO_HF_CHECKPOINT}" "GRPO"' in text
    assert 'w8_biayn.integrations.slime_moonlight_hf_export' in text
    assert '--save-hf "${SFT_HF_SAVE_TEMPLATE}"' in text
    assert '--save-hf "${GRPO_HF_SAVE_TEMPLATE}"' in text
    assert 'if [ "${SAVE_HF_EXPORTS}" = "1" ] && [ "${INLINE_SAVE_HF_EXPORTS}" = "1" ]; then' in text
    assert 'require_hf_export "${SFT_HF_CHECKPOINT}" "SFT"' in text
    assert 'require_hf_export "${GRPO_HF_CHECKPOINT}" "GRPO"' in text
    assert 'GRPO_SKIP_WEIGHT_UPDATE="${SLIME_GRPO_SKIP_WEIGHT_UPDATE:-1}"' in text
    assert 'GRPO_LOAD_WEIGHTS_ONLY="${SLIME_GRPO_LOAD_WEIGHTS_ONLY:-1}"' in text
    assert 'GRPO_SKIP_FINAL_TRAIN_SLEEP="${SLIME_GRPO_SKIP_FINAL_TRAIN_SLEEP:-1}"' in text
    assert '"W8_SLIME_SKIP_FINAL_TRAIN_SLEEP": "${FINAL_TRAIN_SLEEP_SKIP}"' in text
    assert "grpo_skip_final_train_sleep=${GRPO_SKIP_FINAL_TRAIN_SLEEP}" in text
    assert "final_train_sleep_skip=${FINAL_TRAIN_SLEEP_SKIP}" in text
    assert "SLIME_GRPO_SKIP_WEIGHT_UPDATE=1 is only valid for the one-rollout measurement smoke." in text
    assert "write_frozen_sglang_config" in text
    assert "update_weights: false" in text
    assert text.count("--debug-rollout-only") == 3
    assert 'SGLANG_CONFIG_ARGS=(--sglang-config "$(write_frozen_sglang_config "${HF_CHECKPOINT}")")' in text
    assert 'ATTENTION_BACKEND="${SLIME_ATTENTION_BACKEND:-local}"' in text
    assert 'LOCAL_LAYER_SPEC_MODULE="${SLIME_LOCAL_LAYER_SPEC_MODULE:-local}"' in text
    assert 'LOCAL_LAYER_SPEC_NAME="${SLIME_LOCAL_LAYER_SPEC_NAME:-moonlight_local_decoder_block_spec}"' in text
    assert 'SEQUENCE_PARALLEL="${SLIME_SEQUENCE_PARALLEL:-1}"' in text
    assert 'MOE_GROUPED_GEMM="${SLIME_MOE_GROUPED_GEMM:-0}"' in text
    assert 'MICRO_BATCH_SIZE="${SLIME_MICRO_BATCH_SIZE:-1}"' in text
    assert 'SEQ_LENGTH="${SLIME_SEQ_LENGTH:-1024}"' in text
    assert 'USE_DYNAMIC_BATCH_SIZE="${SLIME_USE_DYNAMIC_BATCH_SIZE:-auto}"' in text
    assert 'SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK="${SLIME_SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK:-1}"' in text
    assert '"SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}"' in text
    assert '"SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}"' in text
    assert "sglang_disable_tp_memory_inbalance_check=${SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK}" in text
    assert 'CONVERT_MODEL_ARGS=("${MODEL_ARGS[@]}")' in text
    assert 'SFT_ROLLOUT_BATCH_SIZE="${SLIME_SFT_ROLLOUT_BATCH_SIZE:-2}"' in text
    assert 'GRPO_ROLLOUT_BATCH_SIZE="${SLIME_GRPO_ROLLOUT_BATCH_SIZE:-2}"' in text
    assert 'SFT_SKIP_FINAL_TRAIN_SLEEP="${SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP:-1}"' in text
    assert "sft_skip_final_train_sleep=${SFT_SKIP_FINAL_TRAIN_SLEEP}" in text
    assert 'if [ "${SFT_SKIP_FINAL_TRAIN_SLEEP}" = "1" ]; then' in text
    assert 'if [ "${arg}" = "--moe-grouped-gemm" ]; then' in text
    assert "filter_model_args" in text
    assert '"${CONVERT_MODEL_ARGS[@]}"' in text
    assert '--attention-backend "${ATTENTION_BACKEND}"' in text
    assert 'PERF_ARGS+=(--sequence-parallel)' in text
    assert '[ "${SEQUENCE_PARALLEL}" = "auto" ] && [ "${ATTENTION_BACKEND}" != "local" ]' in text
    assert 'PERF_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}")' in text
    assert 'PERF_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE}")' in text
    assert '--seq-length "${SEQ_LENGTH}"' in text
    assert '[ "${USE_DYNAMIC_BATCH_SIZE}" = "auto" ] && [ "${ATTENTION_BACKEND}" != "local" ]' in text
    assert "--no-persist-layer-norm" in text
    assert 'MEGATRON_TO_HF_MODE="${SLIME_MEGATRON_TO_HF_MODE:-raw}"' in text
    assert '--megatron-to-hf-mode "${MEGATRON_TO_HF_MODE}"' in text
    assert 'EXTRA_ARGS+=(--no-persist-layer-norm --spec "${LOCAL_LAYER_SPEC_MODULE}" "${LOCAL_LAYER_SPEC_NAME}")' in text
    assert '"NVTE_UNFUSED_ATTN", "1"' not in text
    assert '"local": {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"}' in text
    assert '"flash": {"NVTE_FLASH_ATTN": "1", "NVTE_FUSED_ATTN": "0", "NVTE_UNFUSED_ATTN": "0"}' in text
    assert 'require_checkpoint "${SFT_SAVE_DIR}" "SFT"' in text
    assert '--ref-load "${SLIME_GRPO_REF_LOAD_DIR:-${SFT_SAVE_DIR}}"' in text
    assert '--load "${SLIME_GRPO_ACTOR_LOAD_DIR:-${SFT_SAVE_DIR}}"' in text
    assert "CKPT_ARGS+=(--no-load-optim --no-load-rng)" in text
    assert "grpo_load_weights_only=${GRPO_LOAD_WEIGHTS_ONLY}" in text
    assert 'require_checkpoint "${GRPO_SAVE_DIR}" "GRPO"' in text
    assert "--advantage-estimator grpo" in text
    assert "--reward-key score" in text
    assert "--custom-rm-path w8_biayn.integrations.slime_cpp_perf.reward_func" in text


def test_moonlight_local_spec_patches_mla_core_attention() -> None:
    text = LOCAL_SPEC.read_text(encoding="utf-8")

    assert "def moonlight_local_decoder_block_spec" in text
    assert "get_gpt_decoder_block_spec" in text
    assert "MoonlightLocalDotProductAttention" in text
    assert "k_channels=None" in text
    assert "v_channels=None" in text
    assert "def forward(self, query, key, value, *args, packed_seq_params=None, **kwargs)" in text
    assert "packed_mla = query.ndim == key.ndim == value.ndim == 3" in text
    assert "query = query.unsqueeze(1)" in text
    assert "super().forward(query, key, value, *args, packed_seq_params=None, **kwargs)" in text
    assert "attention_submodules.core_attention = core_attention" in text
    assert "SequenceParallelSafeWrappedTorchNorm" in text
    assert "torch.nn.RMSNorm" in text
    assert "def _patch_sequence_parallel_rmsnorm" in text
    assert 'replace_wrapped_norm(layer_submodules, "input_layernorm")' in text
    assert 'replace_wrapped_norm(attention_submodules, "q_layernorm")' in text
    assert 'if getattr(args, "sequence_parallel", False):' in text
    assert "block_spec.layer_norm = WrappedTorchNorm" in text
    assert "from megatron.core.extensions.transformer_engine import TENorm" not in text


def test_slime_train_entry_skips_only_final_train_sleep_for_measurement() -> None:
    text = TRAIN_ENTRY.read_text(encoding="utf-8")

    assert "W8_SLIME_SKIP_FINAL_TRAIN_SLEEP" in text
    assert "def _patch_final_train_sleep" in text
    assert "actor_module.MegatronTrainRayActor.train = patched_train" in text
    assert "actor_module.MegatronTrainRayActor.save_model = patched_save_model" in text
    assert 'LOGGER.info("Skipping final actor sleep before checkpoint for one-rollout measurement.")' in text
    assert "rollout_id == self.args.num_rollout - 1" in text
    assert "actor_module.save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)" in text
    assert "if skip_final_sleep and final_rollout:" in text
    assert "ray.get(rollout_manager.dispose.remote())" in text


def test_moonlight_cpp_perf_runner_configures_wandb_and_local_sandbox() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in text.lower().splitlines() if line.strip() and not line.lstrip().startswith("#")
    )

    assert "--use-wandb" in text
    assert "--wandb-project" in text
    assert "slime-moonlight-cpp-perf" in text
    assert "--wandb-run-id" in text
    assert "W8_BIAYN_DATA_DIR" in text
    assert "W8_CPP_SANDBOX_IMAGE" in text
    assert "W8_CPP_SANDBOX_CPU" in text
    assert "DOCKER_HOST" in text
    assert "e2b" not in executable
    assert "browsergym" not in executable
    assert "webarena" not in executable


def test_moonlight_cpp_perf_readme_has_exact_operator_order() -> None:
    text = README.read_text(encoding="utf-8")

    assert "base checkpoint" in text
    assert "after SLIME SFT" in text
    assert "after SLIME GRPO" in text
    assert "does not use E2B" in text
    assert "local Docker sandbox" in text
    assert "uv run w8-biayn data pie download" in text
    assert "uv run w8-biayn data pie build-full-tasks" in text
    assert ".w8-biayn/data/tasks-small" in text
    assert "--limit-per-split 50" in text
    assert "--min-train 2" in text
    assert "--min-validation 4" in text
    assert "bash examples/slime/moonlight_cpp_perf/prepare_data.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/eval_base.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/sft.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/eval_sft.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/grpo.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/eval_grpo.sh" in text
    assert "bash examples/slime/moonlight_cpp_perf/compare.sh" in text
    assert "comparison.json" in text
    assert "rollout-only" in text
    assert "HuggingFace export" in text
    assert "Megatron-to-HF export mode: `raw`" in text
    assert "standalone HuggingFace exporter: `on`" in text
    assert "optimizer CPU offload: `off`" in text
    assert "distributed timeout: `60` minutes" in text
    assert "inline SLIME `--save-hf` exporter" in text
    assert "SLIME_SFT_SKIP_FINAL_TRAIN_SLEEP=1" in text
    assert "SLIME_GRPO_SKIP_WEIGHT_UPDATE=0" in text
    assert "Megatron sequence length: `1024`" in text
    assert "SGLang TP memory-balance check: `off`" in text
    assert "SLIME_SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1" in text
    assert "Do not treat a completed launch as proof of learning" in text
