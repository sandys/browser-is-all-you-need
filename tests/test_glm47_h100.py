from __future__ import annotations

import os
import runpy
import subprocess
import sys
import tarfile
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


SFT_RUNNER = Path("scripts/train_sft.sh")
GRPO_RUNNER = Path("scripts/train_grpo.sh")
GLM47_H100_SFT_RUNNER = Path("examples/sft.sh")
GLM47_H100_GRPO_RUNNER = Path("examples/grpo.sh")
GLM47_AIDER_LIUM_GRPO_RUNNER = Path("examples/lium/aider_grpo_2ep.sh")
GLM47_AIDER_LIUM_EVAL_RUNNER = Path("examples/lium/aider_fixed26_eval.py")
GLM47_H100_MODAL_RUNNER = Path("examples/modal/modal_app.py")
GLM47_AIDER_EVAL_MODAL_RUNNER = Path("examples/modal/aider_eval_app.py")
GLM47_H100_CONVERTER = Path("scripts/convert_checkpoint.sh")
GLM47_H100_RUNTIME = Path("Dockerfile")
MILES_SCRIPTS = (
    SFT_RUNNER,
    GRPO_RUNNER,
    GLM47_H100_SFT_RUNNER,
    GLM47_H100_GRPO_RUNNER,
    GLM47_AIDER_LIUM_GRPO_RUNNER,
    GLM47_H100_CONVERTER,
)


def test_glm47_h100_lora_r16_scripts_are_present_and_executable() -> None:
    for script in MILES_SCRIPTS:
        assert script.exists(), script
        assert os.access(script, os.X_OK), script


def test_glm47_h100_lora_r16_scripts_are_bash_syntax_valid() -> None:
    for script in MILES_SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_grpo_runner_forwards_response_boundary_controls() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'ROLLOUT_SKIP_SPECIAL_TOKENS="${MILES_ROLLOUT_SKIP_SPECIAL_TOKENS:-0}"' in text
    assert 'ROLLOUT_STOP_TOKEN_IDS="${MILES_ROLLOUT_STOP_TOKEN_IDS:-}"' in text
    assert 'ROLLOUT_ARGS+=(--rollout-skip-special-tokens)' in text
    assert 'ROLLOUT_ARGS+=(--rollout-stop-token-ids "${ROLLOUT_STOP_TOKEN_ID_ARGS[@]}")' in text


def test_glm47_h100_wrappers_select_fast_8x_h100_defaults() -> None:
    for script in (GLM47_H100_SFT_RUNNER, GLM47_H100_GRPO_RUNNER):
        text = script.read_text(encoding="utf-8")
        assert 'MILES_MODEL_ARGS_FILE="${MILES_MODEL_ARGS_FILE:-glm4.7-flash.sh}"' in text
        assert 'MILES_HF_CHECKPOINT="${MILES_HF_CHECKPOINT:-/root/models/GLM-4.7-Flash}"' in text
        assert 'MILES_REF_LOAD_DIR="${MILES_REF_LOAD_DIR:-${MILES_HF_CHECKPOINT}_torch_dist_tp4_pp1_ep8}"' in text
        assert 'MILES_GPUS_PER_NODE="${MILES_GPUS_PER_NODE:-8}"' in text
        assert 'MILES_TENSOR_MODEL_PARALLEL_SIZE="${MILES_TENSOR_MODEL_PARALLEL_SIZE:-4}"' in text
        assert 'MILES_PIPELINE_MODEL_PARALLEL_SIZE="${MILES_PIPELINE_MODEL_PARALLEL_SIZE:-1}"' in text
        assert 'MILES_CONTEXT_PARALLEL_SIZE="${MILES_CONTEXT_PARALLEL_SIZE:-1}"' in text
        assert 'MILES_EXPERT_MODEL_PARALLEL_SIZE="${MILES_EXPERT_MODEL_PARALLEL_SIZE:-8}"' in text
        assert 'MILES_EXPERT_TENSOR_PARALLEL_SIZE="${MILES_EXPERT_TENSOR_PARALLEL_SIZE:-1}"' in text
        assert 'MILES_SEQ_LENGTH="${MILES_SEQ_LENGTH:-4096}"' in text
        assert 'MILES_MAX_TOKENS_PER_GPU="${MILES_MAX_TOKENS_PER_GPU:-16384}"' in text
        assert 'MILES_RECOMPUTE_GRANULARITY="${MILES_RECOMPUTE_GRANULARITY:-selective}"' in text
        assert 'MILES_USE_DYNAMIC_BATCH_SIZE="${MILES_USE_DYNAMIC_BATCH_SIZE:-1}"' in text
        assert 'MILES_BALANCE_DATA="${MILES_BALANCE_DATA:-1}"' in text
        assert 'MILES_MOE_TOKEN_DISPATCHER_TYPE="${MILES_MOE_TOKEN_DISPATCHER_TYPE:-flex}"' in text
        assert 'MILES_MOE_ENABLE_DEEPEP="${MILES_MOE_ENABLE_DEEPEP:-1}"' in text
        assert 'NVSHMEM_DISABLE_NCCL="${NVSHMEM_DISABLE_NCCL:-1}"' in text
        assert 'MILES_ATTENTION_BACKEND="${MILES_ATTENTION_BACKEND:-flash}"' in text
        assert 'MILES_SGLANG_ENABLE_DP_ATTENTION="${MILES_SGLANG_ENABLE_DP_ATTENTION:-1}"' in text
        assert 'MILES_SGLANG_DP_SIZE="${MILES_SGLANG_DP_SIZE:-8}"' in text
        assert 'MILES_SGLANG_ENABLE_DP_LM_HEAD="${MILES_SGLANG_ENABLE_DP_LM_HEAD:-1}"' in text
        assert 'MILES_SGLANG_MOE_DENSE_TP_SIZE="${MILES_SGLANG_MOE_DENSE_TP_SIZE:-1}"' in text
        assert 'MILES_SGLANG_SPECULATIVE="${MILES_SGLANG_SPECULATIVE:-0}"' in text
        assert 'MILES_SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${MILES_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}"' in text
        assert 'MILES_EXPERTS_SHARED_OUTER_LORAS="${MILES_EXPERTS_SHARED_OUTER_LORAS:-1}"' in text
        assert 'MILES_TRAIN_MODULE="${MILES_TRAIN_MODULE:-glm47_posttraining.integrations.miles_train_with_glm47_bridge}"' in text
        assert 'GLM47_CPP_SANDBOX_BACKEND="${GLM47_CPP_SANDBOX_BACKEND:-local}"' in text

    grpo_text = GLM47_H100_GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'MILES_NUM_ROLLOUT="${MILES_NUM_ROLLOUT:-100}"' in grpo_text
    assert 'MILES_ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-32}"' in grpo_text
    assert 'MILES_N_SAMPLES_PER_PROMPT="${MILES_N_SAMPLES_PER_PROMPT:-8}"' in grpo_text
    assert 'MILES_GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-256}"' in grpo_text
    assert 'MILES_ROLLOUT_MAX_RESPONSE_LEN="${MILES_ROLLOUT_MAX_RESPONSE_LEN:-1536}"' in grpo_text
    assert 'MILES_ROLLOUT_TEMPERATURE="${MILES_ROLLOUT_TEMPERATURE:-1.0}"' in grpo_text
    assert 'MILES_EVAL_MAX_RESPONSE_LEN="${MILES_EVAL_MAX_RESPONSE_LEN:-1536}"' in grpo_text
    assert 'MILES_EVAL_INTERVAL="${MILES_EVAL_INTERVAL:-20}"' in grpo_text
    assert 'MILES_SAVE_INTERVAL="${MILES_SAVE_INTERVAL:-10}"' in grpo_text
    assert 'MILES_LR="${MILES_LR:-2e-6}"' in grpo_text
    assert 'MILES_NO_REF="${MILES_NO_REF:-1}"' in grpo_text
    assert 'MILES_SGLANG_MEM_FRACTION_STATIC="${MILES_SGLANG_MEM_FRACTION_STATIC:-0.75}"' in grpo_text
    assert 'MILES_SGLANG_SERVER_CONCURRENCY="${MILES_SGLANG_SERVER_CONCURRENCY:-1024}"' in grpo_text
    assert 'MILES_SGLANG_CUDA_GRAPH_MAX_BS="${MILES_SGLANG_CUDA_GRAPH_MAX_BS:-64}"' in grpo_text
    assert 'MILES_SGLANG_MAX_RUNNING_REQUESTS="${MILES_SGLANG_MAX_RUNNING_REQUESTS:-256}"' in grpo_text
    assert 'GLM47_CPP_REWARD_WORKERS="${GLM47_CPP_REWARD_WORKERS:-32}"' in grpo_text
    assert (
        'MILES_APPLY_CHAT_TEMPLATE_KWARGS="${MILES_APPLY_CHAT_TEMPLATE_KWARGS:-{\\"enable_thinking\\": false}}"'
        in grpo_text
    )

    sft_text = GLM47_H100_SFT_RUNNER.read_text(encoding="utf-8")
    assert 'MILES_ROLLOUT_BATCH_SIZE="${MILES_ROLLOUT_BATCH_SIZE:-32}"' in sft_text
    assert 'MILES_GLOBAL_BATCH_SIZE="${MILES_GLOBAL_BATCH_SIZE:-32}"' in sft_text
    assert 'MILES_SAVE_INTERVAL="${MILES_SAVE_INTERVAL:-1000}"' in sft_text
    assert 'MILES_NO_REF="${MILES_NO_REF:-1}"' in sft_text
    assert 'MILES_SGLANG_MEM_FRACTION_STATIC="${MILES_SGLANG_MEM_FRACTION_STATIC:-0.60}"' in sft_text
    assert 'MILES_SGLANG_CUDA_GRAPH_MAX_BS="${MILES_SGLANG_CUDA_GRAPH_MAX_BS:-16}"' in sft_text
    assert 'MILES_SGLANG_MAX_RUNNING_REQUESTS="${MILES_SGLANG_MAX_RUNNING_REQUESTS:-64}"' in sft_text
    assert 'MILES_LORA_BASE_CPU_BACKUP="${MILES_LORA_BASE_CPU_BACKUP:-0}"' in sft_text
    assert 'MILES_EXTRA_ARGS="--no-offload-train${MILES_EXTRA_ARGS:+ ${MILES_EXTRA_ARGS}}"' in sft_text


def test_glm47_h100_wandb_lineage_reaches_ray_workers_and_receipts() -> None:
    expected_job_types = {
        GLM47_H100_SFT_RUNNER: "sft",
        GLM47_H100_GRPO_RUNNER: "grpo",
    }
    for script, job_type in expected_job_types.items():
        text = script.read_text(encoding="utf-8")
        assert 'GLM47_EXPERIMENT_ID="${GLM47_EXPERIMENT_ID:-${RUN_ID}}"' in text
        assert 'MILES_WANDB_GROUP="${MILES_WANDB_GROUP:-${GLM47_EXPERIMENT_ID}}"' in text
        assert f'MILES_WANDB_JOB_TYPE="${{MILES_WANDB_JOB_TYPE:-{job_type}}}"' in text
        assert 'WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${GLM47_EXPERIMENT_ID}}"' in text
        assert f'WANDB_JOB_TYPE="${{WANDB_JOB_TYPE:-{job_type}}}"' in text

    for runner in (SFT_RUNNER, GRPO_RUNNER):
        text = runner.read_text(encoding="utf-8")
        assert "wandb_job_type=${WANDB_JOB_TYPE}" in text
        assert "experiment_id=${EXPERIMENT_ID}" in text
        assert "GLM47_EXPERIMENT_ID" in text
        assert "WANDB_JOB_TYPE" in text
        assert "WANDB_RUN_GROUP" in text
        assert "WANDB_TAGS" in text
        assert '"${REPO_ROOT}/scripts/publish_results.py" "${finalize_args[@]}"' in text
        assert "finalize_args=(\n    finalize-stage" in text
        assert '--run-log "${LOG_FILE}"' in text
        assert '--rollout-dump-dir "${rollout_dump_dir}"' in text
        assert '--checkpoint-dir "${SAVE_DIR}"' in text
        assert '--sync-metrics-dir "${GLM47_SYNC_METRICS_DIR}"' in text
        assert '--timing-status "${GLM47_TIMING_STATUS:-unverified}"' in text
        assert "wall_s=$((SECONDS - STAGE_STARTED_AT))" in text
        assert 'finalize_wandb "${STAGE_STATUS}"' in text

    sft_runner_text = SFT_RUNNER.read_text(encoding="utf-8")
    grpo_runner_text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert '"WANDB_RUN_ID": "${WANDB_RUN_ID}",' in sft_runner_text
    assert '\\"WANDB_RUN_ID\\": \\"${WANDB_RUN_ID}\\",' in grpo_runner_text


def test_glm47_h100_converter_matches_runner_layout() -> None:
    text = GLM47_H100_CONVERTER.read_text(encoding="utf-8")
    assert 'MODEL_ARGS_FILE="${MILES_MODEL_ARGS_FILE:-glm4.7-flash.sh}"' in text
    assert 'HF_CHECKPOINT="${MILES_HF_CHECKPOINT:-/root/models/GLM-4.7-Flash}"' in text
    assert 'REF_LOAD_DIR="${MILES_REF_LOAD_DIR:-${HF_CHECKPOINT}_torch_dist_tp4_pp1_ep8}"' in text
    assert 'TP_SIZE="${MILES_TENSOR_MODEL_PARALLEL_SIZE:-4}"' in text
    assert 'PP_SIZE="${MILES_PIPELINE_MODEL_PARALLEL_SIZE:-1}"' in text
    assert 'EP_SIZE="${MILES_EXPERT_MODEL_PARALLEL_SIZE:-8}"' in text
    assert 'ETP_SIZE="${MILES_EXPERT_TENSOR_PARALLEL_SIZE:-1}"' in text
    assert 'CONVERT_NPROC="${MILES_CONVERT_NPROC:-8}"' in text
    # The bridge maps the grouped expert parameter names emitted by conversion.
    assert 'STRIP_GROUPED_GEMM="${GLM47_STRIP_MOE_GROUPED_GEMM:-0}"' in text
    assert 'if [ "${STRIP_GROUPED_GEMM}" = "1" ] && [ "${arg}" = "--moe-grouped-gemm" ]; then' in text
    assert 'convert_hf_to_torch_dist.py' in text
    assert '--expert-model-parallel-size "${EP_SIZE}"' in text
    # Register the GLM-4.7 bridge before invoking the Miles converter.
    assert "-m glm47_posttraining.integrations.miles_convert_with_glm47_bridge" in text
    assert 'CONVERT_PYTHONPATH="${REPO_ROOT}/src:${MEGATRON_DIR}:${PYTHONPATH:-}"' in text


def test_miles_convert_wrapper_registers_bridge_before_exec() -> None:
    import inspect

    from glm47_posttraining.integrations import miles_convert_with_glm47_bridge as wrapper

    source = inspect.getsource(wrapper.main)
    assert source.index("register_glm47_bridge()") < source.index("exec(compile(")
    assert "convert_hf_to_torch_dist.py" in inspect.getsource(wrapper)


def test_miles_convert_wrapper_pp1_patch(tmp_path, monkeypatch) -> None:
    from glm47_posttraining.integrations import miles_convert_with_glm47_bridge as wrapper

    tool = tmp_path / "convert_hf_to_torch_dist.py"
    body = (
        "def get_args(args, world_size):\n"
        f"    {wrapper.PP_OVERRIDE_MARKER}\n"
        "        args.pipeline_model_parallel_size = world_size\n"
        "    return args\n"
    )
    tool.write_text(body, encoding="utf-8")

    # gate off: source untouched
    monkeypatch.delenv("GLM47_KEEP_PP1", raising=False)
    assert wrapper._load_source(tool) == body

    # gate on: override branch neutralized, body still valid python
    monkeypatch.setenv("GLM47_KEEP_PP1", "1")
    patched = wrapper._load_source(tool)
    assert wrapper.PP_OVERRIDE_MARKER not in patched
    assert "if False:" in patched
    compile(patched, str(tool), "exec")

    # gate on but marker missing: fail loud instead of converting a lie
    tool.write_text("def get_args():\n    return None\n", encoding="utf-8")
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="PP-override marker"):
        wrapper._load_source(tool)


def test_glm47_bridge_patches_mbridge_qk_layernorm_mapping(monkeypatch) -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    class FakeGLMBridge:
        _ATTENTION_MAPPING = {
            "self_attention.linear_proj.weight": [
                "model.layers.{layer_number}.self_attn.o_proj.weight"
            ],
        }

    fake_bridge_module = types.ModuleType("mbridge.core.bridge")
    fake_bridge_module._MODEL_REGISTRY = {"glm4_moe_lite": FakeGLMBridge}

    monkeypatch.setattr(miles_glm47_bridge, "_MBRIDGE_PATCHED", False)
    monkeypatch.setitem(sys.modules, "miles_plugins", types.ModuleType("miles_plugins"))
    monkeypatch.setitem(sys.modules, "miles_plugins.mbridge", types.ModuleType("miles_plugins.mbridge"))
    monkeypatch.setitem(sys.modules, "mbridge", types.ModuleType("mbridge"))
    monkeypatch.setitem(sys.modules, "mbridge.core", types.ModuleType("mbridge.core"))
    monkeypatch.setitem(sys.modules, "mbridge.core.bridge", fake_bridge_module)

    miles_glm47_bridge._patch_mbridge_glm47_lite()

    assert FakeGLMBridge._ATTENTION_MAPPING["self_attention.linear_qkv.layer_norm_weight"] == [
        "model.layers.{layer_number}.input_layernorm.weight"
    ]


def test_glm47_bridge_marks_shared_outer_lora_as_ep_replicated(monkeypatch) -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    class FakeShardedTensor:
        def __init__(self, replica_id):
            self.replica_id = replica_id

    class FakeSharedOuterAdapter:
        def __init__(self, is_fc1, replica_id):
            self._is_fc1 = is_fc1
            self._replica_id = replica_id

        def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
            shared_side = "linear_in" if self._is_fc1 else "linear_out"
            per_expert_side = "linear_out" if self._is_fc1 else "linear_in"
            return {
                f"{prefix}{shared_side}.weight": FakeShardedTensor(self._replica_id),
                f"{prefix}{shared_side}._extra_state": FakeShardedTensor(self._replica_id),
                f"{prefix}{per_expert_side}.weight": FakeShardedTensor((0, 0, 0)),
            }

    fake_peft_utils = types.ModuleType("megatron.bridge.peft.utils")
    fake_peft_utils.SharedOuterGroupedExpertAdapter = FakeSharedOuterAdapter

    fake_parallel_state = types.ModuleType("megatron.core.parallel_state")
    fake_parallel_state.get_expert_model_parallel_rank = lambda: 3
    fake_parallel_state.get_expert_model_parallel_world_size = lambda: 4

    fake_core = types.ModuleType("megatron.core")
    fake_core.parallel_state = fake_parallel_state

    monkeypatch.setattr(miles_glm47_bridge, "_SHARED_OUTER_CKPT_PATCHED", False)
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.bridge", types.ModuleType("megatron.bridge"))
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft", types.ModuleType("megatron.bridge.peft"))
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.utils", fake_peft_utils)
    monkeypatch.setitem(sys.modules, "megatron.core", fake_core)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", fake_parallel_state)

    miles_glm47_bridge._patch_shared_outer_expert_adapter_replication()
    assert FakeSharedOuterAdapter._glm47_ep_replica_patched is True

    fc1 = FakeSharedOuterAdapter(is_fc1=True, replica_id=(0, 0, 0)).sharded_state_dict(prefix="a.")
    assert fc1["a.linear_in.weight"].replica_id == (0, 0, 3)
    assert fc1["a.linear_in._extra_state"].replica_id == (0, 0, 3)
    assert fc1["a.linear_out.weight"].replica_id == (0, 0, 0)

    fc2 = FakeSharedOuterAdapter(is_fc1=False, replica_id=(0, 0, 1)).sharded_state_dict(prefix="b.")
    assert fc2["b.linear_out.weight"].replica_id == (0, 0, 7)
    assert fc2["b.linear_out._extra_state"].replica_id == (0, 0, 7)
    assert fc2["b.linear_in.weight"].replica_id == (0, 0, 0)

    int_replica = FakeSharedOuterAdapter(is_fc1=True, replica_id=2).sharded_state_dict(prefix="c.")
    assert int_replica["c.linear_in.weight"].replica_id == 11

    # Re-running the patch must not double-wrap.
    monkeypatch.setattr(miles_glm47_bridge, "_SHARED_OUTER_CKPT_PATCHED", False)
    miles_glm47_bridge._patch_shared_outer_expert_adapter_replication()
    rewrapped = FakeSharedOuterAdapter(is_fc1=True, replica_id=(0, 0, 0)).sharded_state_dict(prefix="d.")
    assert rewrapped["d.linear_in.weight"].replica_id == (0, 0, 3)


def test_glm47_bridge_drops_mtp_adapters_from_sglang_lora_sync() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    sent = []

    class FakeUpdater:
        def __init__(self, num_layers):
            self.args = types.SimpleNamespace(num_layers=num_layers)

        def _send_lora_params(self, hf_named_tensors):
            sent.append(list(hf_named_tensors))
            return [], None

    fake_module = types.ModuleType("miles.backends.megatron_utils.update_weight.update_weight_from_tensor")
    fake_module.UpdateWeightFromTensor = FakeUpdater

    miles_glm47_bridge._apply_sglang_lora_mtp_filter(fake_module)
    assert FakeUpdater._glm47_mtp_filter_patched is True

    tensors = [
        ("base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight", "t0"),
        ("base_model.model.model.layers.46.mlp.gate_proj.lora_B.weight", "t46"),
        ("base_model.model.model.layers.47.mlp.shared_experts.gate_proj.lora_A.weight", "t47"),
    ]
    FakeUpdater(num_layers=47)._send_lora_params(tensors)
    assert [name for name, _ in sent[-1]] == [
        "base_model.model.model.layers.0.self_attn.q_a_proj.lora_A.weight",
        "base_model.model.model.layers.46.mlp.gate_proj.lora_B.weight",
    ]

    # All-MTP payload must pass through unfiltered rather than become empty.
    only_mtp = [("base_model.model.model.layers.47.mlp.gate_proj.lora_A.weight", "t")]
    FakeUpdater(num_layers=47)._send_lora_params(only_mtp)
    assert sent[-1] == only_mtp

    # Double application must not re-wrap.
    miles_glm47_bridge._apply_sglang_lora_mtp_filter(fake_module)
    FakeUpdater(num_layers=47)._send_lora_params(tensors)
    assert len(sent[-1]) == 2


def test_glm47_bridge_orders_sglang_mem_pool_per_expert_first() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    observed = []

    class FakePool:
        def load_lora_weight_to_buffer(self, uid, buffer_id, lora_adapter, *args, **kwargs):
            observed.append([list(layer.weights) for layer in lora_adapter.layers])
            return "ok"

    fake_module = types.ModuleType("sglang.srt.lora.mem_pool")
    fake_module.LoRAMemoryPool = FakePool

    miles_glm47_bridge._apply_sglang_mem_pool_ordering(fake_module)
    assert FakePool._glm47_expert_order_patched is True

    shared_first = types.SimpleNamespace(
        weights={
            "mlp.experts.gate_proj.lora_A.weight": "shared3d",
            "mlp.experts.0.gate_proj.lora_B.weight": "e0",
            "mlp.experts.1.gate_proj.lora_B.weight": "e1",
            "self_attn.o_proj.lora_A.weight": "attn",
        }
    )
    no_experts = types.SimpleNamespace(weights={"self_attn.o_proj.lora_A.weight": "attn"})
    adapter = types.SimpleNamespace(layers=[shared_first, no_experts])

    result = FakePool().load_lora_weight_to_buffer("uid", 0, adapter)
    assert result == "ok"
    assert observed[-1][0] == [
        "mlp.experts.0.gate_proj.lora_B.weight",
        "mlp.experts.1.gate_proj.lora_B.weight",
        "mlp.experts.gate_proj.lora_A.weight",
        "self_attn.o_proj.lora_A.weight",
    ]
    assert observed[-1][1] == ["self_attn.o_proj.lora_A.weight"]


def test_register_glm47_bridge_installs_hooks_without_heavy_imports(monkeypatch) -> None:
    """Registration stays lazy when sitecustomize initializes Ray workers."""

    import importlib.abc

    from glm47_posttraining.integrations import miles_glm47_bridge

    heavy_roots = ("megatron", "mbridge", "miles_plugins", "transformers", "modelopt")
    attempted: list[str] = []

    class RecordingFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in heavy_roots:
                attempted.append(fullname)
            return None

    recorder = RecordingFinder()
    monkeypatch.setattr(miles_glm47_bridge, "_REGISTERED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_MBRIDGE_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_SHARED_OUTER_CKPT_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_LORA_SYNC_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_SGLANG_MEM_POOL_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_ROUTER_CB_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_WARM_START_OPT_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_LORA_TMS_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_LORA_UPDATE_TMS_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_ROLLOUT_DP_SHARD_PATCHED", False)
    monkeypatch.setattr(miles_glm47_bridge, "_CORRECT_SAMPLE_LOG_PATCHED", False)
    before_meta_path = list(sys.meta_path)
    sys.meta_path.insert(0, recorder)
    try:
        miles_glm47_bridge.register_glm47_bridge()
        assert attempted == []
        added = [f for f in sys.meta_path if f is not recorder and f not in before_meta_path]
        # one lazy hook per patch target: mbridge.core.bridge, miles_plugins.mbridge,
        # megatron.bridge.peft.utils, miles update_weight module, sglang mem_pool,
        # miles router_manager, miles lora_utils (optimizer reload), Miles'
        # bridge_lora_helpers (non-nested TMS allocation), Miles' actor
        # (process-group reload inside the live TMS scope), miles.utils.data
        # (aligned raw-reward DP sharding), Miles' rollout logger (global pass@k
        # plus local correct-sample rows), and
        # megatron.bridge for the bridge-class registration
        assert len(added) == 12
    finally:
        sys.meta_path[:] = [f for f in sys.meta_path if f is recorder or f in before_meta_path]
        sys.meta_path.remove(recorder)


def test_grpo_runner_supports_adapter_init_passthrough() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'LORA_ADAPTER_PATH="${MILES_LORA_ADAPTER_PATH:-}"' in text
    assert '--lora-adapter-path "${LORA_ADAPTER_PATH}"' in text


def test_h100_grpo_prepares_hybrid_adapter() -> None:
    text = GLM47_H100_GRPO_RUNNER.read_text(encoding="utf-8")
    assert "scripts/check_runtime.py" in text
    assert "MILES_SKIP_RUNTIME_PREFLIGHT:-0" in text
    assert 'MILES_AUTO_PREPARE_GRPO_ADAPTER:-1' in text
    assert 'MILES_GRPO_ADAPTER_DIR:-${MILES_RUN_ROOT}/adapter_hybrid' in text
    assert 'scripts/prepare_grpo_adapter.py' in text
    assert "--include-native" in text
    assert "--include-training-state" not in text


def test_h100_runtime_aligns_all_flashinfer_packages() -> None:
    text = GLM47_H100_RUNTIME.read_text(encoding="utf-8")
    assert "GLM47_FLASHINFER_VERSION=0.6.12" in text
    assert "GLM47_FLASHINFER_CUDA_INDEX=129" in text
    assert "ENV FLASHINFER_VERSION=${GLM47_FLASHINFER_VERSION}" in text
    assert "ENV FLASHINFER_CUDA_INDEX=${GLM47_FLASHINFER_CUDA_INDEX}" in text
    for package in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
        assert package in text
    assert text.count("--force-reinstall") >= 3
    assert "GLM47_SGLANG_KERNEL_VERSION=0.4.4" in text
    assert "GLM47_TORCH_MEMORY_SAVER_VERSION=0.0.9.post1" in text
    assert "ENV SGLANG_KERNEL_VERSION=${GLM47_SGLANG_KERNEL_VERSION}" in text
    assert "ENV TORCH_MEMORY_SAVER_VERSION=${GLM47_TORCH_MEMORY_SAVER_VERSION}" in text
    assert "https://docs.sglang.ai/whl/cu${FLASHINFER_CUDA_INDEX}/" in text


def test_modal_reproduction_pins_model_image_and_machine_shape() -> None:
    text = GLM47_H100_MODAL_RUNNER.read_text(encoding="utf-8")
    assert 'MODEL_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"' in text
    assert "sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37" in text
    assert 'modal.Image.from_dockerfile(' in text
    assert '"gpu": "H100!:8"' in text
    assert '"cpu": 48.0' in text
    assert '"memory": (262_144, 1_048_576)' in text
    assert '"timeout": 86_400' in text
    assert '"GLM47_CPP_SANDBOX_BACKEND": "local"' in text
    assert '"GLM47_CPP_SANDBOX_UNSHARE_NET": "0"' in text
    assert 'modal.Secret.from_name("wandb-glm47")' in text
    assert 'modal.Secret.from_name("huggingface-token")' in text


def test_modal_aider_profile_binds_objective_adapter_and_safe_reward() -> None:
    text = GLM47_H100_MODAL_RUNNER.read_text(encoding="utf-8")
    assert '"bubblewrap"' in text
    assert 'AIDER_DATASET_KIND = "aider-polyglot-cpp-shadow-grpo"' in text
    assert 'AIDER_TASKS_DIR = f"{ASSETS_DIR}/aider-shadow/tasks/aider_polyglot_cpp_shadow"' in text
    assert '"rubrics"' in text
    assert 'scripts/download_assets.py aider-shadow' in text
    assert 'secrets=[hf_secret]' in text
    assert "def prepare_aider_shadow_asset(" in text
    assert 'volumes={ASSETS_DIR: assets, RUNS_DIR: runs}' in text
    assert "glm47-aider-complement-530-sft-20260721" in text
    assert "glm47-aider-1211-sft-20260718T192250Z" in text
    assert "glm47-aider-1211-530-equal-delta-merge-r32" in text
    assert "f1ea45bc327dc6e28d0287aea75c6b691e99d2ec2f7fdb7f07bbbf5ccd6cf36a" in text
    assert '"MILES_DATA_BUILD_MODULE"' in text
    assert '"glm47_posttraining.integrations.miles_aider_polyglot.reward_func"' in text
    assert '"MILES_ROLLOUT_MAX_RESPONSE_LEN": "4096"' in text
    assert '"MILES_ROLLOUT_STOP_TOKEN_IDS": "154820 154827 154829"' in text
    assert '"MILES_ROLLOUT_SKIP_SPECIAL_TOKENS": "1"' in text
    assert '"MILES_KL_LOSS_COEF": "0.02"' in text
    assert '"MILES_NO_REF": "0"' in text
    assert '"MILES_NUM_ROLLOUT": num_rollout or (' in text
    assert "def aider_preflight(" in text
    assert "def merge_aider(" in text
    assert "def aider_profile(" in text
    assert "def aider_grpo(" in text
    assert 'env["MILES_LORA_RANK"] = lora_rank' in text
    assert 'env["MILES_LORA_ALPHA"] = lora_alpha' in text
    assert "num_rollout=num_rollout" in text


def test_aider_fixed_26_eval_requires_grpo_gate() -> None:
    text = GLM47_AIDER_EVAL_MODAL_RUNNER.read_text(encoding="utf-8")
    assert '"glm47-aider-grpo-training-gate"' in text
    assert '"grpo_lora_r16", "grpo_training_gate.json"' in text
    assert '"training_task_count": EXPECTED_TRAINING_TASK_COUNT' in text
    assert 'os.environ.get("GLM47_EXPECTED_TRAINING_TASK_COUNT", "253")' in text
    assert 'os.environ.get("GLM47_EVAL_LORA_RANK", "16")' in text
    assert '"official_26_role": "external fixed evaluation only"' in text
    assert 'EXPECTED_SOURCE_TENSORS = 9_741' in text
    assert 'EXPECTED_LAYER_47_TENSORS = 207' in text
    assert 'AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"' in text
    assert 'POLYGLOT_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"' in text
    assert "def evaluate_shard(" in text
    assert "def merge_shards(" in text
    assert "elif parallel:" in text


def test_lium_aider_reproduction_pins_inputs_and_fixed_schedule() -> None:
    text = GLM47_AIDER_LIUM_GRPO_RUNNER.read_text(encoding="utf-8")
    for value in (
        "a7e54c0245b97ae78f9b2fa57ff5278844585cf03004254137b6cfc8e91ef157",
        "b72394ab603b4b6faf22370ea70605446f112ab50c883eb61e308e2dd9ab4dd2",
        "dbea7d3e2d6603f278b94c6be134bca83bb5f0ebdc4840eb53898ec5b3affb91",
        "MILES_NUM_ROLLOUT=11",
        "MILES_LORA_RANK=32",
        "MILES_LR=5e-7",
        "MILES_KL_LOSS_COEF=0.02",
        "MILES_ROLLOUT_SKIP_SPECIAL_TOKENS=1",
        "MILES_ROLLOUT_STOP_TOKEN_IDS='154820 154827 154829'",
        "WANDB_MODE=offline",
        "002993b94ddf85e23863e22484459df4b724d91204e5e48c37904a1f34748f00",
        "aider-shadow/tasks/aider_polyglot_cpp_shadow",
    ):
        assert value in text
    assert "verify_sha256" in text
    assert 'source_commit="$(git -C "${repo_root}" rev-parse HEAD)"' in text


def test_lium_fixed_26_eval_pins_benchmark_and_adapter() -> None:
    text = GLM47_AIDER_LIUM_EVAL_RUNNER.read_text(encoding="utf-8")
    for value in (
        "5dc9490bb35f9729ef2c95d00a19ccd30c26339c",
        "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f",
        "046a1018b605aa29f8b8c4f2677f47ce55489105f6766155f4c009798f48abe2",
        "a7e54c0245b97ae78f9b2fa57ff5278844585cf03004254137b6cfc8e91ef157",
        '"--tries", "2"',
        '"--edit-format", "whole"',
        '"temperature": 0.7',
        '"unique_testcases": 26',
    ):
        assert value in text


def test_asset_downloader_pins_the_base_model_revision() -> None:
    module = runpy.run_path("scripts/download_assets.py")
    model = module["ASSETS"]["model"]
    assert model["repo_id"] == "zai-org/GLM-4.7-Flash"
    assert model["default_revision"] == "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
    assert model["destination"] == "GLM-4.7-Flash"
    assert model["verify_checksums"] is False


def test_asset_downloader_pins_the_aider_shadow_revision() -> None:
    module = runpy.run_path("scripts/download_assets.py")
    shadow = module["ASSETS"]["aider-shadow"]
    assert shadow["repo_id"] == "TokenBender/glm47-aider-polyglot-cpp-shadow"
    assert shadow["default_revision"] == "d8f86f752685d5ddc6cece2a08ea8851b395ee83"
    assert shadow["destination"] == "aider-shadow"
    assert shadow["verify_checksums"] is True


def test_h100_runtime_preflight_accepts_aligned_versions() -> None:
    module = runpy.run_path("scripts/check_runtime.py")
    validate = module["validate_miles_h100_runtime"]

    expected = {
        "flashinfer-python": "0.6.12",
        "flashinfer-cubin": "0.6.12",
        "flashinfer-jit-cache": "0.6.12+cu129",
        "sglang-kernel": "0.4.4+cu129",
        "torch-memory-saver": "0.0.9.post1",
    }
    versions = validate(expected.__getitem__)

    assert versions == expected


def test_h100_runtime_preflight_rejects_mismatched_versions() -> None:
    module = runpy.run_path("scripts/check_runtime.py")
    validate = module["validate_miles_h100_runtime"]

    current = {
        "flashinfer-python": "0.6.12",
        "flashinfer-cubin": "0.6.12",
        "flashinfer-jit-cache": "0.6.12+cu129",
        "sglang-kernel": "0.4.4+cu129",
        "torch-memory-saver": "0.0.9.post1",
    }

    mismatched_kernel = {**current, "sglang-kernel": "0.4.2.post2+cu129"}
    with pytest.raises(RuntimeError, match="below the required minimum"):
        validate(mismatched_kernel.__getitem__)

    mismatched_saver = {**current, "torch-memory-saver": "0.0.9"}
    with pytest.raises(RuntimeError, match="below the required minimum"):
        validate(mismatched_saver.__getitem__)

    mixed = {**current, "flashinfer-jit-cache": "0.6.13+cu129"}
    with pytest.raises(RuntimeError, match="versions are not aligned"):
        validate(mixed.__getitem__)


def test_data_asset_extracts_verified_task_bundle(tmp_path) -> None:
    module = runpy.run_path("scripts/download_assets.py")
    extract_task_archive = module["_extract_task_archive"]

    root = tmp_path / "data"
    source = tmp_path / "source"
    (source / "train").mkdir(parents=True)
    (source / "validation").mkdir()
    (source / "train" / "one.json").write_text("{}")
    (source / "validation" / "two.json").write_text("{}")
    root.mkdir()
    (root / "manifest.json").write_text('{"counts": {"copied_tasks": 2}}')
    with tarfile.open(root / "tasks.tar.gz", "w:gz") as handle:
        handle.add(source / "train", arcname="train")
        handle.add(source / "validation", arcname="validation")

    destination = extract_task_archive(root)

    assert (destination / "train" / "one.json").is_file()
    assert (destination / "validation" / "two.json").is_file()


def test_aider_shadow_asset_extracts_verified_archive(tmp_path) -> None:
    import hashlib
    import json

    module = runpy.run_path("scripts/download_assets.py")
    extract = module["_extract_aider_shadow_archive"]
    root = tmp_path / "aider-shadow"
    source = tmp_path / "aider_polyglot_cpp_shadow"
    practice = source / "cpp" / "exercises" / "practice"
    practice.mkdir(parents=True)
    for index in range(253):
        task = practice / f"task-{index:03d}"
        task.mkdir()
        (task / ".rubric.json").write_text("{}")
    source_manifest = {
        "kind": "aider-polyglot-cpp-shadow-rubrics",
        "counts": {"tasks": 253},
    }
    manifest_bytes = (json.dumps(source_manifest) + "\n").encode()
    (source / "manifest.json").write_bytes(manifest_bytes)
    root.mkdir()
    with tarfile.open(root / "aider-shadow-rubrics.tar.gz", "w:gz") as handle:
        handle.add(source, arcname="aider_polyglot_cpp_shadow")
    artifact_manifest = {
        "kind": "glm47-aider-shadow-rubrics-archive",
        "archive": "aider-shadow-rubrics.tar.gz",
        "archive_root": "aider_polyglot_cpp_shadow",
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "counts": {"tasks": 253, "files": 254},
    }
    (root / "artifact_manifest.json").write_text(json.dumps(artifact_manifest))

    destination = extract(root)

    assert destination.name == "aider_polyglot_cpp_shadow"
    assert sum(1 for path in destination.rglob(".rubric.json")) == 253


def test_strip_mtp_adapter_filters_served_layers_and_copies_native_state(tmp_path) -> None:
    module = runpy.run_path("scripts/prepare_grpo_adapter.py")
    clear_generated_outputs = module["clear_generated_outputs"]
    copy_native_state = module["copy_native_state"]
    filter_served_layers = module["filter_served_layers"]

    kept, dropped = filter_served_layers(
        {
            "model.layers.46.x": 46,
            "model.layers.47.x": 47,
            "model.layers.50.x": 50,
            "other": 1,
        },
        num_layers=47,
    )
    assert kept == {"model.layers.46.x": 46, "other": 1}
    assert dropped == ["model.layers.47.x", "model.layers.50.x"]

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "adapter_megatron_tp0_pp0.pt").write_bytes(b"native")
    (src / "training_state_rank0.pt").write_bytes(b"state")
    (src / "ignore.txt").write_text("ignore")
    (dst / "training_state_rank7.pt").write_bytes(b"old")
    (dst / "keep.txt").write_text("keep")

    removed = clear_generated_outputs(dst)
    assert [path.name for path in removed] == ["training_state_rank7.pt"]

    native, training_state = copy_native_state(src, dst)
    assert [path.name for path in native] == ["adapter_megatron_tp0_pp0.pt"]
    assert training_state == []
    assert (dst / "adapter_megatron_tp0_pp0.pt").read_bytes() == b"native"
    assert not (dst / "training_state_rank0.pt").exists()
    assert (dst / "keep.txt").read_text() == "keep"
    assert not (dst / "ignore.txt").exists()

    native, training_state = copy_native_state(src, dst, include_training_state=True)
    assert [path.name for path in native] == ["adapter_megatron_tp0_pp0.pt"]
    assert [path.name for path in training_state] == ["training_state_rank0.pt"]
    assert (dst / "training_state_rank0.pt").read_bytes() == b"state"


def test_grpo_runner_save_interval_is_configurable() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert '--save-interval "${MILES_SAVE_INTERVAL:-1}"' in text


def test_grpo_runner_guards_existing_data_from_forced_rebuild() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    guard = 'if [ ! -f "${DATA_DIR}/grpo/train.jsonl" ]; then'
    assert text.count(guard) == 2
    assert text.index(guard) < text.index('BUILD_DATA_ARGS[@]}"')


def test_warm_start_marks_engine_adapter_preloaded() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    class FakeArgs:
        num_layers = 47
        lora_adapter_path = "/some/adapter"

    class FakeArgsNoWarmStart:
        num_layers = 47
        lora_adapter_path = None

    class FakeUpdater:
        def __init__(self, args):
            self.args = args
            self._lora_loaded = False

        def _send_lora_params(self, hf_named_tensors):
            return hf_named_tensors

    fake_module = types.SimpleNamespace(UpdateWeightFromTensor=FakeUpdater)
    miles_glm47_bridge._apply_sglang_lora_mtp_filter(fake_module)

    assert FakeUpdater(FakeArgs())._lora_loaded is True
    assert FakeUpdater(FakeArgsNoWarmStart())._lora_loaded is False


def test_router_circuit_breaker_patch_disables_breaker_and_widens_queue() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    class FakeRouterArgs:
        def __init__(self):
            self.disable_circuit_breaker = False
            self.queue_size = 100
            self.queue_timeout_secs = 60

        @classmethod
        def from_cli_args(cls, args, use_router_prefix=False):
            return cls()

    fake_module = types.SimpleNamespace(RouterArgs=FakeRouterArgs)
    miles_glm47_bridge._apply_router_cb_patch(fake_module)

    router_args = FakeRouterArgs.from_cli_args(object(), use_router_prefix=True)
    assert router_args.disable_circuit_breaker is True
    assert router_args.queue_size == 4096
    assert router_args.queue_timeout_secs == 1800

    # no double wrap
    miles_glm47_bridge._apply_router_cb_patch(fake_module)
    assert FakeRouterArgs.from_cli_args(object()).queue_size == 4096


def test_router_ready_timeout_patch_enforces_floor(monkeypatch) -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    calls = {}

    def fake_wait(host, port, process=None, timeout=30):
        calls["timeout"] = timeout

    fake_module = types.SimpleNamespace(RouterArgs=None, wait_for_server_ready=fake_wait)
    miles_glm47_bridge._apply_router_cb_patch(fake_module)

    monkeypatch.delenv("GLM47_ROUTER_READY_TIMEOUT_S", raising=False)
    fake_module.wait_for_server_ready("h", 1, None, timeout=30)
    assert calls["timeout"] == 300.0

    # An explicit longer caller timeout wins over a smaller floor.
    monkeypatch.setenv("GLM47_ROUTER_READY_TIMEOUT_S", "45")
    fake_module.wait_for_server_ready("h", 1, None, timeout=60)
    assert calls["timeout"] == 60.0

    # Empty env (Ray runtime-env passthrough default) falls back to 300.
    monkeypatch.setenv("GLM47_ROUTER_READY_TIMEOUT_S", "")
    fake_module.wait_for_server_ready("h", 1, None, timeout=30)
    assert calls["timeout"] == 300.0


def test_grpo_runner_exposes_server_concurrency() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'SGLANG_SERVER_CONCURRENCY="${MILES_SGLANG_SERVER_CONCURRENCY:-512}"' in text
    assert '--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"' in text


def test_grpo_runner_eval_prompt_data_is_configurable() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'EVAL_PROMPT_DATA="${MILES_EVAL_PROMPT_DATA:-}"' in text
    assert 'EVAL_NAME="${MILES_EVAL_NAME:-pie_cpp}"' in text
    assert '--eval-prompt-data "${EVAL_NAME}" "${EVAL_PROMPT_DATA:-${DATA_DIR}/eval/validation.jsonl}"' in text


def test_grpo_runner_supports_raw_extra_args() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'read -r -a EXTRA_ARGS <<< "${MILES_EXTRA_ARGS}"' in text


def test_sft_runner_supports_raw_extra_args() -> None:
    text = SFT_RUNNER.read_text(encoding="utf-8")
    assert 'read -r -a EXTRA_ARGS <<< "${MILES_EXTRA_ARGS}"' in text
    assert "extra_args=${MILES_EXTRA_ARGS:-}" in text


def test_miles_runners_expose_h100_throughput_knobs() -> None:
    for script in (SFT_RUNNER, GRPO_RUNNER):
        text = script.read_text(encoding="utf-8")
        assert 'USE_DYNAMIC_BATCH_SIZE="${MILES_USE_DYNAMIC_BATCH_SIZE:-1}"' in text
        assert 'BALANCE_DATA="${MILES_BALANCE_DATA:-1}"' in text
        assert 'PERF_ARGS+=(--use-dynamic-batch-size)' in text
        assert 'PERF_ARGS+=(--balance-data)' in text
        assert "use_dynamic_batch_size=${USE_DYNAMIC_BATCH_SIZE}" in text
        assert "balance_data=${BALANCE_DATA}" in text
        assert 'MOE_ENABLE_DEEPEP="${MILES_MOE_ENABLE_DEEPEP:-0}"' in text
        assert 'SGLANG_MAX_RUNNING_REQUESTS="${MILES_SGLANG_MAX_RUNNING_REQUESTS:-}"' in text
        assert 'SGLANG_DP_SIZE="${MILES_SGLANG_DP_SIZE:-${GPUS_PER_NODE}}"' in text
        assert 'SGLANG_ENABLE_DP_ATTENTION="${MILES_SGLANG_ENABLE_DP_ATTENTION:-0}"' in text
        assert 'SGLANG_ENABLE_DP_LM_HEAD="${MILES_SGLANG_ENABLE_DP_LM_HEAD:-0}"' in text
        assert 'SGLANG_MOE_DENSE_TP_SIZE="${MILES_SGLANG_MOE_DENSE_TP_SIZE:-}"' in text
        assert 'SGLANG_SPECULATIVE="${MILES_SGLANG_SPECULATIVE:-0}"' in text
        assert 'SGLANG_DISABLE_CUSTOM_ALL_REDUCE="${MILES_SGLANG_DISABLE_CUSTOM_ALL_REDUCE:-0}"' in text
        assert 'PERF_ARGS+=(--moe-enable-deepep)' in text
        assert 'SGLANG_ARGS+=(--sglang-enable-dp-attention --sglang-dp-size "${SGLANG_DP_SIZE}")' in text
        assert 'SGLANG_ARGS+=(--sglang-enable-dp-lm-head)' in text
        assert 'SGLANG_ARGS+=(--sglang-moe-dense-tp-size "${SGLANG_MOE_DENSE_TP_SIZE}")' in text
        assert "--sglang-speculative-algorithm EAGLE" in text
        assert 'SGLANG_ARGS+=(--sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}")' in text
        assert 'SGLANG_ARGS+=(--sglang-disable-custom-all-reduce)' in text
        assert "moe_enable_deepep=${MOE_ENABLE_DEEPEP}" in text
        assert "sglang_speculative=${SGLANG_SPECULATIVE}" in text
        assert "utilization.memory,power.draw" in text

    grpo_text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert '\\"NVSHMEM_DISABLE_NCCL\\": \\"${NVSHMEM_DISABLE_NCCL:-}\\"' in grpo_text
    sft_text = SFT_RUNNER.read_text(encoding="utf-8")
    assert '"NVSHMEM_DISABLE_NCCL": os.environ.get("NVSHMEM_DISABLE_NCCL", "")' in sft_text


def test_miles_runners_gate_recompute_behind_env() -> None:
    for script in (SFT_RUNNER, GRPO_RUNNER):
        text = script.read_text(encoding="utf-8")
        assert 'RECOMPUTE_GRANULARITY="${MILES_RECOMPUTE_GRANULARITY:-selective}"' in text
        assert 'case "${RECOMPUTE_GRANULARITY}" in' in text
        assert "PERF_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1)" in text
        assert "PERF_ARGS+=(--recompute-granularity selective)" in text
        assert "recompute_granularity=${RECOMPUTE_GRANULARITY}" in text
        assert "  --recompute-granularity full\n" not in text


def test_runners_gate_docker_preflight_on_sandbox_backend() -> None:
    for script in (SFT_RUNNER, GRPO_RUNNER):
        text = script.read_text(encoding="utf-8")
        assert 'if [ "${GLM47_CPP_SANDBOX_BACKEND:-docker}" != "local" ]; then' in text
        # the docker checks live inside the backend gate, not at top level
        gate = text.index('if [ "${GLM47_CPP_SANDBOX_BACKEND:-docker}" != "local" ]; then')
        docker_check = text.index("Missing docker CLI inside container")
        assert gate < docker_check
    grpo_text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert "GLM47_CPP_SANDBOX_BACKEND=local but g++ is missing" in grpo_text


def test_grpo_runner_prefers_mini_eval_when_present() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert '[ -f "${DATA_DIR}/eval/validation_mini126.jsonl" ]' in text
    assert 'EVAL_PROMPT_DATA="${DATA_DIR}/eval/validation_mini126.jsonl"' in text
    fallback = text.index("validation_mini126.jsonl")
    eval_args = text.index('--eval-prompt-data "${EVAL_NAME}"')
    assert fallback < eval_args


def test_grpo_runner_ref_load_is_optional() -> None:
    text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert 'if [ "${MILES_NO_REF:-0}" != "1" ]; then' in text
    assert text.count('--ref-load "${REF_LOAD_DIR}"') == 1


def test_sft_runner_ref_load_is_optional() -> None:
    text = SFT_RUNNER.read_text(encoding="utf-8")
    assert 'if [ "${MILES_NO_REF:-0}" != "1" ]; then' in text
    assert text.count('--ref-load "${REF_LOAD_DIR}"') == 1


def test_miles_runners_forward_offline_wandb_mode() -> None:
    sft_text = SFT_RUNNER.read_text(encoding="utf-8")
    grpo_text = GRPO_RUNNER.read_text(encoding="utf-8")
    assert '"WANDB_MODE",' in sft_text
    assert '\\"WANDB_MODE\\": \\"${WANDB_MODE:-online}\\"' in grpo_text


def test_warm_start_reloads_optimizer_master_params() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    calls = []

    class FakeOptimizer:
        def reload_model_params(self):
            calls.append("reloaded")

    def fake_load(model, adapter_path, *, optimizer=None, opt_param_scheduler=None):
        return True, 244

    fake_module = types.SimpleNamespace(load_lora_adapter=fake_load)
    miles_glm47_bridge._apply_warm_start_optimizer_reload(fake_module)

    loaded, iteration = fake_module.load_lora_adapter([], "/x", optimizer=FakeOptimizer())
    assert (loaded, iteration) == (True, 244)
    assert calls == ["reloaded"]

    # not-loaded path must not touch the optimizer
    def fake_load_fail(model, adapter_path, *, optimizer=None, opt_param_scheduler=None):
        return False, None

    fake_module2 = types.SimpleNamespace(load_lora_adapter=fake_load_fail)
    miles_glm47_bridge._apply_warm_start_optimizer_reload(fake_module2)
    fake_module2.load_lora_adapter([], "/x", optimizer=FakeOptimizer())
    assert calls == ["reloaded"]


def test_colocate_lora_buffers_suspend_outer_tms_region(monkeypatch) -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    transitions: list[bool] = []
    observations: list[tuple[bool, bool, bool, object]] = []
    created_pools: list[object] = []
    active_pool = None

    class FakeCDLL:
        interesting = True

        def tms_get_interesting_region(self):
            return self.interesting

        def tms_set_interesting_region(self, enabled):
            self.interesting = bool(enabled)
            transitions.append(self.interesting)

    cdll = FakeCDLL()

    class FakeCuda:
        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def MemPool():
            pool = object()
            created_pools.append(pool)
            return pool

        @staticmethod
        @contextmanager
        def use_mem_pool(pool):
            nonlocal active_pool
            previous = active_pool
            active_pool = pool
            try:
                yield
            finally:
                active_pool = previous

    torch_module = types.ModuleType("torch")
    torch_module.cuda = FakeCuda

    memory_saver = types.SimpleNamespace(
        _impl=types.SimpleNamespace(_binary_wrapper=types.SimpleNamespace(cdll=cdll))
    )
    tms_module = types.ModuleType("torch_memory_saver")
    tms_module.torch_memory_saver = memory_saver

    class FakeBuffer:
        def __init__(self, *args, **kwargs):
            observations.append(
                (
                    cdll.interesting,
                    kwargs["disable_param_buffers_cpu_backup"],
                    kwargs["disable_grad_buffers_cpu_backup"],
                    active_pool,
                )
            )

    param_module = types.ModuleType("megatron.core.distributed.param_and_grad_buffer")
    param_module._ParamAndGradBuffer = FakeBuffer
    lora_utils = types.ModuleType("miles.backends.megatron_utils.lora_utils")
    lora_utils._param_grad_buffer_patched = False
    bridge_helpers = types.SimpleNamespace(
        patch_param_grad_buffer_for_colocate_mode_lora=lambda: None
    )

    monkeypatch.setitem(sys.modules, "torch_memory_saver", tms_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.distributed.param_and_grad_buffer",
        param_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "miles.backends.megatron_utils.lora_utils",
        lora_utils,
    )

    miles_glm47_bridge._apply_colocate_lora_tms_region_patch(bridge_helpers)
    bridge_helpers.patch_param_grad_buffer_for_colocate_mode_lora()
    FakeBuffer()
    FakeBuffer()

    assert observations == [
        (False, False, False, created_pools[0]),
        (False, False, False, created_pools[0]),
    ]
    assert len(created_pools) == 1
    assert transitions == [False, True, False, True]
    assert cdll.interesting is True
    assert lora_utils._param_grad_buffer_patched is True


def test_colocate_weight_sync_reloads_and_destroys_process_groups_inside_tms() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    events: list[str] = []

    @contextmanager
    def disable():
        events.append("tms-enter")
        try:
            yield
        finally:
            events.append("tms-exit")

    class FakeActor:
        pass

    fake_module = types.SimpleNamespace(
        MegatronTrainRayActor=FakeActor,
        torch_memory_saver=types.SimpleNamespace(disable=disable),
        nullcontext=lambda: contextmanager(lambda: (yield))(),
        reload_process_groups=lambda: events.append("reload"),
        destroy_process_groups=lambda: events.append("destroy"),
        print_memory=lambda label: events.append(label),
        timer=lambda fn: fn,
        dist=types.SimpleNamespace(get_rank=lambda: 0),
        logger=types.SimpleNamespace(warning=lambda *args: None, info=lambda *args: None),
        ray=types.SimpleNamespace(),
        random=types.SimpleNamespace(),
        get_gloo_group=lambda: None,
        is_lora_enabled=lambda args: False,
    )
    miles_glm47_bridge._apply_colocate_lora_update_tms_scope(fake_module)

    actor = FakeActor()
    actor.args = types.SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=True,
        debug_skip_weight_update=False,
        ci_test=False,
        keep_old_actor=False,
    )
    actor.weight_updater = types.SimpleNamespace(
        update_weights=lambda: events.append("sync")
    )
    info = types.SimpleNamespace(
        rollout_engines=[],
        rollout_engine_lock=None,
        has_new_engines=False,
        engine_gpu_counts=[],
        engine_gpu_offsets=[],
    )

    actor.update_weights(info)

    assert events == [
        "tms-enter",
        "reload",
        "before update_weights",
        "sync",
        "after update_weights",
        "destroy",
        "tms-exit",
    ]


def test_colocate_sleep_allows_rank_metadata_to_be_uninitialized() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    events: list[str] = []

    class FakeActor:
        pass

    fake_module = types.SimpleNamespace(
        MegatronTrainRayActor=FakeActor,
        clear_memory=lambda **kwargs: events.append("clear"),
        destroy_process_groups=lambda: events.append("destroy"),
        print_memory=lambda label: events.append(label),
        timer=lambda fn: fn,
        torch_memory_saver=types.SimpleNamespace(
            pause=lambda **kwargs: events.append("pause")
        ),
        is_lora_enabled=lambda args: False,
        log_cpu_memory=lambda *args: events.append("log-cpu"),
    )
    miles_glm47_bridge._apply_colocate_lora_update_tms_scope(fake_module)

    actor = FakeActor()
    actor.args = types.SimpleNamespace(offload_train=True)

    actor.sleep()

    assert events == [
        "clear",
        "before offload model",
        "destroy",
        "pause",
        "after offload model",
    ]


def test_colocate_weight_sync_destroys_process_groups_before_tms_on_failure() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    events: list[str] = []

    @contextmanager
    def disable():
        events.append("tms-enter")
        try:
            yield
        finally:
            events.append("tms-exit")

    class FakeActor:
        pass

    original_data = object()
    staged_data = object()

    class FakeParam:
        data = original_data

    class FakeCpuTensor:
        @staticmethod
        def to(*, device, non_blocking):
            assert device == "cuda:0"
            assert non_blocking is False
            return staged_data

        @staticmethod
        def numel():
            return 16

        @staticmethod
        def element_size():
            return 2

    param = FakeParam()

    def fail_sync():
        events.append("sync")
        raise RuntimeError("sync failed")

    fake_module = types.SimpleNamespace(
        MegatronTrainRayActor=FakeActor,
        torch_memory_saver=types.SimpleNamespace(disable=disable),
        nullcontext=lambda: contextmanager(lambda: (yield))(),
        reload_process_groups=lambda: events.append("reload"),
        destroy_process_groups=lambda: events.append("destroy"),
        print_memory=lambda label: events.append(label),
        timer=lambda fn: fn,
        torch=types.SimpleNamespace(
            cuda=types.SimpleNamespace(synchronize=lambda: events.append("cuda-sync"))
        ),
        dist=types.SimpleNamespace(get_rank=lambda: 0),
        logger=types.SimpleNamespace(warning=lambda *args: None, info=lambda *args: None),
        ray=types.SimpleNamespace(),
        random=types.SimpleNamespace(),
        get_gloo_group=lambda: None,
        is_lora_enabled=lambda args: True,
    )
    miles_glm47_bridge._apply_colocate_lora_update_tms_scope(fake_module)

    actor = FakeActor()
    actor.args = types.SimpleNamespace(
        debug_train_only=False,
        debug_rollout_only=False,
        offload_train=True,
        debug_skip_weight_update=False,
        ci_test=False,
        keep_old_actor=False,
    )
    actor.weight_updater = types.SimpleNamespace(update_weights=fail_sync)
    actor._glm47_lora_sync_snapshots = [(param, FakeCpuTensor(), "cuda:0")]
    info = types.SimpleNamespace(
        rollout_engines=[],
        rollout_engine_lock=None,
        has_new_engines=False,
        engine_gpu_counts=[],
        engine_gpu_offsets=[],
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        actor.update_weights(info)

    assert events == [
        "tms-enter",
        "reload",
        "cuda-sync",
        "before update_weights",
        "sync",
        "cuda-sync",
        "destroy",
        "tms-exit",
    ]
    assert param.data is original_data


def test_lora_sync_snapshot_copies_unique_adapter_parameters_to_cpu() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    copied: list[tuple[str, bool]] = []

    class FakeParam:
        def __init__(self, device):
            self.device = device

        def detach(self):
            return self

        def to(self, *, device, copy):
            copied.append((device, copy))
            return object()

    adapter = FakeParam("cuda:3")
    lora = FakeParam("cuda:3")
    base = FakeParam("cuda:3")

    class FakeChunk:
        @staticmethod
        def named_parameters():
            return [
                ("decoder.weight", base),
                ("decoder.adapter.linear_in.weight", adapter),
                ("decoder.adapter.linear_in.alias", adapter),
                ("decoder.lora_A", lora),
            ]

    snapshots = miles_glm47_bridge._snapshot_lora_parameters([FakeChunk()])

    assert [(param, device) for param, _, device in snapshots] == [
        (adapter, "cuda:3"),
        (lora, "cuda:3"),
    ]
    assert copied == [("cpu", True), ("cpu", True)]


def test_rollout_data_dp_sharding_keeps_raw_rewards_aligned() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    timer_state = types.SimpleNamespace(seq_lens=None)
    payload = {
        "partition": [3, 0],
        "tokens": ["rank-local-row-3", "rank-local-row-0"],
        "response_lengths": [13, 10],
        "total_lengths": [10, 11, 12, 13],
        "raw_reward": [0.0, 0.25, -0.5, 1.0],
    }
    fake_module = types.SimpleNamespace(
        ray=types.SimpleNamespace(get=lambda inner: dict(inner)),
        Timer=lambda: timer_state,
        split_train_data_by_dp_raw=lambda *args, **kwargs: None,
        process_rollout_data=lambda *args: None,
    )
    miles_glm47_bridge._apply_rollout_data_dp_sharding(fake_module)

    refs = [types.SimpleNamespace(inner=payload), types.SimpleNamespace(inner={})]
    args = types.SimpleNamespace(delay_split_train_data_by_dp=False)
    result = fake_module.process_rollout_data(
        args,
        refs,
        0,
        2,
        witness_info=None,
    )

    assert result["tokens"] == ["rank-local-row-3", "rank-local-row-0"]
    assert result["response_lengths"] == [13, 10]
    assert result["total_lengths"] == [13, 10]
    assert result["raw_reward"] == [0.0, 0.25, -0.5, 1.0]
    assert result["_glm47_local_raw_reward"] == [1.0, 0.0]
    assert timer_state.seq_lens == [10, 11, 12, 13]


def test_rollout_data_dp_sharding_preserves_delayed_witness_split() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    timer_state = types.SimpleNamespace(seq_lens=None)
    captured: dict[str, object] = {}
    payload = {
        "tokens": ["global-row-0", "global-row-1"],
        "total_lengths": [10, 11],
        "raw_reward": [0.25, 1.0],
    }

    def split_train_data_by_dp_raw(args, raw, *, dp_size):
        captured.update(args=args, raw=raw, dp_size=dp_size)
        return [
            {
                **raw,
                "partition": [1],
                "tokens": ["global-row-1"],
            },
            {
                **raw,
                "partition": [0],
                "tokens": ["global-row-0"],
            },
        ]

    fake_module = types.SimpleNamespace(
        ray=types.SimpleNamespace(get=lambda inner: dict(inner)),
        Timer=lambda: timer_state,
        split_train_data_by_dp_raw=split_train_data_by_dp_raw,
        process_rollout_data=lambda *args, **kwargs: None,
    )
    miles_glm47_bridge._apply_rollout_data_dp_sharding(fake_module)

    args = types.SimpleNamespace(delay_split_train_data_by_dp=True)
    ref = types.SimpleNamespace(inner=payload)
    witness_info = types.SimpleNamespace(witness_ids=[101, 102])
    result = fake_module.process_rollout_data(
        args,
        ref,
        0,
        2,
        witness_info=witness_info,
    )

    assert captured == {
        "args": args,
        "raw": {**payload, "seq_witness_ids": [101, 102]},
        "dp_size": 2,
    }
    assert result["tokens"] == ["global-row-1"]
    assert result["total_lengths"] == [11]
    assert result["_glm47_local_raw_reward"] == [1.0]
    assert timer_state.seq_lens == [10, 11]


def test_correct_sample_logging_uses_global_rewards_only_for_passrate() -> None:
    from glm47_posttraining.integrations import miles_glm47_bridge

    views: list[tuple[str, list[float]]] = []

    def original_log_passrate(rollout_id, args, rollout_data):
        del rollout_id, args
        views.append(("passrate", list(rollout_data["raw_reward"])))

    fake_module = types.SimpleNamespace(log_passrate=original_log_passrate)

    def original_log_rollout_data(rollout_id, args, rollout_data):
        views.append(("aggregate", list(rollout_data["raw_reward"])))
        fake_module.log_passrate(rollout_id, args, rollout_data)
        views.append(("correct-samples", list(rollout_data["raw_reward"])))

    fake_module.log_rollout_data = original_log_rollout_data
    miles_glm47_bridge._apply_correct_sample_logging(fake_module)

    global_rewards = [0.0, 0.25, -0.5, 1.0]
    local_rewards = [1.0, 0.0]
    rollout_data = {
        "raw_reward": global_rewards,
        "_glm47_local_raw_reward": local_rewards,
    }
    fake_module.log_rollout_data(
        0,
        types.SimpleNamespace(log_correct_samples=True),
        rollout_data,
    )

    assert views == [
        ("aggregate", local_rewards),
        ("passrate", global_rewards),
        ("correct-samples", local_rewards),
    ]
    assert rollout_data["raw_reward"] is global_rewards
    assert rollout_data["_glm47_local_raw_reward"] is local_rewards
