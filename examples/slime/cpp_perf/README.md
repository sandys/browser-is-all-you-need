# SLIME PIE C++ Moonlight RL

This example is the PIE C++ optimization lane for the pinned SLIME sidecar. It
keeps the conservative Moonlight/AIME 4-GPU runtime profile from the ReTool
example, but uses SLIME's stock one-turn generation plus the repo C++ reward
harness instead of the ReTool Python tool loop.

## Files

- `run_moonlight_cpp_perf_rl.sh`: Moonlight SLIME GRPO launcher for PIE C++ tasks.
- `generate_with_cpp_perf.py`: custom reward hook that calls `w8_biayn.cpp_perf.reward.compute_reward`.
- `requirements.txt`: Python packages needed by this example inside the SLIME container.

## 4-GPU Profile

This profile keeps the same conservative AIME runtime configuration, but swaps
the task trajectory and reward from ReTool math to PIE C++ optimization. The
model sees only the PIE `v0` program and visible tests. The SLIME `label` is the
task id, and `metadata.task_path` points to the copied task JSON so the reward
hook can load hidden tests and the `v1` oracle for scoring.

Build PIE tasks and the SLIME JSONL bundle from the repo environment before
starting the SLIME container:

```bash
cd <repo-root>

RUN_ID="r$(date -u +%Y%m%d%H%M%S)"
uv run w8-biayn data pie download --out .w8-biayn/data/pie
uv run w8-biayn data pie prepare-full --source-root .w8-biayn/data/pie --out .w8-biayn/data/pie-full --force
uv run w8-biayn data pie measure-coverage --prepared-root .w8-biayn/data/pie-full --out .w8-biayn/data/pie-full/coverage.json --report-out .w8-biayn/data/pie-full/coverage-report.json
uv run w8-biayn data pie build-full-tasks --prepared-root .w8-biayn/data/pie-full --coverage-json .w8-biayn/data/pie-full/coverage.json --out .w8-biayn/data/tasks-full --min-train 1000 --min-validation 100 --min-test 100 --force
uv run w8-biayn data slime build --tasks-dir .w8-biayn/data/tasks-full --out .w8-biayn/data/slime-pie --profile full-official --run-id "$RUN_ID" --min-train-tasks 1000 --min-validation-tasks 100
uv run w8-biayn cpp harness preflight --cpu 3
```

Then run inside the SLIME container. `SLIME_CUSTOM_GENERATE_FUNCTION_PATH=` is
intentional: C++ optimization uses SLIME's stock one-turn generation, not the
Python-tool ReTool loop.

```bash
cd /workspace/browser-is-all-you-need
pip install -r examples/slime/cpp_perf/requirements.txt

SLIME_PROMPT_DATA=.w8-biayn/data/slime-pie/train.jsonl \
W8_BIAYN_SLIME_TASK_ROOT=.w8-biayn/data/slime-pie \
SLIME_CUSTOM_GENERATE_FUNCTION_PATH= \
SLIME_CUSTOM_RM_PATH=generate_with_cpp_perf.reward_func \
SLIME_REWARD_KEY=score \
SLIME_NUM_GPUS=4 \
SLIME_NUM_ROLLOUT=1 \
SLIME_ROLLOUT_BATCH_SIZE=4 \
SLIME_N_SAMPLES_PER_PROMPT=1 \
SLIME_GLOBAL_BATCH_SIZE=4 \
SLIME_MAX_RESPONSE_LEN=256 \
SLIME_MAX_TOKENS_PER_GPU=4096 \
SLIME_ROLLOUT_NUM_GPUS_PER_ENGINE=4 \
SLIME_SGLANG_MEM_FRACTION=0.45 \
SLIME_SGLANG_CUDA_GRAPH_MAX_BS=16 \
SLIME_SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1 \
SLIME_OPTIMIZER_CPU_OFFLOAD=1 \
SLIME_TRAIN_MEMORY_MARGIN_BYTES=268435456 \
SLIME_RAY_MEMORY_USAGE_THRESHOLD=0.99 \
SLIME_CONVERT_NPROC=4 \
SLIME_DISABLE_EVAL=1 \
SLIME_SAVE_INTERVAL=1000 \
SLIME_WANDB_PROJECT=slime-pie-cpp \
SLIME_WANDB_GROUP=pie-cpp-4gpu-smoke \
SLIME_WANDB_RUN_ID=moonlight-pie-cpp-4gpu-$(date -u +%Y%m%d%H%M%S) \
bash examples/slime/cpp_perf/run_moonlight_cpp_perf_rl.sh
```

The generated SLIME bundle is local state under `.w8-biayn/data/slime-pie/` and
contains `train.jsonl`, `validation.jsonl`, `test.jsonl`, copied task JSON under
`tasks/`, and `_w8_data_manifest.json`. The reward hook uses the host Docker
socket mounted by `.w8-biayn/slime/run-container.sh`; the C++ sandbox image must
be buildable or already present as `w8-biayn-cpp-perf:latest`.
