"""Benchmark ladder for the C++ performance-RL thesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BenchmarkTier = Literal["smoke", "train", "eval", "transfer"]


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    tier: BenchmarkTier
    harness: str
    command: str
    metric: str
    reason: str
    notes: str = ""


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="pie-one-smoke",
        tier="smoke",
        harness="PIE + runtime CPU-time harness + SkyRL bridge",
        command="uv run w8-biayn launch cpp-smoke",
        metric="GLM-5.1 loads and one C++ optimization prompt generates",
        reason="Proves the SkyPilot/GCP/SkyRL bridge before reward-loop work scales.",
    ),
    BenchmarkSpec(
        key="pie-10-task",
        tier="smoke",
        harness="PIE data conversion + Docker C++ harness",
        command="uv run w8-biayn data pie build-tasks --limit 10 ...",
        metric="10 validated tasks with coverage-passing visible and hidden tests",
        reason="Creates the first runnable RL pool slice without exposing oracle solutions to prompts.",
    ),
    BenchmarkSpec(
        key="skyrl-data-bundle",
        tier="smoke",
        harness="Validated task JSON to SkyRL SFT/GRPO data",
        command="uv run w8-biayn data skyrl build --tasks-dir .w8-biayn/data/tasks --out .w8-biayn/data/skyrl",
        metric="SFT JSONL, GRPO parquet, task copies, and checksum manifest are written",
        reason="Makes dataset conversion a reproducible project deliverable before training.",
    ),
    BenchmarkSpec(
        key="pie-heldout-effk",
        tier="eval",
        harness="PIE held-out tasks + gem5 final measurement",
        command="uv run w8-biayn benchmarks show pie-heldout-effk",
        metric="eff@k versus base GLM-5.1 and Claude baselines",
        reason="Primary public claim metric for correctness plus performance consistency.",
        notes="Use ENAMEL's estimator for right-censored timeouts and Mercury Beyond as a secondary metric.",
    ),
    BenchmarkSpec(
        key="sft-cold-start",
        tier="train",
        harness="PIE v0 to v1 supervised fine-tuning",
        command="uv run w8-biayn launch cpp-sft --accelerators A100:8",
        metric="valid-format rate and correct-and-faster rate on held-out PIE tasks",
        reason="Teaches the model the response format and basic C++ optimization before GRPO.",
    ),
    BenchmarkSpec(
        key="grpo-tiny",
        tier="train",
        harness="SkyRL/rLLM GRPO on a small PIE slice",
        command="uv run w8-biayn launch cpp-grpo --accelerators A100:8 --train-batch-size 16 --n-samples-per-prompt 4",
        metric="reward trend up on fixed 10-100 task slice",
        reason="Confirms the C++ reward environment is wired into GRPO before scaling.",
    ),
)


def list_benchmarks() -> tuple[BenchmarkSpec, ...]:
    return BENCHMARKS


def get_benchmark(key: str) -> BenchmarkSpec:
    for benchmark in BENCHMARKS:
        if benchmark.key == key:
            return benchmark
    known = ", ".join(benchmark.key for benchmark in BENCHMARKS)
    raise KeyError(f"Unknown benchmark {key!r}. Known benchmarks: {known}")
