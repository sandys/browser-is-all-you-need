"""Benchmark scorecard for the browser-to-mobile RL thesis."""

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
        key="miniwob-smoke",
        tier="smoke",
        harness="BrowserGym + SkyRL",
        command="uv run w8-biayn launch miniwob",
        metric="task success / reward",
        reason="Cheap end-to-end SkyPilot/SkyRL sanity check before spending DOMDiff or Tinker budget.",
    ),
    BenchmarkSpec(
        key="domdiff-local-live",
        tier="smoke",
        harness="w8-biayn DOMDiff",
        command="uv run w8-biayn domdiff local smoke --image android-world-domdiff:local",
        metric="local KVM + CDP + quick tunnel + reward adapter health + live DOMDiff evaluation",
        reason="Proves the prebuilt WootzApp/Android image works locally before spending GCP training budget.",
    ),
    BenchmarkSpec(
        key="webvoyager-domdiff-heldout",
        tier="eval",
        harness="SkyRL + WebVoyager DOMDiff",
        command=(
            "uv run w8-biayn launch r3 --with-local-domdiff "
            "--benchmark webvoyager-domdiff-heldout"
        ),
        metric="judge success rate on held-out WebVoyager no-anti-bot tasks",
        reason="Primary browser-use benchmark for live websites through the mobile browser target.",
        notes="Use the SWE-RL held-out no-anti-bot split as the first target dataset.",
    ),
    BenchmarkSpec(
        key="harbor-domdiff-browser-swe",
        tier="eval",
        harness="Harbor + SkyRL + DOMDiff reward",
        command=(
            "uv run w8-biayn launch r3 --with-local-domdiff "
            "--benchmark harbor-domdiff-browser-swe"
        ),
        metric="rubric reward and DOMDiff success over converted browser/SWE tasks",
        reason="Exercises app-building/browser-preview tasks where DOMDiff is a verifier, not just an observation channel.",
        notes=(
            "Uses the two packaged SWE-RL Harbor tasks with definitive DOMDiff rubrics. "
            "Task containers run on the GCP trainer VM; the local DOMDiff reward service is reached through Cloudflare."
        ),
    ),
    BenchmarkSpec(
        key="webarena-browsergym",
        tier="eval",
        harness="BrowserGym WebArena",
        command=(
            "uv run w8-biayn launch webarena "
            "--webarena-archives-gcs gs://<bucket>/webarena"
        ),
        metric="WebArena task success",
        reason="Reproducible self-hosted web benchmark that is less live-site brittle than WebVoyager.",
    ),
    BenchmarkSpec(
        key="androidworld-transfer",
        tier="transfer",
        harness="AndroidWorld",
        command="uv run w8-biayn benchmarks show androidworld-transfer",
        metric="AndroidWorld task success rate",
        reason="Direct test of the pitch that browser-trained policies transfer to mobile-use.",
        notes="Run after browser benchmarks; use the DOMDiff image or `devjangid/android-world-mobile-tablet` as the target.",
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
