"""Parse SkyRL training metrics from a run directory (torch-free).

SkyRL logs ``timing/*`` (Timer) and vLLM Prometheus metrics. This reads them back
from a run directory — a structured ``metrics.json`` and/or scraped ``*.log`` files —
so kernel or trainer experiments can show where the step time goes. C++ performance
RL is usually rollout and measurement heavy, so this report should be used to identify
the actual bottleneck instead of assuming a training-kernel speedup moves end-to-end
time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Metric namespaces SkyRL emits that are worth surfacing in a perf report.
METRIC_PREFIXES = ("timing/", "reward/", "environment/", "vllm/")

# Matches `key=value` or `key: value` with a numeric value (trailing units like the
# README's `670.2853s` are ignored — only the leading number is captured).
_KV = re.compile(r"([A-Za-z0-9_./@+-]+)\s*[=:]\s*(-?\d+(?:\.\d+)?)")


def parse_metrics_text(text: str) -> dict[str, float]:
    """Extract namespaced numeric metrics from free-form log text."""

    out: dict[str, float] = {}
    for key, value in _KV.findall(text):
        if key.startswith(METRIC_PREFIXES):
            try:
                out[key] = float(value)
            except ValueError:
                continue
    return out


def _flatten(obj: object, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(_flatten(value, f"{path}/"))
            elif isinstance(value, bool):
                continue
            elif isinstance(value, (int, float)):
                flat[path] = float(value)
    return flat


def load_run_metrics(run_dir: str | Path) -> dict[str, float]:
    """Load metrics from ``metrics.json`` and/or any ``*.log`` under ``run_dir``."""

    run = Path(run_dir)
    if not run.exists():
        raise FileNotFoundError(f"run directory not found: {run}")

    metrics: dict[str, float] = {}
    metrics_json = run / "metrics.json"
    if metrics_json.is_file():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        for key, value in _flatten(data).items():
            if key.startswith(METRIC_PREFIXES):
                metrics[key] = value
    for log in sorted(run.rglob("*.log")):
        metrics.update(parse_metrics_text(log.read_text(encoding="utf-8", errors="ignore")))
    return metrics


def summarize(metrics: dict[str, float]) -> dict[str, float]:
    """Derive the fractions that matter for the kernel-lane honesty check."""

    summary: dict[str, float] = {}
    step = metrics.get("timing/step")
    if step:
        generate = metrics.get("timing/generate")
        if generate is not None:
            summary["generation_fraction"] = generate / step
        train = metrics.get("timing/train_critic_and_policy")
        if train is not None:
            summary["train_fraction"] = train / step
    return summary
