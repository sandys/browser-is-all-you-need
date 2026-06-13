"""Performance profiling/reporting for w8-biayn (torch-free).

Parses SkyRL's emitted metrics (``timing/*`` Timers and the vLLM Prometheus scraper)
back from a run directory so the kernel-R&D lane can report where step time actually
goes. C++ performance RL is rollout and measurement heavy, so training kernels should
be evaluated against the measured bottleneck rather than assumed to affect total step
time.
"""

from __future__ import annotations

from .report import load_run_metrics, parse_metrics_text, summarize

__all__ = ["load_run_metrics", "parse_metrics_text", "summarize"]
