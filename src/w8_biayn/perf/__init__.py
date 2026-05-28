"""Performance profiling/reporting for w8-biayn (torch-free).

Parses SkyRL's emitted metrics (``timing/*`` Timers and the vLLM Prometheus scraper)
back from a run directory so the kernel-R&D lane can report where step time actually
goes — honestly. The Harbor R3 step is generation-bound, so training-kernel work is
not expected to move ``timing/step``; the report says so.
"""

from __future__ import annotations

from .report import load_run_metrics, parse_metrics_text, summarize

__all__ = ["load_run_metrics", "parse_metrics_text", "summarize"]
