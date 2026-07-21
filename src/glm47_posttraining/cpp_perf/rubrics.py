"""Static rubric templates and simple evaluation-epoch risk adaptation."""

from __future__ import annotations

from typing import Iterable, Mapping


STANDARD = "standard"
STATE_CONCURRENCY = "state_concurrency"
PERFORMANCE_INTENSIVE = "performance_intensive"

RUBRIC_TEMPLATES: dict[str, tuple[str, ...]] = {
    STANDARD: ("correctness", "reasoning", "memory_safety", "cpp_quality"),
    STATE_CONCURRENCY: (
        "correctness",
        "reasoning",
        "memory_safety",
        "thread_safety",
        "cpp_quality",
    ),
    PERFORMANCE_INTENSIVE: (
        "correctness",
        "reasoning",
        "memory_safety",
        "runtime",
        "cpp_quality",
    ),
}
ALL_RUBRICS = tuple(dict.fromkeys(name for names in RUBRIC_TEMPLATES.values() for name in names))
PASS_THRESHOLDS = {
    "correctness": 1.0,
    "reasoning": 0.1,
    "memory_safety": 0.1,
    "thread_safety": 0.1,
    "runtime": 0.1,
    "cpp_quality": 0.1,
}


def active_rubrics(category: str) -> tuple[str, ...]:
    """Return the immutable rubric template for a task category."""

    try:
        return RUBRIC_TEMPLATES[category]
    except KeyError as exc:
        raise ValueError(f"unknown rubric category: {category}") from exc


def normalize_risks(
    category: str,
    risks: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Normalize active non-negative risks; empty input means uniform weights."""

    active = active_rubrics(category)
    if risks:
        unknown = set(risks) - set(ALL_RUBRICS)
        if unknown:
            raise ValueError(f"unknown rubric risks: {sorted(unknown)}")
        selected = {name: max(0.0, float(risks.get(name, 0.0))) for name in active}
    else:
        selected = {name: 1.0 for name in active}
    total = sum(selected.values())
    if total <= 0.0:
        selected = {name: 1.0 for name in active}
        total = float(len(active))
    return {name: value / total for name, value in selected.items()}


def weighted_rubric_reward(
    scores: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    """Compute the single weighted-sum reward used by GRPO."""

    return round(sum(float(weights[name]) * float(scores.get(name, 0.0)) for name in weights), 6)


def update_risks_ema(
    category: str,
    previous_risks: Mapping[str, float],
    pass_rates: Mapping[str, float],
    *,
    alpha: float = 0.8,
) -> dict[str, float]:
    """Update active risks once per evaluation epoch using failure-rate EMA."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    updated: dict[str, float] = {}
    for name in active_rubrics(category):
        old = max(0.0, float(previous_risks.get(name, 0.5)))
        pass_rate = min(1.0, max(0.0, float(pass_rates.get(name, 0.0))))
        updated[name] = alpha * old + (1.0 - alpha) * (1.0 - pass_rate)
    return updated


def rubric_pass_rates(records: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Derive per-rubric pass rates from flattened evaluation records."""

    outcomes: dict[str, list[bool]] = {name: [] for name in ALL_RUBRICS}
    for record in records:
        scores = record.get("rubric_scores")
        score_map = scores if isinstance(scores, Mapping) else {}
        active = record.get("active_rubrics")
        names = active if isinstance(active, list) else list(ALL_RUBRICS)
        for raw_name in names:
            name = str(raw_name)
            if name not in outcomes:
                continue
            raw_score = score_map.get(name, record.get(f"rubric_{name}"))
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                outcomes[name].append(float(raw_score) >= PASS_THRESHOLDS[name])
    return {
        name: sum(values) / len(values)
        for name, values in outcomes.items()
        if values
    }
