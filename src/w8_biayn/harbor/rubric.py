from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_FACT_TYPES = frozenset(
    {
        "domdiff_subscore",
        "dom_element_present",
        "dom_element_attribute_equals",
        "dom_element_style_property",
        "signal_threshold",
    }
)

_ALLOWED_DOMDIFF_KEYS = frozenset(
    {
        "structuralSimilarity",
        "textSimilarity",
        "layoutSimilarity",
        "styleSimilarity",
    }
)


@dataclass(frozen=True)
class FactResult:
    fact_id: str
    fact_type: str
    confidence: float
    satisfied: bool
    reason: str


@dataclass(frozen=True)
class RubricEvaluation:
    score: float
    pass_threshold: float
    passed: bool
    domdiff_total: float
    domdiff_summary: dict[str, float]
    group_scores: dict[str, float]
    fact_results: list[FactResult]


def validate_harbor_rubric(rubric: dict[str, Any]) -> None:
    fact_groups = rubric.get("fact_groups")
    if not isinstance(fact_groups, list) or not fact_groups:
        raise ValueError("rubric.json must contain non-empty fact_groups")

    fact_ids: set[str] = set()
    fact_type_by_id: dict[str, str] = {}
    group_names: set[str] = set()
    fact_groups_by_name: dict[str, list[dict[str, Any]]] = {}
    for group in fact_groups:
        group_name = str(group.get("name", "")).strip()
        if not group_name:
            raise ValueError("rubric groups must have a non-empty name")
        group_names.add(group_name)
        facts = group.get("facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"rubric group {group_name} has no facts")
        fact_groups_by_name[group_name] = facts
        for fact in facts:
            fact_id = str(fact.get("id", "")).strip()
            if not fact_id:
                raise ValueError(f"rubric group {group_name} has fact without id")
            fact_ids.add(fact_id)
            fact_type = str(fact.get("type", "")).strip()
            if fact_type not in SUPPORTED_FACT_TYPES:
                raise ValueError(f"unsupported Harbor rubric fact type: {fact_type}")
            fact_type_by_id[fact_id] = fact_type

    pass_requirements = rubric.get("pass_requirements", {})
    if not isinstance(pass_requirements, dict):
        raise ValueError("pass_requirements must be an object when provided")

    required_fact_ids = pass_requirements.get("required_fact_ids", [])
    if required_fact_ids:
        if not isinstance(required_fact_ids, list):
            raise ValueError("pass_requirements.required_fact_ids must be a list")
        missing_fact_ids = sorted(
            str(fact_id)
            for fact_id in required_fact_ids
            if str(fact_id).strip() not in fact_ids
        )
        if missing_fact_ids:
            raise ValueError(f"pass_requirements references unknown fact ids: {missing_fact_ids}")

    required_group_ids = pass_requirements.get("required_group_ids", [])
    if required_group_ids:
        if not isinstance(required_group_ids, list):
            raise ValueError("pass_requirements.required_group_ids must be a list")
        missing_group_ids = sorted(
            str(group_id)
            for group_id in required_group_ids
            if str(group_id).strip() not in group_names
        )
        if missing_group_ids:
            raise ValueError(f"pass_requirements references unknown group ids: {missing_group_ids}")

    domdiff_floors = pass_requirements.get("domdiff_floors", {})
    if domdiff_floors:
        if not isinstance(domdiff_floors, dict):
            raise ValueError("pass_requirements.domdiff_floors must be an object")
        unknown = sorted(str(key) for key in domdiff_floors if str(key) not in _ALLOWED_DOMDIFF_KEYS)
        if unknown:
            raise ValueError(f"pass_requirements.domdiff_floors has unknown keys: {unknown}")
    del fact_groups_by_name, fact_type_by_id


def evaluate_harbor_rubric(
    rubric: dict[str, Any],
    *,
    domdiff_summary: dict[str, Any] | None,
    dom_state: dict[str, Any] | None,
    signal_metrics: dict[str, Any] | None,
) -> RubricEvaluation:
    validate_harbor_rubric(rubric)
    domdiff_summary = {
        str(key): _clamp_unit_float(value)
        for key, value in (domdiff_summary or {}).items()
    }
    dom_state = dom_state or {}
    signal_metrics = signal_metrics or {}
    pass_threshold = _clamp_unit_float(rubric.get("pass_threshold", 0.85))

    fact_results: list[FactResult] = []
    weighted_total = 0.0
    total_weight = 0.0
    group_scores: dict[str, float] = {}

    for group in rubric.get("fact_groups", []):
        group_name = str(group.get("name", "")).strip()
        group_weight = _positive_float(group.get("weight", 1.0))
        group_score = 0.0
        group_fact_weight = 0.0
        for fact in group.get("facts", []):
            result = _evaluate_fact(
                fact,
                domdiff_summary=domdiff_summary,
                dom_state=dom_state,
                signal_metrics=signal_metrics,
            )
            fact_results.append(result)
            fact_weight = _positive_float(fact.get("weight", 1.0))
            group_score += fact_weight * result.confidence
            group_fact_weight += fact_weight
        if group_fact_weight <= 0:
            continue
        normalized_group_score = group_score / group_fact_weight
        group_scores[group_name] = normalized_group_score
        weighted_total += group_weight * normalized_group_score
        total_weight += group_weight

    score = 0.0 if total_weight <= 0 else weighted_total / total_weight
    domdiff_total = _compute_domdiff_total(domdiff_summary)

    pass_requirements = rubric.get("pass_requirements", {}) if isinstance(rubric.get("pass_requirements", {}), dict) else {}
    required_fact_ids = {
        str(fact_id).strip()
        for fact_id in pass_requirements.get("required_fact_ids", [])
        if str(fact_id).strip()
    }
    required_group_ids = {
        str(group_id).strip()
        for group_id in pass_requirements.get("required_group_ids", [])
        if str(group_id).strip()
    }
    domdiff_floors = {
        str(key): _clamp_unit_float(value)
        for key, value in pass_requirements.get("domdiff_floors", {}).items()
    }

    fact_map = {result.fact_id: result for result in fact_results}
    required_facts_ok = all(
        fact_id in fact_map and fact_map[fact_id].satisfied
        for fact_id in required_fact_ids
    )
    required_groups_ok = all(group_scores.get(group_id, 0.0) > 0.0 for group_id in required_group_ids)
    domdiff_floors_ok = all(
        domdiff_summary.get(key, 0.0) >= threshold
        for key, threshold in domdiff_floors.items()
    )

    return RubricEvaluation(
        score=score,
        pass_threshold=pass_threshold,
        passed=score >= pass_threshold and required_facts_ok and required_groups_ok and domdiff_floors_ok,
        domdiff_total=domdiff_total,
        domdiff_summary={key: domdiff_summary.get(key, 0.0) for key in _ALLOWED_DOMDIFF_KEYS},
        group_scores=group_scores,
        fact_results=fact_results,
    )


def _evaluate_fact(
    fact: dict[str, Any],
    *,
    domdiff_summary: dict[str, float],
    dom_state: dict[str, Any],
    signal_metrics: dict[str, Any],
) -> FactResult:
    fact_id = str(fact.get("id", "fact")).strip() or "fact"
    fact_type = str(fact.get("type", "")).strip()

    if fact_type == "domdiff_subscore":
        key = str(fact.get("key", "")).strip()
        score = _clamp_unit_float(domdiff_summary.get(key, 0.0))
        return FactResult(
            fact_id=fact_id,
            fact_type=fact_type,
            confidence=score,
            satisfied=score > 0.0,
            reason=f"{key}={score:.4f}" if key else "missing key",
        )

    if fact_type == "dom_element_present":
        selector = str(fact.get("selector", "")).strip()
        matched_nodes = _matching_dom_nodes(selector, dom_state)
        matched = bool(matched_nodes)
        return FactResult(
            fact_id=fact_id,
            fact_type=fact_type,
            confidence=1.0 if matched else 0.0,
            satisfied=matched,
            reason=f"selector={selector}" if matched else f"selector_missing={selector}",
        )

    if fact_type == "dom_element_attribute_equals":
        selector = str(fact.get("selector", "")).strip()
        attribute = str(fact.get("attribute", "")).strip().lower()
        expected_value = str(fact.get("expected_value", "")).strip()
        matched_nodes = _matching_dom_nodes(selector, dom_state)
        for node in matched_nodes:
            actual_value = _node_attribute_value(node, attribute)
            if actual_value == expected_value:
                return FactResult(
                    fact_id=fact_id,
                    fact_type=fact_type,
                    confidence=1.0,
                    satisfied=True,
                    reason=f"{attribute}={actual_value}",
                )
        actual_value = _node_attribute_value(matched_nodes[0], attribute) if matched_nodes else ""
        return FactResult(
            fact_id=fact_id,
            fact_type=fact_type,
            confidence=0.0,
            satisfied=False,
            reason=(
                f"{attribute}={actual_value} expected {expected_value}"
                if matched_nodes
                else f"selector_missing={selector}"
            ),
        )

    if fact_type == "dom_element_style_property":
        selector = str(fact.get("selector", "")).strip()
        property_name = str(fact.get("property", "")).strip().lower()
        expected_value = str(fact.get("expected_value", "")).strip()
        matched_nodes = _matching_dom_nodes(selector, dom_state)
        for node in matched_nodes:
            actual_value = _node_style_property(node, property_name)
            if actual_value == expected_value:
                return FactResult(
                    fact_id=fact_id,
                    fact_type=fact_type,
                    confidence=1.0,
                    satisfied=True,
                    reason=f"{property_name}={actual_value}",
                )
        actual_value = _node_style_property(matched_nodes[0], property_name) if matched_nodes else ""
        return FactResult(
            fact_id=fact_id,
            fact_type=fact_type,
            confidence=0.0,
            satisfied=False,
            reason=(
                f"{property_name}={actual_value} expected {expected_value}"
                if matched_nodes
                else f"selector_missing={selector}"
            ),
        )

    if fact_type == "signal_threshold":
        metric = str(fact.get("metric", "")).strip()
        op = str(fact.get("comparison", "gte")).strip() or "gte"
        threshold = _raw_float(fact.get("threshold", 0.0))
        value = _raw_float(signal_metrics.get(metric, 0.0))
        satisfied = value <= threshold if op == "lte" else value >= threshold
        return FactResult(
            fact_id=fact_id,
            fact_type=fact_type,
            confidence=1.0 if satisfied else 0.0,
            satisfied=satisfied,
            reason=f"{metric}={value:.4f} {op} {threshold:.4f}",
        )

    raise ValueError(f"unsupported Harbor rubric fact type: {fact_type}")


def _matching_dom_nodes(selector: str, dom_state: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = dom_state.get("nodes", [])
    if not selector or not isinstance(nodes, list):
        return []
    selector_lower = selector.lower()
    has_combinator = " " in selector or ">" in selector or "+" in selector or "~" in selector
    parsed = _parse_simple_selector(selector_lower)
    matches: list[dict[str, Any]] = []

    for node in nodes:
        css_selector = str(node.get("cssSelector", "")).lower()
        if selector_lower in css_selector:
            matches.append(node)
            continue
        if has_combinator:
            continue
        if _node_matches_simple_selector(node, parsed):
            matches.append(node)
    return matches


def _node_matches_simple_selector(node: dict[str, Any], parsed: dict[str, Any]) -> bool:
    tag = str(node.get("tagName", "")).lower()
    if parsed["tag"] and parsed["tag"] != tag:
        return False

    attr_map = _node_attributes(node)
    node_classes = {cls.lower() for cls in str(attr_map.get("class", "")).split() if cls}
    if not set(parsed["classes"]).issubset(node_classes):
        return False

    for attr_name, operator, expected_value in parsed["attrs"]:
        actual_value = attr_map.get(attr_name)
        if actual_value is None:
            return False
        actual_text = str(actual_value).lower()
        if operator == "exists":
            continue
        if operator == "=":
            if expected_value is None or actual_text != expected_value:
                return False
            continue
        if operator == "*=":
            if expected_value is None or expected_value not in actual_text:
                return False
            continue
        return False

    return True


def _node_attributes(node: dict[str, Any]) -> dict[str, str]:
    attrs = node.get("attributes", [])
    if isinstance(attrs, dict):
        return {str(k).lower(): str(v) for k, v in attrs.items()}

    flat: dict[str, str] = {}
    if isinstance(attrs, list):
        if attrs and all(isinstance(item, dict) for item in attrs):
            for item in attrs:
                key = str(item.get("name", "")).lower().strip()
                if not key:
                    continue
                value = item.get("value")
                flat[key] = "" if value is None else str(value)
            return flat
        for i in range(0, len(attrs) - 1, 2):
            key = str(attrs[i]).lower()
            value = attrs[i + 1]
            flat[key] = "" if value is None else str(value)
    return flat


def _node_attribute_value(node: dict[str, Any], attribute: str) -> str:
    return _node_attributes(node).get(attribute.lower(), "")


def _node_style_property(node: dict[str, Any], property_name: str) -> str:
    property_name = property_name.strip().lower()
    if not property_name:
        return ""
    attributes = _node_attributes(node)
    inline_style = attributes.get("style", "")
    for chunk in inline_style.split(";"):
        if ":" not in chunk:
            continue
        key, value = chunk.split(":", 1)
        if key.strip().lower() == property_name:
            return value.strip()
    key_styles = node.get("keyStyles", {}) or {}
    if isinstance(key_styles, dict):
        for key, value in key_styles.items():
            if str(key).strip().lower() == property_name:
                return str(value).strip()
    return ""


def _parse_simple_selector(selector: str) -> dict[str, Any]:
    tag_match = re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*", selector)
    tag = tag_match.group(0).lower() if tag_match else ""
    classes = re.findall(r"\.([a-zA-Z0-9_-]+)", selector)
    attrs: list[tuple[str, str, str | None]] = []
    for match in re.finditer(r"\[([^\]\*\=~\^\$\|]+)(\*=|=)?(?:['\"]?([^'\"\]]+)['\"]?)?\]", selector):
        attrs.append(
            (
                match.group(1).strip().lower(),
                match.group(2) or "exists",
                match.group(3).lower() if match.group(3) else None,
            )
        )
    return {
        "tag": tag,
        "classes": classes,
        "attrs": attrs,
    }


def validate_required_assertions_contract(
    *,
    fact_groups_by_name: dict[str, list[dict[str, Any]]],
    required_fact_ids: list[Any],
    required_group_ids: list[Any],
    fact_type_by_id: dict[str, str],
) -> None:
    required_group_id_set = {str(group_id).strip() for group_id in required_group_ids if str(group_id).strip()}
    if "required_assertions" not in required_group_id_set:
        raise ValueError("pass_requirements.required_group_ids must include required_assertions")
    required_assertions = fact_groups_by_name.get("required_assertions", [])
    if len(required_assertions) < 2:
        raise ValueError("required_assertions must contain at least 2 facts")
    required_fact_id_set = {str(fact_id).strip() for fact_id in required_fact_ids if str(fact_id).strip()}
    if len(required_fact_id_set) < 2:
        raise ValueError("pass_requirements.required_fact_ids must contain at least 2 facts")
    if not any(
        fact_type_by_id.get(fact_id, "") in {"dom_element_attribute_equals", "dom_element_style_property"}
        for fact_id in required_fact_id_set
    ):
        raise ValueError("required_assertions must include at least one attribute- or style-based required fact")


def _compute_domdiff_total(summary: dict[str, float]) -> float:
    structural = summary.get("structuralSimilarity", 0.0)
    text = summary.get("textSimilarity", 0.0)
    layout = summary.get("layoutSimilarity", 0.0)
    style = summary.get("styleSimilarity", 0.0)
    return (
        0.35 * structural
        + 0.15 * text
        + 0.30 * layout
        + 0.20 * style
    )


def _positive_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return parsed if parsed > 0 else 1.0


def _raw_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed


def _clamp_unit_float(value: Any) -> float:
    parsed = _raw_float(value)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed
