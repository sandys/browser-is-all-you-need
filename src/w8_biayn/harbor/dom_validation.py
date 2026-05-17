from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_ERROR_TEXT_MARKERS = (
    '"page":"/_error"',
    '"statuscode":500',
    "module not found:",
    "application error",
    "internal server error",
    "chunkloaderror",
    "daytona preview - warning",
    "preview url warning",
)

_IGNORE_TAGS = {
    "html",
    "head",
    "meta",
    "link",
    "style",
    "script",
    "noscript",
    "path",
    "svg",
}

_IGNORE_SELECTOR_MARKERS = (
    "__next-route-announcer__",
    "__next-build-watcher",
    "nextjs-portal",
    "nextjs-toast",
    "nextjs-dev-tools",
    "react-devtools",
)


@dataclass(frozen=True)
class DOMValidationResult:
    valid: bool
    reason: str
    meaningful_node_count: int
    markers: tuple[str, ...] = field(default_factory=tuple)


def validate_dom_state(dom_state: dict[str, Any] | None) -> DOMValidationResult:
    if not isinstance(dom_state, dict):
        return DOMValidationResult(False, "dom_state_missing", 0)

    nodes = dom_state.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return DOMValidationResult(False, "dom_state_nodes_missing", 0)

    matched_markers = tuple(_find_error_markers(dom_state, nodes))
    if matched_markers:
        return DOMValidationResult(
            False,
            f"dom_state_error_markers={','.join(matched_markers)}",
            _meaningful_node_count(nodes),
            matched_markers,
        )

    meaningful_node_count = _meaningful_node_count(nodes)
    if meaningful_node_count < 10:
        return DOMValidationResult(
            False,
            f"dom_state_too_sparse(meanings={meaningful_node_count})",
            meaningful_node_count,
        )

    return DOMValidationResult(True, "ok", meaningful_node_count)


def _find_error_markers(dom_state: dict[str, Any], nodes: list[dict[str, Any]]) -> list[str]:
    haystacks: list[str] = []
    for value in (
        dom_state.get("url", ""),
        dom_state.get("title", ""),
        dom_state.get("textContent", ""),
    ):
        text = str(value or "").strip().lower()
        if text:
            haystacks.append(text)

    for node in nodes:
        for key in ("textContent", "name", "value", "title", "url"):
            text = str(node.get(key, "") or "").strip().lower()
            if text:
                haystacks.append(text)

    joined = "\n".join(haystacks)
    return [marker for marker in _ERROR_TEXT_MARKERS if marker in joined]


def _meaningful_node_count(nodes: list[dict[str, Any]]) -> int:
    count = 0
    for node in nodes:
        if _is_meaningful_node(node):
            count += 1
    return count


def _is_meaningful_node(node: dict[str, Any]) -> bool:
    selector = str(node.get("cssSelector", "") or "").lower()
    if any(marker in selector for marker in _IGNORE_SELECTOR_MARKERS):
        return False

    tag = str(node.get("tagName") or node.get("nodeName") or "").strip().lower()
    if tag in _IGNORE_TAGS:
        return False

    if tag == "#text":
        text = str(node.get("textContent", "") or "").strip()
        return bool(text)

    return True
