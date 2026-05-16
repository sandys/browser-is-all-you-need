"""Live ChromiumRL CDP helpers used by the w8-biayn reward service.

The current V1 reward path runs DOM/state comparison inside the ChromiumRL
worker and returns reward facts over HTTP. This module intentionally keeps only
the low-level CDP commands needed by that worker and by ``CDPPool``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


FEEDBACK_MISSING = -1


class ChromiumRLNotAvailableError(Exception):
    """Raised when WootzApp's ChromiumRL CDP domain is not available."""


class ChromiumRLSignalCollectionError(Exception):
    """Raised when ChromiumRL signal collection fails in strict mode."""


@dataclass(frozen=True)
class ChromiumRLScoringConfig:
    """Thresholds used by the live ChromiumRL worker quality summary."""

    layout_time_max_ms: float = 200.0
    cls_max_score: float = 0.25
    paint_time_max_ms: float = 100.0
    style_recalc_max_ms: float = 50.0
    layer_count_good: int = 5
    layer_count_poor: int = 30
    gpu_memory_good_mb: float = 10.0
    gpu_memory_poor_mb: float = 100.0
    lcp_good_ms: float = 1000.0
    lcp_poor_ms: float = 4000.0
    abs_layout_weight: float = 0.15
    abs_cls_weight: float = 0.15
    abs_compositor_weight: float = 0.10
    abs_touch_weight: float = 0.10
    abs_paint_weight: float = 0.10
    abs_style_weight: float = 0.10
    abs_layer_weight: float = 0.10
    abs_gpu_weight: float = 0.10
    abs_lcp_weight: float = 0.10


async def verify_chromiumrl_available(
    cdp_client,
    session_id: Optional[str] = None,
) -> None:
    """Verify the custom ChromiumRL CDP domain is available."""

    try:
        await cdp_client.send("ChromiumRL.enable", {}, session_id=session_id)
        await cdp_client.send("ChromiumRL.disable", {}, session_id=session_id)
        logger.info("ChromiumRL domain verified available")
    except Exception as exc:
        raise ChromiumRLNotAvailableError(
            f"WootzApp with ChromiumRL domain required. Got: {exc}"
        ) from exc


async def enable_chromiumrl_tracing(
    cdp_client,
    capture_touch_traces: bool = True,
    capture_layout_timings: bool = True,
    capture_cls_attribution: bool = True,
    capture_compositor_layers: bool = True,
    session_id: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> str:
    """Enable ChromiumRL tracing and return the ChromiumRL tracing session id."""

    params = {
        "captureTouchTraces": capture_touch_traces,
        "captureLayoutTimings": capture_layout_timings,
        "captureCLSAttribution": capture_cls_attribution,
        "captureCompositorLayers": capture_compositor_layers,
    }
    if timeout_s is not None and timeout_s > 0:
        result = await cdp_client.send(
            "ChromiumRL.enable",
            params,
            timeout=float(timeout_s),
            session_id=session_id,
        )
    else:
        result = await cdp_client.send(
            "ChromiumRL.enable",
            params,
            session_id=session_id,
        )
    chromiumrl_session_id = result.get("sessionId")
    if chromiumrl_session_id is None:
        raise RuntimeError(
            "ChromiumRL.enable returned no sessionId key "
            f"(result={result})"
        )
    normalized = str(chromiumrl_session_id).strip()
    if not normalized:
        raise RuntimeError(
            "ChromiumRL.enable returned empty sessionId "
            f"(result={result})"
        )
    return normalized


async def disable_chromiumrl_tracing(
    cdp_client,
    session_id: Optional[str] = None,
) -> None:
    """Disable ChromiumRL tracing."""

    await cdp_client.send("ChromiumRL.disable", {}, session_id=session_id)


async def collect_touch_traces(
    cdp_client,
    root_node_id: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect ChromiumRL touch event traces."""

    params: dict[str, Any] = {}
    if root_node_id is not None:
        params["rootNodeId"] = root_node_id
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    result = await cdp_client.send("ChromiumRL.getTouchTraces", params, session_id=session_id)
    return {
        "traces": result.get("traces", []),
        "totalEvents": result.get("totalEvents", 0),
        "captureDuration": result.get("captureDuration", 0),
    }


async def collect_layout_timings(
    cdp_client,
    frame_id: str,
    threshold_ms: Optional[float] = None,
    include_zero_time: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect ChromiumRL per-element layout timings."""

    params: dict[str, Any] = {"frameId": frame_id}
    if threshold_ms is not None:
        params["thresholdMs"] = threshold_ms
    if include_zero_time:
        params["includeZeroTime"] = include_zero_time

    result = await cdp_client.send("ChromiumRL.getLayoutTimings", params, session_id=session_id)
    return {
        "elements": result.get("elements", []),
        "frameMetrics": result.get("frameMetrics", {}),
    }


async def collect_cls_attribution(
    cdp_client,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    include_sources: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect ChromiumRL CLS attribution entries."""

    params: dict[str, Any] = {"includeSources": include_sources}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    result = await cdp_client.send("ChromiumRL.getCLSAttribution", params, session_id=session_id)
    return {
        "entries": result.get("entries", []),
        "totalCLS": result.get("totalCLS", 0.0),
        "exceedsGoodThreshold": result.get("exceedsGoodThreshold", False),
    }


async def collect_compositor_layers(
    cdp_client,
    include_paint_info: bool = True,
    include_transforms: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect ChromiumRL compositor layer information."""

    result = await cdp_client.send(
        "ChromiumRL.getCompositorLayers",
        {
            "includePaintInfo": include_paint_info,
            "includeTransforms": include_transforms,
        },
        session_id=session_id,
    )
    return {
        "layers": result.get("layers", []),
        "totalGPUMemory": result.get("totalGPUMemory", 0),
        "pendingTextureUploads": result.get("pendingTextureUploads", 0),
    }


async def get_main_frame_id(
    cdp_client,
    session_id: Optional[str] = None,
    command_timeout_s: Optional[float] = None,
) -> str:
    """Return the root frame id for ChromiumRL layout collection."""

    if command_timeout_s is not None and command_timeout_s > 0:
        result = await cdp_client.send(
            "Page.getFrameTree",
            {},
            timeout=float(command_timeout_s),
            session_id=session_id,
        )
    else:
        result = await cdp_client.send("Page.getFrameTree", {}, session_id=session_id)
    frame_tree = result.get("frameTree", {})
    root_frame = frame_tree.get("frame", {})
    return root_frame.get("id", "")


async def navigate_and_enable_chromiumrl(
    cdp_client,
    url: str,
    wait_time: float = 2.0,
    capture_touch_traces: bool = True,
    capture_layout_timings: bool = True,
    capture_cls_attribution: bool = True,
    capture_compositor_layers: bool = True,
    session_id: Optional[str] = None,
    cdp_command_timeout_s: Optional[float] = None,
    enable_timeout_s: Optional[float] = None,
) -> str:
    """Navigate, enable ChromiumRL on the warmed renderer, then reload."""

    if cdp_command_timeout_s is not None and cdp_command_timeout_s > 0:
        await cdp_client.send(
            "Page.navigate",
            {"url": url},
            timeout=float(cdp_command_timeout_s),
            session_id=session_id,
        )
    else:
        await cdp_client.send("Page.navigate", {"url": url}, session_id=session_id)

    await asyncio.sleep(wait_time)
    effective_enable_timeout: Optional[float] = None
    if enable_timeout_s is not None and enable_timeout_s > 0:
        effective_enable_timeout = float(enable_timeout_s)
    elif cdp_command_timeout_s is not None and cdp_command_timeout_s > 0:
        effective_enable_timeout = float(cdp_command_timeout_s)

    chromiumrl_session_id = await enable_chromiumrl_tracing(
        cdp_client,
        capture_touch_traces=capture_touch_traces,
        capture_layout_timings=capture_layout_timings,
        capture_cls_attribution=capture_cls_attribution,
        capture_compositor_layers=capture_compositor_layers,
        session_id=session_id,
        timeout_s=effective_enable_timeout,
    )

    if cdp_command_timeout_s is not None and cdp_command_timeout_s > 0:
        await cdp_client.send(
            "Page.reload",
            {},
            timeout=float(cdp_command_timeout_s),
            session_id=session_id,
        )
    else:
        await cdp_client.send("Page.reload", {}, session_id=session_id)
    await asyncio.sleep(wait_time)
    return chromiumrl_session_id


async def _collect_signals_impl(
    send_cmd,
    get_frame_id,
    strict: bool,
    log_prefix: str,
) -> Dict[str, Any]:
    """Shared signal collection backend used by service and CDPPool."""

    errors: list[str] = []
    signals: Dict[str, Any] = {
        "touch_traces": [],
        "touch_total_events": 0,
        "layout_elements": [],
        "frame_metrics": {},
        "cls_entries": [],
        "total_cls": 0.0,
        "cls_exceeds_threshold": False,
        "compositor_layers": [],
        "total_gpu_memory": 0,
        "pending_texture_uploads": 0,
        "command_error_count": 0,
        "timeout_error_count": 0,
    }

    def _record_error(context: str, exc: Exception) -> None:
        errors.append(f"{context} failed: {exc}")
        signals["command_error_count"] = int(signals["command_error_count"]) + 1
        if "timed out" in str(exc).lower():
            signals["timeout_error_count"] = int(signals["timeout_error_count"]) + 1

    frame_id = None
    try:
        frame_id = await get_frame_id()
        logger.debug("%s Got main frame ID: %s", log_prefix, frame_id)
    except Exception as exc:
        _record_error("frame_id", exc)
        logger.error("%s ERROR getting frame ID: %s", log_prefix, exc)

    try:
        result = await send_cmd("ChromiumRL.getTouchTraces", {})
        signals["touch_traces"] = result.get("traces", [])
        signals["touch_total_events"] = result.get("totalEvents", 0)
    except Exception as exc:
        _record_error("touch_traces", exc)
        logger.error("%s ERROR collecting touch traces: %s", log_prefix, exc)

    if frame_id:
        try:
            result = await send_cmd("ChromiumRL.getLayoutTimings", {"frameId": frame_id})
            signals["layout_elements"] = result.get("elements", [])
            signals["frame_metrics"] = result.get("frameMetrics", {})
        except Exception as exc:
            _record_error("layout_timings", exc)
            logger.error("%s ERROR collecting layout timings: %s", log_prefix, exc)

    try:
        result = await send_cmd("ChromiumRL.getCLSAttribution", {})
        signals["cls_entries"] = result.get("entries", [])
        signals["total_cls"] = result.get("totalCLS", 0.0)
        signals["cls_exceeds_threshold"] = result.get("exceedsGoodThreshold", False)
    except Exception as exc:
        _record_error("cls_attribution", exc)
        logger.error("%s ERROR collecting CLS attribution: %s", log_prefix, exc)

    try:
        result = await send_cmd(
            "ChromiumRL.getCompositorLayers",
            {"includePaintInfo": True, "includeTransforms": False},
        )
        signals["compositor_layers"] = result.get("layers", [])
        signals["total_gpu_memory"] = result.get("totalGPUMemory", 0)
        signals["pending_texture_uploads"] = result.get("pendingTextureUploads", 0)
    except Exception as exc:
        _record_error("compositor_layers", exc)
        logger.error("%s ERROR collecting compositor layers: %s", log_prefix, exc)

    logger.info(
        "%s Signal collection complete: touch=%d layout=%d cls=%d compositor=%d gpu=%.1fKB",
        log_prefix,
        len(signals["touch_traces"]),
        len(signals["layout_elements"]),
        len(signals["cls_entries"]),
        len(signals["compositor_layers"]),
        signals["total_gpu_memory"] / 1024,
    )

    if strict and errors:
        error_msg = (
            f"ChromiumRL signal collection failed with {len(errors)} errors:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )
        logger.error(error_msg)
        raise ChromiumRLSignalCollectionError(error_msg)

    return signals


async def collect_all_chromiumrl_signals(
    cdp_client,
    frame_id: Optional[str] = None,
    strict: bool = True,
    session_id: Optional[str] = None,
    command_timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Collect all live ChromiumRL signals via a direct CDP client."""

    async def _send(method: str, params: dict) -> Dict[str, Any]:
        if command_timeout_s is not None and command_timeout_s > 0:
            return await cdp_client.send(
                method,
                params,
                timeout=float(command_timeout_s),
                session_id=session_id,
            )
        return await cdp_client.send(method, params, session_id=session_id)

    if frame_id is not None:
        async def _get_frame_id() -> str:
            return frame_id
    else:
        async def _get_frame_id() -> str:
            return await get_main_frame_id(
                cdp_client,
                session_id=session_id,
                command_timeout_s=command_timeout_s,
            )

    return await _collect_signals_impl(
        send_cmd=_send,
        get_frame_id=_get_frame_id,
        strict=strict,
        log_prefix="[ChromiumRL]",
    )


async def save_dom_state(
    cdp_client,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Save current DOM state via ChromiumRL.saveDOMState."""

    try:
        result = await cdp_client.send("ChromiumRL.saveDOMState", {}, session_id=session_id)
        return result.get("state", {})
    except Exception:
        logger.debug("ChromiumRL.saveDOMState not available")
        return {}


async def compare_dom_state(
    cdp_client,
    reference_state: Dict[str, Any],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare current DOM against a reference state."""

    if not reference_state or not reference_state.get("nodes"):
        return {}
    try:
        result = await cdp_client.send(
            "ChromiumRL.compareDOMState",
            {"referenceState": reference_state},
            session_id=session_id,
        )
        return result.get("result", {})
    except Exception:
        logger.debug("ChromiumRL.compareDOMState not available")
        return {}


def compute_dom_comparison_score(comparison: Dict[str, Any]) -> float:
    """Compute a [0, 1] score from ChromiumRL.compareDOMState output."""

    summary = comparison.get("summary", {})
    if not summary:
        return FEEDBACK_MISSING

    structural = summary.get("structuralSimilarity", 0.0)
    text = summary.get("textSimilarity", 0.0)
    layout = summary.get("layoutSimilarity", 0.0)
    style = summary.get("styleSimilarity", 0.0)
    return 0.30 * structural + 0.30 * text + 0.20 * layout + 0.20 * style
