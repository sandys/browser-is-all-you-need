"""
ChromiumRL evaluation HTTP service.

A persistent FastAPI server that wraps CDPPool and exposes ChromiumRL
DOMdiff evaluation as an HTTP API. Runs on the local host alongside the
Android emulator. The Harbor worker calls this service after the sandbox-side
verifier reaches `ready_for_chromiumrl`.

Endpoints:
    POST /evaluate  — Navigate to target_url, collect DOMdiff + quality signals.
    POST /evaluate_async — Start evaluation in the background.
    GET  /evaluate_jobs/{job_id} — Poll a background evaluation.
    GET  /health    — Health check with available tab count.

Usage:
    uvicorn w8_rl.harbor_env.chromiumrl_service:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DAYTONA_WARNING_MARKERS = (
    "daytona preview - warning",
    "preview url warning",
)
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_PROXY_ROUTE_TTL_SEC = int(os.environ.get("CHROMIUMRL_PROXY_ROUTE_TTL_SEC", "600") or "600")


def _default_host_preview_proxy_base_url() -> str:
    explicit = os.environ.get("CHROMIUMRL_HOST_PREVIEW_PROXY_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    port = int(os.environ.get("W8RL_HOST_PREVIEW_PROXY_PORT", "18080") or "18080")
    try:
        with open("/proc/net/route", "r", encoding="utf-8") as route_file:
            for line in route_file.read().splitlines()[1:]:
                fields = line.split()
                if len(fields) < 4:
                    continue
                destination, gateway, flags = fields[1], fields[2], fields[3]
                if destination != "00000000":
                    continue
                if (int(flags, 16) & 0x2) == 0:
                    continue
                gateway_ip = socket.inet_ntoa(bytes.fromhex(gateway)[::-1])
                if gateway_ip:
                    return f"http://{gateway_ip}:{port}"
    except (OSError, ValueError):
        pass
    return f"http://host.docker.internal:{port}"


_HOST_PREVIEW_PROXY_BASE_URL = _default_host_preview_proxy_base_url()


def _default_browser_proxy_base_url() -> str:
    explicit = os.environ.get("CHROMIUMRL_BROWSER_PROXY_BASE_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    port = int(os.environ.get("CHROMIUMRL_BROWSER_PROXY_PORT", "8080") or "8080")
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            host = probe.getsockname()[0]
        finally:
            probe.close()
        if host:
            return f"http://{host}:{port}"
    except OSError:
        pass
    return f"http://10.0.2.2:{port}"


_BROWSER_PROXY_BASE_URL = _default_browser_proxy_base_url()


class EvaluateRequest(BaseModel):
    target_url: str
    timeout_sec: int = 120
    session_id: str = ""
    reference_state: dict[str, Any] = Field(default_factory=dict)
    extra_headers: dict[str, str] = Field(default_factory=dict)


class EvaluateResponse(BaseModel):
    quality: float = 0.0
    domdiff_total: float = 0.0
    domdiff_summary: dict[str, float] = Field(default_factory=dict)
    dom_state: dict[str, Any] = Field(default_factory=dict)
    signal_metrics: dict[str, float] = Field(default_factory=dict)
    signal_count: int = 0
    timeout_count: int = 0
    compositor_layers: int = 0
    request_id: str = ""
    eval_mode: str = ""
    browser_context_supported: bool | None = None
    service_active_evals_at_start: int = 0
    service_max_inflight_after_finish: int = 0
    service_shared_tab_lock_acquire_total: int = 0
    fallback_mode: str = ""
    error: str = ""


class EvaluateJobSubmitResponse(BaseModel):
    job_id: str
    status: str


class EvaluateJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str = ""


class HealthResponse(BaseModel):
    status: str
    available_tabs: int
    wootzapp_verified: bool
    fallback_mode: bool = False


class StatsResponse(BaseModel):
    connected: bool = False
    cdp_url: str = ""
    active_contexts: int = 0
    context_ids: list[str] = Field(default_factory=list)
    browser_context_supported: bool | None = None
    eval_started_total: int = 0
    eval_completed_total: int = 0
    eval_mode_browser_context_total: int = 0
    eval_mode_shared_tab_total: int = 0
    shared_tab_lock_acquire_total: int = 0
    active_evals: int = 0
    max_inflight_evals: int = 0
    last_request_id: str = ""
    fallback_mode: bool = False


@dataclass
class ServiceTelemetry:
    browser_context_supported: bool | None = None
    eval_started_total: int = 0
    eval_completed_total: int = 0
    eval_mode_browser_context_total: int = 0
    eval_mode_shared_tab_total: int = 0
    shared_tab_lock_acquire_total: int = 0
    active_evals: int = 0
    max_inflight_evals: int = 0
    last_request_id: str = ""

    def reset_counters(self) -> None:
        self.eval_started_total = 0
        self.eval_completed_total = 0
        self.eval_mode_browser_context_total = 0
        self.eval_mode_shared_tab_total = 0
        self.shared_tab_lock_acquire_total = 0
        self.active_evals = 0
        self.max_inflight_evals = 0
        self.last_request_id = ""


@dataclass(frozen=True)
class PreviewProxyRoute:
    upstream_origin: str
    headers: dict[str, str]
    created_at: float


_preview_proxy_routes: dict[str, PreviewProxyRoute] = {}
_preview_proxy_targets: dict[str, str] = {}


def _requires_browser_proxy(target_url: str) -> bool:
    if str(os.environ.get("CHROMIUMRL_DISABLE_BROWSER_PREVIEW_PROXY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return False
    try:
        parsed = urllib.parse.urlsplit(target_url)
    except Exception:
        return False
    hostname = str(parsed.hostname or "").strip().lower()
    return hostname.endswith("daytonaproxy01.net")


def _normalize_preview_target_url(target_url: str) -> str:
    parsed = urllib.parse.urlsplit(target_url)
    scheme = parsed.scheme or "https"
    if _requires_browser_proxy(target_url):
        scheme = "https"
    path = parsed.path or "/"
    return urllib.parse.urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def _register_preview_proxy_route(
    target_url: str,
    headers: dict[str, str],
) -> tuple[str, str]:
    normalized_target_url = _normalize_preview_target_url(target_url)
    parsed = urllib.parse.urlsplit(normalized_target_url)
    route_id = uuid4().hex
    _preview_proxy_routes[route_id] = PreviewProxyRoute(
        upstream_origin=f"{parsed.scheme}://{parsed.netloc}",
        headers=dict(headers),
        created_at=time.time(),
    )
    _preview_proxy_targets[normalized_target_url] = route_id
    proxied_url = f"{_BROWSER_PROXY_BASE_URL}/preview/{route_id}{parsed.path or '/'}"
    if parsed.query:
        proxied_url += f"?{parsed.query}"
    return route_id, proxied_url


def _prune_expired_preview_proxy_routes() -> None:
    cutoff = time.time() - max(_PROXY_ROUTE_TTL_SEC, 60)
    expired_ids = [
        route_id
        for route_id, route in _preview_proxy_routes.items()
        if route.created_at < cutoff
    ]
    for route_id in expired_ids:
        _preview_proxy_routes.pop(route_id, None)
    stale_targets = [
        target_url
        for target_url, route_id in _preview_proxy_targets.items()
        if route_id not in _preview_proxy_routes
    ]
    for target_url in stale_targets:
        _preview_proxy_targets.pop(target_url, None)


def _build_upstream_preview_url(route: PreviewProxyRoute, path: str, query: str) -> str:
    upstream_url = route.upstream_origin.rstrip("/")
    if path:
        upstream_url += "/" + path.lstrip("/")
    else:
        upstream_url += "/"
    if query:
        upstream_url += f"?{query}"
    return upstream_url


def _rewrite_preview_redirect_location(
    *,
    route_id: str,
    route: PreviewProxyRoute,
    upstream_url: str,
    location: str,
) -> str:
    candidate = str(location or "").strip()
    if not candidate:
        return candidate
    absolute_target = urllib.parse.urljoin(upstream_url, candidate)
    parsed = urllib.parse.urlsplit(absolute_target)
    if f"{parsed.scheme}://{parsed.netloc}" != route.upstream_origin:
        return candidate
    local_redirect = f"{_BROWSER_PROXY_BASE_URL}/preview/{route_id}{parsed.path or '/'}"
    if parsed.query:
        local_redirect += f"?{parsed.query}"
    return local_redirect


def _rewrite_preview_response_headers(
    *,
    route_id: str,
    route: PreviewProxyRoute,
    upstream_url: str,
    upstream_headers: dict[str, str],
) -> dict[str, str]:
    response_headers = {
        key: value
        for key, value in upstream_headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
    for key, value in list(response_headers.items()):
        if key.lower() == "location":
            response_headers[key] = _rewrite_preview_redirect_location(
                route_id=route_id,
                route=route,
                upstream_url=upstream_url,
                location=value,
            )
        elif key.lower() == "refresh":
            match = re.match(r"^\s*([0-9.]+)\s*;\s*url=(.+?)\s*$", str(value), flags=re.IGNORECASE)
            if match:
                delay, refresh_target = match.groups()
                rewritten_target = _rewrite_preview_redirect_location(
                    route_id=route_id,
                    route=route,
                    upstream_url=upstream_url,
                    location=refresh_target.strip().strip("'\""),
                )
                response_headers[key] = f"{delay};url={rewritten_target}"
    return response_headers


def _rewrite_preview_html_body(route_id: str, body: bytes) -> bytes:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body
    prefix = f"/preview/{route_id}"
    replacements = {
        'href="/': f'href="{prefix}/',
        "href='/": f"href='{prefix}/",
        'src="/': f'src="{prefix}/',
        "src='/": f"src='{prefix}/",
        'action="/': f'action="{prefix}/',
        "action='/": f"action='{prefix}/",
        'content="/': f'content="{prefix}/',
        "content='/": f"content='{prefix}/",
        "url(/": f"url({prefix}/",
        "url('/": f"url('{prefix}/",
        'url("/': f'url("{prefix}/',
    }
    for needle, replacement in replacements.items():
        text = text.replace(needle, replacement)
    return text.encode("utf-8")


def _fetch_preview_proxy_response(
    *,
    upstream_url: str,
    method: str,
    headers: dict[str, str],
) -> tuple[int, dict[str, str], bytes]:
    if _requires_browser_proxy(upstream_url):
        return _fetch_preview_proxy_response_via_host(
            upstream_url=upstream_url,
            method=method,
            headers=headers,
        )
    return _fetch_preview_proxy_response_direct(
        upstream_url=upstream_url,
        method=method,
        headers=headers,
    )


def _fetch_preview_proxy_response_direct(
    *,
    upstream_url: str,
    method: str,
    headers: dict[str, str],
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        upstream_url,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            return int(resp.status), dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), dict(exc.headers.items()), exc.read()


def _fetch_preview_proxy_response_via_host(
    *,
    upstream_url: str,
    method: str,
    headers: dict[str, str],
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        _HOST_PREVIEW_PROXY_BASE_URL.rstrip("/") + "/fetch",
        data=json.dumps(
            {
                "upstream_url": upstream_url,
                "method": method,
                "headers": headers,
                "timeout_s": 60,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=65) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return (
            502,
            {"Content-Type": "text/plain; charset=utf-8"},
            f"host preview proxy HTTPError: {exc.code} {detail}".encode("utf-8"),
        )
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return (
            502,
            {"Content-Type": "text/plain; charset=utf-8"},
            f"host preview proxy error: {type(exc).__name__}: {exc}".encode("utf-8"),
        )
    if not isinstance(payload, dict):
        return (
            502,
            {"Content-Type": "text/plain; charset=utf-8"},
            b"host preview proxy returned non-object payload",
        )
    status_code = int(payload.get("status_code", 502) or 502)
    response_headers = payload.get("headers", {})
    if not isinstance(response_headers, dict):
        response_headers = {}
    body_b64 = str(payload.get("body_b64", "") or "")
    try:
        body = base64.b64decode(body_b64.encode("ascii")) if body_b64 else b""
    except (ValueError, OSError):
        body = b"host preview proxy returned invalid body encoding"
        status_code = 502
    return (
        status_code,
        {str(key): str(value) for key, value in response_headers.items()},
        body,
    )


@dataclass
class CDPManager:
    """Manages a local CDPPoolBase instance (no Ray)."""

    cdp_url: str = "ws://localhost:9224"
    _pool: object = field(default=None, repr=False)
    _verified: bool = False
    _fallback_mode: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _telemetry: ServiceTelemetry = field(default_factory=ServiceTelemetry, repr=False)

    async def initialize(self) -> None:
        from w8_rl.browser.cdp_pool import CDPPoolBase

        self._pool = CDPPoolBase(cdp_url=self.cdp_url)
        try:
            connected = await self._pool.connect()
            if not connected:
                raise RuntimeError(f"Failed to connect to browser at {self.cdp_url}")
            logger.info("Connected to browser at %s", self.cdp_url)

            try:
                await self._pool.verify_chromiumrl()
                self._verified = True
                self._fallback_mode = False
                logger.info("ChromiumRL domain verified (WootzApp present)")
            except Exception:
                if not _allow_cdp_fallback():
                    raise
                self._verified = False
                self._fallback_mode = True
                logger.warning(
                    "ChromiumRL CDP domain is unavailable; continuing with "
                    "CHROMIUMRL_ALLOW_CDP_FALLBACK=1 degraded DOM snapshot mode",
                    exc_info=True,
                )

            # Probe BrowserContext support for parallel evaluations
            bc_supported = await self._pool.supports_browser_context()
            self._telemetry.browser_context_supported = bool(bc_supported)
            logger.info(
                "BrowserContext support: %s (parallel ChromiumRL evaluations %s)",
                bc_supported,
                "enabled" if bc_supported else "disabled — using serialized shared-tab",
            )
        except Exception:
            self._verified = False
            self._fallback_mode = False
            try:
                await self._pool.disconnect()
            except Exception:
                logger.debug("Failed to disconnect CDP pool after init error", exc_info=True)
            self._pool = None
            raise

    async def evaluate(
        self,
        target_url: str,
        timeout_sec: int = 120,
        reference_state: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if self._pool is None:
            raise RuntimeError("CDPManager not initialized")

        request_id = uuid4().hex[:12]
        # BrowserContext mode: each evaluation gets an isolated context,
        # so concurrent evaluations are safe without serialization.
        # Falls back to shared-tab + lock when BrowserContext is unavailable.
        use_browser_context = self._pool.is_browser_context_supported()
        eval_mode = "browser_context" if use_browser_context else "shared_tab"
        self._telemetry.browser_context_supported = bool(use_browser_context)
        self._telemetry.eval_started_total += 1
        self._telemetry.last_request_id = request_id
        self._telemetry.active_evals += 1
        active_evals_at_start = self._telemetry.active_evals
        self._telemetry.max_inflight_evals = max(
            self._telemetry.max_inflight_evals,
            self._telemetry.active_evals,
        )
        if use_browser_context:
            self._telemetry.eval_mode_browser_context_total += 1
        else:
            self._telemetry.eval_mode_shared_tab_total += 1
        started_at = time.perf_counter()
        result: dict[str, Any] | None = None
        logger.info(
            "ChromiumRL eval start request_id=%s mode=%s active_evals=%d target_url=%s",
            request_id,
            eval_mode,
            active_evals_at_start,
            target_url,
        )
        if use_browser_context:
            try:
                result = await self._evaluate_in_context(
                    target_url,
                    timeout_sec,
                    reference_state or {},
                    extra_headers or {},
                )
                if self._is_recoverable_cdp_error(result):
                    logger.warning(
                        "ChromiumRL evaluation hit recoverable CDP failure for %s; retrying",
                        target_url,
                    )
                    if await self._wait_for_live_cdp(timeout_sec=min(timeout_sec, 120)):
                        result = await self._evaluate_in_context(
                            target_url,
                            timeout_sec,
                            reference_state or {},
                            extra_headers or {},
                        )
                if self._is_browser_context_unsupported_error(result):
                    logger.warning(
                        "ChromiumRL evaluation discovered BrowserContext is unsupported for %s; "
                        "retrying this evaluation via shared-tab fallback",
                        target_url,
                    )
                    eval_mode = "shared_tab"
                    if self._telemetry.eval_mode_browser_context_total > 0:
                        self._telemetry.eval_mode_browser_context_total -= 1
                    self._telemetry.eval_mode_shared_tab_total += 1
                    self._telemetry.shared_tab_lock_acquire_total += 1
                    async with self._lock:
                        result = await self._evaluate_in_shared_tab(
                            target_url,
                            timeout_sec,
                            reference_state or {},
                            extra_headers or {},
                        )
            finally:
                self._telemetry.active_evals = max(0, self._telemetry.active_evals - 1)
                self._telemetry.eval_completed_total += 1
                duration_ms = (time.perf_counter() - started_at) * 1000.0
                success = bool(result) and not bool(result.get("error", ""))
                logger.info(
                    "ChromiumRL eval end request_id=%s mode=%s success=%s duration_ms=%.1f active_evals=%d max_inflight=%d",
                    request_id,
                    eval_mode,
                    success,
                    duration_ms,
                    self._telemetry.active_evals,
                    self._telemetry.max_inflight_evals,
                )
        else:
            self._telemetry.shared_tab_lock_acquire_total += 1
            try:
                async with self._lock:
                    result = await self._evaluate_in_shared_tab(
                        target_url,
                        timeout_sec,
                        reference_state or {},
                        extra_headers or {},
                    )
                if self._is_recoverable_cdp_error(result):
                    logger.warning(
                        "ChromiumRL evaluation hit recoverable CDP failure for %s; retrying",
                        target_url,
                    )
                    if await self._wait_for_live_cdp(timeout_sec=min(timeout_sec, 120)):
                        self._telemetry.shared_tab_lock_acquire_total += 1
                        async with self._lock:
                            result = await self._evaluate_in_shared_tab(
                                target_url,
                                timeout_sec,
                                reference_state or {},
                                extra_headers or {},
                            )
            finally:
                self._telemetry.active_evals = max(0, self._telemetry.active_evals - 1)
                self._telemetry.eval_completed_total += 1
                duration_ms = (time.perf_counter() - started_at) * 1000.0
                success = bool(result) and not bool(result.get("error", ""))
                logger.info(
                    "ChromiumRL eval end request_id=%s mode=%s success=%s duration_ms=%.1f active_evals=%d max_inflight=%d",
                    request_id,
                    eval_mode,
                    success,
                    duration_ms,
                    self._telemetry.active_evals,
                    self._telemetry.max_inflight_evals,
                )
        if result is None:
            result = self._error_result("empty evaluation result")
        result.update(
            {
                "request_id": request_id,
                "eval_mode": eval_mode,
                "browser_context_supported": self._telemetry.browser_context_supported,
                "service_active_evals_at_start": active_evals_at_start,
                "service_max_inflight_after_finish": self._telemetry.max_inflight_evals,
                "service_shared_tab_lock_acquire_total": self._telemetry.shared_tab_lock_acquire_total,
            }
        )
        return result

    async def _evaluate_in_context(
        self,
        target_url: str,
        timeout_sec: int,
        reference_state: dict[str, Any],
        extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        """Evaluate using an isolated BrowserContext (parallel-safe)."""
        pool = self._pool
        context = None
        try:
            context = await pool.acquire_context(url="about:blank")
            return await self._evaluate_with_session(
                pool=pool,
                context=context,
                target_url=target_url,
                timeout_sec=timeout_sec,
                reference_state=reference_state,
                extra_headers=extra_headers,
            )
        except Exception as exc:
            logger.exception("ChromiumRL evaluation failed (BrowserContext) for %s", target_url)
            return self._error_result(str(exc))
        finally:
            if context is not None:
                try:
                    await pool.release_context(context["context_id"])
                except Exception:
                    logger.debug("Failed to release BrowserContext", exc_info=True)

    async def _evaluate_in_shared_tab(
        self,
        target_url: str,
        timeout_sec: int,
        reference_state: dict[str, Any],
        extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        """Evaluate using shared tab (serialized, fallback for old APK)."""
        pool = self._pool
        context = None
        try:
            context = await pool.acquire_shared_tab(url="about:blank", fresh=True)
            return await self._evaluate_with_session(
                pool=pool,
                context=context,
                target_url=target_url,
                timeout_sec=timeout_sec,
                reference_state=reference_state,
                extra_headers=extra_headers,
            )
        except Exception as exc:
            logger.exception("ChromiumRL evaluation failed (shared tab) for %s", target_url)
            return self._error_result(str(exc))
        finally:
            if context is not None:
                try:
                    if extra_headers:
                        await pool.send(
                            "Network.setExtraHTTPHeaders",
                            {"headers": {}},
                            session_id=context["session_id"],
                            timeout=5.0,
                        )
                except Exception:
                    logger.debug("Failed to clear context headers", exc_info=True)
                try:
                    await pool.release_context(context["context_id"])
                except Exception:
                    logger.debug("Failed to release context", exc_info=True)

    async def _evaluate_with_session(
        self,
        *,
        pool: Any,
        context: dict[str, Any],
        target_url: str,
        timeout_sec: int,
        reference_state: dict[str, Any],
        extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        """Core evaluation logic shared by BrowserContext and shared-tab paths."""
        session_id = context["session_id"]
        proxy_route_id: str | None = None

        if self._fallback_mode:
            return await self._evaluate_with_cdp_fallback(
                pool=pool,
                session_id=session_id,
                target_url=target_url,
                timeout_sec=timeout_sec,
                reference_state=reference_state,
                extra_headers=extra_headers,
            )

        from w8_rl.rewards.chromiumrl import (
            ChromiumRLScoringConfig,
            _collect_signals_impl,
            compare_dom_state,
            compute_dom_comparison_score,
            enable_chromiumrl_tracing,
            get_main_frame_id,
            save_dom_state,
        )

        if extra_headers:
            normalized_headers = {
                str(key): str(value)
                for key, value in extra_headers.items()
                if str(key).strip() and str(value).strip()
            }
            if normalized_headers:
                await pool.send("Network.enable", {}, session_id=session_id, timeout=10.0)
                await pool.send(
                    "Network.setExtraHTTPHeaders",
                    {"headers": normalized_headers},
                    session_id=session_id,
                    timeout=10.0,
                )

        cdp_timeout = min(timeout_sec, 120)
        enable_timeout = min(timeout_sec, 15)
        navigation_url = target_url
        if _requires_browser_proxy(target_url):
            proxy_route_id, navigation_url = _register_preview_proxy_route(
                target_url,
                normalized_headers if extra_headers else {},
            )
            logger.info(
                "Using local preview proxy for %s via %s",
                target_url,
                navigation_url,
            )

        try:
            await pool.send(
                "Page.navigate",
                {"url": navigation_url},
                session_id=session_id,
                timeout=cdp_timeout,
            )
            await asyncio.sleep(3.0)
            warm_dom_state = await self._wait_for_rendered_page(
                pool,
                session_id=session_id,
                target_url=target_url,
                timeout_sec=timeout_sec,
            )

            chromiumrl_session_id = await enable_chromiumrl_tracing(
                pool,
                session_id=session_id,
                timeout_s=enable_timeout,
            )
            logger.info(
                "ChromiumRL enabled for %s session=%s",
                target_url,
                chromiumrl_session_id,
            )

            await pool.send(
                "Page.reload",
                {},
                session_id=session_id,
                timeout=cdp_timeout,
            )
            await asyncio.sleep(3.0)
            dom_state = await self._wait_for_rendered_page(
                pool,
                session_id=session_id,
                target_url=target_url,
                timeout_sec=timeout_sec,
            )

            async def send_cmd(method, params):
                return await pool.send(method, params, session_id=session_id, timeout=10.0)

            async def frame_id():
                return await get_main_frame_id(pool, session_id=session_id, command_timeout_s=10.0)

            signals = await _collect_signals_impl(
                send_cmd=send_cmd,
                get_frame_id=frame_id,
                strict=False,
                log_prefix=f"chromiumrl_service[{target_url[:60]}]",
            )
            if not dom_state:
                dom_state = warm_dom_state
            comparison = await compare_dom_state(pool, reference_state, session_id=session_id)
            domdiff_summary = comparison.get("summary", {}) if isinstance(comparison, dict) else {}
            domdiff_total = compute_dom_comparison_score(comparison)
            domdiff_operation_counts = {}
            if isinstance(comparison, dict):
                for key in (
                    "insertions",
                    "deletions",
                    "moves",
                    "attributeChanges",
                    "textChanges",
                    "layoutChanges",
                    "styleChanges",
                    "typeChanges",
                ):
                    value = comparison.get(key, [])
                    domdiff_operation_counts[key] = len(value) if isinstance(value, list) else 0

            quality = _compute_quality(signals, ChromiumRLScoringConfig())
            return {
                "quality": quality,
                "domdiff_total": max(0.0, float(domdiff_total)) if domdiff_total >= 0 else 0.0,
                "domdiff_summary": {
                    key: float(domdiff_summary.get(key, 0.0) or 0.0)
                    for key in (
                        "structuralSimilarity",
                        "textSimilarity",
                        "layoutSimilarity",
                        "styleSimilarity",
                    )
                },
                "domdiff_operation_counts": domdiff_operation_counts,
                "dom_state": dom_state if isinstance(dom_state, dict) else {},
                "signal_metrics": _signal_metrics(signals),
                "signal_count": len(signals) - signals.get("command_error_count", 0),
                "timeout_count": signals.get("timeout_error_count", 0),
                "compositor_layers": len(signals.get("compositor_layers", [])),
            }
        finally:
            if proxy_route_id is not None:
                stale_targets = [
                    url
                    for url, route_id in _preview_proxy_targets.items()
                    if route_id == proxy_route_id
                ]
                for url in stale_targets:
                    _preview_proxy_targets.pop(url, None)
                _preview_proxy_routes.pop(proxy_route_id, None)

    async def _evaluate_with_cdp_fallback(
        self,
        *,
        pool: Any,
        session_id: str,
        target_url: str,
        timeout_sec: int,
        reference_state: dict[str, Any],
        extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        """Degraded evaluation for browser builds without ChromiumRL CDP APIs.

        This is intentionally opt-in and exists to exercise the Harbor/rLLM
        plumbing when the local APK is Chromium-compatible but does not expose
        WootzApp's custom ChromiumRL domain. Real training should run with the
        native domain so performance and visual signals are available.
        """
        proxy_route_id: str | None = None
        normalized_headers: dict[str, str] = {}
        if extra_headers:
            normalized_headers = {
                str(key): str(value)
                for key, value in extra_headers.items()
                if str(key).strip() and str(value).strip()
            }

        navigation_url = target_url
        try:
            await self._enable_standard_cdp_domains(pool, session_id=session_id)
            if normalized_headers:
                await pool.send("Network.enable", {}, session_id=session_id, timeout=10.0)
                await pool.send(
                    "Network.setExtraHTTPHeaders",
                    {"headers": normalized_headers},
                    session_id=session_id,
                    timeout=10.0,
                )

            if _requires_browser_proxy(target_url):
                proxy_route_id, navigation_url = _register_preview_proxy_route(
                    target_url,
                    normalized_headers,
                )
                logger.info(
                    "Using local preview proxy for fallback eval %s via %s",
                    target_url,
                    navigation_url,
                )

            await pool.send(
                "Page.navigate",
                {"url": navigation_url},
                session_id=session_id,
                timeout=min(timeout_sec, 120),
            )
            await asyncio.sleep(3.0)
            await self._wait_for_standard_render(pool, session_id=session_id, timeout_sec=timeout_sec)
            dom_state = await self._capture_standard_dom_state(pool, session_id=session_id)
            comparison = _compare_standard_dom_state(reference_state, dom_state)
            return {
                "quality": comparison["total"],
                "domdiff_total": comparison["total"],
                "domdiff_summary": comparison["summary"],
                "domdiff_operation_counts": comparison["operation_counts"],
                "dom_state": dom_state,
                "signal_metrics": {},
                "signal_count": 0,
                "timeout_count": 0,
                "compositor_layers": 0,
                "fallback_mode": "cdp_dom_snapshot",
            }
        finally:
            if proxy_route_id is not None:
                stale_targets = [
                    url
                    for url, route_id in _preview_proxy_targets.items()
                    if route_id == proxy_route_id
                ]
                for url in stale_targets:
                    _preview_proxy_targets.pop(url, None)
                _preview_proxy_routes.pop(proxy_route_id, None)

    async def _enable_standard_cdp_domains(self, pool: Any, *, session_id: str) -> None:
        for method in ("Page.enable", "Runtime.enable", "DOM.enable", "Network.enable"):
            try:
                await pool.send(method, {}, session_id=session_id, timeout=5.0)
            except Exception:
                logger.debug("Fallback CDP domain enable failed: %s", method, exc_info=True)

    async def _wait_for_standard_render(self, pool: Any, *, session_id: str, timeout_sec: int) -> None:
        deadline = asyncio.get_running_loop().time() + min(max(timeout_sec, 10), 120)
        while asyncio.get_running_loop().time() < deadline:
            try:
                result = await pool.send(
                    "Runtime.evaluate",
                    {
                        "expression": "document.readyState",
                        "returnByValue": True,
                    },
                    session_id=session_id,
                    timeout=5.0,
                )
                value = result.get("result", {}).get("value")
                if value in {"interactive", "complete"}:
                    return
            except Exception:
                logger.debug("Fallback readyState probe failed", exc_info=True)
            await asyncio.sleep(1.0)

    async def _capture_standard_dom_state(self, pool: Any, *, session_id: str) -> dict[str, Any]:
        script = r"""
(() => {
  function attrList(el) {
    const out = [];
    for (const a of Array.from(el.attributes || [])) {
      out.push(String(a.name), String(a.value));
    }
    return out;
  }
  function cssSelector(el) {
    if (!el || !el.tagName) return "";
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === Node.ELEMENT_NODE && parts.length < 8) {
      let part = cur.tagName.toLowerCase();
      if (cur.id) {
        part += "#" + CSS.escape(cur.id);
        parts.unshift(part);
        break;
      }
      const cls = Array.from(cur.classList || []).slice(0, 3).map(CSS.escape);
      if (cls.length) part += "." + cls.join(".");
      let nth = 1;
      let sib = cur;
      while ((sib = sib.previousElementSibling)) {
        if (sib.tagName === cur.tagName) nth++;
      }
      part += `:nth-of-type(${nth})`;
      parts.unshift(part);
      cur = cur.parentElement;
    }
    return parts.join(" > ");
  }
  function visibleFromStyle(style, rect) {
    return style.display !== "none" &&
      style.visibility !== "hidden" &&
      Number(style.opacity || "1") > 0 &&
      rect.width > 0 &&
      rect.height > 0;
  }
  const viewport = {
    width: window.innerWidth || 0,
    height: window.innerHeight || 0,
    deviceScaleFactor: window.devicePixelRatio || 1
  };
  const nodes = Array.from(document.querySelectorAll("*")).slice(0, 4000).map((el, index) => {
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    const selector = cssSelector(el);
    const text = (el.innerText || el.textContent || "").trim().replace(/\s+/g, " ");
    return {
      nodeId: index + 1,
      tagName: el.tagName.toLowerCase(),
      parentId: 0,
      siblingIndex: index,
      stablePath: selector,
      cssSelector: selector,
      xpath: "",
      attributes: attrList(el),
      textContent: text.slice(0, 1200),
      bounds: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
      isVisible: visibleFromStyle(style, rect),
      isInViewport: rect.bottom >= 0 && rect.right >= 0 && rect.top <= viewport.height && rect.left <= viewport.width,
      zIndex: Number.parseInt(style.zIndex, 10) || 0,
      keyStyles: {
        display: style.display,
        color: style.color,
        backgroundColor: style.backgroundColor,
        fontSize: style.fontSize,
        fontWeight: style.fontWeight,
        opacity: style.opacity,
        visibility: style.visibility
      },
      fingerprint: `${el.tagName.toLowerCase()}|${selector}|${text.slice(0, 80)}`
    };
  });
  return {
    nodes,
    viewport,
    title: document.title || "",
    url: location.href,
    textContent: document.body ? (document.body.innerText || document.body.textContent || "").slice(0, 20000) : ""
  };
})()
"""
        result = await pool.send(
            "Runtime.evaluate",
            {
                "expression": script,
                "returnByValue": True,
                "awaitPromise": False,
            },
            session_id=session_id,
            timeout=30.0,
        )
        value = result.get("result", {}).get("value")
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _error_result(error: str) -> dict[str, Any]:
        return {
            "quality": 0.0,
            "domdiff_total": 0.0,
            "domdiff_summary": {},
            "domdiff_operation_counts": {},
            "dom_state": {},
            "signal_metrics": {},
            "signal_count": 0,
            "timeout_count": 0,
            "compositor_layers": 0,
            "request_id": "",
            "eval_mode": "",
            "browser_context_supported": None,
            "service_active_evals_at_start": 0,
            "service_max_inflight_after_finish": 0,
            "service_shared_tab_lock_acquire_total": 0,
            "fallback_mode": "",
            "error": error,
        }

    def get_stats(self) -> dict[str, Any]:
        if self._pool is None:
            return {
                "connected": False,
                "browser_context_supported": self._telemetry.browser_context_supported,
                "eval_started_total": self._telemetry.eval_started_total,
                "eval_completed_total": self._telemetry.eval_completed_total,
                "eval_mode_browser_context_total": self._telemetry.eval_mode_browser_context_total,
                "eval_mode_shared_tab_total": self._telemetry.eval_mode_shared_tab_total,
                "shared_tab_lock_acquire_total": self._telemetry.shared_tab_lock_acquire_total,
                "active_evals": self._telemetry.active_evals,
                "max_inflight_evals": self._telemetry.max_inflight_evals,
                "last_request_id": self._telemetry.last_request_id,
                "fallback_mode": self._fallback_mode,
            }
        stats = self._pool.get_stats()
        stats.update(
            {
                "browser_context_supported": self._telemetry.browser_context_supported,
                "eval_started_total": self._telemetry.eval_started_total,
                "eval_completed_total": self._telemetry.eval_completed_total,
                "eval_mode_browser_context_total": self._telemetry.eval_mode_browser_context_total,
                "eval_mode_shared_tab_total": self._telemetry.eval_mode_shared_tab_total,
                "shared_tab_lock_acquire_total": self._telemetry.shared_tab_lock_acquire_total,
                "active_evals": self._telemetry.active_evals,
                "max_inflight_evals": self._telemetry.max_inflight_evals,
                "last_request_id": self._telemetry.last_request_id,
                "fallback_mode": self._fallback_mode,
            }
        )
        return stats

    def reset_stats(self) -> dict[str, Any]:
        self._telemetry.reset_counters()
        if self._pool is not None:
            self._telemetry.browser_context_supported = self._pool.is_browser_context_supported()
        return self.get_stats()

    async def is_healthy(self) -> bool:
        if self._pool is None:
            return False
        try:
            await self._pool.send("Browser.getVersion", timeout=5.0)
            return True
        except Exception as exc:
            logger.warning("ChromiumRL health ping failed: %s", exc)
            return False

    async def _wait_for_live_cdp(self, timeout_sec: int) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while asyncio.get_running_loop().time() < deadline:
            if await self.is_healthy():
                return True
            await asyncio.sleep(2)
        return False

    @staticmethod
    def _is_recoverable_cdp_error(result: dict[str, Any]) -> bool:
        error = str(result.get("error", "") or "").lower()
        if not error:
            return False
        markers = (
            "cdp connection closed",
            "failed to connect to cdp",
            "cannot connect to host",
            "clientconnectorerror",
        )
        return any(marker in error for marker in markers)

    @staticmethod
    def _is_browser_context_unsupported_error(result: dict[str, Any]) -> bool:
        error = str(result.get("error", "") or "").strip().lower()
        if not error:
            return False
        return "browsercontext is not supported by this cdp endpoint" in error

    async def _wait_for_rendered_page(
        self,
        pool: Any,
        *,
        session_id: str,
        target_url: str,
        timeout_sec: int,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(max(timeout_sec, 10), 120)
        last_dom_state: dict[str, Any] = {}
        warning_click_attempted = False

        while loop.time() < deadline:
            last_dom_state = await self._safe_save_dom_state(pool, session_id=session_id)
            if _is_daytona_preview_warning(last_dom_state):
                if not warning_click_attempted:
                    logger.warning(
                        "Daytona preview warning encountered for %s; attempting in-browser continue",
                        target_url,
                    )
                    await pool.send(
                        "Runtime.evaluate",
                        {
                            "expression": """
(() => {
  const button = document.querySelector("button.btn-primary, button.btn, button[type='submit'], input[type='submit']");
  if (button) {
    button.click();
    return "clicked";
  }
  const form = document.querySelector("form");
  if (form) {
    form.submit();
    return "submitted";
  }
  return "missing";
})()
""",
                            "returnByValue": True,
                        },
                        session_id=session_id,
                        timeout=5.0,
                    )
                    warning_click_attempted = True
                await asyncio.sleep(1.0)
                continue

            if _has_rendered_content(last_dom_state):
                if warning_click_attempted:
                    logger.info("Daytona preview warning cleared for %s", target_url)
                return last_dom_state

            await asyncio.sleep(1.0)

        logger.warning(
            "Rendered page did not become ready for %s before timeout; title=%r url=%r",
            target_url,
            last_dom_state.get("title", ""),
            last_dom_state.get("url", ""),
        )
        return last_dom_state

    async def _safe_save_dom_state(
        self,
        pool: Any,
        *,
        session_id: str,
    ) -> dict[str, Any]:
        from w8_rl.rewards.chromiumrl import save_dom_state

        try:
            dom_state = await save_dom_state(pool, session_id=session_id)
        except Exception:
            logger.debug("save_dom_state failed during preview warning flow", exc_info=True)
            return {}
        return dom_state if isinstance(dom_state, dict) else {}


def _compute_quality(signals: dict[str, Any], config: Any) -> float:
    scores = []

    frame_metrics = signals.get("frame_metrics", {})
    layout_duration = frame_metrics.get("layoutDuration", 0)
    if layout_duration >= 0:
        layout_score = max(0.0, 1.0 - layout_duration / (config.layout_time_max_ms / 1000.0))
        scores.append(("layout", layout_score, config.abs_layout_weight))

    total_cls = signals.get("total_cls", -1)
    if total_cls >= 0:
        cls_score = max(0.0, 1.0 - total_cls / config.cls_max_score)
        scores.append(("cls", cls_score, config.abs_cls_weight))

    layers = signals.get("compositor_layers", [])
    if layers:
        n = len(layers)
        if n <= config.layer_count_good:
            comp_score = 1.0
        elif n >= config.layer_count_poor:
            comp_score = 0.0
        else:
            comp_score = 1.0 - (n - config.layer_count_good) / (
                config.layer_count_poor - config.layer_count_good
            )
        scores.append(("compositor", comp_score, config.abs_compositor_weight))

    gpu_mem = signals.get("total_gpu_memory", -1)
    if gpu_mem >= 0:
        gpu_mb = gpu_mem / (1024 * 1024)
        if gpu_mb <= config.gpu_memory_good_mb:
            gpu_score = 1.0
        elif gpu_mb >= config.gpu_memory_poor_mb:
            gpu_score = 0.0
        else:
            gpu_score = 1.0 - (gpu_mb - config.gpu_memory_good_mb) / (
                config.gpu_memory_poor_mb - config.gpu_memory_good_mb
            )
        scores.append(("gpu", gpu_score, 0.10))

    if not scores:
        return 0.0

    total_weight = sum(w for _, _, w in scores)
    if total_weight == 0:
        return 0.0
    return sum(s * w for _, s, w in scores) / total_weight


def _signal_metrics(signals: dict[str, Any]) -> dict[str, float]:
    frame_metrics = signals.get("frame_metrics", {}) if isinstance(signals, dict) else {}
    layout_duration = frame_metrics.get("layoutDuration", 0.0)
    gpu_bytes = signals.get("total_gpu_memory", 0.0) if isinstance(signals, dict) else 0.0
    compositor_layers = len(signals.get("compositor_layers", [])) if isinstance(signals, dict) else 0
    return {
        "layout_duration_ms": float(layout_duration or 0.0) * 1000.0,
        "total_cls": float(signals.get("total_cls", 0.0) or 0.0),
        "gpu_memory_mb": float(gpu_bytes or 0.0) / (1024.0 * 1024.0),
        "compositor_layer_count": float(compositor_layers),
    }


def _is_daytona_preview_warning(dom_state: dict[str, Any]) -> bool:
    if not isinstance(dom_state, dict):
        return False

    haystacks: list[str] = []
    for value in (
        dom_state.get("title", ""),
        dom_state.get("textContent", ""),
        dom_state.get("url", ""),
    ):
        text = str(value or "").strip().lower()
        if text:
            haystacks.append(text)

    nodes = dom_state.get("nodes", [])
    if isinstance(nodes, list):
        for node in nodes[:200]:
            for key in ("textContent", "name", "value", "title"):
                text = str(node.get(key, "") or "").strip().lower()
                if text:
                    haystacks.append(text)

    joined = "\n".join(haystacks)
    return any(marker in joined for marker in _DAYTONA_WARNING_MARKERS)


def _has_rendered_content(dom_state: dict[str, Any]) -> bool:
    if not isinstance(dom_state, dict):
        return False

    if str(dom_state.get("title", "") or "").strip():
        return True

    text_nodes = 0
    nodes = dom_state.get("nodes", [])
    if not isinstance(nodes, list):
        return False

    for node in nodes[:400]:
        text = str(node.get("textContent", "") or "").strip()
        if len(text) >= 2:
            text_nodes += 1
            if text_nodes >= 3:
                return True

    return False


def _allow_cdp_fallback() -> bool:
    return str(os.environ.get("CHROMIUMRL_ALLOW_CDP_FALLBACK", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _compare_standard_dom_state(
    reference_state: dict[str, Any],
    live_state: dict[str, Any],
) -> dict[str, Any]:
    ref_nodes = _node_list(reference_state)
    live_nodes = _node_list(live_state)
    ref_selectors = _selector_set(ref_nodes)
    live_selectors = _selector_set(live_nodes)
    selector_union = ref_selectors | live_selectors
    structural = (
        len(ref_selectors & live_selectors) / len(selector_union)
        if selector_union
        else _count_similarity(len(ref_nodes), len(live_nodes))
    )

    ref_text = _state_text(reference_state, ref_nodes)
    live_text = _state_text(live_state, live_nodes)
    text = difflib.SequenceMatcher(
        None,
        ref_text[:20000],
        live_text[:20000],
        autojunk=False,
    ).ratio() if ref_text or live_text else structural

    ref_visible = sum(1 for node in ref_nodes if bool(node.get("isVisible")))
    live_visible = sum(1 for node in live_nodes if bool(node.get("isVisible")))
    layout = _count_similarity(ref_visible, live_visible)

    style = _style_similarity(ref_nodes, live_nodes)
    total = max(0.0, min(1.0, 0.30 * structural + 0.30 * text + 0.20 * layout + 0.20 * style))
    operation_counts = {
        "insertions": max(0, len(live_selectors - ref_selectors)),
        "deletions": max(0, len(ref_selectors - live_selectors)),
        "moves": 0,
        "attributeChanges": 0,
        "textChanges": 0 if text >= 0.98 else 1,
        "layoutChanges": 0 if layout >= 0.98 else 1,
        "styleChanges": 0 if style >= 0.98 else 1,
        "typeChanges": 0,
    }
    return {
        "total": total,
        "summary": {
            "structuralSimilarity": max(0.0, min(1.0, float(structural))),
            "textSimilarity": max(0.0, min(1.0, float(text))),
            "layoutSimilarity": max(0.0, min(1.0, float(layout))),
            "styleSimilarity": max(0.0, min(1.0, float(style))),
        },
        "operation_counts": operation_counts,
    }


def _node_list(state: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = state.get("nodes") if isinstance(state, dict) else []
    return [node for node in nodes if isinstance(node, dict)] if isinstance(nodes, list) else []


def _selector_set(nodes: list[dict[str, Any]]) -> set[str]:
    selectors: set[str] = set()
    for node in nodes:
        selector = str(node.get("cssSelector") or node.get("stablePath") or "").strip()
        if not selector:
            fingerprint = str(node.get("fingerprint") or "").strip()
            tag = str(node.get("tagName") or "").strip()
            text = str(node.get("textContent") or "").strip()[:80]
            selector = fingerprint or f"{tag}|{text}"
        if selector:
            selectors.add(selector)
    return selectors


def _state_text(state: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    text = str(state.get("textContent") or "").strip() if isinstance(state, dict) else ""
    if text:
        return re.sub(r"\s+", " ", text)
    pieces = [str(node.get("textContent") or "").strip() for node in nodes]
    return re.sub(r"\s+", " ", " ".join(piece for piece in pieces if piece))


def _count_similarity(reference_count: int, live_count: int) -> float:
    max_count = max(reference_count, live_count)
    if max_count <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(reference_count - live_count) / max_count)


def _style_similarity(ref_nodes: list[dict[str, Any]], live_nodes: list[dict[str, Any]]) -> float:
    ref_by_selector = {
        str(node.get("cssSelector") or node.get("stablePath") or "").strip(): node
        for node in ref_nodes
    }
    live_by_selector = {
        str(node.get("cssSelector") or node.get("stablePath") or "").strip(): node
        for node in live_nodes
    }
    shared = [selector for selector in ref_by_selector if selector and selector in live_by_selector]
    if not shared:
        return _count_similarity(len(ref_nodes), len(live_nodes))
    keys = ("display", "visibility", "opacity", "fontSize", "fontWeight")
    matches = 0
    total = 0
    for selector in shared[:1000]:
        ref_style = ref_by_selector[selector].get("keyStyles", {})
        live_style = live_by_selector[selector].get("keyStyles", {})
        if not isinstance(ref_style, dict) or not isinstance(live_style, dict):
            continue
        for key in keys:
            total += 1
            if str(ref_style.get(key) or "") == str(live_style.get(key) or ""):
                matches += 1
    return matches / total if total else 0.0


_cdp_manager = CDPManager(cdp_url=os.environ.get("CDP_URL", "ws://localhost:9224"))
_evaluate_jobs: dict[str, dict[str, Any]] = {}
_EVALUATE_JOB_TTL_SEC = int(os.environ.get("CHROMIUMRL_EVALUATE_JOB_TTL_SEC", "3600") or "3600")


def _prune_evaluate_jobs() -> None:
    now = time.time()
    stale = [
        job_id
        for job_id, job in _evaluate_jobs.items()
        if now - float(job.get("updated_at", job.get("created_at", now))) > _EVALUATE_JOB_TTL_SEC
    ]
    for job_id in stale:
        _evaluate_jobs.pop(job_id, None)


async def _run_evaluate_job(job_id: str, req: EvaluateRequest) -> None:
    job = _evaluate_jobs.get(job_id)
    if job is None:
        return
    job.update({"status": "running", "updated_at": time.time()})
    try:
        result = await _cdp_manager.evaluate(
            target_url=req.target_url,
            timeout_sec=req.timeout_sec,
            reference_state=req.reference_state,
            extra_headers=req.extra_headers,
        )
        job.update({"status": "done", "result": result, "updated_at": time.time()})
    except Exception as exc:  # noqa: BLE001 - surfaced through polling response
        logger.exception("ChromiumRL async evaluation failed job_id=%s", job_id)
        job.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": time.time(),
            }
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ChromiumRL service starting...")
    logger.info("ChromiumRL browser preview proxy base URL: %s", _BROWSER_PROXY_BASE_URL)
    timeout_sec = int(os.environ.get("CHROMIUMRL_INIT_TIMEOUT_SEC", "180") or "180")
    retry_interval_sec = float(os.environ.get("CHROMIUMRL_INIT_RETRY_INTERVAL_SEC", "5") or "5")
    deadline = asyncio.get_running_loop().time() + max(timeout_sec, 1)
    last_exc: Exception | None = None

    while asyncio.get_running_loop().time() < deadline:
        try:
            await _cdp_manager.initialize()
            logger.info("ChromiumRL service ready")
            break
        except Exception as exc:
            last_exc = exc
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            logger.warning(
                "ChromiumRL service init not ready yet: %s (%s). Retrying in %.1fs, remaining %.1fs",
                type(exc).__name__,
                exc,
                retry_interval_sec,
                remaining,
            )
            await asyncio.sleep(retry_interval_sec)
    else:
        logger.exception("FATAL: ChromiumRL service failed to start")
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("ChromiumRL service failed to start")
    yield
    logger.info("ChromiumRL service shutting down")


app = FastAPI(title="ChromiumRL Evaluation Service", lifespan=lifespan)


@app.api_route("/preview/{route_id}/{path:path}", methods=["GET", "HEAD"])
async def preview_proxy(route_id: str, path: str, request: Request) -> Response:
    _prune_expired_preview_proxy_routes()
    route = _preview_proxy_routes.get(route_id)
    if route is None:
        raise HTTPException(status_code=404, detail="preview proxy route expired")

    upstream_url = _build_upstream_preview_url(route, path, str(request.url.query))
    forward_headers = dict(route.headers)
    user_agent = str(request.headers.get("user-agent", "")).strip()
    accept = str(request.headers.get("accept", "")).strip()
    if user_agent:
        forward_headers.setdefault("User-Agent", user_agent)
    if accept:
        forward_headers.setdefault("Accept", accept)

    status_code, upstream_headers, body = await asyncio.to_thread(
        _fetch_preview_proxy_response,
        upstream_url=upstream_url,
        method=request.method,
        headers=forward_headers,
    )
    response_headers = _rewrite_preview_response_headers(
        route_id=route_id,
        route=route,
        upstream_url=upstream_url,
        upstream_headers=upstream_headers,
    )
    content_type = str(response_headers.get("Content-Type", "")).lower()
    if "text/html" in content_type and body:
        body = _rewrite_preview_html_body(route_id, body)
        response_headers.pop("Content-Length", None)
    return Response(
        content=b"" if request.method == "HEAD" else body,
        status_code=status_code,
        headers=response_headers,
    )


@app.api_route("/accept-daytona-preview-warning", methods=["POST", "GET"])
async def accept_daytona_preview_warning(request: Request) -> Response:
    redirect_target = str(request.query_params.get("redirect", "")).strip()
    if not redirect_target or not _requires_browser_proxy(redirect_target):
        raise HTTPException(status_code=400, detail="invalid preview redirect target")

    normalized_redirect_target = _normalize_preview_target_url(redirect_target)
    parsed = urllib.parse.urlsplit(normalized_redirect_target)
    upstream_accept_url = (
        f"{parsed.scheme}://{parsed.netloc}/accept-daytona-preview-warning?"
        f"{urllib.parse.urlencode({'redirect': normalized_redirect_target})}"
    )
    status_code, upstream_headers, body = await asyncio.to_thread(
        _fetch_preview_proxy_response,
        upstream_url=upstream_accept_url,
        method=request.method,
        headers={},
    )

    route_id = _preview_proxy_targets.get(normalized_redirect_target)
    if status_code < 400 and route_id:
        local_redirect = f"{_BROWSER_PROXY_BASE_URL}/preview/{route_id}{parsed.path or '/'}"
        if parsed.query:
            local_redirect += f"?{parsed.query}"
        return Response(status_code=302, headers={"Location": local_redirect})

    response_headers = {
        key: value
        for key, value in upstream_headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS
    }
    return Response(
        content=b"" if request.method == "HEAD" else body,
        status_code=status_code,
        headers=response_headers,
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    if not req.target_url:
        raise HTTPException(status_code=400, detail="target_url is required")

    result = await _cdp_manager.evaluate(
        target_url=req.target_url,
        timeout_sec=req.timeout_sec,
        reference_state=req.reference_state,
        extra_headers=req.extra_headers,
    )
    return EvaluateResponse(**result)


@app.post("/evaluate_async", response_model=EvaluateJobSubmitResponse)
async def evaluate_async(req: EvaluateRequest) -> EvaluateJobSubmitResponse:
    if not req.target_url:
        raise HTTPException(status_code=400, detail="target_url is required")

    _prune_evaluate_jobs()
    job_id = uuid4().hex
    _evaluate_jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": "",
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    asyncio.create_task(_run_evaluate_job(job_id, req))
    return EvaluateJobSubmitResponse(job_id=job_id, status="queued")


@app.get("/evaluate_jobs/{job_id}", response_model=EvaluateJobStatusResponse)
async def evaluate_job(job_id: str) -> EvaluateJobStatusResponse:
    _prune_evaluate_jobs()
    job = _evaluate_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="evaluate job not found")
    return EvaluateJobStatusResponse(
        job_id=job_id,
        status=str(job.get("status") or "unknown"),
        result=job.get("result") if isinstance(job.get("result"), dict) else None,
        error=str(job.get("error") or ""),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    stats = _cdp_manager.get_stats()
    live = await _cdp_manager.is_healthy()
    return HealthResponse(
        status="ok" if stats.get("connected") and live else "degraded",
        available_tabs=1,
        wootzapp_verified=_cdp_manager._verified and live,
        fallback_mode=_cdp_manager._fallback_mode and live,
    )


@app.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    return StatsResponse(**_cdp_manager.get_stats())


@app.post("/stats/reset", response_model=StatsResponse)
async def reset_stats() -> StatsResponse:
    return StatsResponse(**_cdp_manager.reset_stats())
