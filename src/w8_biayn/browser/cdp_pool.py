"""CDP connection pool for the w8-biayn ChromiumRL service."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from w8_biayn.browser.cdp_client import CDPClient, CDPConnectionError

logger = logging.getLogger(__name__)


class AccessibilityTreeCaptureError(RuntimeError):
    """Raised when accessibility capture fails for a page under evaluation."""


@dataclass
class PooledContext:
    """Represents an isolated browser context acquired from the pool."""
    context_id: str
    target_id: str
    session_id: str
    owns_target: bool = False


class CDPPoolBase:
    """
    Base class for CDP connection pool.

    Manages a single browser-level WebSocket connection and provides
    isolated BrowserContexts for each rollout.

    Each BrowserContext has separate:
    - Cookies
    - localStorage / IndexedDB
    - Cache storage
    - Network socket pool

    This enables parallel rollouts without state pollution.
    """

    def __init__(
        self,
        cdp_url: str = "ws://localhost:9224",
        max_retries: int = 30,
        initial_delay: float = 1.0,
        adb_host: str | None = None,
    ):
        """
        Initialize the CDP pool.

        Args:
            cdp_url: WebSocket URL for CDP connection
            max_retries: Max connection retry attempts
            initial_delay: Initial retry delay in seconds
            adb_host: ADB host for browser reset (e.g. "container:android-world")
        """
        self._cdp_url = cdp_url
        self._max_retries = max_retries
        self._initial_delay = initial_delay
        self._cdp: CDPClient | None = None
        self._contexts: dict[str, PooledContext] = {}
        self._lock = asyncio.Lock()
        self._connected = False
        self._browser_context_supported: bool | None = None
        self._adb_host = adb_host
        self._needs_browser_reset = False

    async def connect(self) -> bool:
        """
        Connect shared CDPClient to browser.

        Uses browser-level WebSocket for sessionId routing.
        Tests BrowserContext support during connection.

        Returns:
            True if connection succeeded and BrowserContext is supported,
            False if connected but BrowserContext is NOT supported (mobile Chrome)
        """
        async with self._lock:
            if self._connected and self._cdp is not None:
                # Already connected
                return self._browser_context_supported

            logger.info(f"CDPPool connecting to {self._cdp_url}")

            self._cdp = CDPClient(
                ws_url=self._cdp_url,
                max_retries=self._max_retries,
                initial_delay=self._initial_delay,
                use_browser_websocket=True,  # CRITICAL: browser-level for sessionId routing
            )

            # Wait for CDP to be ready
            if not await self._cdp.wait_for_ready(timeout=120.0):
                raise CDPConnectionError(f"CDP not ready at {self._cdp_url}")

            # Connect to browser-level WebSocket
            await self._cdp.connect()

            self._connected = True

            # BrowserContext support is detected on first acquire_context().
            logger.info("CDPPool connected successfully")
            return True

    async def _test_browser_context_support(self) -> bool:
        """
        Test if the browser supports Target.createBrowserContext.

        Mobile Chrome (Android) does NOT support BrowserContext isolation.
        Desktop Chrome and headless Chrome DO support it.

        Returns:
            True if BrowserContext is supported
        """
        if not self._cdp:
            return False

        try:
            # Try to create a test context
            context_id = await self._cdp.create_browser_context()
            if context_id:
                # Clean up the test context
                await self._cdp.dispose_browser_context(context_id)
                logger.debug("CDPPool: BrowserContext support confirmed")
                return True
            return False
        except Exception as e:
            logger.debug(f"CDPPool: BrowserContext not supported: {e}")
            return False

    def is_browser_context_supported(self) -> bool:
        """Check if BrowserContext isolation is supported."""
        return self._browser_context_supported is True

    async def supports_browser_context(self) -> bool:
        """
        Probe BrowserContext support with a stable actor API.

        Returns:
            True when BrowserContext is supported.
            False when unsupported (for example Android mobile Chrome).
        """
        if self._browser_context_supported is not None:
            return self._browser_context_supported
        if not self._connected:
            await self.connect()
        self._browser_context_supported = await self._test_browser_context_support()
        return self._browser_context_supported

    async def disconnect(self) -> None:
        """Disconnect from browser and release all contexts."""
        async with self._lock:
            if not self._connected:
                return

            logger.info("CDPPool disconnecting...")

            # Release all contexts
            for context_id in list(self._contexts.keys()):
                ctx = self._contexts.pop(context_id, None)
                if ctx:
                    try:
                        await self._cdp.close_target(ctx.target_id)
                        await self._cdp.dispose_browser_context(ctx.context_id)
                    except Exception as e:
                        logger.warning(f"Error releasing context {context_id}: {e}")

            # Close CDP connection
            if self._cdp:
                await self._cdp.close()
                self._cdp = None

            self._connected = False
            logger.info("CDPPool disconnected")

    async def is_connected(self) -> bool:
        """Check if pool is connected."""
        if not self._connected or self._cdp is None:
            return False
        return await self._cdp.is_connected()

    async def acquire_context(self, url: str = "about:blank") -> dict:
        """
        Create isolated BrowserContext for a rollout.

        Each context has separate cookies, localStorage, cache, etc.

        Args:
            url: Initial URL to navigate to (default: about:blank)

        Returns:
            Dict with context_id, target_id, session_id
        """
        if not self._connected or self._cdp is None:
            raise CDPConnectionError("CDPPool not connected")

        async with self._lock:
            # Fast path: skip CDP call when we already know it's unsupported
            if self._browser_context_supported is False:
                raise CDPConnectionError(
                    "BrowserContext is not supported by this CDP endpoint."
                )

            # Create isolated browser context
            try:
                context_id = await self._cdp.create_browser_context()
                self._browser_context_supported = True
            except Exception as e:
                self._browser_context_supported = False
                raise CDPConnectionError(
                    "BrowserContext is not supported by this CDP endpoint."
                ) from e

            # Create target (tab) in context
            target_id, session_id = await self._cdp.create_target_in_context(
                url, context_id
            )

            # Store context info
            ctx = PooledContext(
                context_id=context_id,
                target_id=target_id,
                session_id=session_id,
            )
            self._contexts[context_id] = ctx

            logger.debug(
                f"CDPPool acquired context: context_id={context_id}, "
                f"target_id={target_id}, session_id={session_id}"
            )

            return {
                "context_id": context_id,
                "target_id": target_id,
                "session_id": session_id,
            }

    async def acquire_shared_tab(
        self,
        url: str = "about:blank",
        *,
        fresh: bool = False,
    ) -> dict:
        """
        Acquire an existing browser tab without BrowserContext isolation.

        Used when BrowserContext is not supported (mobile Chrome / WootzApp).
        By default it attaches to the first existing page target. When
        ``fresh=True``, it creates a temporary dedicated tab instead.

        This does NOT provide isolation — callers must ensure serial access.

        Args:
            url: Initial URL to navigate to (default: about:blank)

        Returns:
            Dict with context_id, target_id, session_id
        """
        if not self._connected or self._cdp is None:
            raise CDPConnectionError("CDPPool not connected")

        async with self._lock:
            owns_target = False
            if fresh:
                create_result = await self._cdp.send("Target.createTarget", {
                    "url": url,
                })
                target_id = create_result.get("targetId")
                if not target_id:
                    raise CDPConnectionError("Failed to create target tab")
                owns_target = True
            else:
                # Get existing targets
                result = await self._cdp.send("Target.getTargets", {})
                targets = result.get("targetInfos", [])

                # Find a page target
                page_target = next(
                    (t for t in targets if t.get("type") == "page"), None
                )
                if not page_target:
                    # Create a new tab if no page exists
                    create_result = await self._cdp.send("Target.createTarget", {
                        "url": url,
                    })
                    target_id = create_result.get("targetId")
                    if not target_id:
                        raise CDPConnectionError("Failed to create target tab")
                    owns_target = True
                else:
                    target_id = page_target["targetId"]

            # Attach to the target to get a sessionId
            attach_result = await self._cdp.send("Target.attachToTarget", {
                "targetId": target_id,
                "flatten": True,
            })
            session_id = attach_result.get("sessionId")
            if not session_id:
                raise CDPConnectionError(
                    f"Failed to attach to target {target_id}: no sessionId"
                )

            # Enable required domains for the session
            await self._ensure_session_domains(session_id)

            # Navigate to requested URL
            if url != "about:blank":
                await self._cdp.send("Page.navigate", {"url": url}, session_id=session_id)
                await asyncio.sleep(1.0)

            # Use a synthetic context_id since we have no real BrowserContext
            synthetic_context_id = f"shared-tab-{target_id}"

            ctx = PooledContext(
                context_id=synthetic_context_id,
                target_id=target_id,
                session_id=session_id,
                owns_target=owns_target,
            )
            self._contexts[synthetic_context_id] = ctx

            logger.info(
                f"CDPPool acquired shared tab (no BrowserContext): "
                f"target_id={target_id}, session_id={session_id}, fresh={fresh}"
            )

            return {
                "context_id": synthetic_context_id,
                "target_id": target_id,
                "session_id": session_id,
            }

    async def release_context(self, context_id: str) -> bool:
        """
        Release BrowserContext after rollout completes.

        Args:
            context_id: Context ID from acquire_context()

        Returns:
            True if released successfully
        """
        if not self._cdp:
            return False

        async with self._lock:
            ctx = self._contexts.pop(context_id, None)
            if not ctx:
                logger.warning(f"CDPPool: context {context_id} not found")
                return False

            try:
                if ctx.context_id.startswith("shared-tab-"):
                    if ctx.owns_target:
                        await self._cdp.close_target(ctx.target_id)
                        logger.debug(f"CDPPool closed owned shared tab: {context_id}")
                    else:
                        # Shared tab mode: detach session but keep the tab alive
                        try:
                            await self._cdp.send("Target.detachFromTarget", {
                                "sessionId": ctx.session_id,
                            })
                        except Exception as e:
                            logger.debug(f"CDPPool: best-effort detach failed: {e}")
                        logger.debug(f"CDPPool released shared tab: {context_id}")
                else:
                    # BrowserContext mode: close tab and dispose context
                    await self._cdp.close_target(ctx.target_id)
                    await self._cdp.dispose_browser_context(ctx.context_id)
                    logger.debug(f"CDPPool released context: {context_id}")
                return True
            except Exception as e:
                logger.warning(f"CDPPool: error releasing context {context_id}: {e}")
                return False

    async def get_context_count(self) -> int:
        """Get number of active contexts."""
        return len(self._contexts)

    # =========================================================================
    # Session health helpers
    # =========================================================================

    async def _validate_and_reattach(self, context_id: str) -> str | None:
        """Validate shared-tab session; re-attach if stale.

        Returns:
            New session_id if re-attached, None if session was already healthy.
        """
        ctx = self._contexts.get(context_id)
        if not ctx or not ctx.context_id.startswith("shared-tab-"):
            return None  # Not a shared-tab context, skip

        # Test session health with a lightweight probe
        try:
            await self._cdp.send(
                "Runtime.evaluate",
                {"expression": "1"},
                session_id=ctx.session_id,
                timeout=3.0,
            )
            return None  # Session is fine
        except Exception:
            pass  # Session is dead, re-attach

        logger.warning(
            "CDPPool: session %s is stale, re-attaching to %s",
            ctx.session_id,
            ctx.target_id,
        )
        try:
            attach_result = await self._cdp.send("Target.attachToTarget", {
                "targetId": ctx.target_id,
                "flatten": True,
            })
            new_session_id = attach_result.get("sessionId")
            if new_session_id:
                ctx.session_id = new_session_id
                # Re-enable core domains
                await self._ensure_session_domains(new_session_id)
                logger.info("CDPPool: re-attached with new session %s", new_session_id)
                return new_session_id
        except Exception as e:
            logger.error("CDPPool: re-attach failed: %s", e)

        return None

    async def validate_session(self, context_id: str) -> str | None:
        """Validate and optionally re-attach session.

        Returns:
            New session_id if changed, None if session was healthy.
        """
        return await self._validate_and_reattach(context_id)

    async def _ensure_session_domains(self, session_id: str) -> None:
        """Re-enable core domains on the session. Idempotent."""
        if not self._cdp or not session_id:
            return
        for domain in ("Page", "DOM", "Runtime"):
            try:
                await self._cdp.send(f"{domain}.enable", {}, session_id=session_id)
            except Exception:
                pass  # Best-effort; will fail loudly on the actual command

    # =========================================================================
    # Proxy methods that accept session_id for routing
    # =========================================================================

    async def navigate(
        self,
        url: str,
        wait_time: float,
        session_id: str,
    ) -> None:
        """
        Navigate to URL in a specific context.

        Args:
            url: URL to navigate to
            wait_time: Time to wait after navigation
            session_id: Session ID for routing to correct tab
        """
        if not self._cdp:
            raise CDPConnectionError("CDPPool not connected")
        await self._ensure_session_domains(session_id)
        await self._cdp.navigate(url, wait_time, session_id=session_id)

    async def capture_screenshot(
        self,
        session_id: str,
        format: str = "png",
    ) -> str:
        """
        Capture screenshot from a specific context.

        Args:
            session_id: Session ID for routing
            format: Image format (png or jpeg)

        Returns:
            Base64-encoded screenshot
        """
        if not self._cdp:
            raise CDPConnectionError("CDPPool not connected")
        return await self._cdp.capture_screenshot(format=format, session_id=session_id)

    async def set_desktop_mode(
        self,
        width: int,
        height: int,
        session_id: str,
        device_scale_factor: float = 1.0,
        mobile: bool = False,
        user_agent: str | None = None,
    ) -> None:
        """
        Set desktop viewport in a specific context.

        Args:
            width: Viewport width
            height: Viewport height
            session_id: Session ID for routing
            device_scale_factor: Device scale factor
            mobile: Mobile emulation mode
            user_agent: User agent string
        """
        if not self._cdp:
            raise CDPConnectionError("CDPPool not connected")
        await self._cdp.set_desktop_mode(
            width=width,
            height=height,
            device_scale_factor=device_scale_factor,
            mobile=mobile,
            user_agent=user_agent,
            session_id=session_id,
        )

    async def evaluate(
        self,
        expression: str,
        session_id: str,
        return_by_value: bool = True,
        await_promise: bool = False,
    ) -> dict:
        """
        Evaluate JavaScript in a specific context.

        Args:
            expression: JavaScript expression
            session_id: Session ID for routing
            return_by_value: Return result by value
            await_promise: Wait for promise resolution

        Returns:
            Evaluation result
        """
        if not self._cdp:
            raise CDPConnectionError("CDPPool not connected")
        return await self._cdp.evaluate(
            expression,
            return_by_value=return_by_value,
            await_promise=await_promise,
            session_id=session_id,
        )

    async def get_document(
        self,
        session_id: str,
        depth: int = -1,
    ) -> dict:
        """
        Get DOM document from a specific context.

        Args:
            session_id: Session ID for routing
            depth: Depth of tree to return (-1 = full tree)

        Returns:
            Document root node
        """
        if not self._cdp:
            raise CDPConnectionError("CDPPool not connected")
        return await self._cdp.get_document(depth=depth, session_id=session_id)

    async def send(
        self,
        method: str,
        params: dict | None = None,
        session_id: str | None = None,
        timeout: float = 10.0,
    ) -> Any:
        """
        Send raw CDP command.

        Args:
            method: CDP method name
            params: CDP method parameters
            session_id: Session ID for routing (optional)
            timeout: Timeout in seconds

        Returns:
            CDP response
        """
        if not self._cdp:
            raise CDPConnectionError("CDPPool not connected")
        return await self._cdp.send(
            method, params, timeout=timeout, session_id=session_id
        )

    async def verify_chromiumrl(self) -> None:
        """Verify ChromiumRL CDP domain is available. Raises if not WootzApp."""
        from w8_biayn.rewards.chromiumrl import verify_chromiumrl_available

        # Need a session for verification — use shared tab
        tab_info = await self.acquire_shared_tab(url="about:blank", fresh=True)
        session_id = tab_info["session_id"]
        try:
            await verify_chromiumrl_available(self._cdp, session_id=session_id)
        finally:
            await self.release_context(tab_info["context_id"])
        logger.info("CDPPool: ChromiumRL domain verified (WootzApp confirmed)")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "connected": self._connected,
            "cdp_url": self._cdp_url,
            "active_contexts": len(self._contexts),
            "context_ids": list(self._contexts.keys()),
        }

    # =========================================================================
    # Atomic ChromiumRL evaluation (single-eval lane)
    # =========================================================================

    def _find_context_by_session(self, session_id: str) -> PooledContext | None:
        """Find a pooled context by its session_id."""
        for ctx in self._contexts.values():
            if ctx.session_id == session_id:
                return ctx
        return None

    async def _reset_browser(self) -> bool:
        """ADB force-stop + restart WootzApp for clean CDP state."""
        if not self._adb_host:
            return False

        logger.warning(
            "CDPPool: browser reset requested for adb_host=%s, but w8-biayn "
            "does not bundle the SGLang ADB reset helper; continuing without reset",
            self._adb_host,
        )
        return False

    async def evaluate_chromiumrl(
        self,
        url: str,
        session_id: str,
        wait_time: float = 3.0,
        reference_dom_state: dict | None = None,
        capture_a11y: bool = False,
    ) -> tuple[dict, str]:
        """Atomic navigate + enable + collect ChromiumRL signals.

        Runs the full sequence in one actor call so no other env can
        interleave CDP commands between navigate and signal collection.
        Requires max_concurrency=1 on the CDPPool actor to prevent async
        interleaving of concurrent evaluate_chromiumrl calls.

        Steps:
            1. ADB browser reset (if flagged, for clean renderer)
            2. Re-acquire shared tab (if reset happened)
            3. Navigate + enable ChromiumRL (warmup pattern)
            4. Collect all signals

        Returns:
            Tuple of (signals_dict, session_id) — session_id may change
            after browser reset + re-acquire.
        """
        # Step 0: ADB browser reset for clean state (if flagged)
        if self._needs_browser_reset and self._adb_host:
            logger.warning("CDPPool: resetting browser before evaluation")
            reset_ok = await self._reset_browser()
            if reset_ok:
                # Re-acquire shared tab after reset
                tab_info = await self.acquire_shared_tab(url="about:blank")
                session_id = tab_info["session_id"]
            else:
                logger.warning("CDPPool: browser reset failed, continuing with existing session")
        else:
            # Validate existing session
            ctx = self._find_context_by_session(session_id)
            if ctx:
                new_sid = await self._validate_and_reattach(ctx.context_id)
                if new_sid:
                    session_id = new_sid

        # Preserve the live session for return — fresh eval tab session is
        # ephemeral and must NOT leak to callers (causes "Session not found").
        stable_session_id = session_id

        # Create a FRESH dedicated tab for evaluation (like capture_gold_signals.py).
        # Shared-tab reuse can leave stale renderer state that prevents compositor
        # layer capture. A fresh tab ensures clean renderer hooks.
        eval_target_id = None
        try:
            result = await self._cdp.send("Target.createTarget", {"url": "about:blank"})
            eval_target_id = result.get("targetId")
            attach_result = await self._cdp.send("Target.attachToTarget", {
                "targetId": eval_target_id,
                "flatten": True,
            })
            eval_session_id = attach_result.get("sessionId")
            if eval_session_id:
                logger.warning(
                    "CDPPool: created fresh eval tab target=%s session=%s",
                    eval_target_id, eval_session_id,
                )
                session_id = eval_session_id
            else:
                logger.warning("CDPPool: fresh tab attach failed, using existing session")
                eval_target_id = None  # Don't try to close it
        except Exception as e:
            logger.warning("CDPPool: fresh tab creation failed (%s), using existing session", e)
            eval_target_id = None

        try:
            # Re-enable domains before navigate (idempotent; ensures renderer hooks work)
            await self._ensure_session_domains(session_id)
            logger.warning("CDPPool: domains re-enabled on session %s before navigate", session_id)

            # Step 1: Navigate + enable ChromiumRL (warmup pattern)
            # Direct CDP calls — no Ray remote hops (we're inside the actor)
            from w8_biayn.rewards.chromiumrl import (
                navigate_and_enable_chromiumrl,
                _collect_signals_impl,
                get_main_frame_id,
            )
            chromiumrl_sid = await navigate_and_enable_chromiumrl(
                self._cdp, url, wait_time=wait_time, session_id=session_id,
            )
            logger.warning(
                "CDPPool: navigate_and_enable_chromiumrl done, chromiumrl_session=%s",
                chromiumrl_sid,
            )

            # Step 2: Collect all signals
            async def _send(method, params):
                return await self._cdp.send(method, params, session_id=session_id)

            async def _get_frame_id():
                return await get_main_frame_id(self._cdp, session_id=session_id)

            signals = await _collect_signals_impl(
                send_cmd=_send,
                get_frame_id=_get_frame_id,
                strict=True,
                log_prefix="[ChromiumRL Atomic]",
            )

            # Diagnostic: summarize what signals were actually collected
            _layers = len(signals.get("compositor_layers", []))
            _layout = len(signals.get("layout_elements", []))
            _gpu_kb = signals.get("total_gpu_memory", 0) / 1024
            logger.warning(
                "CDPPool: signals collected: compositor=%d layout=%d gpu=%.1fKB",
                _layers, _layout, _gpu_kb,
            )

            # Capture accessibility tree (if requested, for rubric mode)
            if capture_a11y:
                try:
                    a11y_result = await self._cdp.send(
                        "Accessibility.getFullAXTree", {}, session_id=session_id,
                    )
                    a11y_nodes = a11y_result.get("nodes", [])
                    if not a11y_nodes:
                        raise AccessibilityTreeCaptureError(
                            f"Accessibility.getFullAXTree returned empty for {url}"
                        )
                    signals["a11y_nodes"] = a11y_nodes
                    logger.warning(
                        "[A11Y] CDPPool: captured %d a11y nodes for %s",
                        len(a11y_nodes), url,
                    )
                except Exception as e:
                    if isinstance(e, AccessibilityTreeCaptureError):
                        raise
                    raise AccessibilityTreeCaptureError(
                        f"Accessibility.getFullAXTree failed for {url}: {e}"
                    ) from e

            # Save DOM state (best-effort) and optionally compare against reference
            from w8_biayn.rewards.chromiumrl import (
                save_dom_state,
                compare_dom_state,
                compute_dom_comparison_score,
            )
            dom_state = await save_dom_state(self._cdp, session_id=session_id)
            if dom_state and dom_state.get("nodes"):
                logger.warning("CDPPool: dom_state saved (%d nodes)",
                               len(dom_state["nodes"]))
            if dom_state:
                signals["dom_state"] = dom_state

            if reference_dom_state and dom_state:
                comparison = await compare_dom_state(
                    self._cdp, reference_dom_state, session_id=session_id,
                )
                dom_score = compute_dom_comparison_score(comparison)
                if dom_score >= 0:  # FEEDBACK_MISSING = -1
                    signals["dom_comparison_score"] = dom_score
                    signals["dom_comparison_detail"] = comparison
                    logger.warning("CDPPool: dom_comparison_score=%.4f", dom_score)
            elif reference_dom_state and not dom_state:
                logger.warning("CDPPool: dom_comparison skipped, save_dom_state empty")

            # Only flag browser reset when renderer is genuinely broken
            if _layers == 0 and _layout == 0:
                self._needs_browser_reset = True
                logger.warning("CDPPool: empty compositor+layout, flagging browser reset")

        finally:
            # ALWAYS close eval tab, even on exception/cancel
            if eval_target_id:
                try:
                    await self._cdp.send("Target.closeTarget", {"targetId": eval_target_id})
                    logger.warning("CDPPool: closed eval tab %s", eval_target_id)
                except Exception as e:
                    logger.warning("CDPPool: eval tab close failed: %s", e)
                # Let CDP settle between queued evals before next fresh tab
                await asyncio.sleep(0.3)

        return signals, stable_session_id
