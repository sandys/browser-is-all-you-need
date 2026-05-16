"""
Chrome DevTools Protocol client for browser automation.

Connects to a running browser instance (typically Android Chrome)
via WebSocket for screenshot capture and page navigation.
"""

import asyncio
import json
import logging
import aiohttp
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import AsyncIterator

logger = logging.getLogger(__name__)


class CDPConnectionError(Exception):
    """Raised when CDP connection fails after all retries."""

    pass


class CDPClient:
    """
    Chrome DevTools Protocol client.
    Connects to a running browser instance via WebSocket.

    Includes robust retry logic with exponential backoff for production use.
    Features auto-reconnection when connection drops during operation.
    """

    def __init__(
        self,
        ws_url: str = "ws://localhost:9224",
        max_retries: int = 30,
        initial_delay: float = 1.0,
        max_delay: float = 10.0,
        backoff_factor: float = 1.5,
        auto_reconnect: bool = True,
        reconnect_attempts: int = 5,
        use_browser_websocket: bool = True,
    ):
        self.ws_url = ws_url
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self._message_id = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._listening = False
        self._reconnecting = False  # Prevent concurrent reconnection attempts
        self._receiver_task: Optional[asyncio.Task] = None
        self._reset_lock = asyncio.Lock()

        # Retry configuration
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

        # Auto-reconnection configuration
        self.auto_reconnect = auto_reconnect
        self.reconnect_attempts = reconnect_attempts
        self._reconnect_backoff_factor = 2.0
        self._reconnect_max_delay = 16.0

        # Browser WebSocket mode for per_worker_tabs
        # When True, connects to browser-level WebSocket (/devtools/browser/...)
        # which supports sessionId routing to multiple tabs.
        # When False (default), connects to page-level WebSocket (/devtools/page/...)
        # which is direct to a single page.
        self._use_browser_websocket = use_browser_websocket

    def _get_http_url(self) -> str:
        """Convert WebSocket URL to HTTP URL for discovery."""
        http_url = self.ws_url.replace("ws://", "http://").replace("wss://", "https://")
        if http_url.endswith("/json"):
            return http_url
        # Remove trailing slash if present
        http_url = http_url.rstrip("/")
        return f"{http_url}/json"

    def _rewrite_ws_url(self, ws_debug_url: str) -> str:
        """Rewrite Chrome's localhost WebSocket URL to use our target host."""
        import urllib.parse

        # Get base URL from our configured ws_url
        base_ws = self.ws_url.replace("http://", "ws://").replace("https://", "wss://")
        if base_ws.endswith("/json"):
            base_ws = base_ws[:-5]
        base_ws = base_ws.rstrip("/")

        # Extract path from Chrome's URL (e.g., /devtools/page/ABC123)
        parsed = urllib.parse.urlparse(ws_debug_url)
        return f"{base_ws}{parsed.path}"

    async def wait_for_ready(self, timeout: float = 120.0) -> bool:
        """
        Wait for CDP endpoint to be ready before attempting connection.
        Returns True if ready, False if timeout.
        """
        http_url = self._get_http_url().replace("/json", "/json/version")
        headers = {"Host": "localhost"}
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Waiting for CDP to be ready at {http_url} (timeout: {timeout}s)")

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(http_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            logger.info(f"CDP ready: {data.get('Browser', 'Unknown browser')}")
                            return True
            except Exception as e:
                logger.debug(f"CDP not ready yet: {e}")
            await asyncio.sleep(2)

        logger.error(f"CDP not ready after {timeout}s")
        return False

    async def cleanup_stale_tabs(self, timeout: float = 10.0) -> bool:
        """
        Clean up stale browser tabs using browser-level WebSocket.

        This connects directly to the browser endpoint (not a page) and
        closes all existing page targets, then creates a fresh about:blank tab.

        This is essential between tasks to prevent Page.enable timeouts
        caused by accumulated browser state.

        Returns True if cleanup succeeded, False otherwise.
        """
        # Get browser WebSocket URL
        http_url = self._get_http_url().replace("/json", "/json/version")
        headers = {"Host": "localhost"}

        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                # Get browser debugger URL
                async with session.get(http_url) as resp:
                    if resp.status != 200:
                        logger.warning(f"Cannot get browser info: {resp.status}")
                        return False
                    data = await resp.json()
                    browser_ws_url = data.get("webSocketDebuggerUrl", "")

                if not browser_ws_url:
                    logger.warning("No browser WebSocket URL available")
                    return False

                # Rewrite URL to use our target host
                browser_ws_url = self._rewrite_ws_url(browser_ws_url)
                logger.info(f"Cleaning up stale tabs via {browser_ws_url}")

                # Connect to browser-level WebSocket
                async with session.ws_connect(
                    browser_ws_url,
                    timeout=timeout,
                    headers={"Host": "localhost"}
                ) as ws:
                    msg_id = 1

                    # Get all targets
                    await ws.send_json({
                        "id": msg_id,
                        "method": "Target.getTargets",
                        "params": {}
                    })

                    response = await asyncio.wait_for(ws.receive_json(), timeout=5.0)
                    targets = response.get("result", {}).get("targetInfos", [])

                    # Close all page targets
                    closed_count = 0
                    for target in targets:
                        if target.get("type") == "page":
                            msg_id += 1
                            await ws.send_json({
                                "id": msg_id,
                                "method": "Target.closeTarget",
                                "params": {"targetId": target.get("targetId")}
                            })
                            try:
                                await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                                closed_count += 1
                            except asyncio.TimeoutError:
                                pass

                    # Create a fresh about:blank tab
                    msg_id += 1
                    await ws.send_json({
                        "id": msg_id,
                        "method": "Target.createTarget",
                        "params": {"url": "about:blank"}
                    })
                    await asyncio.wait_for(ws.receive_json(), timeout=5.0)

                    logger.info(f"Cleaned up {closed_count} stale tabs, created fresh tab")
                    return True

        except Exception as e:
            logger.warning(f"Tab cleanup failed: {e}")
            return False

    # ============================================
    # BrowserContext Methods (for parallel rollouts)
    # ============================================

    async def create_browser_context(self) -> str:
        """
        Create isolated browser context (like incognito).

        Each context has separate:
        - Cookies
        - localStorage / IndexedDB
        - Cache storage
        - Network socket pool (NetworkContext)

        Returns:
            browserContextId string
        """
        result = await self.send("Target.createBrowserContext", {
            "disposeOnDetach": True,
        })
        context_id = result.get("browserContextId")
        logger.info(f"Created BrowserContext: {context_id}")
        return context_id

    async def dispose_browser_context(self, context_id: str) -> bool:
        """Dispose browser context and all its tabs."""
        try:
            await self.send("Target.disposeBrowserContext", {
                "browserContextId": context_id,
            })
            logger.info(f"Disposed BrowserContext: {context_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to dispose BrowserContext {context_id}: {e}")
            return False

    async def create_target_in_context(
        self,
        url: str,
        context_id: str,
    ) -> tuple[str, str]:
        """
        Create a new tab in a specific browser context.

        Returns:
            Tuple of (targetId, sessionId)
        """
        # Create target in context
        result = await self.send("Target.createTarget", {
            "url": url,
            "browserContextId": context_id,
        })
        target_id = result.get("targetId")

        # Attach to target to get sessionId for routing
        attach_result = await self.send("Target.attachToTarget", {
            "targetId": target_id,
            "flatten": True,  # Required for sessionId routing
        })
        session_id = attach_result.get("sessionId")

        # Enable required domains for this session
        await self.send("Page.enable", session_id=session_id)
        await self.send("DOM.enable", session_id=session_id)
        await self.send("Runtime.enable", session_id=session_id)

        return target_id, session_id

    async def close_target(self, target_id: str) -> bool:
        """Close a specific target/tab."""
        try:
            result = await self.send("Target.closeTarget", {"targetId": target_id})
            return result.get("success", False)
        except Exception as e:
            logger.warning(f"Failed to close target {target_id}: {e}")
            return False

    async def connect(self):
        """Connect to CDP WebSocket with robust retry and exponential backoff."""
        http_url = self._get_http_url()
        logger.info(f"Connecting to CDP discovery at {http_url}")

        # Chrome rejects external Host headers. Force localhost.
        headers = {"Host": "localhost"}
        delay = self.initial_delay
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"CDP connection attempt {attempt}/{self.max_retries}")

                # Discovery: Get WebSocket URL
                timeout = aiohttp.ClientTimeout(total=10)
                async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
                    if self._use_browser_websocket:
                        # BROWSER-LEVEL WebSocket: Connect to browser endpoint
                        # This enables sessionId routing for per_worker_tabs mode
                        version_url = http_url.replace("/json", "/json/version")
                        async with session.get(version_url) as resp:
                            if resp.status != 200:
                                raise RuntimeError(f"CDP version returned status {resp.status}")

                            data = await resp.json()
                            ws_debug_url = data.get("webSocketDebuggerUrl")
                            if not ws_debug_url:
                                raise RuntimeError("No browser WebSocket URL in /json/version")

                            logger.info(f"Using browser-level WebSocket: {ws_debug_url}")
                    else:
                        # PAGE-LEVEL WebSocket: Connect to first page target (default)
                        async with session.get(http_url) as resp:
                            if resp.status != 200:
                                raise RuntimeError(f"CDP discovery returned status {resp.status}")

                            text = await resp.text()
                            if not text.strip():
                                raise RuntimeError("CDP discovery returned empty response")

                            try:
                                targets = json.loads(text)
                            except json.JSONDecodeError as e:
                                raise RuntimeError(f"CDP discovery returned invalid JSON: {e}")

                            if not isinstance(targets, list):
                                raise RuntimeError(f"CDP discovery returned non-list: {type(targets)}")

                            # Find a page target
                            page_target = next(
                                (t for t in targets if t.get("type") == "page"), None
                            )
                            if not page_target and targets:
                                page_target = targets[0]

                            if not page_target:
                                raise RuntimeError("No debuggable pages found")

                            if "webSocketDebuggerUrl" not in page_target:
                                raise RuntimeError(f"Target missing webSocketDebuggerUrl: {page_target}")

                            ws_debug_url = page_target["webSocketDebuggerUrl"]
                            logger.info(f"Found target: {page_target.get('title', 'untitled')} -> {ws_debug_url}")

                # Rewrite URL to use our host instead of localhost
                final_ws_url = self._rewrite_ws_url(ws_debug_url)
                logger.info(f"Rewritten WebSocket URL: {final_ws_url}")

                # Connect WebSocket
                self.session = aiohttp.ClientSession(headers=headers)
                self.ws = await self.session.ws_connect(
                    final_ws_url,
                    timeout=10,  # Reduced from 30s to fail fast
                    heartbeat=10,
                    max_msg_size=50 * 1024 * 1024,  # 50MB for large accessibility trees
                )
                self._listening = True

                # Start receiver task and give it a moment to start
                self._receiver_task = asyncio.create_task(self._receive_messages())
                await asyncio.sleep(0.1)  # Allow receiver to start

                # Enable required domains (only for page-level WebSocket)
                # When using browser-level WebSocket, domains are enabled per-session
                # after attaching to a tab via Target.attachToTarget
                if not self._use_browser_websocket:
                    await self.send("Page.enable")
                    await self.send("DOM.enable")
                    await self.send("Runtime.enable")
                    await self.send("Accessibility.enable")

                # Validate connection is working
                version = await self.send("Browser.getVersion")
                logger.info(f"Connected to {version.get('product', 'Unknown')}")

                return

            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                logger.warning(
                    f"CDP connection attempt {attempt}/{self.max_retries} failed: {type(e).__name__}: {e}"
                )

                # Clean up failed connection attempt
                if self.session:
                    try:
                        await self.session.close()
                    except Exception:
                        pass
                    self.session = None
                self.ws = None
                self._listening = False

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)

        raise CDPConnectionError(
            f"Failed to connect to CDP at {self.ws_url} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def send(
        self,
        method: str,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
        session_id: Optional[str] = None,
    ) -> Any:
        """Send CDP command and wait for response with timeout.

        Automatically attempts reconnection if the connection is dead.

        Args:
            method: CDP method name (e.g., "Page.navigate")
            params: CDP method parameters
            timeout: Timeout in seconds
            session_id: Optional session ID for tab targeting (per_worker_tabs mode).
                       When specified, the command is routed to a specific tab's session.
        """
        # Check connection and attempt reconnect if needed
        if not self.ws or self.ws.closed or not self._listening:
            if self.auto_reconnect and not self._reconnecting:
                logger.warning(f"CDP not connected for '{method}', attempting reconnect...")
                if not await self.ensure_connected():
                    raise RuntimeError("CDP WebSocket not connected and reconnection failed")
            else:
                raise RuntimeError("CDP WebSocket not connected")

        self._message_id += 1
        msg_id = self._message_id

        message = {"id": msg_id, "method": method}
        if session_id:
            message["sessionId"] = session_id  # Route to specific tab
        if params:
            message["params"] = params

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[msg_id] = future

        try:
            await self.ws.send_json(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"CDP command '{method}' timed out after {timeout}s")
        except Exception as e:
            self._pending.pop(msg_id, None)
            # If send failed due to connection issue, try reconnect and retry once
            if self.auto_reconnect and ("closed" in str(e).lower() or "connection" in str(e).lower()):
                logger.warning(f"CDP send failed ({e}), attempting reconnect and retry...")
                if await self.ensure_connected():
                    # Retry the send once after reconnection
                    return await self._send_internal(method, params, timeout, session_id)
            raise

    async def _send_internal(
        self,
        method: str,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
        session_id: Optional[str] = None,
    ) -> Any:
        """Internal send without auto-reconnect (used for retry after reconnection)."""
        if not self.ws or self.ws.closed:
            raise RuntimeError("CDP WebSocket not connected")

        self._message_id += 1
        msg_id = self._message_id

        message = {"id": msg_id, "method": method}
        if session_id:
            message["sessionId"] = session_id  # Route to specific tab
        if params:
            message["params"] = params

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[msg_id] = future

        try:
            await self.ws.send_json(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"CDP command '{method}' timed out after {timeout}s")
        except Exception as e:
            self._pending.pop(msg_id, None)
            raise

    async def _receive_messages(self):
        """Background task to receive CDP messages."""
        logger.debug("CDP message receiver started")
        try:
            while self._listening and self.ws and not self.ws.closed:
                try:
                    msg = await self.ws.receive()
                    logger.debug(f"CDP received message type: {msg.type}")
                except Exception as e:
                    logger.error(f"Error receiving message: {e}")
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON from CDP: {msg.data[:100]}")
                        continue

                    if "id" in data and data["id"] in self._pending:
                        future = self._pending.pop(data["id"])
                        if "error" in data:
                            future.set_exception(Exception(data["error"].get("message", str(data["error"]))))
                        else:
                            future.set_result(data.get("result", {}))
                    elif "method" in data:
                        # CDP event - could be logged or handled
                        logger.debug(f"CDP event: {data['method']}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"CDP WebSocket error: {msg.data}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("CDP WebSocket closed by server")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSING:
                    logger.info("CDP WebSocket closing")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    logger.info(f"CDP WebSocket close frame received: {msg.data}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"CDP WebSocket receive error: {e}")
        finally:
            self._listening = False
            # Cancel pending futures
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError("CDP connection closed"))
            self._pending.clear()

    async def is_connected(self) -> bool:
        """Check if connection is still alive."""
        if not self.ws or self.ws.closed or not self._listening:
            return False
        try:
            await self.send("Browser.getVersion", timeout=5.0)
            return True
        except Exception:
            return False

    async def close(self):
        """Close CDP connection gracefully."""
        self._listening = False
        if self._receiver_task and not self._receiver_task.done():
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
        if self.ws and not self.ws.closed:
            try:
                await self.ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
        if self.session and not self.session.closed:
            try:
                await self.session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
        self.ws = None
        self.session = None
        self._receiver_task = None

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect to CDP after connection loss.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        if self._reconnecting:
            logger.debug("Reconnection already in progress")
            return False

        self._reconnecting = True
        logger.warning("CDP connection lost, attempting reconnection...")

        try:
            # Close existing connection cleanly
            await self.close()

            # Small delay before reconnecting
            await asyncio.sleep(1.0)

            # Attempt reconnection with reduced retries and wider backoff
            original_max_retries = self.max_retries
            original_backoff_factor = self.backoff_factor
            original_max_delay = self.max_delay
            self.max_retries = self.reconnect_attempts
            self.backoff_factor = self._reconnect_backoff_factor
            self.max_delay = self._reconnect_max_delay

            try:
                await self.connect()
                logger.info("CDP reconnection successful")
                return True
            except CDPConnectionError as e:
                logger.error(f"CDP reconnection failed: {e}")
                return False
            finally:
                self.max_retries = original_max_retries
                self.backoff_factor = original_backoff_factor
                self.max_delay = original_max_delay

        finally:
            self._reconnecting = False

    async def ensure_connected(self) -> bool:
        """
        Ensure CDP connection is alive, reconnecting if necessary.

        Returns:
            True if connected (or reconnected successfully), False otherwise.
        """
        if self.ws and not self.ws.closed and self._listening:
            return True

        if not self.auto_reconnect:
            return False

        return await self.reconnect()

    async def reset_for_capture(self, timeout_s: float = 20.0) -> None:
        """
        Atomic reset: tear down CDP state and reconnect with clean tabs.

        Used between gold/patched captures to ensure no state leakage.
        Serialized via _reset_lock to prevent concurrent resets.

        Raises:
            CDPConnectionError: If reconnection or tab cleanup fails.
        """
        async with self._reset_lock:
            # 1. Best-effort disable ChromiumRL tracing
            try:
                await asyncio.wait_for(
                    self.send("ChromiumRL.disable"),
                    timeout=5.0,
                )
            except Exception:
                pass  # May not be enabled, that's fine

            # 2. Fully tear down current transport
            await self.close()

            # 3. Reconnect fresh
            await asyncio.wait_for(self.connect(), timeout=timeout_s)

            # 4. Clean tab state
            ok = await asyncio.wait_for(
                self.cleanup_stale_tabs(),
                timeout=timeout_s,
            )
            if not ok:
                raise CDPConnectionError(
                    "cleanup_stale_tabs failed after reconnect"
                )

    async def capture_screenshot(
        self,
        format: str = "png",
        session_id: Optional[str] = None,
    ) -> str:
        """Capture screenshot and return as base64 string.

        Args:
            format: Image format (png or jpeg)
            session_id: Optional session ID for tab targeting
        """
        result = await self.send(
            "Page.captureScreenshot",
            {"format": format},
            session_id=session_id,
            timeout=30.0,
        )
        return result["data"]

    async def navigate(
        self,
        url: str,
        wait_time: float = 3.0,
        session_id: Optional[str] = None,
    ):
        """Navigate to a URL and wait for it to load.

        Args:
            url: URL to navigate to
            wait_time: Time to wait after navigation
            session_id: Optional session ID for tab targeting
        """
        try:
            result = await self.send(
                "Page.navigate",
                {"url": url},
                session_id=session_id,
            )
            logger.debug(f"Navigation result: {result}")
        except Exception as e:
            logger.warning(f"Navigation error (may be expected): {e}")
        # Wait for page to load
        await asyncio.sleep(wait_time)

    async def reload(self, wait_time: float = 2.0):
        """Reload the current page."""
        await self.send("Page.reload")
        await asyncio.sleep(wait_time)

    async def set_desktop_mode(
        self,
        width: int = 1280,
        height: int = 720,
        device_scale_factor: float = 1.0,
        mobile: bool = False,
        user_agent: str = None,
        session_id: Optional[str] = None,
    ):
        """
        Set desktop viewport and user agent via CDP Emulation.

        This is the most reliable way to get desktop rendering on Android Chrome,
        as it overrides the browser's default mobile behavior at the protocol level.

        Args:
            width: Viewport width
            height: Viewport height
            device_scale_factor: Device scale factor
            mobile: Mobile emulation mode
            user_agent: User agent string
            session_id: Optional session ID for tab targeting
        """
        if user_agent is None:
            user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )

        logger.info(f"Setting desktop mode: {width}x{height}, mobile={mobile}")

        # Set device metrics (viewport size)
        await self.send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": width,
                "height": height,
                "deviceScaleFactor": device_scale_factor,
                "mobile": mobile,
            },
            session_id=session_id,
        )

        # Set user agent
        await self.send(
            "Emulation.setUserAgentOverride",
            {
                "userAgent": user_agent,
            },
            session_id=session_id,
        )

        logger.info(f"Desktop mode set: {width}x{height}, UA={user_agent[:50]}...")

    # ============================================
    # DOM Snapshot Domain
    # ============================================

    async def capture_dom_snapshot(
        self,
        computed_styles: list = None,
        include_paint_order: bool = True,
        include_dom_rects: bool = True,
    ) -> Dict:
        """
        Capture complete DOM snapshot with layout information.

        Args:
            computed_styles: List of CSS properties to capture
            include_paint_order: Include paint order information
            include_dom_rects: Include bounding rectangles

        Returns:
            DOMSnapshot.captureSnapshot result
        """
        if computed_styles is None:
            computed_styles = ["display", "visibility", "opacity", "position"]

        return await self.send("DOMSnapshot.captureSnapshot", {
            "computedStyles": computed_styles,
            "includePaintOrder": include_paint_order,
            "includeDOMRects": include_dom_rects,
        })

    # ============================================
    # DOM Domain Extensions
    # ============================================

    async def get_document(
        self,
        depth: int = -1,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Get the root DOM node.

        Args:
            depth: Depth of tree to return (-1 = full tree)
            session_id: Optional session ID for tab targeting

        Returns:
            Dict with 'root' node
        """
        return await self.send(
            "DOM.getDocument",
            {"depth": depth},
            session_id=session_id,
        )

    # ============================================
    # Runtime Domain Extensions
    # ============================================

    async def evaluate(
        self,
        expression: str,
        return_by_value: bool = True,
        await_promise: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate JavaScript expression in page context.

        Args:
            expression: JavaScript expression to evaluate
            return_by_value: Return result by value (vs remote object)
            await_promise: Wait for promise resolution
            session_id: Optional session ID for tab targeting

        Returns:
            Dict with 'result' containing evaluation result
        """
        return await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": await_promise,
            },
            session_id=session_id,
        )

    # ============================================
    # RolloutContext for Parallel Rollouts
    # ============================================

    @asynccontextmanager
    async def rollout_context(
        self, url: str = "about:blank"
    ) -> "AsyncIterator[RolloutContext]":
        """
        Context manager for isolated rollout.

        Creates an isolated BrowserContext with its own tab, enabling parallel
        rollouts without state pollution between them.

        Usage:
            async with cdp.rollout_context("http://task.example.com") as ctx:
                await ctx.set_desktop_mode()
                screenshot = await ctx.capture_screenshot()
            # Automatically cleaned up

        Args:
            url: Initial URL to navigate to (default: about:blank)

        Yields:
            RolloutContext with isolated browser state
        """
        context_id = await self.create_browser_context()
        target_id, session_id = await self.create_target_in_context(url, context_id)

        ctx = RolloutContext(
            context_id=context_id,
            target_id=target_id,
            session_id=session_id,
            cdp=self,
        )

        try:
            yield ctx
        finally:
            await ctx.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()


@dataclass
class RolloutContext:
    """
    Encapsulates an isolated rollout environment.

    Each RolloutContext has its own BrowserContext (like incognito) with
    separate cookies, localStorage, cache, and network sockets. This enables
    parallel rollouts without state pollution.

    Created via CDPClient.rollout_context() context manager:
        async with cdp.rollout_context("http://task.example.com") as ctx:
            await ctx.set_desktop_mode()
            screenshot = await ctx.capture_screenshot()
        # Automatically cleaned up
    """

    context_id: str
    target_id: str
    session_id: str
    cdp: CDPClient

    async def navigate(self, url: str, wait_time: float = 3.0) -> None:
        """Navigate to URL in this context's tab."""
        await self.cdp.navigate(url, wait_time, session_id=self.session_id)

    async def capture_screenshot(self, format: str = "png") -> str:
        """Capture screenshot from this context's tab."""
        return await self.cdp.capture_screenshot(format, session_id=self.session_id)

    async def evaluate(self, expression: str) -> Dict:
        """Evaluate JavaScript in this context's tab."""
        return await self.cdp.evaluate(expression, session_id=self.session_id)

    async def set_desktop_mode(
        self,
        width: int = 1280,
        height: int = 720,
        device_scale_factor: float = 1.0,
        mobile: bool = False,
        user_agent: str = None,
    ) -> None:
        """Set desktop viewport in this context's tab."""
        await self.cdp.set_desktop_mode(
            width=width,
            height=height,
            device_scale_factor=device_scale_factor,
            mobile=mobile,
            user_agent=user_agent,
            session_id=self.session_id,
        )

    async def get_document(self, depth: int = -1) -> Dict:
        """Get DOM document from this context's tab."""
        return await self.cdp.get_document(depth, session_id=self.session_id)

    async def close(self) -> None:
        """Close this context's tab and dispose the browser context."""
        await self.cdp.close_target(self.target_id)
        await self.cdp.dispose_browser_context(self.context_id)
