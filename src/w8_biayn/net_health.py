"""Network reachability probes + a launch watchdog that reports honestly.

A paid launch once stalled ~10 minutes in silent googleapiclient retries
because this machine's DNS could not resolve compute.googleapis.com -- and the
operator learned it from the operator, not the tooling. Every reachability
problem must bubble up loudly: a preflight check before money is spent, and a
background watchdog that emits net_degraded / net_recovered events into the
launch-event trail (console + W&B table) the moment connectivity changes.

Pure-stdlib on purpose: the checks must work exactly when the network does not.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

#: endpoints a launch depends on: GCP control plane, GCS, GitHub (clones), W&B.
DEFAULT_ENDPOINTS: tuple[tuple[str, int], ...] = (
    ("compute.googleapis.com", 443),
    ("storage.googleapis.com", 443),
    ("github.com", 443),
    ("api.wandb.ai", 443),
)

#: substrings that identify a network/reachability failure in tool output.
_ERROR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("unable to find the server", "dns_failure"),
    ("temporary failure in name resolution", "dns_failure"),
    ("name or service not known", "dns_failure"),
    ("could not resolve host", "dns_failure"),
    ("failed to connect", "connect_failure"),
    ("connection refused", "connect_failure"),
    ("connection reset", "connect_failure"),
    ("network is unreachable", "connect_failure"),
    ("connection timed out", "timeout"),
    ("timed out", "timeout"),
    ("could not fetch origin", "connect_failure"),
)


@dataclass(frozen=True)
class EndpointStatus:
    host: str
    port: int
    ok: bool
    error: str = ""

    def describe(self) -> str:
        state = "ok" if self.ok else self.error or "failed"
        return f"{self.host}:{self.port}={state}"


def probe(host: str, port: int, *, timeout: float = 5.0) -> EndpointStatus:
    """DNS-resolve and TCP-connect one endpoint; never raises."""

    try:
        socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return EndpointStatus(host, port, False, f"dns_failure({exc.errno})")
    except Exception as exc:  # noqa: BLE001 - a probe must never take the caller down
        return EndpointStatus(host, port, False, f"dns_error({type(exc).__name__})")
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
    except Exception as exc:  # noqa: BLE001
        return EndpointStatus(host, port, False, f"connect_failure({type(exc).__name__})")
    return EndpointStatus(host, port, True)


def check_endpoints(
    endpoints: Sequence[tuple[str, int]] = DEFAULT_ENDPOINTS,
    *,
    probe_fn: Callable[[str, int], EndpointStatus] | None = None,
) -> list[EndpointStatus]:
    fn = probe_fn or (lambda host, port: probe(host, port))
    return [fn(host, port) for host, port in endpoints]


def classify_error_text(text: str) -> str | None:
    """Map tool stderr/exception text to a network-failure kind, or None."""

    lowered = (text or "").lower()
    for signature, kind in _ERROR_SIGNATURES:
        if signature in lowered:
            return kind
    return None


class NetWatchdog(threading.Thread):
    """Background reachability monitor emitting events on state transitions.

    ``on_event(event, detail)`` fires with ``net_degraded`` when an endpoint
    stops being reachable, ``net_recovered`` when it comes back, and a periodic
    ``net_still_degraded`` heartbeat while it stays down -- so a silent stall
    (e.g. local WSL2 DNS dying mid-provision) is visible in the console log and
    the pipeline's launch-events table without anyone asking.
    """

    def __init__(
        self,
        *,
        on_event: Callable[[str, str], None],
        endpoints: Sequence[tuple[str, int]] = DEFAULT_ENDPOINTS,
        interval_s: float = 60.0,
        still_degraded_every: int = 5,
        probe_fn: Callable[[str, int], EndpointStatus] | None = None,
    ) -> None:
        super().__init__(name="w8-net-watchdog", daemon=True)
        self._on_event = on_event
        self._endpoints = tuple(endpoints)
        self._interval_s = interval_s
        self._still_every = max(1, still_degraded_every)
        self._probe_fn = probe_fn or (lambda host, port: probe(host, port))
        self._stop = threading.Event()
        self._down: dict[str, str] = {}
        self._down_cycles: dict[str, int] = {}

    def stop(self) -> None:
        self._stop.set()

    def check_once(self) -> None:
        """One probe sweep; emits transition events. Split out for tests."""

        for host, port in self._endpoints:
            status = self._probe_fn(host, port)
            key = f"{host}:{port}"
            if status.ok and key in self._down:
                self._down.pop(key, None)
                self._down_cycles.pop(key, None)
                self._emit("net_recovered", key)
            elif not status.ok and key not in self._down:
                self._down[key] = status.error
                self._down_cycles[key] = 0
                self._emit("net_degraded", f"{key} {status.error}")
            elif not status.ok:
                self._down_cycles[key] = self._down_cycles.get(key, 0) + 1
                if self._down_cycles[key] % self._still_every == 0:
                    self._emit("net_still_degraded", f"{key} {status.error}")

    def run(self) -> None:  # pragma: no cover - thin loop over check_once
        while not self._stop.wait(self._interval_s):
            try:
                self.check_once()
            except Exception:  # noqa: BLE001 - the watchdog must never die loudly
                pass

    def _emit(self, event: str, detail: str) -> None:
        try:
            self._on_event(event, detail)
        except Exception:  # noqa: BLE001
            pass


def preflight(
    endpoints: Iterable[tuple[str, int]] = DEFAULT_ENDPOINTS,
    *,
    attempts: int = 3,
    sleep_s: float = 10.0,
    probe_fn: Callable[[str, int], EndpointStatus] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[EndpointStatus]:
    """Probe all endpoints, retrying failures; returns the FINAL statuses.

    Retries absorb a transient blip so a launch is not aborted for a hiccup,
    while a genuinely dead network fails fast BEFORE a paid box exists.
    """

    endpoint_list = list(endpoints)
    statuses = check_endpoints(endpoint_list, probe_fn=probe_fn)
    for _ in range(attempts - 1):
        failed = [(s.host, s.port) for s in statuses if not s.ok]
        if not failed:
            break
        sleep_fn(sleep_s)
        retried = {(s.host, s.port): s for s in check_endpoints(failed, probe_fn=probe_fn)}
        statuses = [retried.get((s.host, s.port), s) for s in statuses]
    return statuses
