"""Tests for the network-health probes, watchdog, and launch wiring."""

from __future__ import annotations

from pathlib import Path

from w8_biayn.net_health import (
    DEFAULT_ENDPOINTS,
    EndpointStatus,
    NetWatchdog,
    check_endpoints,
    classify_error_text,
    preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def _probe_factory(down: set[str]):
    def fake_probe(host: str, port: int) -> EndpointStatus:
        if host in down:
            return EndpointStatus(host, port, False, "dns_failure(-2)")
        return EndpointStatus(host, port, True)

    return fake_probe


def test_classify_error_text_maps_network_signatures() -> None:
    assert classify_error_text("Unable to find the server at compute.googleapis.com") == "dns_failure"
    assert classify_error_text("fatal: unable to access '...': Failed to connect to github.com") == "connect_failure"
    assert classify_error_text("error: Could not fetch origin") == "connect_failure"
    assert classify_error_text("read tcp ... connection timed out") == "timeout"
    assert classify_error_text("AssertionError: shapes differ") is None


def test_watchdog_emits_transitions_and_heartbeat() -> None:
    events: list[tuple[str, str]] = []
    down: set[str] = set()
    dog = NetWatchdog(
        on_event=lambda e, d: events.append((e, d)),
        endpoints=(("compute.googleapis.com", 443),),
        still_degraded_every=2,
        probe_fn=_probe_factory(down),
    )

    dog.check_once()
    assert events == []  # healthy: silent

    down.add("compute.googleapis.com")
    dog.check_once()
    assert events[-1][0] == "net_degraded" and "dns_failure" in events[-1][1]

    dog.check_once()  # still down (cycle 1): quiet
    dog.check_once()  # still down (cycle 2): heartbeat
    assert events[-1][0] == "net_still_degraded"

    down.clear()
    dog.check_once()
    assert events[-1] == ("net_recovered", "compute.googleapis.com:443")
    # events must never fire while healthy
    dog.check_once()
    assert events[-1][0] == "net_recovered"


def test_preflight_retries_failures_then_reports_final_state() -> None:
    calls: list[str] = []
    healed = {"count": 0}

    def flaky_probe(host: str, port: int) -> EndpointStatus:
        calls.append(host)
        if host == "github.com" and healed["count"] < 1:
            healed["count"] += 1
            return EndpointStatus(host, port, False, "connect_failure(TimeoutError)")
        return EndpointStatus(host, port, True)

    statuses = preflight(
        (("compute.googleapis.com", 443), ("github.com", 443)),
        attempts=3,
        probe_fn=flaky_probe,
        sleep_fn=lambda _s: None,
    )

    assert all(s.ok for s in statuses)
    # only the FAILED endpoint is re-probed on retry
    assert calls.count("compute.googleapis.com") == 1
    assert calls.count("github.com") == 2


def test_check_endpoints_covers_the_launch_dependencies() -> None:
    hosts = {host for host, _port in DEFAULT_ENDPOINTS}
    assert {"compute.googleapis.com", "storage.googleapis.com", "github.com", "api.wandb.ai"} <= hosts
    statuses = check_endpoints((("h", 1),), probe_fn=lambda h, p: EndpointStatus(h, p, True))
    assert statuses[0].describe() == "h:1=ok"


def test_launcher_wires_preflight_watchdog_and_net_events() -> None:
    text = (ROOT / "src/w8_biayn/cloud_launch.py").read_text(encoding="utf-8")
    assert "net_health.preflight()" in text
    assert "net_health.NetWatchdog(on_event=_launch_event)" in text
    assert "watchdog.stop()" in text
    assert "GCP endpoints unreachable after retries" in text  # fail fast pre-spend
    assert 'run.summary["pipeline/net_degraded_events"]' in text
