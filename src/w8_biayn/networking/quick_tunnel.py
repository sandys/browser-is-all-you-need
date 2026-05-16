"""Cloudflare quick-tunnel helpers.

Account-less quick tunnels occasionally publish ``*.trycloudflare.com`` names
before the trainer host's resolver can resolve those names. For framework
smokes we still want the HTTPS request to preserve the original hostname for
SNI and the Host header, but connect to a Cloudflare edge address resolved from
``trycloudflare.com``. ``aiohttp`` supports this with a custom resolver.
"""

from __future__ import annotations

import asyncio
import socket
import urllib.error
from urllib.parse import urlparse

import aiohttp
from aiohttp.abc import AbstractResolver


def is_trycloudflare_host(host: str | None) -> bool:
    return bool(host and host.lower().endswith(".trycloudflare.com"))


def trycloudflare_connect_to_args(url: str) -> list[str]:
    """Return curl ``--connect-to`` args for a trycloudflare URL.

    The URL hostname remains the request hostname, which preserves SNI and the
    public Host header. Only the TCP connect destination is changed.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if not is_trycloudflare_host(host):
        return []
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    target_port = 443 if parsed.scheme == "https" else 80
    return ["--connect-to", f"{host}:{port}:trycloudflare.com:{target_port}"]


def is_dns_resolution_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, socket.gaierror):
        return True
    text = str(exc).lower()
    return (
        "name or service not known" in text
        or "temporary failure in name resolution" in text
        or "nodename nor servname provided" in text
        or "name resolution" in text
    )


def aiohttp_connector_for_url(url: str) -> aiohttp.TCPConnector | None:
    parsed = urlparse(url)
    if not is_trycloudflare_host(parsed.hostname):
        return None
    return aiohttp.TCPConnector(resolver=TryCloudflareResolver())


class TryCloudflareResolver(AbstractResolver):
    """Resolve quick-tunnel hostnames through the base trycloudflare edge."""

    def __init__(self, edge_host: str = "trycloudflare.com") -> None:
        self.edge_host = edge_host
        self._default = aiohttp.resolver.DefaultResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[dict]:
        if not is_trycloudflare_host(host):
            return await self._default.resolve(host, port, family)

        infos = await asyncio.to_thread(
            socket.getaddrinfo,
            self.edge_host,
            port,
            family,
            socket.SOCK_STREAM,
        )
        addresses: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for fam, _socktype, proto, _canonname, sockaddr in infos:
            address = sockaddr[0]
            resolved_port = sockaddr[1]
            key = (address, resolved_port)
            if key in seen:
                continue
            seen.add(key)
            addresses.append(
                {
                    "hostname": host,
                    "host": address,
                    "port": resolved_port,
                    "family": fam,
                    "proto": proto,
                    "flags": socket.AI_NUMERICHOST,
                }
            )
        return addresses

    async def close(self) -> None:
        await self._default.close()
