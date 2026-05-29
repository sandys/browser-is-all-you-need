"""Host-side fetch helper for ChromiumRL preview proxying.

ChromiumRL sometimes needs WootzApp to navigate through a local URL while a
host process fetches the real Daytona preview with bearer-style preview
headers. The browser calls ChromiumRL's /preview route; ChromiumRL then calls
this /fetch helper to retrieve the upstream HTML/assets.
"""

from __future__ import annotations

import base64
import urllib.error
import urllib.request
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field


class FetchRequest(BaseModel):
    upstream_url: str
    method: str = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: int = 60


app = FastAPI(title="W8 ChromiumRL Host Preview Proxy")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/fetch")
async def fetch(req: FetchRequest) -> dict[str, Any]:
    method = req.method.upper()
    if method not in {"GET", "HEAD"}:
        method = "GET"
    request = urllib.request.Request(
        req.upstream_url,
        headers={str(k): str(v) for k, v in req.headers.items()},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=max(1, req.timeout_s)) as response:
            body = b"" if method == "HEAD" else response.read()
            return {
                "status_code": int(response.status),
                "headers": dict(response.headers.items()),
                "body_b64": base64.b64encode(body).decode("ascii"),
            }
    except urllib.error.HTTPError as exc:
        body = b"" if method == "HEAD" else exc.read()
        return {
            "status_code": int(exc.code),
            "headers": dict(exc.headers.items()),
            "body_b64": base64.b64encode(body).decode("ascii"),
        }
