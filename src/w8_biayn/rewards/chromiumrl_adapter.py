"""Small ChromiumRL/DOMDiff reward HTTP adapter.

This module is copied into the prebuilt Android DOMDiff container at runtime.
It intentionally contains only reward-service code; the browser, AndroidWorld
source, WootzApp APK, and Docker image recipe stay outside this repository.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from websocket import WebSocket, create_connection


def _cdp_url() -> str:
    return os.environ.get("CDP_URL", "ws://127.0.0.1:9224").rstrip("/")


def _http_url(cdp_url: str) -> str:
    if cdp_url.startswith("wss://"):
        return "https://" + cdp_url[len("wss://") :]
    if cdp_url.startswith("ws://"):
        return "http://" + cdp_url[len("ws://") :]
    return cdp_url


def _rewrite_ws_url(cdp_url: str, discovered: str) -> str:
    configured = urllib.parse.urlsplit(cdp_url)
    found = urllib.parse.urlsplit(discovered)
    scheme = "wss" if configured.scheme in {"wss", "https"} else "ws"
    netloc = configured.netloc or found.netloc
    return urllib.parse.urlunsplit((scheme, netloc, found.path, found.query, ""))


def _host_header(cdp_url: str) -> list[str]:
    parsed = urllib.parse.urlsplit(cdp_url)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return ["Host: localhost"]
    return []


def _json_request(url: str, *, timeout: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Host": "localhost"} if "127.0.0.1" in url else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected object response from {url}")
    return data


class CdpSession:
    """Minimal synchronous CDP client for one isolated page target."""

    def __init__(self, cdp_url: str | None = None) -> None:
        self.cdp_url = (cdp_url or _cdp_url()).rstrip("/")
        self.http_url = _http_url(self.cdp_url)
        self.ws: WebSocket | None = None
        self.msg_id = 0
        self.session_id = ""
        self.target_id = ""

    def __enter__(self) -> "CdpSession":
        version = _json_request(f"{self.http_url}/json/version", timeout=30)
        browser_ws = _rewrite_ws_url(self.cdp_url, str(version["webSocketDebuggerUrl"]))
        self.ws = create_connection(
            browser_ws,
            timeout=30,
            header=_host_header(self.cdp_url),
            suppress_origin=True,
        )
        self.target_id = self.send("Target.createTarget", {"url": "about:blank"})["targetId"]
        attached = self.send(
            "Target.attachToTarget",
            {"targetId": self.target_id, "flatten": True},
            route_to_session=False,
        )
        self.session_id = str(attached["sessionId"])
        for method in ("Page.enable", "Runtime.enable", "DOM.enable", "Network.enable"):
            try:
                self.send(method)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self.target_id:
                self.send(
                    "Target.closeTarget",
                    {"targetId": self.target_id},
                    route_to_session=False,
                    timeout=5,
                )
        except Exception:
            pass
        if self.ws is not None:
            self.ws.close()

    def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        route_to_session: bool = True,
        timeout: int = 30,
    ) -> dict[str, Any]:
        if self.ws is None:
            raise RuntimeError("CDP websocket is not connected")
        self.msg_id += 1
        msg_id = self.msg_id
        message: dict[str, Any] = {"id": msg_id, "method": method}
        if params is not None:
            message["params"] = params
        if route_to_session and self.session_id:
            message["sessionId"] = self.session_id

        self.ws.settimeout(timeout)
        self.ws.send(json.dumps(message))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.ws.recv()
            response = json.loads(raw)
            if response.get("id") != msg_id:
                continue
            if "error" in response:
                raise RuntimeError(f"CDP error in {method}: {response['error']}")
            result = response.get("result", {})
            return result if isinstance(result, dict) else {}
        raise TimeoutError(f"Timed out waiting for {method}")

    def wait_ready_state(self, timeout: int) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                result = self.send(
                    "Runtime.evaluate",
                    {"expression": "document.readyState", "returnByValue": True},
                    timeout=5,
                )
                value = ((result.get("result") or {}).get("value") or "").lower()
                if value in {"interactive", "complete"}:
                    return
            except Exception:
                pass
            time.sleep(1.0)


class EvaluateRequest(BaseModel):
    target_url: str
    timeout_sec: int = 120
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
    eval_mode: str = "shared_tab"
    error: str = ""


class EvaluateJobSubmitResponse(BaseModel):
    job_id: str
    status: str


class EvaluateJobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str = ""


@dataclass
class Job:
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str = ""


app = FastAPI(title="w8-biayn ChromiumRL DOMDiff Reward Adapter")
_jobs: dict[str, Job] = {}
_work_queue: "queue.Queue[tuple[str, EvaluateRequest]]" = queue.Queue()


def _score_from_comparison(comparison: dict[str, Any]) -> tuple[float, dict[str, float]]:
    summary_raw = comparison.get("summary") if isinstance(comparison, dict) else {}
    summary = {
        key: float((summary_raw or {}).get(key, 0.0) or 0.0)
        for key in (
            "structuralSimilarity",
            "textSimilarity",
            "layoutSimilarity",
            "styleSimilarity",
        )
    }
    values = [value for value in summary.values() if value > 0.0]
    if not values:
        return 0.0, summary
    return max(0.0, min(1.0, sum(values) / len(values))), summary


def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    request_id = uuid4().hex[:12]
    try:
        with CdpSession() as cdp:
            if req.extra_headers:
                cdp.send(
                    "Network.setExtraHTTPHeaders",
                    {"headers": {str(k): str(v) for k, v in req.extra_headers.items()}},
                    timeout=10,
                )
            cdp.send("Page.navigate", {"url": req.target_url}, timeout=min(req.timeout_sec, 60))
            cdp.wait_ready_state(timeout=min(req.timeout_sec, 120))
            time.sleep(2.0)
            dom_result = cdp.send("ChromiumRL.saveDOMState", timeout=min(req.timeout_sec, 90))
            dom_state = dom_result.get("state", dom_result)
            if not isinstance(dom_state, dict):
                dom_state = {}
            comparison: dict[str, Any] = {}
            if req.reference_state:
                result = cdp.send(
                    "ChromiumRL.compareDOMState",
                    {"referenceState": req.reference_state},
                    timeout=min(req.timeout_sec, 120),
                )
                comparison = result.get("result", result)
                if not isinstance(comparison, dict):
                    comparison = {}
            quality, summary = _score_from_comparison(comparison)
            if not req.reference_state and dom_state:
                quality = 1.0
            return EvaluateResponse(
                quality=quality,
                domdiff_total=quality,
                domdiff_summary=summary,
                dom_state=dom_state,
                signal_metrics={},
                signal_count=0,
                timeout_count=0,
                compositor_layers=0,
                request_id=request_id,
                eval_mode="shared_tab",
            ).model_dump()
    except Exception as exc:
        return EvaluateResponse(request_id=request_id, error=f"{type(exc).__name__}: {exc}").model_dump()


def _worker() -> None:
    while True:
        job_id, req = _work_queue.get()
        job = _jobs[job_id]
        job.status = "running"
        try:
            result = evaluate(req)
            if result.get("error"):
                job.status = "error"
                job.error = str(result["error"])
            else:
                job.status = "done"
                job.result = result
        except Exception as exc:
            job.status = "error"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            _work_queue.task_done()


threading.Thread(target=_worker, name="chromiumrl-evaluate-worker", daemon=True).start()


@app.get("/health")
def health() -> dict[str, Any]:
    cdp = _cdp_url()
    try:
        version = _json_request(f"{_http_url(cdp)}/json/version", timeout=10)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "cdp_url": cdp,
        "browser": version.get("Browser", ""),
        "webSocketDebuggerUrl": bool(version.get("webSocketDebuggerUrl")),
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in _jobs.values():
        counts[job.status] = counts.get(job.status, 0) + 1
    return {"jobs": counts, "queued": _work_queue.qsize(), "cdp_url": _cdp_url()}


@app.post("/evaluate")
def evaluate_sync(req: EvaluateRequest) -> dict[str, Any]:
    result = evaluate(req)
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.post("/evaluate_async", response_model=EvaluateJobSubmitResponse)
def evaluate_async(req: EvaluateRequest) -> EvaluateJobSubmitResponse:
    job_id = uuid4().hex
    _jobs[job_id] = Job()
    _work_queue.put((job_id, req))
    return EvaluateJobSubmitResponse(job_id=job_id, status="queued")


@app.get("/evaluate_jobs/{job_id}", response_model=EvaluateJobStatusResponse)
def evaluate_job(job_id: str) -> EvaluateJobStatusResponse:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return EvaluateJobStatusResponse(
        job_id=job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )
