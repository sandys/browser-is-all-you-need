"""Live DOMDiff evaluation for Harbor preview URLs."""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from w8_biayn.domdiff import trycloudflare_connect_to_args

from .rubric import RubricEvaluation, evaluate_harbor_rubric


def evaluate_preview_url(
    *,
    chromiumrl_url: str,
    task_dir: str | Path,
    preview_url: str,
    preview_headers: dict[str, str] | None = None,
    timeout_sec: int = 120,
) -> tuple[dict[str, Any], RubricEvaluation]:
    task_path = Path(task_dir)
    gold = _read_json(task_path / "gold_signals.json")
    rubric = _read_json(task_path / "rubric.json")
    reference_state = gold.get("dom_state", gold)
    eval_doc = _post_chromiumrl_evaluate(
        chromiumrl_url,
        {
            "target_url": preview_url,
            "timeout_sec": timeout_sec,
            "reference_state": reference_state,
            "extra_headers": {str(k): str(v) for k, v in (preview_headers or {}).items()},
        },
    )
    rubric_eval = evaluate_harbor_rubric(
        rubric,
        domdiff_summary=_domdiff_summary(eval_doc),
        dom_state=eval_doc.get("dom_state") if isinstance(eval_doc.get("dom_state"), dict) else {},
        signal_metrics=(
            eval_doc.get("signal_metrics")
            if isinstance(eval_doc.get("signal_metrics"), dict)
            else {}
        ),
    )
    return eval_doc, rubric_eval


def _post_chromiumrl_evaluate(chromiumrl_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("transport") or "async").strip().lower()
    if mode in {"sync", "direct"}:
        return _post_chromiumrl_evaluate_sync(chromiumrl_url, payload)
    try:
        return _post_chromiumrl_evaluate_async(chromiumrl_url, payload)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return _post_chromiumrl_evaluate_sync(chromiumrl_url, payload)
        raise


def _post_chromiumrl_evaluate_sync(chromiumrl_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = chromiumrl_url.rstrip("/") + "/evaluate"
    doc = _json_http_request(
        endpoint,
        method="POST",
        payload=payload,
        timeout=int(payload.get("timeout_sec", 120) or 120) + 10,
    )
    if not isinstance(doc, dict):
        raise RuntimeError("ChromiumRL /evaluate returned non-object JSON")
    return doc


def _post_chromiumrl_evaluate_async(chromiumrl_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = chromiumrl_url.rstrip("/")
    submit_doc = _json_http_request(
        base + "/evaluate_async",
        method="POST",
        payload=payload,
        timeout=30,
    )
    if not isinstance(submit_doc, dict) or not submit_doc.get("job_id"):
        raise RuntimeError("ChromiumRL /evaluate_async returned no job_id")
    job_id = str(submit_doc["job_id"])
    timeout_sec = int(payload.get("timeout_sec", 120) or 120)
    poll_timeout = max(timeout_sec + 300, 600)
    deadline = time.time() + poll_timeout
    status_endpoint = base + f"/evaluate_jobs/{urllib.parse.quote(job_id)}"
    while time.time() < deadline:
        status_doc = _json_http_request(status_endpoint, timeout=30)
        if not isinstance(status_doc, dict):
            raise RuntimeError("ChromiumRL evaluate job returned non-object JSON")
        status = str(status_doc.get("status") or "").lower()
        if status == "done":
            result = status_doc.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("ChromiumRL evaluate job completed without result")
            return result
        if status == "error":
            raise RuntimeError(str(status_doc.get("error") or "ChromiumRL evaluate job failed"))
        time.sleep(3)
    raise TimeoutError(f"ChromiumRL evaluate job {job_id} did not finish within {poll_timeout}s")


def _json_http_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError:
        raise
    except Exception as exc:
        connect_to_args = trycloudflare_connect_to_args(url)
        if not connect_to_args:
            raise
        body = _curl_json_request(
            url,
            method=method,
            payload=payload,
            timeout=timeout,
            connect_to_args=connect_to_args,
        )
    doc = json.loads(body)
    if not isinstance(doc, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return doc


def _curl_json_request(
    url: str,
    *,
    method: str,
    payload: dict[str, Any] | None,
    timeout: int,
    connect_to_args: list[str],
) -> str:
    cmd = [
        "curl",
        "-fsSL",
        "--retry",
        "6",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "10",
        "--max-time",
        str(timeout),
        *connect_to_args,
    ]
    input_text = None
    if payload is not None:
        cmd.extend(["-H", "Content-Type: application/json", "--data-binary", "@-"])
        input_text = json.dumps(payload)
    elif method != "GET":
        cmd.extend(["-X", method])
    cmd.append(url)
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout * 7 + 20,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl fallback failed for {url}: {result.stderr.strip()}")
    return result.stdout


def _domdiff_summary(doc: dict[str, Any]) -> dict[str, Any]:
    summary = doc.get("domdiff_summary")
    if isinstance(summary, dict):
        return summary
    return {
        "structuralSimilarity": doc.get("structuralSimilarity", doc.get("domdiff_structural", 0.0)),
        "textSimilarity": doc.get("textSimilarity", doc.get("domdiff_text", 0.0)),
        "layoutSimilarity": doc.get("layoutSimilarity", doc.get("domdiff_layout", 0.0)),
        "styleSimilarity": doc.get("styleSimilarity", doc.get("domdiff_style", 0.0)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise RuntimeError(f"Expected object JSON at {path}")
    return doc


def _is_dns_resolution_error(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, socket.gaierror):
        return True
    text = str(exc).lower()
    return (
        "name or service not known" in text
        or "temporary failure in name resolution" in text
        or "nodename nor servname provided" in text
        or "name resolution" in text
    )
