#!/usr/bin/env python3
"""Strict one-call OpenRouter transport for the fixed26 Luna Low evaluation.

The server accepts Aider's local OpenAI-compatible chat requests, rejects any
request that appears to contain hidden evaluation material, and performs
exactly one upstream HTTP request.  It never retries and never writes the API
key or Authorization header to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


EXPECTED_AIDER_MODEL = "gpt-5.6-luna"
OPENROUTER_MODEL = "openai/gpt-5.6-luna"
ALLOWED_RESPONSE_MODELS = {
    OPENROUTER_MODEL,
    "openai/gpt-5.6-luna-20260709",
}
EXPECTED_REASONING_EFFORT = os.environ.get("LUNA_EXPECTED_REASONING_EFFORT", "low")
EXPECTED_PROVIDER = "OpenAI"
MAX_OUTPUT_TOKENS = 32_768
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
ALLOW_AUTHORIZED_FEEDBACK = os.environ.get("LUNA_ALLOW_AUTHORIZED_FEEDBACK") == "1"
EXPECTED_TREE_SHA256 = os.environ.get("LUNA_EXPECTED_TREE_SHA256", "")
FORBIDDEN_REQUEST_KEYS = {
    "tools",
    "tool_choice",
    "plugins",
    "models",
    "provider",
    "web_search",
}
FORBIDDEN_FEEDBACK_MARKERS = (
    "See the testing errors above.",
    "The tests are correct, don't try and change them.",
    "FAILED:\n",
    "test cases:",
    "assertions:",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in {
                "text",
                "input_text",
            }:
                raise ValueError("non-text message content is forbidden")
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("text message part has no string text")
            parts.append(text)
        return "\n".join(parts)
    raise ValueError(f"unsupported message content type: {type(content).__name__}")


def normalize_excerpt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class HiddenCorpus:
    def __init__(self, practice_root: Path) -> None:
        self.practice_root = practice_root.resolve()
        records: list[dict[str, Any]] = []
        names: set[str] = set()
        full_texts: list[tuple[str, str]] = []
        excerpts: list[tuple[str, str]] = []

        # Public instructions and starter files may legitimately repeat API
        # names or boilerplate found in tests/examples.  Such overlap is not a
        # leak, so remove it from the forbidden-signature set.
        public_parts: list[str] = []
        for task_root in sorted(path for path in self.practice_root.iterdir() if path.is_dir()):
            config = json.loads((task_root / ".meta" / "config.json").read_text(encoding="utf-8"))
            public_paths = set(config.get("files", {}).get("solution", []))
            public_paths.update(
                path.relative_to(task_root).as_posix()
                for path in (task_root / ".docs").rglob("*")
                if path.is_file()
            )
            for relative in sorted(public_paths):
                path = task_root / relative
                if path.is_file():
                    public_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        normalized_public = normalize_excerpt("\n".join(public_parts))
        lower_public = normalized_public.lower()

        for task_root in sorted(path for path in self.practice_root.iterdir() if path.is_dir()):
            config = json.loads((task_root / ".meta" / "config.json").read_text(encoding="utf-8"))
            files = config.get("files", {})
            relative_paths = set(files.get("test", [])) | set(files.get("example", []))
            relative_paths.add(".meta/tests.toml")
            relative_paths.update(
                path.relative_to(task_root).as_posix()
                for path in (task_root / "test").rglob("*")
                if path.is_file()
            )
            for relative in sorted(relative_paths):
                path = task_root / relative
                if not path.is_file():
                    continue
                content = path.read_text(encoding="utf-8", errors="replace")
                identity = f"{task_root.name}/{relative}"
                records.append(
                    {
                        "path": identity,
                        "sha256": sha256_bytes(content.encode("utf-8")),
                        "size": len(content.encode("utf-8")),
                    }
                )
                basename = Path(relative).name
                if basename.lower() not in lower_public:
                    names.add(basename)
                normalized = normalize_excerpt(content)
                if len(normalized) >= 80 and normalized not in normalized_public:
                    full_texts.append((identity, normalized))
                lines = [normalize_excerpt(line) for line in content.splitlines()]
                lines = [line for line in lines if line]
                for index in range(max(0, len(lines) - 4)):
                    excerpt = " ".join(lines[index : index + 5])
                    if len(excerpt) >= 160 and excerpt not in normalized_public:
                        excerpts.append((f"{identity}:window-{index + 1}", excerpt))

        self.records = records
        self.names = sorted(names)
        self.full_texts = full_texts
        self.excerpts = excerpts
        self.sha256 = sha256_bytes(canonical_json_bytes(records))

    def audit(self, request: dict[str, Any]) -> dict[str, Any]:
        violations: list[str] = []
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            violations.append("messages must be a nonempty list")
            messages = []

        text_parts: list[str] = []
        roles: list[str] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                violations.append(f"messages[{index}] is not an object")
                continue
            role = message.get("role")
            roles.append(str(role))
            if role not in {"system", "user", "assistant"}:
                violations.append(f"forbidden message role {role!r}")
            try:
                text_parts.append(content_text(message.get("content")))
            except ValueError as exc:
                violations.append(f"messages[{index}]: {exc}")

        prompt = "\n".join(text_parts)
        normalized_prompt = normalize_excerpt(prompt)
        lower_prompt = prompt.lower()

        for key in sorted(FORBIDDEN_REQUEST_KEYS & set(request)):
            violations.append(f"forbidden request field {key}")
        if request.get("model") != EXPECTED_AIDER_MODEL:
            violations.append(f"unexpected Aider model {request.get('model')!r}")
        if request.get("reasoning_effort") != EXPECTED_REASONING_EFFORT:
            violations.append(
                f"reasoning effort is not exactly {EXPECTED_REASONING_EFFORT}"
            )
        if request.get("stream") not in {None, False}:
            violations.append("streaming is forbidden")

        for marker in FORBIDDEN_FEEDBACK_MARKERS:
            if marker.lower() in lower_prompt:
                violations.append(f"test-feedback marker present: {marker}")
        for name in self.names:
            if name.lower() in lower_prompt:
                violations.append(f"hidden filename present: {name}")
        for identity, full_text in self.full_texts:
            if full_text in normalized_prompt:
                violations.append(f"complete hidden file present: {identity}")
        for identity, excerpt in self.excerpts:
            if excerpt in normalized_prompt:
                violations.append(f"hidden five-line excerpt present: {identity}")

        return {
            "status": "passed" if not violations else "rejected",
            "violations": sorted(set(violations)),
            "hidden_corpus_sha256": self.sha256,
            "hidden_file_count": len(self.records),
            "hidden_filename_count": len(self.names),
            "hidden_excerpt_count": len(self.excerpts),
            "message_count": len(messages),
            "message_roles": roles,
            "prompt_chars": len(prompt),
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        }


class ProxyState:
    def __init__(
        self,
        *,
        artifact_dir: Path,
        practice_root: Path,
        feedback_receipts_root: Path | None,
        max_concurrency: int,
        timeout_seconds: int,
    ) -> None:
        self.artifact_dir = artifact_dir.resolve()
        self.calls_dir = self.artifact_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key.startswith("sk-or-"):
            raise RuntimeError("OPENROUTER_API_KEY is missing or malformed")
        self.hidden = HiddenCorpus(practice_root)
        self.feedback_receipts_root = (
            feedback_receipts_root.resolve() if feedback_receipts_root else None
        )
        if ALLOW_AUTHORIZED_FEEDBACK and self.feedback_receipts_root is None:
            raise RuntimeError(
                "authorized feedback requires --feedback-receipts-root"
            )
        self.semaphore = threading.BoundedSemaphore(max_concurrency)
        self.timeout_seconds = timeout_seconds
        self.counter_lock = threading.Lock()
        self.counter = len([path for path in self.calls_dir.glob("call-*") if path.is_dir()])

    def next_call_id(self, request_sha256: str) -> str:
        with self.counter_lock:
            self.counter += 1
            number = self.counter
        return f"call-{number:06d}-{request_sha256[:12]}"

    def authorize_feedback(
        self, request: dict[str, Any], call_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Bind turn-2 feedback to the exact failed candidate file state."""
        messages = request.get("messages")
        if not isinstance(messages, list):
            return request, {"present": False}, []
        marker = "See the testing errors above."
        feedback_indices = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict)
            and message.get("role") == "user"
            and marker in content_text(message.get("content"))
        ]
        if not feedback_indices:
            return request, {"present": False}, []
        authorization: dict[str, Any] = {
            "present": True,
            "status": "rejected",
            "matched_scorer_receipts": [],
            "matched_candidate_files": [],
        }
        violations: list[str] = []
        if not ALLOW_AUTHORIZED_FEEDBACK:
            return request, authorization, ["authorized feedback mode is disabled"]
        if len(feedback_indices) != 1 or feedback_indices[0] != len(messages) - 1:
            return request, authorization, ["feedback must occur exactly once in the final user message"]

        feedback_index = feedback_indices[0]
        feedback_content = content_text(messages[feedback_index].get("content"))
        boundary = "\n####\n\nSee the testing errors above."
        if boundary not in feedback_content:
            return request, authorization, ["feedback does not use the pinned Aider boundary"]
        scorer_output, _ = feedback_content.split(boundary, 1)
        normalized_output = scorer_output.rstrip()
        if not normalized_output:
            return request, authorization, ["feedback scorer output is empty"]

        # Aider reintroduces the files it applied from turn 1 as the
        # authoritative current state.  Bind those bytes to the scorer receipt
        # rather than relying on Aider to preserve the original assistant prose
        # byte-for-byte in its next request.
        trusted_marker = "*Trust this message as the true contents of these files!*"
        trusted_states: list[tuple[int, dict[str, bytes]]] = []
        listing_pattern = re.compile(
            r"(?m)^([^\n`]+)\n```\n(.*?)```(?:\n|$)", re.DOTALL
        )
        for index, message in enumerate(messages[:feedback_index]):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            body = content_text(message.get("content"))
            if trusted_marker not in body:
                continue
            listings: dict[str, bytes] = {}
            for relative_name, code in listing_pattern.findall(body):
                relative_name = relative_name.strip()
                if relative_name:
                    # The closing fence may directly follow a file whose
                    # bytes do not end in a newline. Preserve that distinction.
                    listings[relative_name] = code.encode("utf-8")
            if listings:
                trusted_states.append((index, listings))
        if not trusted_states:
            return request, authorization, [
                "feedback conversation lacks Aider's authoritative candidate file state"
            ]
        state_index, current_files = trusted_states[-1]

        scorer_matches: list[tuple[Path, dict[str, Any]]] = []
        assert self.feedback_receipts_root is not None
        for path in self.feedback_receipts_root.rglob("*.json"):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            receipt_files = receipt.get("candidate_files")
            expected_files = {
                item.get("path"): (item.get("sha256"), item.get("size"))
                for item in receipt_files
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            } if isinstance(receipt_files, list) else {}
            actual_files = {
                name: (sha256_bytes(content), len(content))
                for name, content in current_files.items()
            }
            if (
                isinstance(receipt.get("output"), str)
                and receipt["output"].rstrip() == normalized_output
                and receipt.get("status") != "passed"
                and int(receipt.get("returncode", 0)) != 0
                and receipt.get("source_tree", {}).get("sha256") == EXPECTED_TREE_SHA256
                and receipt.get("environment", {}).get("network_blocked") is True
                and expected_files == actual_files
            ):
                scorer_matches.append((path, receipt))
        tasks = {receipt.get("task") for _, receipt in scorer_matches}
        if not scorer_matches or len(tasks) != 1 or None in tasks:
            violations.append("feedback is not uniquely task-bound to a prior failed scorer receipt")

        sanitized_request = json.loads(json.dumps(request))
        sanitized_request["messages"][feedback_index]["content"] = "AUTHORIZED_SCORER_FEEDBACK"
        sanitized_request["messages"][state_index]["content"] = "AUTHORIZED_CANDIDATE_FILE_STATE"
        for index, message in enumerate(sanitized_request["messages"][:feedback_index]):
            if message.get("role") == "assistant":
                # Candidate-generated text is not harness-provided hidden material.
                message["content"] = "AUTHORIZED_PRIOR_ASSISTANT_CONTENT"

        authorization.update(
            {
                "status": "authorized" if not violations else "rejected",
                "task": next(iter(tasks)) if len(tasks) == 1 else None,
                "feedback_output_sha256": sha256_bytes(normalized_output.encode("utf-8")),
                "matched_scorer_receipts": [
                    {
                        "path": str(path.relative_to(self.feedback_receipts_root)),
                        "sha256": sha256_bytes(path.read_bytes()),
                        "request_id": receipt.get("request_id"),
                    }
                    for path, receipt in scorer_matches
                ],
                "matched_candidate_files": [
                    {
                        "path": name,
                        "sha256": sha256_bytes(content),
                        "size": len(content),
                    }
                    for name, content in sorted(current_files.items())
                ],
                "candidate_state_message_index": state_index,
            }
        )
        return sanitized_request, authorization, violations

    def complete(self, request: dict[str, Any]) -> tuple[str, dict[str, int], str]:
        request_sha256 = sha256_bytes(canonical_json_bytes(request))
        call_id = self.next_call_id(request_sha256)
        call_dir = self.calls_dir / call_id
        call_dir.mkdir()
        write_json(call_dir / "aider_request.json", request)
        started = time.time()
        audit_request, feedback_authorization, feedback_violations = self.authorize_feedback(
            request, call_id
        )
        audit = self.hidden.audit(audit_request)
        audit["feedback_authorization"] = feedback_authorization
        response_text = ""
        response_payload: dict[str, Any] = {}
        response_headers: dict[str, str] = {}
        http_status = 0
        violations = list(audit["violations"]) + feedback_violations
        upstream_attempts = 0

        upstream_body = {
            "model": OPENROUTER_MODEL,
            "messages": request.get("messages", []),
            "reasoning": {"effort": EXPECTED_REASONING_EFFORT, "exclude": True},
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
            "provider": {
                "only": ["openai"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            },
        }
        write_json(call_dir / "sanitized_upstream_request.json", upstream_body)

        if not violations:
            upstream_request = urllib.request.Request(
                OPENROUTER_URL,
                data=canonical_json_bytes(upstream_body),
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "fixed26-strict-one-shot-eval",
                },
            )
            try:
                with self.semaphore:
                    upstream_attempts = 1
                    with urllib.request.urlopen(
                        upstream_request, timeout=self.timeout_seconds
                    ) as response:
                        http_status = response.status
                        response_headers = {
                            key.lower(): value
                            for key, value in response.headers.items()
                            if key.lower().startswith("x-openrouter")
                            or key.lower() in {"content-type", "date"}
                        }
                        raw = response.read()
                response_payload = json.loads(raw)
            except urllib.error.HTTPError as exc:
                http_status = exc.code
                body = exc.read().decode("utf-8", errors="replace")
                try:
                    response_payload = json.loads(body)
                except json.JSONDecodeError:
                    response_payload = {"error": {"message": body[:2000]}}
                violations.append(f"OpenRouter HTTP {exc.code}")
            except Exception as exc:
                violations.append(f"OpenRouter transport error: {type(exc).__name__}: {exc}")

        actual_model = response_payload.get("model")
        actual_provider = response_payload.get("provider")
        if not violations:
            if actual_model not in ALLOWED_RESPONSE_MODELS:
                violations.append(f"unexpected response model {actual_model!r}")
            if actual_provider != EXPECTED_PROVIDER:
                violations.append(f"unexpected provider {actual_provider!r}")
            choices = response_payload.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                violations.append("response does not contain exactly one choice")
            else:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    response_text = content
                else:
                    violations.append("response content is not text")
            if not response_text.strip():
                violations.append("empty candidate response")

        usage = response_payload.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        safe_response = dict(response_payload)
        safe_response.pop("choices", None)
        write_json(call_dir / "upstream_metadata.json", safe_response)
        (call_dir / "response.txt").write_text(response_text, encoding="utf-8")
        receipt = {
            "schema_version": 1,
            "kind": "luna-low-openrouter-strict-one-shot-call",
            "call_id": call_id,
            "status": "accepted" if not violations else "rejected",
            "request_sha256": request_sha256,
            "response_sha256": sha256_bytes(response_text.encode("utf-8")),
            "model_requested": OPENROUTER_MODEL,
            "model_returned": actual_model,
            "provider_returned": actual_provider,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "reasoning_excluded_from_response": True,
            "tools_present": False,
            "plugins_present": False,
            "web_search_enabled": False,
            "provider_fallbacks_allowed": False,
            "provider_requires_parameters": True,
            "provider_data_collection": "deny",
            "provider_zdr_required": False,
            "upstream_http_attempts": upstream_attempts,
            "http_status": http_status,
            "response_headers": response_headers,
            "usage": usage,
            "leak_audit": audit,
            "feedback_authorization": feedback_authorization,
            "violations": sorted(set(violations)),
            "duration_seconds": round(time.time() - started, 6),
        }
        write_json(call_dir / "receipt.json", receipt)

        if violations:
            response_text = "EVALUATION_TRANSPORT_REJECTED\n"
        normalized_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        return response_text, normalized_usage, call_id


class Handler(BaseHTTPRequestHandler):
    server_version = "LunaLowOpenRouterStrictProxy/1"

    @property
    def state(self) -> ProxyState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"", "/health"}:
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": EXPECTED_AIDER_MODEL,
                    "openrouter_model": OPENROUTER_MODEL,
                    "reasoning_effort": EXPECTED_REASONING_EFFORT,
                    "provider": EXPECTED_PROVIDER,
                    "calls": self.state.counter,
                    "hidden_corpus_sha256": self.state.hidden.sha256,
                },
            )
            return
        if self.path.rstrip("/").endswith("/models"):
            self.send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": EXPECTED_AIDER_MODEL,
                            "object": "model",
                            "owned_by": "openrouter-strict-transport",
                        }
                    ],
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            content, usage, call_id = self.state.complete(request)
            self.send_json(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-{call_id}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": EXPECTED_AIDER_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )
        except Exception as exc:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"type": type(exc).__name__, "message": str(exc)}},
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--practice-root", type=Path, required=True)
    parser.add_argument("--feedback-receipts-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    state = ProxyState(
        artifact_dir=args.artifact_dir,
        practice_root=args.practice_root,
        feedback_receipts_root=args.feedback_receipts_root,
        max_concurrency=args.max_concurrency,
        timeout_seconds=args.timeout_seconds,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.state = state  # type: ignore[attr-defined]
    write_json(
        args.artifact_dir / "proxy_receipt.json",
        {
            "schema_version": 1,
            "kind": "luna-low-openrouter-strict-one-shot-proxy",
            "status": "running",
            "host": args.host,
            "port": server.server_address[1],
            "model": EXPECTED_AIDER_MODEL,
            "openrouter_model": OPENROUTER_MODEL,
            "reasoning_effort": EXPECTED_REASONING_EFFORT,
            "provider": EXPECTED_PROVIDER,
            "identity_gate": {
                "expected_model": EXPECTED_AIDER_MODEL,
                "expected_reasoning_effort": EXPECTED_REASONING_EFFORT,
                "passed": True,
            },
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "tools": "omitted",
            "plugins": "omitted",
            "web_search": "not enabled",
            "provider_fallbacks": False,
            "provider_data_collection": "deny",
            "provider_zdr_required": False,
            "upstream_retries": 0,
            "authorized_feedback_enabled": ALLOW_AUTHORIZED_FEEDBACK,
            "feedback_receipts_root": (
                str(state.feedback_receipts_root) if state.feedback_receipts_root else None
            ),
            "hidden_corpus_sha256": state.hidden.sha256,
            "hidden_file_count": len(state.hidden.records),
            "max_concurrency": args.max_concurrency,
            "timeout_seconds": args.timeout_seconds,
            "pid": os.getpid(),
            "started_at_unix": time.time(),
        },
    )
    print(
        f"listening http://{args.host}:{server.server_address[1]} "
        f"model={OPENROUTER_MODEL} effort={EXPECTED_REASONING_EFFORT}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
