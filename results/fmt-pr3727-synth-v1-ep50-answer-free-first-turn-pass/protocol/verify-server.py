#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

LORA_NAME = "glm-4.7-flash-grpo"


def post(url: str, payload: dict[str, object], timeout: int = 600) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local-eval"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc


def completion(port: int, use_lora: bool) -> dict[str, object]:
    body: dict[str, object] = {
        "model": LORA_NAME,
        "messages": [{
            "role": "user",
            "content": "Write a C++17 function `int answer()` that returns 42. Reply with only the code.",
        }],
        "temperature": 0.0,
        "max_tokens": 48,
        "logprobs": True,
    }
    if use_lora:
        body["lora_path"] = LORA_NAME
    payload = post(f"http://127.0.0.1:{port}/v1/chat/completions", body)
    choice = payload["choices"][0]
    logprobs = [
        round(entry["logprob"], 6)
        for entry in (choice.get("logprobs") or {}).get("content") or []
    ]
    return {"content": choice["message"]["content"], "logprobs": logprobs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--adapter-path", default="/adapter")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{args.port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(5)
    else:
        raise SystemExit("SGLang did not become healthy within 30 minutes")

    loaded = None
    load_payload = {"lora_name": LORA_NAME, "lora_path": args.adapter_path}
    for endpoint in ("/load_lora_adapter", "/v1/load_lora_adapter"):
        try:
            loaded = post(f"http://127.0.0.1:{args.port}{endpoint}", load_payload)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    if loaded is None:
        raise SystemExit("SGLang exposes no LoRA loading endpoint")

    with_lora = completion(args.port, True)
    without_lora = completion(args.port, False)
    if with_lora == without_lora:
        raise SystemExit("LoRA activation probe did not diverge from base")
    receipt = {
        "schema_version": 1,
        "status": "diverged",
        "load_response": loaded,
        "with_lora": with_lora,
        "without_lora": without_lora,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
