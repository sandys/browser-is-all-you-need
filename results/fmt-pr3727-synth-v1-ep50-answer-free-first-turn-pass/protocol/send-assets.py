#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess

from huggingface_hub import get_token

HOST = "pipeshift@216.81.245.237"
KEY = "/Users/tokenbender/.ssh/wootzapp-training"
REMOTE_RECEIVER = "/tmp/receive-answer-free-assets.py"


def main() -> None:
    token = get_token()
    if not token:
        raise SystemExit("local Hugging Face token is unavailable")
    request = {
        "token": token,
        "base_destination": "/home/pipeshift/models/zai-org--GLM-4.7-Flash",
        "adapter_destination": "/home/pipeshift/checkpoints/glm47-synth-v1-100ep-a8d386d6b5bedf761cbe4effe2fdc36a97ad62bc",
    }
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-i",
        KEY,
        HOST,
        f"python3 {REMOTE_RECEIVER}",
    ]
    completed = subprocess.run(
        command,
        input=json.dumps(request),
        text=True,
        check=True,
        capture_output=True,
    )
    print(completed.stdout, end="")


if __name__ == "__main__":
    main()
