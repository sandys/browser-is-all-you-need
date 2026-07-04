#!/usr/bin/env python3
"""Deprecated shim: the GCP GLM launcher moved into the w8-biayn CLI.

Use `uv run --extra cloud w8-biayn launch glm47-full ...` instead. This file
delegates to `w8_biayn.cli` so existing invocations keep working.
"""

from __future__ import annotations

import sys


def main() -> int:
    sys.stderr.write(
        "launch_gcp_h100_full.py moved into the w8-biayn CLI; forwarding to "
        "`w8-biayn launch glm47-full`. Update your invocation.\n"
    )
    from typer.main import get_command

    from w8_biayn.cli import app

    command = get_command(app)
    return command.main(["launch", "glm47-full", *sys.argv[1:]], standalone_mode=False) or 0


if __name__ == "__main__":
    raise SystemExit(main())
