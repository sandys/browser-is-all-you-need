"""Resolve LoRA CLI flags for SLIME/Megatron lanes.

The pinned SLIME parser tolerates unknown Megatron arguments, so a LoRA lane
must verify the runtime actually advertises LoRA flags before appending them.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z][A-Za-z0-9-]*")

ENABLE_FLAGS = (
    "--enable-lora",
    "--use-lora",
    "--use-peft-lora",
    "--enable-peft-lora",
)
RANK_FLAGS = (
    "--lora-rank",
    "--lora-r",
    "--peft-lora-rank",
    "--peft-lora-r",
    "--adapter-rank",
)
ALPHA_FLAGS = (
    "--lora-alpha",
    "--peft-lora-alpha",
    "--adapter-alpha",
)
DROPOUT_FLAGS = (
    "--lora-dropout",
    "--peft-lora-dropout",
    "--adapter-dropout",
)
TARGET_FLAGS = (
    "--lora-target-modules",
    "--peft-lora-target-modules",
    "--adapter-target-modules",
)


@dataclass(frozen=True)
class LoraArgPlan:
    """Resolved LoRA arguments for a particular runtime help surface."""

    args: list[str]
    rank_flag: str
    enable_flag: str | None = None
    alpha_flag: str | None = None
    dropout_flag: str | None = None
    target_modules_flag: str | None = None


def available_flags(help_text: str) -> set[str]:
    """Extract long CLI flags from argparse/Megatron help text."""

    return set(FLAG_RE.findall(help_text))


def build_lora_args(
    help_text: str,
    *,
    rank: int = 16,
    alpha: str | None = None,
    dropout: str | None = None,
    target_modules: str | None = None,
) -> LoraArgPlan:
    """Build LoRA args only when the runtime exposes known LoRA flags."""

    flags = available_flags(help_text)
    rank_flag = _choose_flag(flags, RANK_FLAGS)
    if rank_flag is None:
        raise ValueError(
            "The active SLIME/Megatron runtime does not expose a supported LoRA rank flag. "
            f"Expected one of: {', '.join(RANK_FLAGS)}. Refresh or patch the SLIME/Megatron runtime "
            "before running the rank-16 LoRA lane."
        )

    args: list[str] = []
    enable_flag = _choose_flag(flags, ENABLE_FLAGS)
    if enable_flag is not None:
        args.append(enable_flag)

    args.extend([rank_flag, str(rank)])

    alpha_flag = _append_optional_value(args, flags, ALPHA_FLAGS, alpha, "alpha")
    dropout_flag = _append_optional_value(args, flags, DROPOUT_FLAGS, dropout, "dropout")
    target_modules_flag = _append_optional_value(args, flags, TARGET_FLAGS, target_modules, "target modules")

    return LoraArgPlan(
        args=args,
        rank_flag=rank_flag,
        enable_flag=enable_flag,
        alpha_flag=alpha_flag,
        dropout_flag=dropout_flag,
        target_modules_flag=target_modules_flag,
    )


def _choose_flag(flags: set[str], candidates: Sequence[str]) -> str | None:
    for flag in candidates:
        if flag in flags:
            return flag
    return None


def _append_optional_value(
    args: list[str],
    flags: set[str],
    candidates: Sequence[str],
    value: str | None,
    label: str,
) -> str | None:
    if value is None:
        return None
    flag = _choose_flag(flags, candidates)
    if flag is None:
        raise ValueError(
            f"SLIME_LORA_{label.upper().replace(' ', '_')} was set, but the active runtime does not expose "
            f"a supported LoRA {label} flag. Expected one of: {', '.join(candidates)}."
        )
    args.extend([flag, value])
    return flag


def _read_help_text(args: argparse.Namespace) -> str:
    if args.help_file:
        return Path(args.help_file).read_text(encoding="utf-8")
    if args.help_command:
        proc = subprocess.run(
            shlex.split(args.help_command),
            check=False,
            capture_output=True,
            text=True,
        )
        output = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
        if proc.returncode not in (0, 2):
            raise RuntimeError(
                f"LoRA help command failed with exit code {proc.returncode}: {args.help_command}\n{output}"
            )
        return output
    raise ValueError("one of --help-file or --help-command is required")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="Resolve LoRA args from runtime help text")
    resolve.add_argument("--help-file")
    resolve.add_argument("--help-command")
    resolve.add_argument("--rank", type=int, default=16)
    resolve.add_argument("--alpha")
    resolve.add_argument("--dropout")
    resolve.add_argument("--target-modules")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "resolve":
        help_text = _read_help_text(args)
        plan = build_lora_args(
            help_text,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=args.target_modules,
        )
        print(shlex.join(plan.args))
        return
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
