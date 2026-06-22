#!/usr/bin/env python3
"""Convert ScaleCUA action annotations into OSWorld Qwen tool-call SFT rows."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)
CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
REJECT_ACTIONS = {"response", "open_app", "long_press", "tripleclick"}


def iter_annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def platform_for_record(record: dict[str, Any], source: Path) -> str:
    haystack = f"{source.name} {record.get('image', '')}".lower()
    for platform in ("windows", "ubuntu", "mac", "web", "android", "iphone"):
        if platform in haystack:
            return platform
    return "unknown"


def human_instruction(record: dict[str, Any]) -> str:
    for message in record.get("conversations") or []:
        if message.get("from") == "human" and isinstance(message.get("value"), str):
            return message["value"].replace("<image>", "").strip()
    return ""


def assistant_text(record: dict[str, Any]) -> str:
    parts = []
    for message in record.get("conversations") or []:
        if message.get("from") in {"gpt", "assistant"} and isinstance(message.get("value"), str):
            parts.append(message["value"])
    return "\n".join(parts)


def literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal_value(item) for item in node.elts]
    raise ValueError("unsupported argument expression")


def parse_action_call(action: str) -> tuple[str, list[Any], dict[str, Any]]:
    first_line = action.strip().splitlines()[0].strip()
    match = CALL_RE.match(first_line)
    if not match:
        raise ValueError("malformed action call")
    name, args_src = match.groups()
    expr = ast.parse(f"_f({args_src})", mode="eval").body
    if not isinstance(expr, ast.Call):
        raise ValueError("malformed action arguments")
    args = [literal_value(arg) for arg in expr.args]
    kwargs = {kw.arg: literal_value(kw.value) for kw in expr.keywords if kw.arg is not None}
    return name.lower(), args, kwargs


def number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise ValueError("expected number")


def coord_from_args(args: list[Any], kwargs: dict[str, Any], width: int, height: int) -> list[int]:
    if "x" in kwargs and "y" in kwargs:
        x, y = number(kwargs["x"]), number(kwargs["y"])
    elif "coordinate" in kwargs and isinstance(kwargs["coordinate"], list) and len(kwargs["coordinate"]) >= 2:
        x, y = number(kwargs["coordinate"][0]), number(kwargs["coordinate"][1])
    elif len(args) >= 2:
        x, y = number(args[0]), number(args[1])
    else:
        raise ValueError("missing coordinate")
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return [round(x * width), round(y * height)]
    return [round(x), round(y)]


def string_arg(args: list[Any], kwargs: dict[str, Any], *names: str) -> str:
    for name in names:
        if name in kwargs:
            return str(kwargs[name])
    if args:
        return str(args[0])
    raise ValueError("missing string argument")


def list_arg(args: list[Any], kwargs: dict[str, Any], *names: str) -> list[str]:
    for name in names:
        if name in kwargs:
            value = kwargs[name]
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]
    return [str(item) for item in args]


def scale_scroll(value: float) -> int:
    if -1.0 <= value <= 1.0:
        return int(round(value * 500))
    return int(round(value))


def convert_action(name: str, args: list[Any], kwargs: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    if name in REJECT_ACTIONS:
        raise ValueError(f"unsupported action {name}")
    if name in {"click", "left_click", "tap"}:
        return {"action": "left_click", "coordinate": coord_from_args(args, kwargs, width, height)}
    if name in {"rightclick", "right_click"}:
        return {"action": "right_click", "coordinate": coord_from_args(args, kwargs, width, height)}
    if name in {"doubleclick", "double_click"}:
        return {"action": "double_click", "coordinate": coord_from_args(args, kwargs, width, height)}
    if name in {"moveto", "mouse_move"}:
        return {"action": "mouse_move", "coordinate": coord_from_args(args, kwargs, width, height)}
    if name in {"dragto", "drag", "left_click_drag"}:
        converted = {"action": "left_click_drag", "coordinate": coord_from_args(args, kwargs, width, height)}
        if "duration" in kwargs:
            converted["duration"] = number(kwargs["duration"])
        return converted
    if name in {"write", "type", "input", "text"}:
        return {"action": "type", "text": string_arg(args, kwargs, "text", "content")}
    if name in {"press", "key", "hotkey", "keydown", "keyup"}:
        keys = list_arg(args, kwargs, "key", "keys")
        return {"action": "key", "keys": keys}
    if name in {"scroll", "swipe"}:
        value = kwargs.get("pixels", kwargs.get("amount", kwargs.get("dy", args[0] if args else 0)))
        return {"action": "scroll", "pixels": scale_scroll(number(value))}
    if name == "wait":
        return {"action": "wait"}
    if name in {"terminate", "done", "success"}:
        return {"action": "terminate", "status": "success"}
    if name in {"failure", "fail"}:
        return {"action": "terminate", "status": "failure"}
    if name == "navigate_home":
        return {"action": "key", "keys": ["home"]}
    if name == "navigate_back":
        return {"action": "key", "keys": ["alt", "left"]}
    raise ValueError(f"unsupported action {name}")


def tool_call(arguments: dict[str, Any]) -> str:
    payload = {"name": "computer_use", "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=True, separators=(",", ": ")) + "\n</tool_call>"


def output_row(record: dict[str, Any], source: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    instruction = human_instruction(record)
    user_prompt = (
        "<image>\n"
        "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        "Previous actions:\n"
        "None"
    )
    return {
        "image": record.get("image"),
        "width": record.get("width"),
        "height": record.get("height"),
        "source": source.name,
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": tool_call(arguments)},
        ],
    }


def reject_row(record: dict[str, Any], source: Path, line_no: int, reason: str, action: str | None) -> dict[str, Any]:
    return {
        "source": source.name,
        "line": line_no,
        "image": record.get("image"),
        "reason": reason,
        "action": action,
    }


def convert(args: argparse.Namespace) -> tuple[int, int]:
    platforms = {platform.lower() for platform in args.platforms}
    accepted = 0
    rejected = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.rejects.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f, args.rejects.open("w", encoding="utf-8") as reject_f:
        for path in iter_annotation_files(args.annotations):
            with path.open(encoding="utf-8") as in_f:
                for line_no, line in enumerate(in_f, 1):
                    if accepted >= args.limit:
                        return accepted, rejected
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        reject_f.write(json.dumps({"source": path.name, "line": line_no, "reason": str(exc)}) + "\n")
                        rejected += 1
                        continue
                    platform = platform_for_record(record, path)
                    if platforms and platform not in platforms:
                        continue
                    actions = ACTION_RE.findall(assistant_text(record))
                    if not actions:
                        continue
                    if len(actions) != 1:
                        reject_f.write(json.dumps(reject_row(record, path, line_no, "expected exactly one action", None)) + "\n")
                        rejected += 1
                        continue
                    action = actions[0].strip()
                    try:
                        name, parsed_args, kwargs = parse_action_call(action)
                        arguments = convert_action(name, parsed_args, kwargs, int(record["width"]), int(record["height"]))
                    except Exception as exc:
                        reject_f.write(json.dumps(reject_row(record, path, line_no, str(exc), action), ensure_ascii=True) + "\n")
                        rejected += 1
                        continue
                    out_f.write(json.dumps(output_row(record, path, arguments), ensure_ascii=True) + "\n")
                    accepted += 1
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="ScaleCUA annotation JSONL file or directory.")
    parser.add_argument("--limit", type=int, required=True, help="Maximum accepted rows to write.")
    parser.add_argument("--platforms", nargs="*", default=[], help="Platform substrings to include.")
    parser.add_argument("--out", type=Path, required=True, help="Output converted JSONL path.")
    parser.add_argument("--rejects", type=Path, required=True, help="Output rejected JSONL path.")
    return parser.parse_args()


def main() -> None:
    accepted, rejected = convert(parse_args())
    print(f"accepted={accepted}")
    print(f"rejected={rejected}")


if __name__ == "__main__":
    main()
