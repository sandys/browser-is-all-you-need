from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "convert_scalecua_to_osworld_toolcalls.py"
spec = importlib.util.spec_from_file_location("scalecua_converter", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_click_converts_to_osworld_tool_call():
    name, args, kwargs = converter.parse_action_call("click(x=0.5, y=0.25)")
    action = converter.convert_action(name, args, kwargs, 1920, 1080)

    assert action == {"action": "left_click", "coordinate": [960, 270]}
    assert converter.tool_call(action) == (
        '<tool_call>\n{"name": "computer_use","arguments": {"action": "left_click","coordinate": [960,270]}}\n</tool_call>'
    )


def test_converter_writes_accepted_and_rejected_rows(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    source = annotations / "data_windows_action_grounding.jsonl"
    rows = [
        {
            "image": "windows/example.png",
            "conversations": [
                {"from": "human", "value": "<image>\nClick the OK button"},
                {"from": "gpt", "value": "<action>\nclick(x=0.25, y=0.5)\n</action>"},
            ],
            "width": 1000,
            "height": 800,
        },
        {
            "image": "windows/example-2.png",
            "conversations": [
                {"from": "human", "value": "<image>\nOpen the app"},
                {"from": "gpt", "value": "<action>\nopen_app(name='Calendar')\n</action>"},
            ],
            "width": 1000,
            "height": 800,
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    rejects = tmp_path / "rejects.jsonl"

    args = converter.argparse.Namespace(
        annotations=annotations,
        limit=10,
        platforms=["windows", "ubuntu", "mac", "web"],
        out=out,
        rejects=rejects,
    )
    accepted, rejected = converter.convert(args)

    assert accepted == 1
    assert rejected == 1
    converted = json.loads(out.read_text(encoding="utf-8"))
    assistant = converted["messages"][1]["content"]
    assert "<tool_call>" in assistant
    assert "<action>" not in assistant
    assert '"coordinate": [250,400]' in assistant
    reject = json.loads(rejects.read_text(encoding="utf-8"))
    assert "open_app" in reject["reason"]
