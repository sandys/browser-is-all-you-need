from __future__ import annotations

import json

import pytest

from w8_biayn.scalecua_sft import load_jsonl, mask_prompt_tokens, qwen_messages_from_row


def row() -> dict[str, object]:
    return {
        "image": "ubuntu/step_1.png",
        "messages": [
            {
                "role": "user",
                "content": "<image>\nPlease generate the next move.\n\nInstruction: Click OK",
            },
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name":"computer_use","arguments":{"action":"left_click","coordinate":[1,2]}}\n</tool_call>',
            },
        ],
    }


def test_qwen_messages_from_converted_row():
    messages = qwen_messages_from_row(row())

    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0] == {"type": "image", "image": "ubuntu/step_1.png"}
    assert messages[0]["content"][1]["type"] == "text"
    assert "<image>" not in messages[0]["content"][1]["text"]
    assert messages[1]["role"] == "assistant"
    assert "<tool_call>" in messages[1]["content"]


def test_qwen_messages_requires_tool_call():
    bad = row()
    bad["messages"][1]["content"] = "click(1, 2)"

    with pytest.raises(ValueError, match="tool_call"):
        qwen_messages_from_row(bad)


def test_load_jsonl_limit(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row()) for _ in range(3)) + "\n", encoding="utf-8")

    assert len(load_jsonl(path, limit=2)) == 2


def test_mask_prompt_tokens_masks_prompt_and_padding():
    class FakeLabels:
        def __init__(self) -> None:
            self.values = [
                [10, 11, 12, 0],
                [20, 21, 0, 0],
            ]

        def __getitem__(self, item):
            row_idx, col = item
            if isinstance(col, slice):
                return RowSlice(self.values[row_idx], col)
            return self.values[row_idx][col]

        def __setitem__(self, item, value):
            if isinstance(item, Mask):
                for row_idx, row_values in enumerate(self.values):
                    for col_idx, should_mask in enumerate(item.values[row_idx]):
                        if should_mask:
                            row_values[col_idx] = value
                return
            row_idx, col = item
            if isinstance(col, slice):
                start, stop, step = col.indices(len(self.values[row_idx]))
                for index in range(start, stop, step):
                    self.values[row_idx][index] = value
                return
            self.values[row_idx][col] = value

        def __eq__(self, other):
            return Mask([[value == other for value in row_values] for row_values in self.values])

    class RowSlice:
        def __init__(self, row_values, col) -> None:
            self.row_values = row_values
            self.col = col

        def __setitem__(self, item, value):
            start, stop, step = self.col.indices(len(self.row_values))
            for index in range(start, stop, step):
                self.row_values[index] = value

    class Mask:
        def __init__(self, values) -> None:
            self.values = values

    labels = FakeLabels()
    masked = mask_prompt_tokens(labels, [2, 1], pad_token_id=0)

    assert masked.values == [
        [-100, -100, 12, -100],
        [-100, 21, -100, -100],
    ]
