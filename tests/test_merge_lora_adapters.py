from __future__ import annotations

import runpy

import pytest


torch = pytest.importorskip("torch")
merge_state = runpy.run_path("scripts/merge_lora_adapters.py")["merge_state"]


def test_merge_state_represents_equal_weight_delta_exactly() -> None:
    left_a = torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
    left_b = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [2.0, 3.0]]
    )
    right_a = torch.tensor([[2.0, 0.0, 1.0], [1.0, 3.0, 2.0]])
    right_b = torch.tensor(
        [[0.0, 4.0], [5.0, 0.0], [2.0, 1.0], [3.0, 2.0]]
    )
    merged = merge_state(
        {
            "layer.lora_A.weight": left_a,
            "layer.lora_B.weight": left_b,
        },
        {
            "layer.lora_A.weight": right_a,
            "layer.lora_B.weight": right_b,
        },
        rank=2,
        input_scale=2.0,
        output_scale=1.0,
        left_weight=0.5,
        right_weight=0.5,
    )
    actual = merged["layer.lora_B.weight"] @ merged["layer.lora_A.weight"]
    expected = 0.5 * 2.0 * (left_b @ left_a) + 0.5 * 2.0 * (right_b @ right_a)
    assert torch.equal(actual, expected)


def test_merge_state_supports_outer_expert_rank_dimension() -> None:
    left = {
        "experts.lora_A.weight": torch.ones(1, 2, 3),
        "experts.lora_B.weight": torch.ones(1, 4, 2),
    }
    right = {name: tensor * 2 for name, tensor in left.items()}
    merged = merge_state(
        left,
        right,
        rank=2,
        input_scale=2.0,
        output_scale=1.0,
        left_weight=0.5,
        right_weight=0.5,
    )
    assert merged["experts.lora_A.weight"].shape == (1, 4, 3)
    assert merged["experts.lora_B.weight"].shape == (1, 4, 4)


def test_merge_state_supports_megatron_adapter_factor_names() -> None:
    left = {
        "module.adapter.linear_in.weight": torch.ones(4, 3),
        "module.adapter.linear_out.weight": torch.ones(5, 4),
    }
    right = {name: tensor * 2 for name, tensor in left.items()}
    merged = merge_state(
        left,
        right,
        rank=16,
        input_scale=2.0,
        output_scale=1.0,
        left_weight=0.5,
        right_weight=0.5,
    )
    assert merged["module.adapter.linear_in.weight"].shape == (8, 3)
    assert merged["module.adapter.linear_out.weight"].shape == (5, 8)


def test_merge_state_rejects_different_non_lora_tensors() -> None:
    with pytest.raises(ValueError, match="non-LoRA tensors differ"):
        merge_state(
            {"metadata": torch.tensor([1])},
            {"metadata": torch.tensor([2])},
            rank=2,
            input_scale=2.0,
            output_scale=1.0,
            left_weight=0.5,
            right_weight=0.5,
        )
