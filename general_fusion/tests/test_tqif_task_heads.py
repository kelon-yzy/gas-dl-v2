from __future__ import annotations

import pytest
import torch

from gf.dl.task_heads import (
    FixedTotalTargetHead,
    TargetSlotRegressionHead,
    VariableTotalTargetHead,
    build_tqif_task_head,
)


def test_tqif_h0_and_str_head_contracts() -> None:
    torch.manual_seed(17)
    representations = torch.randn(4, 3, 8)
    h0 = TargetSlotRegressionHead(8, 3)
    str_head = FixedTotalTargetHead(8, 3, total=100.0)

    h0_output = h0(representations)
    str_output = str_head(representations)
    assert h0_output.shape == str_output.shape == (4, 3)
    assert torch.isfinite(h0_output).all()
    assert torch.all(str_output >= 0.0)
    assert str_output.sum(dim=-1).tolist() == pytest.approx([100.0] * 4, abs=1e-5)


def test_tqif_variable_total_head_is_positive_and_closed() -> None:
    torch.manual_seed(29)
    head = VariableTotalTargetHead(8, 3, total_hidden_dim=16)
    output = head(torch.randn(5, 3, 8))
    assert output.shape == (5, 3)
    assert torch.isfinite(output).all()
    assert torch.all(output >= 0.0)
    assert torch.all(output.sum(dim=-1) > 0.0)


def test_tqif_head_builder_supports_fixed_and_variable_total_contracts() -> None:
    representations = torch.randn(2, 3, 8)
    fixed = build_tqif_task_head(
        {"id": "STR", "total": 100.0, "temperature": 1.0},
        input_dim=8,
        target_count=3,
    )
    variable = build_tqif_task_head(
        {"id": "VAR_TOTAL", "total_hidden_dim": 16},
        input_dim=8,
        target_count=3,
    )
    assert fixed(representations).sum(dim=-1).tolist() == pytest.approx([100.0, 100.0], abs=1e-5)
    assert torch.all(variable(representations) >= 0.0)


def test_build_tqif_task_head_rejects_wrong_slot_shape() -> None:
    head = build_tqif_task_head({"id": "H0"}, input_dim=8, target_count=3)
    with pytest.raises(ValueError, match="slot representations"):
        head(torch.randn(2, 8))
