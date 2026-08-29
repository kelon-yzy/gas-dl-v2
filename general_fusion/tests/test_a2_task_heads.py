from __future__ import annotations

import pytest
import torch

from gf.dl.evaluation import evaluate_output_constraints
from gf.dl.task_heads import (
    FixedTotalSoftmaxHead,
    RegressionHead,
    SimplexProjectionHead,
    SparsemaxHead,
    project_to_simplex,
    sparsemax,
)


def test_simplex_projection_is_nonnegative_and_fixed_total() -> None:
    raw = torch.tensor([[2.0, -1.0, 4.0], [-3.0, -2.0, -1.0]])
    projected = project_to_simplex(raw)
    assert torch.all(projected >= 0.0)
    assert projected.sum(dim=-1).tolist() == pytest.approx([100.0, 100.0])
    diagnostics = evaluate_output_constraints(projected.detach().numpy())
    assert diagnostics["negative_rate"] == 0.0
    assert diagnostics["composition_sum_mae"] == pytest.approx(0.0, abs=1e-5)


def test_h1_wraps_h0_without_adding_trainable_mapping() -> None:
    base = RegressionHead(4, 3)
    head = SimplexProjectionHead(4, 3, base_head=base)
    assert list(head.parameters()) == list(base.parameters())
    output = head(torch.randn(5, 4))
    assert output.shape == (5, 3)
    assert torch.all(output >= 0.0)
    assert output.sum(dim=-1).tolist() == pytest.approx([100.0] * 5, abs=2e-5)


def test_h2_is_strictly_positive_and_h3_can_emit_exact_zero() -> None:
    h2 = FixedTotalSoftmaxHead(2, 3)
    h3 = SparsemaxHead(2, 3)
    inputs = torch.zeros(4, 2)
    h2_output = h2(inputs)
    assert torch.all(h2_output > 0.0)
    assert h2_output.sum(dim=-1).tolist() == pytest.approx([100.0] * 4, abs=1e-5)

    probabilities = sparsemax(torch.tensor([[3.0, 1.0, -2.0], [0.0, 0.0, 0.0]]))
    assert torch.allclose(probabilities[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(probabilities[1], torch.tensor([1.0 / 3.0] * 3))
    h3_output = h3(inputs)
    assert torch.all(h3_output >= 0.0)
    assert h3_output.sum(dim=-1).tolist() == pytest.approx([100.0] * 4, abs=1e-5)


def test_invalid_fixed_total_head_arguments_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="temperature"):
        FixedTotalSoftmaxHead(2, 3, temperature=0.0)
    with pytest.raises(ValueError, match="total"):
        SparsemaxHead(2, 3, total=0.0)
