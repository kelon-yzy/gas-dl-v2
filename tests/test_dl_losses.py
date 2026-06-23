from __future__ import annotations

import pytest
import torch

from common.composition import ILR_N2_FIRST_TRANSFORM
from dl.training.losses import (
    FREE_COMPONENT_MSE_LOSS,
    LOSS_REGISTRY,
    WEIGHTED_COMPONENT_MSE_LOSS,
    WEIGHTED_FREE_COMPONENT_MSE_LOSS,
    build_loss,
    loss_config_name,
    validate_loss_model_output,
    validate_loss_target_transform,
)


def test_registry_contains_baseline_losses():
    assert {
        "mse",
        "compositional_mse",
        "ilr_mse",
        FREE_COMPONENT_MSE_LOSS,
        WEIGHTED_COMPONENT_MSE_LOSS,
        WEIGHTED_FREE_COMPONENT_MSE_LOSS,
        "mae",
        "smooth_l1",
        "huber",
    }.issubset(LOSS_REGISTRY)


def test_build_loss_from_name():
    loss = build_loss("mse")
    value = loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
    assert torch.isclose(value, torch.tensor(2.0))


def test_build_semantic_mse_alias():
    loss = build_loss("ilr_mse")
    value = loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
    assert torch.isclose(value, torch.tensor(2.0))


def test_build_loss_from_config_passes_kwargs():
    loss = build_loss({"name": "smooth_l1", "beta": 0.5})
    assert loss.beta == 0.5


def test_loss_config_name_accepts_name_or_config():
    assert loss_config_name("mse") == "mse"
    assert loss_config_name({"name": "smooth_l1", "beta": 0.5}) == "smooth_l1"


def test_validate_loss_target_transform_accepts_configured_loss():
    validate_loss_target_transform({"name": "smooth_l1", "beta": 0.5}, None)
    with pytest.raises(ValueError, match="ilr_mse requires target_transform"):
        validate_loss_target_transform({"name": "ilr_mse"}, None)


def test_free_component_mse_ignores_closure_component():
    loss = build_loss(FREE_COMPONENT_MSE_LOSS)
    pred = torch.tensor([[10.0, 75.0, 5.0, 10.0], [12.0, 70.0, 8.0, 10.0]])
    target = torch.tensor([[11.0, 73.0, 5.0, 11.0], [10.0, 72.0, 7.0, 11.0]])
    target_with_different_n2 = target.clone()
    target_with_different_n2[:, 3] = torch.tensor([99.0, -99.0])

    value = loss(pred, target)
    value_with_different_n2 = loss(pred, target_with_different_n2)
    expected = torch.mean((pred[:, :3] - target[:, :3]) ** 2)

    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(value_with_different_n2, expected)


def test_free_component_mse_accepts_configured_free_component_count():
    loss = build_loss({"name": FREE_COMPONENT_MSE_LOSS, "free_components": 2})
    pred = torch.tensor([[1.0, 2.0, 100.0, 100.0]])
    target = torch.tensor([[0.0, 0.0, -100.0, -100.0]])

    value = loss(pred, target)

    torch.testing.assert_close(value, torch.tensor(2.5))


def test_free_component_mse_rejects_invalid_shapes():
    loss = build_loss(FREE_COMPONENT_MSE_LOSS)
    with pytest.raises(ValueError, match="2D tensors"):
        loss(torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="shapes must match"):
        loss(torch.zeros(2, 4), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="at least 3 component columns"):
        loss(torch.zeros(2, 2), torch.zeros(2, 2))


def test_free_component_mse_rejects_target_transform_pairing():
    with pytest.raises(ValueError, match="without target_transform"):
        validate_loss_target_transform(FREE_COMPONENT_MSE_LOSS, ILR_N2_FIRST_TRANSFORM)


def test_weighted_component_mse_uses_inverse_train_variance():
    train_targets = torch.tensor([[0.0, 0.0, 0.0, 0.0], [2.0, 4.0, 8.0, 16.0]])
    loss = build_loss(
        {"name": WEIGHTED_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"},
        train_targets=train_targets,
    )
    pred = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    target = torch.zeros_like(pred)

    value = loss(pred, target)

    torch.testing.assert_close(loss.component_weights, torch.tensor([1.0, 0.25, 0.0625, 0.015625]))
    torch.testing.assert_close(value, torch.tensor(1.0))


def test_weighted_free_component_mse_ignores_closure_component():
    loss = build_loss({"name": WEIGHTED_FREE_COMPONENT_MSE_LOSS, "component_weights": [1.0, 2.0, 3.0]})
    pred = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    target = torch.zeros_like(pred)
    target_with_different_n2 = target.clone()
    target_with_different_n2[:, 3] = -100.0

    value = loss(pred, target)
    value_with_different_n2 = loss(pred, target_with_different_n2)
    expected = torch.mean(torch.tensor([[1.0, 8.0, 27.0]]))

    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(value_with_different_n2, expected)


def test_weighted_mse_requires_train_targets_for_inverse_variance():
    with pytest.raises(ValueError, match="requires train_targets"):
        build_loss({"name": WEIGHTED_COMPONENT_MSE_LOSS, "weighting": "inverse_train_var"})


def test_weighted_mse_rejects_target_transform_pairing():
    with pytest.raises(ValueError, match="without target_transform"):
        validate_loss_target_transform(WEIGHTED_COMPONENT_MSE_LOSS, ILR_N2_FIRST_TRANSFORM)


def test_free_component_mse_requires_gas_head_output():
    validate_loss_model_output(
        FREE_COMPONENT_MSE_LOSS,
        model_name="phase_window_tcn",
        model_kwargs={"output_mode": "gas_head"},
    )
    with pytest.raises(ValueError, match="output_mode='gas_head'"):
        validate_loss_model_output(
            FREE_COMPONENT_MSE_LOSS,
            model_name="phase_window_tcn",
            model_kwargs={"output_mode": "raw4"},
        )
    with pytest.raises(ValueError, match="out_dim=4"):
        validate_loss_model_output(
            FREE_COMPONENT_MSE_LOSS,
            model_name="phase_window_tcn",
            model_kwargs={"output_mode": "gas_head", "out_dim": 3},
        )
    with pytest.raises(ValueError, match="gas-head DL model"):
        validate_loss_model_output(FREE_COMPONENT_MSE_LOSS, model_name="cnn1d", model_kwargs={})


def test_weighted_mse_requires_gas_head_output():
    validate_loss_model_output(
        WEIGHTED_COMPONENT_MSE_LOSS,
        model_name="handcraft_mlp",
        model_kwargs={},
    )
    validate_loss_model_output(
        WEIGHTED_FREE_COMPONENT_MSE_LOSS,
        model_name="phase_window_tcn",
        model_kwargs={"output_mode": "gas_head"},
    )
    with pytest.raises(ValueError, match="output_mode='gas_head'"):
        validate_loss_model_output(
            WEIGHTED_FREE_COMPONENT_MSE_LOSS,
            model_name="phase_window_tcn",
            model_kwargs={"output_mode": "raw4"},
        )


def test_weighted_component_mse_allows_non_gas_head():
    # weighted_component_mse 监督全 4 列，与 head 无关，phase_window_tcn 上应放开 raw4 / softmax100 / gas_head。
    for output_mode in ("raw4", "softmax100", "gas_head"):
        validate_loss_model_output(
            WEIGHTED_COMPONENT_MSE_LOSS,
            model_name="phase_window_tcn",
            model_kwargs={"output_mode": output_mode, "out_dim": 4},
        )
    with pytest.raises(ValueError, match="out_dim=4"):
        validate_loss_model_output(
            WEIGHTED_COMPONENT_MSE_LOSS,
            model_name="phase_window_tcn",
            model_kwargs={"output_mode": "raw4", "out_dim": 3},
        )
    with pytest.raises(ValueError, match="output_mode"):
        validate_loss_model_output(
            WEIGHTED_COMPONENT_MSE_LOSS,
            model_name="phase_window_tcn",
            model_kwargs={"output_mode": "unknown"},
        )


def test_build_unknown_loss_raises():
    with pytest.raises(ValueError, match="imaginary"):
        build_loss({"name": "imaginary"})


def test_weighted_component_mse_allows_cnn1d_tcn_fusion_raw4():
    # cnn1d_tcn_fusion + raw4 + weighted_component_mse 应为合法组合。
    validate_loss_model_output(
        WEIGHTED_COMPONENT_MSE_LOSS,
        model_name="cnn1d_tcn_fusion",
        model_kwargs={"output_mode": "raw4", "out_dim": 4},
    )


def test_cnn1d_tcn_fusion_free_component_mse_requires_gas_head():
    # free_component_mse 是闭包类损失，要求 gas_head。
    with pytest.raises(ValueError, match="output_mode='gas_head'"):
        validate_loss_model_output(
            FREE_COMPONENT_MSE_LOSS,
            model_name="cnn1d_tcn_fusion",
            model_kwargs={"output_mode": "raw4"},
        )
    # gas_head 应通过
    validate_loss_model_output(
        FREE_COMPONENT_MSE_LOSS,
        model_name="cnn1d_tcn_fusion",
        model_kwargs={"output_mode": "gas_head"},
    )


def test_cnn1d_tcn_fusion_weighted_free_component_requires_gas_head():
    with pytest.raises(ValueError, match="output_mode='gas_head'"):
        validate_loss_model_output(
            WEIGHTED_FREE_COMPONENT_MSE_LOSS,
            model_name="cnn1d_tcn_fusion",
            model_kwargs={"output_mode": "raw4"},
        )
    validate_loss_model_output(
        WEIGHTED_FREE_COMPONENT_MSE_LOSS,
        model_name="cnn1d_tcn_fusion",
        model_kwargs={"output_mode": "gas_head"},
    )


def test_cnn1d_tcn_fusion_rejects_unknown_output_mode():
    with pytest.raises(ValueError, match="output_mode"):
        validate_loss_model_output(
            WEIGHTED_COMPONENT_MSE_LOSS,
            model_name="cnn1d_tcn_fusion",
            model_kwargs={"output_mode": "unknown"},
        )
