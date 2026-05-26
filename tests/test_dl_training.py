from __future__ import annotations

import torch

from dl.training.losses import LOSS_REGISTRY, build_loss
from dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from sim.core.schema import COMPONENT_FIELDS


class TestLosses:
    def test_registry_contains_baseline_losses(self):
        assert {"mse", "mae", "smooth_l1", "huber"}.issubset(LOSS_REGISTRY)

    def test_build_loss_from_name(self):
        loss = build_loss("mse")
        value = loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        assert torch.isclose(value, torch.tensor(2.0))

    def test_build_loss_from_config_passes_kwargs(self):
        loss = build_loss({"name": "smooth_l1", "beta": 0.5})
        assert loss.beta == 0.5

    def test_build_unknown_loss_raises(self):
        try:
            build_loss({"name": "imaginary"})
        except ValueError as exc:
            assert "imaginary" in str(exc)
        else:
            raise AssertionError("build_loss should reject unknown loss names")


class TestRegressionMetrics:
    def test_regression_metrics_for_perfect_prediction(self):
        y_true = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]])
        metrics = regression_metrics(y_true, y_true)
        assert metrics == RegressionMetrics(mae=0.0, rmse=0.0, r2=1.0)

    def test_regression_metrics_for_nonperfect_prediction(self):
        y_true = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        y_pred = torch.tensor([[1.0, 1.0], [1.0, 5.0]])
        metrics = regression_metrics(y_pred, y_true)
        assert metrics.mae == 1.0
        assert round(metrics.rmse, 6) == round((6.0 / 4.0) ** 0.5, 6)
        assert metrics.r2 < 1.0

    def test_constant_target_r2_is_one_for_exact_match(self):
        y_true = torch.ones(3, 4)
        metrics = regression_metrics(y_true, y_true)
        assert metrics.r2 == 1.0

    def test_constant_target_r2_is_zero_for_mismatch(self):
        y_true = torch.ones(3, 4)
        y_pred = torch.zeros(3, 4)
        metrics = regression_metrics(y_pred, y_true)
        assert metrics.r2 == 0.0

    def test_component_metrics_use_v4_component_fields(self):
        y_true = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]])
        y_pred = y_true + 0.1
        metrics = component_regression_metrics(y_pred, y_true)
        assert tuple(metrics) == COMPONENT_FIELDS
        for value in metrics.values():
            assert isinstance(value, RegressionMetrics)
            assert round(value.mae, 6) == 0.1

    def test_metric_shape_mismatch_raises(self):
        try:
            regression_metrics(torch.zeros(2, 4), torch.zeros(2, 3))
        except ValueError as exc:
            assert "shapes must match" in str(exc)
        else:
            raise AssertionError("regression_metrics should reject shape mismatch")

    def test_component_name_mismatch_raises(self):
        try:
            component_regression_metrics(torch.zeros(2, 4), torch.zeros(2, 4), component_names=("x",))
        except ValueError as exc:
            assert "component_names length" in str(exc)
        else:
            raise AssertionError("component_regression_metrics should reject name mismatch")
