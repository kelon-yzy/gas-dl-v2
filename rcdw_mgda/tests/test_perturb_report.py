"""测试 scripts.perturb 的结构化报告辅助函数。"""
from __future__ import annotations

import pytest
import torch

from rcdw.sim.core.schema import GAS_DISPLAY_NAMES
from scripts.perturb import (
    _error_calibration_payload,
    _summarize_model_outputs,
    _thresholded_relative_error_payload,
)


GAS_NAMES = list(GAS_DISPLAY_NAMES)
MODAL_NAMES = ["NDIR", "TCD", "US"]


def test_thresholded_relative_error_filters_low_reference_values() -> None:
    pred = torch.tensor(
        [
            [0.02, 0.20, 0.80],
            [0.10, 0.05, 0.85],
        ],
        dtype=torch.float32,
    )
    ref = torch.tensor(
        [
            [0.001, 0.10, 0.90],
            [0.05, 0.00, 0.95],
        ],
        dtype=torch.float32,
    )

    payload = _thresholded_relative_error_payload(
        pred, ref, gas_names=GAS_NAMES, threshold=0.01
    )

    assert payload["threshold"] == pytest.approx(0.01)
    assert payload["by_gas"]["O2"]["count"] == 1
    assert payload["by_gas"]["CO2"]["count"] == 1
    assert payload["by_gas"]["N2"]["count"] == 2
    assert payload["overall"]["count"] == 4


def test_error_calibration_payload_reports_best_modality_accuracy() -> None:
    y_ref = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]], dtype=torch.float32)
    y_modal = torch.stack(
        [
            y_ref + 0.01,
            y_ref + 0.20,
            y_ref + 0.40,
        ],
        dim=1,
    )
    e_pred = torch.stack(
        [
            torch.full_like(y_ref, 0.01),
            torch.full_like(y_ref, 0.20),
            torch.full_like(y_ref, 0.40),
        ],
        dim=1,
    )

    payload = _error_calibration_payload(
        y_modal,
        e_pred,
        y_ref,
        modal_names=MODAL_NAMES,
        gas_names=GAS_NAMES,
    )

    assert payload["best_modality_accuracy"]["overall"] == pytest.approx(1.0)
    assert payload["by_modality_gas"]["NDIR"]["O2"]["actual_mae_mean"] == pytest.approx(
        0.01
    )


def test_summarize_model_outputs_contains_raw_and_hard_suppress_metrics() -> None:
    y_ref = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]], dtype=torch.float32)
    y_modal = torch.stack(
        [
            y_ref + 0.01,
            y_ref + 0.10,
            y_ref + 0.20,
        ],
        dim=1,
    )
    out = {
        "C": y_ref + 0.02,
        "Y_modal": y_modal,
        "E_pred": torch.stack(
            [
                torch.full_like(y_ref, 0.01),
                torch.full_like(y_ref, 0.10),
                torch.full_like(y_ref, 0.20),
            ],
            dim=1,
        ),
        "W": torch.full((2, 3, 3), 1.0 / 3.0),
    }

    payload, w_final = _summarize_model_outputs(
        out,
        y_ref,
        deg_cfg={"ratio": 4.0, "cap": 0.04},
        modal_names=MODAL_NAMES,
        gas_names=GAS_NAMES,
    )

    assert "raw" in payload
    assert "hard_suppress" in payload
    assert "degraded_rate" in payload
    assert "raw_thresholded_relative_error" in payload
    assert "hard_suppress_thresholded_relative_error" in payload
    assert w_final.shape == (2, 3, 3)
    assert payload["degraded_any_rate"] > 0.0
