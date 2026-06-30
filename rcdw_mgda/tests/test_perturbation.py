"""测试扰动注入（v1.2 §9.1 12 维通道布局）。"""
from __future__ import annotations

import pytest
import torch

from rcdw.models.single_modal import (
    IDX_NDIR_CO2,
    IDX_TCS,
    IDX_T_C,
    IDX_US_SPEED,
)
from rcdw.perturbation.inject import PERTURBATION_KINDS, inject


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    return torch.randn(8, 8, 12).abs() + 0.5


def test_all_kinds_valid(sample_input):
    for kind in PERTURBATION_KINDS:
        out = inject(sample_input, kind, 0.05)
        assert out.shape == sample_input.shape


def test_inject_rejects_legacy_6_channel():
    """v1.2: 旧 6 维布局应被拒绝（防止误用旧 toy 数据）。"""
    x = torch.randn(4, 8, 6)
    with pytest.raises(ValueError, match="12-channel"):
        inject(x, "optical_atten", 0.1)


def test_zero_level_unchanged(sample_input):
    """level=0 时输出与输入相同（确定性扰动）。"""
    for kind in ["optical_atten", "temperature"]:
        out = inject(sample_input, kind, 0.0)
        torch.testing.assert_close(out, sample_input)


def test_optical_atten_decreases_ndir_co2(sample_input):
    """v1.2 §9.1: optical_atten 目标改为 IDX_NDIR_CO2 (=0)。"""
    out = inject(sample_input, "optical_atten", 0.1)
    assert out[..., IDX_NDIR_CO2].mean() < sample_input[..., IDX_NDIR_CO2].mean()
    # 其他通道不应被改变
    for idx in [IDX_TCS, IDX_US_SPEED, IDX_T_C]:
        torch.testing.assert_close(out[..., idx], sample_input[..., idx])


def test_thermal_targets_tcs(sample_input):
    """thermal 应仅改 IDX_TCS。"""
    torch.manual_seed(0)
    out = inject(sample_input, "thermal", 0.1)
    diff_tcs = (out[..., IDX_TCS] - sample_input[..., IDX_TCS]).abs().mean()
    assert diff_tcs > 0
    # NDIR/US/T 不变
    for idx in [IDX_NDIR_CO2, IDX_US_SPEED, IDX_T_C]:
        torch.testing.assert_close(out[..., idx], sample_input[..., idx])


def test_ultrasonic_targets_us_speed(sample_input):
    """v1.2 §9.1: ultrasonic 目标改为 IDX_US_SPEED (=8)。"""
    torch.manual_seed(0)
    out = inject(sample_input, "ultrasonic", 0.1)
    diff_us = (out[..., IDX_US_SPEED] - sample_input[..., IDX_US_SPEED]).abs().mean()
    assert diff_us > 0
    for idx in [IDX_NDIR_CO2, IDX_TCS, IDX_T_C]:
        torch.testing.assert_close(out[..., idx], sample_input[..., idx])


def test_temperature_shift_on_t_c(sample_input):
    """v1.2 §9.1: temperature 目标改为 IDX_T_C (=2)。"""
    out = inject(sample_input, "temperature", 0.1)
    expected_shift = 0.1 * 80.0
    diff = (out[..., IDX_T_C] - sample_input[..., IDX_T_C]).mean()
    assert diff.item() == pytest.approx(expected_shift, abs=0.01)
    # 其他通道不变
    for idx in [IDX_NDIR_CO2, IDX_TCS, IDX_US_SPEED]:
        torch.testing.assert_close(out[..., idx], sample_input[..., idx])


def test_input_not_mutated(sample_input):
    original = sample_input.clone()
    inject(sample_input, "thermal", 0.1)
    torch.testing.assert_close(sample_input, original)


def test_invalid_kind_raises(sample_input):
    with pytest.raises(ValueError, match="unknown perturbation"):
        inject(sample_input, "nonexistent", 0.1)
