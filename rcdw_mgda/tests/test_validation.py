"""测试 RCDW validation/integrity 的 10 项不变量。

对应方案 §7.1 / §11.1。
"""
from __future__ import annotations

import numpy as np
import pytest

from rcdw.sim.core.schema import (
    COMPONENT_FIELDS,
    LEGACY_CONDITION_FIELDS,
    SLOW_CHANNELS,
)
from rcdw.sim.generation.conditions import generate_condition_rows
from rcdw.sim.packaging.splits import build_default_split_rows
from rcdw.sim.validation.integrity import validate_benchmark_assets


def _label_array(conds):
    return np.array(
        [[float(r[f]) for f in COMPONENT_FIELDS] for r in conds], dtype=np.float32
    )


def _mock_arrays(n, t, w_us=200, w_fm=500):
    return {
        "slow": np.zeros((n, t, len(SLOW_CHANNELS)), dtype=np.float32),
        "ultrasonic": np.zeros((n, t, w_us), dtype=np.int16),
        "ultrasonic_scale": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_tof_s": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_tof_observed_s": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_peak_index": np.zeros((n, t), dtype=np.int32),
        "ultrasonic_sound_speed_m_per_s": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_sound_speed_estimated_m_per_s": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_alpha_true_npm": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_tof_quality": np.zeros((n, t), dtype=np.float32),
        "ultrasonic_tof_accepted": np.zeros((n, t), dtype=np.int8),
        "fiber_mic": np.zeros((n, t, w_fm), dtype=np.int16),
        "fiber_mic_scale": np.zeros((n, t), dtype=np.float32),
    }


def test_validation_passes_on_clean_input():
    conds = generate_condition_rows(8, seed=42)
    splits = build_default_split_rows(conds, seed=42)
    summary = validate_benchmark_assets(conds, splits)
    assert summary["status"] == "pass"
    assert summary["sequence_count"] == 8
    assert summary["mixture_count"] == 8


def test_validation_rejects_legacy_field():
    conds = generate_condition_rows(4, seed=42)
    conds[0]["base_condition_id"] = "X"  # 注入 LEGACY 字段
    splits = build_default_split_rows(conds, seed=42)
    with pytest.raises(ValueError, match="legacy condition fields"):
        validate_benchmark_assets(conds, splits)


def test_validation_rejects_duplicate_sequence_id():
    conds = generate_condition_rows(4, seed=42)
    conds[1]["sequence_id"] = conds[0]["sequence_id"]
    splits = build_default_split_rows(conds, seed=42)
    with pytest.raises(ValueError, match="sequence_id must be unique"):
        validate_benchmark_assets(conds, splits)


def test_validation_rejects_duplicate_mixture_id():
    conds = generate_condition_rows(4, seed=42)
    conds[1]["mixture_id"] = conds[0]["mixture_id"]
    splits = build_default_split_rows(conds, seed=42)
    with pytest.raises(ValueError, match="mixture_id must be unique"):
        validate_benchmark_assets(conds, splits)


def test_validation_rejects_component_sum_off():
    conds = generate_condition_rows(4, seed=42)
    conds[0]["x_O2"] = "30.0"  # 故意把和搞错
    splits = build_default_split_rows(conds, seed=42)
    with pytest.raises(ValueError, match="component\\+background total"):
        validate_benchmark_assets(conds, splits)


def test_validation_rejects_split_missing_sequence():
    conds = generate_condition_rows(5, seed=42)
    splits = build_default_split_rows(conds, seed=42)
    # 删掉 train 的第一个 sequence
    splits["train"] = splits["train"][1:]
    with pytest.raises(ValueError, match="split rows must cover every sequence"):
        validate_benchmark_assets(conds, splits)


def test_validation_array_shape_mismatch_rejected():
    conds = generate_condition_rows(4, seed=42)
    splits = build_default_split_rows(conds, seed=42)
    labels = _label_array(conds)
    arrays = _mock_arrays(n=4, t=16)
    # 故意把 slow 通道数改错
    arrays["slow"] = np.zeros((4, 16, 99), dtype=np.float32)
    with pytest.raises(ValueError, match="slow channel axis"):
        validate_benchmark_assets(conds, splits, arrays, labels)


def test_validation_array_shapes_pass():
    conds = generate_condition_rows(4, seed=42)
    splits = build_default_split_rows(conds, seed=42)
    labels = _label_array(conds)
    arrays = _mock_arrays(n=4, t=16)
    summary = validate_benchmark_assets(conds, splits, arrays, labels)
    assert summary["status"] == "pass"


def test_validation_scaler_passthrough_check_passes():
    conds = generate_condition_rows(4, seed=42)
    splits = build_default_split_rows(conds, seed=42)
    scaler = {
        "channel_entries": [
            {"channel": "ultrasonic_peak_index", "strategy": "passthrough"},
            {"channel": "ultrasonic_tof_quality", "strategy": "passthrough"},
            {"channel": "V_NDIR_CO2", "strategy": "z_score", "mean": 1.0, "std": 0.1},
        ]
    }
    summary = validate_benchmark_assets(
        conds,
        splits,
        scaler=scaler,
        expected_passthrough_channels=(
            "ultrasonic_peak_index",
            "ultrasonic_tof_quality",
        ),
    )
    assert summary["status"] == "pass"
    assert summary["scaler_passthrough_status"] == "pass"


def test_validation_scaler_passthrough_missing_raises():
    """方案 v1.2 §6.5 不变量: 应被标 passthrough 的通道若标了 z_score, validation 应失败。"""
    conds = generate_condition_rows(4, seed=42)
    splits = build_default_split_rows(conds, seed=42)
    scaler_bad = {
        "channel_entries": [
            {"channel": "ultrasonic_peak_index", "strategy": "z_score",
             "mean": 100.0, "std": 50.0},  # 错: 应是 passthrough
        ]
    }
    with pytest.raises(ValueError, match="must be marked strategy='passthrough'"):
        validate_benchmark_assets(
            conds,
            splits,
            scaler=scaler_bad,
            expected_passthrough_channels=("ultrasonic_peak_index",),
        )


def test_validation_scaler_passthrough_skip_channels_missing_raises():
    """默认 ultrasonic passthrough 通道即使不在 slow channel_entries 中,也必须记录到 skip_channels。"""
    conds = generate_condition_rows(3, seed=1)
    splits = build_default_split_rows(conds, seed=1)
    scaler_bad = {
        "channel_entries": [
            {"channel": "V_NDIR_CO2", "strategy": "z_score", "mean": 1.0, "std": 0.1},
        ],
        "skip_channels": [],
    }
    with pytest.raises(ValueError, match="skip_channels"):
        validate_benchmark_assets(
            conds,
            splits,
            scaler=scaler_bad,
            expected_passthrough_channels=("ultrasonic_peak_index",),
        )


def test_validation_split_names_no_extrapolation():
    """方案 §2.6: SPLIT_NAMES 不应含 extrapolation。"""
    from rcdw.sim.core.schema import SPLIT_NAMES

    assert SPLIT_NAMES == ("train", "val", "test")
    assert "extrapolation" not in SPLIT_NAMES


def test_legacy_field_blacklist():
    """方案 §4.8: LEGACY_CONDITION_FIELDS 不应被任何 condition 引入。"""
    assert set(LEGACY_CONDITION_FIELDS) == {
        "base_condition_id",
        "noise_seed_index",
        "noise_seed",
    }
