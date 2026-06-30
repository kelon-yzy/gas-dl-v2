"""测试 RCDW packaging：arrays / manifest / splits / scalers / index / io。

对应方案 §6.1-§6.6 / §11.1。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rcdw.sim.core.schema import (
    COMPONENT_FIELDS,
    SCHEMA_VERSION,
    SLOW_CHANNELS,
    SLOW_MODAL_GROUPS,
    SPLIT_NAMES,
)
from rcdw.sim.packaging.arrays import write_arrays
from rcdw.sim.packaging.index import build_sequence_index_rows
from rcdw.sim.packaging.io import write_csv, write_json
from rcdw.sim.packaging.manifest import build_manifest
from rcdw.sim.packaging.scalers import (
    DEFAULT_PASSTHROUGH_CHANNELS,
    fit_z_score_scalers,
)
from rcdw.sim.packaging.splits import (
    build_default_split_rows,
    build_split_groups,
)


# ---- io ----


def test_write_csv_roundtrip(tmp_path):
    rows = [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]
    path = tmp_path / "test.csv"
    write_csv(path, ("a", "b"), rows)
    text = path.read_text(encoding="utf-8")
    assert "a,b" in text
    assert "1,2" in text
    assert "3,4" in text


def test_write_json_creates_parent(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.json"
    write_json(path, {"key": "value", "n": 42})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"key": "value", "n": 42}


# ---- index ----


def test_build_sequence_index_rows():
    conditions = [
        {"sequence_id": "RCDW-Q000001", "mixture_id": "RCDW-M000001"},
        {"sequence_id": "RCDW-Q000002", "mixture_id": "RCDW-M000002"},
    ]
    rows = build_sequence_index_rows(
        conditions, stage_profile="standard_exposure", timesteps=128, dt_s=0.5
    )
    assert len(rows) == 2
    assert rows[0]["sequence_id"] == "RCDW-Q000001"
    assert rows[0]["stage_profile"] == "standard_exposure"
    assert rows[0]["n_timesteps"] == "128"
    assert rows[0]["dt_s"] == "0.5"


# ---- splits ----


def _mock_conditions(n: int = 10) -> list[dict[str, str]]:
    return [
        {
            "sequence_id": f"RCDW-Q{i+1:06d}",
            "mixture_id": f"RCDW-M{i+1:06d}",
        }
        for i in range(n)
    ]


def test_build_split_groups_ratio_70_15_15():
    conds = _mock_conditions(20)
    groups = build_split_groups(conds, seed=42)
    assert set(groups.keys()) == {"train", "val", "test"}
    # 不含 extrapolation (与 HG 主线差异)
    assert "extrapolation" not in groups
    total = sum(len(g) for g in groups.values())
    assert total == 20
    # 70/15/15 → 14/3/3
    assert len(groups["train"]) == 14
    assert len(groups["val"]) == 3
    assert len(groups["test"]) == 3


def test_build_split_groups_default_test_ratio_is_0_15():
    """方案 v1.1: test_ratio 默认 0.15(非 HG 的 0.10)。"""
    # 100 个 group → train=70, val=15, test=15
    conds = _mock_conditions(100)
    groups = build_split_groups(conds, seed=42)
    assert len(groups["train"]) == 70
    assert len(groups["val"]) == 15
    assert len(groups["test"]) == 15


def test_split_no_overlap_by_mixture():
    conds = _mock_conditions(30)
    groups = build_split_groups(conds, seed=42)
    a = groups["train"] & groups["val"]
    b = groups["train"] & groups["test"]
    c = groups["val"] & groups["test"]
    assert not a
    assert not b
    assert not c


def test_build_default_split_rows_covers_all_sequences():
    conds = _mock_conditions(20)
    rows = build_default_split_rows(conds, seed=42)
    assert set(rows.keys()) == set(SPLIT_NAMES)
    all_seqs = [r["sequence_id"] for name in SPLIT_NAMES for r in rows[name]]
    assert len(all_seqs) == 20
    assert len(set(all_seqs)) == 20


def test_split_reproducibility():
    conds = _mock_conditions(30)
    a = build_default_split_rows(conds, seed=42)
    b = build_default_split_rows(conds, seed=42)
    for name in SPLIT_NAMES:
        assert a[name] == b[name]


# ---- scalers ----


def _mock_slow_matrix(n_seq: int = 10, t: int = 8) -> np.ndarray:
    """构造 (n, t, 7) slow 矩阵: 7 个 SLOW_CHANNELS 顺序。"""
    rng = np.random.default_rng(0)
    return rng.normal(loc=1.5, scale=0.3, size=(n_seq, t, len(SLOW_CHANNELS))).astype(
        np.float32
    )


def test_fit_z_score_scalers_train_only():
    """scaler 应仅用 train_indexes 拟合(不读 val/test 数据)。"""
    matrix = _mock_slow_matrix(n_seq=20)
    seq_scaler, modal_scaler = fit_z_score_scalers(
        matrix,
        train_indexes=list(range(14)),
        channel_names=SLOW_CHANNELS,
        modal_groups=SLOW_MODAL_GROUPS,
        skip_channels=(),  # 暂关闭 passthrough, 检查纯 z_score
    )
    assert seq_scaler["fit_scope"] == "train_split_only"
    assert seq_scaler["method"] == "z_score"
    assert len(seq_scaler["mean"]) == len(SLOW_CHANNELS)
    # train-only: mean 应等于 matrix[:14] 的均值
    expected_mean = matrix[:14].mean(axis=(0, 1))
    np.testing.assert_allclose(seq_scaler["mean"], expected_mean, rtol=1e-5)


def test_fit_z_score_scalers_modal_groups_present():
    matrix = _mock_slow_matrix()
    _, modal_scaler = fit_z_score_scalers(
        matrix,
        train_indexes=[0, 1, 2, 3, 4, 5],
        channel_names=SLOW_CHANNELS,
        modal_groups=SLOW_MODAL_GROUPS,
        skip_channels=(),
    )
    stats = modal_scaler["modal_stats"]
    assert set(stats.keys()) == set(SLOW_MODAL_GROUPS.keys())
    for modal_name, channels in SLOW_MODAL_GROUPS.items():
        entry = stats[modal_name]
        assert entry["channel_names"] == list(channels)
        assert len(entry["mean"]) == len(channels)


def test_fit_z_score_scalers_no_train_raises():
    matrix = _mock_slow_matrix()
    with pytest.raises(ValueError, match="train sequences"):
        fit_z_score_scalers(
            matrix,
            train_indexes=[],
            channel_names=SLOW_CHANNELS,
            modal_groups=SLOW_MODAL_GROUPS,
        )


def test_fit_z_score_scalers_skip_channels_passthrough():
    """方案 §6.5 v1.2: 跳过通道应标 strategy=passthrough。"""
    matrix = _mock_slow_matrix()
    # 这里 SLOW_CHANNELS 仅 7 项, 跳过通道集合中只有那些与 SLOW_CHANNELS 相交的项
    # 才会生效。RCDW slow 矩阵不含 ultrasonic_* 通道, 所以这里测试跳过 V_TCS 作示例。
    seq_scaler, modal_scaler = fit_z_score_scalers(
        matrix,
        train_indexes=list(range(6)),
        channel_names=SLOW_CHANNELS,
        modal_groups=SLOW_MODAL_GROUPS,
        skip_channels=("V_TCS",),
    )
    tcs_idx = SLOW_CHANNELS.index("V_TCS")
    entry = seq_scaler["channel_entries"][tcs_idx]
    assert entry["channel"] == "V_TCS"
    assert entry["strategy"] == "passthrough"
    # 非跳过通道仍走 z_score
    co2_idx = SLOW_CHANNELS.index("V_NDIR_CO2")
    co2_entry = seq_scaler["channel_entries"][co2_idx]
    assert co2_entry["strategy"] == "z_score"
    assert "mean" in co2_entry
    assert "std" in co2_entry


def test_default_passthrough_channels_match_v1_2_spec():
    """v1.2 §6.5: 默认跳过 peak_index / tof_quality / tof_accepted。"""
    assert set(DEFAULT_PASSTHROUGH_CHANNELS) == {
        "ultrasonic_peak_index",
        "ultrasonic_tof_quality",
        "ultrasonic_tof_accepted",
    }


# ---- arrays.write_arrays ----


def _mock_arrays(n_seq: int = 3, t: int = 8, w_us: int = 1000, w_fm: int = 2000):
    return {
        "slow": np.zeros((n_seq, t, len(SLOW_CHANNELS)), dtype=np.float32),
        "ultrasonic": np.zeros((n_seq, t, w_us), dtype=np.int16),
        "ultrasonic_scale": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_tof_s": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_tof_observed_s": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_peak_index": np.zeros((n_seq, t), dtype=np.int32),
        "ultrasonic_sound_speed_m_per_s": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_sound_speed_estimated_m_per_s": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_alpha_true_npm": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_tof_quality": np.zeros((n_seq, t), dtype=np.float32),
        "ultrasonic_tof_accepted": np.zeros((n_seq, t), dtype=np.int8),
        "fiber_mic": np.zeros((n_seq, t, w_fm), dtype=np.int16),
        "fiber_mic_scale": np.zeros((n_seq, t), dtype=np.float32),
    }


def test_write_arrays_creates_all_files(tmp_path):
    arrays = _mock_arrays()
    labels = np.zeros((3, 3), dtype=np.float32)
    sequence_ids = ["RCDW-Q000001", "RCDW-Q000002", "RCDW-Q000003"]
    shapes = write_arrays(
        tmp_path, arrays, labels, sequence_ids,
        SLOW_CHANNELS, COMPONENT_FIELDS, "memmap",
    )
    expected = [
        "slow.npy",
        "ultrasonic_int16.npy",
        "ultrasonic_scale.npy",
        "ultrasonic_tof_s.npy",
        "ultrasonic_tof_observed_s.npy",
        "ultrasonic_peak_index.npy",
        "ultrasonic_sound_speed_m_per_s.npy",
        "ultrasonic_sound_speed_estimated_m_per_s.npy",
        "ultrasonic_alpha_true_npm.npy",
        "ultrasonic_tof_quality.npy",
        "ultrasonic_tof_accepted.npy",
        "fiber_mic_int16.npy",
        "fiber_mic_scale.npy",
    ]
    for name in expected:
        assert (tmp_path / "sequences" / name).is_file(), f"{name} missing"
    assert (tmp_path / "labels" / "y.npy").is_file()
    assert (tmp_path / "metadata" / "sequence_ids.npy").is_file()
    assert (tmp_path / "metadata" / "slow_channel_names.npy").is_file()
    assert (tmp_path / "metadata" / "label_names.npy").is_file()
    # shape 返回字典
    assert shapes["slow"] == [3, 8, len(SLOW_CHANNELS)]
    assert shapes["y"] == [3, 3]


def test_write_arrays_memmap_loadable(tmp_path):
    """落盘后可被 np.load 加载。"""
    arrays = _mock_arrays()
    arrays["slow"][0, 0, 0] = 1.234
    labels = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    write_arrays(
        tmp_path, arrays, labels, ["a", "b", "c"],
        SLOW_CHANNELS, COMPONENT_FIELDS, "memmap",
    )
    loaded = np.load(tmp_path / "sequences" / "slow.npy")
    assert loaded.shape == (3, 8, len(SLOW_CHANNELS))
    assert loaded[0, 0, 0] == pytest.approx(1.234, abs=1e-5)


# ---- manifest ----


def test_build_manifest_rcdw_defaults():
    manifest = build_manifest(
        dataset_slug="rcdw-smoke",
        sequence_count=64,
        seed=42,
        timesteps=32,
        dt_s=0.5,
        storage="memmap",
        multi_path_phase="steady",
        stage_profile="standard_exposure",
        stage_jitter=0.0,
        phase_schedule={"name": "standard_exposure", "segments": []},
        sampling_strategy="lhs",
        path_lms=(0.2, 0.3, 0.4),
        optical_absorption_backend="hitran_hapi_v1",
        shapes={"slow": [64, 32, 7]},
        slow_channels=SLOW_CHANNELS,
        labels=COMPONENT_FIELDS,
    )
    assert manifest["schema_version"] == "rcdw-benchmark-1"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["composition_scheme"] == "rcdw_o2_co2_n2"
    assert manifest["primary_key"] == "mixture_id"
    assert manifest["instance_key"] == "sequence_id"
    assert manifest["slow_channels"] == list(SLOW_CHANNELS)
    assert manifest["labels"] == list(COMPONENT_FIELDS)
    assert manifest["train_modalities"] == ["slow", "ultrasonic"]
    assert manifest["background_fields"] == []


def test_build_manifest_with_scaler_metadata():
    manifest = build_manifest(
        dataset_slug="rcdw-smoke",
        sequence_count=64,
        seed=42,
        timesteps=32,
        dt_s=0.5,
        storage="memmap",
        multi_path_phase="steady",
        stage_profile="standard_exposure",
        stage_jitter=0.0,
        phase_schedule={"name": "standard_exposure", "segments": []},
        sampling_strategy="lhs",
        path_lms=(0.3,),
        optical_absorption_backend="hitran_hapi_v1",
        shapes={},
        slow_channels=SLOW_CHANNELS,
        labels=COMPONENT_FIELDS,
        scaler_metadata={
            "passthrough_channels": list(DEFAULT_PASSTHROUGH_CHANNELS),
            "peak_index_strategy": "skip",
        },
    )
    assert "scaler_metadata" in manifest
    assert manifest["scaler_metadata"]["peak_index_strategy"] == "skip"
