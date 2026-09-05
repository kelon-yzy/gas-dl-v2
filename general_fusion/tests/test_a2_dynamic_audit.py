"""A2-DYN 难度审计的回归测试（规划文档 15 的 F7）。

覆盖六类路径：门控组合逻辑（failed_requirements 各分支）、
``_horizon_indices`` 的 exposure_end 失效规则、F1 的配对比值、
O-KIN / O-KIN-OBS 反演的显式失败路径、Jacobian 秩与条件数、
``eligible_dynamic_axes`` 推导。

需要冻结完整包的测试（``data/a2_dynamic_v1``）在包缺失时整组 skip；
纯函数与伪造输入的测试不依赖数据包，任何 clone 环境都能跑。
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gf.sim.a2_dynamic_audit as audit_module
from gf.sim.a2_dynamic_audit import (
    DEVELOPMENT_SPLITS,
    EARLY_HORIZONS,
    FAMILIES,
    _audit_dynamic_non_degenerate,
    _audit_jacobian,
    _condition_number,
    _horizon_indices,
    _kinetic_oracle_predictions,
    _observed_admission_budgets,
    _paired_late_reference_evidence,
    _per_horizon_jacobian_summary,
    run_a2_dynamic_difficulty_audit,
)
from gf.sim.a2_dynamic_dataset import DynamicDataset, _calibration_profiles, load_a2_dynamic_dataset
from gf.sim.a2_sensor_devices import SensorDeviceError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
FROZEN_PACKAGE_DIR = PROJECT_ROOT / "data" / "a2_dynamic_v1"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="session")
def frozen_configs() -> dict[str, dict[str, object]]:
    return {
        "data": _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json"),
        "eval": _load_json(CONFIG_ROOT / "eval" / "a2_dynamic_eval.json"),
        "experiment": _load_json(CONFIG_ROOT / "experiment" / "a2_dynamic_protocol.json"),
        "a2h": _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2h_v2.json"),
    }


@pytest.fixture(scope="session")
def frozen_package(frozen_configs: dict[str, dict[str, object]]) -> DynamicDataset:
    """冻结完整包（6300 行，只读）。包缺失时整组 skip。"""
    if not (FROZEN_PACKAGE_DIR / "manifest.json").is_file():
        pytest.skip("frozen a2_dynamic_v1 package is not available in data/")
    return load_a2_dynamic_dataset(FROZEN_PACKAGE_DIR)


def _slice_dataset(
    full: DynamicDataset,
    rows: list[int] | np.ndarray,
) -> DynamicDataset:
    """从完整包按行切片出合法子集（行间相互独立，切片安全）。"""
    rows = np.asarray(rows, dtype=np.int64)
    return replace(
        full,
        records=tuple(full.records[int(index)] for index in rows),
        signals=full.signals[rows],
        valid_mask=full.valid_mask[rows],
        quality=full.quality[rows],
        target=full.target[rows],
        phase_id=full.phase_id[rows],
        observation_index=np.arange(len(rows), dtype=np.int64),
        inlet_composition=full.inlet_composition[rows],
        inlet_coefficient=full.inlet_coefficient[rows],
        chamber_composition=full.chamber_composition[rows],
        equilibrium_reference_signals=full.equilibrium_reference_signals[rows],
        clean_device_signals=full.clean_device_signals[rows],
        device_states=full.device_states[rows],
        privileged_parameters=full.privileged_parameters[rows],
        device_audit={
            key: value[rows] for key, value in full.device_audit.items()
        },
    )


def _rows_of(
    full: DynamicDataset,
    *,
    family: str,
    split: str,
    limit: int | None = None,
    noise_profile_id: str | None = None,
) -> list[int]:
    rows = [
        index
        for index, record in enumerate(full.records)
        if record["family"] == family
        and record["split"] == split
        and (noise_profile_id is None or record["noise_profile_id"] == noise_profile_id)
    ]
    return rows if limit is None else rows[:limit]


# ---------------------------------------------------------------- 纯函数层


def test_horizon_indices_marks_exposure_end_rows_invalid() -> None:
    """F7: cutoff 落进 recovery（>= exposure_end）的行必须标记为 -1。"""
    data_config = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2_dynamic_v1.json")
    dt_s = float(data_config["dt_s"])
    time_s = np.arange(0.0, 200.0, dt_s)
    # 三条序列：P150 cutoff 分别为 onset+149.8 s；end 190 s 时有效，
    # end 150 s（=cutoff）与 end 140 s 时都必须失效。
    records = [
        {
            "exposure_onset_s": 30.0,
            "exposure_end_s": 190.0,
            "noise_profile_id": "NOISE-1X",
        },
        {
            "exposure_onset_s": 30.0,
            "exposure_end_s": 150.0,
            "noise_profile_id": "NOISE-1X",
        },
        {
            "exposure_onset_s": 30.0,
            "exposure_end_s": 140.0,
            "noise_profile_id": "NOISE-1X",
        },
    ]
    indices = _horizon_indices(time_s, records, data_config)
    horizons = {h["horizon_id"]: h for h in data_config["prefix_horizons"]}
    assert "P150" in horizons and "FULL" in horizons
    # 前两条（end=190 / end=150=cutoff）P150 判定：cutoff >= end 即失效
    assert indices["P150"][0] >= 0
    assert indices["P150"][1] == -1
    assert indices["P150"][2] == -1
    for horizon_id, horizon in horizons.items():
        if horizon_id == "FULL":
            continue
        for row in range(3):
            cutoff = (
                records[row]["exposure_onset_s"]
                + float(horizon["exposure_after_s"])
                - dt_s
            )
            expect_invalid = cutoff >= records[row]["exposure_end_s"]
            assert (indices[horizon_id][row] == -1) == expect_invalid, (
                f"{horizon_id} row {row}: cutoff={cutoff}, end={records[row]['exposure_end_s']}"
            )


def test_paired_late_reference_evidence_uses_row_intersection() -> None:
    """F7/F1: 配对比值的分子分母必须来自同一批行，且可复算。"""
    target_ranges = np.asarray([20.0, 20.0, 20.0], dtype=np.float64)
    rows = np.arange(30, dtype=np.int64)
    group_ids = tuple(f"g{index:04d}" for index in range(30))
    dataset = SimpleNamespace(target=np.full((30, 3), 50.0), group_ids=group_ids)
    # 伪造"一半行晚期失效"的场景：early 30 行全有效（奇行误差 10、偶行
    # 误差 4），late 只覆盖偶行 15 行（误差 10）。naive 口径会混入奇行的
    # 大误差，配对口径只看到偶行——两者必须不同。
    early_indices = rows
    late_indices = rows[::2]
    early_predictions = np.broadcast_to(
        np.where(rows % 2 == 0, 46.0, 40.0)[:, None], (30, 3)
    )
    late_predictions = np.broadcast_to(np.full((15, 1), 40.0), (15, 3))
    evidence = _paired_late_reference_evidence(
        dataset,  # type: ignore[arg-type]
        target_ranges,
        early_indices=early_indices,
        early_predictions=early_predictions,
        late_indices=late_indices,
        late_predictions=late_predictions,
        late_reference_horizon="P150",
    )
    assert evidence["early_row_count"] == 30
    assert evidence["late_row_count"] == 15
    assert evidence["paired_row_count"] == 15
    # 配对比值只在交集（偶行）上计算；RNMAE = 每分量 MAE/range 的均值。
    assert evidence["B-LAST_val_macro_RNMAE_early_paired"] == pytest.approx(4.0 / 20.0)
    assert evidence["B-LAST_val_macro_RNMAE_late_paired"] == pytest.approx(10.0 / 20.0)
    assert evidence["relative_degradation"] == pytest.approx(0.2 / 0.5 - 1.0)
    # naive（各自全部行）口径混入奇行误差 10/20=0.5 → 与配对口径不同。
    naive_early = 0.5 * (0.2 + 0.5)  # 奇行 0.5、偶行 0.2 各一半
    naive_late = 0.5
    assert evidence["relative_degradation"] != pytest.approx(naive_early / naive_late - 1.0)


def test_paired_evidence_without_overlap_returns_none_degradation() -> None:
    evidence = _paired_late_reference_evidence(
        SimpleNamespace(target=np.zeros((0, 3)), group_ids=()),  # type: ignore[arg-type]
        np.asarray([20.0, 20.0, 20.0]),
        early_indices=np.asarray([1, 2], dtype=np.int64),
        early_predictions=np.zeros((2, 3)),
        late_indices=np.asarray([9], dtype=np.int64),
        late_predictions=np.zeros((1, 3)),
        late_reference_horizon="P150",
    )
    assert evidence["paired_row_count"] == 0
    assert evidence["relative_degradation"] is None


def test_condition_number_diagonal_and_singular() -> None:
    assert _condition_number(np.eye(3)) == pytest.approx(1.0)
    assert _condition_number(np.diag([2.0, 1.0, 1.0e-14])) == pytest.approx(2.0e14)
    # 最小奇异值低于 1e-15 门槛时按奇异处理（inf）。
    assert _condition_number(np.diag([2.0, 1.0, 1.0e-16])) == math_inf()
    assert _condition_number(np.zeros((3, 2))) == math_inf()


def math_inf() -> float:
    return float(np.inf)


def test_per_horizon_jacobian_summary_aggregates_declared_samples() -> None:
    samples = []
    for index in range(20):
        sample = {
            f"fixed_rank_{horizon}": 2 for horizon in EARLY_HORIZONS
        }
        for horizon in EARLY_HORIZONS:
            sample[f"fixed_rank_{horizon}"] = 2
            sample[f"fixed_condition_number_{horizon}"] = 10.0 + float(index)
        sample["row"] = index
        samples.append(sample)
    summary = _per_horizon_jacobian_summary(samples)
    for horizon in EARLY_HORIZONS:
        assert summary[horizon]["sample_count"] == 20
        assert summary[horizon]["fixed_full_rank_fraction"] == 1.0
        assert summary[horizon]["fixed_condition_number_p95"] == pytest.approx(
            float(np.percentile(np.arange(20) + 10.0, 95))
        )


# ---------------------------------------------------------------- 数据包依赖层


def test_kinetic_oracle_observed_mode_records_explicit_failures(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F7: O-KIN-OBS 对观测越界的行显式记失败并 NaN 占位，不 raise。"""
    data_config = frozen_configs["data"]
    experiment_config = frozen_configs["experiment"]
    rows = _rows_of(
        frozen_package,
        family="D-IID",
        split="val",
        noise_profile_id="NOISE-1X",
        limit=4,
    )
    dataset = _slice_dataset(frozen_package, rows)
    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    endpoint = int(horizon_indices["P015"][0])
    # 篡改 NDIR 观测（channel 2）为恒定零电压：超出该行注册噪声预算。
    # 数组在 __post_init__ 后只读，必须先改副本再构造子集。
    signals = np.array(dataset.signals, copy=True)
    signals[:, 2, endpoint, 0] = 0.0
    subset = replace(dataset, signals=signals)
    calibrations = _calibration_profiles(data_config, frozen_configs["a2h"])
    noise_base = np.asarray(
        experiment_config["pilot"]["observation_noise_std_by_sensor"], dtype=np.float64
    )
    indices = np.arange(len(rows), dtype=np.int64)
    budgets = _observed_admission_budgets(
        subset,
        indices,
        data_config=data_config,
        calibrations=calibrations,
        noise_base=noise_base,
    )
    failures: list[dict[str, object]] = []
    predictions = _kinetic_oracle_predictions(
        subset,
        indices,
        np.full(len(rows), endpoint, dtype=np.int64),
        data_config=data_config,
        kinetic_cache={},
        heos_interpolation_cache={},
        input_mode="observed",
        admission_budgets=budgets,
        inversion_failures=failures,
    )
    assert len(failures) == len(rows)
    assert np.all(np.isnan(predictions))
    # 失败按行显式记录（stage 来自反演失败位置），reason 保留完整消息。
    stages = {str(item["stage"]) for item in failures}
    assert stages == {"ndir_ratio_domain"}
    assert all(str(item["reason"]) for item in failures)


def test_kinetic_oracle_clean_mode_raises_instead_of_silently_failing(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F7: clean 模式（O-KIN）不吞错——越界输入显式 raise。"""
    data_config = frozen_configs["data"]
    rows = _rows_of(frozen_package, family="D-IID", split="val", limit=1)
    dataset = _slice_dataset(frozen_package, rows)
    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    endpoint = int(horizon_indices["P015"][0])
    clean = np.array(dataset.clean_device_signals, copy=True)
    clean[0, 2, :] = 0.0  # NDIR 恒定零电压：域外
    subset = replace(dataset, clean_device_signals=clean)
    with pytest.raises(SensorDeviceError):
        _kinetic_oracle_predictions(
            subset,
            np.asarray([0], dtype=np.int64),
            np.asarray([endpoint], dtype=np.int64),
            data_config=data_config,
            kinetic_cache={},
            heos_interpolation_cache={},
        )


def test_kinetic_oracle_observed_tof_boundary_failure(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F7: 超声 ToF 观测越界按 tof_quantization_boundary 记失败。"""
    data_config = frozen_configs["data"]
    experiment_config = frozen_configs["experiment"]
    rows = _rows_of(
        frozen_package,
        family="D-IID",
        split="val",
        noise_profile_id="NOISE-1X",
        limit=4,
    )
    dataset = _slice_dataset(frozen_package, rows)
    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    endpoint = int(horizon_indices["P015"][0])
    # ToF 单位 s，量级 1e-3；置为 1 s 必远超注册预算。
    signals = np.array(dataset.signals, copy=True)
    signals[:, 0, endpoint, 0] = 1.0
    subset = replace(dataset, signals=signals)
    calibrations = _calibration_profiles(data_config, frozen_configs["a2h"])
    noise_base = np.asarray(
        experiment_config["pilot"]["observation_noise_std_by_sensor"], dtype=np.float64
    )
    indices = np.arange(len(rows), dtype=np.int64)
    budgets = _observed_admission_budgets(
        subset,
        indices,
        data_config=data_config,
        calibrations=calibrations,
        noise_base=noise_base,
    )
    failures: list[dict[str, object]] = []
    predictions = _kinetic_oracle_predictions(
        subset,
        indices,
        np.full(len(rows), endpoint, dtype=np.int64),
        data_config=data_config,
        kinetic_cache={},
        heos_interpolation_cache={},
        input_mode="observed",
        admission_budgets=budgets,
        inversion_failures=failures,
    )
    assert len(failures) == len(rows)
    assert np.all(np.isnan(predictions))
    stages = {str(item["stage"]) for item in failures}
    assert stages == {"tof_quantization_boundary"} or stages == {"tof_zero_he_boundary"}


def test_kinetic_oracle_observed_error_exceeds_clean_and_scales_with_noise(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F2/F7: O-KIN-OBS 输入接对——误差 > O-KIN，且 5X 档噪声 > 1X 档。"""
    data_config = frozen_configs["data"]
    experiment_config = frozen_configs["experiment"]
    calibrations = _calibration_profiles(data_config, frozen_configs["a2h"])
    noise_base = np.asarray(
        experiment_config["pilot"]["observation_noise_std_by_sensor"], dtype=np.float64
    )
    one_x = _rows_of(
        frozen_package,
        family="D-NOISE-DRIFT",
        split="train",
        noise_profile_id="NOISE-1X",
        limit=10,
    )
    # 开发 split 内噪声按族分层（train 1X / val 2X / stress_val 5X）：
    # 用同族最高噪声档 5X 与 1X 对比，5 倍 scale 差足以分辨。
    high_noise = _rows_of(
        frozen_package,
        family="D-NOISE-DRIFT",
        split="stress_val",
        noise_profile_id="NOISE-5X",
        limit=10,
    )
    if not one_x or not high_noise:
        pytest.skip("frozen package lacks the required noise profile rows")
    dataset = _slice_dataset(frozen_package, one_x + high_noise)
    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    # 反演与预算都作用于子集内的行号（__post_init__ 后行号是子集局部的）。
    local_indices = np.arange(len(one_x) + len(high_noise), dtype=np.int64)
    endpoints = horizon_indices["P015"][local_indices]
    budgets = _observed_admission_budgets(
        dataset,
        local_indices,
        data_config=data_config,
        calibrations=calibrations,
        noise_base=noise_base,
    )
    failures: list[dict[str, object]] = []
    observed = _kinetic_oracle_predictions(
        dataset,
        local_indices,
        endpoints,
        data_config=data_config,
        kinetic_cache={},
        heos_interpolation_cache={},
        input_mode="observed",
        admission_budgets=budgets,
        inversion_failures=failures,
    )
    clean = _kinetic_oracle_predictions(
        dataset,
        local_indices,
        endpoints,
        data_config=data_config,
        kinetic_cache={},
        heos_interpolation_cache={},
    )
    ok = np.isfinite(observed).all(axis=1)
    assert np.any(ok), "observed-mode inversion failed on every row"
    observed_error = np.abs(observed[ok] - dataset.target[local_indices[ok]]).mean()
    clean_error = np.abs(clean[ok] - dataset.target[local_indices[ok]]).mean()
    assert observed_error > clean_error
    split = int(len(one_x))
    one_x_ok = ok[:split]
    high_noise_ok = ok[split:]
    assert np.any(one_x_ok) and np.any(high_noise_ok)
    one_x_error = np.abs(
        observed[:split][one_x_ok] - dataset.target[local_indices[:split][one_x_ok]]
    ).mean()
    high_noise_error = np.abs(
        observed[split:][high_noise_ok]
        - dataset.target[local_indices[split:][high_noise_ok]]
    ).mean()
    assert high_noise_error > one_x_error, (
        "O-KIN-OBS must be sensitive to the registered noise scale "
        "(NOISE-5X rows must degrade more than NOISE-1X rows)"
    )


def test_dynamic_non_degenerate_scaled_threshold_rejects_high_noise_rows(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F4/F7: NOISE-10X 行的判据按 white_noise_scale 缩放（unscaled 通过、scaled 不通过）。"""
    data_config = frozen_configs["data"]
    experiment_config = frozen_configs["experiment"]
    noise_profiles = {str(p["noise_profile_id"]): p for p in data_config["noise_profiles"]}
    one_x_scale = float(noise_profiles["NOISE-1X"]["white_noise_scale"])
    assert float(noise_profiles["NOISE-10X"]["white_noise_scale"]) == pytest.approx(
        10.0 * one_x_scale
    )
    noise_base = np.asarray(
        experiment_config["pilot"]["observation_noise_std_by_sensor"], dtype=np.float64
    )
    base_rows = _rows_of(frozen_package, family="D-IID", split="train", limit=2)
    for other_family in FAMILIES:
        if other_family != "D-IID":
            base_rows += _rows_of(
                frozen_package, family=other_family, split="train", limit=1
            )
    dataset = _slice_dataset(frozen_package, base_rows)
    # 中幅 clean 信号：每通道 p2p ≈ 12×σ_base。对 NOISE-1X 行超过 5σ 判据
    # （active），对 NOISE-10X 行低于 50σ 判据（inactive）；transition 段
    # 方差比 = var(ramp)/σ² = 144/12 = 12 ≥ 4，不引入额外退化。
    # 数组在 __post_init__ 后只读，必须先改副本再构造子集。
    amplitude = 12.0 * noise_base  # sensor 顺序对齐 channel 顺序
    ramp = np.linspace(-0.5, 0.5, dataset.timesteps, dtype=np.float64)[None, None, :]
    shaped = amplitude[None, :, None] * ramp  # (1, 3, T)
    clean = np.broadcast_to(shaped, dataset.clean_device_signals.shape).astype(np.float32)
    records = [
        dict(record) | {"noise_profile_id": "NOISE-1X"} for record in dataset.records
    ]
    # D-IID 首行保持 NOISE-1X 作对照，第二行（索引 1）改为 NOISE-10X。
    records[1] = dict(records[1]) | {"noise_profile_id": "NOISE-10X"}
    subset = replace(
        dataset,
        clean_device_signals=clean,
        records=tuple(records),
    )
    result = _audit_dynamic_non_degenerate(subset, data_config, experiment_config)
    # 未缩放口径两行全 active（p2p=12σ > 5σ）；缩放后 10X 行失效。
    assert result["active_channel_fraction_unscaled"]["D-IID"] == pytest.approx(1.0)
    assert result["active_channel_fraction"]["D-IID"] == pytest.approx(0.5)
    assert result["status"] == "FAIL"


def test_dynamic_non_degenerate_development_passes_and_test_fails(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F4 真实回归：开发 split PASS、test split（剔除 pure）FAIL（§6.4 结论）。"""
    data_config = frozen_configs["data"]
    experiment_config = frozen_configs["experiment"]
    dev_rows = [i for i, r in enumerate(frozen_package.records) if r["split"] != "test"]
    development = _slice_dataset(frozen_package, dev_rows)
    dev_result = _audit_dynamic_non_degenerate(development, data_config, experiment_config)
    assert dev_result["status"] == "PASS"
    assert dev_result["global_active_channel_fraction"] >= 0.95
    test_result = _audit_dynamic_non_degenerate(
        frozen_package,
        data_config,
        experiment_config,
        subset_split="test",
        exclude_pure=True,
    )
    assert test_result["status"] == "FAIL"
    assert test_result["global_active_channel_fraction"] < 0.95
    assert test_result["global_active_channel_fraction_unscaled"] == pytest.approx(1.0)


def test_audit_jacobian_sampling_declares_scope_and_passes(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
) -> None:
    """F5/F7: Jacobian 审计跑通，口径为采样声明（sampled/total_rows 存在）。

    每 family × split 只放 1 行（18 样本），保证单测在 60 s 内；全量 216
    样本由 S4 的 pipeline 重跑承担。
    """
    rows: list[int] = []
    for family in FAMILIES:
        for split in DEVELOPMENT_SPLITS:
            rows.extend(_rows_of(frozen_package, family=family, split=split, limit=1))
    tiny = _slice_dataset(frozen_package, rows)
    result = _audit_jacobian(
        tiny,
        frozen_configs["data"],
        frozen_configs["eval"],
    )
    assert result["status"] == "PASS"
    assert result["sampled_row_counts"]["D-IID/train"] == 1
    assert result["total_row_counts"]["D-IID/train"] == 1
    assert result["sampled_rows"] == result["sample_count"]
    assert result["sampled_rows"] == len(rows)
    assert result["fixed_full_rank_fraction"] == 1.0
    assert all(result["checks"].values())
    for horizon in EARLY_HORIZONS:
        assert "fixed_condition_number_p95" in result["per_horizon"][horizon]
        assert "joint_target_condition_number_p95" not in result["per_horizon"][horizon]


def test_family_gate_failed_requirements_each_failure_branch(
    frozen_package: DynamicDataset,
    frozen_configs: dict[str, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F7: 门控组合——各失败分支正确进入 failed_requirements。"""
    data_config = frozen_configs["data"]
    eval_config = frozen_configs["eval"]
    experiment_config = frozen_configs["experiment"]
    a2h_config = frozen_configs["a2h"]
    gate_rows: list[int] = []
    for family in FAMILIES:
        for split in DEVELOPMENT_SPLITS:
            gate_rows.extend(_rows_of(frozen_package, family=family, split=split, limit=1))
    gate_dataset = _slice_dataset(frozen_package, gate_rows)

    def run_with(schema_ok, dynamic_ok, jacobian_ok, difficulty_cfg) -> dict[str, object]:
        monkeypatch.setattr(audit_module, "_audit_schema", lambda *a, **k: {"status": "PASS" if schema_ok else "FAIL"})
        monkeypatch.setattr(audit_module, "_audit_physics", lambda *a, **k: {"status": "PASS" if schema_ok else "FAIL"})
        monkeypatch.setattr(
            audit_module, "_audit_dynamic_non_degenerate",
            lambda *a, **k: {"status": "PASS" if dynamic_ok else "FAIL"},
        )
        monkeypatch.setattr(audit_module, "_audit_jacobian", lambda *a, **k: {"status": "PASS" if jacobian_ok else "FAIL"})
        monkeypatch.setattr(
            audit_module,
            "_audit_baselines",
            lambda *a, **k: _fake_baselines(eval_config, **difficulty_cfg),
        )
        monkeypatch.setattr(audit_module, "validate_a2_dynamic_records", lambda *a, **k: None)
        return run_a2_dynamic_difficulty_audit(
            gate_dataset,
            data_config=data_config,
            eval_config=eval_config,
            experiment_config=experiment_config,
            a2h_config=a2h_config,
        )

    # 全过 → DIFFICULTY_QUALIFIED，D-IID + 两个压力轴，无 failed requirements。
    result = run_with(True, True, True, {})
    assert result["status"] == "DIFFICULTY_QUALIFIED"
    assert result["failed_requirements"] == []
    assert set(result["eligible_dynamic_axes"]) == {"D-KINETICS", "D-PROTOCOL"}

    # D-IID 配对行不足 → D-IID FAILED，进入 failed_requirements。
    result = run_with(True, True, True, {"iid_paired_rows": 10})
    assert "D-IID" in result["failed_requirements"]
    assert result["family_gate"]["D-IID"]["status"] == "FAILED"

    # 仅 D-IID 合格 → 压力轴不足。
    result = run_with(True, True, True, {"pressure_axes": ("D-KINETICS",)})
    assert "two_independent_pressure_axes" in result["failed_requirements"]

    # 各全局审计失败。
    assert "schema_or_physics" in run_with(False, True, True, {})["failed_requirements"]
    assert "dynamic_non_degenerate" in run_with(True, False, True, {})["failed_requirements"]
    assert "jacobian" in run_with(True, True, False, {})["failed_requirements"]

    # headroom 全 None（全部反演失败）→ 家族失败。
    result = run_with(True, True, True, {"headroom": "none"})
    assert result["family_gate"]["D-IID"]["status"] == "FAILED"
    assert "D-IID" in result["failed_requirements"]

    monkeypatch.undo()


def _fake_baselines(
    eval_config: dict[str, object],
    *,
    iid_paired_rows: int | None = None,
    pressure_axes: tuple[str, ...] = ("D-KINETICS", "D-PROTOCOL"),
    headroom: str = "pass",
) -> dict[str, object]:
    """构造 _audit_baselines 的最小伪造输出（与 run_a2_dynamic_difficulty_audit 消费字段对齐）。"""
    difficulty_gate = eval_config["qualification_gates"]["dynamic_difficulty"]
    min_relative = float(difficulty_gate["min_relative_degradation"])
    min_headroom = float(difficulty_gate["min_oracle_headroom_vs_last"])
    qualified = ["D-IID", *pressure_axes]
    families: dict[str, object] = {}
    for family in FAMILIES:
        difficulty: dict[str, object] = {}
        for horizon in EARLY_HORIZONS:
            is_qualified = family in qualified
            if headroom == "none":
                headroom_value = None
            else:
                headroom_value = min_headroom + 0.1 if is_qualified else min_headroom - 0.1
            difficulty[horizon] = {
                "paired_row_count": (
                    iid_paired_rows
                    if iid_paired_rows is not None and family == "D-IID"
                    else 60
                ),
                "relative_degradation": (
                    min_relative + 0.1 if is_qualified else min_relative - 0.1
                ),
                "oracle_headroom_vs_last": headroom_value,
                "O-KIN-OBS_val_inversion_failure_fraction": 0.0,
            }
        families[family] = {
            "baseline_fit_status": "PASS",
            "oracle_fit_status": "PASS",
            "difficulty": difficulty,
        }
    return {
        "baseline_registry": ["B-LAST", "O-EQ", "O-KIN", "O-KIN-OBS"],
        "horizon_order": ["P005", "P015", "P030", "P060", "P120", "P150"],
        "late_reference_horizon": str(difficulty_gate["late_reference_horizon"]),
        "secondary_late_reference_horizon": str(
            difficulty_gate["secondary_late_reference_horizon"]
        ),
        "pairing": "valid_at_both_horizons",
        "families": families,
    }
