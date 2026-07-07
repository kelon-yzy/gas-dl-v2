"""SPXY + OOD 数据集划分单元测试。

覆盖 docs/掘进通风/spxy_split_implementation_plan.md §5.3 要求：
1. X/Y scaler 在距离计算前生效
2. _spxy_select_train 结果确定、无重复、比例正确
3. 向量化实现与小 N 朴素实现结果一致
4. extrapolation selector 不调用 SPXY，且能输出非零 OOD 诊断；退化时显式失败
5. train/val/test/extrapolation 四集合互斥且总数守恒
6. split_policy、spxy_alpha、extrapolation_strategy 写入 summary
7. 旧 random_mixture_id_split_v4 行为不变
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.sim.core.schema import SPLIT_NAMES
from tv3.sim.packaging.spxy_split import (
    SpxySplitError,
    _build_scaled_split_features,
    _kmeans_boundary_select,
    _lhs_boundary_select,
    _select_extrapolation_indices,
    _spxy_select_train,
    _spxy_select_train_naive,
    _y_margin_ood_select,
    build_lhs_stratified_split_with_summary,
    build_spxy_split_rows,
    build_spxy_split_with_summary,
)


def _make_test_data(n: int = 40, timesteps: int = 16, seed: int = 42):
    """构造测试数据：真实 conditions + 随机 arrays（SPXY 只用统计特征，不要求物理真实）。"""
    from tv3.sim.generation.tunnel_ventilation.conditions import (
        generate_tunnel_ventilation_condition_rows,
    )

    conditions = generate_tunnel_ventilation_condition_rows(n, seed=seed)
    rng = np.random.default_rng(seed)
    arrays = {
        "slow": rng.normal(0.0, 1.0, (n, timesteps, 7)).astype(np.float32),
        "ultrasonic_tof_s": rng.normal(1e-4, 1e-6, (n, timesteps)).astype(np.float32),
        "ultrasonic_sound_speed_m_per_s": rng.normal(340.0, 5.0, (n, timesteps)).astype(np.float32),
        "ultrasonic_alpha_true_npm": rng.normal(0.5, 0.1, (n, timesteps)).astype(np.float32),
    }
    labels = np.array(
        [[float(c["x_CO2"]), float(c["x_O2"]), float(c["x_N2"])] for c in conditions],
        dtype=np.float32,
    )
    return conditions, arrays, labels


# ---------------------------------------------------------------------------
# 1. X/Y scaler
# ---------------------------------------------------------------------------


def test_scaler_applied_to_X_and_Y():
    """X 各维均值/方差受控；Y 同理。距离不被单一大量级特征支配。"""
    conditions, arrays, labels = _make_test_data(n=40)
    X_scaled, y_scaled, y_basis = _build_scaled_split_features(conditions, arrays, labels)
    assert X_scaled.shape[0] == 40
    # 慢通道 35 维 + 超声 7 维 = 42 维
    assert X_scaled.shape[1] == 42
    assert np.allclose(X_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), 1.0, atol=1e-6)
    assert y_basis.shape == (40, 2)  # CO2, O2 两个自由度
    assert np.allclose(y_scaled.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(y_scaled.std(axis=0), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. SPXY 确定性 / 无重复 / 比例
# ---------------------------------------------------------------------------


def test_spxy_select_train_deterministic_unique_size():
    conditions, arrays, labels = _make_test_data(n=40)
    X_scaled, y_scaled, _ = _build_scaled_split_features(conditions, arrays, labels)
    train1, rem1 = _spxy_select_train(X_scaled, y_scaled, train_size=20, alpha=0.5)
    train2, rem2 = _spxy_select_train(X_scaled, y_scaled, train_size=20, alpha=0.5)
    assert np.array_equal(train1, train2)  # 确定性
    assert len(train1) == 20
    assert len(set(train1.tolist())) == 20  # 无重复
    assert len(rem1) == 20
    assert not (set(train1.tolist()) & set(rem1.tolist()))  # train/remainder 互斥


# ---------------------------------------------------------------------------
# 3. 向量化 vs 朴素实现一致
# ---------------------------------------------------------------------------


def test_vectorized_matches_naive():
    conditions, arrays, labels = _make_test_data(n=30)
    X_scaled, y_scaled, _ = _build_scaled_split_features(conditions, arrays, labels)
    train_v, rem_v = _spxy_select_train(X_scaled, y_scaled, train_size=15, alpha=0.5)
    train_n, rem_n = _spxy_select_train_naive(X_scaled, y_scaled, train_size=15, alpha=0.5)
    assert np.array_equal(train_v, train_n)
    assert np.array_equal(rem_v, rem_n)


def test_vectorized_matches_naive_alpha_zero():
    """α=0 退化为纯 Y 距离，向量化与朴素仍一致。"""
    conditions, arrays, labels = _make_test_data(n=30)
    X_scaled, y_scaled, _ = _build_scaled_split_features(conditions, arrays, labels)
    train_v, _ = _spxy_select_train(X_scaled, y_scaled, train_size=15, alpha=0.0)
    train_n, _ = _spxy_select_train_naive(X_scaled, y_scaled, train_size=15, alpha=0.0)
    assert np.array_equal(train_v, train_n)


# ---------------------------------------------------------------------------
# 4. extrapolation selector 独立于 SPXY / 退化显式失败
# ---------------------------------------------------------------------------


def test_extrapolation_selector_independent_of_spxy():
    """OOD selector 不调用 SPXY，直接基于 Y/X 几何选点。"""
    conditions, arrays, labels = _make_test_data(n=60)
    X_scaled, _, y_basis = _build_scaled_split_features(conditions, arrays, labels)
    ext_idx = _select_extrapolation_indices(
        y_basis=y_basis,
        X_scaled=X_scaled,
        n_ext=3,
        strategy="y_margin_ood",
        seed=42,
        interior_quantiles=(0.10, 0.90),
        n_bins=4,
        n_clusters=8,
    )
    assert len(ext_idx) == 3
    assert len(set(ext_idx.tolist())) == 3
    assert set(ext_idx.tolist()) <= set(range(60))


def test_y_margin_ood_degenerates_when_no_boundary():
    """interior_quantiles=(0,1) 时所有点都是 interior，boundary 为空 → 显式失败。"""
    conditions, arrays, labels = _make_test_data(n=60)
    _, _, y_basis = _build_scaled_split_features(conditions, arrays, labels)
    with pytest.raises(SpxySplitError, match="boundary candidates 不足"):
        _y_margin_ood_select(y_basis, n_ext=3, interior_quantiles=(0.0, 1.0))


def test_y_margin_ood_degenerates_when_interior_too_small():
    """interior 分位过窄导致 interior 样本 <4，无法构成 2D 凸包 → 显式失败。"""
    conditions, arrays, labels = _make_test_data(n=60)
    _, _, y_basis = _build_scaled_split_features(conditions, arrays, labels)
    with pytest.raises(SpxySplitError, match="interior domain 样本不足"):
        _y_margin_ood_select(y_basis, n_ext=3, interior_quantiles=(0.49, 0.51))


def test_lhs_boundary_select():
    conditions, arrays, labels = _make_test_data(n=60)
    _, _, y_basis = _build_scaled_split_features(conditions, arrays, labels)
    ext_idx = _lhs_boundary_select(y_basis, n_ext=3, n_bins=4, seed=42)
    assert len(ext_idx) == 3
    assert len(set(ext_idx.tolist())) == 3


def test_kmeans_boundary_select():
    conditions, arrays, labels = _make_test_data(n=60)
    X_scaled, _, _ = _build_scaled_split_features(conditions, arrays, labels)
    ext_idx = _kmeans_boundary_select(X_scaled, n_ext=3, n_clusters=8, seed=42)
    assert len(ext_idx) == 3
    assert len(set(ext_idx.tolist())) == 3


# ---------------------------------------------------------------------------
# 5. 四集合互斥且总数守恒
# ---------------------------------------------------------------------------


def test_spxy_four_splits_disjoint_and_conserved():
    conditions, arrays, labels = _make_test_data(n=40)
    rows, _ = build_spxy_split_with_summary(
        conditions, arrays, labels, seed=42, extrapolation_strategy="y_margin_ood"
    )
    all_seqs = [r["sequence_id"] for name in SPLIT_NAMES for r in rows[name]]
    assert len(all_seqs) == 40  # 总数守恒
    assert len(set(all_seqs)) == 40  # 互斥无重复
    # extrapolation 非空（n_ext = round(40*0.05) = 2）
    assert len(rows["extrapolation"]) == 2


def test_lhs_stratified_four_splits_disjoint_and_conserved():
    conditions, _, labels = _make_test_data(n=40)
    rows, summary = build_lhs_stratified_split_with_summary(conditions, labels, seed=42)
    assert summary["split_policy"] == "lhs_stratified_split_v1"
    all_seqs = [r["sequence_id"] for name in SPLIT_NAMES for r in rows[name]]
    assert len(all_seqs) == 40
    assert len(set(all_seqs)) == 40


# ---------------------------------------------------------------------------
# 6. summary 写入 policy / alpha / strategy
# ---------------------------------------------------------------------------


def test_summary_records_policy_alpha_strategy():
    conditions, arrays, labels = _make_test_data(n=40)
    rows, summary = build_spxy_split_with_summary(
        conditions, arrays, labels, seed=42, alpha=0.3, extrapolation_strategy="lhs_boundary"
    )
    assert summary["split_policy"] == "spxy_v1:lhs_boundary"
    assert summary["spxy_alpha"] == pytest.approx(0.3)
    assert summary["extrapolation_strategy"] == "lhs_boundary"
    assert "diagnostics" in summary
    # 诊断含每个 split 的 sequence_count 与 CO2/O2 范围
    for name in SPLIT_NAMES:
        assert "sequence_count" in summary["diagnostics"][name]
        if summary["diagnostics"][name]["sequence_count"] > 0:
            assert "co2_range" in summary["diagnostics"][name]
            assert "o2_range" in summary["diagnostics"][name]


def test_diagnostics_nn_to_train_distance_for_non_train_splits():
    """val/test/extrapolation 应输出到 train 的最近邻 Y 距离分布。"""
    conditions, arrays, labels = _make_test_data(n=60)
    rows, summary = build_spxy_split_with_summary(
        conditions, arrays, labels, seed=42, extrapolation_strategy="y_margin_ood"
    )
    diag = summary["diagnostics"]
    for name in ("val", "test", "extrapolation"):
        if diag[name]["sequence_count"] > 0:
            assert "nn_to_train_y_distance" in diag[name]
            assert diag[name]["nn_to_train_y_distance"]["max"] >= 0.0
    # train 含 X/Y pairwise 距离摘要
    assert "x_pairwise_distance" in diag["train"]
    assert "y_pairwise_distance" in diag["train"]


# ---------------------------------------------------------------------------
# 7. 旧 random split 行为不变
# ---------------------------------------------------------------------------


def test_random_split_unchanged_and_deterministic():
    from tv3.sim.packaging.splits import build_default_split_rows

    conditions, _, _ = _make_test_data(n=20)
    rows1 = build_default_split_rows(conditions, seed=42)
    rows2 = build_default_split_rows(conditions, seed=42)
    for name in SPLIT_NAMES:
        assert rows1[name] == rows2[name]  # seed 可复现
    total = sum(len(rows1[n]) for n in SPLIT_NAMES)
    assert total == 20


def test_build_spxy_split_rows_interface_matches_random():
    """build_spxy_split_rows 返回 dict[str, list[dict]]，与 build_default_split_rows 接口一致。"""
    conditions, arrays, labels = _make_test_data(n=40)
    rows = build_spxy_split_rows(
        conditions, arrays, labels, seed=42, extrapolation_strategy="y_margin_ood"
    )
    assert set(rows.keys()) == set(SPLIT_NAMES)
    for name in SPLIT_NAMES:
        for row in rows[name]:
            assert set(row.keys()) == {"sequence_id", "mixture_id"}


# ---------------------------------------------------------------------------
# α 边界值
# ---------------------------------------------------------------------------


def test_alpha_extremes_run_without_error():
    """α=1.0 退化为 KS（仅 X），α=0.0 退化为纯 Y，均应正常运行。"""
    conditions, arrays, labels = _make_test_data(n=40)
    for alpha in (0.0, 1.0):
        rows = build_spxy_split_rows(
            conditions, arrays, labels, seed=42, alpha=alpha, extrapolation_strategy="y_margin_ood"
        )
        total = sum(len(rows[n]) for n in SPLIT_NAMES)
        assert total == 40


# ---------------------------------------------------------------------------
# spec 校验
# ---------------------------------------------------------------------------


def test_validate_spec_rejects_spxy_without_ood():
    from tv3.sim.generation.tunnel_ventilation.benchmark import (
        TunnelVentilationBenchmarkGenerationSpec,
        _validate_spec,
    )

    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="t",
        sequence_count=8,
        seed=1,
        timesteps=8,
        split_strategy="spxy_v1",
        extrapolation_strategy="none",
    )
    with pytest.raises(ValueError, match="extrapolation_strategy"):
        _validate_spec(spec)


def test_validate_spec_rejects_ood_on_random():
    from tv3.sim.generation.tunnel_ventilation.benchmark import (
        TunnelVentilationBenchmarkGenerationSpec,
        _validate_spec,
    )

    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="t",
        sequence_count=8,
        seed=1,
        timesteps=8,
        split_strategy="random",
        extrapolation_strategy="y_margin_ood",
    )
    with pytest.raises(ValueError, match="仅在 split_strategy='spxy_v1' 下有效"):
        _validate_spec(spec)


# ---------------------------------------------------------------------------
# 端到端：spxy_v1 + y_margin_ood 生成完整 benchmark
# ---------------------------------------------------------------------------


def test_benchmark_spxy_v1_end_to_end(tmp_path):
    from tv3.sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-spxy-test",
        sequence_count=32,
        seed=20260704,
        timesteps=16,
        dt_s=0.5,
        storage="memmap",
        optical_absorption_backend="empirical_v1",
        workers=1,
        split_strategy="spxy_v1",
        spxy_alpha=0.5,
        extrapolation_strategy="y_margin_ood",
    )
    result = generate_tunnel_ventilation_benchmark_dataset(tmp_path, spec)
    summary_path = Path(result["output_dir"]) / "splits" / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["split_policy"] == "spxy_v1:y_margin_ood"
    assert summary["spxy_alpha"] == pytest.approx(0.5)
    assert summary["extrapolation_strategy"] == "y_margin_ood"
    assert "diagnostics" in summary
    # 四集合总数守恒
    total = sum(summary["splits"][n]["sequence_count"] for n in SPLIT_NAMES)
    assert total == 32


def test_benchmark_lhs_stratified_end_to_end(tmp_path):
    from tv3.sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-lhs-test",
        sequence_count=32,
        seed=20260704,
        timesteps=16,
        dt_s=0.5,
        storage="memmap",
        optical_absorption_backend="empirical_v1",
        workers=1,
        split_strategy="lhs_stratified_split_v1",
    )
    result = generate_tunnel_ventilation_benchmark_dataset(tmp_path, spec)
    summary_path = Path(result["output_dir"]) / "splits" / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["split_policy"] == "lhs_stratified_split_v1"
    total = sum(summary["splits"][n]["sequence_count"] for n in SPLIT_NAMES)
    assert total == 32


# ---------------------------------------------------------------------------
# 重算脚本：复用已有数据集，硬链接物理数据，只换 splits/
# ---------------------------------------------------------------------------


def test_recompute_split_end_to_end(tmp_path):
    """从 random 数据集重算 spxy / lhs 变体：物理数据硬链接（同 inode），splits 不同。"""
    import os
    import subprocess
    import sys

    from tv3.sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    project_root = Path(__file__).resolve().parents[1]
    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-src",
        sequence_count=32,
        seed=20260704,
        timesteps=16,
        dt_s=0.5,
        storage="memmap",
        optical_absorption_backend="empirical_v1",
        workers=1,
    )
    generate_tunnel_ventilation_benchmark_dataset(tmp_path, spec)
    source = tmp_path / "tv3-src"

    cases = [
        (["--split-strategy", "spxy_v1", "--extrapolation-strategy", "y_margin_ood"], "spxy_v1:y_margin_ood"),
        (["--split-strategy", "lhs_stratified_split_v1"], "lhs_stratified_split_v1"),
    ]
    for extra_args, expected_policy in cases:
        output = tmp_path / f"tv3-out-{expected_policy.replace(':', '_')}"
        result = subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "recompute_tv3_split.py"),
                "--source-dir", str(source),
                "--output-dir", str(output),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        assert result.returncode == 0, result.stderr
        summary = json.loads((output / "splits" / "split_summary.json").read_text(encoding="utf-8"))
        assert summary["split_policy"] == expected_policy
        total = sum(summary["splits"][n]["sequence_count"] for n in SPLIT_NAMES)
        assert total == 32
        # 物理数据硬链接（同一 inode，不占额外磁盘）
        src_inode = os.stat(source / "sequences" / "slow.npy").st_ino
        dst_inode = os.stat(output / "sequences" / "slow.npy").st_ino
        assert src_inode == dst_inode
        # splits 与 source（random）不同
        assert summary["split_policy"] != "random_mixture_id_split_v4"


def test_recompute_split_rejects_spxy_without_ood(tmp_path):
    """脚本校验：spxy_v1 必须配 OOD selector，否则非零退出。"""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "recompute_tv3_split.py"),
            "--source-dir", str(tmp_path / "nonexistent"),
            "--output-dir", str(tmp_path / "out"),
            "--split-strategy", "spxy_v1",
            "--extrapolation-strategy", "none",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    assert result.returncode != 0
