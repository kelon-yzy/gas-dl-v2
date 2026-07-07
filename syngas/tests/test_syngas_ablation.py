"""阶段 Ⅱ ablation 机制测试。

覆盖两类新机制：
- channel 子集选择（Ⅱ-1 CO 通道 ablation）：DL `V4BenchmarkDataset` 与
  Ridge `load_feature_matrix` 两条路径按保留通道名过滤 slow 列。
- crosstalk 开关透传（Ⅱ-2）：`enable_co_crosstalk` 从 benchmark spec 透传到
  物理层 `main_sensor_features`，并写入 manifest policy。

slow 通道顺序（syngas_schema.SLOW_CHANNELS）：
  V_NDIR_CH4(0) V_NDIR_CO2(1) V_NDIR_CO(2) V_TCS(3)
  T_C(4) P_MPa(5) H_RH(6) L_m(7) piston_position_m(8)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from sg.dl.cli import build_parser as build_dl_cli_parser, run as run_dl_cli
from sg.dl.data.dataset import V4BenchmarkDataset
from sg.ml.features import MLFeatureConfig, load_feature_matrix
from sg.sim.generation.syngas import (
    SyngasBenchmarkGenerationSpec,
    generate_syngas_benchmark_dataset,
)
from sg.sim.generation.syngas.acoustic_physics import main_sensor_features


# 组 B：去 CO NDIR（保留 8 通道）
DROP_CO_CHANNELS = (
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)
# 组 C：仅 CO 光学 + 环境（保留 6 通道）
CO_ONLY_CHANNELS = (
    "V_NDIR_CO",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)


def _generate_smoke(root: Path, slug: str, *, enable_co_crosstalk: bool = False) -> Path:
    generate_syngas_benchmark_dataset(
        root,
        SyngasBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=16,
            seed=20260626,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
            enable_co_crosstalk=enable_co_crosstalk,
        ),
    )
    return root / slug


@pytest.fixture(scope="module")
def smoke_dataset(tmp_path_factory) -> Path:
    """只读的 sg4 smoke 数据集，channel 子集测试共享，避免重复生成。"""
    root = tmp_path_factory.mktemp("sg4_ablation")
    return _generate_smoke(root, "sg4-ablation-smoke")


# ---------------------------------------------------------------------------
# Ⅱ-1 DL channel 子集
# ---------------------------------------------------------------------------


class TestDLChannelSubset:
    def test_default_keeps_all_9_channels(self, smoke_dataset: Path):
        ds = V4BenchmarkDataset(smoke_dataset, split="train", modalities=("slow",))
        x, _ = ds[0]  # NTC: (T, C)
        assert x.shape[1] == 9

    def test_drop_co_channel_gives_8(self, smoke_dataset: Path):
        ds = V4BenchmarkDataset(
            smoke_dataset, split="train", modalities=("slow",), slow_channels=DROP_CO_CHANNELS
        )
        x, _ = ds[0]
        assert x.shape[1] == 8

    def test_co_only_gives_6(self, smoke_dataset: Path):
        ds = V4BenchmarkDataset(
            smoke_dataset, split="train", modalities=("slow",), slow_channels=CO_ONLY_CHANNELS
        )
        x, _ = ds[0]
        assert x.shape[1] == 6

    def test_selected_columns_match_full(self, smoke_dataset: Path):
        """选列结果等于全通道取对应 index 列（V_NDIR_CO=2, T_C=4）。"""
        full = V4BenchmarkDataset(smoke_dataset, split="train", modalities=("slow",))
        sub = V4BenchmarkDataset(
            smoke_dataset, split="train", modalities=("slow",), slow_channels=("V_NDIR_CO", "T_C")
        )
        x_full, _ = full[0]
        x_sub, _ = sub[0]
        np.testing.assert_allclose(x_sub[:, 0].numpy(), x_full[:, 2].numpy())
        np.testing.assert_allclose(x_sub[:, 1].numpy(), x_full[:, 4].numpy())

    def test_order_follows_requested_not_schema(self, smoke_dataset: Path):
        """列顺序遵循请求顺序，不是 schema 原顺序。"""
        sub = V4BenchmarkDataset(
            smoke_dataset, split="train", modalities=("slow",), slow_channels=("T_C", "V_NDIR_CO")
        )
        full = V4BenchmarkDataset(smoke_dataset, split="train", modalities=("slow",))
        x_sub, _ = sub[0]
        x_full, _ = full[0]
        np.testing.assert_allclose(x_sub[:, 0].numpy(), x_full[:, 4].numpy())  # T_C first
        np.testing.assert_allclose(x_sub[:, 1].numpy(), x_full[:, 2].numpy())  # V_NDIR_CO second

    def test_unknown_channel_raises(self, smoke_dataset: Path):
        with pytest.raises(ValueError, match="Unknown slow channel"):
            V4BenchmarkDataset(
                smoke_dataset, split="train", modalities=("slow",), slow_channels=("V_NOPE",)
            )

    def test_cli_drop_co_in_channels_8(self, smoke_dataset: Path, tmp_path: Path):
        """CLI 端到端：--slow-channels 去 CO 后 in_channels 自动推断为 8。"""
        output_dir = tmp_path / "runs" / "dropco"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir", str(smoke_dataset),
                "--output-dir", str(output_dir),
                "--model", "cnn1d",
                "--model-kwargs", '{"hidden_channels":[4],"kernel_size":3,"dropout":0.0}',
                "--loss", "mse",
                "--slow-channels", ",".join(DROP_CO_CHANNELS),
                "--epochs", "1",
                "--batch-size", "4",
                "--eval-splits", "val",
            ]
        )
        payload = run_dl_cli(args)
        assert payload["model_config"]["in_channels"] == 8
        assert payload["model_config"]["out_dim"] == 4
        assert payload["slow_channels"] == list(DROP_CO_CHANNELS)


# ---------------------------------------------------------------------------
# Ⅱ-1 Ridge channel 子集
# ---------------------------------------------------------------------------


class TestRidgeChannelSubset:
    def test_full_feature_count(self, smoke_dataset: Path):
        m = load_feature_matrix(smoke_dataset, split="train", config=MLFeatureConfig(modalities=("slow",)))
        assert m.x.shape[1] == 63  # 9 channels × 7 stats

    def test_drop_co_feature_count(self, smoke_dataset: Path):
        m = load_feature_matrix(
            smoke_dataset,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), slow_channels=DROP_CO_CHANNELS),
        )
        assert m.x.shape[1] == 56  # 8 channels × 7 stats
        assert all(":V_NDIR_CO:" not in name for name in m.feature_names)

    def test_co_only_feature_count(self, smoke_dataset: Path):
        m = load_feature_matrix(
            smoke_dataset,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), slow_channels=CO_ONLY_CHANNELS),
        )
        assert m.x.shape[1] == 42  # 6 channels × 7 stats
        assert any(":V_NDIR_CO:" in name for name in m.feature_names)
        assert all(":V_NDIR_CH4:" not in name for name in m.feature_names)

    def test_subset_columns_match_full(self, smoke_dataset: Path):
        """子集特征值等于全特征里对应通道的列。"""
        full = load_feature_matrix(smoke_dataset, split="train", config=MLFeatureConfig(modalities=("slow",)))
        sub = load_feature_matrix(
            smoke_dataset,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), slow_channels=("V_NDIR_CO",)),
        )
        # mean 统计量块内，V_NDIR_CO 是全特征第 2 列（index 2）
        full_idx = full.feature_names.index("slow:V_NDIR_CO:mean")
        sub_idx = sub.feature_names.index("slow:V_NDIR_CO:mean")
        # float32 下 numpy 对 (N,T,9) 与 (N,T,1) 的 mean(axis=1) 累加路径不同，
        # 末位有 ~5e-7 舍入差异；放宽容差，验证目标是"选对了列"而非 bit-exact。
        np.testing.assert_allclose(sub.x[:, sub_idx], full.x[:, full_idx], rtol=1e-5)

    def test_unknown_channel_raises(self, smoke_dataset: Path):
        with pytest.raises(ValueError, match="Unknown slow channel"):
            load_feature_matrix(
                smoke_dataset,
                split="train",
                config=MLFeatureConfig(modalities=("slow",), slow_channels=("V_NOPE",)),
            )


# ---------------------------------------------------------------------------
# Ⅱ-2 crosstalk 透传（物理层）
# ---------------------------------------------------------------------------


@pytest.fixture
def high_co2_condition() -> dict[str, str]:
    return {
        "x_H2": "20.0",
        "x_CH4": "2.0",
        "x_CO2": "30.0",
        "x_N2": "3.0",
        "x_CO": "45.0",
        "T_C": "25.0",
        "P_MPa": "0.5",
        "H_RH": "50.0",
        "L_m": "1.0",
    }


class TestCrosstalkPassthrough:
    def test_co_observed_pure_in_step1(self, high_co2_condition):
        res = main_sensor_features(high_co2_condition, random.Random(1), enable_co_crosstalk=False)
        assert abs(res["absorption_co_observed"] - res["absorption_co_true"]) < 1e-9

    def test_co_observed_receives_co2_leak_in_step2(self, high_co2_condition):
        """Step 2：CO 通道 observed > true（CO2 泄漏进来）。"""
        res = main_sensor_features(high_co2_condition, random.Random(1), enable_co_crosstalk=True)
        assert res["absorption_co_observed"] > res["absorption_co_true"]

    def test_ch4_channel_unaffected_by_co_crosstalk(self, high_co2_condition):
        """CH4 通道 observed 不随 CO 串扰开关变化（ch4_channel_co_response=0）。"""
        off = main_sensor_features(high_co2_condition, random.Random(1), enable_co_crosstalk=False)
        on = main_sensor_features(high_co2_condition, random.Random(1), enable_co_crosstalk=True)
        assert abs(off["absorption_ch4_observed"] - on["absorption_ch4_observed"]) < 1e-9


# ---------------------------------------------------------------------------
# Ⅱ-2 crosstalk 透传（benchmark 生成层）
# ---------------------------------------------------------------------------


class TestCrosstalkBenchmark:
    def test_manifest_policy_step1_default(self, tmp_path: Path):
        ds_dir = _generate_smoke(tmp_path, "sg4-ct-step1")
        spec_json = json.loads((ds_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))
        assert spec_json["optical_crosstalk_policy"] == "syngas_empirical_3x3_step1_co_pure"
        assert spec_json["enable_co_crosstalk"] is False

    def test_manifest_policy_step2_when_enabled(self, tmp_path: Path):
        ds_dir = _generate_smoke(tmp_path, "sg4-ct-step2", enable_co_crosstalk=True)
        spec_json = json.loads((ds_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))
        assert spec_json["optical_crosstalk_policy"] == "syngas_empirical_3x3_step2_co2_co_crosstalk"
        assert spec_json["enable_co_crosstalk"] is True

    def test_co_channel_data_differs_env_channel_same(self, tmp_path: Path):
        """同 seed 下，step2 改变 V_NDIR_CO(2) 数据，但环境通道 T_C(4) 不变。"""
        d1 = _generate_smoke(tmp_path, "sg4-ct-off", enable_co_crosstalk=False)
        d2 = _generate_smoke(tmp_path, "sg4-ct-on", enable_co_crosstalk=True)
        slow1 = np.load(d1 / "sequences" / "slow.npy")
        slow2 = np.load(d2 / "sequences" / "slow.npy")
        assert not np.allclose(slow1[:, :, 2], slow2[:, :, 2])  # V_NDIR_CO 受串扰
        np.testing.assert_allclose(slow1[:, :, 4], slow2[:, :, 4])  # T_C 不受影响
