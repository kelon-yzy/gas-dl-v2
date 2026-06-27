"""合成气场景下 DL 训练管线的端到端测试。

验证：
- CLI 能直接消费 sg4-smoke benchmark（manifest.composition_scheme="syngas"）
- in_channels == 9（含 V_NDIR_CO）
- out_dim == 4（4 列预测目标，不含 x_N2）
- component_metrics 键集合 == ("x_H2", "x_CH4", "x_CO2", "x_CO")
- conditional_metrics 按 co_bins 分箱（不是 n2_bins）
- val_sum_abs_error 为 None（syngas 场景不计算）
- 闭包类 loss 在 syngas 下被拒绝
- hydrogen_ng 路径不受影响
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dl.cli import build_parser as build_dl_cli_parser, run as run_dl_cli
from dl.training.losses import (
    COMPOSITIONAL_MSE_LOSS,
    FREE_COMPONENT_MSE_LOSS,
    ILR_MSE_LOSS,
    WEIGHTED_COMPONENT_MSE_LOSS,
    WEIGHTED_FREE_COMPONENT_MSE_LOSS,
    validate_loss_composition_scheme,
)
from sim.core.syngas_schema import COMPONENT_FIELDS as SG_COMPONENT_FIELDS
from sim.generation.syngas import (
    SyngasBenchmarkGenerationSpec,
    generate_syngas_benchmark_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_syngas_smoke_dataset(tmp_path: Path, slug: str = "sg4-dl-smoke", sequences: int = 16) -> Path:
    generate_syngas_benchmark_dataset(
        tmp_path,
        SyngasBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260626,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


# ---------------------------------------------------------------------------
# Loss 校验
# ---------------------------------------------------------------------------


class TestLossCompositionSchemeValidation:
    def test_closure_losses_rejected_for_syngas(self):
        for loss_name in (
            COMPOSITIONAL_MSE_LOSS,
            ILR_MSE_LOSS,
            FREE_COMPONENT_MSE_LOSS,
            WEIGHTED_FREE_COMPONENT_MSE_LOSS,
        ):
            with pytest.raises(ValueError, match="closure"):
                validate_loss_composition_scheme(loss_name, "syngas")

    def test_open_losses_accepted_for_syngas(self):
        for loss_name in (
            "mse",
            "mae",
            "smooth_l1",
            "huber",
            WEIGHTED_COMPONENT_MSE_LOSS,
        ):
            # 不应抛错
            validate_loss_composition_scheme(loss_name, "syngas")

    def test_all_losses_accepted_for_hydrogen_ng(self):
        """hydrogen_ng 场景下闭包类 loss 仍然合法。"""
        for loss_name in (
            "mse",
            COMPOSITIONAL_MSE_LOSS,
            FREE_COMPONENT_MSE_LOSS,
            WEIGHTED_COMPONENT_MSE_LOSS,
        ):
            validate_loss_composition_scheme(loss_name, "hydrogen_ng")


# ---------------------------------------------------------------------------
# CLI 端到端
# ---------------------------------------------------------------------------


class TestDLCliSyngas:
    def test_cli_trains_syngas_benchmark(self, tmp_path: Path):
        dataset_dir = _make_syngas_smoke_dataset(tmp_path)
        output_dir = tmp_path / "runs" / "syngas-cnn1d"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "cnn1d",
                "--model-kwargs",
                '{"hidden_channels":[4],"kernel_size":3,"dropout":0.0}',
                "--loss",
                "mse",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--lr",
                "0.001",
                "--eval-splits",
                "val,test",
            ]
        )

        payload = run_dl_cli(args)

        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "checkpoint.pt").is_file()
        # 9 个慢通道
        assert payload["model_config"]["in_channels"] == 9
        # 4 列预测目标
        assert payload["model_config"]["out_dim"] == 4
        assert set(payload["evaluations"]) == {"val", "test"}
        # component_metrics 键来自 syngas 标签
        val_comp_keys = set(payload["evaluations"]["val"]["component_metrics"])
        assert val_comp_keys == set(SG_COMPONENT_FIELDS)
        # conditional_metrics 按 CO 分箱
        cond_keys = set(payload["evaluations"]["val"]["conditional_metrics"])
        assert cond_keys == {"co_bins", "ch4_bins"}
        # syngas 不计算 sum_abs_error
        assert payload["evaluations"]["val"]["sum_abs_error"] is None

    def test_cli_rejects_closure_loss_on_syngas(self, tmp_path: Path):
        dataset_dir = _make_syngas_smoke_dataset(tmp_path, slug="sg4-loss-reject")
        output_dir = tmp_path / "runs" / "syngas-bad-loss"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "cnn1d",
                "--model-kwargs",
                '{"hidden_channels":[4],"kernel_size":3,"dropout":0.0}',
                "--loss",
                FREE_COMPONENT_MSE_LOSS,
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--eval-splits",
                "val",
            ]
        )
        with pytest.raises(ValueError, match="closure"):
            run_dl_cli(args)

    def test_cli_weighted_component_mse_on_syngas(self, tmp_path: Path):
        """weighted_component_mse 在 syngas 上应能跑通，按 4 列方差倒数加权。

        weighted loss 复杂 config 通过 --config JSON 文件传入（--loss 是 choices）。
        """
        dataset_dir = _make_syngas_smoke_dataset(tmp_path, slug="sg4-weighted")
        output_dir = tmp_path / "runs" / "syngas-weighted"
        config_path = tmp_path / "sg4_weighted_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "output_dir": str(output_dir),
                    "model": "cnn1d",
                    "model_kwargs": {"hidden_channels": [4], "kernel_size": 3, "dropout": 0.0},
                    "loss": {
                        "name": WEIGHTED_COMPONENT_MSE_LOSS,
                        "weighting": "inverse_train_var",
                        "component_count": 4,
                    },
                    "epochs": 1,
                    "batch_size": 4,
                    "eval_splits": "val",
                }
            ),
            encoding="utf-8",
        )
        parser = build_dl_cli_parser()
        args = parser.parse_args(["--config", str(config_path)])
        payload = run_dl_cli(args)
        assert payload["model_config"]["out_dim"] == 4
        assert payload["evaluations"]["val"]["loss"] >= 0.0


# ---------------------------------------------------------------------------
# Trainer 单元
# ---------------------------------------------------------------------------


class TestTrainerCompositionScheme:
    def test_trainer_rejects_target_transform_on_syngas(self):
        """syngas 不允许 ILR/ALR target_transform（闭包假设不成立）。"""
        import torch
        from torch import nn
        from common.composition import TargetTransformSpec
        from dl.training.trainer import Trainer

        model = nn.Linear(3, 4)
        opt = torch.optim.Adam(model.parameters())
        spec = TargetTransformSpec(name="ilr_n2_first", epsilon=1e-4)
        with pytest.raises(ValueError, match="syngas"):
            Trainer(
                model=model,
                optimizer=opt,
                loss_fn=nn.MSELoss(),
                device="cpu",
                target_transform=spec,
                composition_scheme="syngas",
            )

    def test_trainer_rejects_unknown_scheme(self):
        import torch
        from torch import nn
        from dl.training.trainer import Trainer

        model = nn.Linear(3, 4)
        opt = torch.optim.Adam(model.parameters())
        with pytest.raises(ValueError, match="composition_scheme"):
            Trainer(
                model=model,
                optimizer=opt,
                loss_fn=nn.MSELoss(),
                device="cpu",
                composition_scheme="bogus_scheme",
            )


# ---------------------------------------------------------------------------
# 隔离性
# ---------------------------------------------------------------------------


class TestHydrogenNgUnaffected:
    def test_legacy_hg_benchmark_still_loads_with_default_scheme(self, tmp_path: Path):
        """hydrogen_ng 数据集即使 manifest 无 composition_scheme 字段也能 fallback。"""
        from sim.generation.benchmark import (
            BenchmarkGenerationSpec,
            generate_benchmark_dataset,
        )

        generate_benchmark_dataset(
            tmp_path,
            BenchmarkGenerationSpec(
                dataset_slug="hg-dl-smoke",
                sequence_count=8,
                seed=7,
                timesteps=16,
                storage="npz",
                optical_absorption_backend="empirical_v1",
            ),
        )
        dataset_dir = tmp_path / "hg-dl-smoke"
        output_dir = tmp_path / "runs" / "hg-baseline"
        parser = build_dl_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(output_dir),
                "--model",
                "cnn1d",
                "--model-kwargs",
                '{"hidden_channels":[4],"kernel_size":3,"dropout":0.0}',
                "--loss",
                "mse",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--eval-splits",
                "val",
            ]
        )
        payload = run_dl_cli(args)
        # hg：8 通道、按 N2 分箱、计算 sum_abs_error
        assert payload["model_config"]["in_channels"] == 8
        assert payload["model_config"]["out_dim"] == 4
        cond_keys = set(payload["evaluations"]["val"]["conditional_metrics"])
        assert cond_keys == {"n2_bins", "ch4_bins"}
        assert payload["evaluations"]["val"]["sum_abs_error"] is not None
