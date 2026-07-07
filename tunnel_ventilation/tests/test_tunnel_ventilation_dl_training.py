"""掘进通风场景下 DL 训练管线的端到端测试。

验证：
- CLI 能直接消费 tv3-smoke benchmark（manifest.composition_scheme="tunnel_ventilation"）
- in_channels == 7（无 V_NDIR_CH4 / V_NDIR_CO）
- out_dim == 3（3 列预测目标：x_CO2, x_O2, x_N2）
- component_metrics 键集合 == ("x_CO2", "x_O2", "x_N2")
- conditional_metrics 按 o2_bins / co2_bins 分箱（tv3 无 x_CH4）
- sum_abs_error 不为 None（tv3 数据 sum=100% 闭包，可计算监控值）
- 闭包类 loss 在 tv3 下被拒绝
- target_transform 在 tv3 下被拒绝
- hydrogen_ng / syngas 路径不受影响
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tv3.dl.cli import build_parser as build_dl_cli_parser, run as run_dl_cli
from tv3.dl.training.losses import (
    COMPOSITIONAL_MSE_LOSS,
    FREE_COMPONENT_MSE_LOSS,
    ILR_MSE_LOSS,
    WEIGHTED_COMPONENT_MSE_LOSS,
    WEIGHTED_FREE_COMPONENT_MSE_LOSS,
    validate_loss_composition_scheme,
)
from tv3.sim.core.tunnel_ventilation_schema import COMPONENT_FIELDS as TV_COMPONENT_FIELDS
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-dl-smoke", sequences: int = 16) -> Path:
    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260704,
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
    def test_closure_losses_rejected_for_tunnel_ventilation(self):
        for loss_name in (
            COMPOSITIONAL_MSE_LOSS,
            ILR_MSE_LOSS,
            FREE_COMPONENT_MSE_LOSS,
            WEIGHTED_FREE_COMPONENT_MSE_LOSS,
        ):
            with pytest.raises(ValueError, match="closure"):
                validate_loss_composition_scheme(loss_name, "tunnel_ventilation")

    def test_open_losses_accepted_for_tunnel_ventilation(self):
        for loss_name in (
            "mse",
            "mae",
            "smooth_l1",
            "huber",
            WEIGHTED_COMPONENT_MSE_LOSS,
        ):
            # 不应抛错
            validate_loss_composition_scheme(loss_name, "tunnel_ventilation")

    def test_closure_losses_still_rejected_for_syngas(self):
        """syngas 闭包 loss 拒绝未被破坏。"""
        for loss_name in (COMPOSITIONAL_MSE_LOSS, FREE_COMPONENT_MSE_LOSS):
            with pytest.raises(ValueError, match="closure"):
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


class TestTv3BaselineScript:
    def test_tv3_baseline_script_seed_plan_and_summary_excludes_failures(self, tmp_path: Path):
        import importlib.util

        script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_tv3_baseline.py"
        spec = importlib.util.spec_from_file_location("run_tv3_baseline_test", script_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ok_metrics = tmp_path / "ok_metrics.json"
        fail_metrics = tmp_path / "fail_metrics.json"
        ok_metrics.write_text(
            json.dumps(
                {
                    "evaluations": {
                        "val": {
                            "metrics": {"mae": 1.0},
                            "component_metrics": {"x_CO2": {"mae": 0.1}},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        fail_metrics.write_text(
            json.dumps(
                {
                    "evaluations": {
                        "val": {
                            "metrics": {"mae": 9.0},
                            "component_metrics": {"x_CO2": {"mae": 9.0}},
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        records = [
            {"model": "cnn1d", "seed": 42, "status": "ok", "metrics_path": str(ok_metrics)},
            {"model": "tcn", "seed": 123, "status": "fail", "metrics_path": str(fail_metrics)},
        ]

        summary = module._summarize_from_disk(records)

        assert module.SEEDS == (42, 123, 456)
        assert set(summary) == {"cnn1d"}
        assert summary["cnn1d"]["seed42"]["val"]["metrics"] == {"mae": 1.0}

# ---------------------------------------------------------------------------
# CLI 端到端
# ---------------------------------------------------------------------------


class TestDLCliTunnelVentilation:
    def test_cli_trains_tv3_benchmark(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path)
        output_dir = tmp_path / "runs" / "tv3-cnn1d"
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
        # 7 个慢通道（无 V_NDIR_CH4 / V_NDIR_CO）
        assert payload["model_config"]["in_channels"] == 7
        # 3 列预测目标
        assert payload["model_config"]["out_dim"] == 3
        assert set(payload["evaluations"]) == {"val", "test"}
        # component_metrics 键来自 tv3 标签
        val_comp_keys = set(payload["evaluations"]["val"]["component_metrics"])
        assert val_comp_keys == set(TV_COMPONENT_FIELDS)
        # conditional_metrics 按 O2 / CO2 分箱（tv3 无 x_CH4）
        cond_keys = set(payload["evaluations"]["val"]["conditional_metrics"])
        assert cond_keys == {"o2_bins", "co2_bins"}
        # tv3 数据 sum=100% 闭包，sum_abs_error 可计算
        assert payload["evaluations"]["val"]["sum_abs_error"] is not None

    def test_cli_rejects_closure_loss_on_tv3(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-loss-reject")
        output_dir = tmp_path / "runs" / "tv3-bad-loss"
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

    def test_cli_weighted_component_mse_on_tv3(self, tmp_path: Path):
        """weighted_component_mse 在 tv3 上应能跑通，按 3 列方差倒数加权。

        weighted loss 复杂 config 通过 --config JSON 文件传入（--loss 是 choices）。
        """
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-weighted")
        output_dir = tmp_path / "runs" / "tv3-weighted"
        config_path = tmp_path / "tv3_weighted_config.json"
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
                        "component_count": 3,
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
        assert payload["model_config"]["out_dim"] == 3
        assert payload["evaluations"]["val"]["loss"] >= 0.0


# ---------------------------------------------------------------------------
# Trainer 单元
# ---------------------------------------------------------------------------


class TestTrainerCompositionScheme:
    def test_trainer_rejects_target_transform_on_tunnel_ventilation(self):
        """tv3 不允许 ILR/ALR target_transform（不使用闭包残差头）。"""
        import torch
        from torch import nn
        from tv3.common.composition import TargetTransformSpec
        from tv3.dl.training.trainer import Trainer

        model = nn.Linear(3, 3)
        opt = torch.optim.Adam(model.parameters())
        spec = TargetTransformSpec(name="ilr_n2_first", epsilon=1e-4)
        with pytest.raises(ValueError, match="tunnel_ventilation"):
            Trainer(
                model=model,
                optimizer=opt,
                loss_fn=nn.MSELoss(),
                device="cpu",
                target_transform=spec,
                composition_scheme="tunnel_ventilation",
            )

    def test_trainer_accepts_tunnel_ventilation_scheme(self):
        """tv3 scheme 应被接受（无 target_transform 时）。"""
        import torch
        from torch import nn
        from tv3.dl.training.trainer import Trainer

        model = nn.Linear(3, 3)
        opt = torch.optim.Adam(model.parameters())
        trainer = Trainer(
            model=model,
            optimizer=opt,
            loss_fn=nn.MSELoss(),
            device="cpu",
            composition_scheme="tunnel_ventilation",
        )
        assert trainer.composition_scheme == "tunnel_ventilation"

    def test_trainer_rejects_target_transform_on_syngas(self):
        """syngas target_transform 拒绝未被破坏。"""
        import torch
        from torch import nn
        from tv3.common.composition import TargetTransformSpec
        from tv3.dl.training.trainer import Trainer

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
        from tv3.dl.training.trainer import Trainer

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
        from tv3.sim.generation.benchmark import (
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
