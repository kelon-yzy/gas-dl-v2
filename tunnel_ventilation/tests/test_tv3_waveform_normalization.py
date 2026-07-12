"""tv3 raw 波形归一化与 fusion 改动测试（waveform_normalization_plan 落地）。

覆盖：
- 层 1：dataset per-timestep z-score 数值正确性、CLI normalize_waveforms 端到端传参
- 层 2：CNN1DTCNFusionRegressor fusion_layer_norm 开关与 forward 通路
- 层 3：FiLMModulation 恒等初始化、GatedFusion 通路、fusion_mode 分支与非法值拒绝
- losses.py：weighted_component_mse weighting="fixed" + loss_weights 解析与冲突校验
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tv3.dl.cli import (
    _build_augment_config,
    _resolve_args,
    build_parser as build_dl_cli_parser,
    run as run_dl_cli,
)
from tv3.dl.data.dataset import V4BenchmarkDataset
from tv3.dl.models.cnn1d_tcn_fusion import (
    CNN1DTCNFusionRegressor,
    FiLMModulation,
    GatedFusion,
)
from tv3.dl.training.losses import WEIGHTED_COMPONENT_MSE_LOSS, build_loss
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-norm-smoke", sequences: int = 16) -> Path:
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


def _small_fusion_kwargs(**overrides: object) -> dict[str, object]:
    """小尺寸 cnn1d_tcn_fusion 构造参数，匹配 tv3 raw3 + 无 fiber。"""
    kwargs: dict[str, object] = {
        "in_channels": 5007,  # 7 slow + 5000 ultrasonic
        "out_dim": 3,
        "slow_channels": 7,
        "ultrasonic_channels": 5000,
        "fiber_mic_channels": 0,
        "waveform_embedding_dim": 8,
        "waveform_adc_scale": 1.0,
        "acoustic_channels": [4],
        "acoustic_kernel_size": 3,
        "slow_hidden_dim": 8,
        "slow_embedding_dim": 8,
        "tcn_channels": [4],
        "tcn_kernel_size": 3,
        "shared_hidden_dims": [8, 4],
        "output_mode": "raw3",
    }
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# 层 1：z-score 数值正确性
# ---------------------------------------------------------------------------


class TestWaveformZScore:
    def test_normalize_waveforms_produces_unit_variance_per_timestep(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path)
        dataset = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("ultrasonic",),
            dequantize_waveforms=True,
            normalize_waveforms=True,
            lazy=True,
        )
        x, _ = dataset[0]
        assert x.shape[1] == 5000
        # 每帧 z-score 后 mean≈0、std≈1
        per_frame_mean = x.numpy().mean(axis=-1)
        per_frame_std = x.numpy().std(axis=-1)
        assert np.allclose(per_frame_mean, 0.0, atol=1e-4)
        assert np.allclose(per_frame_std, 1.0, atol=1e-3)

    def test_normalize_off_preserves_dequantized_voltage(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-norm-off")
        on_ds = V4BenchmarkDataset(
            dataset_dir, split="train", modalities=("ultrasonic",),
            dequantize_waveforms=True, normalize_waveforms=True, lazy=True,
        )
        off_ds = V4BenchmarkDataset(
            dataset_dir, split="train", modalities=("ultrasonic",),
            dequantize_waveforms=True, normalize_waveforms=False, lazy=True,
        )
        x_on, _ = on_ds[0]
        x_off, _ = off_ds[0]
        # 归一化前后不应相等（z-score 改变了尺度）
        assert not np.allclose(x_on.numpy(), x_off.numpy())
        # 关闭时输出不是 unit variance：dequantize 后原始电压为毫伏级（std 远小于 1），
        # 这正是 v2 失效的尺度失衡证据；开启 z-score 后才对齐到 unit variance。
        assert not np.allclose(x_off.numpy().std(axis=-1), 1.0, atol=0.1)

    def test_waveform_stats_preserve_pre_normalization_amplitude(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-wave-stats")
        dataset = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic"),
            dequantize_waveforms=True,
            normalize_waveforms=True,
            waveform_stats_features=("log_std", "log_max_abs"),
            lazy=True,
        )
        x, _ = dataset[0]
        assert x.shape[1] == 5009  # 7 slow + 2 waveform stats + 5000 waveform samples
        stats = x.numpy()[:, 7:9]
        waveform = x.numpy()[:, 9:]
        assert np.isfinite(stats).all()
        assert np.any(stats > 0.0)
        assert np.allclose(waveform.mean(axis=-1), 0.0, atol=1e-4)
        assert np.allclose(waveform.std(axis=-1), 1.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 层 2：fusion_layer_norm
# ---------------------------------------------------------------------------


class TestFusionLayerNorm:
    def test_layer_norm_modules_present_when_enabled(self):
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_layer_norm=True))
        assert model.ultrasonic_norm is not None
        assert model.slow_norm is not None
        assert model.fiber_mic_norm is None  # fiber_mic_channels=0

    def test_layer_norm_absent_by_default(self):
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs())
        assert model.fusion_layer_norm is False
        assert model.ultrasonic_norm is None
        assert model.slow_norm is None

    def test_forward_with_layer_norm_runs(self):
        torch.manual_seed(0)
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_layer_norm=True))
        x = torch.randn(2, 5, 5007)
        out = model(x)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 层 3：FiLM + GatedFusion
# ---------------------------------------------------------------------------


class TestFiLMGatedFusion:
    def test_film_identity_init_after_apply(self):
        """FiLM 恒等初始化需在父模型 _init_weights 之后存活：γ=1, β=0。"""
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_layer_norm=True, fusion_mode="film_gate"))
        film = model.film_modulation
        assert film is not None
        feature_dim = film.feature_dim
        # weight 应全 0（被 apply 重置后由 reset_identity 重建）
        assert torch.all(film.proj.weight == 0.0)
        # bias 前半段 γ=1，后半段 β=0
        assert torch.allclose(film.proj.bias[:feature_dim], torch.ones(feature_dim))
        assert torch.allclose(film.proj.bias[feature_dim:], torch.zeros(feature_dim))

    def test_film_is_identity_at_init(self):
        """恒等初始化下 FiLM(ultrasonic, slow) == ultrasonic。"""
        torch.manual_seed(0)
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_layer_norm=True, fusion_mode="film_gate"))
        ultrasonic_emb = torch.randn(2, 5, 8)
        slow_emb = torch.randn(2, 5, 8)
        out = model.film_modulation(ultrasonic_emb, slow_emb)
        assert torch.allclose(out, ultrasonic_emb, atol=1e-6)

    def test_film_gate_requires_layer_norm(self):
        with pytest.raises(ValueError, match="fusion_layer_norm"):
            CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_mode="film_gate"))

    def test_film_gate_forward_runs(self):
        torch.manual_seed(0)
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_layer_norm=True, fusion_mode="film_gate"))
        x = torch.randn(2, 5, 5007)
        out = model(x)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_film_gate_with_fiber_mic(self):
        """film_gate 分支在有 fiber_mic 时也应跑通（三分支 GatedFusion）。"""
        torch.manual_seed(0)
        kwargs = _small_fusion_kwargs(
            fusion_layer_norm=True,
            fusion_mode="film_gate",
            fiber_mic_channels=5000,
            in_channels=10007,  # 7 + 5000 + 5000
        )
        model = CNN1DTCNFusionRegressor(**kwargs)
        x = torch.randn(2, 5, 10007)
        out = model(x)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_invalid_fusion_mode_rejected(self):
        with pytest.raises(ValueError, match="fusion_mode"):
            CNN1DTCNFusionRegressor(**_small_fusion_kwargs(fusion_mode="bogus"))

    def test_gated_fusion_output_dim_equals_concat(self):
        """GatedFusion 输出维度应等于各分支维度之和（与 concat 一致）。"""
        fusion = GatedFusion(8, 8)
        embs = [torch.randn(2, 5, 8), torch.randn(2, 5, 8)]
        out = fusion(*embs)
        assert out.shape == (2, 5, 16)

    def test_gated_fusion_is_neutral_at_init(self):
        fusion = GatedFusion(8, 8)
        embs = [torch.randn(2, 5, 8), torch.randn(2, 5, 8)]
        assert torch.allclose(fusion(*embs), torch.cat(embs, dim=-1), atol=1e-6)

    def test_film_modulation_rejects_bad_dims(self):
        with pytest.raises(ValueError):
            FiLMModulation(context_dim=0, feature_dim=8)
        with pytest.raises(ValueError):
            FiLMModulation(context_dim=8, feature_dim=0)


# ---------------------------------------------------------------------------
# losses.py：fixed weighting
# ---------------------------------------------------------------------------


class TestFixedLossWeighting:
    def test_fixed_weighting_parses_loss_weights(self):
        loss = build_loss(
            {
                "name": WEIGHTED_COMPONENT_MSE_LOSS,
                "weighting": "fixed",
                "loss_weights": [1.0, 2.0, 1.0],
                "component_count": 3,
            }
        )
        assert loss.component_count == 3
        assert torch.allclose(loss.component_weights, torch.tensor([1.0, 2.0, 1.0]))

    def test_fixed_weighting_requires_loss_weights(self):
        with pytest.raises(ValueError, match="loss_weights"):
            build_loss(
                {
                    "name": WEIGHTED_COMPONENT_MSE_LOSS,
                    "weighting": "fixed",
                    "component_count": 3,
                }
            )

    def test_fixed_weighting_rejects_component_weights_conflict(self):
        with pytest.raises(ValueError, match="component_weights"):
            build_loss(
                {
                    "name": WEIGHTED_COMPONENT_MSE_LOSS,
                    "weighting": "fixed",
                    "loss_weights": [1.0, 2.0, 1.0],
                    "component_weights": [1.0, 1.0, 1.0],
                    "component_count": 3,
                }
            )

    def test_fixed_loss_applies_per_component_weights(self):
        """[1,2,1] 加权下，第 2 列误差贡献应被放大 2×。"""
        loss = build_loss(
            {
                "name": WEIGHTED_COMPONENT_MSE_LOSS,
                "weighting": "fixed",
                "loss_weights": [1.0, 2.0, 1.0],
                "component_count": 3,
            }
        )
        pred = torch.zeros(1, 3)
        target = torch.tensor([[1.0, 0.0, 0.0]])  # 仅第 1 列误差 1
        loss_col1 = loss(pred, target)
        target2 = torch.tensor([[0.0, 1.0, 0.0]])  # 仅第 2 列误差 1
        loss_col2 = loss(pred, target2)
        assert torch.allclose(loss_col2, 2.0 * loss_col1)

    def test_inverse_train_var_supports_mean_one_normalization(self):
        train_targets = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 4.0, 6.0],
            ],
            dtype=torch.float32,
        )
        loss = build_loss(
            {
                "name": WEIGHTED_COMPONENT_MSE_LOSS,
                "weighting": "inverse_train_var",
                "weight_normalization": "mean_one",
                "component_count": 3,
            },
            train_targets=train_targets,
        )
        assert torch.allclose(loss.component_weights.mean(), torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# CLI / model 配置边界
# ---------------------------------------------------------------------------


class TestNormalizationConfigGuards:
    def test_augment_infers_tv3_slow_channel_count(self):
        config = _build_augment_config(
            {"amplitude_scale_range": [0.9, 1.1]},
            ("slow", "ultrasonic"),
            slow_channel_count=7,
        )
        assert config is not None
        assert config.amplitude_apply_from_channel == 7

    def test_boolean_optional_cli_can_disable_config_true(self, tmp_path: Path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(tmp_path),
                    "output_dir": str(tmp_path / "out"),
                    "dequantize_waveforms": True,
                    "normalize_waveforms": True,
                }
            ),
            encoding="utf-8",
        )
        parser = build_dl_cli_parser()
        args = _resolve_args(parser.parse_args(["--config", str(config_path), "--no-normalize-waveforms"]))
        assert args.dequantize_waveforms is True
        assert args.normalize_waveforms is False

    def test_normalize_requires_dequantize(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-bad-norm")
        args = argparse.Namespace(
            config=None,
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "runs" / "bad-norm",
            model="cnn1d_tcn_fusion",
            model_kwargs={
                "output_mode": "raw3",
                "slow_channels": 7,
                "ultrasonic_channels": 5000,
                "fiber_mic_channels": 0,
                "waveform_adc_scale": 1.0,
            },
            modalities="slow,ultrasonic",
            slow_channels=None,
            input_format=None,
            scaler_path=None,
            resume_from=None,
            window=None,
            phase_windows=None,
            phase_stats_path=None,
            dequantize_waveforms=False,
            normalize_waveforms=True,
            augment=None,
            augment_seed=0,
            target_transform=None,
            epochs=1,
            batch_size=4,
            num_workers=0,
            seed=42,
            device="cpu",
            loss="mse",
            optimizer="adamw",
            lr=1e-3,
            weight_decay=0.0,
            grad_clip_norm=0.0,
            eval_splits="val",
            checkpoint_name="checkpoint.pt",
            json=False,
        )
        with pytest.raises(SystemExit):
            run_dl_cli(args)

    def test_normalize_requires_adc_scale_one(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-bad-scale")
        args = argparse.Namespace(
            config=None,
            dataset_dir=dataset_dir,
            output_dir=tmp_path / "runs" / "bad-scale",
            model="cnn1d_tcn_fusion",
            model_kwargs={
                "output_mode": "raw3",
                "slow_channels": 7,
                "ultrasonic_channels": 5000,
                "fiber_mic_channels": 0,
                "waveform_adc_scale": 5.0,
            },
            modalities="slow,ultrasonic",
            slow_channels=None,
            input_format=None,
            scaler_path=None,
            resume_from=None,
            window=None,
            phase_windows=None,
            phase_stats_path=None,
            dequantize_waveforms=True,
            normalize_waveforms=True,
            augment=None,
            augment_seed=0,
            target_transform=None,
            epochs=1,
            batch_size=4,
            num_workers=0,
            seed=42,
            device="cpu",
            loss="mse",
            optimizer="adamw",
            lr=1e-3,
            weight_decay=0.0,
            grad_clip_norm=0.0,
            eval_splits="val",
            checkpoint_name="checkpoint.pt",
            json=False,
        )
        with pytest.raises(SystemExit):
            run_dl_cli(args)


class TestRawOutputPrior:
    def test_raw3_output_prior_initializes_bias(self):
        model = CNN1DTCNFusionRegressor(**_small_fusion_kwargs(raw_output_prior=[1.0, 2.0, 3.0]))
        assert torch.allclose(model.output_head.bias, torch.tensor([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# CLI 端到端：normalize_waveforms 传参
# ---------------------------------------------------------------------------


class TestCLINormalizeWaveforms:
    def test_cli_records_normalize_waveforms(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-cli-norm")
        output_dir = tmp_path / "runs" / "tv3-norm"
        config_path = tmp_path / "tv3_norm_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "output_dir": str(output_dir),
                    "model": "cnn1d_tcn_fusion",
                    "model_kwargs": {
                        "output_mode": "raw3",
                        "slow_channels": 7,
                        "ultrasonic_channels": 5000,
                        "fiber_mic_channels": 0,
                        "waveform_embedding_dim": 8,
                        "waveform_adc_scale": 1.0,
                        "acoustic_channels": [4],
                        "acoustic_kernel_size": 3,
                        "slow_hidden_dim": 8,
                        "slow_embedding_dim": 8,
                        "tcn_channels": [4],
                        "tcn_kernel_size": 3,
                        "shared_hidden_dims": [8, 4],
                    },
                    "modalities": "slow,ultrasonic",
                    "dequantize_waveforms": True,
                    "normalize_waveforms": True,
                    "waveform_stats_features": "log_std,log_max_abs",
                    "epochs": 1,
                    "batch_size": 4,
                    "loss": {
                        "name": WEIGHTED_COMPONENT_MSE_LOSS,
                        "weighting": "fixed",
                        "loss_weights": [1.0, 2.0, 1.0],
                        "component_count": 3,
                    },
                    "eval_splits": "val",
                    "device": "cpu",
                }
            ),
            encoding="utf-8",
        )
        parser = build_dl_cli_parser()
        args = parser.parse_args(["--config", str(config_path)])
        payload = run_dl_cli(args)

        assert payload["normalize_waveforms"] is True
        assert payload["dequantize_waveforms"] is True
        # run_config.json 也应记录
        run_config = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
        assert run_config["normalize_waveforms"] is True
        assert payload["model_config"]["out_dim"] == 3
        assert payload["model_config"]["slow_channels"] == 9
        assert payload["waveform_stats_features"] == ["log_std", "log_max_abs"]
        assert run_config["waveform_stats_features"] == ["log_std", "log_max_abs"]
        assert len(payload["model_config"]["raw_output_prior"]) == 3
        assert payload["resolved_loss"]["component_weights"] == [1.0, 2.0, 1.0]
        assert run_config["resolved_loss"]["component_weights"] == [1.0, 2.0, 1.0]
        assert payload["evaluations"]["val"]["loss"] >= 0.0
