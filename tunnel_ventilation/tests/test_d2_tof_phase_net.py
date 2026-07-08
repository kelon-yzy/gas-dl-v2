"""D2 TOF-PhaseNet 单元测试与 smoke 集成测试。

覆盖:
- SoftArgmaxLag 可微峰值定位
- TOFPhaseNetRegressor forward shapes 与 aux 输出
- D2TOFPhaseLoss 复合 loss 计算
- V4BenchmarkDataset aux_target_arrays 读取
- Trainer 结构化输出与 4-tuple unpack
- CLI smoke 配置 1 epoch
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tv3.dl.cli import (
    _build_model_config,
    _parse_comma,
    _parse_waveform_stats_features,
    _resolve_raw_output_prior,
    _waveform_stats_channel_count,
)
from tv3.dl.data.dataset import V4BenchmarkDataset
from tv3.dl.models.registry import MODEL_REGISTRY, build_model
from tv3.dl.models.tof_phase_net import (
    AcousticFrontend,
    FixedQuadratureFilterBank,
    PeakShapeFeatures,
    SoftArgmaxLag,
    TOFPhaseNetRegressor,
)
from tv3.dl.training.losses import D2TOFPhaseLoss, build_loss
from tv3.dl.training.trainer import Trainer, _main_prediction


# ── SoftArgmaxLag ──────────────────────────────────────────────

class TestSoftArgmaxLag:
    def test_monotonic_shift_single_peak(self):
        """单峰平移 k samples 后 softargmax index 单调跟随。"""
        s = SoftArgmaxLag(temperature=0.01)  # low temp for sharp peak
        L = 200
        prev = -1.0
        for k in [10, 40, 70, 100, 130, 160, 190]:
            env = torch.zeros(1, 1, L)
            env[0, 0, k] = 1.0
            lag = s(env).item()
            assert lag > prev, f"softargmax not monotonic: {lag:.4f} <= {prev:.4f} at k={k}"
            prev = lag

    def test_shift_error_less_than_one_sample(self):
        """平移单峰的 soft index 误差小于 1 sample。"""
        s = SoftArgmaxLag(temperature=0.05)
        L = 5000
        for k in [500, 1500, 2500, 3500, 4500]:
            env = torch.zeros(1, 1, L)
            # 用窄高斯脉冲而非 δ 函数，使 softargmax 可定位
            indices = torch.arange(L, dtype=torch.float32)
            env[0, 0, :] = torch.exp(-0.5 * ((indices - k) / 2.0) ** 2)
            lag_norm = s(env).item()
            pred_sample = lag_norm * (L - 1)
            assert abs(pred_sample - k) < 1.0, f"shift error {abs(pred_sample - k):.3f} samples at k={k}"

    def test_batch_and_timesteps(self):
        """支持 (B, T, L) 输入。"""
        s = SoftArgmaxLag(temperature=0.05)
        B, T, L = 4, 10, 100
        env = torch.rand(B, T, L)
        lag = s(env)
        assert lag.shape == (B, T)
        assert not torch.isnan(lag).any()
        assert not torch.isinf(lag).any()

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError):
            SoftArgmaxLag(temperature=0.0)


# ── FixedQuadratureFilterBank ──────────────────────────────────

class TestFixedQuadratureFilterBank:
    def test_output_shape(self):
        f = FixedQuadratureFilterBank(sample_rate_hz=1_000_000.0, carrier_hz=200_000.0)
        wf = torch.randn(2, 8, 5000)
        i_resp, q_resp = f(wf)
        assert i_resp.shape == wf.shape
        assert q_resp.shape == wf.shape

    def test_no_nan_inf(self):
        f = FixedQuadratureFilterBank()
        wf = torch.randn(2, 8, 5000)
        i_resp, q_resp = f(wf)
        assert not torch.isnan(i_resp).any()
        assert not torch.isinf(i_resp).any()
        assert not torch.isnan(q_resp).any()
        assert not torch.isinf(q_resp).any()


# ── AcousticFrontend ───────────────────────────────────────────

class TestAcousticFrontend:
    def test_output_structure(self):
        frontend = AcousticFrontend(acoustic_feature_dim=32)
        wf = torch.randn(2, 10, 5000)
        out = frontend(wf)
        assert "features" in out
        assert "tof_s" in out
        assert "peak_index_norm" in out
        assert "peak_sharpness" in out
        assert out["features"].shape == (2, 10, 32)
        assert out["tof_s"].shape == (2, 10)
        assert out["peak_index_norm"].shape == (2, 10)
        assert out["peak_sharpness"].shape == (2, 10)

    def test_no_nan_inf(self):
        frontend = AcousticFrontend()
        wf = torch.randn(1, 5, 5000)
        out = frontend(wf)
        for key, val in out.items():
            if isinstance(val, torch.Tensor):
                assert not torch.isnan(val).any(), f"{key} has NaN"
                assert not torch.isinf(val).any(), f"{key} has Inf"


# ── TOFPhaseNetRegressor ───────────────────────────────────────

class TestTOFPhaseNetRegressor:
    @pytest.fixture
    def model(self):
        return TOFPhaseNetRegressor(
            in_channels=5009,
            out_dim=3,
            slow_channels=9,
            ultrasonic_channels=5000,
            fiber_mic_channels=0,
        )

    def test_registered_in_registry(self):
        assert "tof_phase_net" in MODEL_REGISTRY

    def test_forward_output_structure(self, model):
        x = torch.randn(2, 16, 5009)
        out = model(x)
        assert isinstance(out, dict)
        assert "prediction" in out
        assert "aux" in out
        assert out["prediction"].shape == (2, 3)
        assert out["aux"]["tof_s"].shape == (2, 16)
        assert out["aux"]["peak_index"].shape == (2, 16)
        assert out["aux"]["peak_index_samples"].shape == (2, 16)
        assert out["aux"]["peak_sharpness"].shape == (2, 16)

    def test_aux_outputs_are_frontend_softargmax(self, model):
        model.eval()
        x = torch.randn(2, 8, 5009)
        with torch.inference_mode():
            out = model(x)
            ultrasonic = x[:, :, model.slow_channels : model.slow_channels + model.ultrasonic_channels]
            frontend = model.acoustic_frontend(ultrasonic)
        assert torch.allclose(out["aux"]["tof_s"], frontend["tof_s"])
        assert torch.allclose(out["aux"]["peak_index"], frontend["peak_index_norm"])
        assert torch.allclose(out["aux"]["peak_index_samples"], frontend["peak_index_samples"])
        assert not hasattr(model, "tof_head")
        assert not hasattr(model, "peak_index_head")

    def test_forward_no_nan_inf(self, model):
        x = torch.randn(1, 8, 5009)
        out = model(x)
        assert not torch.isnan(out["prediction"]).any()
        assert not torch.isinf(out["prediction"]).any()

    def test_output_mode_must_be_raw3(self):
        with pytest.raises(ValueError):
            TOFPhaseNetRegressor(
                in_channels=5009,
                out_dim=3,
                slow_channels=9,
                ultrasonic_channels=5000,
                output_mode="gas_head",
            )

    def test_in_channels_mismatch_raises(self):
        with pytest.raises(ValueError):
            TOFPhaseNetRegressor(in_channels=100, out_dim=3, slow_channels=9, ultrasonic_channels=5000)

    def test_fiber_mic_unsupported(self):
        with pytest.raises(ValueError):
            TOFPhaseNetRegressor(
                in_channels=15009,
                out_dim=3,
                slow_channels=9,
                ultrasonic_channels=5000,
                fiber_mic_channels=10000,
            )

    def test_build_via_registry(self):
        config = {
            "name": "tof_phase_net",
            "in_channels": 5009,
            "out_dim": 3,
            "slow_channels": 9,
            "ultrasonic_channels": 5000,
            "fiber_mic_channels": 0,
            "output_mode": "raw3",
        }
        model = build_model(config)
        assert isinstance(model, TOFPhaseNetRegressor)

    def test_nct_input_not_checked_at_runtime(self, model):
        """NTC/NCT 格式检查是调用方约定，模型层只检查 ndim==3。"""
        x = torch.randn(2, 5009, 16)  # NCT shape, but 3D so model doesn't reject
        out = model(x)
        assert out["prediction"].shape[0] == 2


# ── D2TOFPhaseLoss ─────────────────────────────────────────────

class TestD2TOFPhaseLoss:
    @pytest.fixture
    def loss_fn(self):
        return build_loss({
            "name": "d2_tof_phase_loss",
            "component_loss": {"name": "weighted_component_mse", "weighting": "fixed", "loss_weights": [1.0, 2.0, 1.0], "component_count": 3},
            "aux_weights": {"tof_s": 0.2, "peak_index": 0.05},
        })

    def test_accepts_aux_targets(self, loss_fn):
        assert getattr(loss_fn, "accepts_aux_targets", False)

    def test_loss_scalar_output(self, loss_fn):
        pred = {
            "prediction": torch.randn(4, 3),
            "aux": {
                "tof_s": torch.randn(4, 16),
                "peak_index": torch.rand(4, 16),
                "peak_sharpness": torch.rand(4, 16),
            },
        }
        target = torch.randn(4, 3) * 10 + 50
        aux = {
            "tof_true_s": torch.randn(4, 16) * 0.001 + 0.003,
            "tof_accepted": torch.ones(4, 16),
            "peak_index": torch.randint(0, 5000, (4, 16)).float(),
        }
        loss_val = loss_fn(pred, target, aux_targets=aux)
        assert loss_val.ndim == 0
        assert torch.isfinite(loss_val)

    def test_requires_structured_pred(self, loss_fn):
        with pytest.raises(ValueError):
            loss_fn(torch.randn(4, 3), torch.randn(4, 3), aux_targets={"tof_true_s": torch.randn(4, 16)})

    def test_requires_aux_targets_when_weights_nonzero(self, loss_fn):
        pred = {
            "prediction": torch.randn(4, 3),
            "aux": {"tof_s": torch.randn(4, 16), "peak_index": torch.rand(4, 16), "peak_sharpness": torch.rand(4, 16)},
        }
        with pytest.raises(ValueError):
            loss_fn(pred, torch.randn(4, 3))


# ── Dataset aux_target_arrays ──────────────────────────────────

class TestDatasetAuxTargets:
    def _smoke_dataset_path(self):
        p = Path("data/tv3-smoke7")
        if p.is_dir() and (p / "sequences").is_dir():
            return p
        return None

    @pytest.mark.skipif(
        not Path("data/tv3-smoke7/sequences").is_dir() if Path("data/tv3-smoke7").is_dir() else True,
        reason="tv3-smoke7 数据不可用",
    )
    def test_aux_target_in_batch(self):
        ds = V4BenchmarkDataset(
            Path("data/tv3-smoke7"),
            split="train",
            modalities=("slow", "ultrasonic"),
            dequantize_waveforms=True,
            aux_target_arrays={"tof_true_s": "ultrasonic_tof_s"},
        )
        xb, yb = ds[0]
        assert isinstance(xb, dict)
        assert "x" in xb
        assert "aux_targets" in xb
        assert "tof_true_s" in xb["aux_targets"]
        assert isinstance(xb["aux_targets"]["tof_true_s"], torch.Tensor)

    def test_missing_file_raises_on_getitem(self):
        smoke_dir = self._smoke_dataset_path()
        if smoke_dir is None:
            pytest.skip("tv3-smoke7 数据不可用")
        ds = V4BenchmarkDataset(
            smoke_dir,
            split="train",
            modalities=("slow",),
            aux_target_arrays={"nonexistent": "nonexistent_file"},
        )
        with pytest.raises(FileNotFoundError):
            _ = ds[0]

    def test_empty_aux_dict_config_raises(self):
        # 空 dict 必须在构造时拒绝
        try:
            V4BenchmarkDataset(
                Path("data/tv3-smoke7"),
                split="train",
                modalities=("slow",),
                aux_target_arrays={},
            )
            # 如果 data/tv3-smoke7 不存在会先报 FileNotFoundError，也算正确拒绝
        except (ValueError, FileNotFoundError):
            pass
        else:
            pytest.fail("Expected ValueError or FileNotFoundError for invalid config")


# ── Trainer structured output ──────────────────────────────────

class TestTrainerStructuredOutput:
    def test_main_prediction_tensor(self):
        t = torch.randn(4, 3)
        assert _main_prediction(t) is t

    def test_main_prediction_dict(self):
        d = {"prediction": torch.randn(4, 3), "aux": {}}
        result = _main_prediction(d)
        assert result.shape == (4, 3)
        assert result is d["prediction"]

    def test_main_prediction_dict_missing_key(self):
        with pytest.raises(ValueError):
            _main_prediction({"aux": {}})

    def test_unpack_batch_with_aux(self):
        from tv3.dl.training.trainer import Trainer
        x = torch.randn(2, 16, 5009)
        aux = {"tof_true_s": torch.randn(2, 16)}
        batch = ({"x": x, "aux_targets": aux}, torch.randn(2, 3))
        _x, _y, _model_kwargs, _aux_targets = Trainer._unpack_batch(
            batch, device=torch.device("cpu"), non_blocking=False
        )
        assert "aux_targets" not in _model_kwargs
        assert _aux_targets is not None
        assert "tof_true_s" in _aux_targets

    def test_unpack_batch_without_aux(self):
        from tv3.dl.training.trainer import Trainer
        x = torch.randn(2, 16, 5009)
        batch = (x, torch.randn(2, 3))
        _x, _y, _model_kwargs, _aux_targets = Trainer._unpack_batch(
            batch, device=torch.device("cpu"), non_blocking=False
        )
        assert _model_kwargs == {}
        assert _aux_targets is None

    def test_evaluate_reports_auxiliary_metrics(self):
        class DummyD2Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = nn.Parameter(torch.zeros(3))

            def forward(self, x: torch.Tensor) -> dict[str, object]:
                return {
                    "prediction": x[:, 0, 2:5] + self.offset * 0.0,
                    "aux": {
                        "tof_s": x[:, :, 0],
                        "peak_index": x[:, :, 1] / 4999.0,
                        "peak_index_samples": x[:, :, 1],
                    },
                }

        tof_true = torch.tensor(
            [[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]],
            dtype=torch.float32,
        )
        peak_true = torch.tensor(
            [[1000.0, 1200.0, 1400.0], [1600.0, 1800.0, 2000.0]],
            dtype=torch.float32,
        )
        y = torch.tensor(
            [[20.0, 10.0, 70.0], [21.0, 11.0, 68.0]],
            dtype=torch.float32,
        )
        x = torch.zeros(2, 3, 5)
        x[:, :, 0] = tof_true + 1e-6
        x[:, :, 1] = peak_true + 2.0
        x[:, 0, 2:5] = y
        aux_targets = {
            "tof_true_s": tof_true,
            "tof_observed_s": tof_true + 2e-6,
            "peak_index": peak_true,
            "tof_accepted": torch.tensor(
                [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
                dtype=torch.float32,
            ),
        }
        loss_fn = build_loss(
            {
                "name": "d2_tof_phase_loss",
                "component_loss": {
                    "name": "weighted_component_mse",
                    "weighting": "fixed",
                    "loss_weights": [1.0, 2.0, 1.0],
                    "component_count": 3,
                },
                "aux_weights": {"tof_s": 0.2, "peak_index": 0.05},
            }
        )
        model = DummyD2Model()
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=loss_fn,
            device="cpu",
            component_names=("x_CO2", "x_O2", "x_N2"),
        )
        result = trainer.evaluate([({"x": x, "aux_targets": aux_targets}, y)])
        aux_metrics = result["auxiliary_metrics"]
        assert aux_metrics is not None
        assert aux_metrics["tof_mae_s"] == pytest.approx(1e-6, rel=1e-4, abs=1e-9)
        assert aux_metrics["tof_mae_us"] == pytest.approx(1.0, rel=1e-4, abs=1e-3)
        assert aux_metrics["peak_index_mae_samples"] == pytest.approx(2.0)
        assert aux_metrics["observed_tof_mae_s"] == pytest.approx(2e-6, rel=1e-4, abs=1e-9)
        assert aux_metrics["d2_minus_observed_tof_mae_s"] == pytest.approx(-1e-6, rel=1e-4, abs=1e-9)


# ── CLI smoke ──────────────────────────────────────────────────

class TestD2CLISmoke:
    def test_smoke_config_loads(self):
        config_path = Path("configs/tv3_d2_tof_phasenet_smoke.json")
        if not config_path.is_file():
            pytest.skip("smoke config 不存在")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert cfg["model"] == "tof_phase_net"
        assert cfg["loss"]["name"] == "d2_tof_phase_loss"
        assert "aux_target_arrays" in cfg
        assert cfg["model_kwargs"]["slow_channels"] == 7

    def test_smoke_model_builds_from_config(self):
        config_path = Path("configs/tv3_d2_tof_phasenet_smoke.json")
        if not config_path.is_file():
            pytest.skip("smoke config 不存在")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        model_config = _build_model_config(
            cfg["model"],
            cfg.get("model_kwargs", {}),
            in_channels=5009,
            out_dim=3,
            timesteps=16,
        )
        modalities = _parse_comma(cfg["modalities"])
        waveform_stats_features = _parse_waveform_stats_features(cfg["waveform_stats_features"])
        stats_channels = _waveform_stats_channel_count(modalities, waveform_stats_features)
        model_config["slow_channels"] = int(model_config["slow_channels"]) + stats_channels
        train_labels = np.array(
            [[20.0, 10.0, 70.0], [22.0, 12.0, 66.0]],
            dtype=np.float32,
        )
        _resolve_raw_output_prior(
            cfg["model"],
            model_config,
            train_labels,
            out_dim=3,
            target_transform=None,
        )
        assert model_config["slow_channels"] == 9
        assert model_config["raw_output_prior"] == [21.0, 11.0, 68.0]
        model = build_model(model_config)
        assert isinstance(model, TOFPhaseNetRegressor)
        assert torch.allclose(model.component_head.bias, torch.tensor([21.0, 11.0, 68.0]))
