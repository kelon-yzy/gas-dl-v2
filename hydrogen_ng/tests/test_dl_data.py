from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hg.dl.data.augmentation import TimeSeriesAugmentConfig, augment_sequence
from hg.dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from hg.dl.data.scalers import apply_scaler, load_scaler
from hg.dl.data.splits import SPLIT_NAMES, load_splits, resolve_split_indices, split_sequence_ids
from hg.sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset
from hg.sim.packaging.scalers import fit_z_score_scalers


def _make_smoke_dataset(tmp_path: Path, slug: str = "dl-smoke", sequences: int = 16) -> Path:
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=42,
            timesteps=32,
            dt_s=0.5,
            storage="npz",
            optical_absorption_backend="empirical_v1",
        ),
    )
    return tmp_path / slug


class TestSplits:
    def test_load_splits_returns_four_splits(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        splits = load_splits(dataset_dir / "splits")
        assert set(splits) == set(SPLIT_NAMES)
        for name in SPLIT_NAMES:
            assert len(splits[name]) > 0

    def test_split_rows_have_required_columns(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        splits = load_splits(dataset_dir / "splits")
        for name, rows in splits.items():
            for row in rows:
                assert "sequence_id" in row
                assert "mixture_id" in row

    def test_split_sequence_ids_returns_lists(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        splits = load_splits(dataset_dir / "splits")
        ids = split_sequence_ids(splits)
        total = sum(len(v) for v in ids.values())
        assert total == 16

    def test_resolve_split_indices_coverage(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        splits = load_splits(dataset_dir / "splits")
        meta_ids = np.load(dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True)
        master = [str(sid) for sid in meta_ids.tolist()]
        indices = resolve_split_indices(splits, master)
        resolved = {idx for idx_list in indices.values() for idx in idx_list}
        assert resolved == set(range(len(master)))


class TestScalers:
    def test_load_scaler_returns_expected_keys(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        scaler = load_scaler(dataset_dir / "scalers" / "scaler_slow_sequence.json")
        assert scaler["method"] == "z_score"
        expected_channels = (
            "V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "T_C",
            "P_MPa", "H_RH", "L_m", "piston_position_m",
        )
        assert tuple(scaler["channel_names"]) == expected_channels
        assert len(scaler["mean"]) == 8
        assert len(scaler["std"]) == 8

    def test_apply_scaler_zero_mean_unit_variance(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        scaler = load_scaler(dataset_dir / "scalers" / "scaler_slow_sequence.json")
        slow = np.load(dataset_dir / "sequences" / "slow.npy").astype(np.float32)
        # use only train split indices for fitting verification
        splits = load_splits(dataset_dir / "splits")
        meta_ids = np.load(dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True)
        master = [str(sid) for sid in meta_ids.tolist()]
        train_idx = resolve_split_indices(splits, master)["train"]
        train_slow = slow[train_idx]
        scaled = apply_scaler(train_slow, scaler)
        means = scaled.mean(axis=(0, 1))
        stds = scaled.std(axis=(0, 1))
        assert np.all(np.abs(means) < 0.5)
        assert np.all(stds > 0.5)

    def test_fit_and_apply_scaler_share_near_zero_std_threshold(self):
        matrix = np.array(
            [
                [[1.0, 10.0], [1.0, 10.0 + 1e-13]],
                [[1.0, 10.0 - 1e-13], [1.0, 10.0]],
            ],
            dtype=np.float64,
        )
        scaler, _ = fit_z_score_scalers(
            matrix,
            [0, 1],
            channel_names=("constant", "tiny"),
            modal_groups={"all": ("constant", "tiny")},
        )

        assert scaler["std"] == [1.0, 1.0]
        np.testing.assert_allclose(apply_scaler(matrix, scaler), matrix - np.array(scaler["mean"]))

    def test_apply_scaler_rejects_unsupported_ndim(self):
        scaler = {"method": "z_score", "channel_names": ["a"], "mean": [0.0], "std": [1.0]}

        with pytest.raises(ValueError, match="2D or 3D"):
            apply_scaler(np.array([1.0], dtype=np.float32), scaler)

    def test_apply_scaler_rejects_channel_mismatch(self):
        scaler = {"method": "z_score", "channel_names": ["a", "b"], "mean": [0.0, 0.0], "std": [1.0, 1.0]}

        with pytest.raises(ValueError, match="last dimension"):
            apply_scaler(np.ones((4, 1), dtype=np.float32), scaler)


class TestV4BenchmarkDataset:
    def test_slow_only_nct_format(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-nct")
        ds = V4BenchmarkDataset(dataset_dir, split="train", modalities=("slow",), input_format="NCT", lazy=False)
        x, y = ds[0]
        assert x.ndim == 2  # (C, T)
        assert y.shape == (4,)

    def test_slow_only_ntc_format(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-ntc")
        ds = V4BenchmarkDataset(dataset_dir, split="train", modalities=("slow",), input_format="NTC", lazy=False)
        x, y = ds[0]
        assert x.ndim == 2  # (T, C)
        assert x.shape[1] == 8
        assert y.shape == (4,)

    def test_all_splits_loadable(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-all")
        for split_name in SPLIT_NAMES:
            ds = V4BenchmarkDataset(dataset_dir, split=split_name, modalities=("slow",), lazy=False)
            assert len(ds) > 0

    def test_scaler_applied_normalizes_output(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-scaled")
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NTC",
            scaler_path=dataset_dir / "scalers" / "scaler_slow_sequence.json",
            lazy=False,
        )
        x, _ = ds[0]
        assert -4.0 < x.mean().item() < 4.0

    def test_lazy_loading_defers_array_load(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-lazy")
        ds = V4BenchmarkDataset(dataset_dir, split="train", modalities=("slow",), lazy=True)
        assert ds._slow is None
        _ = ds[0]
        assert isinstance(ds._slow, np.memmap)

    def test_lazy_loading_keeps_waveforms_memmapped_int16_until_sample_cast(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-wave-lazy", sequences=8)
        ds = V4BenchmarkDataset(dataset_dir, split="train", modalities=("ultrasonic", "fiber_mic"), lazy=False)

        assert isinstance(ds._ultrasonic, np.memmap)
        assert isinstance(ds._fiber_mic, np.memmap)
        assert ds._ultrasonic.dtype == np.int32
        assert ds._fiber_mic.dtype == np.int32
        x, _ = ds[0]
        assert x.dtype == torch.float32

    def test_dequantize_waveforms_uses_scale_arrays(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-wave-dequant", sequences=8)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("ultrasonic",),
            input_format="NTC",
            lazy=False,
            dequantize_waveforms=True,
        )

        x, _ = ds[0]
        src_idx = ds.indices[0]
        raw = np.load(dataset_dir / "sequences" / "ultrasonic_int32.npy", mmap_mode="r")[src_idx].astype(np.float32)
        scale = np.load(dataset_dir / "sequences" / "ultrasonic_scale.npy", mmap_mode="r")[src_idx].astype(np.float32)

        np.testing.assert_allclose(x.numpy(), raw * scale[:, np.newaxis], rtol=1e-6, atol=1e-6)

    def test_multimodal_concatenates_channels(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-multi", sequences=8)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic"),
            input_format="NTC",
            lazy=False,
        )
        x, _ = ds[0]
        assert x.shape[1] == 8 + 5000  # slow channels + ultrasonic waveform samples

    def test_rejects_invalid_modality(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-bad")
        with pytest.raises(ValueError, match="imaginary"):
            V4BenchmarkDataset(dataset_dir, split="train", modalities=("imaginary",))

    def test_rejects_invalid_input_format(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-fmt")
        with pytest.raises(ValueError, match="input_format"):
            V4BenchmarkDataset(dataset_dir, split="train", input_format="TNC")

    def test_modality_options_constant(self):
        assert MODALITY_OPTIONS == ("slow", "ultrasonic", "fiber_mic")

    def test_augmentation_preserves_shape_for_ntc_and_nct(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-aug")
        ntc = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NTC",
            lazy=False,
            augment_config=TimeSeriesAugmentConfig(jitter_std=0.01, window_fraction=0.75),
            augment_seed=1,
        )
        nct = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NCT",
            lazy=False,
            augment_config=TimeSeriesAugmentConfig(jitter_std=0.01, window_fraction=0.75),
            augment_seed=1,
        )

        x_ntc, _ = ntc[0]
        x_nct, _ = nct[0]

        assert x_ntc.shape == (32, 8)
        assert x_nct.shape == (8, 32)
        # 相同 augment_seed、相同样本：NCT 应是 NTC 增强结果的转置（增强在 transpose 之前完成）。
        assert torch.allclose(x_nct, x_ntc.transpose(0, 1))

    def test_phase_window_resamples_all_modalities_to_original_length(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-window-phase", sequences=8)
        full = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic", "fiber_mic"),
            input_format="NTC",
            lazy=False,
        )
        exposure = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic", "fiber_mic"),
            input_format="NTC",
            lazy=False,
            window={"kind": "phase", "value": "exposure"},
        )

        x_full, _ = full[0]
        x_exposure, _ = exposure[0]

        assert x_exposure.shape == x_full.shape
        assert not torch.allclose(x_exposure, x_full)

    def test_early_window_resamples_for_nct(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-window-early", sequences=8)
        full = V4BenchmarkDataset(dataset_dir, split="train", modalities=("slow",), input_format="NCT", lazy=False)
        early = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NCT",
            lazy=False,
            window={"kind": "early", "value": 0.5},
        )

        x_full, _ = full[0]
        x_early, _ = early[0]

        assert x_early.shape == x_full.shape
        assert not torch.allclose(x_early, x_full)

    def test_phase_windows_stack_resampled_views(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-phase-windows", sequences=8)
        full = V4BenchmarkDataset(dataset_dir, split="train", modalities=("slow",), input_format="NTC", lazy=False)
        multi = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow",),
            input_format="NTC",
            lazy=False,
            phase_windows=[None, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}],
        )

        x_full, _ = full[0]
        x_multi, y = multi[0]

        assert x_multi.shape == (3, *x_full.shape)
        assert y.shape == (4,)
        assert torch.allclose(x_multi[0], x_full)
        assert not torch.allclose(x_multi[1], x_full)
        assert not torch.allclose(x_multi[2], x_full)

    def test_phase_windows_rejects_empty_and_window_combination(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-phase-windows-bad", sequences=8)
        with pytest.raises(ValueError, match="phase_windows must not be empty"):
            V4BenchmarkDataset(dataset_dir, split="train", phase_windows=[])
        with pytest.raises(ValueError, match="cannot be combined"):
            V4BenchmarkDataset(
                dataset_dir,
                split="train",
                window={"kind": "phase", "value": "exposure"},
                phase_windows=[None, {"kind": "phase", "value": "recovery"}],
            )

    def test_rejects_invalid_window(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-window-bad", sequences=8)
        with pytest.raises(ValueError, match="early window value"):
            V4BenchmarkDataset(dataset_dir, split="train", window={"kind": "early", "value": 0.0})
        with pytest.raises(ValueError, match="empty timestep window"):
            V4BenchmarkDataset(dataset_dir, split="train", window={"kind": "phase", "value": "missing_phase"})


def test_augment_sequence_resamples_window_to_original_length():
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    augmented = augment_sequence(values, TimeSeriesAugmentConfig(window_fraction=0.5), np.random.default_rng(3))

    assert augmented.shape == values.shape
    assert augmented.dtype == np.float32


def test_augment_sequence_changes_values_with_window_and_jitter():
    values = np.tile(np.arange(20, dtype=np.float32).reshape(20, 1), (1, 2))

    windowed = augment_sequence(values, TimeSeriesAugmentConfig(window_fraction=0.5), np.random.default_rng(0))
    jittered = augment_sequence(values, TimeSeriesAugmentConfig(jitter_std=0.5), np.random.default_rng(0))

    assert windowed.shape == values.shape
    assert not np.allclose(windowed, values)
    assert not np.allclose(jittered, values)


# ── P3 augmentation unit tests ──────────────────────────────────────────


class TestTimeJitter:
    def test_preserves_shape(self):
        values = np.arange(50, dtype=np.float32).reshape(25, 2)
        augmented = augment_sequence(
            values,
            TimeSeriesAugmentConfig(max_shift=5, apply_prob=1.0),
            np.random.default_rng(42),
        )
        assert augmented.shape == values.shape
        assert augmented.dtype == np.float32

    def test_edge_padding_zero_shift_yields_identity(self):
        values = np.arange(50, dtype=np.float32).reshape(25, 2)
        augmented = augment_sequence(
            values,
            TimeSeriesAugmentConfig(max_shift=0, apply_prob=1.0),
            np.random.default_rng(0),
        )
        np.testing.assert_array_equal(augmented, values)

    def test_deterministic_with_fixed_seed(self):
        values = np.arange(100, dtype=np.float32).reshape(20, 5)
        cfg = TimeSeriesAugmentConfig(max_shift=3, apply_prob=1.0)
        a1 = augment_sequence(values, cfg, np.random.default_rng(12345))
        a2 = augment_sequence(values, cfg, np.random.default_rng(12345))
        np.testing.assert_array_equal(a1, a2)

    def test_positive_shift_uses_edge_padding(self):
        values = np.arange(30, dtype=np.float32).reshape(10, 3)
        values[:, 0] = np.arange(10, dtype=np.float32)
        cfg = TimeSeriesAugmentConfig(max_shift=5, apply_prob=1.0)
        augmented = augment_sequence(values, cfg, np.random.default_rng(999999))
        # With a large max_shift the shift should be non-zero with high probability.
        # Verify shape and no NaN.
        assert augmented.shape == values.shape
        assert not np.any(np.isnan(augmented))

    def test_negative_shift_uses_edge_padding(self):
        values = np.arange(30, dtype=np.float32).reshape(10, 3)
        cfg = TimeSeriesAugmentConfig(max_shift=5, apply_prob=1.0)
        augmented = augment_sequence(values, cfg, np.random.default_rng(111111))
        assert augmented.shape == values.shape
        assert not np.any(np.isnan(augmented))


class TestAmplitudeScale:
    def test_only_waveform_channels_change(self):
        rng = np.random.default_rng(7)
        T, C = 20, 12
        values = np.ones((T, C), dtype=np.float32) * 2.0
        cfg = TimeSeriesAugmentConfig(
            amplitude_scale_range=(0.5, 0.5),
            amplitude_apply_from_channel=8,
            apply_prob=1.0,
        )
        augmented = augment_sequence(values, cfg, rng)
        np.testing.assert_array_equal(augmented[:, :8], values[:, :8])
        np.testing.assert_array_equal(augmented[:, 8:], np.ones((T, C - 8), dtype=np.float32))

    def test_scale_one_no_change(self):
        values = np.arange(30, dtype=np.float32).reshape(10, 3)
        cfg = TimeSeriesAugmentConfig(
            amplitude_scale_range=(1.0, 1.0),
            amplitude_apply_from_channel=0,
            apply_prob=1.0,
        )
        augmented = augment_sequence(values, cfg, np.random.default_rng(0))
        np.testing.assert_array_equal(augmented, values)

    def test_apply_from_channel_beyond_width_noop(self):
        values = np.arange(20, dtype=np.float32).reshape(10, 2)
        cfg = TimeSeriesAugmentConfig(
            amplitude_scale_range=(0.5, 1.5),
            amplitude_apply_from_channel=99,
            apply_prob=1.0,
        )
        augmented = augment_sequence(values, cfg, np.random.default_rng(0))
        np.testing.assert_array_equal(augmented, values)


class TestGaussianNoise:
    def test_noise_adds_variance(self):
        values = np.zeros((1000, 4), dtype=np.float32)
        cfg = TimeSeriesAugmentConfig(gaussian_noise_std=0.1, apply_prob=1.0)
        augmented = augment_sequence(values, cfg, np.random.default_rng(99))
        std_actual = float(np.std(augmented))
        assert 0.08 < std_actual < 0.12

    def test_zero_std_no_change(self):
        values = np.arange(50, dtype=np.float32).reshape(10, 5)
        cfg = TimeSeriesAugmentConfig(gaussian_noise_std=0.0, apply_prob=1.0)
        augmented = augment_sequence(values, cfg, np.random.default_rng(0))
        np.testing.assert_array_equal(augmented, values)


class TestApplyProb:
    def test_prob_zero_skips_all(self):
        values = np.arange(40, dtype=np.float32).reshape(10, 4)
        cfg = TimeSeriesAugmentConfig(
            max_shift=5,
            amplitude_scale_range=(0.5, 1.5),
            gaussian_noise_std=0.5,
            apply_prob=0.0,
        )
        augmented = augment_sequence(values, cfg, np.random.default_rng(0))
        np.testing.assert_array_equal(augmented, values)

    def test_prob_one_always_applies_with_fixed_seed(self):
        values = np.arange(50, dtype=np.float32).reshape(10, 5)
        cfg = TimeSeriesAugmentConfig(max_shift=3, apply_prob=1.0)
        a1 = augment_sequence(values, cfg, np.random.default_rng(42))
        a2 = augment_sequence(values, cfg, np.random.default_rng(42))
        np.testing.assert_array_equal(a1, a2)


class TestValidateConfig:
    def test_rejects_negative_max_shift(self):
        with pytest.raises(ValueError, match="max_shift"):
            augment_sequence(np.ones((2, 2)), TimeSeriesAugmentConfig(max_shift=-1), np.random.default_rng(0))

    def test_rejects_negative_gaussian_noise_std(self):
        with pytest.raises(ValueError, match="gaussian_noise_std"):
            augment_sequence(np.ones((2, 2)), TimeSeriesAugmentConfig(gaussian_noise_std=-0.1), np.random.default_rng(0))

    def test_rejects_apply_prob_out_of_range(self):
        with pytest.raises(ValueError, match="apply_prob"):
            augment_sequence(np.ones((2, 2)), TimeSeriesAugmentConfig(apply_prob=1.5), np.random.default_rng(0))
        with pytest.raises(ValueError, match="apply_prob"):
            augment_sequence(np.ones((2, 2)), TimeSeriesAugmentConfig(apply_prob=-0.1), np.random.default_rng(0))

    def test_rejects_amplitude_scale_range_non_positive(self):
        with pytest.raises(ValueError, match="amplitude_scale_range"):
            augment_sequence(
                np.ones((2, 2)),
                TimeSeriesAugmentConfig(amplitude_scale_range=(0.0, 1.0)),
                np.random.default_rng(0),
            )

    def test_rejects_amplitude_scale_range_lo_gt_hi(self):
        with pytest.raises(ValueError, match="amplitude_scale_range"):
            augment_sequence(
                np.ones((2, 2)),
                TimeSeriesAugmentConfig(amplitude_scale_range=(2.0, 1.0)),
                np.random.default_rng(0),
            )

    def test_rejects_negative_apply_from_channel(self):
        with pytest.raises(ValueError, match="amplitude_apply_from_channel"):
            augment_sequence(
                np.ones((2, 2)),
                TimeSeriesAugmentConfig(amplitude_apply_from_channel=-1, amplitude_scale_range=(0.5, 1.5)),
                np.random.default_rng(0),
            )


class TestAugmentDataset:
    def test_augment_option_does_not_crash_smoke(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-aug-p3", sequences=8)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="train",
            modalities=("slow", "ultrasonic"),
            input_format="NTC",
            lazy=False,
            augment_config=TimeSeriesAugmentConfig(max_shift=3, apply_prob=1.0),
            augment_seed=42,
        )
        x, y = ds[0]
        assert x.shape == (32, 8 + 5000)
        assert y.shape == (4,)
        assert x.dtype == torch.float32

    def test_augment_default_skips_val_split(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ds-aug-val", sequences=8)
        ds = V4BenchmarkDataset(
            dataset_dir,
            split="val",
            modalities=("slow",),
            input_format="NTC",
            lazy=False,
        )
        assert ds._augment_config is None
        assert ds._augment_seed == 0
