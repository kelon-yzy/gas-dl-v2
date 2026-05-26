from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from dl.data.scalers import apply_scaler, load_scaler
from dl.data.splits import SPLIT_NAMES, load_splits, resolve_split_indices, split_sequence_ids
from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset
from sim.packaging.scalers import fit_z_score_scalers


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
        assert ds._ultrasonic.dtype == np.int16
        assert ds._fiber_mic.dtype == np.int16
        x, _ = ds[0]
        assert x.dtype == torch.float32

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
        assert x.shape[1] == 8 + 1000  # slow channels + ultrasonic waveform samples

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
