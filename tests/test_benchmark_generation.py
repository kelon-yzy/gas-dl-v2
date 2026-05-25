import csv
import json

import numpy as np

from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generate_benchmark_dataset_writes_v4_assets(tmp_path):
    summary = generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(dataset_slug="wv4-smoke", sequence_count=16, seed=7),
    )

    dataset_dir = tmp_path / "wv4-smoke"
    condition_rows = _read_csv(dataset_dir / "condition_grid_sequence.csv")
    index_rows = _read_csv(dataset_dir / "sequence_index.csv")
    label_rows = _read_csv(dataset_dir / "sequence_labels.csv")
    train_rows = _read_csv(dataset_dir / "splits" / "train.csv")
    val_rows = _read_csv(dataset_dir / "splits" / "val.csv")
    test_rows = _read_csv(dataset_dir / "splits" / "test.csv")
    extrapolation_rows = _read_csv(dataset_dir / "splits" / "extrapolation.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))

    assert summary["dataset_slug"] == "wv4-smoke"
    assert summary["sequence_count"] == 16
    assert len(condition_rows) == 16
    assert len(index_rows) == 16
    assert len(label_rows) == 16
    assert manifest["schema_version"] == "v4-benchmark-1"
    assert manifest["path_lms"] == [0.2, 0.25, 0.3, 0.35, 0.4]

    forbidden_fields = {"base_condition_id", "noise_seed_index", "noise_seed"}
    assert forbidden_fields.isdisjoint(condition_rows[0])
    assert condition_rows[0]["mixture_id"].startswith("M")
    assert condition_rows[0]["sequence_id"].startswith("Q")
    assert condition_rows[0]["mixture_id"] != condition_rows[0]["sequence_id"]

    split_sequence_ids = {
        row["sequence_id"]
        for rows in (train_rows, val_rows, test_rows, extrapolation_rows)
        for row in rows
    }
    assert split_sequence_ids == {row["sequence_id"] for row in condition_rows}


def test_generated_condition_rows_have_component_sum_100(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(dataset_slug="wv4-components", sequence_count=8, seed=11),
    )

    rows = _read_csv(tmp_path / "wv4-components" / "condition_grid_sequence.csv")

    for row in rows:
        total = sum(float(row[name]) for name in ("x_H2", "x_CH4", "x_CO2", "x_N2"))
        assert abs(total - 100.0) < 1e-5


def test_generate_benchmark_dataset_writes_npz_storage_arrays_and_metadata(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-npz",
            sequence_count=5,
            seed=19,
            timesteps=8,
            dt_s=0.25,
            storage="npz",
            multi_path_phase="off",
        ),
    )

    dataset_dir = tmp_path / "wv4-npz"
    condition_rows = _read_csv(dataset_dir / "condition_grid_sequence.csv")
    slow_rows = _read_csv(dataset_dir / "sequences" / "slow_sequence_long.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((dataset_dir / "quality" / "validation_summary.json").read_text(encoding="utf-8"))

    y = np.load(dataset_dir / "labels" / "y.npy")
    sequence_ids = np.load(dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True)
    slow_channel_names = np.load(dataset_dir / "metadata" / "slow_channel_names.npy", allow_pickle=True)
    label_names = np.load(dataset_dir / "metadata" / "label_names.npy", allow_pickle=True)
    slow = np.load(dataset_dir / "sequences" / "slow.npy")
    ultrasonic = np.load(dataset_dir / "sequences" / "ultrasonic_int16.npy")
    fiber_mic = np.load(dataset_dir / "sequences" / "fiber_mic_int16.npy")
    bundle = np.load(dataset_dir / "sequences" / "waveform_sequence.npz")

    assert manifest["storage"] == "npz"
    assert manifest["shapes"]["slow"] == [5, 8, len(slow_channel_names)]
    assert validation["status"] == "pass"
    assert y.shape == (5, 4)
    assert slow.shape == (5, 8, len(slow_channel_names))
    assert ultrasonic.shape[:2] == (5, 8)
    assert fiber_mic.shape[:2] == (5, 8)
    assert bundle["slow"].shape == slow.shape
    assert list(sequence_ids.astype(str)) == [row["sequence_id"] for row in condition_rows]
    assert list(label_names.astype(str)) == ["x_H2", "x_CH4", "x_CO2", "x_N2"]
    assert len(slow_rows) == 5 * 8


def test_generate_benchmark_dataset_uses_configured_path_lms(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-path-lms",
            sequence_count=4,
            seed=31,
            timesteps=8,
            storage="npz",
            multi_path_phase="steady",
            path_lms=(0.25, 0.35),
        ),
    )

    dataset_dir = tmp_path / "wv4-path-lms"
    slow_rows = _read_csv(dataset_dir / "sequences" / "slow_sequence_long.csv")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    waveform_spec = json.loads((dataset_dir / "metadata" / "waveform_spec.json").read_text(encoding="utf-8"))

    first_sequence_rows = [row for row in slow_rows if row["sequence_id"] == "Q000001"]

    assert manifest["path_lms"] == [0.25, 0.35]
    assert waveform_spec["path_lms"] == [0.25, 0.35]
    assert first_sequence_rows[4]["phase_id"] == "steady"
    assert first_sequence_rows[5]["phase_id"] == "steady"
    assert first_sequence_rows[4]["L_m"] == "0.25000"
    assert first_sequence_rows[5]["L_m"] == "0.35000"


def test_generate_benchmark_dataset_writes_memmap_storage_arrays_without_npz(tmp_path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="wv4-memmap",
            sequence_count=4,
            seed=23,
            timesteps=8,
            storage="memmap",
        ),
    )

    dataset_dir = tmp_path / "wv4-memmap"

    assert (dataset_dir / "sequences" / "slow.npy").is_file()
    assert (dataset_dir / "sequences" / "ultrasonic_int16.npy").is_file()
    assert (dataset_dir / "sequences" / "fiber_mic_int16.npy").is_file()
    assert not (dataset_dir / "sequences" / "waveform_sequence.npz").exists()


def test_generate_benchmark_dataset_rejects_invalid_storage(tmp_path):
    try:
        generate_benchmark_dataset(
            tmp_path,
            BenchmarkGenerationSpec(dataset_slug="bad-storage", sequence_count=4, seed=1, storage="csv"),
        )
    except ValueError as exc:
        assert "storage must be one of" in str(exc)
    else:
        raise AssertionError("invalid storage was accepted")
