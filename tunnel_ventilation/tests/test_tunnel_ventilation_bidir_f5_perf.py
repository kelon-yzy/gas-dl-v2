"""F5 performance: frame-cache workers/reuse and shared slow-block equivalence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.bidir_arm_features import (
    arm_specs,
    assemble_arm_feature_matrix,
    build_arm_feature_caches,
    compute_shared_slow_windowed_block,
)
from tv3.ml.bidir_features import (
    FEATURE_BUILDER,
    BIDIR_RAW_DSP_SCHEMA_VERSION,
)
from tv3.pipeline.build_tv3_bidir_features import (
    FRAME_ARRAY_SPECS,
    SEQ_SCALAR_SPECS,
    _chunk_indices,
    _try_reuse_existing_cache,
    build_tv3_bidir_feature_cache,
    default_bidir_workers,
)
from tv3.sim.core.tunnel_ventilation_bidir_schema import COMPOSITION_SCHEME, SCHEMA_VERSION
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
)
from tv3.sim.generation.waveforms import WaveformSpec, simulate_bidirectional_waveform_measurement


def _write_minimal_bidir_dataset(root: Path, *, n_seq: int = 8, n_t: int = 8) -> Path:
    """Hand-built tiny bidir dataset (avoids Windows memmap publish locks)."""
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("labels", "metadata", "sequences", "splits"):
        (root / sub).mkdir(exist_ok=True)

    sequence_ids = [f"Q{i + 1:06d}" for i in range(n_seq)]
    mixture_ids = [f"M{i // 2 + 1:06d}" for i in range(n_seq)]
    slow_names = (
        "V_NDIR_CO2",
        "V_TCS",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    )
    phases = ("baseline", "exposure", "steady", "recovery")
    path_cycle = np.linspace(0.20, 0.28, n_t, dtype=np.float64)

    spec = WaveformSpec(
        noise_std_v=0.0,
        trigger_jitter_std_s=0.0,
        waveform_dtype="int16",
        per_timestep_scale=True,
    )
    n_samples = int(spec.waveform_samples)
    wave_ab = np.empty((n_seq, n_t, n_samples), dtype=np.int16)
    wave_ba = np.empty((n_seq, n_t, n_samples), dtype=np.int16)
    scale_ab = np.empty((n_seq, n_t), dtype=np.float32)
    scale_ba = np.empty((n_seq, n_t), dtype=np.float32)
    slow = np.zeros((n_seq, n_t, len(slow_names)), dtype=np.float32)
    c_true = np.empty((n_seq, n_t), dtype=np.float32)

    cond_rows = []
    phase_rows = []
    for s_idx, (sid, mid) in enumerate(zip(sequence_ids, mixture_ids, strict=True)):
        v_path = 0.5 if (s_idx // 2) % 2 == 0 else 2.0
        cond_rows.append(
            {
                "sequence_id": sid,
                "mixture_id": mid,
                "v_path_m_per_s": f"{v_path:.6f}",
                "x_CO2": "1.0",
                "x_O2": "20.0",
                "x_N2": "79.0",
                "flow_scenario": "steady",
            }
        )
        for t_idx in range(n_t):
            path_m = float(path_cycle[t_idx])
            phase = phases[t_idx % len(phases)]
            phase_rows.append(
                {
                    "sequence_id": sid,
                    "timestep": str(t_idx),
                    "phase_id": phase,
                }
            )
            meas = simulate_bidirectional_waveform_measurement(
                x_h2=0.0,
                x_ch4=0.0,
                x_co2=1.0,
                x_n2=79.0,
                t_c=20.0,
                p_mpa=0.101325,
                h_rh=40.0,
                l_m=path_m,
                seed=20260723 + s_idx * 100 + t_idx,
                spec=spec,
                v_path_m_per_s=v_path,
                sound_speed_fn=hidden_sound_speed_v2,
                attenuation_fn=hidden_attenuation_v2,
                extra_gas_kwargs={"x_o2": 20.0},
            )
            ab = meas["ab"]
            ba = meas["ba"]
            wave_ab[s_idx, t_idx] = np.asarray(ab["waveform_int"], dtype=np.int16)
            wave_ba[s_idx, t_idx] = np.asarray(ba["waveform_int"], dtype=np.int16)
            scale_ab[s_idx, t_idx] = float(ab["scale_factor"])
            scale_ba[s_idx, t_idx] = float(ba["scale_factor"])
            slow[s_idx, t_idx] = np.asarray(
                [0.1, 0.2, 20.0, 0.101325, 40.0, path_m, 0.0], dtype=np.float32
            )
            c_true[s_idx, t_idx] = float(meas["sound_speed_m_per_s"])

    np.save(root / "sequences" / "ultrasonic_ab_int16.npy", wave_ab)
    np.save(root / "sequences" / "ultrasonic_ba_int16.npy", wave_ba)
    np.save(root / "sequences" / "ultrasonic_ab_scale.npy", scale_ab)
    np.save(root / "sequences" / "ultrasonic_ba_scale.npy", scale_ba)
    np.save(root / "sequences" / "slow.npy", slow)
    np.save(root / "sequences" / "ultrasonic_sound_speed_m_per_s.npy", c_true)
    np.save(root / "labels" / "y.npy", np.tile(np.array([1.0, 20.0, 79.0], dtype=np.float32), (n_seq, 1)))
    np.save(root / "metadata" / "sequence_ids.npy", np.asarray(sequence_ids))
    np.save(root / "metadata" / "label_names.npy", np.asarray(["x_CO2", "x_O2", "x_N2"]))
    np.save(root / "metadata" / "slow_channel_names.npy", np.asarray(slow_names))

    (root / "metadata" / "waveform_spec.json").write_text(
        json.dumps(
            {
                "ultrasonic": {
                    "sample_rate_hz": float(spec.sample_rate_hz),
                    "daq_full_scale_v": float(spec.daq_full_scale_v),
                    "center_frequency_hz": float(spec.center_frequency_hz),
                    "waveform_dtype": "int16",
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "composition_scheme": COMPOSITION_SCHEME,
                "dataset_slug": "tv3-bidir-perf-hand",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (root / "condition_grid_sequence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cond_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cond_rows)
    with (root / "sequences" / "slow_sequence_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence_id", "timestep", "phase_id"])
        writer.writeheader()
        writer.writerows(phase_rows)

    # 4/1/1/2 split by sequence index (train-heavy for template/calibration).
    split_map = {
        "train": list(range(0, 4)),
        "val": [4],
        "test": [5],
        "extrapolation": [6, 7],
    }
    for split_name, indices in split_map.items():
        with (root / "splits" / f"{split_name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence_id", "mixture_id"])
            writer.writeheader()
            for i in indices:
                writer.writerow({"sequence_id": sequence_ids[i], "mixture_id": mixture_ids[i]})
    (root / "splits" / "split_summary.json").write_text(
        json.dumps({"split_policy": "hand_built_perf_v1"}, indent=2),
        encoding="utf-8",
    )
    return root


def test_default_bidir_workers_and_chunking():
    assert default_bidir_workers(1) == 1
    assert default_bidir_workers(100) >= 1
    chunks = _chunk_indices(range(10), workers=3)
    assert [len(c) for c in chunks] == [4, 4, 2]
    assert [x for chunk in chunks for x in chunk] == list(range(10))


def test_frame_cache_workers_match_serial(tmp_path: Path):
    dataset_dir = _write_minimal_bidir_dataset(tmp_path / "data")
    serial = build_tv3_bidir_feature_cache(
        dataset_dir,
        cache_dir=tmp_path / "cache_serial",
        overwrite=True,
        workers=1,
    )
    parallel = build_tv3_bidir_feature_cache(
        dataset_dir,
        cache_dir=tmp_path / "cache_parallel",
        overwrite=True,
        workers=2,
    )
    assert serial["build_signature"] == parallel["build_signature"]
    serial_cal = json.loads(
        (Path(serial["cache_dir"]) / "session_delay_calibration.json").read_text(encoding="utf-8")
    )
    parallel_cal = json.loads(
        (Path(parallel["cache_dir"]) / "session_delay_calibration.json").read_text(encoding="utf-8")
    )
    assert serial_cal == parallel_cal

    names = (
        list(FRAME_ARRAY_SPECS)
        + [
            "ultrasonic_sound_speed_ab_raw_dsp_m_per_s",
            "ultrasonic_sound_speed_ba_raw_dsp_m_per_s",
        ]
        + list(SEQ_SCALAR_SPECS)
        + ["template_ab", "template_ba"]
    )
    for name in names:
        a = np.load(Path(serial["cache_dir"]) / f"{name}.npy")
        b = np.load(Path(parallel["cache_dir"]) / f"{name}.npy")
        np.testing.assert_array_equal(a, b)


def test_frame_cache_skip_if_exists(tmp_path: Path):
    dataset_dir = _write_minimal_bidir_dataset(tmp_path / "data")
    cache_dir = tmp_path / "cache"
    first = build_tv3_bidir_feature_cache(dataset_dir, cache_dir=cache_dir, overwrite=True, workers=1)
    assert first.get("reused") is False
    second = build_tv3_bidir_feature_cache(dataset_dir, cache_dir=cache_dir, overwrite=False, workers=1)
    assert second.get("reused") is True
    assert second["build_signature"] == first["build_signature"]

    reused = _try_reuse_existing_cache(cache_dir, overwrite=False)
    assert reused is not None
    assert reused["feature_builder"] == FEATURE_BUILDER
    assert reused["schema_version"] == BIDIR_RAW_DSP_SCHEMA_VERSION


def test_shared_slow_block_matches_inline_assemble():
    n, t = 4, 8
    rng = np.random.default_rng(1)
    slow = rng.normal(size=(n, t, 7)).astype(np.float32)
    slow_names = (
        "V_NDIR_CO2",
        "V_TCS",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    )
    sequence_ids = tuple(f"Q{i}" for i in range(n))
    phase_lookup = {
        sid: tuple(["baseline", "exposure", "steady", "recovery"] * 2) for sid in sequence_ids
    }
    labels = np.tile(np.array([1.0, 20.0, 79.0], dtype=np.float32), (n, 1))
    specs = arm_specs()
    frames = {name: rng.normal(size=(n, t)).astype(np.float32) for name in specs["A3"].frame_arrays}
    scalars = {name: rng.normal(size=(n,)).astype(np.float32) for name in specs["A3"].sequence_scalars}
    shared = compute_shared_slow_windowed_block(
        slow=slow,
        slow_channel_names=slow_names,
        sequence_ids=sequence_ids,
        phase_lookup=phase_lookup,
    )
    with_shared = assemble_arm_feature_matrix(
        slow=slow,
        slow_channel_names=slow_names,
        sequence_ids=sequence_ids,
        labels=labels,
        label_names=("x_CO2", "x_O2", "x_N2"),
        phase_lookup=phase_lookup,
        frame_arrays=frames,
        sequence_scalars=scalars,
        arm=specs["A3"],
        shared_slow_block=shared,
    )
    inline = assemble_arm_feature_matrix(
        slow=slow,
        slow_channel_names=slow_names,
        sequence_ids=sequence_ids,
        labels=labels,
        label_names=("x_CO2", "x_O2", "x_N2"),
        phase_lookup=phase_lookup,
        frame_arrays=frames,
        sequence_scalars=scalars,
        arm=specs["A3"],
        shared_slow_block=None,
    )
    assert with_shared.feature_names == inline.feature_names
    np.testing.assert_array_equal(with_shared.x, inline.x)


def test_build_arm_feature_caches_writes_all_arms(tmp_path: Path):
    dataset_dir = _write_minimal_bidir_dataset(tmp_path / "data")
    frame = build_tv3_bidir_feature_cache(
        dataset_dir,
        cache_dir=tmp_path / "frames",
        overwrite=True,
        workers=1,
    )
    manifests = build_arm_feature_caches(
        dataset_dir, ("A1", "A3", "A5"), frame_cache_dir=frame["cache_dir"]
    )
    assert set(manifests) == {"A1", "A3", "A5"}
    for arm_id, manifest in manifests.items():
        assert manifest["feature_count"] > 0
        names_path = (
            Path(dataset_dir)
            / "features"
            / "rocket"
            / arm_specs()[arm_id].feature_builder
            / "feature_names.json"
        )
        assert names_path.is_file()
