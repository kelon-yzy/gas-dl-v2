"""F2 tests: bidirectional smoke benchmark generation + packaging contract."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.pipeline.generate_tunnel_ventilation_benchmark import BIDIR_SMOKE_PRESET, main as gen_main
from tv3.sim.core.tunnel_ventilation_bidir_schema import (
    COMPOSITION_SCHEME,
    SCHEMA_VERSION,
    SIM_REVISION_TAG,
)
from tv3.sim.generation.tunnel_ventilation.benchmark import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


def test_bidir_smoke_preset_enables_bidirectional():
    assert BIDIR_SMOKE_PRESET["bidirectional"] is True
    assert BIDIR_SMOKE_PRESET["dataset"] == "tv3-bidir-smoke"
    assert BIDIR_SMOKE_PRESET["skip_fiber_mic"] is True


def test_bidir_formal_6000_preset_shape():
    from tv3.pipeline.generate_tunnel_ventilation_benchmark import BIDIR_FORMAL_6000_PRESET

    assert BIDIR_FORMAL_6000_PRESET["dataset"] == "tv3-bidir-6000"
    assert BIDIR_FORMAL_6000_PRESET["sequences"] == 6000
    assert BIDIR_FORMAL_6000_PRESET["bidirectional"] is True
    assert BIDIR_FORMAL_6000_PRESET["timesteps"] == 512


def test_generate_tiny_bidir_benchmark(tmp_path: Path):
    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-bidir-unit",
        sequence_count=4,
        seed=20260721,
        timesteps=4,
        dt_s=0.5,
        storage="memmap",
        workers=1,
        skip_fiber_mic=True,
        bidirectional=True,
        split_strategy="random",
    )
    result = generate_tunnel_ventilation_benchmark_dataset(tmp_path, spec)
    out = Path(result["output_dir"])
    assert result["bidirectional"] is True
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["composition_scheme"] == COMPOSITION_SCHEME
    assert result["validation"]["status"] == "pass"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["sim_revision"]["tag"] == SIM_REVISION_TAG
    assert "ultrasonic_ab" in manifest["shapes"]
    assert "ultrasonic_ba" in manifest["shapes"]
    assert "ultrasonic" not in manifest["shapes"]
    assert "V_NDIR_CH4" not in manifest["slow_channels"]

    assert (out / "sequences" / "ultrasonic_ab_int16.npy").is_file()
    assert (out / "sequences" / "ultrasonic_ba_int16.npy").is_file()
    assert (out / "sequences" / "ultrasonic_alpha_true_npm.npy").is_file()
    assert not (out / "sequences" / "ultrasonic_int16.npy").is_file()

    ab = np.load(out / "sequences" / "ultrasonic_ab_int16.npy", mmap_mode="r")
    ba = np.load(out / "sequences" / "ultrasonic_ba_int16.npy", mmap_mode="r")
    assert ab.shape[0] == 4
    assert ba.shape == ab.shape
    assert ab.dtype == np.int16

    cond = (out / "condition_grid_sequence.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "v_path_m_per_s" in cond
    assert "flow_scenario" in cond


def test_cli_bidirectional_flag_writes_dataset(tmp_path: Path):
    code = gen_main(
        [
            "--output-root",
            str(tmp_path),
            "--dataset",
            "tv3-bidir-cli",
            "--bidirectional",
            "--skip-fiber-mic",
            "--sequences",
            "4",
            "--timesteps",
            "4",
            "--workers",
            "1",
            "--seed",
            "20260721",
        ]
    )
    assert code == 0
    out = tmp_path / "tv3-bidir-cli"
    assert out.is_dir()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_slug"] == "tv3-bidir-cli"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["sim_revision"]["tag"] == SIM_REVISION_TAG
