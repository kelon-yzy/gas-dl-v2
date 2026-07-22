"""Wide composition domain (CO2 0.03–10 / O2 15–25): F0'+F1 gates."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from tv3.pipeline.generate_tunnel_ventilation_benchmark import (
    apply_composition_domain_dataset_slug,
)
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    PROCESSING_PARAMS,
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
    main_sensor_features,
)
from tv3.sim.generation.tunnel_ventilation.bidir_registry import (
    audit_f0_gate,
    audit_f0_gate_wide,
    load_f0_registry,
    load_f0_registry_wide,
)
from tv3.sim.generation.tunnel_ventilation.conditions import (
    TUNNEL_VENTILATION_RANGES,
    WIDE_COMPOSITION_RANGES,
    generate_tunnel_ventilation_bidir_condition_rows,
    generate_tunnel_ventilation_condition_rows,
    resolve_composition_ranges,
)
from tv3.sim.generation.tunnel_ventilation.flow_physics import (
    bidirectional_transit_times_s,
    reciprocal_sum_path_velocity_m_per_s,
    reciprocal_sum_sound_speed_m_per_s,
)
from tv3.sim.generation.waveforms import (
    WaveformSpec,
    simulate_bidirectional_waveform_measurement,
    simulate_waveform_measurement,
)


def test_wide_ranges_constants_and_resolve():
    assert TUNNEL_VENTILATION_RANGES.co2 == (0.03, 5.00)
    assert TUNNEL_VENTILATION_RANGES.o2 == (18.00, 21.20)
    assert WIDE_COMPOSITION_RANGES.co2 == (0.03, 10.00)
    assert WIDE_COMPOSITION_RANGES.o2 == (15.00, 25.00)
    assert WIDE_COMPOSITION_RANGES.n2_min == 65.00
    assert WIDE_COMPOSITION_RANGES.n2_max == 84.97
    assert resolve_composition_ranges("narrow") is TUNNEL_VENTILATION_RANGES
    assert resolve_composition_ranges("wide") is WIDE_COMPOSITION_RANGES
    with pytest.raises(ValueError, match="composition_domain"):
        resolve_composition_ranges("ambient")


def test_narrow_default_unchanged_for_unidirectional_and_bidir():
    uni = generate_tunnel_ventilation_condition_rows(8, seed=11)
    bidir = generate_tunnel_ventilation_bidir_condition_rows(8, seed=11)
    for rows in (uni, bidir):
        for row in rows:
            co2 = float(row["x_CO2"])
            o2 = float(row["x_O2"])
            n2 = float(row["x_N2"])
            assert TUNNEL_VENTILATION_RANGES.co2[0] - 1e-6 <= co2 <= TUNNEL_VENTILATION_RANGES.co2[1] + 1e-6
            assert TUNNEL_VENTILATION_RANGES.o2[0] - 1e-6 <= o2 <= TUNNEL_VENTILATION_RANGES.o2[1] + 1e-6
            assert abs(co2 + o2 + n2 - 100.0) <= 1e-5


def test_wide_lhs_covers_corners_and_closure():
    rows = generate_tunnel_ventilation_bidir_condition_rows(
        64,
        seed=20260722,
        ranges=WIDE_COMPOSITION_RANGES,
    )
    co2 = np.array([float(r["x_CO2"]) for r in rows])
    o2 = np.array([float(r["x_O2"]) for r in rows])
    n2 = np.array([float(r["x_N2"]) for r in rows])
    assert co2.min() >= WIDE_COMPOSITION_RANGES.co2[0] - 1e-6
    assert co2.max() <= WIDE_COMPOSITION_RANGES.co2[1] + 1e-6
    assert o2.min() >= WIDE_COMPOSITION_RANGES.o2[0] - 1e-6
    assert o2.max() <= WIDE_COMPOSITION_RANGES.o2[1] + 1e-6
    assert n2.min() >= WIDE_COMPOSITION_RANGES.n2_min - 1e-6
    assert n2.max() <= WIDE_COMPOSITION_RANGES.n2_max + 1e-6
    assert np.allclose(co2 + o2 + n2, 100.0, atol=1e-5)
    # LHS should reach near both ends of the widened axes
    assert co2.max() > 7.0
    assert o2.min() < 17.0
    assert o2.max() > 23.0


@pytest.mark.parametrize(
    "x_co2,x_o2",
    [
        (10.0, 15.0),
        (10.0, 25.0),
        (0.03, 15.0),
        (0.03, 25.0),
        (1.0, 20.0),  # prior-table anchor still inside wide
    ],
)
def test_wide_corner_sound_speed_finite_and_closure(x_co2: float, x_o2: float):
    x_n2 = 100.0 - x_co2 - x_o2
    assert x_n2 >= WIDE_COMPOSITION_RANGES.n2_min - 1e-9
    c = hidden_sound_speed_v2(
        x_h2=0.0, x_ch4=0.0, x_co2=x_co2, x_n2=x_n2, t_c=20.0, x_o2=x_o2
    )
    assert math.isfinite(c) and c > 300.0
    t_ab, t_ba = bidirectional_transit_times_s(0.25, c, 1.0)
    assert abs(reciprocal_sum_sound_speed_m_per_s(0.25, t_ab, t_ba) - c) <= 1e-9
    assert abs(reciprocal_sum_path_velocity_m_per_s(0.25, t_ab, t_ba) - 1.0) <= 1e-9


def test_wide_co2_10pct_ndir_and_acoustic_amplitude_floor():
    """Worst-case CO2=10%, L=0.3m: NDIR unsaturated; acoustic amp above floor."""
    x_co2, x_o2, x_n2 = 10.0, 20.0, 70.0
    l_m = 0.3
    att = hidden_attenuation_v2(
        x_h2=0.0,
        x_ch4=0.0,
        x_co2=x_co2,
        x_n2=x_n2,
        t_c=20.0,
        p_mpa=0.101325,
        h_rh=40.0,
        x_o2=x_o2,
    )
    amp = math.exp(-float(att["alpha_true_v2"]) * l_m)
    assert amp > 0.1  # above 0.1 V-ish floor used by NDIR clamp analogy / plan noise floor
    cond = {
        "x_CO2": str(x_co2),
        "x_O2": str(x_o2),
        "x_N2": str(x_n2),
        "T_C": "20.0",
        "P_MPa": "0.101325",
        "H_RH": "40.0",
        "L_m": str(l_m),
    }
    feat = main_sensor_features(cond, random.Random(0))
    assert feat["ndir_co2_saturated"] is False
    assert float(feat["absorption_co2_true"]) < PROCESSING_PARAMS["optical_saturation_absorbance"]
    assert float(feat["V_NDIR_CO2"]) > 0.1


def test_wide_zero_flow_bidir_matches_unidirectional():
    """At wide corner, v=0 AB shot equals unidirectional; AB/BA true TOF identical."""
    spec = WaveformSpec(
        noise_std_v=0.0,
        trigger_jitter_std_s=0.0,
        waveform_dtype="int16",
        per_timestep_scale=True,
    )
    kw = dict(
        x_h2=0.0,
        x_ch4=0.0,
        x_co2=10.0,
        x_n2=70.0,
        t_c=20.0,
        p_mpa=0.101325,
        h_rh=40.0,
        l_m=0.3,
        seed=4242,
        spec=spec,
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs={"x_o2": 20.0},
    )
    with_v0 = simulate_waveform_measurement(**kw, path_velocity_m_per_s=0.0)
    uni = simulate_waveform_measurement(**kw)
    for key in ("tof_true_s", "tof_observed_s", "peak_index", "sound_speed_m_per_s", "alpha_true_npm"):
        assert with_v0[key] == uni[key]
    np.testing.assert_array_equal(with_v0["waveform_int"], uni["waveform_int"])

    pair = simulate_bidirectional_waveform_measurement(**kw, v_path_m_per_s=0.0)
    assert abs(float(pair["tof_true_ab_s"]) - float(pair["tof_true_ba_s"])) <= 1e-15
    assert abs(float(pair["reciprocal_sum_path_velocity_m_per_s"])) <= 1e-12
    assert abs(float(pair["sound_speed_m_per_s"]) - float(uni["sound_speed_m_per_s"])) <= 1e-12


def test_dataset_slug_wide_suffix():
    assert apply_composition_domain_dataset_slug("tv3-bidir-smoke", "narrow") == "tv3-bidir-smoke"
    assert apply_composition_domain_dataset_slug("tv3-bidir-smoke", "wide") == "tv3-bidir-smoke-wide"
    assert apply_composition_domain_dataset_slug("tv3-bidir-smoke-wide", "wide") == "tv3-bidir-smoke-wide"


def test_validate_spec_write_once_slug_isolation():
    from tv3.sim.generation.tunnel_ventilation.benchmark import (
        TunnelVentilationBenchmarkGenerationSpec,
        _validate_spec,
    )

    with pytest.raises(ValueError, match="-wide"):
        _validate_spec(
            TunnelVentilationBenchmarkGenerationSpec(
                dataset_slug="tv3-bidir-6000",
                sequence_count=8,
                seed=1,
                timesteps=4,
                dt_s=0.5,
                bidirectional=True,
                composition_domain="wide",
            )
        )
    with pytest.raises(ValueError, match="composition_domain='narrow'"):
        _validate_spec(
            TunnelVentilationBenchmarkGenerationSpec(
                dataset_slug="tv3-bidir-6000-wide",
                sequence_count=8,
                seed=1,
                timesteps=4,
                dt_s=0.5,
                bidirectional=True,
                composition_domain="narrow",
            )
        )


def test_f2_registry_check_requires_f0_passed(tmp_path: Path):
    import importlib.util

    from tv3.sim.generation.tunnel_ventilation.bidir_registry import (
        default_config_dir,
        sha256_file,
    )

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tv3_bidir_f2_benchmark_audit.py"
    spec = importlib.util.spec_from_file_location("tv3_bidir_f2", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    registry = default_config_dir() / "parameter_registry_wide.json"
    actual = sha256_file(registry)
    f0 = tmp_path / "f0_verdict.json"
    f0.write_text(
        json.dumps(
            {
                "audit": {
                    "passed": False,
                    "verdict": "inconclusive_parameter_bounds",
                    "registry_sha256": actual,
                }
            }
        ),
        encoding="utf-8",
    )
    matched, info = mod._check_registry_sha256_matches_f0(f0, composition_domain="wide")
    assert matched is False
    assert info["hash_matched"] is True
    assert info["f0_passed"] is False


def test_f2_registry_check_requires_f0_passed(tmp_path: Path):
    import importlib.util
    import hashlib

    from tv3.sim.generation.tunnel_ventilation.bidir_registry import (
        default_config_dir,
        sha256_file,
    )

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tv3_bidir_f2_benchmark_audit.py"
    spec = importlib.util.spec_from_file_location("tv3_bidir_f2_audit", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    registry_path = default_config_dir() / "parameter_registry_wide.json"
    actual = sha256_file(registry_path)
    f0 = tmp_path / "f0_verdict.json"
    f0.write_text(
        json.dumps(
            {
                "audit": {
                    "passed": False,
                    "verdict": "inconclusive_parameter_bounds",
                    "registry_sha256": actual,
                }
            }
        ),
        encoding="utf-8",
    )
    matched, info = mod._check_registry_sha256_matches_f0(f0, composition_domain="wide")
    assert matched is False
    assert info["hash_matched"] is True
    assert info["f0_passed"] is False


def test_f0_narrow_registry_untouched_and_wide_gate_passes():
    narrow = load_f0_registry()
    wide = load_f0_registry_wide()
    assert Path(narrow["path"]).name == "parameter_registry.json"
    assert Path(wide["path"]).name == "parameter_registry_wide.json"
    assert narrow["sha256"] != wide["sha256"]
    # Frozen narrow hash from F0 (2026-07-21)
    assert narrow["sha256"].startswith("dc61d9e7")
    audit_n = audit_f0_gate()
    assert audit_n["passed"] is True
    audit_w = audit_f0_gate_wide()
    assert audit_w["passed"] is True, audit_w["issues"]
    assert audit_w["composition_domain"] == "wide_hazard_v1"
    assert audit_w["verdict"] == "f0_registry_frozen"
    ranges = wide["registry"]["composition_anchor"]["composition_ranges"]
    assert ranges["x_CO2"]["max"] == 10.0
    assert ranges["x_O2"]["min"] == 15.0


def test_generate_tiny_bidir_wide_benchmark(tmp_path: Path):
    from tv3.sim.generation.tunnel_ventilation.benchmark import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-bidir-unit-wide",
        sequence_count=8,
        seed=20260722,
        timesteps=4,
        dt_s=0.5,
        storage="memmap",
        workers=1,
        skip_fiber_mic=True,
        bidirectional=True,
        composition_domain="wide",
        split_strategy="random",
    )
    result = generate_tunnel_ventilation_benchmark_dataset(tmp_path, spec)
    assert result["composition_domain"] == "wide"
    assert result["validation"]["status"] == "pass"
    out = Path(result["output_dir"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sim_revision"]["composition_domain"] == "wide"
    assert manifest["sim_revision"]["composition_domain_tag"] == "wide_hazard_v1"
    assert manifest["sim_revision"]["composition_ranges"]["x_CO2"] == [0.03, 10.0]
    assert manifest["sim_revision"]["composition_ranges"]["x_O2"] == [15.0, 25.0]
    assert manifest["sim_revision"]["f0_registry_file"] == "parameter_registry_wide.json"
    assert len(manifest["sim_revision"]["f0_registry_sha256"]) == 64
    assert manifest["sim_revision"]["f0_registry_sha256"].startswith("c7137f54")
    import csv

    with (out / "condition_grid_sequence.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    o2 = [float(r["x_O2"]) for r in rows]
    co2 = [float(r["x_CO2"]) for r in rows]
    assert min(o2) >= 15.0 - 1e-6 and max(o2) <= 25.0 + 1e-6
    assert min(co2) >= 0.03 - 1e-6 and max(co2) <= 10.0 + 1e-6


def test_wide_rejected_on_unidirectional_spec():
    from tv3.sim.generation.tunnel_ventilation.benchmark import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug="tv3-unit-bad-wide",
        sequence_count=4,
        seed=1,
        timesteps=4,
        bidirectional=False,
        composition_domain="wide",
    )
    with pytest.raises(ValueError, match="F-line only"):
        generate_tunnel_ventilation_benchmark_dataset(Path("."), spec)
