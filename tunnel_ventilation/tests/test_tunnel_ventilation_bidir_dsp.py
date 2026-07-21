"""F3 tests: raw_dsp_bidirectional_v1 estimator gates and deploy-path hygiene."""
from __future__ import annotations

import numpy as np
import pytest

from tv3.ml.bidir_features import (
    FEATURE_BUILDER,
    assert_no_oracle_inputs,
    calibrate_session_delay_s,
    extract_bidir_frame_pair,
    freeze_session_delay_calibration,
    reciprocity_residual_s,
    true_fixed_delay_s,
)
from tv3.ml.raw_dsp_features import RawDSPConfig, exact_simulator_template
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
)
from tv3.sim.generation.tunnel_ventilation.flow_physics import (
    reciprocal_sum_path_velocity_m_per_s,
    reciprocal_sum_sound_speed_m_per_s,
)
from tv3.sim.generation.waveforms import WaveformSpec, simulate_bidirectional_waveform_measurement

_TV3_KW = dict(
    x_h2=0.0,
    x_ch4=0.0,
    x_co2=1.0,
    x_n2=79.0,
    t_c=20.0,
    p_mpa=0.101325,
    h_rh=40.0,
)
_EXTRA = {"x_o2": 20.0}


def _quiet_spec(**overrides) -> WaveformSpec:
    base = dict(
        noise_std_v=0.0,
        trigger_jitter_std_s=0.0,
        waveform_dtype="int16",
        per_timestep_scale=True,
    )
    base.update(overrides)
    return WaveformSpec(**base)


def test_feature_builder_name_matches_schema():
    assert FEATURE_BUILDER == "raw_dsp_bidirectional_v1"


def test_assert_no_oracle_inputs_rejects_true_arrays():
    with pytest.raises(ValueError, match="oracle"):
        assert_no_oracle_inputs(["ultrasonic_ab", "ultrasonic_tof_true_ab_s"])
    assert_no_oracle_inputs(["ultrasonic_ab", "ultrasonic_ba", "slow"])


def test_true_fixed_delay_splits_asymmetry():
    tau_ab = true_fixed_delay_s(
        system_delay_s=80e-6, cable_delay_s=2e-6, delay_asymmetry_s=0.2e-6, direction="ab"
    )
    tau_ba = true_fixed_delay_s(
        system_delay_s=80e-6, cable_delay_s=2e-6, delay_asymmetry_s=0.2e-6, direction="ba"
    )
    assert tau_ab == pytest.approx(82.1e-6)
    assert tau_ba == pytest.approx(81.9e-6)


def test_session_delay_calibration_recovers_intercept():
    rng = np.random.default_rng(0)
    tau_true = 82e-6
    c_eff = 340.0
    tofs = []
    paths = []
    phases = []
    accepted = []
    for _ in range(5):
        l_values = np.asarray([0.18, 0.20, 0.22, 0.25, 0.28], dtype=np.float64)
        noise = rng.normal(0.0, 0.02e-6, size=l_values.shape)
        tof = tau_true + l_values / c_eff + noise
        tofs.append(tof)
        paths.append(l_values)
        phases.append(tuple("steady" for _ in l_values))
        accepted.append(np.ones(l_values.shape, dtype=bool))
    tau_hat, c_hat, n_used = calibrate_session_delay_s(tofs, paths, phases, accepted)
    assert n_used == 5
    assert abs(tau_hat - tau_true) <= 0.10e-6
    assert abs(c_hat - c_eff) <= 0.5


def test_reciprocal_sum_zero_noise_frame_pair():
    spec = _quiet_spec()
    template = exact_simulator_template(
        {
            "center_frequency_hz": spec.center_frequency_hz,
            "burst_cycles": spec.burst_cycles,
            "transducer_bandwidth_hz": spec.transducer_bandwidth_hz,
            "transducer_ringdown_cycles": spec.transducer_ringdown_cycles,
            "sample_rate_hz": spec.sample_rate_hz,
        }
    )
    config = RawDSPConfig(
        sample_rate_hz=spec.sample_rate_hz,
        carrier_frequency_hz=spec.center_frequency_hz,
    )
    l_m = 0.25
    v_path = 2.0
    meas = simulate_bidirectional_waveform_measurement(
        **_TV3_KW,
        l_m=l_m,
        seed=7,
        spec=spec,
        v_path_m_per_s=v_path,
        delay_asymmetry_s=0.0,
        jitter_correlation="independent",
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs=_EXTRA,
    )
    tau = float(spec.system_delay_s + spec.cable_delay_s)
    pair = extract_bidir_frame_pair(
        np.asarray(meas["ab"]["waveform_float"], dtype=np.float64),
        np.asarray(meas["ba"]["waveform_float"], dtype=np.float64),
        path_length_m=l_m,
        template_ab=template,
        template_ba=template,
        tau_ab_s=tau,
        tau_ba_s=tau,
        daq_full_scale_v=float(spec.daq_full_scale_v),
        config=config,
    )
    assert pair.accepted_pair
    assert abs(pair.sound_speed_m_per_s - float(meas["sound_speed_m_per_s"])) <= 0.05
    assert abs(pair.v_path_m_per_s - v_path) <= 0.05
    assert abs(pair.reciprocity_residual_s) <= 1e-15

    t_ab = float(meas["tof_true_ab_s"])
    t_ba = float(meas["tof_true_ba_s"])
    assert reciprocal_sum_sound_speed_m_per_s(l_m, t_ab, t_ba) == pytest.approx(
        float(meas["sound_speed_m_per_s"]), abs=1e-9
    )
    assert reciprocal_sum_path_velocity_m_per_s(l_m, t_ab, t_ba) == pytest.approx(v_path, abs=1e-9)


def test_reciprocity_residual_vs_sequence_ref():
    residual = reciprocity_residual_s(0.25, 340.0, 340.1)
    assert residual == pytest.approx(0.25 * (1 / 340.0 - 1 / 340.1), abs=1e-15)


def test_shared_midpair_delay_beats_direction_split():
    """With zero asymmetry, shared mid-pair τ cancels opposite AB/BA intercept noise."""
    rng = np.random.default_rng(1)
    tau_true = 82e-6
    tof_ab_seqs = []
    tof_ba_seqs = []
    paths = []
    phases = []
    accepted = []
    for _ in range(20):
        l_values = np.asarray([0.18, 0.20, 0.22, 0.25, 0.28], dtype=np.float64)
        v = float(rng.uniform(-3.0, 3.0))
        c = 342.0
        noise_ab = rng.normal(0.0, 0.5e-6, size=l_values.shape)
        noise_ba = rng.normal(0.0, 0.5e-6, size=l_values.shape)
        t_ab = tau_true + l_values / (c + v) + noise_ab
        t_ba = tau_true + l_values / (c - v) + noise_ba
        tof_ab_seqs.append(t_ab)
        tof_ba_seqs.append(t_ba)
        paths.append(l_values)
        phases.append(tuple("steady" for _ in l_values))
        accepted.append(np.ones(l_values.shape, dtype=bool))

    from tv3.ml.bidir_features import calibrate_session_delay_shared_s

    tau_hat, _c_hat, n_used = calibrate_session_delay_shared_s(
        tof_ab_seqs, tof_ba_seqs, paths, phases, accepted, accepted
    )
    assert n_used == 20
    assert abs(tau_hat - tau_true) <= 0.25e-6


def test_freeze_session_delay_digest_stable():
    a = freeze_session_delay_calibration(
        tau_ab_s=82e-6,
        tau_ba_s=82e-6,
        c_eff_ab_m_per_s=340.0,
        c_eff_ba_m_per_s=340.0,
        n_sequences_ab=3,
        n_sequences_ba=3,
    )
    b = freeze_session_delay_calibration(
        tau_ab_s=82e-6,
        tau_ba_s=82e-6,
        c_eff_ab_m_per_s=340.0,
        c_eff_ba_m_per_s=340.0,
        n_sequences_ab=3,
        n_sequences_ba=3,
    )
    assert a.digest == b.digest
    assert len(a.digest) == 64
