import pytest

from sim.generation.waveforms import (
    FiberMicSpec,
    WaveformSpec,
    simulate_fiber_mic_measurement,
    simulate_waveform_measurement,
)


_STATE = {
    "x_h2": 10.0,
    "x_ch4": 55.0,
    "x_co2": 5.0,
    "x_n2": 30.0,
    "t_c": 25.0,
    "p_mpa": 1.0,
    "h_rh": 50.0,
    "l_m": 0.3,
}


def test_ultrasonic_waveform_uses_observed_tof_and_delay_correction():
    spec = WaveformSpec(noise_std_v=0.0, trigger_jitter_std_s=0.0)

    result = simulate_waveform_measurement(**_STATE, seed=123, spec=spec)

    assert result["tof_observed_s"] == pytest.approx(
        result["tof_true_s"] + spec.system_delay_s + spec.cable_delay_s,
        rel=1e-12,
    )
    assert result["sound_speed_estimated_m_per_s"] == pytest.approx(
        result["sound_speed_m_per_s"],
        rel=1e-12,
    )
    assert result["peak_index"] == round(result["tof_observed_s"] * spec.sample_rate_hz)
    assert result["tof_quality"] > 0.99
    assert result["tof_accepted"] == 1
    assert result["scale_factor"] == pytest.approx(spec.daq_full_scale_v / spec.adc_max_int16)


def test_fiber_mic_waveform_runs_probe_pressure_to_phase_demod_chain():
    spec = FiberMicSpec(max_reflections=0)

    result = simulate_fiber_mic_measurement(**_STATE, seed=456, spec=spec)

    assert result["probe_pressure_peak_pa"] > 0.0
    assert result["phase_peak_rad"] == pytest.approx(result["probe_pressure_peak_pa"] * spec.probe.pressure_sensitivity_rad_per_pa)
    assert result["demod_peak_v"] > 0.0
    assert result["scale_factor"] == pytest.approx(spec.probe.daq_full_scale_v / spec.adc_max_int16)
    assert spec.model_name == "fiber_interferometric_proxy_v1"
    assert spec.fiber_optical_demodulation_model == "linear_phase_demodulation_proxy_v1"
