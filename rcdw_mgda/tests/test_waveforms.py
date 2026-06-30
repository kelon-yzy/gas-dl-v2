"""测试 RCDW 超声 + 光纤麦克风波形仿真。

对应方案 §5.6 / §11.1。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rcdw.sim.generation.waveforms import (
    FiberMicSpec,
    WaveformSpec,
    generate_burst_pulse,
    simulate_fiber_mic_measurement,
    simulate_waveform_measurement,
    transducer_response_pulse,
)


def _default_kwargs():
    return dict(
        x_o2=10.0,
        x_co2=5.0,
        x_n2=85.0,
        t_c=25.0,
        p_mpa=0.1,
        h_rh=40.0,
        l_m=0.5,
        seed=42,
    )


def test_waveform_spec_samples():
    spec = WaveformSpec()
    assert spec.waveform_samples == int(round(200000 * 0.005))  # 1000


def test_burst_pulse_shape_and_amplitude():
    pulse = generate_burst_pulse(
        center_frequency_hz=40000.0, burst_cycles=8, sample_rate_hz=200000
    )
    assert pulse.dtype == np.float32
    assert pulse.shape[0] == int(round(8 * 200000 / 40000.0))
    assert pulse.max() > 0 and pulse.min() < 0


def test_transducer_response_pulse_normalized():
    pulse = transducer_response_pulse(WaveformSpec())
    peak = float(np.max(np.abs(pulse)))
    assert math.isclose(peak, 1.0, abs_tol=1e-5)


def test_ultrasonic_waveform_returns_int16_and_metadata():
    spec = WaveformSpec()
    result = simulate_waveform_measurement(spec=spec, **_default_kwargs())
    assert result["waveform_int16"].dtype == np.int16
    assert result["waveform_int16"].shape == (spec.waveform_samples,)
    for key in (
        "tof_true_s",
        "tof_observed_s",
        "peak_index",
        "tof_quality",
        "tof_accepted",
        "alpha_true_npm",
        "sound_speed_m_per_s",
        "sound_speed_estimated_m_per_s",
    ):
        assert key in result


def test_ultrasonic_tof_estimate_close_to_truth():
    spec = WaveformSpec()
    result = simulate_waveform_measurement(spec=spec, **_default_kwargs())
    # delay_correction_s 校正后, sound_speed_estimated 应接近真值
    rel_err = abs(
        result["sound_speed_estimated_m_per_s"] - result["sound_speed_m_per_s"]
    ) / result["sound_speed_m_per_s"]
    assert rel_err < 0.05  # 5% 容差


def test_ultrasonic_tof_quality_range():
    spec = WaveformSpec()
    result = simulate_waveform_measurement(spec=spec, **_default_kwargs())
    assert 0.0 <= result["tof_quality"] <= 1.0
    assert result["tof_accepted"] in (0, 1)


def test_ultrasonic_invalid_path_length_rejected():
    spec = WaveformSpec()
    kwargs = _default_kwargs()
    kwargs["l_m"] = 0.0
    with pytest.raises(ValueError, match="l_m"):
        simulate_waveform_measurement(spec=spec, **kwargs)


def test_ultrasonic_signature_does_not_accept_legacy_gases():
    """方案 §11.1: 旧 x_h2 / x_ch4 参数应被拒绝。"""
    spec = WaveformSpec()
    kwargs = _default_kwargs()
    with pytest.raises(TypeError):
        simulate_waveform_measurement(spec=spec, x_h2=10.0, **kwargs)
    with pytest.raises(TypeError):
        simulate_waveform_measurement(spec=spec, x_ch4=15.0, **kwargs)


def test_fiber_mic_waveform_shape():
    spec = FiberMicSpec()
    result = simulate_fiber_mic_measurement(spec=spec, **_default_kwargs())
    assert result["waveform_int16"].dtype == np.int16
    assert result["waveform_int16"].shape == (spec.waveform_samples,)


def test_fiber_mic_does_not_accept_legacy_gases():
    spec = FiberMicSpec()
    kwargs = _default_kwargs()
    with pytest.raises(TypeError):
        simulate_fiber_mic_measurement(spec=spec, x_h2=5.0, **kwargs)


def test_acoustic_attenuation_affects_amplitude():
    """衰减增大（高 CO2 + 高 RH）应使波形 peak 下降。"""
    spec = WaveformSpec()
    base = simulate_waveform_measurement(spec=spec, **_default_kwargs())
    high = _default_kwargs()
    high["x_co2"] = 20.0
    high["x_n2"] = 70.0
    high["h_rh"] = 80.0
    high_atten = simulate_waveform_measurement(spec=spec, **high)
    assert high_atten["alpha_true_npm"] > base["alpha_true_npm"]


def test_spec_acoustic_model_label():
    """spec 中的声学模型标签应与 acoustic_physics 模块一致。"""
    assert WaveformSpec().acoustic_attenuation_model == "linear_mixing_v1"
    assert FiberMicSpec().acoustic_attenuation_model == "linear_mixing_v1"
