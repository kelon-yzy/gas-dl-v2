import numpy as np
import pytest

from hg.sim.generation.waveforms import (
    FiberMicSpec,
    WaveformSpec,
    generate_burst_pulse,
    simulate_fiber_mic_measurement,
    simulate_waveform_measurement,
    transducer_response_pulse,
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
    # 用 over-sample 互相关验证实际波形的亚样本 TOF 定位，替代循环论证的
    # peak_index == round(tof_observed_s * fs) 断言（实现记录的 round 值）。
    # 若分数延迟退化（frac 被忽略），脉冲落在整数 round 位置，与期望差 frac
    # （0~0.5 样本），超出 0.1 样本容差，测试失败。
    from scipy.signal import correlate, resample_poly

    pulse = transducer_response_pulse(spec)
    pulse_argmax = int(np.argmax(np.abs(pulse)))
    up = 20
    pulse_up = resample_poly(pulse, up, 1)
    wf_up = resample_poly(result["waveform_float"], up, 1)
    corr = correlate(wf_up, pulse_up, mode="valid")
    measured_start = np.argmax(np.abs(corr)) / up
    expected_start = result["tof_observed_s"] * spec.sample_rate_hz - pulse_argmax
    assert measured_start == pytest.approx(expected_start, abs=0.1)
    assert result["tof_quality"] > 0.99
    assert result["tof_accepted"] == 1
    assert result["scale_factor"] == pytest.approx(spec.daq_full_scale_v / spec.adc_max)


def test_fiber_mic_waveform_runs_probe_pressure_to_phase_demod_chain():
    spec = FiberMicSpec(max_reflections=0)

    result = simulate_fiber_mic_measurement(**_STATE, seed=456, spec=spec)

    assert result["probe_pressure_peak_pa"] > 0.0
    assert result["phase_peak_rad"] == pytest.approx(result["probe_pressure_peak_pa"] * spec.probe.pressure_sensitivity_rad_per_pa)
    assert result["demod_peak_v"] > 0.0
    assert result["scale_factor"] == pytest.approx(spec.probe.daq_full_scale_v / spec.adc_max)
    assert spec.model_name == "fiber_interferometric_proxy_v1"
    assert spec.fiber_optical_demodulation_model == "linear_phase_demodulation_proxy_v1"


def test_fractional_shift_distinguishes_subsample_offsets():
    """分数延迟必须在亚样本级产生可区分波形，不能退化为整数 roll。

    若 _lagrange_fractional_shift 退化为只做整数移位（丢弃 frac），则 0.0 与 0.3
    偏移输出完全相同，本测试失败。参考 docs/物理模型严格化实施计划.md §3.5。
    """
    from scipy.signal import resample_poly

    from hg.sim.generation.waveforms import _lagrange_fractional_shift

    # 带通脉冲信号（与超声主脉冲同形态），放置在远离边界处避免边界效应干扰
    signal = np.zeros(512, dtype=np.float32)
    signal[200:240] = generate_burst_pulse()

    shifted_0 = _lagrange_fractional_shift(signal, 0.0)
    shifted_03 = _lagrange_fractional_shift(signal, 0.3)

    # 核心断言：0.3 偏移必须产生与 0 偏移不同的波形（frac 生效）
    assert not np.allclose(shifted_0, shifted_03, atol=1e-6)

    # over-sample 10 倍后比较绝对值峰值索引，验证实际亚样本位移 ≈ 0.3
    # （shifted_03 是 shifted_0 右移 0.3，峰值应右移约 3 个 up-sample）
    up = 10
    peak_ref = int(np.argmax(np.abs(resample_poly(shifted_0, up, 1))))
    peak_test = int(np.argmax(np.abs(resample_poly(shifted_03, up, 1))))
    lag_samples = (peak_test - peak_ref) / up
    assert lag_samples == pytest.approx(0.3, abs=0.2)


def test_fractional_shift_no_wraparound_at_boundary():
    """零填充移位：边界外的能量不环绕到数组另一端。

    旧实现用 np.roll 做整数移位，会把移出的能量卷到另一端；本测试锁定零填充
    行为——正向延迟不污染末尾，负向延迟不污染开头。
    """
    from hg.sim.generation.waveforms import _lagrange_fractional_shift

    # 靠近开头的脉冲，正向延迟 3.3 样本：末尾应保持零（无环绕）
    signal_head = np.zeros(64, dtype=np.float32)
    signal_head[2] = 1.0
    shifted = _lagrange_fractional_shift(signal_head, 3.3)
    assert np.all(shifted[-8:] == 0.0), "正向延迟不应把能量环绕到数组末尾"

    # 靠近末尾的脉冲，负向延迟 3.3 样本：开头应保持零（无环绕）
    signal_tail = np.zeros(64, dtype=np.float32)
    signal_tail[60] = 1.0
    shifted_neg = _lagrange_fractional_shift(signal_tail, -3.3)
    assert np.all(shifted_neg[:8] == 0.0), "负向延迟不应把能量环绕到数组开头"
