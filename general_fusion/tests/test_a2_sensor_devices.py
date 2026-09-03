from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gf.sim.a2_dynamic_physics import evaluate_shared_physics
from gf.sim.ar_he_co2 import A2DYN_SOUND_SPEED_MODEL_ID, SYSTEM_DELAY_S, a2dyn_cp_t_virial_sound_speed
from gf.sim.a2_sensor_devices import (
    SensorDeviceError,
    UltrasonicLockError,
    acquire_ultrasonic_tof,
    estimate_ndir_equilibrium_co2_series,
    estimate_ultrasonic_quality_series,
    estimate_ultrasonic_tof_series,
    quantization_plateau_lengths,
    simulate_ndir,
    simulate_tcd,
    ultrasonic_signal_amplitude,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (PROJECT_ROOT / "configs" / "data" / "ar_he_co2_a2_dynamic_v1.json").read_text(encoding="utf-8")
)


def _ultrasonic_profile(index: int = 0) -> dict:
    return CONFIG["hardware_profiles"]["ultrasonic"]["candidates"][index]


def _thermal_profile() -> dict:
    return CONFIG["hardware_profiles"]["thermal"]["profiles"][0]


def _ndir_profile() -> dict:
    return CONFIG["hardware_profiles"]["ndir"]["profiles"][0]


def test_ultrasonic_waveform_xcorr_estimates_shared_physics_tof_without_persisting_waveform() -> None:
    composition = [90.0, 5.0, 5.0]
    profile = _ultrasonic_profile()
    result = acquire_ultrasonic_tof(
        composition,
        temperature_k=298.15,
        pressure_pa=101325.0,
        profile=profile,
    )
    expected = evaluate_shared_physics(
        [composition],
        temperature_k=298.15,
        pressure_pa=101325.0,
        path_length_m=profile["path_length_m"],
    )["tof_s"][0]
    assert result.lock_status
    assert result.tof_s is not None
    assert abs(result.tof_s - expected) <= 1.0 / profile["adc_rate_hz"]
    assert result.peak_correlation > 0.9
    assert result.waveform_samples is None

    retained = acquire_ultrasonic_tof(
        composition,
        temperature_k=298.15,
        pressure_pa=101325.0,
        profile=profile,
        retain_waveform=True,
    )
    assert retained.waveform_samples is not None
    assert retained.waveform_samples.ndim == 1


def test_ultrasonic_uses_explicit_registered_a2dyn_sound_speed_model() -> None:
    composition = [20.0, 30.0, 50.0]
    profile = _ultrasonic_profile()
    result = acquire_ultrasonic_tof(
        composition,
        temperature_k=298.15,
        pressure_pa=101325.0,
        profile=profile,
        sound_speed_model_id=A2DYN_SOUND_SPEED_MODEL_ID,
    )
    expected = profile["path_length_m"] / a2dyn_cp_t_virial_sound_speed(
        {"Ar": 0.2, "He": 0.3, "CO2": 0.5},
        298.15,
        101325.0,
    ) + SYSTEM_DELAY_S
    assert result.lock_status
    assert result.tof_s is not None
    assert abs(result.tof_s - expected) <= 1.0 / profile["adc_rate_hz"]

    with pytest.raises(ValueError, match="unsupported sound speed model"):
        acquire_ultrasonic_tof(
            composition,
            temperature_k=298.15,
            pressure_pa=101325.0,
            profile=profile,
            sound_speed_model_id="unknown",
        )


def test_ultrasonic_parabolic_refinement_and_lock_failure_are_explicit() -> None:
    result = acquire_ultrasonic_tof(
        [80.0, 10.0, 10.0],
        temperature_k=303.15,
        pressure_pa=105000.0,
        profile=_ultrasonic_profile(1),
    )
    assert result.lock_status
    assert result.tof_s is not None
    assert result.estimated_tof_uncertainty_s > 0.0

    with pytest.raises(UltrasonicLockError, match="no theoretical ToF fallback"):
        acquire_ultrasonic_tof(
            [80.0, 10.0, 10.0],
            temperature_k=303.15,
            pressure_pa=105000.0,
            profile=_ultrasonic_profile(),
            internal_noise_std=10.0,
            rng=np.random.default_rng(4),
        )
    unlocked = acquire_ultrasonic_tof(
        [80.0, 10.0, 10.0],
        temperature_k=303.15,
        pressure_pa=105000.0,
        profile=_ultrasonic_profile(),
        internal_noise_std=10.0,
        rng=np.random.default_rng(4),
        strict=False,
    )
    assert not unlocked.lock_status
    assert unlocked.tof_s is None


def test_ultrasonic_low_frequency_estimator_and_attenuation_are_explicit() -> None:
    profile = _ultrasonic_profile(1)
    theoretical = np.asarray([0.000501234, 0.000498765])
    estimated = estimate_ultrasonic_tof_series(theoretical, profile)

    assert estimated.shape == theoretical.shape
    assert np.all(np.abs(estimated - theoretical) <= 0.25 / profile["adc_rate_hz"])
    assert ultrasonic_signal_amplitude([0.0, 0.0, 100.0], profile) < ultrasonic_signal_amplitude(
        [0.0, 100.0, 0.0], profile
    )


def test_ultrasonic_quality_surrogate_depends_on_registered_composition_attenuation() -> None:
    compositions = np.asarray(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        dtype=np.float64,
    )

    quality = estimate_ultrasonic_quality_series(
        compositions,
        profile=_ultrasonic_profile(1),
        internal_noise_std=0.01,
        multipath_profile={"components": []},
    )

    assert np.ptp(quality["peak_correlation"]) > 0.0
    assert np.ptp(quality["snr"]) > 0.0
    assert np.ptp(quality["estimated_tof_uncertainty_s"]) > 0.0
    assert np.all(quality["lock_status"])


def test_ndir_equilibrium_inverse_uses_clean_prefix_and_registered_curve() -> None:
    compositions = np.asarray(
        [[100.0, 0.0, 0.0], [80.0, 0.0, 20.0], [80.0, 0.0, 20.0]],
        dtype=np.float64,
    )
    profile = _ndir_profile()
    simulated = simulate_ndir(
        compositions,
        temperature_k=298.15,
        pressure_pa=101325.0,
        dt_s=0.2,
        profile=profile,
    )

    recovered = estimate_ndir_equilibrium_co2_series(
        simulated.clean_voltage_v,
        temperature_k=298.15,
        pressure_pa=101325.0,
        dt_s=0.2,
        profile=profile,
    )

    assert np.allclose(recovered, compositions[:, 2], rtol=0.0, atol=0.03)


def test_tcd_energy_balance_and_nominal_steady_state_parity() -> None:
    composition = [90.0, 5.0, 5.0]
    compositions = np.asarray([composition] * 8)
    result = simulate_tcd(
        compositions,
        temperature_k=298.15,
        dt_s=0.5,
        profile=_thermal_profile(),
    )
    expected = evaluate_shared_physics(
        compositions,
        temperature_k=298.15,
        pressure_pa=101325.0,
        path_length_m=0.2,
    )["thermal_voltage_v"]
    assert np.allclose(result.clean_voltage_v, expected, atol=1.0e-12)
    assert np.max(np.abs(result.energy_balance_residual_w)) < 1.0e-12
    assert np.all(result.heater_resistance_ohm > 0.0)

    dynamic = simulate_tcd(
        np.asarray([[100.0, 0.0, 0.0], [100.0, 0.0, 0.0], [20.0, 30.0, 50.0], [20.0, 30.0, 50.0]]),
        temperature_k=298.15,
        dt_s=0.5,
        profile=_thermal_profile(),
    )
    assert dynamic.heater_temperature_k[2] != dynamic.heater_temperature_k[1]
    assert np.max(np.abs(dynamic.energy_balance_residual_w)) < 1.0e-12


def test_ndir_zero_point_active_reference_and_high_range_saturation_audit() -> None:
    zero = simulate_ndir(
        np.asarray([[100.0, 0.0, 0.0]] * 5),
        temperature_k=298.15,
        pressure_pa=101325.0,
        dt_s=0.5,
        profile=_ndir_profile(),
        initial_ratio=1.0,
    )
    assert np.allclose(zero.clean_voltage_v, 2.5, atol=1.0e-12)
    assert np.allclose(zero.active_reference_ratio, 1.0, atol=1.0e-12)
    assert zero.saturation_fraction == 0.0

    high = simulate_ndir(
        np.asarray([[0.0, 0.0, 100.0]] * 40),
        temperature_k=298.15,
        pressure_pa=101325.0,
        dt_s=0.5,
        profile=_ndir_profile(),
        initial_ratio=1.0,
    )
    assert high.clean_voltage_v[-1] < high.clean_voltage_v[0]
    assert high.clean_voltage_v[-1] > 0.0
    assert high.saturation_fraction == 0.0
    assert quantization_plateau_lengths(np.round(high.clean_voltage_v[:, None], 2))[0] >= 1

    low = simulate_ndir(
        np.asarray([[99.5, 0.0, 0.5]] * 5),
        temperature_k=298.15,
        pressure_pa=101325.0,
        dt_s=0.5,
        profile=_ndir_profile(),
    )
    repeated = simulate_ndir(
        np.asarray([[99.5, 0.0, 0.5]] * 5),
        temperature_k=298.15,
        pressure_pa=101325.0,
        dt_s=0.5,
        profile=_ndir_profile(),
    )
    assert abs(low.clean_voltage_v[-1] - zero.clean_voltage_v[-1]) >= 1.0e-5
    assert np.array_equal(low.clean_voltage_v, repeated.clean_voltage_v)


def test_ndir_rejects_out_of_range_composition_without_clipping() -> None:
    invalid = dict(_ndir_profile())
    invalid["range_max_mol_pct"] = 90.0
    with pytest.raises(SensorDeviceError):
        simulate_ndir(
            [[0.0, 0.0, 100.0]],
            temperature_k=298.15,
            pressure_pa=101325.0,
            dt_s=0.5,
            profile=invalid,
        )
