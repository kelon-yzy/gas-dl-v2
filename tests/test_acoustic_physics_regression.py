import random

import pytest

from sim.generation.acoustic_physics import (
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
    main_sensor_features,
)


# ---- hidden_sound_speed_v2 ----

_SOUND_SPEED_CASES = [
    # (x_h2, x_ch4, x_co2, x_n2, t_c), expected
    ((10.0, 50.0, 10.0, 30.0, 25.0), 486.3),
    ((0.0, 60.0, 15.0, 25.0, 30.0), 399.04999999999995),
    ((20.0, 40.0, 20.0, 20.0, 20.0), 560.8000000000001),
    ((5.0, 70.0, 5.0, 20.0, 35.0), 467.5),
    ((0.0, 0.0, 0.0, 100.0, 25.0), 353.0),
]


@pytest.mark.parametrize("inputs,expected", _SOUND_SPEED_CASES)
def test_hidden_sound_speed_v2_regression(inputs, expected):
    result = hidden_sound_speed_v2(*inputs)
    assert result == pytest.approx(expected, rel=1e-12)


# ---- hidden_attenuation_v2 ----

_ATTEN_CASE_1_INPUTS = (10.0, 50.0, 10.0, 30.0, 25.0, 1.0, 50.0)
_ATTEN_CASE_1_EXPECTED = {
    "alpha_ch4_v2": 0.12568657024180233,
    "alpha_classical_v2": 0.000346343507339856,
    "alpha_co2_v2": 0.2792491236633517,
    "alpha_h2_diffusion_v2": 2.003409210374726e-09,
    "alpha_h2o_v2": 0.00010540769151257129,
    "alpha_true_v2": 0.4053874471074157,
    "c_mix_used": 486.3,
    "f_relax_ch4_eff": 888230.9400444116,
    "f_relax_co2_eff": 276994.8948941024,
    "f_relax_h2o_eff": 986923.2667160128,
    "h_w_pct_eff": 0.15835172653553672,
}

_ATTEN_CASE_2_INPUTS = (0.0, 60.0, 15.0, 25.0, 30.0, 0.8, 60.0)
_ATTEN_CASE_2_EXPECTED = {
    "alpha_ch4_v2": 0.2026325407603578,
    "alpha_classical_v2": 0.00043654442174454616,
    "alpha_co2_v2": 0.6294128618727635,
    "alpha_h2_diffusion_v2": 0.0,
    "alpha_h2o_v2": 0.0003224203488054903,
    "alpha_true_v2": 0.8328043674036714,
    "c_mix_used": 399.04999999999995,
    "f_relax_ch4_eff": 805329.3856402665,
    "f_relax_co2_eff": 222126.19230518906,
    "f_relax_h2o_eff": 789538.6133728103,
    "h_w_pct_eff": 0.31826320036690464,
}

_ATTEN_CASE_3_INPUTS = (20.0, 40.0, 20.0, 20.0, 20.0, 1.2, 40.0)
_ATTEN_CASE_3_EXPECTED = {
    "alpha_ch4_v2": 0.08385097276324477,
    "alpha_classical_v2": 0.00028618927008979965,
    "alpha_co2_v2": 0.40659704887975584,
    "alpha_h2_diffusion_v2": 1.9353284772013118e-09,
    "alpha_h2o_v2": 3.749489553969569e-05,
    "alpha_true_v2": 0.4907717077439585,
    "c_mix_used": 560.8000000000001,
    "f_relax_ch4_eff": 923760.177646188,
    "f_relax_co2_eff": 331993.74631674215,
    "f_relax_h2o_eff": 1184307.9200592153,
    "h_w_pct_eff": 0.07790941576170851,
}


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (_ATTEN_CASE_1_INPUTS, _ATTEN_CASE_1_EXPECTED),
        (_ATTEN_CASE_2_INPUTS, _ATTEN_CASE_2_EXPECTED),
        (_ATTEN_CASE_3_INPUTS, _ATTEN_CASE_3_EXPECTED),
    ],
)
def test_hidden_attenuation_v2_regression(inputs, expected):
    result = hidden_attenuation_v2(*inputs)
    for key, expected_val in expected.items():
        assert result[key] == pytest.approx(expected_val, rel=1e-12), f"mismatch on {key}"


# ---- main_sensor_features ----

_COND_1 = {
    "x_H2": "10.0", "x_CH4": "50.0", "x_CO2": "10.0", "x_N2": "30.0",
    "T_C": "25.0", "P_MPa": "1.0", "H_RH": "50.0", "L_m": "0.3",
}
_FEAT_1_EXPECTED = {
    "A_fft_max": 809.2448435088161,
    "Amp": 0.8853498005431114,
    "TOF": 0.0006964708752173117,
    "V_NDIR_CH4": 1.5586492732964947,
    "V_NDIR_CO2": 1.517607833837499,
    "V_TCS": 1.4639783071517982,
    "absorption_ch4_cross_from_co2": 0.017220000000000003,
    "absorption_ch4_observed": 0.47222000000000003,
    "absorption_ch4_true": 0.455,
    "absorption_co2_cross_from_ch4": 0.0054600000000000004,
    "absorption_co2_observed": 0.49746,
    "absorption_co2_true": 0.492,
    "f_peak": 40052.7644689981,
    "ndir_ch4_saturated": False,
    "ndir_co2_saturated": False,
    "optical_baseline_drift_ch4_observed": -0.0005103531351315484,
    "optical_baseline_drift_co2_observed": -0.00598941365736383,
    "thermal_baseline_drift_observed": 0.0009969550322031459,
}

_COND_2 = {
    "x_H2": "5.0", "x_CH4": "70.0", "x_CO2": "5.0", "x_N2": "20.0",
    "T_C": "30.0", "P_MPa": "0.9", "H_RH": "60.0", "L_m": "0.25",
}
_FEAT_2_EXPECTED = {
    "A_fft_max": 834.9959922899055,
    "Amp": 0.9223876470031603,
    "TOF": 0.0006194258177001045,
    "V_NDIR_CH4": 1.3283303226746737,
    "V_NDIR_CO2": 1.8917630965233587,
    "V_TCS": 1.381089436670308,
    "absorption_ch4_cross_from_co2": 0.009539249999999997,
    "absorption_ch4_observed": 0.63203925,
    "absorption_ch4_true": 0.6225,
    "absorption_co2_cross_from_ch4": 0.007470000000000001,
    "absorption_co2_observed": 0.2800199999999999,
    "absorption_co2_true": 0.2725499999999999,
    "f_peak": 40019.09199303792,
    "ndir_ch4_saturated": False,
    "ndir_co2_saturated": False,
    "optical_baseline_drift_ch4_observed": 0.007507546194645476,
    "optical_baseline_drift_co2_observed": 0.005951017432173414,
    "thermal_baseline_drift_observed": 0.006973568110038521,
}

_COND_3 = {
    "x_H2": "0.0", "x_CH4": "40.0", "x_CO2": "15.0", "x_N2": "45.0",
    "T_C": "22.0", "P_MPa": "1.5", "H_RH": "30.0", "L_m": "0.4",
}
_FEAT_3_EXPECTED = {
    "A_fft_max": 730.8949428891135,
    "Amp": 0.8293913667105622,
    "TOF": 0.0011450608159545197,
    "V_NDIR_CH4": 1.6635456975018763,
    "V_NDIR_CO2": 1.2398854570782656,
    "V_TCS": 1.2102371246758152,
    "absorption_ch4_cross_from_co2": 0.024869250000000002,
    "absorption_ch4_observed": 0.39076925000000007,
    "absorption_ch4_true": 0.36590000000000006,
    "absorption_co2_cross_from_ch4": 0.004390800000000001,
    "absorption_co2_observed": 0.7149408,
    "absorption_co2_true": 0.71055,
    "f_peak": 39793.07051251411,
    "ndir_ch4_saturated": False,
    "ndir_co2_saturated": False,
    "optical_baseline_drift_ch4_observed": -0.014125256409527352,
    "optical_baseline_drift_co2_observed": -0.011873739834561593,
    "thermal_baseline_drift_observed": -0.003209729538938888,
}


@pytest.mark.parametrize(
    "condition,seed,expected",
    [
        (_COND_1, 42, _FEAT_1_EXPECTED),
        (_COND_2, 123, _FEAT_2_EXPECTED),
        (_COND_3, 999, _FEAT_3_EXPECTED),
    ],
)
def test_main_sensor_features_regression(condition, seed, expected):
    rng = random.Random(seed)
    result = main_sensor_features(condition, rng)
    for key, expected_val in expected.items():
        assert result[key] == pytest.approx(expected_val, rel=1e-12), f"mismatch on {key}"
