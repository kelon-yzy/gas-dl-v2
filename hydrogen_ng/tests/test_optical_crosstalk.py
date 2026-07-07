import pytest

from hg.sim.generation.optical_crosstalk import OpticalCrosstalkSpec, apply_optical_crosstalk


def test_apply_optical_crosstalk_uses_default_matrix():
    result = apply_optical_crosstalk(absorption_ch4=0.46, absorption_co2=0.72)

    assert result == {
        "absorption_ch4_true": pytest.approx(0.46, rel=1e-12),
        "absorption_co2_true": pytest.approx(0.72, rel=1e-12),
        "absorption_ch4_cross_from_co2": pytest.approx(0.0252, rel=1e-12),
        "absorption_co2_cross_from_ch4": pytest.approx(0.00552, rel=1e-12),
        "absorption_ch4_observed": pytest.approx(0.4852, rel=1e-12),
        "absorption_co2_observed": pytest.approx(0.72552, rel=1e-12),
    }


def test_apply_optical_crosstalk_accepts_explicit_matrix():
    result = apply_optical_crosstalk(
        absorption_ch4=0.4,
        absorption_co2=0.8,
        spec=OpticalCrosstalkSpec(ch4_channel_co2_response=0.05, co2_channel_ch4_response=0.02),
    )

    assert result["absorption_ch4_cross_from_co2"] == pytest.approx(0.04, rel=1e-12)
    assert result["absorption_co2_cross_from_ch4"] == pytest.approx(0.008, rel=1e-12)
    assert result["absorption_ch4_observed"] == pytest.approx(0.44, rel=1e-12)
    assert result["absorption_co2_observed"] == pytest.approx(0.808, rel=1e-12)
