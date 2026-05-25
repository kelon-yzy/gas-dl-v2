import numpy as np
import pytest

from sim.generation.spectral import (
    NDIRFilter,
    TabulatedSpectrum,
    compute_tabulated_ndir_absorbance,
    gaussian_filter,
    integrate_channel_absorbance,
)


def test_integrate_channel_absorbance_matches_constant_optical_depth():
    wavenumber = np.linspace(2300.0, 2400.0, 101)
    optical_depth = np.full_like(wavenumber, 0.42)
    response = np.ones_like(wavenumber)

    result = integrate_channel_absorbance(
        wavenumber_cm1=wavenumber,
        optical_depth=optical_depth,
        filter_response=response,
    )

    assert result["absorbance_observed"] == pytest.approx(0.42, rel=1e-12)
    assert result["transmittance_channel"] == pytest.approx(np.exp(-0.42), rel=1e-12)


def test_gaussian_filter_peaks_at_center():
    wavenumber = np.linspace(2320.0, 2360.0, 81)
    response = gaussian_filter(wavenumber, NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=20.0))

    assert response.argmax() == 40
    assert response[40] == pytest.approx(1.0, rel=1e-12)
    assert response[30] == pytest.approx(response[50], rel=1e-12)


def test_tabulated_backend_reports_cross_response_and_backend_metadata():
    wavenumber = np.linspace(2300.0, 2400.0, 101)
    co2_coeff = np.exp(-0.5 * ((wavenumber - 2340.0) / 8.0) ** 2) * 0.012
    ch4_coeff = np.exp(-0.5 * ((wavenumber - 2365.0) / 10.0) ** 2) * 0.002
    spectra = (
        TabulatedSpectrum("CO2", wavenumber, co2_coeff, "synthetic-co2-v1"),
        TabulatedSpectrum("CH4", wavenumber, ch4_coeff, "synthetic-ch4-v1"),
    )

    result = compute_tabulated_ndir_absorbance(
        spectra=spectra,
        concentrations_pct={"CO2": 8.0, "CH4": 60.0},
        path_length_m=0.3,
        filter_spec=NDIRFilter(channel="co2", center_cm1=2340.0, fwhm_cm1=24.0),
    )

    assert result["backend"] == "tabulated_spectrum_v1"
    assert result["filter_center_cm1"] == 2340.0
    assert result["filter_fwhm_cm1"] == 24.0
    assert result["absorbance_by_gas"]["CO2"] > result["absorbance_by_gas"]["CH4"]
    assert result["absorbance_observed"] > result["absorbance_by_gas"]["CO2"]
    assert result["source_version"] == {"CO2": "synthetic-co2-v1", "CH4": "synthetic-ch4-v1"}
