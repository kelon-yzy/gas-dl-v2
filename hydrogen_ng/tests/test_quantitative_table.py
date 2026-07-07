from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hg.sim.generation.spectral import TabulatedSpectrum
from hg.sim.generation.spectral.quantitative_table import (
    convert_quantitative_coeff_to_per_percent_m,
    load_quantitative_spectrum_csv,
    resample_spectrum_to_grid,
)


def test_convert_quantitative_coeff_units_to_per_percent_m():
    values = np.array([1.0, 2.0], dtype=np.float64)

    np.testing.assert_allclose(
        convert_quantitative_coeff_to_per_percent_m(values, unit="per_percent_m"),
        np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(
        convert_quantitative_coeff_to_per_percent_m(values, unit="per_fraction_m"),
        np.array([0.01, 0.02]),
    )
    np.testing.assert_allclose(
        convert_quantitative_coeff_to_per_percent_m(values, unit="per_ppm_m"),
        np.array([10000.0, 20000.0]),
    )


def test_convert_quantitative_coeff_rejects_unknown_unit():
    with pytest.raises(ValueError, match="Unsupported quantitative spectrum unit"):
        convert_quantitative_coeff_to_per_percent_m(np.array([1.0]), unit="mystery")


def test_load_quantitative_spectrum_csv_sorts_rows_and_sets_metadata(tmp_path: Path):
    path = tmp_path / "co2.csv"
    path.write_text(
        "wavenumber_cm1,absorption_coeff\n"
        "2302,0.3\n"
        "2300,0.1\n"
        "2301,0.2\n",
        encoding="utf-8",
    )

    spectrum = load_quantitative_spectrum_csv(path, gas="CO2", unit="per_fraction_m", source_version="nist-test")

    assert spectrum.gas == "CO2"
    assert spectrum.source_version == "nist-test"
    np.testing.assert_allclose(spectrum.wavenumber_cm1, np.array([2300.0, 2301.0, 2302.0]))
    np.testing.assert_allclose(spectrum.absorption_coeff_per_percent_m, np.array([0.001, 0.002, 0.003]))


def test_load_quantitative_spectrum_csv_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("nu,coeff\n2300,0.1\n2301,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_quantitative_spectrum_csv(path, gas="CH4", unit="per_percent_m")


def test_load_quantitative_spectrum_csv_rejects_duplicate_wavenumber(tmp_path: Path):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "wavenumber_cm1,absorption_coeff\n"
        "2300,0.1\n"
        "2300,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        load_quantitative_spectrum_csv(path, gas="CH4", unit="per_percent_m")


def test_resample_spectrum_to_grid_interpolates_without_extrapolation():
    spectrum = TabulatedSpectrum(
        gas="CO2",
        wavenumber_cm1=np.array([2300.0, 2302.0, 2304.0]),
        absorption_coeff_per_percent_m=np.array([0.1, 0.3, 0.5]),
        source_version="synthetic",
    )

    resampled = resample_spectrum_to_grid(spectrum, np.array([2301.0, 2302.0, 2303.0]))

    np.testing.assert_allclose(resampled.absorption_coeff_per_percent_m, np.array([0.2, 0.3, 0.4]))


def test_resample_spectrum_to_grid_rejects_extrapolation():
    spectrum = TabulatedSpectrum(
        gas="CH4",
        wavenumber_cm1=np.array([3000.0, 3001.0]),
        absorption_coeff_per_percent_m=np.array([0.1, 0.2]),
        source_version="synthetic",
    )

    with pytest.raises(ValueError, match="does not cover target grid"):
        resample_spectrum_to_grid(spectrum, np.array([2999.0, 3000.0]))
