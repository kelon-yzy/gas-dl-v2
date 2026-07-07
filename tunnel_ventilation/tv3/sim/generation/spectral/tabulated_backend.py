from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tv3.sim.generation.spectral.filters import NDIRFilter, gaussian_filter
from tv3.sim.generation.spectral.integration import integrate_channel_absorbance


@dataclass(frozen=True, slots=True)
class TabulatedSpectrum:
    gas: str
    wavenumber_cm1: np.ndarray
    absorption_coeff_per_percent_m: np.ndarray
    source_version: str


@dataclass(frozen=True, slots=True)
class PreparedTabulatedSpectra:
    spectra: tuple[TabulatedSpectrum, ...]
    wavenumber_cm1: np.ndarray
    filter_response: np.ndarray
    filter_spec: NDIRFilter


def compute_tabulated_ndir_absorbance(
    *,
    spectra: tuple[TabulatedSpectrum, ...],
    concentrations_pct: dict[str, float],
    path_length_m: float,
    filter_spec: NDIRFilter,
) -> dict[str, object]:
    prepared = prepare_tabulated_spectra(spectra=spectra, filter_spec=filter_spec)
    return compute_prepared_tabulated_ndir_absorbance(
        prepared=prepared,
        concentrations_pct=concentrations_pct,
        path_length_m=path_length_m,
    )


def prepare_tabulated_spectra(
    *,
    spectra: tuple[TabulatedSpectrum, ...],
    filter_spec: NDIRFilter,
) -> PreparedTabulatedSpectra:
    if len(spectra) == 0:
        raise ValueError("spectra must contain at least one gas")
    wavenumber = spectra[0].wavenumber_cm1.astype(np.float64)
    for spectrum in spectra:
        if spectrum.wavenumber_cm1.shape != wavenumber.shape or not np.allclose(spectrum.wavenumber_cm1, wavenumber):
            raise ValueError("all spectra must share the same wavenumber grid")
        if spectrum.absorption_coeff_per_percent_m.shape != wavenumber.shape:
            raise ValueError("absorption_coeff_per_percent_m must match wavenumber_cm1 shape")
    return PreparedTabulatedSpectra(
        spectra=spectra,
        wavenumber_cm1=wavenumber,
        filter_response=gaussian_filter(wavenumber, filter_spec),
        filter_spec=filter_spec,
    )


def compute_prepared_tabulated_ndir_absorbance(
    *,
    prepared: PreparedTabulatedSpectra,
    concentrations_pct: dict[str, float],
    path_length_m: float,
) -> dict[str, object]:
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    wavenumber = prepared.wavenumber_cm1
    filter_response = prepared.filter_response
    filter_spec = prepared.filter_spec
    optical_depth = np.zeros_like(wavenumber, dtype=np.float64)
    absorbance_by_gas = {}
    source_versions = {}

    for spectrum in prepared.spectra:
        concentration = float(concentrations_pct.get(spectrum.gas, 0.0))
        if concentration < 0.0:
            raise ValueError("concentrations_pct values must be >= 0")
        gas_optical_depth = spectrum.absorption_coeff_per_percent_m.astype(np.float64) * concentration * path_length_m
        optical_depth += gas_optical_depth
        gas_integral = integrate_channel_absorbance(
            wavenumber_cm1=wavenumber,
            optical_depth=gas_optical_depth,
            filter_response=filter_response,
        )
        absorbance_by_gas[spectrum.gas] = gas_integral["absorbance_observed"]
        source_versions[spectrum.gas] = spectrum.source_version

    channel = integrate_channel_absorbance(
        wavenumber_cm1=wavenumber,
        optical_depth=optical_depth,
        filter_response=filter_response,
    )
    return {
        "absorbance_observed": channel["absorbance_observed"],
        "absorbance_by_gas": absorbance_by_gas,
        "transmittance_channel": channel["transmittance_channel"],
        "filter_center_cm1": float(filter_spec.center_cm1),
        "filter_fwhm_cm1": float(filter_spec.fwhm_cm1),
        "backend": "tabulated_spectrum_v1",
        "source_version": source_versions,
    }
