from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np

from sim.generation.spectral.cache import SpectralCacheKey, read_cached_spectrum, write_cached_spectrum
from sim.generation.spectral.filters import NDIRFilter
from sim.generation.spectral.tabulated_backend import TabulatedSpectrum, compute_tabulated_ndir_absorbance


HITRAN_ABSORPTION_BACKEND = "hitran_hapi_v1"
BOLTZMANN_J_PER_K = 1.380649e-23
STANDARD_ATMOSPHERE_PA = 101325.0


@dataclass(frozen=True, slots=True)
class HitranGasSpec:
    gas: str
    table_name: str
    molecule_id: int
    isotopologue_id: int
    source_version: str = HITRAN_ABSORPTION_BACKEND


@dataclass(frozen=True, slots=True)
class HitranGridSpec:
    wavenumber_min_cm1: float
    wavenumber_max_cm1: float
    wavenumber_step_cm1: float
    temperature_k: float
    pressure_atm: float


def compute_hitran_ndir_absorbance(
    *,
    gas_specs: tuple[HitranGasSpec, ...],
    concentrations_pct: dict[str, float],
    path_length_m: float,
    filter_spec: NDIRFilter,
    grid_spec: HitranGridSpec,
    cache_root: Path | str,
    hapi_module: object | None = None,
) -> dict[str, object]:
    spectra = tuple(
        _spectrum_for_gas(gas_spec=gas_spec, grid_spec=grid_spec, cache_root=cache_root, hapi_module=hapi_module)
        for gas_spec in gas_specs
    )
    result = compute_tabulated_ndir_absorbance(
        spectra=spectra,
        concentrations_pct=concentrations_pct,
        path_length_m=path_length_m,
        filter_spec=filter_spec,
    )
    return result | {"backend": HITRAN_ABSORPTION_BACKEND}


def _spectrum_for_gas(
    *,
    gas_spec: HitranGasSpec,
    grid_spec: HitranGridSpec,
    cache_root: Path | str,
    hapi_module: object | None,
) -> TabulatedSpectrum:
    key = _cache_key(gas_spec, grid_spec)
    cached = read_cached_spectrum(cache_root, key)
    if cached is None:
        hapi = hapi_module if hapi_module is not None else _load_hapi()
        _fetch_table(hapi, gas_spec, grid_spec, cache_root)
        wavenumber_cm1, absorption_coeff_cm1 = _absorption_coefficient(hapi, gas_spec, grid_spec)
        write_cached_spectrum(
            cache_root,
            key,
            wavenumber_cm1=wavenumber_cm1,
            absorption_coeff_cm1=absorption_coeff_cm1,
        )
    else:
        wavenumber_cm1, absorption_coeff_cm1 = cached
    return _tabulated_spectrum_from_hitran_coefficients(gas_spec, wavenumber_cm1, absorption_coeff_cm1, grid_spec)


def _fetch_table(hapi: object, gas_spec: HitranGasSpec, grid_spec: HitranGridSpec, cache_root: Path | str) -> None:
    table_name = _hapi_table_name(gas_spec, grid_spec)
    hapi.db_begin(str(Path(cache_root)))
    hapi.fetch(
        table_name,
        gas_spec.molecule_id,
        gas_spec.isotopologue_id,
        grid_spec.wavenumber_min_cm1,
        grid_spec.wavenumber_max_cm1,
    )


def _absorption_coefficient(hapi: object, gas_spec: HitranGasSpec, grid_spec: HitranGridSpec) -> tuple[np.ndarray, np.ndarray]:
    table_name = _hapi_table_name(gas_spec, grid_spec)
    wavenumber_cm1, absorption_coeff_cm1 = hapi.absorptionCoefficient_Voigt(
        SourceTables=table_name,
        Environment={"T": grid_spec.temperature_k, "p": grid_spec.pressure_atm},
        WavenumberRange=(grid_spec.wavenumber_min_cm1, grid_spec.wavenumber_max_cm1),
        WavenumberStep=grid_spec.wavenumber_step_cm1,
        HITRAN_units=True,
    )
    return np.asarray(wavenumber_cm1, dtype=np.float64), np.asarray(absorption_coeff_cm1, dtype=np.float64)


def _tabulated_spectrum_from_hitran_coefficients(
    gas_spec: HitranGasSpec,
    wavenumber_cm1: np.ndarray,
    absorption_coeff_cm1: np.ndarray,
    grid_spec: HitranGridSpec,
) -> TabulatedSpectrum:
    return TabulatedSpectrum(
        gas=gas_spec.gas,
        wavenumber_cm1=wavenumber_cm1,
        absorption_coeff_per_percent_m=convert_hitran_coeff_to_per_percent_m(
            absorption_coeff_cm1.astype(np.float64),
            temperature_k=grid_spec.temperature_k,
            pressure_atm=grid_spec.pressure_atm,
        ),
        source_version=gas_spec.source_version,
    )


def convert_hitran_coeff_to_per_percent_m(
    absorption_coeff_cm2_per_molecule: np.ndarray,
    *,
    temperature_k: float,
    pressure_atm: float,
) -> np.ndarray:
    """Convert HITRAN k(nu) to optical depth per 1% concentration and 1 m path.

    HAPI with ``HITRAN_units=True`` returns the HITRAN line absorption
    coefficient in cm^2 / molecule. Multiplying it by the ideal-gas column
    number density for 1% volume fraction over 1 meter yields the unit expected
    by ``TabulatedSpectrum.absorption_coeff_per_percent_m``.
    """

    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be > 0")
    if pressure_atm <= 0.0:
        raise ValueError("pressure_atm must be > 0")
    total_number_density_m3 = pressure_atm * STANDARD_ATMOSPHERE_PA / (BOLTZMANN_J_PER_K * temperature_k)
    column_density_per_percent_m_cm2 = total_number_density_m3 * 1e-6
    return absorption_coeff_cm2_per_molecule.astype(np.float64) * column_density_per_percent_m_cm2


def _cache_key(gas_spec: HitranGasSpec, grid_spec: HitranGridSpec) -> SpectralCacheKey:
    return SpectralCacheKey(
        backend=HITRAN_ABSORPTION_BACKEND,
        gas=gas_spec.gas,
        source_version=gas_spec.source_version,
        wavenumber_min_cm1=grid_spec.wavenumber_min_cm1,
        wavenumber_max_cm1=grid_spec.wavenumber_max_cm1,
        wavenumber_step_cm1=grid_spec.wavenumber_step_cm1,
        temperature_k=grid_spec.temperature_k,
        pressure_atm=grid_spec.pressure_atm,
    )


def _hapi_table_name(gas_spec: HitranGasSpec, grid_spec: HitranGridSpec) -> str:
    min_part = _format_table_number(grid_spec.wavenumber_min_cm1)
    max_part = _format_table_number(grid_spec.wavenumber_max_cm1)
    return f"{gas_spec.table_name}_{min_part}_{max_part}"


def _format_table_number(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _load_hapi() -> object:
    try:
        return importlib.import_module("hapi")
    except ModuleNotFoundError as exc:
        raise RuntimeError("HITRAN HAPI is required for cache-miss HITRAN calculations; install hapi or provide hapi_module.") from exc
