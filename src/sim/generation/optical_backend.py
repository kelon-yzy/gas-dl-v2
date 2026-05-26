from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim.generation.gas_state import h2o_mole_percent_from_rh, hitran_pressure_atm, hitran_temperature_k
from sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    HITRAN_ABSORPTION_BACKEND,
    HitranGasSpec,
    HitranGridSpec,
    MissingHitranCacheError,
    SpectralCacheKey,
    TabulatedSpectrum,
    cache_path,
    compute_tabulated_ndir_absorbance,
    convert_hitran_coeff_to_per_percent_m,
    get_default_hitran_grid,
    get_default_ndir_filter,
    hitran_cache_key,
    read_cached_spectrum,
)


EMPIRICAL_ABSORPTION_BACKEND = "empirical_v1"
VALID_OPTICAL_ABSORPTION_BACKENDS = (HITRAN_ABSORPTION_BACKEND, EMPIRICAL_ABSORPTION_BACKEND)
HITRAN_CACHE_POLICY = "cache_only_prechecked"
HITRAN_TP_MODE = "per_condition"
H2O_POLICY = "rh_to_mole_pct"
SPECTRAL_CROSSTALK_POLICY = "spectral_multigas_integral"


@dataclass(frozen=True, slots=True)
class HitranCacheRequirement:
    channel: str
    gas_spec: HitranGasSpec
    grid_spec: HitranGridSpec
    key: SpectralCacheKey
    path: Path


class MissingHitranBenchmarkCacheError(RuntimeError):
    def __init__(self, missing: tuple[HitranCacheRequirement, ...]):
        self.missing = missing
        preview = "; ".join(
            f"{item.channel}/{item.gas_spec.gas}/T={item.grid_spec.temperature_k:.3f}/P={item.grid_spec.pressure_atm:.6f}: {item.path}"
            for item in missing[:8]
        )
        suffix = "" if len(missing) <= 8 else f"; ... {len(missing) - 8} more"
        super().__init__(
            "Missing HITRAN benchmark cache entries. Run pipeline.precompute_hitran_benchmark_cache "
            f"with the same sequence_count, seed, and sampling_strategy first. Missing: {preview}{suffix}"
        )


def build_hitran_grid_for_condition(channel: str, *, t_c: float, p_mpa: float) -> HitranGridSpec:
    default = get_default_hitran_grid(channel)
    return HitranGridSpec(
        wavenumber_min_cm1=default.wavenumber_min_cm1,
        wavenumber_max_cm1=default.wavenumber_max_cm1,
        wavenumber_step_cm1=default.wavenumber_step_cm1,
        temperature_k=hitran_temperature_k(t_c),
        pressure_atm=hitran_pressure_atm(p_mpa),
    )


def collect_hitran_cache_requirements(
    conditions: list[dict[str, str]],
    *,
    cache_root: Path | str,
    channels: tuple[str, ...] = ("ch4", "co2"),
) -> tuple[HitranCacheRequirement, ...]:
    requirements: dict[SpectralCacheKey, HitranCacheRequirement] = {}
    for condition in conditions:
        t_c = float(condition["T_C_base"])
        p_mpa = float(condition["P_MPa_base"])
        for channel in channels:
            grid_spec = build_hitran_grid_for_condition(channel, t_c=t_c, p_mpa=p_mpa)
            for gas_spec in DEFAULT_HITRAN_GAS_SPECS:
                key = hitran_cache_key(gas_spec, grid_spec)
                requirements[key] = HitranCacheRequirement(
                    channel=channel,
                    gas_spec=gas_spec,
                    grid_spec=grid_spec,
                    key=key,
                    path=cache_path(cache_root, key),
                )
    return tuple(
        requirements[key]
        for key in sorted(
            requirements,
            key=lambda item: (item.temperature_k, item.pressure_atm, item.gas, item.wavenumber_min_cm1, item.wavenumber_max_cm1),
        )
    )


def validate_hitran_benchmark_cache(
    conditions: list[dict[str, str]],
    *,
    cache_root: Path | str,
) -> tuple[HitranCacheRequirement, ...]:
    requirements = collect_hitran_cache_requirements(conditions, cache_root=cache_root)
    missing = tuple(requirement for requirement in requirements if read_cached_spectrum(cache_root, requirement.key) is None)
    if missing:
        raise MissingHitranBenchmarkCacheError(missing)
    return requirements


def compute_hitran_optical_absorption(
    condition: dict[str, str],
    *,
    cache_root: Path | str,
    spectra_cache: dict[tuple[str, HitranGridSpec], tuple[TabulatedSpectrum, ...]] | None = None,
) -> dict[str, object]:
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    concentrations_pct = {
        "CH4": float(condition["x_CH4"]),
        "CO2": float(condition["x_CO2"]),
        "H2O": h2o_mole_percent_from_rh(t_c, p_mpa, h_rh),
    }
    path_length_m = float(condition["L_m"])
    by_channel = {
        channel: _compute_hitran_channel_from_cache(
            channel=channel,
            concentrations_pct=concentrations_pct,
            path_length_m=path_length_m,
            filter_spec=get_default_ndir_filter(channel),
            grid_spec=build_hitran_grid_for_condition(channel, t_c=t_c, p_mpa=p_mpa),
            cache_root=cache_root,
            spectra_cache=spectra_cache,
        )
        for channel in ("ch4", "co2")
    }
    ch4 = by_channel["ch4"]
    co2 = by_channel["co2"]
    return {
        "absorption_ch4_true": ch4["absorbance_by_gas"].get("CH4", 0.0),
        "absorption_co2_true": co2["absorbance_by_gas"].get("CO2", 0.0),
        "absorption_ch4_cross_from_co2": ch4["absorbance_observed"] - ch4["absorbance_by_gas"].get("CH4", 0.0),
        "absorption_co2_cross_from_ch4": co2["absorbance_observed"] - co2["absorbance_by_gas"].get("CO2", 0.0),
        "absorption_ch4_observed": ch4["absorbance_observed"],
        "absorption_co2_observed": co2["absorbance_observed"],
        "absorption_by_gas": {
            "ch4": ch4["absorbance_by_gas"],
            "co2": co2["absorbance_by_gas"],
        },
        "optical_absorption_backend": HITRAN_ABSORPTION_BACKEND,
        "optical_source_version": {
            "ch4": ch4["source_version"],
            "co2": co2["source_version"],
        },
        "h2o_pct_eff": concentrations_pct["H2O"],
    }


def hitran_manifest_metadata(cache_root: Path | str) -> dict[str, object]:
    return {
        "hitran_cache_root": str(Path(cache_root)),
        "hitran_cache_policy": HITRAN_CACHE_POLICY,
        "hitran_temperature_pressure_mode": HITRAN_TP_MODE,
        "h2o_policy": H2O_POLICY,
        "optical_crosstalk_policy": SPECTRAL_CROSSTALK_POLICY,
    }


def _compute_hitran_channel_from_cache(
    *,
    channel: str,
    concentrations_pct: dict[str, float],
    path_length_m: float,
    filter_spec: object,
    grid_spec: HitranGridSpec,
    cache_root: Path | str,
    spectra_cache: dict[tuple[str, HitranGridSpec], tuple[TabulatedSpectrum, ...]] | None,
) -> dict[str, object]:
    spectra = _cached_tabulated_spectra(
        channel=channel,
        grid_spec=grid_spec,
        cache_root=cache_root,
        spectra_cache=spectra_cache,
    )
    result = compute_tabulated_ndir_absorbance(
        spectra=spectra,
        concentrations_pct=concentrations_pct,
        path_length_m=path_length_m,
        filter_spec=filter_spec,
    )
    return result | {"backend": HITRAN_ABSORPTION_BACKEND}


def _cached_tabulated_spectra(
    *,
    channel: str,
    grid_spec: HitranGridSpec,
    cache_root: Path | str,
    spectra_cache: dict[tuple[str, HitranGridSpec], tuple[TabulatedSpectrum, ...]] | None,
) -> tuple[TabulatedSpectrum, ...]:
    cache_key = (channel, grid_spec)
    if spectra_cache is not None and cache_key in spectra_cache:
        return spectra_cache[cache_key]
    spectra = tuple(_read_tabulated_spectrum(cache_root=cache_root, gas_spec=gas_spec, grid_spec=grid_spec) for gas_spec in DEFAULT_HITRAN_GAS_SPECS)
    if spectra_cache is not None:
        spectra_cache[cache_key] = spectra
    return spectra


def _read_tabulated_spectrum(*, cache_root: Path | str, gas_spec: HitranGasSpec, grid_spec: HitranGridSpec) -> TabulatedSpectrum:
    key = hitran_cache_key(gas_spec, grid_spec)
    cached = read_cached_spectrum(cache_root, key)
    if cached is None:
        raise MissingHitranCacheError(key, cache_path(cache_root, key))
    wavenumber_cm1, absorption_coeff_cm1 = cached
    return TabulatedSpectrum(
        gas=gas_spec.gas,
        wavenumber_cm1=wavenumber_cm1,
        absorption_coeff_per_percent_m=convert_hitran_coeff_to_per_percent_m(
            absorption_coeff_cm1,
            temperature_k=grid_spec.temperature_k,
            pressure_atm=grid_spec.pressure_atm,
        ),
        source_version=gas_spec.source_version,
    )
