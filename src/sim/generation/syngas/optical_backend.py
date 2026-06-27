"""Syngas (H2/CH4/CO2/CO + N2 background) HITRAN optical backend.

Mirrors hg ``sim.generation.optical_backend`` but for the 3-channel syngas
NDIR setup (ch4 / co2 / co). Each channel's gas list is driven by the
``syngas.channel_gas_specs`` mapping in ``configs/data/spectral-defaults.json``:

    ch4 channel: [CH4, CO2, H2O]   (CH4 self + CO2/H2O crosstalk)
    co2 channel: [CH4, CO2, H2O]   (CO2 self + CH4/H2O crosstalk)
    co  channel: [CO, CO2, H2O]    (CO self + CO2/H2O crosstalk; CH4 has no
                                    fundamental in [1980, 2310] cm^-1)

The CO channel uses an independent grid [1980, 2310] cm^-1 stored under
``syngas.hitran_grids.co``; spectral cache files are keyed by
(gas, wmin, wmax, step) via ``SpectralCacheKey`` so hg ch4/co2 caches are
never shadowed or overwritten.

Cache policy matches hg: ``cache_only_prechecked`` -- generation never calls
``hapi.fetch``. Run ``pipeline.precompute_syngas_hitran_benchmark_cache`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim.generation.gas_state import h2o_mole_percent_from_rh, hitran_pressure_atm, hitran_temperature_k
from sim.generation.optical_backend import (
    HITRAN_CACHE_POLICY,
    HITRAN_TP_MODE,
    H2O_POLICY,
    SPECTRAL_CROSSTALK_POLICY,
    HitranCacheRequirement,
    MissingHitranBenchmarkCacheError,
)
from sim.generation.spectral import (
    HITRAN_ABSORPTION_BACKEND,
    HitranGasSpec,
    HitranGridSpec,
    MissingHitranCacheError,
    NDIRFilter,
    PreparedTabulatedSpectra,
    SpectralCacheKey,
    TabulatedSpectrum,
    cache_path,
    compute_prepared_tabulated_ndir_absorbance,
    convert_hitran_coeff_to_per_percent_m,
    get_syngas_channel_gas_specs,
    get_syngas_hitran_grid,
    get_syngas_ndir_filter,
    hitran_cache_key,
    prepare_tabulated_spectra,
    read_cached_spectrum,
)


SYNGAS_NDIR_CHANNELS: tuple[str, ...] = ("ch4", "co2", "co")
SYNGAS_OPTICAL_CROSSTALK_POLICY = "spectral_multigas_integral_syngas_3channel"


def build_hitran_syngas_grid_for_condition(channel: str, *, t_c: float, p_mpa: float) -> HitranGridSpec:
    """Bind syngas default grid (wmin/wmax/step) to per-condition T/P."""
    default = get_syngas_hitran_grid(channel)
    return HitranGridSpec(
        wavenumber_min_cm1=default.wavenumber_min_cm1,
        wavenumber_max_cm1=default.wavenumber_max_cm1,
        wavenumber_step_cm1=default.wavenumber_step_cm1,
        temperature_k=hitran_temperature_k(t_c),
        pressure_atm=hitran_pressure_atm(p_mpa),
    )


def collect_hitran_syngas_cache_requirements(
    conditions: list[dict[str, str]],
    *,
    cache_root: Path | str,
    channels: tuple[str, ...] = SYNGAS_NDIR_CHANNELS,
) -> tuple[HitranCacheRequirement, ...]:
    """Enumerate all (channel, gas, T, P) cache entries needed for syngas benchmark.

    For each condition's (T_C_base, P_MPa_base) and each NDIR channel, iterate
    over the channel-specific gas list from ``SYNGAS_CHANNEL_GAS_SPECS`` (not
    the hg-wide DEFAULT_HITRAN_GAS_SPECS) so requirements stay minimal.
    """
    requirements: dict[SpectralCacheKey, HitranCacheRequirement] = {}
    for condition in conditions:
        t_c = float(condition["T_C_base"])
        p_mpa = float(condition["P_MPa_base"])
        for channel in channels:
            grid_spec = build_hitran_syngas_grid_for_condition(channel, t_c=t_c, p_mpa=p_mpa)
            for gas_spec in get_syngas_channel_gas_specs(channel):
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
            key=lambda item: (
                item.temperature_k,
                item.pressure_atm,
                item.gas,
                item.wavenumber_min_cm1,
                item.wavenumber_max_cm1,
            ),
        )
    )


def validate_hitran_syngas_benchmark_cache(
    conditions: list[dict[str, str]],
    *,
    cache_root: Path | str,
) -> tuple[HitranCacheRequirement, ...]:
    requirements = collect_hitran_syngas_cache_requirements(conditions, cache_root=cache_root)
    missing = tuple(
        requirement
        for requirement in requirements
        if read_cached_spectrum(cache_root, requirement.key) is None
    )
    if missing:
        raise MissingHitranBenchmarkCacheError(missing)
    return requirements


def compute_hitran_syngas_optical_absorption(
    condition: dict[str, str],
    *,
    cache_root: Path | str,
    spectra_cache: dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None = None,
) -> dict[str, object]:
    """Compute per-channel NDIR absorbance via HITRAN cache, 3-channel syngas.

    Returns a dict with the same shape conventions as hg
    ``compute_hitran_optical_absorption`` but extended for the CO channel:

    - absorption_{ch4,co2,co}_true: self-gas absorbance per channel
    - absorption_{ch4,co2,co}_observed: total (self + crosstalk) per channel
    - absorption_by_gas: nested {channel -> {gas -> absorbance}}
    - h2o_pct_eff, optical_absorption_backend, optical_source_version
    """
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    concentrations_pct = {
        "CH4": float(condition["x_CH4"]),
        "CO2": float(condition["x_CO2"]),
        "CO": float(condition["x_CO"]),
        "H2O": h2o_mole_percent_from_rh(t_c, p_mpa, h_rh),
    }
    path_length_m = float(condition["L_m"])
    by_channel = {
        channel: _compute_hitran_channel_from_cache(
            channel=channel,
            concentrations_pct=concentrations_pct,
            path_length_m=path_length_m,
            filter_spec=get_syngas_ndir_filter(channel),
            grid_spec=build_hitran_syngas_grid_for_condition(channel, t_c=t_c, p_mpa=p_mpa),
            cache_root=cache_root,
            spectra_cache=spectra_cache,
        )
        for channel in SYNGAS_NDIR_CHANNELS
    }
    ch4 = by_channel["ch4"]
    co2 = by_channel["co2"]
    co = by_channel["co"]
    return {
        "absorption_ch4_true": ch4["absorbance_by_gas"].get("CH4", 0.0),
        "absorption_co2_true": co2["absorbance_by_gas"].get("CO2", 0.0),
        "absorption_co_true": co["absorbance_by_gas"].get("CO", 0.0),
        "absorption_ch4_observed": ch4["absorbance_observed"],
        "absorption_co2_observed": co2["absorbance_observed"],
        "absorption_co_observed": co["absorbance_observed"],
        "absorption_ch4_cross": ch4["absorbance_observed"] - ch4["absorbance_by_gas"].get("CH4", 0.0),
        "absorption_co2_cross": co2["absorbance_observed"] - co2["absorbance_by_gas"].get("CO2", 0.0),
        "absorption_co_cross": co["absorbance_observed"] - co["absorbance_by_gas"].get("CO", 0.0),
        "absorption_by_gas": {
            "ch4": ch4["absorbance_by_gas"],
            "co2": co2["absorbance_by_gas"],
            "co": co["absorbance_by_gas"],
        },
        "optical_absorption_backend": HITRAN_ABSORPTION_BACKEND,
        "optical_source_version": {
            "ch4": ch4["source_version"],
            "co2": co2["source_version"],
            "co": co["source_version"],
        },
        "h2o_pct_eff": concentrations_pct["H2O"],
    }


def hitran_syngas_manifest_metadata(cache_root: Path | str) -> dict[str, object]:
    """Manifest fields for syngas HITRAN benchmarks. Mirrors hg but with
    a syngas-specific crosstalk policy tag so downstream loaders can tell
    the two apart."""
    return {
        "hitran_cache_root": str(Path(cache_root)),
        "hitran_cache_policy": HITRAN_CACHE_POLICY,
        "hitran_temperature_pressure_mode": HITRAN_TP_MODE,
        "h2o_policy": H2O_POLICY,
        "optical_crosstalk_policy": SYNGAS_OPTICAL_CROSSTALK_POLICY,
        "ndir_channels": list(SYNGAS_NDIR_CHANNELS),
    }


def _compute_hitran_channel_from_cache(
    *,
    channel: str,
    concentrations_pct: dict[str, float],
    path_length_m: float,
    filter_spec: NDIRFilter,
    grid_spec: HitranGridSpec,
    cache_root: Path | str,
    spectra_cache: dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None,
) -> dict[str, object]:
    prepared = _cached_prepared_tabulated_spectra(
        channel=channel,
        filter_spec=filter_spec,
        grid_spec=grid_spec,
        cache_root=cache_root,
        spectra_cache=spectra_cache,
    )
    result = compute_prepared_tabulated_ndir_absorbance(
        prepared=prepared,
        concentrations_pct=concentrations_pct,
        path_length_m=path_length_m,
    )
    return result | {"backend": HITRAN_ABSORPTION_BACKEND}


def _cached_prepared_tabulated_spectra(
    *,
    channel: str,
    filter_spec: NDIRFilter,
    grid_spec: HitranGridSpec,
    cache_root: Path | str,
    spectra_cache: dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None,
) -> PreparedTabulatedSpectra:
    cache_key = (channel, grid_spec)
    if spectra_cache is not None and cache_key in spectra_cache:
        return spectra_cache[cache_key]
    gas_specs = get_syngas_channel_gas_specs(channel)
    spectra = tuple(
        _read_tabulated_spectrum(cache_root=cache_root, gas_spec=gas_spec, grid_spec=grid_spec)
        for gas_spec in gas_specs
    )
    prepared = prepare_tabulated_spectra(spectra=spectra, filter_spec=filter_spec)
    if spectra_cache is not None:
        spectra_cache[cache_key] = prepared
    return prepared


def _read_tabulated_spectrum(
    *,
    cache_root: Path | str,
    gas_spec: HitranGasSpec,
    grid_spec: HitranGridSpec,
) -> TabulatedSpectrum:
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
