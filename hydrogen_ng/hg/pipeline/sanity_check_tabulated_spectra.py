from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    HitranGasSpec,
    HitranGridSpec,
    TabulatedSpectrum,
    compute_hitran_ndir_absorbance,
    compute_tabulated_ndir_absorbance,
    get_default_hitran_grid,
    get_default_ndir_filter,
)
from sim.generation.spectral.quantitative_table import (
    DEFAULT_COEFF_COLUMN,
    DEFAULT_WAVENUMBER_COLUMN,
    SUPPORTED_QUANTITATIVE_UNITS,
    load_quantitative_spectrum_csv,
    resample_spectrum_to_grid,
)


CHANNEL_TARGET_GAS = {"ch4": "CH4", "co2": "CO2"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare external quantitative spectra with hitran_hapi_v1.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--channels", default="ch4,co2", help="Comma-separated NDIR channels to check.")
    parser.add_argument("--ch4-spectrum")
    parser.add_argument("--co2-spectrum")
    parser.add_argument("--h2o-spectrum")
    parser.add_argument("--unit", choices=SUPPORTED_QUANTITATIVE_UNITS, required=True)
    parser.add_argument("--wavenumber-column", default=DEFAULT_WAVENUMBER_COLUMN)
    parser.add_argument("--coeff-column", default=DEFAULT_COEFF_COLUMN)
    parser.add_argument("--path-length-m", type=float, default=0.3)
    parser.add_argument("--ch4-pct", type=float, default=60.0)
    parser.add_argument("--co2-pct", type=float, default=8.0)
    parser.add_argument("--h2o-pct", type=float, default=0.0)
    return parser


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not channels:
        raise argparse.ArgumentTypeError("--channels must contain at least one channel")
    unknown = [channel for channel in channels if channel not in CHANNEL_TARGET_GAS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported NDIR channel: {unknown[0]!r}")
    return channels


def sanity_check_tabulated_spectra(
    *,
    cache_root: Path | str,
    spectra: tuple[TabulatedSpectrum, ...],
    concentrations_pct: dict[str, float],
    path_length_m: float,
    channels: tuple[str, ...],
    hapi_module: object | None = None,
) -> dict[str, object]:
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    channels = parse_channels(",".join(channels))
    active_gas_specs = _active_hitran_gas_specs(concentrations_pct)
    spectra_by_gas = {spectrum.gas.upper(): spectrum for spectrum in spectra}
    _validate_required_spectra(spectra_by_gas, concentrations_pct)

    results = {}
    for channel in channels:
        filter_spec = get_default_ndir_filter(channel)
        grid_spec = get_default_hitran_grid(channel)
        grid = _grid_from_hitran_spec(grid_spec)
        resampled = tuple(
            resample_spectrum_to_grid(spectra_by_gas[gas], grid)
            for gas, concentration in concentrations_pct.items()
            if concentration > 0.0
        )
        tabulated = compute_tabulated_ndir_absorbance(
            spectra=resampled,
            concentrations_pct=concentrations_pct,
            path_length_m=path_length_m,
            filter_spec=filter_spec,
        )
        hitran = compute_hitran_ndir_absorbance(
            gas_specs=active_gas_specs,
            concentrations_pct=concentrations_pct,
            path_length_m=path_length_m,
            filter_spec=filter_spec,
            grid_spec=grid_spec,
            cache_root=cache_root,
            hapi_module=hapi_module,
        )
        checks = _channel_checks(channel=channel, tabulated=tabulated, hitran=hitran)
        results[channel] = {
            "filter": {
                "center_cm1": filter_spec.center_cm1,
                "fwhm_cm1": filter_spec.fwhm_cm1,
            },
            "grid": {
                "wavenumber_min_cm1": grid_spec.wavenumber_min_cm1,
                "wavenumber_max_cm1": grid_spec.wavenumber_max_cm1,
                "wavenumber_step_cm1": grid_spec.wavenumber_step_cm1,
            },
            "tabulated": _result_summary(tabulated),
            "hitran_hapi_v1": _result_summary(hitran),
            "delta_tabulated_minus_hitran": {
                "absorbance_observed": tabulated["absorbance_observed"] - hitran["absorbance_observed"],
            },
            "checks": checks,
            "overall_ok": all(value for value in checks.values() if isinstance(value, bool)),
        }
    return {
        "cache_root": str(Path(cache_root)),
        "channels": list(channels),
        "path_length_m": path_length_m,
        "concentrations_pct": concentrations_pct,
        "results": results,
    }


def load_cli_spectra(args: argparse.Namespace) -> tuple[TabulatedSpectrum, ...]:
    path_by_gas = {
        "CH4": args.ch4_spectrum,
        "CO2": args.co2_spectrum,
        "H2O": args.h2o_spectrum,
    }
    spectra = []
    for gas, path in path_by_gas.items():
        if path is None:
            continue
        spectra.append(
            load_quantitative_spectrum_csv(
                path,
                gas=gas,
                unit=args.unit,
                wavenumber_column=args.wavenumber_column,
                coeff_column=args.coeff_column,
            )
        )
    return tuple(spectra)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    concentrations = {"CH4": args.ch4_pct, "CO2": args.co2_pct, "H2O": args.h2o_pct}
    summary = sanity_check_tabulated_spectra(
        cache_root=Path(args.cache_root),
        spectra=load_cli_spectra(args),
        concentrations_pct=concentrations,
        path_length_m=args.path_length_m,
        channels=parse_channels(args.channels),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _validate_required_spectra(spectra_by_gas: dict[str, TabulatedSpectrum], concentrations_pct: dict[str, float]) -> None:
    missing = sorted(gas for gas, concentration in concentrations_pct.items() if concentration > 0.0 and gas not in spectra_by_gas)
    if missing:
        raise ValueError(f"Missing quantitative spectra for nonzero concentration gases: {missing}")


def _active_hitran_gas_specs(concentrations_pct: dict[str, float]) -> tuple[HitranGasSpec, ...]:
    active_gases = {gas for gas, concentration in concentrations_pct.items() if concentration > 0.0}
    gas_specs = tuple(spec for spec in DEFAULT_HITRAN_GAS_SPECS if spec.gas in active_gases)
    if not gas_specs:
        raise ValueError("concentrations_pct must contain at least one positive gas concentration")
    missing = sorted(active_gases - {spec.gas for spec in gas_specs})
    if missing:
        raise ValueError(f"No HITRAN gas spec configured for active gases: {missing}")
    return gas_specs


def _grid_from_hitran_spec(grid_spec: HitranGridSpec) -> np.ndarray:
    steps = int(round((grid_spec.wavenumber_max_cm1 - grid_spec.wavenumber_min_cm1) / grid_spec.wavenumber_step_cm1))
    grid = grid_spec.wavenumber_min_cm1 + grid_spec.wavenumber_step_cm1 * np.arange(steps + 1, dtype=np.float64)
    if abs(float(grid[-1]) - grid_spec.wavenumber_max_cm1) > grid_spec.wavenumber_step_cm1 * 1e-6:
        raise ValueError("HITRAN grid range must be divisible by wavenumber_step_cm1")
    return grid


def _channel_checks(*, channel: str, tabulated: dict[str, object], hitran: dict[str, object]) -> dict[str, object]:
    target_gas = CHANNEL_TARGET_GAS[channel]
    tabulated_contrib = tabulated["absorbance_by_gas"]
    hitran_contrib = hitran["absorbance_by_gas"]
    ratio = _safe_ratio(tabulated["absorbance_observed"], hitran["absorbance_observed"])
    return {
        "tabulated_target_gas_is_dominant": _dominant_gas(tabulated_contrib) == target_gas,
        "hitran_target_gas_is_dominant": _dominant_gas(hitran_contrib) == target_gas,
        "tabulated_contributions_nonnegative": _all_nonnegative(tabulated_contrib),
        "hitran_contributions_nonnegative": _all_nonnegative(hitran_contrib),
        "observed_ratio_tabulated_to_hitran": ratio,
        "same_order_of_magnitude": ratio is None or 0.1 <= ratio <= 10.0,
    }


def _result_summary(result: dict[str, object]) -> dict[str, object]:
    return {
        "backend": result["backend"],
        "absorbance_observed": result["absorbance_observed"],
        "absorbance_by_gas": result["absorbance_by_gas"],
        "source_version": result["source_version"],
    }


def _dominant_gas(absorbance_by_gas: dict[str, float]) -> str | None:
    if not absorbance_by_gas:
        return None
    return max(absorbance_by_gas, key=lambda gas: absorbance_by_gas[gas])


def _all_nonnegative(values: dict[str, float]) -> bool:
    return all(value >= -1e-12 for value in values.values())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


if __name__ == "__main__":
    raise SystemExit(main())
