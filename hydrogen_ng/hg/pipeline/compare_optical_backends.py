from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from sim.generation.acoustic_physics import main_sensor_features
from sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    compute_hitran_ndir_absorbance,
    get_default_hitran_grid,
    get_default_ndir_filter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare empirical_v1 NDIR absorption with hitran_hapi_v1.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--x-h2", type=float, default=10.0)
    parser.add_argument("--x-ch4", type=float, default=60.0)
    parser.add_argument("--x-co2", type=float, default=8.0)
    parser.add_argument("--x-n2", type=float, default=22.0)
    parser.add_argument("--t-c", type=float, default=25.0)
    parser.add_argument("--p-mpa", type=float, default=0.101325)
    parser.add_argument("--h-rh", type=float, default=50.0)
    parser.add_argument("--l-m", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser


def compare_optical_backends(
    *,
    cache_root: Path | str,
    condition: dict[str, str],
    seed: int,
    hapi_module: object | None = None,
) -> dict[str, object]:
    empirical = main_sensor_features(condition, random.Random(seed))
    concentrations = {
        "CH4": float(condition["x_CH4"]),
        "CO2": float(condition["x_CO2"]),
        "H2O": 0.0,
    }
    path_length_m = float(condition["L_m"])
    spectral = {
        channel: compute_hitran_ndir_absorbance(
            gas_specs=DEFAULT_HITRAN_GAS_SPECS,
            concentrations_pct=concentrations,
            path_length_m=path_length_m,
            filter_spec=get_default_ndir_filter(channel),
            grid_spec=get_default_hitran_grid(channel),
            cache_root=cache_root,
            hapi_module=hapi_module,
        )
        for channel in ("ch4", "co2")
    }
    return {
        "condition": condition,
        "empirical_v1": {
            "absorption_ch4_observed": empirical["absorption_ch4_observed"],
            "absorption_co2_observed": empirical["absorption_co2_observed"],
        },
        "hitran_hapi_v1": {
            "absorption_ch4_observed": spectral["ch4"]["absorbance_observed"],
            "absorption_co2_observed": spectral["co2"]["absorbance_observed"],
            "absorbance_by_gas": {
                "ch4_channel": spectral["ch4"]["absorbance_by_gas"],
                "co2_channel": spectral["co2"]["absorbance_by_gas"],
            },
        },
        "delta_hitran_minus_empirical": {
            "absorption_ch4_observed": spectral["ch4"]["absorbance_observed"] - empirical["absorption_ch4_observed"],
            "absorption_co2_observed": spectral["co2"]["absorbance_observed"] - empirical["absorption_co2_observed"],
        },
    }


def _condition_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        "x_H2": str(args.x_h2),
        "x_CH4": str(args.x_ch4),
        "x_CO2": str(args.x_co2),
        "x_N2": str(args.x_n2),
        "T_C": str(args.t_c),
        "P_MPa": str(args.p_mpa),
        "H_RH": str(args.h_rh),
        "L_m": str(args.l_m),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = compare_optical_backends(
        cache_root=Path(args.cache_root),
        condition=_condition_from_args(args),
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
