from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hg.sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    compute_hitran_ndir_absorbance,
    get_default_hitran_grid,
    get_default_ndir_filter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute HITRAN NDIR spectra into the local cache.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--channels", default="ch4,co2", help="Comma-separated NDIR channels to precompute.")
    parser.add_argument("--path-length-m", type=float, default=0.3)
    parser.add_argument("--ch4-pct", type=float, default=60.0)
    parser.add_argument("--co2-pct", type=float, default=8.0)
    parser.add_argument("--h2o-pct", type=float, default=1.0)
    return parser


def parse_channels(value: str) -> tuple[str, ...]:
    channels = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if len(channels) == 0:
        raise argparse.ArgumentTypeError("--channels must contain at least one channel")
    for channel in channels:
        if channel not in {"ch4", "co2"}:
            raise argparse.ArgumentTypeError(f"unsupported NDIR channel: {channel!r}")
    return channels


def precompute_hitran_spectra(
    *,
    cache_root: Path | str,
    channels: tuple[str, ...],
    concentrations_pct: dict[str, float],
    path_length_m: float,
    hapi_module: object | None = None,
) -> dict[str, object]:
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    results = {}
    for channel in channels:
        result = compute_hitran_ndir_absorbance(
            gas_specs=DEFAULT_HITRAN_GAS_SPECS,
            concentrations_pct=concentrations_pct,
            path_length_m=path_length_m,
            filter_spec=get_default_ndir_filter(channel),
            grid_spec=get_default_hitran_grid(channel),
            cache_root=cache_root,
            hapi_module=hapi_module,
        )
        results[channel] = {
            "backend": result["backend"],
            "absorbance_observed": result["absorbance_observed"],
            "absorbance_by_gas": result["absorbance_by_gas"],
            "source_version": result["source_version"],
        }
    return {
        "cache_root": str(Path(cache_root)),
        "channels": list(channels),
        "path_length_m": path_length_m,
        "concentrations_pct": concentrations_pct,
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    channels = parse_channels(args.channels)
    summary = precompute_hitran_spectra(
        cache_root=Path(args.cache_root),
        channels=channels,
        concentrations_pct={"CH4": args.ch4_pct, "CO2": args.co2_pct, "H2O": args.h2o_pct},
        path_length_m=args.path_length_m,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
