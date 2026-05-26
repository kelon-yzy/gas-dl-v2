from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sim.generation.benchmark import DEFAULT_HITRAN_CACHE_ROOT
from sim.generation.conditions import generate_condition_rows
from sim.generation.optical_backend import collect_hitran_cache_requirements
from sim.generation.spectral import compute_hitran_ndir_absorbance, get_default_ndir_filter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute all HITRAN cache entries required by a benchmark generation spec.")
    parser.add_argument("--cache-root", default=DEFAULT_HITRAN_CACHE_ROOT)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--sampling-strategy", choices=("lhs", "random"), default="lhs")
    return parser


def precompute_hitran_benchmark_cache(
    *,
    cache_root: Path | str,
    sequence_count: int,
    seed: int,
    sampling_strategy: str,
    hapi_module: object | None = None,
) -> dict[str, object]:
    conditions = generate_condition_rows(sequence_count, seed=seed, sampling_strategy=sampling_strategy)
    requirements = collect_hitran_cache_requirements(conditions, cache_root=cache_root)
    for requirement in requirements:
        compute_hitran_ndir_absorbance(
            gas_specs=(requirement.gas_spec,),
            concentrations_pct={requirement.gas_spec.gas: 1.0},
            path_length_m=1.0,
            filter_spec=get_default_ndir_filter(requirement.channel),
            grid_spec=requirement.grid_spec,
            cache_root=cache_root,
            hapi_module=hapi_module,
            allow_fetch=True,
        )
    return {
        "cache_root": str(Path(cache_root)),
        "sequence_count": sequence_count,
        "seed": seed,
        "sampling_strategy": sampling_strategy,
        "required_cache_entries": len(requirements),
        "conditions": len(conditions),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = precompute_hitran_benchmark_cache(
        cache_root=Path(args.cache_root),
        sequence_count=args.sequences,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
