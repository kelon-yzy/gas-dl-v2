from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Sequence

from sim.generation.benchmark import DEFAULT_HITRAN_CACHE_ROOT, default_worker_count
from sim.generation.conditions import generate_condition_rows
from sim.generation.optical_backend import collect_hitran_cache_requirements
from sim.generation.spectral import compute_hitran_ndir_absorbance, get_default_ndir_filter, read_cached_spectrum
from sim.generation.spectral.hitran_backend import _ensure_hapi_table, _load_hapi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute all HITRAN cache entries required by a benchmark generation spec.")
    parser.add_argument("--cache-root", default=DEFAULT_HITRAN_CACHE_ROOT)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--sampling-strategy", choices=("lhs", "random"), default="lhs")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes for cache precompute (default: CPU count - 2, capped at 24).")
    return parser


def precompute_hitran_benchmark_cache(
    *,
    cache_root: Path | str,
    sequence_count: int,
    seed: int,
    sampling_strategy: str,
    workers: int = 1,
    hapi_module: object | None = None,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    conditions = generate_condition_rows(sequence_count, seed=seed, sampling_strategy=sampling_strategy)
    requirements = collect_hitran_cache_requirements(conditions, cache_root=cache_root)
    if workers == 1 or hapi_module is not None:
        results = [_precompute_requirement(requirement, cache_root=cache_root, hapi_module=hapi_module, allow_fetch=True) for requirement in requirements]
    else:
        _ensure_required_hapi_tables(requirements, cache_root=cache_root)
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(requirements))) as executor:
            futures = [executor.submit(_precompute_requirement, requirement, cache_root, None, False) for requirement in requirements]
            for future in as_completed(futures):
                results.append(future.result())
    return {
        "cache_root": str(Path(cache_root)),
        "sequence_count": sequence_count,
        "seed": seed,
        "sampling_strategy": sampling_strategy,
        "required_cache_entries": len(requirements),
        "computed_cache_entries": sum(1 for result in results if result == "computed"),
        "skipped_cache_entries": sum(1 for result in results if result == "skipped"),
        "conditions": len(conditions),
        "workers": workers,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workers = args.workers if args.workers is not None else default_worker_count(args.sequences)
    summary = precompute_hitran_benchmark_cache(
        cache_root=Path(args.cache_root),
        sequence_count=args.sequences,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        workers=workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _precompute_requirement(requirement, cache_root: Path | str, hapi_module: object | None, allow_fetch: bool) -> str:
    if read_cached_spectrum(cache_root, requirement.key) is not None:
        return "skipped"
    compute_hitran_ndir_absorbance(
        gas_specs=(requirement.gas_spec,),
        concentrations_pct={requirement.gas_spec.gas: 1.0},
        path_length_m=1.0,
        filter_spec=get_default_ndir_filter(requirement.channel),
        grid_spec=requirement.grid_spec,
        cache_root=cache_root,
        hapi_module=hapi_module,
        allow_fetch=allow_fetch,
    )
    return "computed"


def _ensure_required_hapi_tables(requirements, *, cache_root: Path | str) -> None:
    hapi = _load_hapi()
    seen = set()
    for requirement in requirements:
        table_key = (
            requirement.gas_spec.gas,
            requirement.grid_spec.wavenumber_min_cm1,
            requirement.grid_spec.wavenumber_max_cm1,
        )
        if table_key in seen:
            continue
        seen.add(table_key)
        _ensure_hapi_table(hapi, requirement.gas_spec, requirement.grid_spec, cache_root)


if __name__ == "__main__":
    raise SystemExit(main())
