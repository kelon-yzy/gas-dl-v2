"""Syngas HITRAN cache 预计算 CLI。

针对 `sim.generation.syngas.benchmark.generate_syngas_benchmark_dataset` 的
HITRAN 后端预计算所需 (channel, gas, T, P) 谱线缓存。结构与 hg 的
`precompute_hitran_benchmark_cache` 一致，但走 syngas 的 condition rows 和
3 通道 (ch4, co2, co) cache requirements。

用法：
    python -m pipeline.precompute_syngas_hitran_benchmark_cache \
        --cache-root data/hitran_cache \
        --sequences 32 \
        --seed 20260626 \
        --workers 4

输出 JSON 含 cache_root / sequence_count / required / computed / skipped 字段。
计算流程：

1. 用 syngas LHS 采样生成 condition rows（与 benchmark 实际使用同 seed）；
2. 调 ``collect_hitran_syngas_cache_requirements`` 收集所有 (channel, gas, T, P) 槽位；
3. 串行调用 `_ensure_required_hapi_tables`（HAPI fetch，需联网，约 ~50 MB CO/CO2/H2O 谱线）；
4. 并行计算每个槽位的 .npz cache（max 4 workers，避免 HAPI 内存爆）。

注意：与 hg 的 precompute 共用同一 HAPI table 池（按 wmin/wmax 编码 table_name 自动隔离）。
即使先跑 hg 后跑 syngas 也不会覆盖 hg cache。
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import json
from pathlib import Path
from typing import Sequence

from sg.sim.generation.benchmark import default_worker_count
from sg.sim.generation.optical_backend import HitranCacheRequirement
from sg.sim.generation.spectral import (
    compute_hitran_ndir_absorbance,
    get_syngas_ndir_filter,
    read_cached_spectrum,
)
from sg.sim.generation.spectral.hitran_backend import _ensure_hapi_table, _load_hapi
from sg.sim.generation.syngas.benchmark import DEFAULT_HITRAN_CACHE_ROOT
from sg.sim.generation.syngas.conditions import generate_syngas_condition_rows
from sg.sim.generation.syngas.optical_backend import collect_hitran_syngas_cache_requirements


DEFAULT_MAX_HITRAN_PRECOMPUTE_WORKERS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute all HITRAN cache entries required by a syngas benchmark generation spec."
    )
    parser.add_argument("--cache-root", default=DEFAULT_HITRAN_CACHE_ROOT)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--sampling-strategy", choices=("lhs", "random"), default="lhs")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes for memory-heavy HAPI cache precompute (default: CPU count - 2, capped at 4).",
    )
    return parser


def precompute_syngas_hitran_benchmark_cache(
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
    conditions = generate_syngas_condition_rows(
        sequence_count, seed=seed, sampling_strategy=sampling_strategy
    )
    requirements = collect_hitran_syngas_cache_requirements(conditions, cache_root=cache_root)
    if workers == 1 or hapi_module is not None:
        results = [
            _precompute_requirement(
                requirement,
                cache_root=cache_root,
                hapi_module=hapi_module,
                allow_fetch=True,
            )
            for requirement in requirements
        ]
    else:
        _ensure_required_hapi_tables(requirements, cache_root=cache_root)
        results = _precompute_requirements_parallel(
            requirements, cache_root=cache_root, workers=workers
        )
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
    workers = (
        args.workers
        if args.workers is not None
        else default_syngas_hitran_precompute_worker_count(args.sequences)
    )
    summary = precompute_syngas_hitran_benchmark_cache(
        cache_root=Path(args.cache_root),
        sequence_count=args.sequences,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        workers=workers,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def default_syngas_hitran_precompute_worker_count(sequence_count: int | None = None) -> int:
    return min(DEFAULT_MAX_HITRAN_PRECOMPUTE_WORKERS, default_worker_count(sequence_count))


def _precompute_requirements_parallel(
    requirements: Sequence[HitranCacheRequirement],
    *,
    cache_root: Path | str,
    workers: int,
) -> list[str]:
    max_workers = min(workers, len(requirements))
    max_pending = max_workers * 2
    requirement_iter = iter(requirements)
    results: list[str] = []
    pending: dict[Future[str], HitranCacheRequirement] = {}

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for requirement in requirement_iter:
                pending[executor.submit(_precompute_requirement, requirement, cache_root, None, False)] = requirement
                if len(pending) >= max_pending:
                    _collect_completed_requirements(pending, results)
            while pending:
                _collect_completed_requirements(pending, results)
    except BrokenProcessPool as exc:
        raise _broken_process_pool_error() from exc
    return results


def _collect_completed_requirements(
    pending: dict[Future[str], HitranCacheRequirement], results: list[str]
) -> None:
    done, _ = wait(pending, return_when=FIRST_COMPLETED)
    for future in done:
        pending.pop(future)
        try:
            results.append(future.result())
        except BrokenProcessPool as exc:
            raise _broken_process_pool_error() from exc


def _broken_process_pool_error() -> RuntimeError:
    return RuntimeError(
        "A HITRAN cache worker terminated abruptly. This usually indicates host memory exhaustion "
        "or a native-library crash. Existing .npz cache entries are safe to keep; rerun the same "
        "command with fewer workers, for example --workers 2 or --workers 1."
    )


def _precompute_requirement(
    requirement: HitranCacheRequirement,
    cache_root: Path | str,
    hapi_module: object | None,
    allow_fetch: bool,
) -> str:
    if read_cached_spectrum(cache_root, requirement.key) is not None:
        return "skipped"
    compute_hitran_ndir_absorbance(
        gas_specs=(requirement.gas_spec,),
        concentrations_pct={requirement.gas_spec.gas: 1.0},
        path_length_m=1.0,
        filter_spec=get_syngas_ndir_filter(requirement.channel),
        grid_spec=requirement.grid_spec,
        cache_root=cache_root,
        hapi_module=hapi_module,
        allow_fetch=allow_fetch,
    )
    return "computed"


def _ensure_required_hapi_tables(
    requirements: Sequence[HitranCacheRequirement], *, cache_root: Path | str
) -> None:
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
