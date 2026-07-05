"""掘进通风场景 benchmark 生成 CLI。

用法（empirical，阶段 1 唯一支持的后端）：
    python -m pipeline.generate_tunnel_ventilation_benchmark \
        --output-root data \
        --dataset tv3-smoke \
        --sequences 32 \
        --seed 20260704 \
        --timesteps 32 \
        --dt-s 0.5 \
        --optical-absorption-backend empirical_v1

HITRAN 后端阶段 1 未实现，传入 hitran_hapi_v1 会被拒绝。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
)
from sim.generation.phases import PHASE_SCHEDULES
from sim.generation.tunnel_ventilation.benchmark import (
    DEFAULT_WAVEFORM_PATH_LMS,
    TunnelVentilationBenchmarkGenerationSpec,
    default_worker_count,
    generate_tunnel_ventilation_benchmark_dataset,
)


DEFAULT_DATASET = "tv3-smoke"
DEFAULT_SEED = 20260704
DEFAULT_TIMESTEPS = 32
DEFAULT_DT_S = 0.5


def parse_path_lms(value: str) -> tuple[float, ...]:
    path_lms = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(path_lms) == 0:
        raise argparse.ArgumentTypeError("--path-lms must contain at least one comma-separated value")
    if any(path_l_m <= 0.0 for path_l_m in path_lms):
        raise argparse.ArgumentTypeError("--path-lms values must be > 0")
    return path_lms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a v4 tunnel_ventilation benchmark dataset.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--dt-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--storage", choices=("memmap", "npz", "both"), default="memmap")
    parser.add_argument("--multi-path-phase", choices=("off", "baseline", "steady"), default="steady")
    parser.add_argument("--stage-profile", choices=tuple(PHASE_SCHEDULES), default="standard_exposure")
    parser.add_argument("--stage-jitter", type=float, default=0.0)
    parser.add_argument("--sampling-strategy", choices=("lhs", "random"), default="lhs")
    parser.add_argument("--path-lms", type=parse_path_lms, default=DEFAULT_WAVEFORM_PATH_LMS)
    parser.add_argument(
        "--optical-absorption-backend",
        choices=VALID_OPTICAL_ABSORPTION_BACKENDS,
        default=EMPIRICAL_ABSORPTION_BACKEND,
    )
    parser.add_argument("--hitran-cache-root", default="data/hitran_cache_tv3")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--temp-dir", type=str, default=None)
    parser.add_argument("--keep-chunks", action="store_true", default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workers = args.workers if args.workers is not None else default_worker_count(args.sequences)
    spec = TunnelVentilationBenchmarkGenerationSpec(
        dataset_slug=args.dataset,
        sequence_count=args.sequences,
        seed=args.seed,
        timesteps=args.timesteps,
        dt_s=args.dt_s,
        storage=args.storage,
        multi_path_phase=args.multi_path_phase,
        stage_profile=args.stage_profile,
        stage_jitter=args.stage_jitter,
        sampling_strategy=args.sampling_strategy,
        path_lms=args.path_lms,
        optical_absorption_backend=args.optical_absorption_backend,
        hitran_cache_root=args.hitran_cache_root,
        workers=workers,
        chunk_size=args.chunk_size,
        temp_dir=args.temp_dir,
        keep_chunks=args.keep_chunks,
    )
    result = generate_tunnel_ventilation_benchmark_dataset(Path(args.output_root), spec)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
