"""掘进通风场景 benchmark 生成 CLI。

阶段 1 只支持 empirical_v1 光学吸收后端。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from tv3.sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
)
from tv3.sim.generation.phases import PHASE_SCHEDULES
from tv3.sim.generation.tunnel_ventilation.benchmark import (
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
    if not path_lms:
        raise argparse.ArgumentTypeError("--path-lms must contain at least one comma-separated value")
    if any(path_l_m <= 0.0 for path_l_m in path_lms):
        raise argparse.ArgumentTypeError("--path-lms values must be > 0")
    return path_lms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a tv3 tunnel ventilation benchmark dataset.")
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
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--keep-chunks", action="store_true")
    parser.add_argument(
        "--skip-fiber-mic",
        action="store_true",
        help="Skip fiber microphone waveform generation.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("random", "spxy_v1", "lhs_stratified_split_v1"),
        default="random",
    )
    parser.add_argument("--spxy-alpha", type=float, default=0.5)
    parser.add_argument(
        "--extrapolation-strategy",
        choices=("none", "y_margin_ood", "lhs_boundary", "kmeans_boundary"),
        default="none",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workers = args.workers if args.workers is not None else default_worker_count(args.sequences)

    print(
        f"[tv3-gen] dataset={args.dataset} sequences={args.sequences} "
        f"timesteps={args.timesteps} workers={workers} skip_fiber_mic={args.skip_fiber_mic}",
        flush=True,
    )
    started_at = time.perf_counter()
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
        skip_fiber_mic=args.skip_fiber_mic,
        split_strategy=args.split_strategy,
        spxy_alpha=args.spxy_alpha,
        extrapolation_strategy=args.extrapolation_strategy,
    )
    result = generate_tunnel_ventilation_benchmark_dataset(Path(args.output_root), spec)
    elapsed = time.perf_counter() - started_at
    print(
        f"[tv3-gen] done output={result['output_dir']} "
        f"sequences={result['sequence_count']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
