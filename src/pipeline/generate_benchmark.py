from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sim.generation.benchmark import DEFAULT_HITRAN_CACHE_ROOT, DEFAULT_WAVEFORM_PATH_LMS, BenchmarkGenerationSpec, generate_benchmark_dataset
from sim.generation.optical_backend import VALID_OPTICAL_ABSORPTION_BACKENDS


def parse_path_lms(value: str) -> tuple[float, ...]:
    path_lms = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(path_lms) == 0:
        raise argparse.ArgumentTypeError("--path-lms must contain at least one comma-separated value")
    if any(path_l_m <= 0.0 for path_l_m in path_lms):
        raise argparse.ArgumentTypeError("--path-lms values must be > 0")
    return path_lms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a v4 benchmark dataset.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--timesteps", type=int, default=128)
    parser.add_argument("--dt-s", type=float, default=0.5)
    parser.add_argument("--storage", choices=("memmap", "npz", "both"), default="memmap")
    parser.add_argument("--multi-path-phase", choices=("off", "baseline", "steady"), default="steady")
    parser.add_argument("--sampling-strategy", choices=("lhs", "random"), default="lhs")
    parser.add_argument("--path-lms", type=parse_path_lms, default=DEFAULT_WAVEFORM_PATH_LMS)
    parser.add_argument("--optical-absorption-backend", choices=VALID_OPTICAL_ABSORPTION_BACKENDS, default="hitran_hapi_v1")
    parser.add_argument("--hitran-cache-root", default=DEFAULT_HITRAN_CACHE_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_benchmark_dataset(
        Path(args.output_root),
        BenchmarkGenerationSpec(
            dataset_slug=args.dataset,
            sequence_count=args.sequences,
            seed=args.seed,
            timesteps=args.timesteps,
            dt_s=args.dt_s,
            storage=args.storage,
            multi_path_phase=args.multi_path_phase,
            sampling_strategy=args.sampling_strategy,
            path_lms=args.path_lms,
            optical_absorption_backend=args.optical_absorption_backend,
            hitran_cache_root=args.hitran_cache_root,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
