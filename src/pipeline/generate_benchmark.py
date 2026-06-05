from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sim.generation.benchmark import (
    DEFAULT_HITRAN_CACHE_ROOT,
    DEFAULT_WAVEFORM_PATH_LMS,
    TIME_AXIS_PRESETS,
    BenchmarkGenerationSpec,
    default_worker_count,
    generate_benchmark_dataset,
    resolve_time_axis_preset,
)
from sim.generation.optical_backend import VALID_OPTICAL_ABSORPTION_BACKENDS
from sim.generation.phases import PHASE_SCHEDULES

FORMAL_HITRAN_STANDARD_PRESET = "formal-hitran-standard-6000"
GENERAL_DEFAULT_SEED = 20260524
GENERAL_DEFAULT_TIME_AXIS_PRESET = "short"


@dataclass(frozen=True, slots=True)
class ResolvedCliDefaults:
    dataset: str
    sequences: int
    seed: int
    time_axis_preset: str


def parse_path_lms(value: str) -> tuple[float, ...]:
    path_lms = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(path_lms) == 0:
        raise argparse.ArgumentTypeError("--path-lms must contain at least one comma-separated value")
    if any(path_l_m <= 0.0 for path_l_m in path_lms):
        raise argparse.ArgumentTypeError("--path-lms values must be > 0")
    return path_lms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a v4 benchmark dataset.")
    parser.add_argument(
        "--experiment-preset",
        choices=(FORMAL_HITRAN_STANDARD_PRESET,),
        default=None,
        help="Apply a named experiment preset before explicit CLI overrides.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--sequences", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--time-axis-preset", choices=tuple(TIME_AXIS_PRESETS), default=None)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--dt-s", type=float)
    parser.add_argument("--storage", choices=("memmap", "npz", "both"), default="memmap")
    parser.add_argument("--multi-path-phase", choices=("off", "baseline", "steady"), default="steady")
    parser.add_argument("--stage-profile", choices=tuple(PHASE_SCHEDULES), default="standard_exposure")
    parser.add_argument("--stage-jitter", type=float, default=0.0)
    parser.add_argument("--sampling-strategy", choices=("lhs", "random"), default="lhs")
    parser.add_argument("--path-lms", type=parse_path_lms, default=DEFAULT_WAVEFORM_PATH_LMS)
    parser.add_argument("--optical-absorption-backend", choices=VALID_OPTICAL_ABSORPTION_BACKENDS, default="hitran_hapi_v1")
    parser.add_argument("--hitran-cache-root", default=DEFAULT_HITRAN_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=None, help="Worker processes for sequence generation (default: CPU count - 2, capped at 24).")
    parser.add_argument("--chunk-size", type=int, default=None, help="Sequences per worker chunk (default: ceil(sequences / workers)).")
    parser.add_argument("--temp-dir", type=str, default=None, help="Chunk temp directory (default: <dataset staging dir>/.chunks).")
    parser.add_argument("--keep-chunks", action="store_true", default=False, help="Keep chunk temp files for debugging.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    resolved = _resolve_cli_defaults(args, parser)
    time_axis = resolve_time_axis_preset(resolved.time_axis_preset)
    timesteps = args.timesteps if args.timesteps is not None else time_axis.timesteps
    dt_s = args.dt_s if args.dt_s is not None else time_axis.dt_s
    workers = args.workers if args.workers is not None else default_worker_count(resolved.sequences)
    summary = generate_benchmark_dataset(
        Path(args.output_root),
        BenchmarkGenerationSpec(
            dataset_slug=resolved.dataset,
            sequence_count=resolved.sequences,
            seed=resolved.seed,
            timesteps=timesteps,
            dt_s=dt_s,
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
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _resolve_cli_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> ResolvedCliDefaults:
    if args.experiment_preset == FORMAL_HITRAN_STANDARD_PRESET:
        dataset = args.dataset or "wv4-formal-hitran-standard-6000"
        sequences = args.sequences if args.sequences is not None else 6000
        seed = args.seed if args.seed is not None else 20260603
        time_axis_preset = args.time_axis_preset or "standard"
    else:
        if args.dataset is None:
            parser.error("--dataset is required unless --experiment-preset is provided")
        if args.sequences is None:
            parser.error("--sequences is required unless --experiment-preset is provided")
        dataset = args.dataset
        sequences = args.sequences
        seed = args.seed if args.seed is not None else GENERAL_DEFAULT_SEED
        time_axis_preset = args.time_axis_preset or GENERAL_DEFAULT_TIME_AXIS_PRESET

    return ResolvedCliDefaults(
        dataset=dataset,
        sequences=sequences,
        seed=seed,
        time_axis_preset=time_axis_preset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
