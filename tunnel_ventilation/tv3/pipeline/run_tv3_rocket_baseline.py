from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tv3.ml.minirocket_features import (
    MINIROCKET_RAW_BUILDER,
    MINIROCKET_SCALAR_BUILDER,
    DEFAULT_MINIROCKET_KERNEL_LENGTHS,
    DEFAULT_MINIROCKET_NUM_KERNELS,
    DEFAULT_MINIROCKET_KERNEL_SEED,
    MiniRocketFeatureConfig,
)
from tv3.ml.rocket_features import (
    DEFAULT_EARLY_FRACTIONS,
    DEFAULT_FEATURE_BUILDER,
    DEFAULT_PHASE_WINDOWS,
    DEFAULT_ROCKET_SEQUENCE_STATISTICS,
    DEFAULT_PHYSICS_ARRAYS,
    RocketFeatureConfig,
)
from tv3.ml.rocket_training import (
    DEFAULT_RIDGE_ALPHAS,
    rocket_training_payload,
    train_tv3_rocket_regressor,
    write_rocket_training_payload,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_dir": None,
    "output_dir": None,
    "cache_dir": None,
    "feature_set": "physics_stats",
    "head": "ridgecv",
    "feature_builder": None,
    "include_slow": True,
    "slow_channels": None,
    "physics_arrays": ",".join(DEFAULT_PHYSICS_ARRAYS),
    "sequence_statistics": ",".join(DEFAULT_ROCKET_SEQUENCE_STATISTICS),
    "phase_windows": ",".join(DEFAULT_PHASE_WINDOWS),
    "early_fractions": ",".join(str(value) for value in DEFAULT_EARLY_FRACTIONS),
    "num_kernels": DEFAULT_MINIROCKET_NUM_KERNELS,
    "kernel_lengths": ",".join(str(value) for value in DEFAULT_MINIROCKET_KERNEL_LENGTHS),
    "kernel_seed": DEFAULT_MINIROCKET_KERNEL_SEED,
    "raw_zscore": True,
    "train_split": "train",
    "eval_splits": "val,test,extrapolation",
    "ridge_alphas": ",".join(str(value) for value in DEFAULT_RIDGE_ALPHAS),
    "closed_form_alpha": 1.0,
}

FEATURE_SET_TO_BUILDER = {
    "physics_stats": DEFAULT_FEATURE_BUILDER,
    "minirocket_scalar": MINIROCKET_SCALAR_BUILDER,
    "minirocket_raw": MINIROCKET_RAW_BUILDER,
}
MINIROCKET_FEATURE_SETS = ("minirocket_scalar", "minirocket_raw")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tv3 rocket-style regression baseline.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config file. CLI args override config values.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="tv3 dataset root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write metrics.json.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional feature cache directory.")
    parser.add_argument("--feature-set", choices=("physics_stats", "minirocket_scalar", "minirocket_raw"), default=None, help="Feature family to run.")
    parser.add_argument("--head", choices=("ridgecv", "ridge_closed_form"), default=None, help="Regression head.")
    parser.add_argument("--feature-builder", type=str, default=None, help="Feature builder cache name. Overrides feature_set's default mapping if given.")
    parser.add_argument("--include-slow", type=str, default=None, help="true/false.")
    parser.add_argument("--slow-channels", type=str, default=None, help="Comma-separated slow channel allowlist.")
    parser.add_argument("--physics-arrays", type=str, default=None, help="Comma-separated physics array names (R1a only).")
    parser.add_argument("--sequence-statistics", type=str, default=None, help="Comma-separated sequence statistics (R1b cross-timestep pooling).")
    parser.add_argument("--phase-windows", type=str, default=None, help="Comma-separated phase windows (physics_stats only).")
    parser.add_argument("--early-fractions", type=str, default=None, help="Comma-separated early fractions in (0,1] (physics_stats only).")
    parser.add_argument("--num-kernels", type=int, default=None, help="MiniRocket fixed kernel count (R1a/R1b).")
    parser.add_argument("--kernel-lengths", type=str, default=None, help="Comma-separated MiniRocket kernel lengths (R1a/R1b).")
    parser.add_argument("--kernel-seed", type=int, default=None, help="MiniRocket kernel RNG seed (R1a/R1b).")
    parser.add_argument("--raw-zscore", type=str, default=None, help="true/false, R1b per-frame z-score on dequantized waveform.")
    parser.add_argument("--train-split", type=str, default=None, help="Train split name.")
    parser.add_argument("--eval-splits", type=str, default=None, help="Comma-separated eval splits.")
    parser.add_argument("--ridge-alphas", type=str, default=None, help="Comma-separated RidgeCV alphas.")
    parser.add_argument("--closed-form-alpha", type=float, default=None, help="Alpha for ridge_closed_form.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _resolve_args(parser.parse_args(argv))
    if args.dataset_dir is None:
        parser.error("--dataset-dir is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.feature_set not in FEATURE_SET_TO_BUILDER:
        raise ValueError(f"unsupported feature_set {args.feature_set!r}")

    # --feature-builder 未显式覆盖时按 feature_set 取默认映射
    feature_builder = args.feature_builder or FEATURE_SET_TO_BUILDER[args.feature_set]
    is_minirocket = args.feature_set in MINIROCKET_FEATURE_SETS

    if is_minirocket:
        feature_config = _build_minirocket_config(args, feature_builder)
    else:
        feature_config = _build_physics_config(args, feature_builder)

    result = train_tv3_rocket_regressor(
        args.dataset_dir,
        feature_config=feature_config,
        cache_dir=args.cache_dir,
        head=args.head,
        train_split=args.train_split,
        eval_splits=_parse_csv(args.eval_splits),
        ridge_alphas=_parse_float_csv(args.ridge_alphas),
        closed_form_alpha=args.closed_form_alpha,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = write_rocket_training_payload(result, output_dir / "metrics.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _build_physics_config(args: argparse.Namespace, feature_builder: str) -> RocketFeatureConfig:
    return RocketFeatureConfig(
        feature_builder=feature_builder,
        include_slow=args.include_slow,
        slow_channels=_parse_optional_csv(args.slow_channels),
        physics_arrays=_parse_csv(args.physics_arrays),
        sequence_statistics=_parse_csv(args.sequence_statistics),
        phase_windows=_parse_csv(args.phase_windows),
        early_fractions=_parse_float_csv(args.early_fractions),
    )


def _build_minirocket_config(args: argparse.Namespace, feature_builder: str) -> MiniRocketFeatureConfig:
    return MiniRocketFeatureConfig(
        feature_builder=feature_builder,
        include_slow=args.include_slow,
        slow_channels=_parse_optional_csv(args.slow_channels),
        physics_arrays=_parse_csv(args.physics_arrays),
        sequence_statistics=_parse_csv(args.sequence_statistics),
        num_kernels=args.num_kernels,
        kernel_lengths=_parse_int_csv(args.kernel_lengths),
        kernel_seed=args.kernel_seed,
        raw_zscore=_parse_bool(args.raw_zscore),
    )


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = dict(DEFAULT_CONFIG)
    if args.config is not None:
        config.update(_load_config(args.config))
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value
    config["dataset_dir"] = Path(config["dataset_dir"]) if config.get("dataset_dir") is not None else None
    config["output_dir"] = Path(config["output_dir"]) if config.get("output_dir") is not None else None
    config["cache_dir"] = Path(config["cache_dir"]) if config.get("cache_dir") is not None else None
    config["include_slow"] = _parse_bool(config["include_slow"])
    config["raw_zscore"] = _parse_bool(config["raw_zscore"])
    return argparse.Namespace(**config)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("rocket config must be a JSON object")
    allowed = set(DEFAULT_CONFIG)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown rocket config keys: {sorted(unknown)}")
    return payload


def _parse_csv(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        parts = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not parts:
        raise ValueError("comma-separated argument must not be empty")
    return parts


def _parse_optional_csv(value: Any | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = _parse_csv(value)
    return parts or None


def _parse_float_csv(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_csv(value))


def _parse_int_csv(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_csv(value))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
