from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from tv3.ml.extratrees_head import ExtraTreesHeadConfig
from tv3.ml.extratrees_training import train_tv3_extratrees_regressor
from tv3.ml.rocket_features import RocketFeatureConfig
from tv3.ml.rocket_training import write_rocket_training_payload


REQUIRED_CONFIG_KEYS = {
    "dataset_dir",
    "output_dir",
    "feature_builder",
    "include_slow",
    "physics_arrays",
    "sequence_statistics",
    "phase_windows",
    "early_fractions",
    "eval_splits",
    "extratrees_n_estimators",
    "extratrees_max_features",
    "extratrees_min_samples_leaf",
    "extratrees_max_depth",
    "extratrees_n_jobs",
    "seed",
}
OPTIONAL_CONFIG_KEYS = {"cache_dir", "slow_channels", "train_split"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the tv3 R7 observed-feature ExtraTrees regression probe.")
    parser.add_argument("--config", type=Path, required=True, help="R7 ExtraTrees JSON config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    feature_config = _build_feature_config(config)
    result = train_tv3_extratrees_regressor(
        Path(config["dataset_dir"]),
        feature_config=feature_config,
        cache_dir=Path(config["cache_dir"]) if config.get("cache_dir") is not None else None,
        train_split=str(config.get("train_split", "train")),
        eval_splits=_parse_csv(config["eval_splits"]),
        extratrees_config=ExtraTreesHeadConfig(
            n_estimators=int(config["extratrees_n_estimators"]),
            max_features=float(config["extratrees_max_features"]),
            min_samples_leaf=int(config["extratrees_min_samples_leaf"]),
            max_depth=config["extratrees_max_depth"],
            n_jobs=int(config["extratrees_n_jobs"]),
            seed=int(config["seed"]),
        ),
    )
    output_dir = Path(config["output_dir"])
    payload = write_rocket_training_payload(result, output_dir / "metrics.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _build_feature_config(config: dict[str, Any]) -> RocketFeatureConfig:
    return RocketFeatureConfig(
        feature_builder=str(config["feature_builder"]),
        include_slow=_parse_bool(config["include_slow"]),
        slow_channels=_parse_optional_csv(config.get("slow_channels")),
        physics_arrays=_parse_csv(config["physics_arrays"]),
        sequence_statistics=_parse_csv(config["sequence_statistics"]),
        phase_windows=_parse_csv(config["phase_windows"]),
        early_fractions=_parse_float_csv(config["early_fractions"]),
    )


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("R7 ExtraTrees config must be a JSON object")
    unknown = set(payload) - REQUIRED_CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown R7 ExtraTrees config keys: {sorted(unknown)}")
    missing = REQUIRED_CONFIG_KEYS - set(payload)
    if missing:
        raise ValueError(f"missing R7 ExtraTrees config keys: {sorted(missing)}")
    max_depth = payload["extratrees_max_depth"]
    if max_depth is not None and (not isinstance(max_depth, int) or isinstance(max_depth, bool)):
        raise ValueError("extratrees_max_depth must be an integer or null")
    return payload


def _parse_csv(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_optional_csv(value: Any | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    parts = _parse_csv(value)
    return parts or None


def _parse_float_csv(value: Any) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_csv(value))


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
