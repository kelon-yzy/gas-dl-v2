from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hg.common.composition import PERCENT_TOTAL, close_to_unit_interval, resolve_zero_replacement_epsilon
from hg.common.composition import TRAIN_MIN_POSITIVE_HALF_EPSILON
from hg.common.splits import load_splits, resolve_split_indices


DEFAULT_ALR_REFERENCE = "x_CH4"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect composition labels before log-ratio target experiments.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="v4 benchmark dataset root.")
    parser.add_argument(
        "--alr-reference",
        type=str,
        default=DEFAULT_ALR_REFERENCE,
        help="Reference component used for ALR stability diagnostics.",
    )
    parser.add_argument("--json", action="store_true", default=False, help="Print JSON instead of Markdown.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional path to write the inspection report.")
    parser.add_argument("--json-output-path", type=Path, default=None, help="Optional path to write the inspection payload as JSON.")
    return parser


def inspect_composition_labels(
    dataset_dir: Path | str,
    *,
    alr_reference: str = DEFAULT_ALR_REFERENCE,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float64)
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))
    sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_rows = load_splits(dataset_dir / "splits")
    split_indices = resolve_split_indices(split_rows, sequence_ids)
    reference_index = _component_index(label_names, alr_reference)
    train_epsilon = resolve_zero_replacement_epsilon(
        TRAIN_MIN_POSITIVE_HALF_EPSILON,
        labels[split_indices["train"]],
    )

    return {
        "dataset_dir": str(dataset_dir),
        "label_names": label_names,
        "alr_reference": alr_reference,
        "recommended_zero_replacement": {
            "strategy": TRAIN_MIN_POSITIVE_HALF_EPSILON,
            "epsilon": train_epsilon,
            "epsilon_percent": train_epsilon * PERCENT_TOTAL,
        },
        "splits": {
            split: _split_summary(labels[indices], label_names, reference_index)
            for split, indices in split_indices.items()
        },
    }


def format_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Composition Label Inspection",
        "",
        f"- dataset_dir: `{payload['dataset_dir']}`",
        f"- alr_reference: `{payload['alr_reference']}`",
        (
            "- recommended epsilon: "
            f"`{payload['recommended_zero_replacement']['epsilon']:.10g}` "
            f"({payload['recommended_zero_replacement']['strategy']})"
        ),
        "",
        "| split | rows | component | zero count | zero ratio | min % | min positive % | max % |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for split, summary in payload["splits"].items():
        for component, stats in summary["components"].items():
            lines.append(
                f"| {split} | {summary['total_rows']} | {component} | {stats['zero_count']} | "
                f"{stats['zero_ratio']:.6f} | {_format_optional_float(stats['min_percent'])} | "
                f"{_format_optional_float(stats['min_positive_percent'])} | "
                f"{_format_optional_float(stats['max_percent'])} |"
            )
    lines.extend(
        [
            "",
            "| split | ALR reference | log-reference mean | log-reference variance |",
            "|---|---|---:|---:|",
        ]
    )
    for split, summary in payload["splits"].items():
        reference = summary["alr_reference"]
        lines.append(
            f"| {split} | {payload['alr_reference']} | "
            f"{reference['log_mean']:.6f} | {reference['log_variance']:.6f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _split_summary(values_percent: np.ndarray, label_names: tuple[str, ...], reference_index: int) -> dict[str, Any]:
    if values_percent.ndim != 2 or values_percent.shape[1] != len(label_names):
        raise ValueError(
            f"labels must be shaped (N, {len(label_names)}), got {tuple(values_percent.shape)}"
        )
    total_rows = int(values_percent.shape[0])
    if total_rows == 0:
        return {
            "total_rows": 0,
            "components": {
                name: {
                    "zero_count": 0,
                    "zero_ratio": 0.0,
                    "min_percent": None,
                    "min_positive_percent": None,
                    "max_percent": None,
                }
                for name in label_names
            },
            "alr_reference": {"log_mean": 0.0, "log_variance": 0.0},
        }

    unit = close_to_unit_interval(values_percent)
    components = {}
    for index, name in enumerate(label_names):
        column = values_percent[:, index]
        positives = column[column > 0.0]
        components[name] = {
            "zero_count": int(np.count_nonzero(column == 0.0)),
            "zero_ratio": float(np.count_nonzero(column == 0.0) / total_rows),
            "min_percent": float(np.min(column)),
            "min_positive_percent": None if positives.size == 0 else float(np.min(positives)),
            "max_percent": float(np.max(column)),
        }

    reference = unit[:, reference_index]
    if np.any(reference <= 0.0):
        raise ValueError("ALR reference component must be positive for log-reference diagnostics")
    log_reference = np.log(reference)
    return {
        "total_rows": total_rows,
        "components": components,
        "alr_reference": {
            "log_mean": float(np.mean(log_reference)),
            "log_variance": float(np.var(log_reference)),
        },
    }


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


def _component_index(label_names: tuple[str, ...], name: str) -> int:
    try:
        return label_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown component {name!r}. Available: {label_names}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = inspect_composition_labels(args.dataset_dir, alr_reference=args.alr_reference)
    if args.json:
        output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        output = format_markdown_report(payload)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(output, encoding="utf-8")
    if args.json_output_path is not None:
        args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
