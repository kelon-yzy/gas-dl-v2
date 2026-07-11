from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.pipeline.build_tv3_raw_dsp_features import DEFAULT_CACHE_ROOT


AUDIT_SCHEMA_VERSION = "tv3-d2b-frame-fidelity-1"
EVALUATION_SPLITS = ("val", "test", "extrapolation")
PHASE_ORDER = ("baseline", "exposure", "steady", "recovery")
OUTPUT_FILENAMES = (
    "metrics.json",
    "manifest_snapshot.json",
    "peak_error_by_split.csv",
    "peak_error_by_phase.csv",
    "quality_coverage_by_split.csv",
)


@dataclass(frozen=True, slots=True)
class FidelityThresholds:
    peak_mae_samples: float = 0.15
    peak_p95_abs_samples: float = 0.25
    peak_abs_bias_samples: float = 0.05
    sound_speed_mae_m_per_s: float = 0.15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit tv3 D2b RawDSP frame fidelity.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="tv3 dataset root.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="RawDSP cache directory; defaults to the dataset's formal cache path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New audit output directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_d2b_frame_fidelity(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics_path": str(args.output_dir / "metrics.json"),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "passed" else 1


def audit_d2b_frame_fidelity(
    *,
    dataset_dir: Path | str,
    output_dir: Path | str,
    cache_dir: Path | str | None = None,
    thresholds: FidelityThresholds | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else dataset_dir / DEFAULT_CACHE_ROOT
    output_dir = Path(output_dir)
    thresholds = thresholds or FidelityThresholds()

    _prepare_output_dir(output_dir)
    manifest = _load_and_validate_cache_manifest(dataset_dir, cache_dir)
    dataset_sequence_ids = _load_string_array(dataset_dir / "metadata" / "sequence_ids.npy")
    cache_sequence_ids = _load_string_array(cache_dir / "sequence_ids.npy")
    if cache_sequence_ids != dataset_sequence_ids:
        raise ValueError("RawDSP cache sequence_ids do not exactly match the source dataset")

    sample_rate_hz = _load_sample_rate_hz(dataset_dir)
    phase_ids = _load_phase_ids(dataset_dir / "sequences" / "slow_sequence_long.csv", dataset_sequence_ids)
    split_indices = _load_split_indices(dataset_dir, dataset_sequence_ids)
    arrays = _load_audit_arrays(dataset_dir, cache_dir, len(dataset_sequence_ids), len(phase_ids[0]))

    peak_error_samples = arrays["raw_peak_index"] - arrays["observed_tof_s"] * sample_rate_hz
    sound_speed_error = arrays["raw_sound_speed_m_per_s"] - arrays["observed_sound_speed_m_per_s"]
    corrected_tof_error_s = arrays["raw_tof_corrected_s"] - arrays["true_tof_s"]
    if not np.isfinite(peak_error_samples).all():
        raise ValueError("peak error contains non-finite values")
    if not np.isfinite(sound_speed_error).all():
        raise ValueError("sound-speed error contains non-finite values")
    if not np.isfinite(corrected_tof_error_s).all():
        raise ValueError("corrected TOF error contains non-finite values")

    split_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name, indices in split_indices.items():
        index_array = np.asarray(indices, dtype=np.intp)
        metrics = _build_split_metrics(
            peak_error_samples[index_array],
            sound_speed_error[index_array],
            corrected_tof_error_s[index_array],
            arrays["accepted"][index_array],
            arrays["boundary_hit"][index_array],
            arrays["clipped"][index_array],
            thresholds,
            gate_required=split_name in EVALUATION_SPLITS,
        )
        split_metrics[split_name] = metrics
        split_rows.append(_split_csv_row(split_name, metrics))
        quality_rows.append(_quality_csv_row(split_name, metrics))
        phase_rows.extend(
            _phase_csv_rows(
                split_name,
                index_array,
                phase_ids,
                peak_error_samples,
                arrays["accepted"],
            )
        )

    required_metrics = [split_metrics[name] for name in EVALUATION_SPLITS]
    status = "passed" if all(metrics["gate"]["passed"] for metrics in required_metrics) else "failed"
    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": status,
        "primary_population": "all_frames",
        "gate_splits": list(EVALUATION_SPLITS),
        "thresholds": asdict(thresholds),
        "source": {
            "dataset_dir": str(dataset_dir),
            "cache_dir": str(cache_dir),
            "cache_manifest_sha256": _file_sha256(cache_dir / "manifest.json"),
            "cache_build_signature": manifest["build_signature"],
            "template_digest": manifest["template_digest"],
            "template_mode": manifest["template_mode"],
            "template_source_split": manifest["template_source_split"],
            "sample_rate_hz": sample_rate_hz,
        },
        "splits": split_metrics,
    }
    snapshot = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_dir": str(dataset_dir),
        "source_cache_dir": str(cache_dir),
        "cache_manifest_sha256": _file_sha256(cache_dir / "manifest.json"),
        "cache_manifest": manifest,
    }
    _write_json(output_dir / "metrics.json", result)
    _write_json(output_dir / "manifest_snapshot.json", snapshot)
    _write_csv(output_dir / "peak_error_by_split.csv", split_rows)
    _write_csv(output_dir / "peak_error_by_phase.csv", phase_rows)
    _write_csv(output_dir / "quality_coverage_by_split.csv", quality_rows)
    return result


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"audit output already exists: {existing}; choose a new output directory")


def _load_and_validate_cache_manifest(dataset_dir: Path, cache_dir: Path) -> dict[str, Any]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing RawDSP cache manifest: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    required = {
        "schema_version": "tv3-raw-dsp-frame-1",
        "complete_dataset": True,
        "template_mode": "train_baseline_median",
        "template_source_split": "train",
        "diagnostic_only": False,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"RawDSP cache manifest field {key!r} must be {expected!r}, got {manifest.get(key)!r}"
            )
    source_dataset = manifest.get("source_dataset")
    if not isinstance(source_dataset, str) or Path(source_dataset).resolve() != dataset_dir.resolve():
        raise ValueError("RawDSP cache source_dataset does not match --dataset-dir")
    if not manifest.get("build_signature") or not manifest.get("template_digest"):
        raise ValueError("RawDSP cache manifest is missing build_signature or template_digest")
    return manifest


def _load_audit_arrays(
    dataset_dir: Path,
    cache_dir: Path,
    sequence_count: int,
    timesteps: int,
) -> dict[str, np.ndarray]:
    paths = {
        "raw_peak_index": cache_dir / "ultrasonic_peak_index_raw_dsp.npy",
        "raw_tof_corrected_s": cache_dir / "ultrasonic_tof_corrected_raw_dsp_s.npy",
        "raw_sound_speed_m_per_s": cache_dir / "ultrasonic_sound_speed_raw_dsp_m_per_s.npy",
        "accepted": cache_dir / "ultrasonic_raw_dsp_accepted.npy",
        "boundary_hit": cache_dir / "ultrasonic_raw_dsp_boundary_hit.npy",
        "clipped": cache_dir / "ultrasonic_raw_dsp_clipped.npy",
        "observed_tof_s": dataset_dir / "sequences" / "ultrasonic_tof_observed_s.npy",
        "observed_sound_speed_m_per_s": dataset_dir
        / "sequences"
        / "ultrasonic_sound_speed_estimated_m_per_s.npy",
        "true_tof_s": dataset_dir / "sequences" / "ultrasonic_tof_s.npy",
    }
    arrays: dict[str, np.ndarray] = {}
    expected_shape = (sequence_count, timesteps)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing D2b fidelity input: {path}")
        values = np.load(path, mmap_mode="r")
        if values.shape != expected_shape:
            raise ValueError(f"fidelity input shape mismatch for {name}: {values.shape} != {expected_shape}")
        arrays[name] = values
    return arrays


def _load_sample_rate_hz(dataset_dir: Path) -> float:
    waveform_spec = _read_json_object(dataset_dir / "metadata" / "waveform_spec.json")
    ultrasonic = waveform_spec.get("ultrasonic")
    if not isinstance(ultrasonic, dict) or "sample_rate_hz" not in ultrasonic:
        raise ValueError("waveform_spec.json must contain ultrasonic.sample_rate_hz")
    sample_rate_hz = float(ultrasonic["sample_rate_hz"])
    if sample_rate_hz <= 0.0:
        raise ValueError("ultrasonic.sample_rate_hz must be > 0")
    return sample_rate_hz


def _load_split_indices(dataset_dir: Path, sequence_ids: list[str]) -> dict[str, list[int]]:
    split_indices = resolve_split_indices(load_splits(dataset_dir / "splits"), sequence_ids)
    required = {"train", *EVALUATION_SPLITS}
    missing = required - set(split_indices)
    if missing:
        raise ValueError(f"split files are missing required splits: {sorted(missing)}")
    return {name: split_indices[name] for name in ("train", *EVALUATION_SPLITS)}


def _load_phase_ids(path: Path, sequence_ids: list[str]) -> list[tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing phase CSV: {path}")
    by_sequence: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_id", "timestep", "phase_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"phase CSV missing required columns: {sorted(required)}")
        for row in reader:
            by_sequence.setdefault(row["sequence_id"], []).append((int(row["timestep"]), row["phase_id"]))
    phases = [tuple(phase for _, phase in sorted(by_sequence[sequence_id])) for sequence_id in sequence_ids]
    lengths = {len(items) for items in phases}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("phase CSV has inconsistent or empty timestep counts")
    return phases


def _build_split_metrics(
    peak_error_samples: np.ndarray,
    sound_speed_error: np.ndarray,
    corrected_tof_error_s: np.ndarray,
    accepted: np.ndarray,
    boundary_hit: np.ndarray,
    clipped: np.ndarray,
    thresholds: FidelityThresholds,
    *,
    gate_required: bool,
) -> dict[str, Any]:
    peak = _error_metrics(peak_error_samples)
    sound_speed = _error_metrics(sound_speed_error)
    corrected_tof = _error_metrics(corrected_tof_error_s)
    frame_count = int(peak_error_samples.size)
    accepted_count = int(np.count_nonzero(accepted))
    gate_checks = {
        "peak_mae_samples": peak["mae"] <= thresholds.peak_mae_samples,
        "peak_p95_abs_samples": peak["p95_abs"] <= thresholds.peak_p95_abs_samples,
        "peak_abs_bias_samples": abs(peak["bias"]) <= thresholds.peak_abs_bias_samples,
        "sound_speed_mae_m_per_s": sound_speed["mae"] <= thresholds.sound_speed_mae_m_per_s,
    }
    return {
        "frame_count": frame_count,
        "peak_error_samples": peak,
        "sound_speed_error_m_per_s": sound_speed,
        "corrected_tof_error_s": corrected_tof,
        "quality": {
            "accepted_count": accepted_count,
            "accepted_fraction": accepted_count / frame_count,
            "boundary_hit_count": int(np.count_nonzero(boundary_hit)),
            "boundary_hit_fraction": float(np.mean(boundary_hit)),
            "clipped_count": int(np.count_nonzero(clipped)),
            "clipped_fraction": float(np.mean(clipped)),
        },
        "gate": {
            "required": gate_required,
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
        },
    }


def _error_metrics(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mae": float(np.mean(np.abs(flat))),
        "p95_abs": float(np.percentile(np.abs(flat), 95)),
        "bias": float(np.mean(flat)),
    }


def _split_csv_row(split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    peak = metrics["peak_error_samples"]
    speed = metrics["sound_speed_error_m_per_s"]
    corrected_tof = metrics["corrected_tof_error_s"]
    gate = metrics["gate"]
    return {
        "split": split_name,
        "frame_count": metrics["frame_count"],
        "peak_mae_samples": peak["mae"],
        "peak_p95_abs_samples": peak["p95_abs"],
        "peak_bias_samples": peak["bias"],
        "sound_speed_mae_m_per_s": speed["mae"],
        "corrected_tof_mae_s": corrected_tof["mae"],
        "corrected_tof_p95_abs_s": corrected_tof["p95_abs"],
        "corrected_tof_bias_s": corrected_tof["bias"],
        "gate_required": gate["required"],
        "gate_passed": gate["passed"],
    }


def _quality_csv_row(split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    quality = metrics["quality"]
    return {"split": split_name, "frame_count": metrics["frame_count"], **quality}


def _phase_csv_rows(
    split_name: str,
    indices: np.ndarray,
    phase_ids: list[tuple[str, ...]],
    peak_error_samples: np.ndarray,
    accepted: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_phases = np.asarray([phase_ids[index] for index in indices])
    selected_errors = peak_error_samples[indices]
    selected_accepted = accepted[indices]
    for phase in PHASE_ORDER:
        mask = selected_phases == phase
        if not np.any(mask):
            raise ValueError(f"split {split_name!r} has no frames for phase {phase!r}")
        metrics = _error_metrics(selected_errors[mask])
        frame_count = int(np.count_nonzero(mask))
        accepted_count = int(np.count_nonzero(selected_accepted[mask]))
        rows.append(
            {
                "split": split_name,
                "phase": phase,
                "frame_count": frame_count,
                "peak_mae_samples": metrics["mae"],
                "peak_p95_abs_samples": metrics["p95_abs"],
                "peak_bias_samples": metrics["bias"],
                "accepted_count": accepted_count,
                "accepted_fraction": accepted_count / frame_count,
            }
        )
    return rows


def _load_string_array(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing sequence IDs: {path}")
    return [str(value) for value in np.load(path, allow_pickle=True).tolist()]


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
