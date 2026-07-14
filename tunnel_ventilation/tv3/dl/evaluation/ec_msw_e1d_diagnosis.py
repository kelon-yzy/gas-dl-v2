"""E1d frozen representation diagnosis: locate the E1r vs B1 information gap.

E1d does not train new deep nets, does not start E2, and does not rewrite B1/B7.
All probes fit train-only StandardScaler + RidgeCV and evaluate val/test/extrapolation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tv3.dl.evaluation.ec_msw_e1_audit import (
    _batch_x as _e1_batch_x,
    _build_dataset as _build_e1_dataset,
    _build_input_preprocess as _build_e1_input_preprocess,
    _load_model as _load_e1_model,
    _loader as _e1_loader,
)

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.metrics import component_regression_metrics, regression_metrics
from tv3.ml.raw_dsp_features import FORMAL_SLOW_CHANNELS, RAW_DSP_FRAME_SCHEMA_VERSION
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.ml.rocket_features import (
    DEFAULT_EARLY_FRACTIONS,
    DEFAULT_PHASE_WINDOWS,
    DEFAULT_ROCKET_SEQUENCE_STATISTICS,
    RAW_DSP_FRAME_CACHE_ROOT,
    RAW_DSP_PHYSICS_ARRAYS,
    _load_phase_lookup,
    _load_str_array,
    _select_slow_channels,
    _windowed_sequence_features,
    build_tv3_physics_feature_cache,
    d0_raw_dsp_feature_config,
    load_cached_split_feature_matrix,
)
from tv3.sim.core.tunnel_ventilation_schema import COMPONENT_FIELDS
from tv3.sim.packaging.io import write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_SPLITS = ("val", "test", "extrapolation")
SCHEMA_VERSION = "tv3-ec-msw-e1d-1"

E1_AGGREGATION_STATS = ("last", "mean", "max")
B1_SEVEN_STATS = ("mean", "std", "min", "max", "last", "delta", "slope")
B1_FULL_STATS = DEFAULT_ROCKET_SEQUENCE_STATISTICS

PEAK_ARRAY = ("ultrasonic_peak_index_raw_dsp",)
SOUND_SPEED_ARRAY = ("ultrasonic_sound_speed_raw_dsp_m_per_s",)
TOF_CORRECTED_ARRAY = ("ultrasonic_tof_corrected_raw_dsp_s",)
CORR_PEAK_ARRAY = ("ultrasonic_corr_peak",)
PSR_ARRAY = ("ultrasonic_peak_to_sidelobe_ratio",)
SNR_ARRAY = ("ultrasonic_snr_db",)
PEAK_WIDTH_ARRAY = ("ultrasonic_peak_width_samples",)
QUALITY_ARRAYS = ("ultrasonic_raw_dsp_quality", "ultrasonic_raw_dsp_accepted")
DELAY_SCALAR = ("ultrasonic_delay_calibration_s",)
TOF_L_SCALARS = (
    "ultrasonic_tof_l_m_intercept_s",
    "ultrasonic_sound_speed_slope_raw_dsp_m_per_s",
)

DEFAULT_NARROW_O2_WINDOWS = (
    {"id": "o2_18.0_18.8", "low_percent": 18.0, "high_percent": 18.8},
    {"id": "o2_18.8_19.6", "low_percent": 18.8, "high_percent": 19.6},
    {"id": "o2_19.6_20.4", "low_percent": 19.6, "high_percent": 20.4},
    {"id": "o2_20.4_21.2", "low_percent": 20.4, "high_percent": 21.2},
)

DEFAULT_PARITY_GATES = {
    "o2_r2_drop_max": 0.05,
    "co2_n2_r2_drop_max": 0.03,
}
DEFAULT_POSITIVE_CONTROL_R2_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class E1dFeatureSpec:
    """One pre-registered ablation set."""

    name: str
    stage: str
    role: str
    include_slow: bool = True
    slow_channels: tuple[str, ...] = FORMAL_SLOW_CHANNELS
    frame_arrays: tuple[str, ...] = ()
    sequence_scalars: tuple[str, ...] = ()
    sequence_statistics: tuple[str, ...] = B1_SEVEN_STATS
    phase_windows: tuple[str, ...] = ()
    early_fractions: tuple[float, ...] = ()
    is_full_b1: bool = False
    representation_source: str = "raw_dsp"


def default_e1d_specs() -> tuple[E1dFeatureSpec, ...]:
    """Pre-registered ladder: E1d-0 control, then aggregation, calibration, quality."""
    full_b1 = E1dFeatureSpec(
        name="full_b1",
        stage="E1d-0",
        role="positive_control_not_end_to_end_claim",
        frame_arrays=RAW_DSP_PHYSICS_ARRAYS,
        sequence_statistics=B1_FULL_STATS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
        is_full_b1=True,
    )
    e1r_sequence = E1dFeatureSpec(
        name="e1r_sequence_embedding",
        stage="E1d-1",
        role="actual_frozen_e1r_sequence_embedding",
        include_slow=False,
        representation_source="e1r_sequence",
    )
    e1r_peak_lmm = E1dFeatureSpec(
        name="e1r_peak_lmm",
        stage="E1d-1",
        role="actual_e1r_matched_peak_last_mean_max",
        sequence_statistics=E1_AGGREGATION_STATS,
        representation_source="e1r_peak",
    )
    e1r_peak_b1_windows = E1dFeatureSpec(
        name="e1r_peak_b1_windows",
        stage="E1d-1",
        role="actual_e1r_matched_peak_full_b1_windows",
        sequence_statistics=B1_FULL_STATS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
        representation_source="e1r_peak",
    )
    peak_lmm = E1dFeatureSpec(
        name="peak_lmm",
        stage="E1d-1",
        role="e1r_like_last_mean_max_on_peak_coordinate",
        frame_arrays=PEAK_ARRAY,
        sequence_statistics=E1_AGGREGATION_STATS,
    )
    peak_stats7 = E1dFeatureSpec(
        name="peak_stats7",
        stage="E1d-1",
        role="b1_seven_stats_full_sequence_on_peak",
        frame_arrays=PEAK_ARRAY,
        sequence_statistics=B1_SEVEN_STATS,
    )
    peak_stats7_phase = E1dFeatureSpec(
        name="peak_stats7_phase",
        stage="E1d-1",
        role="seven_stats_plus_four_phase_windows_on_peak",
        frame_arrays=PEAK_ARRAY,
        sequence_statistics=B1_SEVEN_STATS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
    )
    peak_b1_windows = E1dFeatureSpec(
        name="peak_b1_windows",
        stage="E1d-1",
        role="peak_only_with_full_b1_stats_phase_early",
        frame_arrays=PEAK_ARRAY,
        sequence_statistics=B1_FULL_STATS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
    )
    # Cumulative ladder from the richest peak-only set in E1d-1.
    base_peak = peak_stats7_phase
    cal_delay = replace(
        base_peak,
        name="peak_phase_plus_delay",
        stage="E1d-2",
        role="add_sequence_delay_calibration",
        sequence_scalars=DELAY_SCALAR,
    )
    cal_tof_corr = replace(
        cal_delay,
        name="peak_phase_plus_delay_tof_corr",
        stage="E1d-2",
        role="add_corrected_tof_frames",
        frame_arrays=PEAK_ARRAY + TOF_CORRECTED_ARRAY,
    )
    cal_tof_l = replace(
        cal_tof_corr,
        name="peak_phase_plus_delay_tof_corr_tofl",
        stage="E1d-2",
        role="add_tof_l_intercept_slope",
        sequence_scalars=DELAY_SCALAR + TOF_L_SCALARS,
    )
    cal_sound = replace(
        cal_tof_l,
        name="peak_phase_plus_calibration_sound",
        stage="E1d-2",
        role="add_estimated_sound_speed_frames",
        frame_arrays=PEAK_ARRAY + TOF_CORRECTED_ARRAY + SOUND_SPEED_ARRAY,
    )
    q_corr = replace(
        cal_sound,
        name="cal_plus_corr_peak",
        stage="E1d-3",
        role="add_correlation_peak",
        frame_arrays=cal_sound.frame_arrays + CORR_PEAK_ARRAY,
    )
    q_psr = replace(
        q_corr,
        name="cal_plus_corr_psr",
        stage="E1d-3",
        role="add_peak_to_sidelobe_ratio",
        frame_arrays=q_corr.frame_arrays + PSR_ARRAY,
    )
    q_snr = replace(
        q_psr,
        name="cal_plus_corr_psr_snr",
        stage="E1d-3",
        role="add_snr_db",
        frame_arrays=q_psr.frame_arrays + SNR_ARRAY,
    )
    q_width = replace(
        q_snr,
        name="cal_plus_quality_width",
        stage="E1d-3",
        role="add_peak_width",
        frame_arrays=q_snr.frame_arrays + PEAK_WIDTH_ARRAY,
    )
    q_accept = replace(
        q_width,
        name="cal_plus_quality_full",
        stage="E1d-3",
        role="add_quality_and_accepted",
        frame_arrays=q_width.frame_arrays + QUALITY_ARRAYS,
    )
    # B1-like cumulative frame set without claiming end-to-end improvement.
    b1_minus_tof_obs = E1dFeatureSpec(
        name="b1_arrays_without_tof_observed",
        stage="E1d-3",
        role="full_b1_physics_minus_redundant_tof_observed",
        frame_arrays=tuple(
            name for name in RAW_DSP_PHYSICS_ARRAYS if name != "ultrasonic_tof_observed_raw_dsp_s"
        ),
        sequence_statistics=B1_FULL_STATS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
    )
    return (
        full_b1,
        e1r_sequence,
        e1r_peak_lmm,
        e1r_peak_b1_windows,
        peak_lmm,
        peak_stats7,
        peak_stats7_phase,
        peak_b1_windows,
        cal_delay,
        cal_tof_corr,
        cal_tof_l,
        cal_sound,
        q_corr,
        q_psr,
        q_snr,
        q_width,
        q_accept,
        b1_minus_tof_obs,
    )


def run_ec_msw_e1d_diagnosis(
    config_path: Path | str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_config(config)

    dataset_dir = _resolve(project_root, config["dataset_dir"])
    output_dir = _resolve(project_root, config["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"E1d output already exists: {output_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    raw_dsp_manifest = _validate_raw_dsp_cache(raw_dsp_dir, dataset_dir)
    run_kind = str(config.get("run_kind", "formal"))

    reference_path = config.get("b1_reference_metrics")
    reference: dict[str, Any] | None = None
    if reference_path is not None:
        reference_path = _resolve(project_root, reference_path)
        if not reference_path.is_file():
            raise FileNotFoundError(f"b1_reference_metrics not found: {reference_path}")
        reference = _read_json(reference_path)

    alphas = tuple(float(value) for value in config["ridge_alphas"])
    gates = dict(config.get("parity_gates", DEFAULT_PARITY_GATES))
    positive_control_tolerance = float(
        config.get("positive_control_r2_tolerance", DEFAULT_POSITIVE_CONTROL_R2_TOLERANCE)
    )
    narrow_windows = list(config.get("narrow_o2_windows", DEFAULT_NARROW_O2_WINDOWS))
    requested = config.get("feature_sets")
    specs = _select_specs(requested)
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    e1r_specs = tuple(spec for spec in specs if spec.representation_source.startswith("e1r_"))
    e1r_matrices: dict[str, dict[str, MLFeatureMatrix]] = {}
    e1r_provenance: dict[str, Any] = {}
    if e1r_specs:
        training_run_dir = _resolve(project_root, config["training_run_dir"])
        e1r_matrices, e1r_provenance = _build_e1r_spec_matrices(
            dataset_dir,
            e1r_specs,
            training_run_dir=training_run_dir,
            project_root=project_root,
            device=str(config["device"]),
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            expected_template_digest=str(raw_dsp_manifest["template_digest"]),
        )

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "dataset_dir": str(dataset_dir),
        "raw_dsp_manifest_sha256": _sha256(raw_dsp_dir / "manifest.json"),
        "raw_dsp_build_signature": raw_dsp_manifest.get("build_signature"),
        "raw_dsp_template_digest": raw_dsp_manifest.get("template_digest"),
        "b1_reference_metrics": None if reference_path is None else str(reference_path),
        "b1_reference_metrics_sha256": None if reference_path is None else _sha256(reference_path),
        "run_kind": run_kind,
        **e1r_provenance,
        "e2_allowed": False,
        "notes": [
            "full_b1 is a positive control only; never claim end-to-end improvement from full RawDSP",
            "no new deep network is trained in E1d",
            "oracle TOF/true sound speed/true alpha/labels are not model inputs",
        ],
    }

    feature_set_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, object]] = []
    narrow_rows: list[dict[str, object]] = []
    full_b1_split_metrics: dict[str, dict[str, Any]] | None = None

    for spec in specs:
        matrices = (
            e1r_matrices[spec.name]
            if spec.representation_source.startswith("e1r_")
            else _build_spec_matrices(dataset_dir, spec)
        )
        train = matrices["train"]
        probe = ScaledRidgeCVRegressor(alphas=alphas).fit(train.x, train.y)
        split_payload: dict[str, Any] = {}
        for split in ("train", *eval_splits):
            matrix = matrices[split]
            y_pred = probe.predict(matrix.x)
            metrics = _composition_metrics(y_pred, matrix.y)
            gate = None
            if reference is not None and split in eval_splits:
                reference_metrics = _reference_component_metrics(reference, split)
                gate = (
                    _positive_control_split_gate(
                        metrics["component_metrics"],
                        reference_metrics,
                        positive_control_tolerance,
                    )
                    if spec.is_full_b1
                    else _parity_split_gate(metrics["component_metrics"], reference_metrics, gates)
                )
                metrics = {
                    **metrics,
                    "reference_component_metrics": reference_metrics,
                    "gate": gate,
                }
            elif full_b1_split_metrics is not None and split in eval_splits:
                control = full_b1_split_metrics[split]["component_metrics"]
                gate = _parity_split_gate(metrics["component_metrics"], control, gates)
                metrics = {
                    **metrics,
                    "control_component_metrics": control,
                    "gate": gate,
                    "control_source": "full_b1_same_run",
                }
            split_payload[split] = metrics
            if split in eval_splits:
                ablation_rows.append(
                    _ablation_row(
                        spec,
                        split,
                        metrics,
                        probe.selected_alpha,
                        feature_count=len(train.feature_names),
                    )
                )
                narrow_rows.extend(
                    _narrow_window_rows(spec.name, split, y_pred, matrix.y, narrow_windows)
                )
        if spec.is_full_b1:
            full_b1_split_metrics = {
                split: split_payload[split] for split in eval_splits if split in split_payload
            }
        feature_set_rows.append(
            {
                "name": spec.name,
                "stage": spec.stage,
                "role": spec.role,
                "is_full_b1": spec.is_full_b1,
                "feature_count": len(train.feature_names),
                "diagnostic_feature_count": _diagnostic_feature_count(train.feature_names),
                "feature_names": list(train.feature_names),
                "frame_arrays": list(spec.frame_arrays),
                "sequence_scalars": list(spec.sequence_scalars),
                "sequence_statistics": list(spec.sequence_statistics),
                "phase_windows": list(spec.phase_windows),
                "early_fractions": list(spec.early_fractions),
                "include_slow": spec.include_slow,
                "representation_source": spec.representation_source,
                "selected_alpha": probe.selected_alpha,
                "splits": split_payload,
            }
        )

    summary = _build_summary(
        feature_set_rows,
        gates,
        reference is not None,
        positive_control_r2_tolerance=positive_control_tolerance,
    )
    verdict = _build_verdict(summary, gates, run_kind=run_kind)

    output_dir.mkdir(parents=True)
    write_json(output_dir / "manifest.json", provenance | {"verdict": verdict["status"]})
    write_json(output_dir / "feature_sets.json", {"feature_sets": feature_set_rows})
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "verdict.json", verdict)
    write_csv(
        output_dir / "ablation_table.csv",
        (
            "stage",
            "feature_set",
            "role",
            "split",
            "feature_count",
            "selected_alpha",
            "x_CO2_r2",
            "x_O2_r2",
            "x_N2_r2",
            "x_CO2_mae",
            "x_O2_mae",
            "x_N2_mae",
            "x_CO2_bias",
            "x_O2_bias",
            "x_N2_bias",
            "sum_abs_error",
            "delta_o2_r2_vs_control",
            "delta_co2_r2_vs_control",
            "delta_n2_r2_vs_control",
            "parity_passed",
        ),
        ablation_rows,
    )
    write_csv(
        output_dir / "narrow_o2_windows.csv",
        (
            "feature_set",
            "split",
            "window_id",
            "low_percent",
            "high_percent",
            "count",
            "mae_percent",
            "rmse_percent",
            "p90_abs_error_percent",
            "bias_percent",
            "local_slope",
        ),
        narrow_rows,
    )
    return output_dir


def build_e1d_feature_matrix(
    dataset_dir: Path | str,
    *,
    split: str,
    spec: E1dFeatureSpec,
) -> MLFeatureMatrix:
    dataset_dir = Path(dataset_dir)
    if spec.representation_source != "raw_dsp":
        raise ValueError(
            f"spec {spec.name!r} requires a frozen E1r run and must be built through "
            "run_ec_msw_e1d_diagnosis"
        )
    if spec.is_full_b1:
        return _build_full_b1_matrix(dataset_dir, split=split)

    splits = load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_indices = resolve_split_indices(splits, master_sequence_ids)[split]
    sequence_ids = tuple(master_sequence_ids[index] for index in split_indices)
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)[split_indices]
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))
    phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv")
    slow_names = tuple(_load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy"))
    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if spec.include_slow:
        slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[split_indices].astype(
            np.float32
        )
        channel_names = slow_names
        if spec.slow_channels is not None:
            slow, channel_names = _select_slow_channels(slow, slow_names, spec.slow_channels)
        # Slow always uses the frozen B1 window contract so ultrasonic ablations are isolated.
        block, names = _windowed_sequence_features(
            slow,
            sequence_ids=sequence_ids,
            channel_names=channel_names,
            phase_lookup=phase_lookup,
            statistics=B1_FULL_STATS,
            source_prefix="slow",
            phase_windows=DEFAULT_PHASE_WINDOWS,
            early_fractions=DEFAULT_EARLY_FRACTIONS,
        )
        blocks.append(block)
        feature_names.extend(names)

    for array_name in spec.frame_arrays:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP frame array: {path}")
        values = np.load(path, mmap_mode="r")[split_indices]
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"{array_name} must be (N, T), got {values.shape}")
        block, names = _windowed_sequence_features(
            values[..., np.newaxis],
            sequence_ids=sequence_ids,
            channel_names=(array_name,),
            phase_lookup=phase_lookup,
            statistics=spec.sequence_statistics,
            source_prefix="physics",
            phase_windows=spec.phase_windows,
            early_fractions=spec.early_fractions,
        )
        blocks.append(block)
        feature_names.extend(names)

    for array_name in spec.sequence_scalars:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP sequence scalar: {path}")
        values = np.load(path)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"{array_name} must be (N,), got {values.shape}")
        selected = values[split_indices].reshape(-1, 1)
        if not np.isfinite(selected).all():
            raise ValueError(f"non-finite sequence scalar values in {array_name}")
        blocks.append(selected)
        feature_names.append(f"seq|{array_name}")

    if not blocks:
        raise ValueError(f"feature spec {spec.name!r} produced no features")
    x = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite features for spec {spec.name!r} split {split!r}")
    return MLFeatureMatrix(
        x=x,
        y=labels,
        feature_names=tuple(feature_names),
        label_names=label_names,
        sequence_ids=sequence_ids,
    )


def _build_e1r_spec_matrices(
    dataset_dir: Path,
    specs: Sequence[E1dFeatureSpec],
    *,
    training_run_dir: Path,
    project_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    expected_template_digest: str,
) -> tuple[dict[str, dict[str, MLFeatureMatrix]], dict[str, Any]]:
    run_config_path = training_run_dir / "run_config.json"
    checkpoint_path = training_run_dir / "checkpoint.pt"
    for path in (run_config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen E1r input not found: {path}")
    run_config = _read_json(run_config_path)
    model_config = run_config.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("E1r run_config missing model_config")
    template_digest = model_config.get("peak_coordinate_template_digest")
    if template_digest != expected_template_digest:
        raise ValueError(
            "E1r template digest does not match the validated RawDSP cache: "
            f"{template_digest!r} != {expected_template_digest!r}"
        )

    torch_device = torch.device(device)
    model = _load_e1_model(run_config, checkpoint_path, device=device)
    input_preprocess = _build_e1_input_preprocess(run_config)
    matrices_by_spec: dict[str, dict[str, MLFeatureMatrix]] = {spec.name: {} for spec in specs}

    for split in ("train", *EVAL_SPLITS):
        dataset = _build_e1_dataset(dataset_dir, split, run_config, project_root=project_root)
        slow_matrix = _build_slow_control_matrix(dataset_dir, split=split)
        sequence_blocks: list[np.ndarray] = []
        coordinate_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        preprocess = None if input_preprocess is None else input_preprocess.to(torch_device)
        with torch.inference_mode():
            for xb, yb in _e1_loader(dataset, batch_size, num_workers, torch_device):
                x = _e1_batch_x(xb, device=torch_device, input_preprocess=preprocess)
                frame_embeddings = model.encode_frames(x)
                if not model.has_peak_coordinate:
                    raise ValueError("E1d requires an E1r checkpoint with a frozen peak coordinate")
                sequence_blocks.append(
                    model.encode_sequence(x, frame_embeddings=frame_embeddings).cpu().numpy()
                )
                coordinate_blocks.append(
                    (
                        frame_embeddings[:, :, :1]
                        * float(model.waveform_encoder.waveform_length - 1)
                    )
                    .cpu()
                    .numpy()
                )
                label_blocks.append(yb.numpy())

        sequence_x = np.concatenate(sequence_blocks).astype(np.float32, copy=False)
        peak_coordinate = np.concatenate(coordinate_blocks).astype(np.float32, copy=False)
        labels = np.concatenate(label_blocks).astype(np.float32, copy=False)
        if not np.array_equal(labels, slow_matrix.y):
            raise ValueError(f"E1r dataset label ordering differs from frozen split for {split}")

        phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv")
        for spec in specs:
            if spec.representation_source == "e1r_sequence":
                x = sequence_x
                names = tuple(f"e1r|sequence_embedding:{index}" for index in range(x.shape[1]))
            elif spec.representation_source == "e1r_peak":
                peak_block, peak_names = _windowed_sequence_features(
                    peak_coordinate,
                    sequence_ids=slow_matrix.sequence_ids,
                    channel_names=("matched_filter_peak_coordinate_samples",),
                    phase_lookup=phase_lookup,
                    statistics=spec.sequence_statistics,
                    source_prefix="e1r",
                    phase_windows=spec.phase_windows,
                    early_fractions=spec.early_fractions,
                )
                x = np.concatenate((slow_matrix.x, peak_block), axis=1).astype(
                    np.float32, copy=False
                )
                names = slow_matrix.feature_names + peak_names
            else:
                raise ValueError(f"unsupported E1r representation source: {spec.representation_source}")
            matrices_by_spec[spec.name][split] = MLFeatureMatrix(
                x=x,
                y=labels,
                feature_names=names,
                label_names=slow_matrix.label_names,
                sequence_ids=slow_matrix.sequence_ids,
            )

    return matrices_by_spec, {
        "e1r_training_run_dir": str(training_run_dir),
        "e1r_checkpoint_path": str(checkpoint_path),
        "e1r_checkpoint_sha256": _sha256(checkpoint_path),
        "e1r_run_config_sha256": _sha256(run_config_path),
        "e1r_template_digest": str(template_digest),
    }


def _build_slow_control_matrix(dataset_dir: Path, *, split: str) -> MLFeatureMatrix:
    return build_e1d_feature_matrix(
        dataset_dir,
        split=split,
        spec=E1dFeatureSpec(
            name="_frozen_slow_control",
            stage="internal",
            role="frozen_b1_slow_contract",
        ),
    )


def _build_spec_matrices(
    dataset_dir: Path,
    spec: E1dFeatureSpec,
) -> dict[str, MLFeatureMatrix]:
    if spec.is_full_b1:
        cache_dir = dataset_dir / "features" / "rocket" / f"e1d_{spec.name}"
        build_tv3_physics_feature_cache(
            dataset_dir,
            cache_dir=cache_dir,
            config=d0_raw_dsp_feature_config(),
        )
        return {
            split: load_cached_split_feature_matrix(dataset_dir, cache_dir, split=split)
            for split in ("train", *EVAL_SPLITS)
        }
    return {
        split: build_e1d_feature_matrix(dataset_dir, split=split, spec=spec)
        for split in ("train", *EVAL_SPLITS)
    }


def _build_full_b1_matrix(dataset_dir: Path, *, split: str) -> MLFeatureMatrix:
    cache_dir = dataset_dir / "features" / "rocket" / "e1d_full_b1"
    if not _full_b1_cache_matches(dataset_dir, cache_dir):
        build_tv3_physics_feature_cache(
            dataset_dir,
            cache_dir=cache_dir,
            config=d0_raw_dsp_feature_config(),
        )
    return load_cached_split_feature_matrix(dataset_dir, cache_dir, split=split)


def _full_b1_cache_matches(dataset_dir: Path, cache_dir: Path) -> bool:
    manifest_path = cache_dir / "feature_manifest.json"
    feature_names_path = cache_dir / "feature_names.json"
    if not manifest_path.is_file() or not feature_names_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    dataset_manifest = _read_json(dataset_dir / "manifest.json")
    expected = d0_raw_dsp_feature_config()
    return (
        manifest.get("dataset_slug") == dataset_manifest.get("dataset_slug")
        and manifest.get("feature_builder") == expected.feature_builder
        and tuple(manifest.get("slow_channels") or ()) == tuple(expected.slow_channels or ())
        and tuple(manifest.get("source_arrays") or ()) == tuple(expected.physics_arrays)
        and tuple(manifest.get("pooling_stats") or ()) == tuple(expected.sequence_statistics)
        and tuple(manifest.get("phase_windows") or ()) == tuple(expected.phase_windows)
        and tuple(float(value) for value in manifest.get("early_fractions") or ())
        == tuple(expected.early_fractions)
    )


def _select_specs(requested: object) -> tuple[E1dFeatureSpec, ...]:
    defaults = default_e1d_specs()
    if requested is None:
        return defaults
    if not isinstance(requested, list) or not requested:
        raise ValueError("feature_sets must be a non-empty list when provided")
    by_name = {spec.name: spec for spec in defaults}
    selected: list[E1dFeatureSpec] = []
    for name in requested:
        if name not in by_name:
            raise ValueError(f"unknown E1d feature set {name!r}; known={sorted(by_name)}")
        selected.append(by_name[str(name)])
    # The positive control must run first so same-run fallback gates are deterministic.
    selected = [spec for spec in selected if not spec.is_full_b1]
    selected.insert(0, by_name["full_b1"])
    return tuple(selected)


def _validate_raw_dsp_cache(raw_dsp_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    manifest_path = raw_dsp_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing RawDSP frame cache manifest: {manifest_path}. "
            "Build train_baseline_median cache before E1d."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != RAW_DSP_FRAME_SCHEMA_VERSION:
        raise ValueError(
            f"RawDSP frame schema mismatch: {manifest.get('schema_version')!r} "
            f"!= {RAW_DSP_FRAME_SCHEMA_VERSION!r}"
        )
    if manifest.get("diagnostic_only") is not False:
        raise ValueError("E1d requires train_baseline_median RawDSP cache, not diagnostic exact template")
    if manifest.get("complete_dataset") is not True:
        raise ValueError("E1d requires a complete-dataset RawDSP cache")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    cached_sequence_ids = _load_str_array(raw_dsp_dir / "sequence_ids.npy")
    if cached_sequence_ids != master_sequence_ids:
        raise ValueError("RawDSP cache sequence_ids do not match dataset metadata ordering")
    required = set(RAW_DSP_PHYSICS_ARRAYS) | set(DELAY_SCALAR) | set(TOF_L_SCALARS) | set(
        TOF_CORRECTED_ARRAY + PSR_ARRAY + PEAK_WIDTH_ARRAY
    )
    missing = [name for name in sorted(required) if not (raw_dsp_dir / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(
            "RawDSP cache is incomplete for E1d; missing arrays: "
            + ", ".join(missing)
            + f" under {raw_dsp_dir}"
        )
    return manifest


def _composition_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    components = component_regression_metrics(y_pred, y_true, COMPONENT_FIELDS)
    component_payload: dict[str, Any] = {}
    for index, name in enumerate(COMPONENT_FIELDS):
        errors = np.asarray(y_pred[:, index] - y_true[:, index], dtype=np.float64)
        component_payload[name] = {
            **asdict(components[name]),
            "bias": float(np.mean(errors)),
            "p90_abs_error": float(np.percentile(np.abs(errors), 90.0)),
        }
    return {
        "sequence_count": int(len(y_true)),
        "metrics": asdict(regression_metrics(y_pred, y_true)),
        "component_metrics": component_payload,
        "sum_abs_error": float(np.mean(np.abs(np.sum(y_pred, axis=1) - 100.0))),
    }


def _reference_component_metrics(reference: Mapping[str, Any], split: str) -> dict[str, dict[str, float]]:
    evaluations = reference.get("evaluations")
    if not isinstance(evaluations, dict) or split not in evaluations:
        raise ValueError(f"B1 reference metrics missing evaluations.{split}")
    component_metrics = evaluations[split].get("component_metrics")
    if not isinstance(component_metrics, dict):
        raise ValueError(f"B1 reference metrics missing component_metrics for {split}")
    payload: dict[str, dict[str, float]] = {}
    for name in COMPONENT_FIELDS:
        entry = component_metrics.get(name)
        if not isinstance(entry, dict) or "r2" not in entry:
            raise ValueError(f"B1 reference missing {split}.{name}.r2")
        payload[name] = {
            "mae": float(entry.get("mae", float("nan"))),
            "rmse": float(entry.get("rmse", float("nan"))),
            "r2": float(entry["r2"]),
        }
    return payload


def _parity_split_gate(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
    gates: Mapping[str, float],
) -> dict[str, Any]:
    o2_drop_max = float(gates["o2_r2_drop_max"])
    co2_n2_drop_max = float(gates["co2_n2_r2_drop_max"])
    deltas = {
        name: float(candidate[name]["r2"]) - float(reference[name]["r2"])
        for name in COMPONENT_FIELDS
    }
    checks = {
        "x_O2_r2_noninferiority": deltas["x_O2"] >= -o2_drop_max,
        "x_CO2_r2_noninferiority": deltas["x_CO2"] >= -co2_n2_drop_max,
        "x_N2_r2_noninferiority": deltas["x_N2"] >= -co2_n2_drop_max,
    }
    return {
        "passed": all(checks.values()),
        "r2_delta_vs_control": deltas,
        "checks": checks,
    }


def _positive_control_split_gate(
    candidate: Mapping[str, Mapping[str, float]],
    reference: Mapping[str, Mapping[str, float]],
    tolerance: float,
) -> dict[str, Any]:
    deltas = {
        name: float(candidate[name]["r2"]) - float(reference[name]["r2"])
        for name in COMPONENT_FIELDS
    }
    checks = {name: abs(delta) <= tolerance for name, delta in deltas.items()}
    return {
        "passed": all(checks.values()),
        "r2_delta_vs_control": deltas,
        "checks": checks,
        "absolute_r2_tolerance": tolerance,
    }


def _ablation_row(
    spec: E1dFeatureSpec,
    split: str,
    metrics: Mapping[str, Any],
    selected_alpha: float,
    *,
    feature_count: int,
) -> dict[str, object]:
    components = metrics["component_metrics"]
    gate = metrics.get("gate") or {}
    deltas = gate.get("r2_delta_vs_control") or {}
    return {
        "stage": spec.stage,
        "feature_set": spec.name,
        "role": spec.role,
        "split": split,
        "feature_count": feature_count,
        "selected_alpha": selected_alpha,
        "x_CO2_r2": components["x_CO2"]["r2"],
        "x_O2_r2": components["x_O2"]["r2"],
        "x_N2_r2": components["x_N2"]["r2"],
        "x_CO2_mae": components["x_CO2"]["mae"],
        "x_O2_mae": components["x_O2"]["mae"],
        "x_N2_mae": components["x_N2"]["mae"],
        "x_CO2_bias": components["x_CO2"]["bias"],
        "x_O2_bias": components["x_O2"]["bias"],
        "x_N2_bias": components["x_N2"]["bias"],
        "sum_abs_error": metrics["sum_abs_error"],
        "delta_o2_r2_vs_control": deltas.get("x_O2"),
        "delta_co2_r2_vs_control": deltas.get("x_CO2"),
        "delta_n2_r2_vs_control": deltas.get("x_N2"),
        "parity_passed": gate.get("passed"),
    }


def _narrow_window_rows(
    feature_set: str,
    split: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    windows: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    o2_true = np.asarray(y_true[:, 1], dtype=np.float64)
    o2_pred = np.asarray(y_pred[:, 1], dtype=np.float64)
    for index, window in enumerate(windows):
        low = float(window["low_percent"])
        high = float(window["high_percent"])
        mask = (o2_true >= low) & (o2_true <= high if index == len(windows) - 1 else o2_true < high)
        true_values = o2_true[mask]
        pred_values = o2_pred[mask]
        errors = pred_values - true_values
        count = int(np.count_nonzero(mask))
        row: dict[str, object] = {
            "feature_set": feature_set,
            "split": split,
            "window_id": str(window["id"]),
            "low_percent": low,
            "high_percent": high,
            "count": count,
            "mae_percent": None,
            "rmse_percent": None,
            "p90_abs_error_percent": None,
            "bias_percent": None,
            "local_slope": None,
        }
        if count > 0:
            row["mae_percent"] = float(np.mean(np.abs(errors)))
            row["rmse_percent"] = float(np.sqrt(np.mean(errors * errors)))
            row["p90_abs_error_percent"] = float(np.percentile(np.abs(errors), 90.0))
            row["bias_percent"] = float(np.mean(errors))
            if count >= 2 and float(np.std(true_values)) > 0.0:
                slope = float(np.polyfit(true_values, pred_values, 1)[0])
                row["local_slope"] = slope
        rows.append(row)
    return rows


def _build_summary(
    feature_set_rows: Sequence[Mapping[str, Any]],
    gates: Mapping[str, float],
    has_external_reference: bool,
    *,
    positive_control_r2_tolerance: float = DEFAULT_POSITIVE_CONTROL_R2_TOLERANCE,
) -> dict[str, Any]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for entry in feature_set_rows:
        missing_splits = [split for split in EVAL_SPLITS if split not in entry["splits"]]
        if missing_splits:
            raise ValueError(
                f"feature set {entry['name']!r} is missing required eval splits: {missing_splits}"
            )
        stage = str(entry["stage"])
        by_stage.setdefault(stage, []).append(
            {
                "name": entry["name"],
                "feature_count": entry["feature_count"],
                "diagnostic_feature_count": entry["diagnostic_feature_count"],
                "role": entry["role"],
                "is_full_b1": entry["is_full_b1"],
                "eval": {
                    split: {
                        "x_O2_r2": entry["splits"][split]["component_metrics"]["x_O2"]["r2"],
                        "x_CO2_r2": entry["splits"][split]["component_metrics"]["x_CO2"]["r2"],
                        "x_N2_r2": entry["splits"][split]["component_metrics"]["x_N2"]["r2"],
                        "parity_passed": (entry["splits"][split].get("gate") or {}).get("passed"),
                        "delta_vs_control": (entry["splits"][split].get("gate") or {}).get(
                            "r2_delta_vs_control"
                        ),
                    }
                    for split in EVAL_SPLITS
                },
            }
        )

    candidates = [
        entry
        for entry in feature_set_rows
        if not entry["is_full_b1"]
        and all(
            (entry["splits"][split].get("gate") or {}).get("passed") is True
            for split in EVAL_SPLITS
        )
    ]
    full_b1 = next((entry for entry in feature_set_rows if entry["is_full_b1"]), None)
    full_count = None if full_b1 is None else int(full_b1["feature_count"])
    full_diagnostic_count = (
        None if full_b1 is None else int(full_b1["diagnostic_feature_count"])
    )
    positive_control_passed = None
    if has_external_reference:
        positive_control_passed = bool(
            full_b1 is not None
            and all(
                (full_b1["splits"][split].get("gate") or {}).get("passed") is True
                for split in EVAL_SPLITS
            )
        )
    compact = []
    for entry in candidates:
        if full_diagnostic_count is None:
            continue
        # Slow is frozen across RawDSP ablations, so compactness concerns the diagnostic block.
        if int(entry["diagnostic_feature_count"]) <= full_diagnostic_count // 2:
            compact.append(entry["name"])

    return {
        "gates": dict(gates),
        "control_source": "b1_reference_metrics" if has_external_reference else "full_b1_same_run",
        "stages": by_stage,
        "parity_passing_sets": [entry["name"] for entry in candidates],
        "compact_parity_passing_sets": compact,
        "full_b1_feature_count": full_count,
        "full_b1_diagnostic_feature_count": full_diagnostic_count,
        "positive_control_passed": positive_control_passed,
        "positive_control_r2_tolerance": positive_control_r2_tolerance,
    }


def _build_verdict(
    summary: Mapping[str, Any],
    gates: Mapping[str, float],
    *,
    run_kind: str = "formal",
) -> dict[str, Any]:
    compact = list(summary.get("compact_parity_passing_sets") or [])
    passing = list(summary.get("parity_passing_sets") or [])
    if run_kind == "smoke":
        status = "smoke_only"
        reason = "Smoke run only verifies the E1d pipeline and cannot authorize a formal conclusion."
        continue_allowed = False
    elif summary.get("positive_control_passed") is not True:
        status = "positive_control_failed"
        reason = (
            "The same-run full B1 positive control did not reproduce frozen B1 non-inferiority "
            "on all eval splits. Stop interpretation and repair cache/protocol provenance."
        )
        continue_allowed = False
    elif compact:
        status = "minimal_deployable_set_found"
        reason = (
            "At least one feature set clearly smaller than full B1 passes O2/CO2/N2 "
            "non-inferiority on all eval splits. A new structured sequence builder may "
            "be implemented and re-audited; E2 remains forbidden until that builder passes."
        )
        continue_allowed = True
    elif passing:
        status = "only_near_full_b1_passes"
        reason = (
            "Only feature sets close to full B1 pass parity. Stop the EC-MSW learned "
            "encoder branch; keep B7; do not package full RawDSP as end-to-end progress."
        )
        continue_allowed = False
    else:
        status = "no_parity_set"
        reason = (
            "No ablation set recovered B1 non-inferiority on all three splits under "
            f"gates={dict(gates)}. Stop learned-encoder expansion; keep B7; E2 forbidden."
        )
        continue_allowed = False
    return {
        "status": status,
        "reason": reason,
        "e2_allowed": False,
        "continue_structured_builder": continue_allowed,
        "compact_parity_passing_sets": compact,
        "parity_passing_sets": passing,
    }


def _diagnostic_feature_count(feature_names: Sequence[str]) -> int:
    return sum("|slow:" not in name for name in feature_names)


def _validate_config(config: Mapping[str, Any]) -> None:
    required = ("dataset_dir", "output_dir", "ridge_alphas")
    for key in required:
        if key not in config:
            raise ValueError(f"E1d config missing required key: {key}")
    alphas = config["ridge_alphas"]
    if (
        not isinstance(alphas, list)
        or not alphas
        or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in alphas)
    ):
        raise ValueError("ridge_alphas must contain finite positive values")

    run_kind = str(config.get("run_kind", "formal"))
    if run_kind not in {"formal", "smoke"}:
        raise ValueError("run_kind must be 'formal' or 'smoke'")
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")

    gates = config.get("parity_gates", DEFAULT_PARITY_GATES)
    if not isinstance(gates, dict) or set(gates) != set(DEFAULT_PARITY_GATES):
        raise ValueError(f"parity_gates must contain exactly {sorted(DEFAULT_PARITY_GATES)}")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in gates.values()):
        raise ValueError("parity_gates values must be finite and non-negative")

    requested = config.get("feature_sets")
    if isinstance(requested, list) and len(set(requested)) != len(requested):
        raise ValueError("feature_sets must not contain duplicates")
    selected = _select_specs(requested)
    e1r_selected = {spec.name for spec in selected if spec.representation_source.startswith("e1r_")}
    if e1r_selected:
        required_e1r_fields = {"training_run_dir", "device", "batch_size", "num_workers"}
        missing = required_e1r_fields.difference(config)
        if missing:
            raise ValueError(f"E1r feature sets require config fields: {sorted(missing)}")
        if int(config["batch_size"]) < 1 or int(config["num_workers"]) < 0:
            raise ValueError("batch_size must be positive and num_workers must be non-negative")
    if run_kind == "formal":
        if "b1_reference_metrics" not in config:
            raise ValueError("formal E1d requires b1_reference_metrics")
        required_actual = {
            "e1r_sequence_embedding",
            "e1r_peak_lmm",
            "e1r_peak_b1_windows",
        }
        if not required_actual.issubset(e1r_selected):
            raise ValueError(
                "formal E1d must include the actual frozen E1r representation sets: "
                f"{sorted(required_actual)}"
            )
    tolerance = float(
        config.get("positive_control_r2_tolerance", DEFAULT_POSITIVE_CONTROL_R2_TOLERANCE)
    )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("positive_control_r2_tolerance must be finite and non-negative")


def _resolve(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
