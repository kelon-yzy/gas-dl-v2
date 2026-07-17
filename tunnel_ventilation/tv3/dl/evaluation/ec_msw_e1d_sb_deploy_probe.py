"""E1d-SB deployable inference probe: waveform → e1d_sb (no LS) → Ridge → raw3.

Audits deploy wiring. Does not open E2. Does not promote LS. Does not replace B7.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tv3.dl.evaluation.ec_msw_e1d_diagnosis import (
    DEFAULT_NARROW_O2_WINDOWS,
    DEFAULT_PARITY_GATES,
    EVAL_SPLITS,
    PROJECT_ROOT,
    _composition_metrics,
    _narrow_window_rows,
    _parity_split_gate,
    _reference_component_metrics,
    _resolve,
    _sha256,
    _validate_raw_dsp_cache,
)
from tv3.ml.e1d_sb_features import (
    E1DSB_FEATURE_BUILDER,
    E1DSB_SPEC_NAME,
    build_e1d_sb_feature_matrix,
    builder_manifest_payload,
    diagnostic_feature_count,
    e1d_sb_builder_info,
)
from tv3.ml.e1d_sb_inference import (
    fit_e1d_sb_inference,
    predict_with_artifact,
    write_inference_artifact,
)
from tv3.ml.rocket_features import RAW_DSP_FRAME_CACHE_ROOT
from tv3.sim.packaging.io import write_csv, write_json

SCHEMA_VERSION = "tv3-ec-msw-e1d-sb-deploy-probe-1"
FULL_B1_DIAGNOSTIC_FEATURE_COUNT = 504
DEFAULT_FEATURE_ALIGN_ATOL = 1.0e-5
FORBIDDEN_OUTPUT_MARKERS = (
    "e1_s",
    "e1r_s",
    "e1d_s",
    "e1d_sb_s",
    "e1d_sb_ls_",
    "e1r_attach_",
)
ALLOWED_OUTPUT_PREFIXES = ("e1d_sb_deploy_probe",)


def run_ec_msw_e1d_sb_deploy_probe(
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
    _assert_safe_output_dir(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"deploy probe output already exists: {output_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    raw_dsp_manifest = _validate_raw_dsp_cache(raw_dsp_dir, dataset_dir)
    run_kind = str(config.get("run_kind", "formal"))
    feature_source = str(config.get("feature_source", "waveform"))
    if feature_source not in {"raw_dsp_cache", "waveform"}:
        raise ValueError(f"unsupported feature_source: {feature_source!r}")
    align_atol = float(config.get("feature_align_atol", DEFAULT_FEATURE_ALIGN_ATOL))

    e1d_sb_gate = _load_status_gate(
        project_root,
        config,
        "e1d_sb_verdict_path",
        run_kind=run_kind,
        required_for_formal=True,
    )
    attachment_gate = _load_status_gate(
        project_root,
        config,
        "attachment_verdict_path",
        run_kind=run_kind,
        required_for_formal=True,
    )
    ls_gate = _load_status_gate(
        project_root,
        config,
        "ls_verdict_path",
        run_kind=run_kind,
        required_for_formal=False,
    )

    reference_path = config.get("b1_reference_metrics")
    reference: dict[str, Any] | None = None
    if reference_path is not None:
        reference_path = _resolve(project_root, reference_path)
        if not reference_path.is_file():
            raise FileNotFoundError(f"b1_reference_metrics not found: {reference_path}")
        reference = _read_json(reference_path)

    alphas = tuple(float(value) for value in config["ridge_alphas"])
    gates = dict(config.get("parity_gates", DEFAULT_PARITY_GATES))
    narrow_windows = list(config.get("narrow_o2_windows", DEFAULT_NARROW_O2_WINDOWS))
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")

    info = e1d_sb_builder_info()
    matrices = {
        split: build_e1d_sb_feature_matrix(
            dataset_dir,
            split=split,
            feature_source=feature_source,  # type: ignore[arg-type]
        )
        for split in ("train", *eval_splits)
    }
    train = matrices["train"]
    if any("snr_weighted_ls" in name for name in train.feature_names):
        raise ValueError("deploy probe must not include LS ablation features")

    feature_alignment = _audit_feature_alignment(
        dataset_dir,
        feature_source=feature_source,
        eval_splits=eval_splits,
        atol=align_atol,
        primary_matrices=matrices,
    )

    probe, artifact = fit_e1d_sb_inference(train, alphas=alphas)
    # Round-trip check: artifact predict must match live probe on train.
    artifact_train_pred = predict_with_artifact(artifact, train.x)
    live_train_pred = probe.predict(train.x)
    if not np.allclose(artifact_train_pred, live_train_pred, rtol=0.0, atol=1e-5):
        raise ValueError("artifact predict diverges from live ScaledRidgeCVRegressor")

    split_payload: dict[str, Any] = {}
    ablation_rows: list[dict[str, object]] = []
    narrow_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split in ("train", *eval_splits):
        matrix = matrices[split]
        y_pred = predict_with_artifact(artifact, matrix.x)
        metrics = _composition_metrics(y_pred, matrix.y)
        gate = None
        if reference is not None and split in eval_splits:
            reference_metrics = _reference_component_metrics(reference, split)
            gate = _parity_split_gate(metrics["component_metrics"], reference_metrics, gates)
            metrics = {
                **metrics,
                "reference_component_metrics": reference_metrics,
                "gate": gate,
            }
        split_payload[split] = metrics
        if split in eval_splits:
            ablation_rows.append(
                _metrics_row(
                    split,
                    metrics,
                    artifact.selected_alpha,
                    feature_count=len(train.feature_names),
                )
            )
            narrow_rows.extend(
                _narrow_window_rows(E1DSB_SPEC_NAME, split, y_pred, matrix.y, narrow_windows)
            )
        for row_index, sequence_id in enumerate(matrix.sequence_ids):
            prediction_rows.append(
                {
                    "split": split,
                    "sequence_id": sequence_id,
                    "x_CO2_pred": float(y_pred[row_index, 0]),
                    "x_O2_pred": float(y_pred[row_index, 1]),
                    "x_N2_pred": float(y_pred[row_index, 2]),
                    "x_CO2_true": float(matrix.y[row_index, 0]),
                    "x_O2_true": float(matrix.y[row_index, 1]),
                    "x_N2_true": float(matrix.y[row_index, 2]),
                }
            )

    diagnostic_count = diagnostic_feature_count(train.feature_names)
    compact = diagnostic_count <= FULL_B1_DIAGNOSTIC_FEATURE_COUNT // 2
    parity_passed = bool(
        reference is not None
        and all(
            (split_payload[split].get("gate") or {}).get("passed") is True for split in eval_splits
        )
    )
    waveform_align_passed = bool(feature_alignment.get("passed"))
    verdict = _build_verdict(
        run_kind=run_kind,
        e1d_sb_gate=e1d_sb_gate,
        attachment_gate=attachment_gate,
        has_reference=reference is not None,
        parity_passed=parity_passed,
        waveform_align_passed=waveform_align_passed,
        compact=compact,
        diagnostic_feature_count=diagnostic_count,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    write_inference_artifact(output_dir / "inference_artifact.json", artifact)
    np.save(output_dir / "predictions_y_pred.npy", np.asarray(
        [[row["x_CO2_pred"], row["x_O2_pred"], row["x_N2_pred"]] for row in prediction_rows],
        dtype=np.float32,
    ))
    write_csv(
        output_dir / "predictions.csv",
        (
            "split",
            "sequence_id",
            "x_CO2_pred",
            "x_O2_pred",
            "x_N2_pred",
            "x_CO2_true",
            "x_O2_true",
            "x_N2_true",
        ),
        prediction_rows,
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
        "e1d_sb_gate": e1d_sb_gate,
        "attachment_gate": attachment_gate,
        "ls_gate": ls_gate,
        "run_kind": run_kind,
        "feature_source": feature_source,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "feature_align_atol": align_atol,
        "feature_alignment": feature_alignment,
        "ls_promoted": False,
        "default_head_remains": "B7",
        "e2_allowed": False,
        "builder": builder_manifest_payload(),
        "notes": [
            "deployable inference probe: waveform/cache → e1d_sb (no LS) → Ridge → raw3",
            "does not replace B7 as the default RawDSP head",
            "LS remains unpromoted regardless of ls_gate status",
            "E2 FiLM/attention/MoE remain forbidden",
        ],
        "verdict": verdict["status"],
    }
    write_json(output_dir / "manifest.json", provenance)

    feature_set_entry = {
        "name": E1DSB_SPEC_NAME,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "role": "deployable_inference_probe_cal_plus_corr_psr_snr",
        "feature_count": len(train.feature_names),
        "diagnostic_feature_count": diagnostic_count,
        "compact": compact,
        "feature_names": list(train.feature_names),
        "frame_arrays": list(info.frame_arrays),
        "sequence_scalars": list(info.sequence_scalars),
        "selected_alpha": artifact.selected_alpha,
        "splits": split_payload,
    }
    write_json(output_dir / "feature_sets.json", {"feature_sets": [feature_set_entry]})

    summary = {
        "gates": gates,
        "control_source": "b1_reference_metrics" if reference is not None else None,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "feature_source": feature_source,
        "feature_count": len(train.feature_names),
        "diagnostic_feature_count": diagnostic_count,
        "full_b1_diagnostic_feature_count": FULL_B1_DIAGNOSTIC_FEATURE_COUNT,
        "compact": compact,
        "parity_passed": parity_passed,
        "waveform_align_passed": waveform_align_passed,
        "ls_promoted": False,
        "default_head_remains": "B7",
        "e2_allowed": False,
        "eval": {
            split: {
                "x_O2_r2": split_payload[split]["component_metrics"]["x_O2"]["r2"],
                "x_CO2_r2": split_payload[split]["component_metrics"]["x_CO2"]["r2"],
                "x_N2_r2": split_payload[split]["component_metrics"]["x_N2"]["r2"],
                "parity_passed": (split_payload[split].get("gate") or {}).get("passed"),
                "delta_vs_control": (split_payload[split].get("gate") or {}).get(
                    "r2_delta_vs_control"
                ),
            }
            for split in eval_splits
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "verdict.json", verdict)
    write_csv(
        output_dir / "ablation_table.csv",
        (
            "feature_set",
            "feature_builder",
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


def _audit_feature_alignment(
    dataset_dir: Path,
    *,
    feature_source: str,
    eval_splits: tuple[str, ...],
    atol: float,
    primary_matrices: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare waveform vs raw_dsp_cache features on eval splits."""
    if feature_source == "waveform":
        compare_source = "raw_dsp_cache"
        primary_label = "waveform"
    else:
        compare_source = "waveform"
        primary_label = "raw_dsp_cache"

    split_rows: dict[str, Any] = {}
    all_passed = True
    for split in eval_splits:
        primary = primary_matrices[split]
        other = build_e1d_sb_feature_matrix(
            dataset_dir,
            split=split,
            feature_source=compare_source,  # type: ignore[arg-type]
        )
        names_match = primary.feature_names == other.feature_names
        max_abs = None
        values_match = False
        if names_match and primary.x.shape == other.x.shape:
            diff = np.abs(primary.x.astype(np.float64) - other.x.astype(np.float64))
            max_abs = float(np.max(diff)) if diff.size else 0.0
            values_match = bool(np.allclose(primary.x, other.x, rtol=0.0, atol=atol))
        passed = bool(names_match and values_match)
        all_passed = all_passed and passed
        split_rows[split] = {
            "primary": primary_label,
            "compare": compare_source,
            "names_match": names_match,
            "shape_primary": list(primary.x.shape),
            "shape_compare": list(other.x.shape),
            "max_abs_diff": max_abs,
            "atol": atol,
            "passed": passed,
        }
    return {
        "primary_feature_source": feature_source,
        "compare_feature_source": compare_source,
        "atol": atol,
        "passed": all_passed,
        "splits": split_rows,
    }


def _metrics_row(
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
        "feature_set": E1DSB_SPEC_NAME,
        "feature_builder": E1DSB_FEATURE_BUILDER,
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


def _build_verdict(
    *,
    run_kind: str,
    e1d_sb_gate: Mapping[str, Any] | None,
    attachment_gate: Mapping[str, Any] | None,
    has_reference: bool,
    parity_passed: bool,
    waveform_align_passed: bool,
    compact: bool,
    diagnostic_feature_count: int,
) -> dict[str, Any]:
    if run_kind == "smoke":
        status = "smoke_only"
        reason = (
            "Smoke run only verifies the e1d_sb deployable inference probe pipeline; "
            "it cannot authorize formal deploy_probe conclusions."
        )
    elif e1d_sb_gate is None or e1d_sb_gate.get("status") != "parity_passed":
        status = "gate_blocked"
        reason = (
            "Formal deploy probe requires e1d_sb_verdict_path with status=parity_passed; "
            f"got {None if e1d_sb_gate is None else e1d_sb_gate.get('status')!r}."
        )
    elif attachment_gate is None or attachment_gate.get("status") != "attachment_passed":
        status = "gate_blocked"
        reason = (
            "Formal deploy probe requires attachment_verdict_path with status=attachment_passed; "
            f"got {None if attachment_gate is None else attachment_gate.get('status')!r}."
        )
    elif not has_reference:
        status = "missing_b1_reference"
        reason = "Formal deploy probe requires frozen b1_reference_metrics."
    elif not compact:
        status = "not_compact"
        reason = (
            f"diagnostic_feature_count={diagnostic_feature_count} exceeds half of full B1 "
            f"diagnostic block ({FULL_B1_DIAGNOSTIC_FEATURE_COUNT // 2})."
        )
    elif not waveform_align_passed:
        status = "deploy_probe_failed"
        reason = (
            "waveform vs raw_dsp_cache feature alignment failed; "
            "repair extract/builder before claiming deploy wiring."
        )
    elif parity_passed:
        status = "deploy_probe_passed"
        reason = (
            "Waveform path aligned with cache and train-only Ridge passed B1 non-inferiority. "
            "Deploy probe may proceed to optional D2 artifact packaging; "
            "B7 remains the default head; LS stays unpromoted; E2 remains forbidden."
        )
    else:
        status = "deploy_probe_failed"
        reason = (
            "Deploy probe Ridge failed B1 non-inferiority on one or more eval splits. "
            "Do not open E2 or promote LS."
        )
    return {
        "status": status,
        "reason": reason,
        "e2_allowed": False,
        "ls_promoted": False,
        "default_head_remains": "B7",
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "spec_name": E1DSB_SPEC_NAME,
        "compact": compact,
        "diagnostic_feature_count": diagnostic_feature_count,
        "parity_passed": parity_passed,
        "waveform_align_passed": waveform_align_passed,
        "e1d_sb_status": None if e1d_sb_gate is None else e1d_sb_gate.get("status"),
        "attachment_status": None if attachment_gate is None else attachment_gate.get("status"),
    }


def _load_status_gate(
    project_root: Path,
    config: Mapping[str, Any],
    key: str,
    *,
    run_kind: str,
    required_for_formal: bool,
) -> dict[str, Any] | None:
    path = config.get(key)
    if path is None:
        if run_kind == "formal" and required_for_formal:
            raise ValueError(f"formal deploy probe requires {key}")
        return None
    resolved = _resolve(project_root, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{key} not found: {resolved}")
    payload = _read_json(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "status": payload.get("status"),
        "e2_allowed": payload.get("e2_allowed"),
        "feature_builder": payload.get("feature_builder"),
    }


def _assert_safe_output_dir(output_dir: Path) -> None:
    name = output_dir.name
    if any(name.startswith(prefix) for prefix in ALLOWED_OUTPUT_PREFIXES):
        return
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        if name.startswith(marker):
            raise ValueError(
                f"deploy probe output_dir basename {name!r} collides with frozen evidence dirs"
            )


def _validate_config(config: Mapping[str, Any]) -> None:
    required = ("dataset_dir", "output_dir", "ridge_alphas", "feature_builder")
    for key in required:
        if key not in config:
            raise ValueError(f"deploy probe config missing required key: {key}")
    if config["feature_builder"] != E1DSB_FEATURE_BUILDER:
        raise ValueError(
            f"feature_builder must be {E1DSB_FEATURE_BUILDER!r}, got {config['feature_builder']!r}"
        )
    if config.get("ls_promoted") is True:
        raise ValueError("deploy probe config must keep ls_promoted=false")
    if config.get("e2_allowed") is True:
        raise ValueError("deploy probe config must keep e2_allowed=false")
    run_kind = str(config.get("run_kind", "formal"))
    if run_kind not in {"smoke", "formal"}:
        raise ValueError(f"unsupported run_kind: {run_kind!r}")
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")
    if run_kind == "formal" and config.get("b1_reference_metrics") is None:
        raise ValueError("formal deploy probe requires b1_reference_metrics")
    if run_kind == "formal" and config.get("e1d_sb_verdict_path") is None:
        raise ValueError("formal deploy probe requires e1d_sb_verdict_path")
    if run_kind == "formal" and config.get("attachment_verdict_path") is None:
        raise ValueError("formal deploy probe requires attachment_verdict_path")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload
