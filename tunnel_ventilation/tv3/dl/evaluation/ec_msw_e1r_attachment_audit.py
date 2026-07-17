"""E1r↔E1d-SB attachment audit: frozen E1r frame fidelity + e1d_sb sequence Ridge.

Probe-only. Does not retrain a deep net. Does not open E2.
Sequence parity uses e1d_sb_cal_plus_corr_psr_snr_v1 (replaces E1r last/mean/max embedding).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from tv3.dl.evaluation.ec_msw_e1_audit import (
    EVAL_SPLITS,
    PROJECT_ROOT,
    _build_dataset,
    _build_input_preprocess,
    _extract_eval_features,
    _extract_train_features,
    _frame_split_gate,
    _load_model,
    _load_peak_targets,
    _peak_error_metrics,
    _resolve,
    _sample_frame_indices,
    _sha256,
    _timesteps,
)
from tv3.dl.evaluation.ec_msw_e1d_diagnosis import (
    DEFAULT_NARROW_O2_WINDOWS,
    DEFAULT_PARITY_GATES,
    _composition_metrics,
    _narrow_window_rows,
    _parity_split_gate,
    _reference_component_metrics,
    _validate_raw_dsp_cache,
)
from tv3.ml.e1d_sb_features import (
    E1DSB_FEATURE_BUILDER,
    E1DSB_SPEC_NAME,
    build_e1d_sb_feature_matrix,
    builder_manifest_payload,
    diagnostic_feature_count,
)
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.ml.rocket_features import RAW_DSP_FRAME_CACHE_ROOT
from tv3.sim.packaging.io import write_csv, write_json

SCHEMA_VERSION = "tv3-ec-msw-e1r-attachment-1"
DEFAULT_FRAME_GATES = {
    "peak_mae_samples_max": 0.15,
    "peak_p95_abs_error_samples_max": 0.25,
    "peak_bias_abs_samples_max": 0.05,
}


def run_ec_msw_e1r_attachment_audit(
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
    training_run_dir = _resolve(project_root, config["training_run_dir"])
    output_dir = _resolve(project_root, config["output_dir"])
    reference_path = _resolve(project_root, config["b1_reference_metrics"])
    checkpoint_path = _resolve(
        project_root,
        config.get("checkpoint_path", training_run_dir / "checkpoint.pt"),
    )
    run_config_path = training_run_dir / "run_config.json"
    run_kind = str(config.get("run_kind", "formal"))
    feature_source = str(config.get("feature_source", "raw_dsp_cache"))
    if feature_source not in {"raw_dsp_cache", "waveform"}:
        raise ValueError(f"unsupported feature_source: {feature_source!r}")

    if output_dir.exists():
        raise FileExistsError(f"E1r attachment output already exists: {output_dir}")
    for path in (dataset_dir, checkpoint_path, run_config_path, reference_path):
        if not path.exists():
            raise FileNotFoundError(f"E1r attachment input not found: {path}")

    e1d_sb_gate = _optional_e1d_sb_gate(config, project_root=project_root, run_kind=run_kind)

    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    raw_dsp_manifest = _validate_raw_dsp_cache(raw_dsp_dir, dataset_dir)
    run_config = _read_json(run_config_path)
    model = _load_model(run_config, checkpoint_path, device=str(config["device"]))
    if not model.has_peak_coordinate:
        raise ValueError(
            "E1r attachment requires an E1r checkpoint with frozen peak_coordinate_template"
        )
    template_digest = run_config.get("model_config", {}).get("peak_coordinate_template_digest")
    if template_digest != raw_dsp_manifest.get("template_digest"):
        raise ValueError(
            "E1r template digest does not match RawDSP cache: "
            f"{template_digest!r} != {raw_dsp_manifest.get('template_digest')!r}"
        )

    reference = _read_json(reference_path)
    frame_payload = _run_frame_fidelity(
        config,
        dataset_dir=dataset_dir,
        run_config=run_config,
        model=model,
        project_root=project_root,
    )

    alphas = tuple(float(value) for value in config["ridge_alphas"])
    parity_gates = dict(config.get("parity_gates", DEFAULT_PARITY_GATES))
    narrow_windows = list(config.get("narrow_o2_windows", DEFAULT_NARROW_O2_WINDOWS))
    matrices = {
        split: build_e1d_sb_feature_matrix(
            dataset_dir,
            split=split,
            feature_source=feature_source,  # type: ignore[arg-type]
        )
        for split in ("train", *EVAL_SPLITS)
    }
    train = matrices["train"]
    probe = ScaledRidgeCVRegressor(alphas=alphas).fit(train.x, train.y)

    parity_payload: dict[str, Any] = {
        "head": "train_only_scaled_ridgecv_on_e1d_sb_sequence_features",
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "feature_source": feature_source,
        "selected_alpha": probe.selected_alpha,
        "feature_count": len(train.feature_names),
        "diagnostic_feature_count": diagnostic_feature_count(train.feature_names),
        "gates": parity_gates,
        "reference_metrics_path": str(reference_path),
        "replaces": "e1r_last_mean_max_sequence_embedding",
        "splits": {},
    }
    narrow_rows: list[dict[str, object]] = []
    for split in EVAL_SPLITS:
        matrix = matrices[split]
        y_pred = probe.predict(matrix.x)
        metrics = _composition_metrics(y_pred, matrix.y)
        reference_metrics = _reference_component_metrics(reference, split)
        gate = _parity_split_gate(metrics["component_metrics"], reference_metrics, parity_gates)
        parity_payload["splits"][split] = {
            **metrics,
            "reference_component_metrics": reference_metrics,
            "gate": gate,
        }
        narrow_rows.extend(
            _narrow_window_rows(E1DSB_SPEC_NAME, split, y_pred, matrix.y, narrow_windows)
        )

    parity_payload["passed"] = all(
        entry["gate"]["passed"] for entry in parity_payload["splits"].values()
    )
    verdict = _build_verdict(
        run_kind=run_kind,
        frame_passed=bool(frame_payload["passed"]),
        parity_passed=bool(parity_payload["passed"]),
        e1d_sb_gate=e1d_sb_gate,
        feature_source=feature_source,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "frame_fidelity.json", frame_payload)
    write_json(output_dir / "b1_parity.json", parity_payload)
    write_json(output_dir / "verdict.json", verdict)
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
    write_json(
        output_dir / "summary.json",
        {
            "run_kind": run_kind,
            "feature_builder": E1DSB_FEATURE_BUILDER,
            "feature_source": feature_source,
            "frame_fidelity_passed": frame_payload["passed"],
            "sequence_parity_passed": parity_payload["passed"],
            "eval": {
                split: {
                    "x_O2_r2": parity_payload["splits"][split]["component_metrics"]["x_O2"]["r2"],
                    "x_CO2_r2": parity_payload["splits"][split]["component_metrics"]["x_CO2"]["r2"],
                    "x_N2_r2": parity_payload["splits"][split]["component_metrics"]["x_N2"]["r2"],
                    "parity_passed": parity_payload["splits"][split]["gate"]["passed"],
                    "delta_vs_control": parity_payload["splits"][split]["gate"]["r2_delta_vs_control"],
                }
                for split in EVAL_SPLITS
            },
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "dataset_dir": str(dataset_dir),
            "training_run_dir": str(training_run_dir),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "run_config_sha256": _sha256(run_config_path),
            "b1_reference_metrics": str(reference_path),
            "b1_reference_metrics_sha256": _sha256(reference_path),
            "raw_dsp_manifest_sha256": _sha256(raw_dsp_dir / "manifest.json"),
            "raw_dsp_template_digest": raw_dsp_manifest.get("template_digest"),
            "e1r_template_digest": template_digest,
            "run_kind": run_kind,
            "feature_source": feature_source,
            "feature_builder": E1DSB_FEATURE_BUILDER,
            "e1d_sb_gate": e1d_sb_gate,
            "builder": builder_manifest_payload(),
            "e2_allowed": False,
            "notes": [
                "frame fidelity from frozen E1r; sequence parity from e1d_sb builder",
                "does not retrain deep net; does not open E2",
                (
                    "feature_source=waveform is the deployable extraction path"
                    if feature_source == "waveform"
                    else "feature_source=raw_dsp_cache reuses validated RawDSP arrays; not a new waveform claim"
                ),
            ],
            "verdict": verdict["status"],
        },
    )
    return output_dir


def _run_frame_fidelity(
    config: Mapping[str, Any],
    *,
    dataset_dir: Path,
    run_config: Mapping[str, Any],
    model: Any,
    project_root: Path,
) -> dict[str, Any]:
    peak_targets = _load_peak_targets(dataset_dir)
    input_preprocess = _build_input_preprocess(dict(run_config))
    datasets = {
        split: _build_dataset(dataset_dir, split, dict(run_config), project_root=project_root)
        for split in ("train", *EVAL_SPLITS)
    }
    train_frames = len(datasets["train"]) * _timesteps(
        datasets["train"], input_preprocess=input_preprocess
    )
    sampled_frame_indices = _sample_frame_indices(
        train_frames,
        maximum=int(config["max_train_probe_frames"]),
        seed=int(config["probe_sample_seed"]),
    )
    _sequence_x, _train_y, train_frame_x, train_peak_y = _extract_train_features(
        model,
        datasets["train"],
        peak_targets,
        sampled_frame_indices,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        device=torch.device(str(config["device"])),
        input_preprocess=input_preprocess,
    )
    alphas = tuple(float(value) for value in config["ridge_alphas"])
    peak_probe = ScaledRidgeCVRegressor(alphas=alphas).fit(train_frame_x, train_peak_y)
    train_peak_error = peak_probe.predict(train_frame_x) - train_peak_y
    frame_gates = dict(config.get("frame_fidelity_gates", DEFAULT_FRAME_GATES))
    frame_payload: dict[str, Any] = {
        "probe": {
            "train_frame_count_total": train_frames,
            "train_frame_count_used": int(len(sampled_frame_indices)),
            "sample_seed": int(config["probe_sample_seed"]),
            "selected_alpha": peak_probe.selected_alpha,
            "target_role": "offline_audit_only_not_model_input_or_training_loss",
            "frontend": "frozen_e1r_peak_coordinate",
        },
        "gates": frame_gates,
        "train_probe_metrics": _peak_error_metrics(train_peak_error),
        "splits": {},
    }
    for split in EVAL_SPLITS:
        _sequence_x, _y_true, peak_error = _extract_eval_features(
            model,
            datasets[split],
            peak_targets,
            peak_probe,
            batch_size=int(config["batch_size"]),
            num_workers=int(config["num_workers"]),
            device=torch.device(str(config["device"])),
            input_preprocess=input_preprocess,
        )
        frame_metrics = _peak_error_metrics(peak_error)
        frame_payload["splits"][split] = {
            **frame_metrics,
            "gate": _frame_split_gate(frame_metrics, frame_gates),
        }
    frame_payload["passed"] = all(
        entry["gate"]["passed"] for entry in frame_payload["splits"].values()
    )
    return frame_payload


def _optional_e1d_sb_gate(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    run_kind: str,
) -> dict[str, Any] | None:
    path_value = config.get("e1d_sb_verdict_path")
    if path_value is None:
        if run_kind == "formal":
            raise ValueError("formal E1r attachment requires e1d_sb_verdict_path")
        return None
    path = _resolve(project_root, path_value)
    if not path.is_file():
        raise FileNotFoundError(f"e1d_sb_verdict_path not found: {path}")
    payload = _read_json(path)
    if payload.get("continue_e1r_attachment") is not True:
        raise ValueError(
            "e1d_sb verdict does not authorize E1r attachment: "
            f"continue_e1r_attachment={payload.get('continue_e1r_attachment')!r}"
        )
    if (
        payload.get("feature_builder") is not None
        and payload.get("feature_builder") != E1DSB_FEATURE_BUILDER
    ):
        raise ValueError(
            f"e1d_sb verdict feature_builder mismatch: {payload.get('feature_builder')!r}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "status": payload.get("status"),
        "continue_e1r_attachment": True,
    }


def _build_verdict(
    *,
    run_kind: str,
    frame_passed: bool,
    parity_passed: bool,
    e1d_sb_gate: Mapping[str, Any] | None,
    feature_source: str,
) -> dict[str, Any]:
    if run_kind == "smoke":
        status = "smoke_only"
        reason = (
            "Smoke run only verifies the E1r↔E1d-SB attachment pipeline; "
            "it cannot authorize a formal conclusion."
        )
    elif not frame_passed:
        status = "frame_fidelity_failed"
        reason = "Frozen E1r frame fidelity failed; stop attachment and repair frontend."
    elif not parity_passed:
        status = "b1_parity_failed"
        reason = (
            "E1r frame fidelity passed, but e1d_sb sequence features failed B1 non-inferiority."
        )
    else:
        status = "attachment_passed"
        reason = (
            "Frozen E1r frame fidelity and e1d_sb sequence Ridge both passed. "
            "Structured sequence attachment is authorized for further optional work; "
            "E2 remains forbidden."
        )
    return {
        "status": status,
        "reason": reason,
        "e2_allowed": False,
        "frame_fidelity_passed": frame_passed,
        "sequence_parity_passed": parity_passed,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "feature_source": feature_source,
        "e1d_sb_gate": dict(e1d_sb_gate) if e1d_sb_gate is not None else None,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    required = (
        "dataset_dir",
        "training_run_dir",
        "output_dir",
        "b1_reference_metrics",
        "device",
        "batch_size",
        "num_workers",
        "ridge_alphas",
        "max_train_probe_frames",
        "probe_sample_seed",
    )
    for key in required:
        if key not in config:
            raise ValueError(f"E1r attachment config missing required key: {key}")
    run_kind = str(config.get("run_kind", "formal"))
    if run_kind not in {"smoke", "formal"}:
        raise ValueError(f"unsupported run_kind: {run_kind!r}")
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")
    feature_builder = config.get("feature_builder", E1DSB_FEATURE_BUILDER)
    if feature_builder != E1DSB_FEATURE_BUILDER:
        raise ValueError(f"feature_builder must be {E1DSB_FEATURE_BUILDER!r}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload
