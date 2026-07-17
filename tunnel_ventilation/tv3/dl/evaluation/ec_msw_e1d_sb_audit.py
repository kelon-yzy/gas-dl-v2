"""E1d-SB audit: Ridge parity for the deployable cal_plus_corr_psr_snr builder.

Does not train a new deep net. Does not open E2. Independent output directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.ml.rocket_features import RAW_DSP_FRAME_CACHE_ROOT
from tv3.sim.packaging.io import write_csv, write_json

SCHEMA_VERSION = "tv3-ec-msw-e1d-sb-1"
FULL_B1_DIAGNOSTIC_FEATURE_COUNT = 504


def run_ec_msw_e1d_sb_audit(
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
        raise FileExistsError(f"E1d-SB output already exists: {output_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    raw_dsp_manifest = _validate_raw_dsp_cache(raw_dsp_dir, dataset_dir)
    run_kind = str(config.get("run_kind", "formal"))
    feature_source = str(config.get("feature_source", "raw_dsp_cache"))
    if feature_source not in {"raw_dsp_cache", "waveform"}:
        raise ValueError(f"unsupported feature_source: {feature_source!r}")

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
    probe = ScaledRidgeCVRegressor(alphas=alphas).fit(train.x, train.y)

    split_payload: dict[str, Any] = {}
    ablation_rows: list[dict[str, object]] = []
    narrow_rows: list[dict[str, object]] = []
    for split in ("train", *eval_splits):
        matrix = matrices[split]
        y_pred = probe.predict(matrix.x)
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
                    probe.selected_alpha,
                    feature_count=len(train.feature_names),
                )
            )
            narrow_rows.extend(
                _narrow_window_rows(E1DSB_SPEC_NAME, split, y_pred, matrix.y, narrow_windows)
            )

    diagnostic_count = diagnostic_feature_count(train.feature_names)
    compact = diagnostic_count <= FULL_B1_DIAGNOSTIC_FEATURE_COUNT // 2
    parity_passed = bool(
        reference is not None
        and all(
            (split_payload[split].get("gate") or {}).get("passed") is True for split in eval_splits
        )
    )
    verdict = _build_verdict(
        run_kind=run_kind,
        has_reference=reference is not None,
        parity_passed=parity_passed,
        compact=compact,
        diagnostic_feature_count=diagnostic_count,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
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
        "feature_source": feature_source,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "e2_allowed": False,
        "builder": builder_manifest_payload(),
        "notes": [
            "E1d-SB audits one compact deployable set only",
            "not an end-to-end deep-net claim",
            "E2 FiLM/attention/MoE remain forbidden",
        ],
        "verdict": verdict["status"],
    }
    write_json(output_dir / "manifest.json", provenance)

    feature_set_entry = {
        "name": E1DSB_SPEC_NAME,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "role": "deployable_compact_cal_plus_corr_psr_snr",
        "feature_count": len(train.feature_names),
        "diagnostic_feature_count": diagnostic_count,
        "compact": compact,
        "feature_names": list(train.feature_names),
        "frame_arrays": list(info.frame_arrays),
        "sequence_scalars": list(info.sequence_scalars),
        "selected_alpha": probe.selected_alpha,
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
    has_reference: bool,
    parity_passed: bool,
    compact: bool,
    diagnostic_feature_count: int,
) -> dict[str, Any]:
    if run_kind == "smoke":
        status = "smoke_only"
        reason = (
            "Smoke run only verifies the E1d-SB builder and audit pipeline; "
            "it cannot authorize formal parity."
        )
        continue_e2e = False
    elif not has_reference:
        status = "missing_b1_reference"
        reason = "Formal E1d-SB requires frozen b1_reference_metrics."
        continue_e2e = False
    elif not compact:
        status = "not_compact"
        reason = (
            f"diagnostic_feature_count={diagnostic_feature_count} exceeds half of full B1 "
            f"diagnostic block ({FULL_B1_DIAGNOSTIC_FEATURE_COUNT // 2})."
        )
        continue_e2e = False
    elif parity_passed:
        status = "parity_passed"
        reason = (
            "Deployable cal_plus_corr_psr_snr builder passed O2/CO2/N2 non-inferiority "
            "on val/test/extrapolation. End-to-end attachment to E1r may be audited next; "
            "E2 remains forbidden."
        )
        continue_e2e = True
    else:
        status = "parity_failed"
        reason = (
            "Builder did not recover B1 non-inferiority on all eval splits. "
            "Stop EC-MSW learned expansion; keep B7."
        )
        continue_e2e = False
    return {
        "status": status,
        "reason": reason,
        "e2_allowed": False,
        "continue_e1r_attachment": continue_e2e,
        "feature_builder": E1DSB_FEATURE_BUILDER,
        "spec_name": E1DSB_SPEC_NAME,
        "compact": compact,
        "diagnostic_feature_count": diagnostic_feature_count,
        "parity_passed": parity_passed,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    required = ("dataset_dir", "output_dir", "ridge_alphas")
    for key in required:
        if key not in config:
            raise ValueError(f"E1d-SB config missing required key: {key}")
    run_kind = str(config.get("run_kind", "formal"))
    if run_kind not in {"smoke", "formal"}:
        raise ValueError(f"unsupported run_kind: {run_kind!r}")
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")
    if run_kind == "formal" and config.get("b1_reference_metrics") is None:
        raise ValueError("formal E1d-SB requires b1_reference_metrics")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload
