"""E1d-SB LS ablation audit: base compact set + SNR-weighted closed-form TOF-L scalars.

Additive only. Does not remove ultrasonic_snr_db. Does not open E2.
Requires formal attachment_passed gate. Independent output directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

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
    E1DSB_LS_FEATURE_BUILDER,
    E1DSB_LS_SPEC_NAME,
    build_e1d_sb_ls_feature_matrix,
    builder_manifest_payload,
    diagnostic_feature_count,
    e1d_sb_ls_builder_info,
)
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.ml.rocket_features import RAW_DSP_FRAME_CACHE_ROOT
from tv3.sim.packaging.io import write_csv, write_json

SCHEMA_VERSION = "tv3-ec-msw-e1d-sb-ls-1"
FULL_B1_DIAGNOSTIC_FEATURE_COUNT = 504
FORBIDDEN_OUTPUT_MARKERS = (
    "e1_s",
    "e1r_s",
    "e1d_s",
    "e1d_sb_s",
    "e1r_attach_",
)


def run_ec_msw_e1d_sb_ls_audit(
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
        raise FileExistsError(f"E1d-SB LS output already exists: {output_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    raw_dsp_manifest = _validate_raw_dsp_cache(raw_dsp_dir, dataset_dir)
    run_kind = str(config.get("run_kind", "formal"))
    feature_source = str(config.get("feature_source", "raw_dsp_cache"))
    if feature_source not in {"raw_dsp_cache", "waveform"}:
        raise ValueError(f"unsupported feature_source: {feature_source!r}")
    weight_mode = str(config.get("snr_ls_weight_mode", "amplitude"))
    if weight_mode not in {"amplitude", "power"}:
        raise ValueError(f"unsupported snr_ls_weight_mode: {weight_mode!r}")

    attachment_gate = _load_attachment_gate(
        project_root, config, run_kind=run_kind
    )

    reference_path = config.get("b1_reference_metrics")
    reference: dict[str, Any] | None = None
    if reference_path is not None:
        reference_path = _resolve(project_root, reference_path)
        if not reference_path.is_file():
            raise FileNotFoundError(f"b1_reference_metrics not found: {reference_path}")
        reference = _read_json(reference_path)

    baseline_summary = None
    baseline_path = config.get("baseline_e1d_sb_summary")
    if baseline_path is None:
        if run_kind == "formal":
            raise ValueError("formal E1d-SB LS requires baseline_e1d_sb_summary")
    else:
        baseline_path = _resolve(project_root, baseline_path)
        if not baseline_path.is_file():
            raise FileNotFoundError(f"baseline_e1d_sb_summary not found: {baseline_path}")
        baseline_summary = _read_json(baseline_path)

    alphas = tuple(float(value) for value in config["ridge_alphas"])
    gates = dict(config.get("parity_gates", DEFAULT_PARITY_GATES))
    narrow_windows = list(config.get("narrow_o2_windows", DEFAULT_NARROW_O2_WINDOWS))
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")

    info = e1d_sb_ls_builder_info()
    if "ultrasonic_snr_db" not in info.frame_arrays:
        raise ValueError("LS ablation must retain ultrasonic_snr_db")

    matrices = {
        split: build_e1d_sb_ls_feature_matrix(
            dataset_dir,
            split=split,
            feature_source=feature_source,  # type: ignore[arg-type]
            snr_ls_weight_mode=weight_mode,
        )
        for split in ("train", *eval_splits)
    }
    train = matrices["train"]
    if not any("ultrasonic_snr_db" in name for name in train.feature_names):
        raise ValueError("LS ablation matrix missing ultrasonic_snr_db features")
    if not any("snr_weighted_ls" in name for name in train.feature_names):
        raise ValueError("LS ablation matrix missing SNR-weighted LS scalars")

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
                _narrow_window_rows(E1DSB_LS_SPEC_NAME, split, y_pred, matrix.y, narrow_windows)
            )

    diagnostic_count = diagnostic_feature_count(train.feature_names)
    compact = diagnostic_count <= FULL_B1_DIAGNOSTIC_FEATURE_COUNT // 2
    parity_passed = bool(
        reference is not None
        and all(
            (split_payload[split].get("gate") or {}).get("passed") is True for split in eval_splits
        )
    )
    delta_vs_baseline = _delta_vs_baseline(
        split_payload,
        baseline_summary,
        eval_splits,
        require=run_kind == "formal",
    )
    verdict = _build_verdict(
        run_kind=run_kind,
        attachment_gate=attachment_gate,
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
        "attachment_gate": attachment_gate,
        "baseline_e1d_sb_summary": None if baseline_path is None else str(baseline_path),
        "baseline_e1d_sb_summary_sha256": (
            None if baseline_path is None else _sha256(baseline_path)
        ),
        "run_kind": run_kind,
        "feature_source": feature_source,
        "feature_builder": E1DSB_LS_FEATURE_BUILDER,
        "snr_ls_weight_mode": weight_mode,
        "e2_allowed": False,
        "builder": builder_manifest_payload(include_snr_weighted_ls=True),
        "notes": [
            "additive SNR-weighted LS ablation on top of cal_plus_corr_psr_snr",
            "frame SNR retained; not a pure TOF-L LS head",
            "E2 FiLM/attention/MoE remain forbidden",
            "B7 remains the default deployable RawDSP head",
        ],
        "verdict": verdict["status"],
    }
    write_json(output_dir / "manifest.json", provenance)

    feature_set_entry = {
        "name": E1DSB_LS_SPEC_NAME,
        "feature_builder": E1DSB_LS_FEATURE_BUILDER,
        "role": "ablation_cal_plus_corr_psr_snr_plus_snr_weighted_ls",
        "feature_count": len(train.feature_names),
        "diagnostic_feature_count": diagnostic_count,
        "compact": compact,
        "feature_names": list(train.feature_names),
        "frame_arrays": list(info.frame_arrays),
        "sequence_scalars": list(info.sequence_scalars),
        "selected_alpha": probe.selected_alpha,
        "snr_ls_weight_mode": weight_mode,
        "splits": split_payload,
    }
    write_json(output_dir / "feature_sets.json", {"feature_sets": [feature_set_entry]})

    summary = {
        "gates": gates,
        "control_source": "b1_reference_metrics" if reference is not None else None,
        "feature_builder": E1DSB_LS_FEATURE_BUILDER,
        "feature_source": feature_source,
        "snr_ls_weight_mode": weight_mode,
        "feature_count": len(train.feature_names),
        "diagnostic_feature_count": diagnostic_count,
        "full_b1_diagnostic_feature_count": FULL_B1_DIAGNOSTIC_FEATURE_COUNT,
        "compact": compact,
        "parity_passed": parity_passed,
        "delta_vs_e1d_sb_baseline": delta_vs_baseline,
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
        "feature_set": E1DSB_LS_SPEC_NAME,
        "feature_builder": E1DSB_LS_FEATURE_BUILDER,
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


def _delta_vs_baseline(
    split_payload: Mapping[str, Any],
    baseline_summary: Mapping[str, Any] | None,
    eval_splits: tuple[str, ...],
    *,
    require: bool = False,
) -> dict[str, Any] | None:
    if baseline_summary is None:
        if require:
            raise ValueError("formal E1d-SB LS requires baseline_e1d_sb_summary")
        return None
    baseline_eval = baseline_summary.get("eval")
    if not isinstance(baseline_eval, dict):
        if require:
            raise ValueError("baseline_e1d_sb_summary must contain an object field 'eval'")
        return None
    out: dict[str, Any] = {}
    missing: list[str] = []
    for split in eval_splits:
        base = baseline_eval.get(split)
        if not isinstance(base, dict):
            missing.append(split)
            continue
        for key in ("x_O2_r2", "x_CO2_r2", "x_N2_r2"):
            if key not in base:
                raise ValueError(
                    f"baseline_e1d_sb_summary.eval[{split!r}] missing required key {key!r}"
                )
        current = split_payload[split]["component_metrics"]
        out[split] = {
            "delta_o2_r2_vs_e1d_sb": current["x_O2"]["r2"] - float(base["x_O2_r2"]),
            "delta_co2_r2_vs_e1d_sb": current["x_CO2"]["r2"] - float(base["x_CO2_r2"]),
            "delta_n2_r2_vs_e1d_sb": current["x_N2"]["r2"] - float(base["x_N2_r2"]),
        }
    if missing:
        if require:
            raise ValueError(
                "baseline_e1d_sb_summary.eval missing required splits: "
                + ", ".join(missing)
            )
        return out or None
    return out


def _attachment_gate_identity_ok(gate: Mapping[str, Any] | None) -> tuple[bool, str]:
    if gate is None:
        return False, "Formal LS ablation requires attachment_verdict_path with attachment_passed."
    if gate.get("status") != "attachment_passed":
        return False, (
            f"attachment verdict status={gate.get('status')!r}; "
            "LS ablation remains blocked."
        )
    if gate.get("feature_builder") != E1DSB_FEATURE_BUILDER:
        return False, (
            "attachment gate feature_builder mismatch: "
            f"expected {E1DSB_FEATURE_BUILDER!r}, got {gate.get('feature_builder')!r}."
        )
    if gate.get("e2_allowed") is not False:
        return False, (
            "attachment gate must record e2_allowed=false; "
            f"got {gate.get('e2_allowed')!r}."
        )
    if gate.get("frame_fidelity_passed") is not True:
        return False, (
            "attachment gate requires frame_fidelity_passed=true; "
            f"got {gate.get('frame_fidelity_passed')!r}."
        )
    if gate.get("sequence_parity_passed") is not True:
        return False, (
            "attachment gate requires sequence_parity_passed=true; "
            f"got {gate.get('sequence_parity_passed')!r}."
        )
    return True, ""


def _build_verdict(
    *,
    run_kind: str,
    attachment_gate: Mapping[str, Any] | None,
    has_reference: bool,
    parity_passed: bool,
    compact: bool,
    diagnostic_feature_count: int,
) -> dict[str, Any]:
    attach_ok, attach_reason = _attachment_gate_identity_ok(attachment_gate)
    if run_kind == "smoke":
        status = "smoke_only"
        reason = (
            "Smoke run only verifies the SNR-weighted LS ablation pipeline; "
            "it cannot authorize formal ablation conclusions."
        )
    elif attachment_gate is None:
        status = "missing_attachment_gate"
        reason = "Formal LS ablation requires attachment_verdict_path with attachment_passed."
    elif not attach_ok:
        status = "attachment_gate_failed"
        reason = attach_reason
    elif not has_reference:
        status = "missing_b1_reference"
        reason = "Formal LS ablation requires frozen b1_reference_metrics."
    elif not compact:
        status = "not_compact"
        reason = (
            f"diagnostic_feature_count={diagnostic_feature_count} exceeds half of full B1 "
            f"diagnostic block ({FULL_B1_DIAGNOSTIC_FEATURE_COUNT // 2})."
        )
    elif parity_passed:
        status = "ls_ablation_passed"
        reason = (
            "Additive SNR-weighted LS builder retained SNR and passed B1 non-inferiority. "
            "This does not open E2; B7 remains the default head."
        )
    else:
        status = "ls_ablation_failed"
        reason = (
            "Additive LS scalars did not recover B1 non-inferiority on all eval splits. "
            "Keep E1d-SB / attachment path; do not open E2."
        )
    return {
        "status": status,
        "reason": reason,
        "e2_allowed": False,
        "snr_retained": True,
        "feature_builder": E1DSB_LS_FEATURE_BUILDER,
        "spec_name": E1DSB_LS_SPEC_NAME,
        "compact": compact,
        "diagnostic_feature_count": diagnostic_feature_count,
        "parity_passed": parity_passed,
        "attachment_status": None if attachment_gate is None else attachment_gate.get("status"),
        "attachment_identity_ok": attach_ok,
    }


def _load_attachment_gate(
    project_root: Path,
    config: Mapping[str, Any],
    *,
    run_kind: str,
) -> dict[str, Any] | None:
    path = config.get("attachment_verdict_path")
    if path is None:
        if run_kind == "formal":
            raise ValueError("formal E1d-SB LS requires attachment_verdict_path")
        return None
    resolved = _resolve(project_root, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"attachment_verdict_path not found: {resolved}")
    payload = _read_json(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "status": payload.get("status"),
        "e2_allowed": payload.get("e2_allowed"),
        "feature_builder": payload.get("feature_builder"),
        "frame_fidelity_passed": payload.get("frame_fidelity_passed"),
        "sequence_parity_passed": payload.get("sequence_parity_passed"),
    }


def _assert_safe_output_dir(output_dir: Path) -> None:
    name = output_dir.name
    for marker in FORBIDDEN_OUTPUT_MARKERS:
        if name.startswith(marker) and not name.startswith("e1d_sb_ls"):
            raise ValueError(
                f"E1d-SB LS output_dir basename {name!r} collides with frozen evidence dirs"
            )
    if name.startswith("e1d_sb_s") and not name.startswith("e1d_sb_ls"):
        raise ValueError(f"refusing to overwrite baseline E1d-SB dir pattern: {name!r}")


def _validate_config(config: Mapping[str, Any]) -> None:
    required = ("dataset_dir", "output_dir", "ridge_alphas", "feature_builder")
    for key in required:
        if key not in config:
            raise ValueError(f"E1d-SB LS config missing required key: {key}")
    if config["feature_builder"] != E1DSB_LS_FEATURE_BUILDER:
        raise ValueError(
            f"feature_builder must be {E1DSB_LS_FEATURE_BUILDER!r}, got {config['feature_builder']!r}"
        )
    if config.get("include_snr_weighted_ls") is not True:
        raise ValueError("E1d-SB LS config requires include_snr_weighted_ls=true")
    run_kind = str(config.get("run_kind", "formal"))
    if run_kind not in {"smoke", "formal"}:
        raise ValueError(f"unsupported run_kind: {run_kind!r}")
    eval_splits = tuple(config.get("eval_splits", EVAL_SPLITS))
    if eval_splits != EVAL_SPLITS:
        raise ValueError(f"eval_splits must be exactly {EVAL_SPLITS}, got {eval_splits}")
    if run_kind == "formal" and config.get("b1_reference_metrics") is None:
        raise ValueError("formal E1d-SB LS requires b1_reference_metrics")
    if run_kind == "formal" and config.get("attachment_verdict_path") is None:
        raise ValueError("formal E1d-SB LS requires attachment_verdict_path")
    if run_kind == "formal" and config.get("baseline_e1d_sb_summary") is None:
        raise ValueError("formal E1d-SB LS requires baseline_e1d_sb_summary")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload
