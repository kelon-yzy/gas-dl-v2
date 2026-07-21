#!/usr/bin/env python3
"""F5 model protocol: five-arm × S-Flow (plus L / S-Y report) with frozen B1/B7 heads.

Does not overwrite unidirectional B1/B7 artefacts. Formal 6000 training belongs on the server.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.ml.bidir_arm_features import (  # noqa: E402
    arm_specs,
    build_arm_feature_cache,
    load_arm_split_matrix,
)
from tv3.ml.bidir_s_flow import (  # noqa: E402
    DEFAULT_OOD_ABS_V_MAX,
    DEFAULT_TRAIN_ABS_V_MAX,
    derive_s_flow_split,
    zero_anchor_sequence_ids,
)
from tv3.ml.bidir_f5_gates import evaluate_f5_gates  # noqa: E402
from tv3.ml.mlp_head import MlpHeadConfig  # noqa: E402
from tv3.ml.ridge_head import ScaledRidgeCVRegressor  # noqa: E402
from tv3.ml.ridge_residual_head import (  # noqa: E402
    DEFAULT_OOF_FOLDS,
    DEFAULT_OOF_SEED,
    OofRidgeResidualMlpRegressor,
)
from tv3.ml.rocket_training import DEFAULT_RIDGE_ALPHAS  # noqa: E402
from tv3.ml.training import evaluate_regressor  # noqa: E402
from tv3.pipeline.build_tv3_bidir_features import build_tv3_bidir_feature_cache  # noqa: E402
from tv3.sim.core.tunnel_ventilation_bidir_schema import (  # noqa: E402
    COMPOSITION_SCHEME,
    FORMAL_FEATURE_BUILDER,
)

# Frozen B7 MLP recipe (same as configs/tv3_d2b_oof_ridge_residual_mlp.json).
FROZEN_B7_MLP = {
    "hidden_dims": (64, 64),
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "lr": 0.001,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
    "loss_weights": (1.0, 2.0, 1.0),
    "standardize_targets": True,
    "out_dim": 3,
    "zero_init_output": True,
}

ARM_IDS = ("A1", "A2", "A3", "A4", "A5")
HEADS = ("b1_ridge", "b7_residual")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics_payload(evaluation) -> dict[str, Any]:
    return {
        "mae": evaluation.metrics.mae,
        "rmse": evaluation.metrics.rmse,
        "r2": evaluation.metrics.r2,
        "component_metrics": {
            name: {"mae": m.mae, "rmse": m.rmse, "r2": m.r2}
            for name, m in evaluation.component_metrics.items()
        },
        "sum_abs_error": evaluation.sum_abs_error,
    }


def _subset_evaluation(evaluation, sequence_ids: Sequence[str]) -> dict[str, Any] | None:
    if not sequence_ids:
        return None
    wanted = set(sequence_ids)
    mask = np.asarray([sid in wanted for sid in evaluation.sequence_ids], dtype=bool)
    if not np.any(mask):
        return None
    y_true = evaluation.targets[mask]
    y_pred = evaluation.predictions[mask]
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    r2 = 1.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    o2_i = 1  # x_CO2, x_O2, x_N2
    o2_err = err[:, o2_i]
    o2_true = y_true[:, o2_i]
    o2_mae = float(np.mean(np.abs(o2_err)))
    o2_ss_tot = float(np.sum((o2_true - np.mean(o2_true)) ** 2))
    o2_r2 = 1.0 if o2_ss_tot < 1e-12 else 1.0 - float(np.sum(o2_err**2)) / o2_ss_tot
    return {
        "n": int(np.sum(mask)),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "o2_mae": o2_mae,
        "o2_r2": float(o2_r2),
    }


def train_arm_head(
    dataset_dir: Path,
    *,
    arm_id: str,
    head: str,
    output_dir: Path,
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    device: str = "cpu",
    seed: int = 20260704,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not overwrite:
        return _read_json(metrics_path)

    train = load_arm_split_matrix(dataset_dir, arm_id, split="train")
    val = load_arm_split_matrix(dataset_dir, arm_id, split="val")
    eval_splits = ("val", "test", "extrapolation")

    if head == "b1_ridge":
        model: Any = ScaledRidgeCVRegressor(alphas=ridge_alphas)
        model.fit(train.x, train.y, feature_names=train.feature_names)
        head_diagnostics: dict[str, Any] = {"selected_alpha": model.selected_alpha}
    elif head == "b7_residual":
        mlp = MlpHeadConfig(
            hidden_dims=FROZEN_B7_MLP["hidden_dims"],
            dropout=FROZEN_B7_MLP["dropout"],
            weight_decay=FROZEN_B7_MLP["weight_decay"],
            lr=FROZEN_B7_MLP["lr"],
            batch_size=FROZEN_B7_MLP["batch_size"],
            max_epochs=FROZEN_B7_MLP["max_epochs"],
            patience=FROZEN_B7_MLP["patience"],
            loss_weights=FROZEN_B7_MLP["loss_weights"],
            standardize_targets=FROZEN_B7_MLP["standardize_targets"],
            out_dim=FROZEN_B7_MLP["out_dim"],
            zero_init_output=FROZEN_B7_MLP["zero_init_output"],
            device=device,
            seed=seed,
        )
        model = OofRidgeResidualMlpRegressor(
            ridge_alphas=ridge_alphas,
            mlp_config=mlp,
            oof_folds=DEFAULT_OOF_FOLDS,
            oof_seed=DEFAULT_OOF_SEED,
        )
        model.fit(
            train.x,
            train.y,
            feature_names=train.feature_names,
            x_val=val.x,
            y_val=val.y,
            label_names=train.label_names,
        )
        head_diagnostics = dict(model.diagnostics)
    else:
        raise ValueError(f"unknown head={head!r}")

    evaluations: dict[str, Any] = {}
    eval_objects = {}
    for split_name in ("train", *eval_splits):
        matrix = train if split_name == "train" else load_arm_split_matrix(dataset_dir, arm_id, split=split_name)
        evaluation = evaluate_regressor(
            model,
            matrix,
            split=split_name,
            composition_scheme="tunnel_ventilation",
        )
        evaluations[split_name] = _metrics_payload(evaluation)
        eval_objects[split_name] = evaluation

    # Zero-anchor subset on test∪val∪extrapolation (report) and on train for hygiene.
    with (dataset_dir / "condition_grid_sequence.csv").open(encoding="utf-8", newline="") as handle:
        conditions = list(csv.DictReader(handle))
    zero_ids = zero_anchor_sequence_ids(conditions)
    zero_anchor_metrics = {
        split_name: _subset_evaluation(eval_objects[split_name], zero_ids)
        for split_name in eval_objects
    }

    # Gate (e) helper: pair ĉ bias vs AB-only polluted c, when arrays exist.
    sound_speed_audit = _sound_speed_bias_audit(dataset_dir)

    payload = {
        "schema_version": "tv3-bidir-model-arm-1",
        "arm_id": arm_id,
        "head": head,
        "feature_builder": arm_specs()[arm_id].feature_builder,
        "deployable": arm_specs()[arm_id].deployable,
        "dataset_dir": str(dataset_dir.resolve()),
        "feature_count": len(train.feature_names),
        "ridge_alphas": list(ridge_alphas),
        "seed": seed,
        "device": device,
        "evaluations": evaluations,
        "zero_anchor_metrics": zero_anchor_metrics,
        "sound_speed_audit": sound_speed_audit,
        "head_diagnostics": head_diagnostics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(metrics_path, payload)
    return payload


def _sound_speed_bias_audit(dataset_dir: Path) -> dict[str, Any] | None:
    frame_dir = dataset_dir / "features" / "raw_dsp_bidir" / FORMAL_FEATURE_BUILDER
    pair_path = frame_dir / "ultrasonic_sound_speed_pair_raw_dsp_m_per_s.npy"
    ab_path = frame_dir / "ultrasonic_sound_speed_ab_raw_dsp_m_per_s.npy"
    true_path = dataset_dir / "sequences" / "ultrasonic_sound_speed_m_per_s.npy"
    if not (pair_path.is_file() and ab_path.is_file() and true_path.is_file()):
        return None
    c_pair = np.load(pair_path, mmap_mode="r").astype(np.float64)
    c_ab = np.load(ab_path, mmap_mode="r").astype(np.float64)
    c_true = np.load(true_path, mmap_mode="r").astype(np.float64)

    def _mean_abs_seq_bias(est: np.ndarray) -> float:
        # Per-sequence signed mean bias, then mean of absolute values so +v/−v
        # errors do not cancel across a symmetric flow distribution.
        abs_biases = []
        for i in range(est.shape[0]):
            mask = np.isfinite(est[i]) & np.isfinite(c_true[i])
            if not np.any(mask):
                continue
            abs_biases.append(abs(float(np.mean(est[i][mask] - c_true[i][mask]))))
        return float(np.mean(abs_biases)) if abs_biases else float("nan")

    rec_path = frame_dir / "reciprocity_residual_p95_s.npy"
    if rec_path.is_file():
        seq_p95 = np.load(rec_path).astype(np.float64)
        rec_p95 = float(np.nanpercentile(seq_p95, 95))
    else:
        rec_p95 = float("nan")
    return {
        "pair_sound_speed_mean_abs_seq_bias_m_per_s": _mean_abs_seq_bias(c_pair),
        "ab_sound_speed_mean_abs_seq_bias_m_per_s": _mean_abs_seq_bias(c_ab),
        "reciprocity_residual_p95_of_seq_p95_s": rec_p95,
    }


def _verify_f4_prerequisite(config: dict[str, Any], *, project_root: Path = _TV3_ROOT) -> dict[str, Any]:
    """Require a passed F4 verdict before F5 training."""
    prereq = config.get("f4_prerequisite")
    if not isinstance(prereq, dict):
        raise ValueError("config.f4_prerequisite is required before F5 train")
    verdict_path = Path(prereq["verdict_path"])
    if not verdict_path.is_absolute():
        verdict_path = (project_root / verdict_path).resolve()
    if not verdict_path.is_file():
        raise FileNotFoundError(f"F4 prerequisite verdict missing: {verdict_path}")
    payload = _read_json(verdict_path)
    expected_passed = bool(prereq.get("expected_stage_passed", True))
    observed_passed = bool(payload.get("passed", payload.get("stage_passed", False)))
    if observed_passed != expected_passed:
        raise RuntimeError(
            f"F4 prerequisite failed: {verdict_path} passed={observed_passed}, "
            f"expected_stage_passed={expected_passed}"
        )
    return {
        "verdict_path": str(verdict_path),
        "passed": observed_passed,
        "verdict": payload.get("verdict"),
        "sha256": _file_sha256(verdict_path),
    }


def run_protocol(config: dict[str, Any], *, stage: str) -> dict[str, Any]:
    source_dir = Path(config["source_dataset_dir"])
    splits_root = Path(config["splits_root"])
    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    overwrite = bool(config.get("overwrite", False))
    device = str(config.get("device", "cpu"))
    seed = int(config.get("seed", 20260704))
    heads = tuple(config.get("heads", ["b1_ridge", "b7_residual"]))
    arms = tuple(config.get("arms", list(ARM_IDS)))
    gates = dict(config["f5_amplitude_gates"])

    s_flow_dir = splits_root / "s_flow"
    summary: dict[str, Any] = {
        "schema_version": "tv3-bidir-model-protocol-1",
        "composition_scheme": COMPOSITION_SCHEME,
        "feature_builder": FORMAL_FEATURE_BUILDER,
        "source_dataset_dir": str(source_dir.resolve()),
        "stage": stage,
    }

    if stage in ("derive", "all"):
        info = derive_s_flow_split(
            source_dir,
            s_flow_dir,
            seed=int(config.get("s_flow_seed", 20260721)),
            train_abs_v_max=float(config.get("train_abs_v_max", DEFAULT_TRAIN_ABS_V_MAX)),
            ood_abs_v_max=float(config.get("ood_abs_v_max", DEFAULT_OOD_ABS_V_MAX)),
        )
        summary["s_flow"] = info
        _write_json(output_root / "s_flow_split_info.json", info)

    if stage in ("features", "all"):
        if not (s_flow_dir / "splits" / "split_summary.json").is_file():
            raise FileNotFoundError(f"S-Flow dataset missing: {s_flow_dir}")
        frame_manifest = build_tv3_bidir_feature_cache(
            s_flow_dir,
            overwrite=overwrite,
            template_max_frames=int(config.get("template_max_frames", 512)),
        )
        summary["frame_cache"] = {
            "cache_dir": frame_manifest["cache_dir"],
            "build_signature": frame_manifest["build_signature"],
        }
        arm_manifests = {}
        for arm_id in arms:
            arm_manifests[arm_id] = build_arm_feature_cache(s_flow_dir, arm_id)
        summary["arm_caches"] = arm_manifests
        _write_json(output_root / "feature_build_summary.json", summary)

    if stage in ("train", "all"):
        summary["f4_prerequisite"] = _verify_f4_prerequisite(config)
        arm_metrics: dict[str, dict[str, Any]] = {}
        for arm_id in arms:
            for head in heads:
                run_dir = output_root / "runs" / f"{arm_id}_{head}"
                payload = train_arm_head(
                    s_flow_dir,
                    arm_id=arm_id,
                    head=head,
                    output_dir=run_dir,
                    device=device,
                    seed=seed,
                    overwrite=overwrite,
                )
                arm_metrics[f"{arm_id}:{head}"] = payload
        _write_json(output_root / "arm_metrics_index.json", {
            key: {
                "metrics_path": str((output_root / "runs" / f"{key.replace(':', '_')}" / "metrics.json").resolve()),
                "test_o2_mae": val["evaluations"]["test"]["component_metrics"]["x_O2"]["mae"],
                "ood_o2_mae": val["evaluations"]["extrapolation"]["component_metrics"]["x_O2"]["mae"],
            }
            for key, val in arm_metrics.items()
        })

        gate_results = {}
        for head in heads:
            if f"A1:{head}" in arm_metrics and f"A3:{head}" in arm_metrics:
                gate_results[head] = evaluate_f5_gates(arm_metrics, gates=gates, head=head)
        summary["gates"] = gate_results
        primary_head = "b1_ridge" if "b1_ridge" in gate_results else next(iter(gate_results), None)
        core_pass = bool(primary_head and gate_results[primary_head]["core_gates_passed"])
        # Selector ΔR² gate (d): required for formal F5 pass; pending until S-Y/S-L exist.
        selector_gate_d = {
            "status": "pending_optional_selectors",
            "passed": False,
            "note": (
                "Criterion d (S-Y/S-L noninferior ΔR²) is not evaluated until secondary "
                "selectors are derived; formal F5 pass and F6 are blocked while pending."
            ),
            "threshold": gates["selector_r2_noninferior_delta"],
        }
        summary["selector_gate_d"] = selector_gate_d
        gate_d_pass = bool(selector_gate_d["passed"])
        stage_passed = bool(core_pass and gate_d_pass)
        if stage_passed:
            verdict = "f5_model_protocol_passed"
            allowed_next = "F6_verdict_backfill"
        elif core_pass and not gate_d_pass:
            verdict = "f5_model_protocol_incomplete"
            allowed_next = None
        else:
            verdict = "f5_model_protocol_failed"
            allowed_next = None
        verdict_payload = {
            "schema_version": "tv3-bidir-f5-verdict-1",
            "verdict": verdict,
            "stage_passed": stage_passed,
            "core_gates_passed": core_pass,
            "selector_gate_d": selector_gate_d,
            "primary_head": primary_head,
            "gates": gate_results,
            "f5_amplitude_gates": gates,
            "allowed_next_stage_on_pass": allowed_next,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest_sha256": _file_sha256(source_dir / "manifest.json"),
        }
        _write_json(output_root / "f5_verdict.json", verdict_payload)
        summary["verdict"] = verdict_payload

    if stage in ("report_selectors", "all") and bool(config.get("derive_secondary_selectors", False)):
        summary["secondary_selectors"] = {
            "status": "skipped_in_code_path",
            "note": (
                "Optional L / S-Y: use scripts/recompute_tv3_split.py on the source bidir "
                "dataset after adapting SPXY X arrays (AB TOF proxy). Primary F5 gates use S-Flow. "
                "Until criterion d is evaluated, f5_verdict remains incomplete and F6 is blocked."
            ),
        }

    _write_json(output_root / "protocol_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=_TV3_ROOT / "configs" / "tv3_bidir_model_protocol.json",
    )
    p.add_argument(
        "--stage",
        choices=("derive", "features", "train", "report_selectors", "all"),
        default="all",
    )
    p.add_argument("--source-dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_json(args.config)
    if args.source_dataset_dir is not None:
        config["source_dataset_dir"] = str(args.source_dataset_dir)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.device is not None:
        config["device"] = args.device
    if args.overwrite:
        config["overwrite"] = True
    # Resolve relative paths against tv3 root.
    for key in ("source_dataset_dir", "splits_root", "output_dir"):
        path = Path(config[key])
        if not path.is_absolute():
            config[key] = str((_TV3_ROOT / path).resolve())
    summary = run_protocol(config, stage=args.stage)
    print(json.dumps({"stage": args.stage, "verdict": summary.get("verdict", {}).get("verdict")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
