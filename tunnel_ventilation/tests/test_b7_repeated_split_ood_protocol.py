"""B7 repeated-split / OOD 协议编排测试。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_protocol_module():
    project_root = Path(__file__).resolve().parents[1]
    module_name = "test_b7_protocol_runner_module"
    spec = importlib.util.spec_from_file_location(
        module_name,
        project_root / "scripts" / "run_b7_repeated_split_ood_protocol.py",
    )
    assert spec is not None and spec.loader is not None
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def test_protocol_matrix_is_complete_and_frozen():
    mod = _load_protocol_module()
    specs = mod.build_protocol_matrix()
    assert len(specs) == 12  # 4 protocols × 3 split seeds
    assert {s.protocol_id for s in specs} == {"R", "L", "S-Y", "S-L"}
    assert {s.split_seed for s in specs} == {20260704, 20260712, 20260720}
    sy = [s for s in specs if s.protocol_id == "S-Y"]
    assert all(s.spxy_x_profile == "observed_v1" for s in sy)
    assert all(s.extrapolation_strategy == "y_margin_ood" for s in sy)
    assert all(s.is_ood_evidence for s in sy)
    sl = [s for s in specs if s.protocol_id == "S-L"]
    assert all(s.extrapolation_strategy == "lhs_boundary" for s in sl)
    assert all(s.is_ood_evidence for s in sl)
    random_specs = [s for s in specs if s.protocol_id == "R"]
    assert all(not s.is_ood_evidence for s in random_specs)
    assert mod.TRAINING_SEEDS == (42, 123, 456)


def test_protocol_pass_requires_positive_mean_delta_and_no_all_negative_ood_seed():
    mod = _load_protocol_module()
    rows = []
    for protocol_id, is_ood in (("R", False), ("L", False), ("S-Y", True), ("S-L", True)):
        for split_seed in (20260704, 20260712, 20260720):
            for training_seed in (42, 123, 456):
                rows.append(
                    {
                        "protocol_id": protocol_id,
                        "split_seed": split_seed,
                        "training_seed": training_seed,
                        "b7_status": "ok",
                        "delta_o2_r2_test": 0.02,
                        "delta_o2_r2_extrapolation": 0.01 if is_ood else 0.005,
                        "is_ood_evidence": is_ood,
                    }
                )
    verdict = mod.evaluate_protocol_pass(rows)
    assert verdict["protocol_pass"] is True

    # S-Y 某个 split seed 三 training seed 全负 → 失败
    for row in rows:
        if row["protocol_id"] == "S-Y" and row["split_seed"] == 20260704:
            row["delta_o2_r2_extrapolation"] = -0.02
    verdict_fail = mod.evaluate_protocol_pass(rows)
    assert verdict_fail["protocol_pass"] is False
    assert verdict_fail["checks"]["S-Y"]["passed"] is False


def test_audit_b7_frozen_rejects_test_early_stopping_monitor():
    mod = _load_protocol_module()
    payload = {
        "head": "oof_ridge_residual_mlp",
        "feature_builder": "d0_raw_dsp_physics_stats_v1",
        "feature_count": 1008,
        "diagnostics": {
            "residual_mlp": {
                "model_config": {"seed": 42, "hidden_dims": [64, 64]},
                "early_stopping": {"monitor": "test_o2_r2"},
            },
            "oof": {"fold_count": 5, "fold_seed": 20260711},
        },
    }
    errors = mod._audit_b7_frozen(payload, training_seed=42)
    assert any("early stopping" in err for err in errors)


def test_protocol_dry_run_writes_manifest(tmp_path):
    mod = _load_protocol_module()
    output_root = tmp_path / "protocol_out"
    rc = mod.main(
        [
            "--dry-run",
            "--output-root",
            str(output_root),
            "--protocol-ids",
            "R,S-Y",
            "--split-seeds",
            "20260704",
            "--training-seeds",
            "42",
        ]
    )
    assert rc == 0
    manifest = json.loads((output_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert manifest["invariants"]["spxy_x_profile_for_ood"] == "spxy_observed_stats_v1"
    assert manifest["invariants"]["early_stopping"] == "val_only"
    assert manifest["training_seeds"] == [42]
    assert len(manifest["matrix"]) == 2
    assert {row["protocol_id"] for row in manifest["matrix"]} == {"R", "S-Y"}


def test_result_matrix_requires_b1_pairing_fields():
    mod = _load_protocol_module()
    spec = mod.ProtocolSplitSpec(
        protocol_id="S-Y",
        split_seed=20260704,
        split_strategy="spxy_v1",
        extrapolation_strategy="y_margin_ood",
        spxy_alpha=0.5,
        spxy_x_profile="observed_v1",
        is_ood_evidence=True,
    )
    b1 = {
        "status": "ok",
        "metrics": {
            "components": {
                "test": {"x_O2": {"r2": 0.48}},
                "extrapolation": {"x_O2": {"r2": 0.37}},
                "val": {"x_O2": {"r2": 0.45}},
            },
            "sum_abs_error": {},
        },
    }
    b7 = {
        "status": "ok",
        "training_seed": 42,
        "metrics": {
            "components": {
                "test": {"x_O2": {"r2": 0.70}, "x_CO2": {"r2": 0.9}, "x_N2": {"r2": 0.8}},
                "extrapolation": {"x_O2": {"r2": 0.61}, "x_CO2": {"r2": 0.85}, "x_N2": {"r2": 0.75}},
                "val": {"x_O2": {"r2": 0.69}, "x_CO2": {"r2": 0.88}, "x_N2": {"r2": 0.79}},
                "train": {"x_O2": {"r2": 0.75}},
            },
            "sum_abs_error": {"test": 1.2},
            "train_val_gap_o2_r2": 0.06,
        },
    }
    row = mod._matrix_row_from_records(spec=spec, b1=b1, b7=b7)
    assert row["delta_o2_r2_test"] == pytest.approx(0.22)
    assert row["delta_o2_r2_extrapolation"] == pytest.approx(0.24)
    assert row["b1_test_o2_r2"] == pytest.approx(0.48)
    assert row["b7_test_o2_r2"] == pytest.approx(0.70)
