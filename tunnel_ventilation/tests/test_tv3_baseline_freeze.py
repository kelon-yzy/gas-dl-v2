"""B1/B7 正式基线冻结审计测试。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_freeze_module():
    project_root = Path(__file__).resolve().parents[1]
    module_name = "test_tv3_baseline_freeze_module"
    spec = importlib.util.spec_from_file_location(module_name, project_root / "scripts" / "freeze_tv3_baseline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_evidence(project_root: Path) -> None:
    b1_config = {
        "dataset_dir": "data/tv3-formal-6000",
        "feature_builder": "d0_raw_dsp_physics_stats_v1",
    }
    b7_config = {
        "dataset_dir": "data/tv3-formal-6000",
        "feature_builder": "d0_raw_dsp_physics_stats_v1",
        "head": "oof_ridge_residual_mlp",
    }
    b1_config_path = project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json"
    b7_config_path = project_root / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json"
    _write_json(b1_config_path, b1_config)
    _write_json(b7_config_path, b7_config)
    matrix = [
        {
            "protocol_id": protocol_id,
            "dataset_name": f"{protocol_id}-{seed}",
            "split_seed": seed,
        }
        for protocol_id in ("R", "L", "S-Y", "S-L")
        for seed in (1, 2, 3)
    ]
    protocol_root = project_root / "outputs" / "tv3_b7_protocol"
    _write_json(
        protocol_root / "protocol_manifest.json",
        {
            "b1_config_sha256": _sha256(b1_config_path),
            "b7_config_sha256": _sha256(b7_config_path),
            "is_complete_formal_matrix": True,
            "training_seeds": [42, 123, 456],
            "matrix": matrix,
            "invariants": {
                "feature_builder": "d0_raw_dsp_physics_stats_v1",
                "feature_count": 1008,
                "head": "oof_ridge_residual_mlp",
                "early_stopping": "val_only",
                "skip_source_raw_dsp_hardlink": True,
            },
        },
    )
    _write_json(
        protocol_root / "split_metrics.json",
        {
            "rows": [{} for _ in range(36)],
            "verdict": {"protocol_pass": True, "matrix_complete": True, "unexpected_row_count": 0},
        },
    )
    run_rows = []
    for row in matrix:
        base = {**row, "status": "ok", "split_hash": f"split-{row['protocol_id']}-{row['split_seed']}"}
        run_rows.append({**base, "stage": "derive", "ood_set_hash": "ood", "x_feature_profile": None})
        raw_dsp_row = {**base, "stage": "raw_dsp", "build_signature": "build", "template_digest": "template"}
        del raw_dsp_row["split_seed"]
        run_rows.append(raw_dsp_row)
    (protocol_root / "runs.jsonl").write_text(
        "\n".join(json.dumps(row) for row in run_rows) + "\n", encoding="utf-8"
    )
    _write_json(
        project_root / "outputs" / "tv3_d2b" / "raw_dsp_frame_fidelity" / "metrics.json",
        {
            "status": "passed",
            "source": {
                "template_mode": "train_baseline_median",
                "template_source_split": "train",
                "cache_build_signature": "build",
                "template_digest": "template",
            },
        },
    )
    _write_json(
        project_root / "outputs" / "tv3_d2b" / "raw_dsp_ridge_provenance" / "metrics.json",
        {
            "feature_builder": "d0_raw_dsp_physics_stats_v1",
            "feature_count": 1008,
            "evaluations": {"val": {"component_metrics": {"x_O2": {"r2": 0.4}}}, "test": {}, "extrapolation": {}},
            "raw_dsp_provenance": {
                "diagnostic_only": False,
                "complete_dataset": True,
                "build_signature": "build",
                "template_digest": "template",
            },
        },
    )


def test_freeze_writes_complete_immutable_evidence(tmp_path, monkeypatch):
    mod = _load_freeze_module()
    _build_evidence(tmp_path)
    monkeypatch.setattr(mod, "_git_output", lambda _root, *args: "commit-123" if args[0] == "rev-parse" else "")

    output_dir = mod.freeze_baseline(tmp_path, tmp_path / "outputs" / "tv3_baseline_freeze")

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    split_hashes = json.loads((output_dir / "split_hashes.json").read_text(encoding="utf-8"))
    assert manifest["git"] == {"commit": "commit-123", "worktree_clean": True}
    assert manifest["contract"]["output"] == "raw3"
    assert len(split_hashes["derived_splits"]) == 12
    assert len(split_hashes["raw_dsp_caches"]) == 12
    with pytest.raises(FileExistsError, match="already exists"):
        mod.freeze_baseline(tmp_path, output_dir)


def test_freeze_refuses_dirty_worktree_before_writing(tmp_path, monkeypatch):
    mod = _load_freeze_module()
    _build_evidence(tmp_path)
    monkeypatch.setattr(mod, "_git_output", lambda _root, *_args: " M tv3/ml/ridge_residual_head.py")
    output_dir = tmp_path / "outputs" / "tv3_baseline_freeze"

    with pytest.raises(RuntimeError, match="clean worktree"):
        mod.freeze_baseline(tmp_path, output_dir)

    assert not output_dir.exists()


def test_freeze_rejects_protocol_config_hash_drift(tmp_path, monkeypatch):
    mod = _load_freeze_module()
    _build_evidence(tmp_path)
    monkeypatch.setattr(mod, "_git_output", lambda _root, *args: "commit-123" if args[0] == "rev-parse" else "")
    manifest_path = tmp_path / "outputs" / "tv3_b7_protocol" / "protocol_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["b7_config_sha256"] = "drifted"
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeError, match="B7 config hash"):
        mod.freeze_baseline(tmp_path, tmp_path / "outputs" / "tv3_baseline_freeze")
