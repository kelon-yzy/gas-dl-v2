"""MEI-0 registry freeze unit tests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tv3.audit.mrs_ei_registry import (
    SCHEMA_VERSION,
    audit_mei0_registries,
    build_narrow_points,
    default_config_dir,
    load_json,
    verify_evidence_manifest,
)

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "tv3_mrs_ei"


def test_default_config_dir_points_at_tv3_mrs_ei():
    assert default_config_dir().name == "tv3_mrs_ei"


def test_narrow_points_count_is_216():
    design = load_json(_CFG / "design_space.json")
    points = build_narrow_points(design)
    assert len(points) == 216
    assert design["narrow_context_grid"]["expected_n_points"] == 216


def test_baseline_k4_frequencies():
    design = load_json(_CFG / "design_space.json")
    assert design["frequency_band"]["baseline_k4_hz"] == [
        25000.0,
        63000.0,
        100000.0,
        200000.0,
    ]


def test_model_family_registers_required_envelopes():
    model = load_json(_CFG / "model_family_registry.json")
    assert model["schema_version"] == SCHEMA_VERSION
    ids = {f["id"] for f in model["model_families"]}
    assert "F0_mrs1_baseline" in ids
    assert "F4_diffraction_near_field" in ids
    assert model["claim_scope"] == "registered_simulation_domain_only"


def test_metric_registry_forbids_waveform_until_authorized():
    metric = load_json(_CFG / "metric_registry.json")
    auth = metric["data_generation_authorization"]
    assert auth["formal_waveform_generation"] == "forbidden_until_authorized"
    assert float(metric["delta_num"]["floor"]) == 0.02


def test_audit_incomplete_before_delta_num_frozen():
    # Source metric_registry may still have frozen_value=null before first freeze.
    metric = load_json(_CFG / "metric_registry.json")
    if metric["delta_num"]["frozen_value"] is None:
        audit = audit_mei0_registries(_CFG, project_root=_ROOT)
        assert audit["passed"] is False
        assert audit["verdict"] == "mei0_registry_incomplete"
        assert any("frozen_value" in msg for msg in audit["issues"])
    else:
        audit = audit_mei0_registries(_CFG, project_root=_ROOT)
        assert audit["verdict"] in {
            "mei0_registry_frozen",
            "mei0_registry_incomplete",
        }


def test_preflight_audit_passes_with_temporary_delta_num():
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_num=False,
    )
    assert audit["issues"] == []
    assert audit["passed"] is True


def test_audit_rejects_tampered_upstream_verdict_hash():
    model = load_json(_CFG / "model_family_registry.json")
    model["lineage"]["mrs2_verdict"]["expected_sha256"] = "0" * 64
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_num=False,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is False
    assert any("mrs2 verdict sha256 mismatch" in item for item in audit["issues"])


def test_audit_rejects_quantitative_model_bound_without_refs():
    model = load_json(_CFG / "model_family_registry.json")
    f1 = next(item for item in model["model_families"] if item["id"] == "F1_humid_air_c_eq")
    f1["refs"] = []
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_num=False,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is False
    assert any("quantitative bound requires" in item for item in audit["issues"])


def test_audit_rejects_d4_equal_cost_eligibility():
    design = load_json(_CFG / "design_space.json")
    d4 = next(item for item in design["design_arms"] if item["id"] == "D4")
    d4["eligible_for_information_gate"] = True
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_num=False,
        registry_overrides={"design_space.json": design},
    )
    assert audit["passed"] is False
    assert any("D4 must be excluded" in item for item in audit["issues"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freeze_writes_hashed_evidence_and_refuses_overwrite(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model = load_json(_CFG / "model_family_registry.json")
    design = load_json(_CFG / "design_space.json")
    metric = load_json(_CFG / "metric_registry.json")

    metric["delta_num"]["recompute_spec"]["n_repeat"] = 2
    metric["delta_num"]["frozen_value"] = None
    metric["delta_num"]["recompute_artifact"] = None

    payloads = {
        "model_family_registry.json": model,
        "design_space.json": design,
        "metric_registry.json": metric,
    }
    for name, payload in payloads.items():
        (config_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    output_dir = tmp_path / "freeze"
    command = [
        sys.executable,
        str(_ROOT / "scripts" / "run_tv3_mei0_registry_freeze.py"),
        "--config-dir",
        str(config_dir),
        "--output-dir",
        str(output_dir),
    ]
    first = subprocess.run(
        command,
        cwd=_ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert first.returncode == 0, first.stdout + first.stderr

    manifest = load_json(output_dir / "evidence_manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        assert _sha256(output_dir / name) == expected
    verdict = load_json(output_dir / "mei0_verdict.json")
    assert _sha256(output_dir / "evidence_manifest.json") == verdict[
        "evidence_manifest"
    ]["sha256"]
    assert verify_evidence_manifest(
        output_dir / "evidence_manifest.json",
        project_root=_ROOT,
        expected_manifest_sha256=verdict["evidence_manifest"]["sha256"],
    ) == []
    repeats = verdict["delta_num_recompute_summary"]["fresh_process_repeats"]
    assert len({item["process_id"] for item in repeats}) == 2
    assert all(item["process_id"] != verdict["delta_num_recompute_summary"]["nominal"]["process_id"] for item in repeats)

    second = subprocess.run(
        command,
        cwd=_ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    assert second.returncode != 0
    assert "refuse overwrite" in second.stderr
