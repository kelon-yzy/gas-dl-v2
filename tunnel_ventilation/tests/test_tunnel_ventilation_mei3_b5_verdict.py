from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tv3.audit.mrs_ei_registry import verify_evidence_manifest

_ROOT = Path(__file__).resolve().parents[1]
_STATUS = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
_B4_FREEZE = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei3_varpro_audit"
    / "freezes"
    / "20260729T120958962354Z_cf7ed57312d9"
)
_B4_MANIFEST_SHA256 = "604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_b5_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_b5_verdict_freeze.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_b5_verdict_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _b4_status():
    status = _load(_STATUS)
    mei3 = status["mei3"]
    mei3.update(
        {
            "phase": "b4_formal_solver_comparison",
            "verdict": "mei3_full_parameter_baseline_retained",
            "freeze_dir": _B4_FREEZE.relative_to(_ROOT).as_posix(),
            "verdict_path": (_B4_FREEZE / "mei3_verdict.json").relative_to(_ROOT).as_posix(),
            "evidence_manifest_path": (
                _B4_FREEZE / "evidence_manifest.json"
            ).relative_to(_ROOT).as_posix(),
            "evidence_manifest_sha256": _B4_MANIFEST_SHA256,
            "smoke_mode": False,
            "b4_passed_solver_gate": False,
        }
    )
    for key in (
        "b5_contract_frozen",
        "b5_data_generated",
        "mei4_baseline",
        "parent_b4_freeze_dir",
        "parent_b4_manifest_sha256",
        "parent_b4_verdict_sha256",
    ):
        mei3.pop(key, None)
    status["allowed_next_stage"] = None
    return status


def test_b5_freezes_verified_b4_verdict_without_generating_data(tmp_path, monkeypatch):
    runner = _load_b5_runner()
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b4_status()), encoding="utf-8")
    output_dir = tmp_path / "b5_freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b5_verdict_freeze.py",
            "--stage-status-path",
            str(stage_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert runner.main() == 0
    assert verify_evidence_manifest(
        output_dir / "evidence_manifest.json", project_root=_ROOT
    ) == []
    verdict = _load(output_dir / "mei3_b5_verdict.json")
    assert verdict["verdict"] == "mei3_full_parameter_baseline_retained"
    assert verdict["mei4_baseline"] == "S1"
    assert verdict["b5_data_generated"] is False
    assert (output_dir / "parent_b4_manifest.json").is_file()
    assert (output_dir / "parent_b4_verdict.json").is_file()

    promoted = _load(stage_path)
    assert promoted["allowed_next_stage"] is None
    assert promoted["mei3"]["phase"] == "b5_verdict_freeze"
    assert promoted["mei3"]["b5_contract_frozen"] is True
    assert promoted["mei3"]["mei4_baseline"] == "S1"
    assert promoted["mei3"]["parent_b4_manifest_sha256"] == _B4_MANIFEST_SHA256
