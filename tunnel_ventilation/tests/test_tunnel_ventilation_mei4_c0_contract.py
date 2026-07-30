from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from scipy.stats import binom

from tv3.audit.mrs_ei_registry import verify_evidence_manifest

_ROOT = Path(__file__).resolve().parents[1]
_STATUS = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runner():
    path = _ROOT / "scripts" / "run_tv3_mei4_c0_contract_freeze.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei4_c0_contract_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c0_freezes_verified_parent_assets_and_promotes_mei4(tmp_path, monkeypatch):
    runner = _load_runner()
    stage_path = tmp_path / "stage_status.json"
    status = _load(_STATUS)
    status.pop("mei4", None)
    stage_path.write_text(json.dumps(status), encoding="utf-8")
    output_dir = tmp_path / "c0_freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei4_c0_contract_freeze.py",
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
    inventory = _load(output_dir / "b4_asset_inventory.json")
    assets = {entry["asset_id"]: entry for entry in inventory["assets"]}
    assert assets["registered_observations"]["tables"]["mixtures"]["row_count"] == 1944
    assert assets["registered_observations"]["tables"]["observation_rows"]["row_count"] == 7776
    assert assets["paired_solutions"]["row_count"] == 1296
    assert "truth_raw3_percent" in inventory["audit_only_fields"]
    assert "truth_raw3_percent" not in inventory["method_read_whitelist"]["paired_solutions"]["row"]

    contract = _load(output_dir / "mei4_execution_contract.json")
    assert contract["frozen_design"]["baseline_solver"] == "S1"
    gate = contract["calibration_gate"]
    frozen_bands = gate["exact_binomial_acceptance_counts"]
    assert set(frozen_bands) == {"0.5", "0.8", "0.9", "0.95"}
    for level_text, frozen_band in frozen_bands.items():
        level = float(level_text)
        assert frozen_band == {
            "lower_inclusive": int(
                binom.ppf(float(gate["sidak_alpha_each"]) / 2.0, 648, level)
            ),
            "upper_inclusive": int(
                binom.isf(float(gate["sidak_alpha_each"]) / 2.0, 648, level)
            ),
        }
    promoted = _load(stage_path)
    assert promoted["allowed_next_stage"] is None
    assert promoted["mei4"]["status"] == "mei4_contract_frozen"
    assert promoted["mei4"]["baseline_solver"] == "S1"
    assert promoted["mei4"]["authorizations"]["registered_sparse_simulation_generation"] != "authorized"
