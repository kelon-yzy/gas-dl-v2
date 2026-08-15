from __future__ import annotations

import json
from pathlib import Path

from tv3.audit.mrs_ei_mei4_c4 import build_cc_sbi_trigger_audit


_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = json.loads((_ROOT / "configs" / "tv3_mrs_ei" / "mei4_execution_contract.json").read_text(encoding="utf-8"))


def _coverage_rows(method: str, passed: bool) -> list[dict]:
    return [
        {
            "method": method,
            "domain": domain,
            "component": component,
            "nominal_level": level,
            "covered": 648 if passed else 0,
            "acceptance_band": {"lower_inclusive": 1, "upper_inclusive": 648},
            "within_acceptance_band": passed,
        }
        for domain in ("test", "ood")
        for component in ("CO2", "O2", "N2")
        for level in (0.5, 0.8, 0.9, 0.95)
    ]


def _mc_report() -> dict:
    return {
        "methods": {
            method: {domain: {"passed": True} for domain in ("test", "ood")}
            for method in ("M1", "M1b", "M2")
        }
    }


def test_c4_trigger_requires_independent_authorization_when_nonlearning_paths_fail():
    audit = build_cc_sbi_trigger_audit(
        execution_contract=_CONTRACT,
        coverage_report={"primary_bands": _coverage_rows("M1", False) + _coverage_rows("M1b", False)},
        laplace_diagnostics={
            "methods": {
                method: {domain: {"truncation_mass_loss": {"median": 0.0}} for domain in ("test", "ood")}
                for method in ("M1", "M1b")
            }
        },
        sbc_report=_mc_report(),
        ppc_report=_mc_report(),
        m2b_report={
            "coverage_report": {"primary_bands": _coverage_rows("M2b", False)},
            "cost": {"forward_calls": 1_000_001},
        },
    )

    assert audit["cc_sbi_triggered"] is True
    assert set(audit["triggered_by"]) == {"T1", "T4"}
    assert audit["authorization_requirement"] == "mei4_cc_sbi_training_draws"
    assert audit["triggers"]["T1"]["method_gate_results"]["M2b"]["complete_calibration_gate_passed"] is False
