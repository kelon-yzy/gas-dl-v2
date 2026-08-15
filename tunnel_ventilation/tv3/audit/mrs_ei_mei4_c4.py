"""Evaluate the frozen MEI-4 C4 CC-SBI entry conditions."""
from __future__ import annotations

from typing import Any, Mapping


_DOMAINS = ("test", "ood")
_COMPONENTS = ("CO2", "O2", "N2")
_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _coverage_gate(rows: Any, method: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError("coverage primary_bands must be a list")
    expected = {(domain, component, level) for domain in _DOMAINS for component in _COMPONENTS for level in _LEVELS}
    selected = [row for row in rows if isinstance(row, Mapping) and row.get("method") == method]
    observed = {
        (str(row.get("domain")), str(row.get("component")), float(row.get("nominal_level")))
        for row in selected
    }
    if observed != expected:
        raise ValueError(f"{method} coverage bands do not match the frozen 24-band gate")
    failed = [
        {
            "domain": row["domain"],
            "component": row["component"],
            "nominal_level": row["nominal_level"],
            "covered": row["covered"],
            "acceptance_band": row["acceptance_band"],
        }
        for row in selected
        if row.get("within_acceptance_band") is not True
    ]
    return {"passed": not failed, "failed_bands": failed}


def _mc_gate(report: Mapping[str, Any], method: str, name: str) -> dict[str, Any]:
    methods = _mapping(report.get("methods"), f"{name}.methods")
    method_rows = _mapping(methods.get(method), f"{name}.methods.{method}")
    failed_domains = []
    for domain in _DOMAINS:
        row = _mapping(method_rows.get(domain), f"{name}.methods.{method}.{domain}")
        if row.get("passed") is not True:
            failed_domains.append(domain)
    return {"passed": not failed_domains, "failed_domains": failed_domains}


def _deterministic_method_gate(
    *,
    method: str,
    coverage_rows: Any,
    sbc_report: Mapping[str, Any],
    ppc_report: Mapping[str, Any],
) -> dict[str, Any]:
    coverage = _coverage_gate(coverage_rows, method)
    sbc = _mc_gate(sbc_report, method, "SBC")
    ppc = _mc_gate(ppc_report, method, "PPC")
    return {
        "method": method,
        "coverage": coverage,
        "sbc": sbc,
        "ppc": ppc,
        "complete_calibration_gate_passed": coverage["passed"] and sbc["passed"] and ppc["passed"],
    }


def _m2b_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    coverage = _coverage_gate(_mapping(report.get("coverage_report"), "M2b.coverage_report").get("primary_bands"), "M2b")
    return {
        "method": "M2b",
        "coverage": coverage,
        "sbc": {"passed": False, "reason": "not_registered_for_M2b_in_frozen_C3_protocol"},
        "ppc": {"passed": False, "reason": "not_registered_for_M2b_in_frozen_C3_protocol"},
        "complete_calibration_gate_passed": False,
        "incomplete_evidence_reason": "M2b has no registered SBC/PPC evidence in the frozen C3 protocol",
    }


def _mode_split_trigger(diagnostics: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    value = diagnostics.get("mode_split_distance_percent")
    if value is None:
        return {
            "threshold_percent": threshold,
            "evidence_available": False,
            "triggered": False,
            "reason": "no_registered_mode_split_artifact",
        }
    distance = float(value)
    return {
        "threshold_percent": threshold,
        "evidence_available": True,
        "mode_split_distance_percent": distance,
        "triggered": distance > threshold,
    }


def _truncation_trigger(diagnostics: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    methods = _mapping(diagnostics.get("methods"), "laplace_diagnostics.methods")
    medians: dict[str, float] = {}
    for method in ("M1", "M1b"):
        method_rows = _mapping(methods.get(method), f"laplace_diagnostics.methods.{method}")
        for domain in _DOMAINS:
            domain_rows = _mapping(method_rows.get(domain), f"laplace_diagnostics.methods.{method}.{domain}")
            loss = _mapping(domain_rows.get("truncation_mass_loss"), f"{method}.{domain}.truncation_mass_loss")
            medians[f"{method}:{domain}"] = float(loss.get("median"))
    return {
        "threshold": threshold,
        "median_truncation_mass_loss": medians,
        "triggered": any(value > threshold for value in medians.values()),
    }


def build_cc_sbi_trigger_audit(
    *,
    execution_contract: Mapping[str, Any],
    coverage_report: Mapping[str, Any],
    laplace_diagnostics: Mapping[str, Any],
    sbc_report: Mapping[str, Any],
    ppc_report: Mapping[str, Any],
    m2b_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the C4 trigger decision from immutable C2/C3 evidence only."""
    policy = _mapping(execution_contract.get("cc_sbi_policy"), "execution_contract.cc_sbi_policy")
    triggers = _mapping(policy.get("trigger_conditions"), "execution_contract.cc_sbi_policy.trigger_conditions")
    coverage_rows = coverage_report.get("primary_bands")
    m1 = _deterministic_method_gate(method="M1", coverage_rows=coverage_rows, sbc_report=sbc_report, ppc_report=ppc_report)
    m1b = _deterministic_method_gate(method="M1b", coverage_rows=coverage_rows, sbc_report=sbc_report, ppc_report=ppc_report)
    m2b = _m2b_gate(m2b_report)
    t1 = {
        "condition": str(triggers["T1"]),
        "active_second_path": "M2b",
        "method_gate_results": {"M1": m1, "M1b": m1b, "M2b": m2b},
        "triggered": not any(row["complete_calibration_gate_passed"] for row in (m1, m1b, m2b)),
    }
    t2 = _mode_split_trigger(laplace_diagnostics, float(_mapping(triggers["T2"], "trigger T2")["mode_split_distance_percent"]))
    t3 = _truncation_trigger(laplace_diagnostics, float(_mapping(triggers["T3"], "trigger T3")["median_truncation_mass_loss"]))
    budget = int(_mapping(triggers["T4"], "trigger T4")["m2b_forward_call_budget"])
    forward_calls = int(_mapping(m2b_report.get("cost"), "M2b.cost")["forward_calls"])
    t4 = {
        "condition": str(triggers["T4"]),
        "m2b_forward_calls": forward_calls,
        "forward_call_budget": budget,
        "triggered": forward_calls > budget,
    }
    trigger_report = {"T1": t1, "T2": t2, "T3": t3, "T4": t4}
    triggered_by = [name for name, row in trigger_report.items() if row["triggered"]]
    return {
        "schema_version": "tunnel-ventilation-mrs-ei-mei4-c4-trigger-audit-1",
        "phase": "c4_cc_sbi_trigger_audit",
        "cc_sbi_triggered": bool(triggered_by),
        "triggered_by": triggered_by,
        "triggers": trigger_report,
        "authorization_requirement": str(policy["requires_independent_authorization"]),
    }


__all__ = ["build_cc_sbi_trigger_audit"]
