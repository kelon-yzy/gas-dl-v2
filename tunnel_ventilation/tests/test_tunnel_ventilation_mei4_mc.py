from __future__ import annotations

import copy
import json
from pathlib import Path

from tv3.audit.mrs_ei_mei4_mc import _all_posteriors, _load_context, _m2b_report, run_ppc, run_sbc


_ROOT = Path(__file__).resolve().parents[1]
_B4 = _ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei3_varpro_audit" / "freezes" / "20260729T120958962354Z_cf7ed57312d9"
_CONTRACT = _ROOT / "configs" / "tv3_mrs_ei" / "mei4_execution_contract.json"


def _smoke_contract() -> dict:
    contract = copy.deepcopy(json.loads(_CONTRACT.read_text(encoding="utf-8")))
    contract["mc_protocol"]["sbc"]["replicates_per_domain_per_method"] = 1
    contract["mc_protocol"]["sbc"]["posterior_draws_per_replicate"] = 8
    contract["mc_protocol"]["ppc"]["y_rep_per_frozen_mixture"] = 2
    contract["mc_protocol"]["m2b"]["resamples_per_mixture"] = 2
    return contract


def test_c3_smoke_preserves_m2_parameter_draws_and_emits_aggregate_reports():
    templates, solver_config, calibration, audit_tables, records = _load_context(_B4)
    contract = _smoke_contract()
    record = next(item for item in records if item.mixture_id == "M000651")
    ood_record = next(item for item in records if item.mixture_id == "M001355")
    posteriors = _all_posteriors(record, solver_config, calibration, contract)
    assert posteriors["M2"].rejected is False
    assert posteriors["M2"].parameter_samples is not None

    sbc = run_sbc(
        templates={"test": templates["test"][:1], "ood": templates["ood"][:1]},
        solver_config=solver_config,
        calibration=calibration,
        contract=contract,
    )
    assert sbc["methods"]["M1"]["test"]["n_replicates"] == 1
    assert len(sbc["methods"]["M1"]["test"]["components"]["O2"]["histogram"]) == 9

    ppc = run_ppc(
        records=[record, ood_record], solver_config=solver_config, calibration=calibration, contract=contract
    )
    assert ppc["methods"]["M1"]["test"]["n_frozen_mixtures"] == 1
    assert ppc["methods"]["M1"]["test"]["whitened_residual_norm_tail_probability"]["n"] <= 1

    m2b = _m2b_report(
        records=[record, ood_record],
        audit_tables=audit_tables,
        solver_config=solver_config,
        calibration=calibration,
        contract=contract,
    )
    assert m2b["resamples_per_mixture"] == 2
    assert len(m2b["coverage_report"]["primary_bands"]) == 24
