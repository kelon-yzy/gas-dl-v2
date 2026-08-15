from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tv3.audit.mrs_ei_mei4_formal import (
    _audit_method,
    _audit_values,
    _flat_row,
    _importance_method,
    _laplace_method,
    _spec_for_record,
    load_frozen_c2_inputs,
)
from tv3.ml.mrs_posterior import weighted_equal_tailed_intervals
from tv3.ml.mrs_posterior import sample_nonnegative_tangent_gaussian
from tv3.ml.mrs_varpro import build_s1_parameterization

_ROOT = Path(__file__).resolve().parents[1]
_B4 = _ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei3_varpro_audit" / "freezes" / "20260729T120958962354Z_cf7ed57312d9"
_CONTRACT = _ROOT / "configs" / "tv3_mrs_ei" / "mei4_execution_contract.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_c2_loader_separates_method_record_from_audit_truth_fields():
    records, solver_config, audit_tables = load_frozen_c2_inputs(_B4)
    assert len(records) == 1296
    assert {record.split for record in records} == {"test", "ood"}
    record = records[0]
    assert "truth_raw3_percent" not in record.s1
    assert not hasattr(record, "truth_raw3_percent")
    truth, crb = _audit_values(audit_tables, record.mixture_id)
    assert truth.shape == (3,)
    assert crb > 0.0
    assert "b1_solver_audit" in solver_config


def test_c2_real_record_constructs_intervals_and_counts_rejections_in_coverage():
    contract = _load(_CONTRACT)
    records, solver_config, audit_tables = load_frozen_c2_inputs(_B4)
    record = records[0]
    spec = _spec_for_record(build_s1_parameterization(solver_config), audit_tables["calibration"], record)
    m1 = _laplace_method(method="M1", record=record, spec=spec, solver_config=solver_config, contract=contract)
    m1b = _laplace_method(method="M1b", record=record, spec=spec, solver_config=solver_config, contract=contract)
    m2 = _importance_method(record, spec, contract, m1)
    assert not m1.rejected
    assert set(m1.intervals or {}) == {"0.5", "0.8", "0.9", "0.95"}
    assert m1b.rejected is True
    assert m1b.rejection_reason == "truncation_interval_numerical_failure"
    row = _flat_row(record, m1)
    assert "x_O2_percent" not in row
    assert "O2_lower_0.95" in row
    truth, crb = _audit_values(audit_tables, record.mixture_id)
    events: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for posterior in (m1, m1b, m2):
        _audit_method(record=record, posterior=posterior, truth=truth, crb_o2=crb, contract=contract, coverage_events=events, metric_rows=metrics, diagnostic_rows=diagnostics)
    assert len(events) == 36
    assert all("covered" in event and "rejected" in event for event in events)
    assert len(diagnostics) == 3
    assert all(np.isfinite(value) for row in metrics for value in row["nll"])


def test_weighted_intervals_respect_nonuniform_mass():
    samples = np.asarray([[1.0, 2.0, 97.0], [2.0, 3.0, 95.0], [10.0, 11.0, 79.0]])
    intervals = weighted_equal_tailed_intervals(samples, [0.8, 0.1, 0.1], levels=[0.5])
    assert intervals["0.5"][0][1] < 2.0


def test_truncation_resolution_check_uses_nested_sobol_prefixes():
    contract = _load(_CONTRACT)
    records, solver_config, audit_tables = load_frozen_c2_inputs(_B4)
    record = records[0]
    spec = _spec_for_record(build_s1_parameterization(solver_config), audit_tables["calibration"], record)
    m1 = _laplace_method(method="M1", record=record, spec=spec, solver_config=solver_config, contract=contract)
    assert m1.laplace is not None
    initial = sample_nonnegative_tangent_gaussian(
        m1.laplace, candidates=65536, minimum_accepted=2048, seed=23
    )
    verified = sample_nonnegative_tangent_gaussian(
        m1.laplace, candidates=131072, minimum_accepted=2048, seed=23
    )
    assert np.allclose(verified.z[: initial.accepted], initial.z)
