"""Evidence checks for the MEI-4 line closure freeze.

Every claim that the tv3 methodology paper cites is registered in
``configs/tv3_mrs_ei/mei4_closure_contract.json`` together with its expected
value. This module recomputes each claim from frozen artifacts so the closure
freeze can record whether the paper text and the evidence still agree.

The module never mutates artifacts and never runs new physics. It only reads
frozen outputs, recomputes summary statistics and compares them with the
registered expectations.
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tv3.audit.mrs_ei_registry import load_json, verify_evidence_manifest

CHECK_REGISTRY: dict[str, Callable[["CheckContext", dict[str, Any], Any], Any]] = {}

STATUS_MATCH = "match"
STATUS_MISMATCH = "mismatch"
STATUS_UNVERIFIABLE = "unverifiable"

DEFAULT_ABS_TOLERANCE = 5e-4
INTERVAL_LEVELS = ("0.5", "0.8", "0.9", "0.95")


def register_check(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register an observation function under ``name``."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if name in CHECK_REGISTRY:
            raise ValueError(f"duplicate check name: {name}")
        CHECK_REGISTRY[name] = func
        return func

    return decorator


@dataclass
class CheckContext:
    """Lazily loaded, read-only view of the frozen artifacts a check needs."""

    project_root: Path
    c2_freeze_dir: Path
    _json_cache: dict[Path, Any] = field(default_factory=dict)
    _csv_cache: dict[Path, list[dict[str, str]]] = field(default_factory=dict)

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def json_at(self, relative: str) -> Any:
        path = self.resolve(relative)
        if path not in self._json_cache:
            self._json_cache[path] = load_json(path)
        return self._json_cache[path]

    def c2_json(self, name: str) -> Any:
        return self.json_at((self.c2_freeze_dir / name).as_posix())

    def c2_rows(self, name: str) -> list[dict[str, str]]:
        path = self.c2_freeze_dir / name
        if path not in self._csv_cache:
            text = path.read_text(encoding="utf-8").splitlines()
            self._csv_cache[path] = list(csv.DictReader(text))
        return self._csv_cache[path]

    def accepted_interval_rows(
        self, domain: str, method: str
    ) -> list[dict[str, str]]:
        name = f"posterior_intervals_{domain}.csv"
        return [
            row
            for row in self.c2_rows(name)
            if row["method"] == method and row["rejected"].strip().lower() == "false"
        ]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one registered claim."""

    check_id: str
    check: str
    status: str
    claim: str
    expected: Any
    observed: Any
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.check_id,
            "check": self.check,
            "status": self.status,
            "claim": self.claim,
            "expected": self.expected,
            "observed": self.observed,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


def _values_agree(expected: Any, observed: Any, tolerance: float) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return bool(expected) == bool(observed)
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(expected) - float(observed)) <= tolerance
    if isinstance(expected, dict) and isinstance(observed, dict):
        if set(expected) != set(observed):
            return False
        return all(_values_agree(expected[k], observed[k], tolerance) for k in expected)
    if isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple)):
        if len(expected) != len(observed):
            return False
        return all(_values_agree(e, o, tolerance) for e, o in zip(expected, observed))
    return expected == observed


def run_checklist(
    context: CheckContext, items: list[dict[str, Any]], *, default_tolerance: float
) -> list[CheckResult]:
    """Recompute every registered claim and compare it with its expectation."""
    results: list[CheckResult] = []
    for item in items:
        check_name = item["check"]
        observer = CHECK_REGISTRY.get(check_name)
        if observer is None:
            raise KeyError(f"unknown check: {check_name}")
        params = item.get("params") or {}
        expected = item.get("expected")
        tolerance = float(item.get("tolerance", default_tolerance))
        try:
            observed = observer(context, params, expected)
        except FileNotFoundError as exc:
            results.append(
                CheckResult(
                    check_id=item["id"],
                    check=check_name,
                    status=STATUS_UNVERIFIABLE,
                    claim=item["claim"],
                    expected=expected,
                    observed=None,
                    detail=f"artifact missing: {exc}",
                )
            )
            continue
        status = (
            STATUS_MATCH
            if _values_agree(expected, observed, tolerance)
            else STATUS_MISMATCH
        )
        results.append(
            CheckResult(
                check_id=item["id"],
                check=check_name,
                status=status,
                claim=item["claim"],
                expected=expected,
                observed=observed,
                detail=item.get("note"),
            )
        )
    return results


@register_check("parent_manifest")
def _observe_parent_manifest(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    manifest_path = context.resolve(params["manifest_path"])
    issues = verify_evidence_manifest(
        manifest_path,
        project_root=context.project_root,
        expected_manifest_sha256=params["manifest_sha256"],
    )
    return {"issues": list(issues), "verified": not issues}


@register_check("json_field")
def _observe_json_field(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> Any:
    node = context.json_at(params["path"])
    for key in params["pointer"]:
        node = node[key] if isinstance(node, dict) else node[int(key)]
    return node


@register_check("c2_rejection")
def _observe_c2_rejection(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    group = context.c2_json("laplace_diagnostics.json")["methods"][params["method"]][
        params["domain"]
    ]
    return {
        "n": group["n"],
        "rejected": group["rejected"],
        "reasons": dict(group["rejection_reasons"]),
    }


@register_check("c2_diagnostic")
def _observe_c2_diagnostic(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> Any:
    group = context.c2_json("laplace_diagnostics.json")["methods"][params["method"]][
        params["domain"]
    ]
    return group[params["field"]].get(params["stat"])


@register_check("c2_rejection_reason_absent")
def _observe_c2_reason_absent(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, int]:
    methods = context.c2_json("laplace_diagnostics.json")["methods"]
    totals = {reason: 0 for reason in params["reasons"]}
    for by_domain in methods.values():
        for group in by_domain.values():
            for reason, count in (group.get("rejection_reasons") or {}).items():
                if reason in totals:
                    totals[reason] += int(count)
    return totals


@register_check("c2_hessian_probes")
def _observe_c2_hessian(
    context: CheckContext, _params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    probes = context.c2_json("laplace_diagnostics.json")["complete_hessian"]
    unavailable = [p for p in probes if p.get("status") != "computed"]
    return {
        "n_probes": len(probes),
        "unavailable": len(unavailable),
        "unavailable_ids": sorted(p["mixture_id"] for p in unavailable),
        "errors": sorted({str(p.get("error")) for p in unavailable}),
    }


@register_check("c2_s1_replay")
def _observe_c2_s1_replay(
    context: CheckContext, _params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    replay = context.c2_json("laplace_diagnostics.json")["s1_replay"]
    return {"passed": bool(replay["passed"]), "n_probes": int(replay["n_probes"])}


@register_check("c2_coverage_band")
def _observe_c2_coverage(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    bands = context.c2_json("coverage_report.json")["primary_bands"]
    for band in bands:
        matches = (
            band["method"] == params["method"]
            and band["domain"] == params["domain"]
            and band["component"] == params["component"]
            and abs(float(band["nominal_level"]) - float(params["nominal_level"])) < 1e-9
        )
        if not matches:
            continue
        n, rejected, covered = band["n"], band["rejected"], band["covered"]
        accepted = n - rejected
        return {
            "n": n,
            "rejected": rejected,
            "covered": covered,
            "unconditional": covered / n,
            "selection_conditional": covered / accepted if accepted else None,
            "within_acceptance_band": bool(band["within_acceptance_band"]),
        }
    raise KeyError(f"coverage band not found: {params}")


@register_check("c2_acceptance_band_pass_count")
def _observe_c2_band_passes(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    bands = context.c2_json("coverage_report.json")["primary_bands"]
    selected = [
        band
        for band in bands
        if band["method"] == params["method"] and band["component"] == params["component"]
    ]
    passed = [b for b in selected if b["within_acceptance_band"]]
    return {
        "n_bands": len(selected),
        "n_passed": len(passed),
        "passed": sorted(f"{b['domain']}@{b['nominal_level']}" for b in passed),
    }


@register_check("c2_interval_width_median")
def _observe_c2_widths(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> list[float]:
    rows = context.accepted_interval_rows(params["domain"], params["method"])
    component = params["component"]
    medians: list[float] = []
    for level in INTERVAL_LEVELS:
        widths = [
            float(row[f"{component}_upper_{level}"]) - float(row[f"{component}_lower_{level}"])
            for row in rows
            if row[f"{component}_upper_{level}"]
        ]
        medians.append(statistics.median(widths))
    return medians


@register_check("c2_interval_lower_min")
def _observe_c2_lower_min(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> float:
    component = params["component"]
    levels = params.get("levels") or list(INTERVAL_LEVELS)
    methods = params.get("methods") or ["M1", "M1b", "M2"]
    values: list[float] = []
    for method in methods:
        for row in context.accepted_interval_rows(params["domain"], method):
            for level in levels:
                cell = row[f"{component}_lower_{level}"]
                if cell:
                    values.append(float(cell))
    if not values:
        raise KeyError(f"no interval lower bounds for {params}")
    return min(values)


@register_check("d0_component_r2")
def _observe_d0_r2(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, float | None]:
    metrics = context.json_at(params["metrics_path"])
    evaluations = metrics["evaluations"]
    component = params["component"]
    return {
        split: (
            evaluations.get(split, {})
            .get("component_metrics", {})
            .get(component, {})
            .get("r2")
        )
        for split in params["splits"]
    }


@register_check("d0_bin_r2_range")
def _observe_d0_bin_range(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, float]:
    metrics = context.json_at(params["metrics_path"])
    bins = metrics["evaluations"][params["split"]]["conditional_metrics"][
        params["conditional"]
    ]["bins"]
    values = [
        entry["component_metrics"][params["component"]]["r2"] for entry in bins.values()
    ]
    return {"n_bins": len(values), "min": min(values), "max": max(values)}


@register_check("config_equivalence")
def _observe_config_equivalence(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    left = context.json_at(params["left"])
    right = context.json_at(params["right"])
    fields = params["fields"]
    differing = [
        key for key in fields if left.get(key) != right.get(key)
    ]
    return {
        "equivalent": not differing,
        "differing_fields": differing,
        "left_feature_builder": left.get("feature_builder"),
        "right_feature_builder": right.get("feature_builder"),
    }


@register_check("b7_protocol_cell")
def _observe_b7_cell(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, float]:
    path = context.resolve(params["matrix_path"])
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    subset = [row for row in rows if row["protocol_id"] == params["protocol_id"]]
    if not subset:
        raise KeyError(f"protocol not in matrix: {params['protocol_id']}")
    ddof = int(params.get("ddof", 0))

    def _stats(column: str) -> tuple[float, float]:
        values = [float(row[column]) for row in subset]
        mean = statistics.mean(values)
        if len(values) - ddof <= 0:
            return mean, 0.0
        spread = statistics.pstdev(values) if ddof == 0 else statistics.stdev(values)
        return mean, spread

    test_mean, test_std = _stats("b7_test_o2_r2")
    ood_mean, ood_std = _stats("b7_extrapolation_o2_r2")
    return {
        "n_rows": len(subset),
        "test_mean": test_mean,
        "test_std": test_std,
        "ood_mean": ood_mean,
        "ood_std": ood_std,
        "delta_test_vs_b1": statistics.mean(
            float(row["delta_o2_r2_test"]) for row in subset
        ),
        "delta_ood_vs_b1": statistics.mean(
            float(row["delta_o2_r2_extrapolation"]) for row in subset
        ),
    }


@register_check("ood_hash_sharing")
def _observe_ood_hash_sharing(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    payload = context.json_at(params["split_hashes_path"])
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in payload["derived_splits"]:
        grouped[row["protocol_id"]].append(row["ood_set_hash"])
    return {
        protocol: {
            "n_seeds": len(hashes),
            "n_unique_ood_sets": len(set(hashes)),
        }
        for protocol, hashes in sorted(grouped.items())
    }


@register_check("mrs2_arm_summary")
def _observe_mrs2_arm(
    context: CheckContext, params: dict[str, Any], _expected: Any
) -> dict[str, Any]:
    verdict = context.json_at(params["verdict_path"])
    arm = verdict["arm_summaries"][params["arm"]]
    return {
        "min_joint_rank": arm["min_joint_rank"],
        "max_joint_rank": arm["max_joint_rank"],
        "median_p90_o2_percent": arm["median_p90_o2_percent"],
    }


def summarize(results: list[CheckResult]) -> dict[str, Any]:
    """Aggregate check outcomes into a freeze-ready summary block."""
    counts: dict[str, int] = defaultdict(int)
    for result in results:
        counts[result.status] += 1
    return {
        "n_checks": len(results),
        "n_match": counts[STATUS_MATCH],
        "n_mismatch": counts[STATUS_MISMATCH],
        "n_unverifiable": counts[STATUS_UNVERIFIABLE],
        "mismatched_ids": [
            r.check_id for r in results if r.status == STATUS_MISMATCH
        ],
        "unverifiable_ids": [
            r.check_id for r in results if r.status == STATUS_UNVERIFIABLE
        ],
    }


__all__ = [
    "CHECK_REGISTRY",
    "CheckContext",
    "CheckResult",
    "INTERVAL_LEVELS",
    "STATUS_MATCH",
    "STATUS_MISMATCH",
    "STATUS_UNVERIFIABLE",
    "register_check",
    "run_checklist",
    "summarize",
]
