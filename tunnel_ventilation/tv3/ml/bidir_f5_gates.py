"""F5 preregistered amplitude gates (frozen at F4; do not retune after seeing scores)."""
from __future__ import annotations

import math
from typing import Any

CRITERION_C_S_LINE_B1 = "s_line_b1"
CRITERION_C_IN_DOMAIN_A1 = "in_domain_a1_v_path_zero"
VALID_CRITERION_C_ANCHORS = (CRITERION_C_S_LINE_B1, CRITERION_C_IN_DOMAIN_A1)


def evaluate_f5_gates(
    arm_metrics: dict[str, dict[str, Any]],
    *,
    gates: dict[str, Any],
    primary_split: str = "extrapolation",
    head: str = "b1_ridge",
) -> dict[str, Any]:
    """Apply preregistered F5 amplitude gates on one head.

    Criterion c:
    - ``s_line_b1`` (narrow default): A3 zero-anchor O₂ MAE vs registered S-line / frozen B1.
    - ``in_domain_a1_v_path_zero`` (wide): A3 zero-anchor vs **in-domain A1** zero-anchor
      (cross-distribution B1 is invalid under the widened composition domain).
    Criterion d is evaluated separately by the protocol runner once S-Y/S-L selectors exist.
    """
    required = ("A1", "A2", "A3")
    for arm_id in required:
        key = f"{arm_id}:{head}"
        if key not in arm_metrics:
            raise KeyError(f"missing arm metrics for gate evaluation: {key}")

    criterion_c_anchor = str(gates.get("criterion_c_anchor", CRITERION_C_S_LINE_B1))
    if criterion_c_anchor not in VALID_CRITERION_C_ANCHORS:
        raise ValueError(
            f"criterion_c_anchor must be one of {list(VALID_CRITERION_C_ANCHORS)}, "
            f"got {criterion_c_anchor!r}"
        )

    def o2_mae(arm_id: str, split: str) -> float:
        payload = arm_metrics[f"{arm_id}:{head}"]
        return float(payload["evaluations"][split]["component_metrics"]["x_O2"]["mae"])

    def o2_r2(arm_id: str, split: str) -> float:
        payload = arm_metrics[f"{arm_id}:{head}"]
        return float(payload["evaluations"][split]["component_metrics"]["x_O2"]["r2"])

    a1_ood = o2_mae("A1", primary_split)
    a2_ood = o2_mae("A2", primary_split)
    a3_ood = o2_mae("A3", primary_split)
    a3_minus_a1 = a1_ood - a3_ood  # positive => A3 better
    a3_minus_a2 = a3_ood - a2_ood  # should be ≤ residual gate

    zero_a3 = arm_metrics[f"A3:{head}"]["zero_anchor_metrics"].get("test")
    zero_a1 = arm_metrics[f"A1:{head}"]["zero_anchor_metrics"].get("test")
    if criterion_c_anchor == CRITERION_C_IN_DOMAIN_A1:
        if zero_a3 is None or zero_a1 is None:
            zero_delta = float("nan")
            zero_pass = False
            c_ref = None
        else:
            c_ref = float(zero_a1["o2_mae"])
            zero_delta = float(zero_a3["o2_mae"] - c_ref)
            zero_pass = bool(zero_delta <= gates["v_path_zero_anchor_delta_mae_max_vol_percent"])
        c_check_key = "c_zero_anchor_noninferior_vs_in_domain_a1"
        c_detail = {
            "criterion_c_anchor": criterion_c_anchor,
            "zero_a3": zero_a3,
            "zero_a1": zero_a1,
            "in_domain_a1_reference_o2_mae": c_ref,
        }
    else:
        if "s_line_b1_reference_o2_mae" not in gates:
            raise KeyError("gates must include s_line_b1_reference_o2_mae for criterion c")
        s_line_ref_raw = gates["s_line_b1_reference_o2_mae"]
        if s_line_ref_raw is None:
            raise ValueError(
                "s_line_b1_reference_o2_mae must be set to a finite float from S-line / frozen B1"
            )
        s_line_ref = float(s_line_ref_raw)
        if not math.isfinite(s_line_ref):
            raise ValueError("s_line_b1_reference_o2_mae must be finite")
        if zero_a3 is None:
            zero_delta = float("nan")
            zero_pass = False
        else:
            zero_delta = float(zero_a3["o2_mae"] - s_line_ref)
            zero_pass = bool(zero_delta <= gates["v_path_zero_anchor_delta_mae_max_vol_percent"])
        c_check_key = "c_zero_anchor_noninferior_vs_s_line_b1"
        c_detail = {
            "criterion_c_anchor": criterion_c_anchor,
            "zero_a3": zero_a3,
            "s_line_b1_reference_o2_mae": s_line_ref,
        }

    ss = arm_metrics[f"A3:{head}"].get("sound_speed_audit") or {}
    pair_bias = ss.get("pair_sound_speed_mean_abs_seq_bias_m_per_s")
    ab_bias = ss.get("ab_sound_speed_mean_abs_seq_bias_m_per_s")
    # Backward-compatible keys from older metrics payloads.
    if pair_bias is None:
        pair_bias = ss.get("pair_sound_speed_bias_m_per_s")
    if ab_bias is None:
        ab_bias = ss.get("ab_sound_speed_bias_m_per_s")
    rec_p95 = ss.get("reciprocity_residual_p95_of_seq_p95_s")
    if rec_p95 is None:
        rec_p95 = ss.get("reciprocity_residual_p95_median_s")
    e_pass = (
        pair_bias is not None
        and ab_bias is not None
        and float(pair_bias) < float(ab_bias)
        and rec_p95 is not None
        and float(rec_p95) <= gates.get("reciprocity_p95_max_s", 1.0e-7)
    )

    checks = {
        "a_a3_beats_a1_ood": {
            "value": a3_minus_a1,
            "threshold_min": gates["a3_minus_a1_o2_mae_min_vol_percent"],
            "passed": bool(a3_minus_a1 >= gates["a3_minus_a1_o2_mae_min_vol_percent"]),
            "detail": {"a1_o2_mae": a1_ood, "a3_o2_mae": a3_ood, "split": primary_split},
        },
        "b_a3_near_a2_ood": {
            "value": a3_minus_a2,
            "threshold_max": gates["a3_minus_a2_o2_mae_max_vol_percent"],
            "passed": bool(a3_minus_a2 <= gates["a3_minus_a2_o2_mae_max_vol_percent"]),
            "detail": {"a2_o2_mae": a2_ood, "a3_o2_mae": a3_ood, "split": primary_split},
        },
        c_check_key: {
            "value": zero_delta,
            "threshold_max": gates["v_path_zero_anchor_delta_mae_max_vol_percent"],
            "passed": zero_pass,
            "detail": c_detail,
        },
        "e_sound_speed_bias_and_reciprocity": {
            "passed": bool(e_pass),
            "detail": {
                "pair_mean_abs_seq_bias": pair_bias,
                "ab_mean_abs_seq_bias": ab_bias,
                "reciprocity_p95_of_seq_p95": rec_p95,
            },
        },
    }
    all_core = all(item["passed"] for item in checks.values())
    return {
        "head": head,
        "primary_split": primary_split,
        "criterion_c_anchor": criterion_c_anchor,
        "checks": checks,
        "core_gates_passed": all_core,
        "a3_test_o2_r2": o2_r2("A3", "test"),
        "a1_test_o2_r2": o2_r2("A1", "test"),
    }


__all__ = [
    "CRITERION_C_IN_DOMAIN_A1",
    "CRITERION_C_S_LINE_B1",
    "VALID_CRITERION_C_ANCHORS",
    "evaluate_f5_gates",
]
