"""MEI-1 forward-envelope and Jacobian-direction audit for MRS-EI.

Does not train networks or generate waveforms. Perturbations wrap the shared
MRS-1 ``relaxation_spectrum``; formulas are not copied into solvers.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tv3.audit.identifiability_v3_mrs import (
    DERIVATIVE_PARAMETERS,
    MrsPoint,
    fisher_rank_crb,
    observation_noise_std,
    single_nuisance_equivalent_o2,
)
from tv3.audit.mrs_ei_registry import build_narrow_points, load_json, sha256_file
from tv3.sim.generation.tunnel_ventilation.relaxation_spectrum import relaxation_spectrum

SpectrumFn = Callable[[MrsPoint, np.ndarray], dict[str, Any]]

_TV3_ROOT = Path(__file__).resolve().parents[2]

_UNREPRESENTED_BLOCKING_IDS = (
    "F2_h2o_relaxation_params",
    "F3_coupled_relaxation",
    "F4_diffraction_near_field",
    "F5_transducer_response",
)


@dataclass(frozen=True)
class EnvelopeSpec:
    family_id: str
    kind: str
    c_eq_relative_correction: float = 0.0
    o2_n2_cross_mix: float = 0.0
    o2_strength_scale: float = 1.0
    diffraction_amp: float = 0.0
    frequency_floor_hz: float = 15000.0
    alpha_ripple_amp: float = 0.0
    synthetic_direction: str | None = None
    relative_observation_rms: float = 0.0
    orthogonal_seed_amp: float = 0.002
    registry_family: str | None = None
    notes: str = ""


def principal_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two vectors in degrees, in [0, 90]."""
    a = np.asarray(u, dtype=np.float64).reshape(-1)
    b = np.asarray(v, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 90.0
    cos = float(np.dot(a, b) / (na * nb))
    cos = min(1.0, max(-1.0, cos))
    ang = math.degrees(math.acos(abs(cos)))
    return float(ang)


def spearman_rank_corr(ranks_a: Sequence[float], ranks_b: Sequence[float]) -> float:
    """Spearman correlation for two rankings over the same items (supports ties)."""
    a = np.asarray(ranks_a, dtype=np.float64)
    b = np.asarray(ranks_b, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def point_key(point: MrsPoint) -> tuple[float, float, float, float, float, float]:
    return (
        round(point.co2_percent, 6),
        round(point.o2_percent, 6),
        round(point.t_c, 6),
        round(point.path_length_m, 6),
        round(point.h_rh, 6),
        round(point.p_mpa, 6),
    )


def baseline_spectrum(point: MrsPoint, f_hz: np.ndarray) -> dict[str, Any]:
    point.validate()
    return relaxation_spectrum(
        point.co2_percent,
        point.o2_percent,
        point.n2_percent,
        point.t_c,
        point.p_mpa,
        point.h_rh,
        f_hz,
    )


def apply_envelope(out: dict[str, Any], point: MrsPoint, spec: EnvelopeSpec) -> dict[str, Any]:
    """Post-process MRS-1 spectrum according to a registered/proxy envelope."""
    result = dict(out)
    f = np.asarray(result["f_hz"], dtype=np.float64)
    c_eq = float(result["c_eq"])
    c_f = np.asarray(result["c_f"], dtype=np.float64).copy()
    alpha_f = np.asarray(result["alpha_f"], dtype=np.float64).copy()
    processes = {k: dict(v) for k, v in (result.get("processes") or {}).items()}

    rebuild = False
    if abs(spec.o2_n2_cross_mix) > 0.0 and "o2" in processes and "n2" in processes:
        g = float(spec.o2_n2_cross_mix)
        d_o2 = float(processes["o2"]["delta_c_m_per_s"])
        d_n2 = float(processes["n2"]["delta_c_m_per_s"])
        processes["o2"]["delta_c_m_per_s"] = (1.0 - g) * d_o2 + g * d_n2
        processes["n2"]["delta_c_m_per_s"] = (1.0 - g) * d_n2 + g * d_o2
        processes["o2"]["alpha_lambda_max"] = math.pi * (
            float(processes["o2"]["delta_c_m_per_s"]) / max(c_eq, 1e-12)
        )
        processes["n2"]["alpha_lambda_max"] = math.pi * (
            float(processes["n2"]["delta_c_m_per_s"]) / max(c_eq, 1e-12)
        )
        rebuild = True

    if abs(spec.o2_strength_scale - 1.0) > 0.0 and "o2" in processes:
        s = float(spec.o2_strength_scale)
        processes["o2"]["delta_c_m_per_s"] = float(processes["o2"]["delta_c_m_per_s"]) * s
        processes["o2"]["alpha_lambda_max"] = float(processes["o2"]["alpha_lambda_max"]) * s
        rebuild = True

    if rebuild:
        disp = np.zeros_like(f)
        alpha_rel = np.zeros_like(f)
        for proc in processes.values():
            fr = float(proc["f_r_hz"])
            dcc = float(proc["delta_c_m_per_s"]) / max(c_eq, 1e-12)
            disp += dcc * (f**2) / (f**2 + fr**2)
            alpha_lambda = float(proc["alpha_lambda_max"]) * 2.0 * f * fr / (f**2 + fr**2)
            alpha_rel = alpha_rel + alpha_lambda * f / max(c_eq, 1e-12)
        alpha_classical = np.asarray(result["alpha_classical_f"], dtype=np.float64)
        c_f = c_eq * (1.0 + disp)
        alpha_f = np.maximum(0.0, alpha_classical + alpha_rel)
        result["processes"] = processes

    if abs(spec.c_eq_relative_correction) > 0.0:
        factor = 1.0 + float(spec.c_eq_relative_correction)
        c_eq = c_eq * factor
        c_f = c_f * factor
        result["c_eq"] = c_eq

    if abs(spec.diffraction_amp) > 0.0:
        floor = max(float(spec.frequency_floor_hz), 1.0)
        rh_scale = float(point.h_rh) / 50.0
        stretch = 1.0 + float(spec.diffraction_amp) * rh_scale * (floor / np.maximum(f, floor)) ** 2
        c_f = c_f / stretch

    if abs(spec.alpha_ripple_amp) > 0.0:
        ripple = 1.0 + float(spec.alpha_ripple_amp) * np.sin(
            2.0 * math.pi * np.log(np.maximum(f, 1.0) / 25000.0)
        )
        alpha_f = np.maximum(0.0, alpha_f * ripple)

    result["c_f"] = c_f
    result["alpha_f"] = alpha_f
    result["envelope_family_id"] = spec.family_id
    return result


def diffraction_seed_delta_tof(
    point: MrsPoint,
    *,
    f_hz: np.ndarray,
    c_f: np.ndarray,
    amp: float,
    frequency_floor_hz: float,
) -> np.ndarray:
    """RH-coupled low-frequency path stretch converted to TOF delta (seed only)."""
    f = np.asarray(f_hz, dtype=np.float64)
    c = np.asarray(c_f, dtype=np.float64)
    floor = max(float(frequency_floor_hz), 1.0)
    rh_scale = float(point.h_rh) / 50.0
    stretch = 1.0 + float(amp) * rh_scale * (floor / np.maximum(f, floor)) ** 2
    c_new = c / stretch
    return point.path_length_m * (1.0 / c_new - 1.0 / c)


def build_aligned_delta_tof(
    *,
    direction: str,
    j_o2: np.ndarray,
    y0: np.ndarray,
    relative_rms: float,
    seed_delta_tof: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build δy parallel / orthogonal / mixed to target O2 Jacobian."""
    j = np.asarray(j_o2, dtype=np.float64).reshape(-1)
    y = np.asarray(y0, dtype=np.float64).reshape(-1)
    seed = np.asarray(seed_delta_tof, dtype=np.float64).reshape(-1)
    if j.shape != y.shape or seed.shape != y.shape:
        raise ValueError("j_o2, y0, and seed_delta_tof must share shape")
    j_norm = float(np.linalg.norm(j))
    if j_norm <= 0.0:
        raise ValueError("o2 jacobian norm must be positive")
    j_unit = j / j_norm

    if direction == "parallel":
        b_unit = j_unit
    elif direction in ("orthogonal", "mixed"):
        orth = seed - j_unit * float(np.dot(seed, j_unit))
        orth_norm = float(np.linalg.norm(orth))
        if orth_norm <= 1e-30:
            fallback = np.linspace(-1.0, 1.0, num=j.size, dtype=np.float64)
            orth = fallback - j_unit * float(np.dot(fallback, j_unit))
            orth_norm = float(np.linalg.norm(orth))
        if orth_norm <= 1e-30:
            raise ValueError("failed to construct orthogonal seed")
        orth_unit = orth / orth_norm
        if direction == "orthogonal":
            b_unit = orth_unit
        else:
            mixed = j_unit + orth_unit
            mixed_norm = float(np.linalg.norm(mixed))
            if mixed_norm <= 1e-30:
                raise ValueError("failed to construct mixed direction")
            b_unit = mixed / mixed_norm
    else:
        raise ValueError(f"unsupported synthetic direction: {direction}")

    y_rms = float(np.sqrt(np.mean(y * y)))
    scale = float(relative_rms) * max(y_rms, 1e-30)
    delta = scale * b_unit
    meta = {
        "angle_to_o2_jacobian_deg": principal_angle_deg(delta, j),
        "relative_rms": float(relative_rms),
        "delta_rms": float(np.sqrt(np.mean(delta * delta))),
        "y_rms": y_rms,
    }
    return delta, meta


def precompute_synthetic_delta_c(
    point: MrsPoint,
    *,
    direction: str,
    relative_observation_rms: float,
    j_o2_baseline: np.ndarray,
    f_hz_baseline: Sequence[float],
    y0_baseline: np.ndarray,
    orthogonal_seed_amp: float = 0.002,
    frequency_floor_hz: float = 15000.0,
) -> dict[str, Any]:
    """Freeze Jacobian-aligned δc(f) on the nominal point (FD-safe additive profile)."""
    f_base = np.asarray(f_hz_baseline, dtype=np.float64)
    base_out = baseline_spectrum(point, f_base)
    c_base = np.asarray(base_out["c_f"], dtype=np.float64)
    seed = diffraction_seed_delta_tof(
        point,
        f_hz=f_base,
        c_f=c_base,
        amp=orthogonal_seed_amp,
        frequency_floor_hz=frequency_floor_hz,
    )
    delta_tof_base, align_meta = build_aligned_delta_tof(
        direction=direction,
        j_o2=j_o2_baseline,
        y0=y0_baseline,
        relative_rms=relative_observation_rms,
        seed_delta_tof=seed,
    )
    delta_c_base = -delta_tof_base * (c_base**2) / max(float(point.path_length_m), 1e-30)
    return {
        "f_hz_baseline": f_base,
        "delta_c_baseline": delta_c_base,
        "align_meta": align_meta,
        "direction": direction,
    }


def apply_frozen_delta_c(
    out: dict[str, Any],
    *,
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a nominal-point δc(f) profile to any (possibly FD-shifted) evaluation."""
    result = dict(out)
    f_query = np.asarray(result["f_hz"], dtype=np.float64)
    c_query = np.asarray(result["c_f"], dtype=np.float64).copy()
    f_base = np.asarray(frozen["f_hz_baseline"], dtype=np.float64)
    delta_c_base = np.asarray(frozen["delta_c_baseline"], dtype=np.float64)
    log_base = np.log(np.maximum(f_base, 1.0))
    log_query = np.log(np.maximum(f_query, 1.0))
    order = np.argsort(log_base)
    delta_c_query = np.interp(log_query, log_base[order], delta_c_base[order])
    c_query = c_query + delta_c_query
    if np.any(c_query <= 0.0):
        raise ValueError("synthetic bias produced non-positive c_f")
    result["c_f"] = c_query
    result["synthetic_bias"] = {
        "direction": frozen["direction"],
        **dict(frozen["align_meta"]),
    }
    result["envelope_family_id"] = f"S_{frozen['direction']}"
    return result


def make_spectrum_fn(
    spec: EnvelopeSpec,
    *,
    frozen_delta_c: Mapping[str, Any] | None = None,
) -> SpectrumFn:
    def _fn(point: MrsPoint, f_hz: np.ndarray) -> dict[str, Any]:
        base = baseline_spectrum(point, f_hz)
        if spec.synthetic_direction is None:
            return apply_envelope(base, point, spec)
        if frozen_delta_c is None:
            raise ValueError("synthetic bias requires frozen delta_c profile")
        return apply_frozen_delta_c(base, frozen=frozen_delta_c)

    return _fn


def observe_tof_alpha_envelope(
    point: MrsPoint,
    *,
    f_hz: Sequence[float],
    fixed_delay_s: float,
    spectrum_fn: SpectrumFn,
) -> dict[str, np.ndarray]:
    if not math.isfinite(fixed_delay_s) or fixed_delay_s < 0.0:
        raise ValueError("fixed_delay_s must be finite and >= 0")
    f = np.asarray(f_hz, dtype=np.float64)
    out = spectrum_fn(point, f)
    c = np.asarray(out["c_f"], dtype=np.float64)
    alpha = np.asarray(out["alpha_f"], dtype=np.float64)
    tof = point.path_length_m / c + fixed_delay_s
    return {"f_hz": f, "c_f": c, "alpha_f": alpha, "tof_s": tof}


def _shift_point(point: MrsPoint, parameter: str, delta: float) -> MrsPoint:
    kwargs = {
        "co2_percent": point.co2_percent,
        "o2_percent": point.o2_percent,
        "t_c": point.t_c,
        "path_length_m": point.path_length_m,
        "h_rh": point.h_rh,
        "p_mpa": point.p_mpa,
    }
    kwargs[parameter] = float(kwargs[parameter]) + float(delta)
    return MrsPoint(**kwargs)


def observation_vector_cfreq(
    point: MrsPoint,
    *,
    f_hz: Sequence[float],
    fixed_delay_s: float,
    spectrum_fn: SpectrumFn,
) -> tuple[np.ndarray, list[str]]:
    obs = observe_tof_alpha_envelope(
        point, f_hz=f_hz, fixed_delay_s=fixed_delay_s, spectrum_fn=spectrum_fn
    )
    values = [float(obs["tof_s"][i]) for i in range(len(obs["f_hz"]))]
    labels = [f"base:tof@{int(fk)}Hz" for fk in obs["f_hz"]]
    return np.asarray(values, dtype=np.float64), labels


def local_cfreq_jacobian(
    point: MrsPoint,
    *,
    f_hz: Sequence[float],
    parameter_steps: Mapping[str, float],
    parameter_bounds: Mapping[str, Sequence[float]],
    fixed_delay_s: float,
    spectrum_fn: SpectrumFn,
    max_relative_step_disagreement: float = 0.01,
) -> dict[str, Any]:
    y0, labels = observation_vector_cfreq(
        point, f_hz=f_hz, fixed_delay_s=fixed_delay_s, spectrum_fn=spectrum_fn
    )
    jac = np.zeros((len(labels), len(DERIVATIVE_PARAMETERS)), dtype=np.float64)
    meta: dict[str, Any] = {}
    for col, parameter in enumerate(DERIVATIVE_PARAMETERS):
        step = float(parameter_steps[parameter])
        lo, hi = map(float, parameter_bounds[parameter])
        current = float(getattr(point, parameter))
        plus = current + step
        minus = current - step
        has_plus = lo <= plus <= hi
        has_minus = lo <= minus <= hi
        if has_plus and has_minus:
            yp, _ = observation_vector_cfreq(
                _shift_point(point, parameter, step),
                f_hz=f_hz,
                fixed_delay_s=fixed_delay_s,
                spectrum_fn=spectrum_fn,
            )
            ym, _ = observation_vector_cfreq(
                _shift_point(point, parameter, -step),
                f_hz=f_hz,
                fixed_delay_s=fixed_delay_s,
                spectrum_fn=spectrum_fn,
            )
            d = (yp - ym) / (2.0 * step)
            scheme = "central"
        elif has_plus:
            yp, _ = observation_vector_cfreq(
                _shift_point(point, parameter, step),
                f_hz=f_hz,
                fixed_delay_s=fixed_delay_s,
                spectrum_fn=spectrum_fn,
            )
            d = (yp - y0) / step
            scheme = "forward"
        elif has_minus:
            ym, _ = observation_vector_cfreq(
                _shift_point(point, parameter, -step),
                f_hz=f_hz,
                fixed_delay_s=fixed_delay_s,
                spectrum_fn=spectrum_fn,
            )
            d = (y0 - ym) / step
            scheme = "backward"
        else:
            raise ValueError(f"no FD stencil for {parameter}")

        half = step * 0.5
        if lo <= current + half <= hi and lo <= current - half <= hi:
            yp_h, _ = observation_vector_cfreq(
                _shift_point(point, parameter, half),
                f_hz=f_hz,
                fixed_delay_s=fixed_delay_s,
                spectrum_fn=spectrum_fn,
            )
            ym_h, _ = observation_vector_cfreq(
                _shift_point(point, parameter, -half),
                f_hz=f_hz,
                fixed_delay_s=fixed_delay_s,
                spectrum_fn=spectrum_fn,
            )
            d_half = (yp_h - ym_h) / (2.0 * half)
            denom = max(float(np.linalg.norm(d)), 1e-30)
            disagreement = float(np.linalg.norm(d_half - d) / denom)
        else:
            disagreement = 0.0
        jac[:, col] = d
        meta[parameter] = {
            "scheme": scheme,
            "step": step,
            "disagreement": disagreement,
            "stable": disagreement <= max_relative_step_disagreement,
        }
    return {
        "jacobian": jac,
        "labels": labels,
        "y0": y0,
        "parameter_meta": meta,
        "all_stable": all(m["stable"] for m in meta.values()),
    }


def evaluate_point_design(
    point: MrsPoint,
    *,
    f_hz: Sequence[float],
    spectrum_fn: SpectrumFn,
    parameter_steps: Mapping[str, float],
    parameter_bounds: Mapping[str, Sequence[float]],
    prior_std: Mapping[str, float],
    jitter_std_s: float,
    relative_amp_std: float,
    fixed_delay_s: float,
    window_width_percent: float = 0.8,
    max_relative_step_disagreement: float = 0.01,
) -> dict[str, Any]:
    loc = local_cfreq_jacobian(
        point,
        f_hz=f_hz,
        parameter_steps=parameter_steps,
        parameter_bounds=parameter_bounds,
        fixed_delay_s=fixed_delay_s,
        spectrum_fn=spectrum_fn,
        max_relative_step_disagreement=max_relative_step_disagreement,
    )
    sigmas = observation_noise_std(
        loc["labels"],
        point=point,
        jitter_std_s=jitter_std_s,
        relative_amp_std=relative_amp_std,
    )
    fish = fisher_rank_crb(
        loc["jacobian"],
        row_sigmas=sigmas,
        parameter_steps=parameter_steps,
        prior_std=prior_std,
    )
    nuisance = single_nuisance_equivalent_o2(
        loc["jacobian"],
        row_sigmas=sigmas,
        parameter_steps=parameter_steps,
        prior_std=prior_std,
        window_width_percent=window_width_percent,
    )
    per = nuisance["per_nuisance"]
    bottleneck = max(per.items(), key=lambda kv: float(kv[1]["fraction_of_window"]))[0]
    return {
        "p90_o2_percent": float(fish["p90_o2_percent"]),
        "joint_rank": int(fish["joint_rank"]),
        "crlb_o2_std_percent": float(fish["crlb_o2_std_percent"]),
        "fisher_aug_invertible": bool(fish["fisher_aug_invertible"]),
        "all_stable": bool(loc["all_stable"]),
        "o2_jacobian": loc["jacobian"][:, 0].copy(),
        "y0": loc["y0"].copy(),
        "bottleneck": bottleneck,
        "worst_nuisance_fraction": float(nuisance["worst_fraction_of_window"]),
        "labels": loc["labels"],
        "row_sigmas": sigmas,
    }


def enumerate_k4_designs(pool_hz: Sequence[float], baseline: Sequence[float]) -> list[tuple[float, ...]]:
    pool = sorted(float(x) for x in pool_hz)
    designs = [tuple(sorted(c)) for c in itertools.combinations(pool, 4)]
    base = tuple(sorted(float(x) for x in baseline))
    if base not in designs:
        designs.append(base)
    designs = [base] + sorted({d for d in designs if d != base})
    return designs


def design_id(freqs: Sequence[float]) -> str:
    return "K4[" + ",".join(str(int(f / 1000)) for f in freqs) + "k]"


def select_audit_points(
    design_space: dict[str, Any],
    *,
    mode: str = "full_narrow_grid",
    stride: int = 1,
) -> list[tuple[str, MrsPoint]]:
    labeled = build_narrow_points(design_space)
    if mode == "full_narrow_grid":
        selected = list(labeled)
    elif mode == "stride_plus_holdouts":
        selected = list(labeled[:: max(int(stride), 1)])
    else:
        raise ValueError(f"unsupported point_sampling.mode: {mode}")

    holdout_defs = (design_space.get("holdout_conditions") or {}).get("definitions") or {}
    windows = {
        w["id"]: float(w["center_percent"])
        for w in design_space["target_direction"]["narrow_windows"]
    }
    existing = {point_key(pt) for _, pt in selected}
    for hid, spec in holdout_defs.items():
        o2 = windows[str(spec["o2_window_id"])]
        pt = MrsPoint(
            float(spec["co2_percent"]),
            o2,
            float(spec["t_c"]),
            float(spec["path_length_m"]),
            float(spec["h_rh"]),
            float(spec["p_mpa"]),
        )
        key = point_key(pt)
        if key not in existing:
            selected.append((f"holdout:{hid}", pt))
            existing.add(key)
    return selected


def summarize_design(
    point_results: list[dict[str, Any]],
) -> dict[str, Any]:
    p90s = [float(r["p90_o2_percent"]) for r in point_results]
    ranks = [int(r["joint_rank"]) for r in point_results]
    bottlenecks = [str(r["bottleneck"]) for r in point_results]
    bot_counts: dict[str, int] = {}
    for b in bottlenecks:
        bot_counts[b] = bot_counts.get(b, 0) + 1
    mode_bot = max(bot_counts.items(), key=lambda kv: kv[1])[0]
    return {
        "n_points": len(point_results),
        "max_p90_o2_percent": float(np.max(p90s)),
        "median_p90_o2_percent": float(np.median(p90s)),
        "min_joint_rank": int(min(ranks)),
        "max_joint_rank": int(max(ranks)),
        "mode_bottleneck": mode_bot,
        "bottleneck_counts": bot_counts,
        "all_invertible": all(bool(r["fisher_aug_invertible"]) for r in point_results),
        "all_stable": all(bool(r["all_stable"]) for r in point_results),
    }


def rank_designs(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "max_p90_o2_percent",
    delta_num: float = 0.0,
) -> list[dict[str, Any]]:
    """Rank designs with ``delta_num`` relative equivalence classes (ties)."""
    rows = [
        {"design_id": did, "metric": float(summary[metric]), **dict(summary)}
        for did, summary in summaries.items()
    ]
    rows.sort(key=lambda r: (r["metric"], r["design_id"]))
    if not rows:
        return rows

    best = float(rows[0]["metric"])
    worst = float(rows[-1]["metric"])
    span_rel = (worst - best) / max(abs(best), 1e-30)
    group_start_metric = best
    group_rank = 0
    group_members: list[str] = []
    for row in rows:
        m = float(row["metric"])
        # Relative to current group start: within delta_num => same equivalence class.
        if (m - group_start_metric) / max(abs(group_start_metric), 1e-30) > float(delta_num):
            group_rank += 1
            group_start_metric = m
            group_members = []
        group_members.append(str(row["design_id"]))
        row["rank"] = group_rank
        row["equivalence_group_start_metric"] = group_start_metric
        row["raw_order"] = None  # filled below

    for i, row in enumerate(rows):
        row["raw_order"] = i

    n_levels = int(rows[-1]["rank"]) + 1 if rows else 0
    for row in rows:
        row["ranking_span_relative"] = float(span_rel)
        row["distinguishable_rank_levels"] = n_levels
        row["ranking_resolvable"] = bool(span_rel > float(delta_num))
    return rows


def angle_summary(angles_deg: Sequence[float], quantiles: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(list(angles_deg), dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "max_deg": None,
            "mean_deg": None,
            "quantiles_deg": {},
        }
    qmap = {
        f"p{int(round(100 * float(q)))}": float(np.quantile(arr, float(q)))
        for q in quantiles
    }
    return {
        "n": int(arr.size),
        "max_deg": float(np.max(arr)),
        "mean_deg": float(np.mean(arr)),
        "quantiles_deg": qmap,
    }


def parse_family_specs(mei1_config: dict[str, Any]) -> list[EnvelopeSpec]:
    specs: list[EnvelopeSpec] = []
    for raw in mei1_config["families"]:
        direction = raw.get("direction")
        specs.append(
            EnvelopeSpec(
                family_id=str(raw["id"]),
                kind=str(raw["kind"]),
                c_eq_relative_correction=float(raw.get("c_eq_relative_correction", 0.0)),
                o2_n2_cross_mix=float(raw.get("o2_n2_cross_mix", 0.0)),
                o2_strength_scale=float(raw.get("o2_strength_scale", 1.0)),
                diffraction_amp=float(raw.get("diffraction_amp", 0.0)),
                frequency_floor_hz=float(raw.get("frequency_floor_hz", 15000.0)),
                alpha_ripple_amp=float(raw.get("alpha_ripple_amp", 0.0)),
                synthetic_direction=str(direction) if direction else None,
                relative_observation_rms=float(raw.get("relative_observation_rms", 0.0)),
                orthogonal_seed_amp=float(raw.get("orthogonal_seed_amp", 0.002)),
                registry_family=raw.get("registry_family"),
                notes=str(raw.get("notes") or ""),
            )
        )
    return specs


def collect_unrepresented_blocking(model_registry: Mapping[str, Any]) -> list[str]:
    fam_by_id = {f["id"]: f for f in model_registry["model_families"]}
    blocking: list[str] = []
    for fid in _UNREPRESENTED_BLOCKING_IDS:
        status = fam_by_id.get(fid, {}).get("status")
        if status == "not_represented":
            blocking.append(fid)
    return blocking


def decide_mei1_verdict(
    *,
    issues: Sequence[str],
    flip_events: Sequence[Mapping[str, Any]],
    unrepresented_blocking: Sequence[str],
    ranking_resolvable: bool,
) -> dict[str, Any]:
    """Gate MEI-1 pass. Proxies/S_* never clear not_represented families."""
    blockers: list[str] = []
    if unrepresented_blocking:
        blockers.append(
            "unrepresented_families_without_flip_proof:"
            + ",".join(unrepresented_blocking)
        )
    if not ranking_resolvable:
        blockers.append("design_ranking_not_resolvable_within_delta_num")
    if flip_events:
        blockers.append("family_stability_flip_events")

    if issues:
        return {
            "verdict": "mei1_audit_failed",
            "allowed_next_stage": None,
            "passed": False,
            "blockers": list(issues) + blockers,
        }
    if blockers:
        return {
            "verdict": "mei1_inconclusive_forward_model",
            "allowed_next_stage": None,
            "passed": False,
            "blockers": blockers,
        }
    return {
        "verdict": "mei1_forward_envelope_supported",
        "allowed_next_stage": "MEI-2_robust_design",
        "passed": True,
        "blockers": [],
    }


def run_mei1_audit(
    *,
    project_root: Path | None = None,
    config_dir: Path | None = None,
    mei1_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else _TV3_ROOT
    cfg_dir = Path(config_dir) if config_dir is not None else root / "configs" / "tv3_mrs_ei"
    mei1 = mei1_config or load_json(cfg_dir / "mei1_forward_envelope.json")
    design = load_json(cfg_dir / "design_space.json")
    metric = load_json(cfg_dir / "metric_registry.json")
    model = load_json(cfg_dir / "model_family_registry.json")
    stage = load_json(cfg_dir / "stage_status.json")

    issues: list[str] = []
    prereq = mei1["mei0_prerequisite"]
    if (stage.get("mei0") or {}).get("verdict") != prereq["expected_verdict"]:
        issues.append(
            f"MEI-0 prerequisite failed: expected {prereq['expected_verdict']}, "
            f"got {(stage.get('mei0') or {}).get('verdict')}"
        )

    delta_num = float((stage.get("mei0") or {}).get("delta_num") or metric["delta_num"]["frozen_value"])
    gates = mei1["stability_gates"]
    inert_ids = set(gates.get("inert_family_ids") or [])
    angle_quantiles = [float(q) for q in gates.get("principal_angle_quantiles") or [0.5, 0.9, 0.95]]
    angle_gate_stat = str(gates.get("principal_angle_gate_stat") or "max_deg")

    fam_by_id = {f["id"]: f for f in model["model_families"]}
    if fam_by_id.get("F2_h2o_relaxation_params", {}).get("status") != "not_represented":
        issues.append("F2_h2o_relaxation_params must remain not_represented at MEI-1")

    comsol_status = "unavailable"
    comsol_cfg = mei1["comsol_holdout"]
    for pattern in comsol_cfg.get("search_globs") or []:
        matches = list(root.glob(pattern))
        if matches:
            comsol_status = "found_but_not_auto_ingested"
            break

    obs = design["observation_baselines"]["registered_mrs2"]
    num = metric["numerical_protocol"]
    eval_kwargs = {
        "parameter_steps": {k: float(v) for k, v in num["finite_difference_steps"].items()},
        "parameter_bounds": {k: list(map(float, v)) for k, v in num["parameter_bounds"].items()},
        "prior_std": {k: float(v) for k, v in obs["prior_std"].items()},
        "jitter_std_s": float(obs["jitter_std_s"]),
        "relative_amp_std": float(obs["relative_amp_std"]),
        "fixed_delay_s": float(design["observation_baselines"]["fixed_delay_s"]),
        "window_width_percent": 0.8,
        "max_relative_step_disagreement": float(num["max_relative_step_disagreement"]),
    }

    sampling = mei1["point_sampling"]
    labeled_points = select_audit_points(
        design,
        mode=str(sampling.get("mode") or "full_narrow_grid"),
        stride=int(sampling.get("stride") or 1),
    )
    points = [pt for _, pt in labeled_points]

    pool = design["frequency_band"]["candidate_pool_hz"]
    baseline = tuple(float(x) for x in design["frequency_band"]["baseline_k4_hz"])
    designs = enumerate_k4_designs(pool, baseline)
    baseline_id = design_id(baseline)

    family_specs = parse_family_specs(mei1)

    # F0 baseline context for Jacobian-aligned synthetic bias and angle gates.
    f0_spec = next(s for s in family_specs if s.family_id == "F0_mrs1_baseline")
    f0_spectrum = make_spectrum_fn(f0_spec)
    f0_baseline_rows: list[dict[str, Any]] = []
    synthetic_context: dict[tuple[float, ...], dict[str, Any]] = {}
    for point in points:
        row = evaluate_point_design(
            point,
            f_hz=baseline,
            spectrum_fn=f0_spectrum,
            **eval_kwargs,
        )
        f0_baseline_rows.append(row)
        synthetic_context[point_key(point)] = {
            "o2_jacobian": row["o2_jacobian"],
            "y0": row["y0"],
        }
    f0_o2_jacs = [r["o2_jacobian"] for r in f0_baseline_rows]

    family_reports: dict[str, Any] = {}
    f0_ranking_ids: list[str] | None = None
    f0_rank_by_id: dict[str, float] = {}
    f0_top_class: set[str] | None = None
    f0_baseline_summary = None
    f0_ranking_meta: dict[str, Any] = {}

    for spec in family_specs:
        # Non-synthetic families share one spectrum_fn; synthetic families freeze
        # δc(f) per nominal grid point so FD shifts remain well-defined.
        shared_fn = None
        if spec.synthetic_direction is None:
            shared_fn = make_spectrum_fn(spec)

        design_summaries: dict[str, Any] = {}
        point_rows_baseline: list[dict[str, Any]] = []
        frozen_by_point: dict[tuple[float, ...], dict[str, Any]] = {}

        for point in points:
            if shared_fn is not None:
                spectrum_fn = shared_fn
            else:
                ctx = synthetic_context[point_key(point)]
                frozen = precompute_synthetic_delta_c(
                    point,
                    direction=str(spec.synthetic_direction),
                    relative_observation_rms=float(spec.relative_observation_rms),
                    j_o2_baseline=np.asarray(ctx["o2_jacobian"], dtype=np.float64),
                    f_hz_baseline=baseline,
                    y0_baseline=np.asarray(ctx["y0"], dtype=np.float64),
                    orthogonal_seed_amp=float(spec.orthogonal_seed_amp),
                    frequency_floor_hz=float(spec.frequency_floor_hz),
                )
                frozen_by_point[point_key(point)] = frozen
                spectrum_fn = make_spectrum_fn(spec, frozen_delta_c=frozen)
            point_rows_baseline.append(
                evaluate_point_design(
                    point,
                    f_hz=baseline,
                    spectrum_fn=spectrum_fn,
                    **eval_kwargs,
                )
            )
        bottlenecks = [r["bottleneck"] for r in point_rows_baseline]
        per_point_angles = [
            principal_angle_deg(j0, r["o2_jacobian"])
            for j0, r in zip(f0_o2_jacs, point_rows_baseline, strict=True)
        ]
        ang_stats = angle_summary(per_point_angles, angle_quantiles)

        for freqs in designs:
            did = design_id(freqs)
            rows = []
            for point in points:
                if shared_fn is not None:
                    spectrum_fn = shared_fn
                else:
                    spectrum_fn = make_spectrum_fn(
                        spec, frozen_delta_c=frozen_by_point[point_key(point)]
                    )
                rows.append(
                    evaluate_point_design(
                        point,
                        f_hz=freqs,
                        spectrum_fn=spectrum_fn,
                        **eval_kwargs,
                    )
                )
            design_summaries[did] = summarize_design(rows)

        ranking = rank_designs(
            design_summaries,
            metric=str(mei1["design_enumeration"]["rank_metric"]),
            delta_num=delta_num,
        )
        rank_by_id = {r["design_id"]: float(r["rank"]) for r in ranking}
        best_rank = min(float(r["rank"]) for r in ranking)
        top_class = {r["design_id"] for r in ranking if float(r["rank"]) == best_rank}
        # Representative top1: lowest raw_order within top class.
        top1 = sorted(
            (r for r in ranking if r["design_id"] in top_class),
            key=lambda r: (r["raw_order"], r["design_id"]),
        )[0]["design_id"]
        ranking_resolvable = bool(ranking[0]["ranking_resolvable"])
        span_rel = float(ranking[0]["ranking_span_relative"])
        n_levels = int(ranking[0]["distinguishable_rank_levels"])

        # Synthetic alignment evidence on baseline geometry (observation delta vs J).
        synthetic_alignment = None
        if spec.synthetic_direction is not None:
            align_angles = []
            for point, f0_row, fam_row in zip(points, f0_baseline_rows, point_rows_baseline, strict=True):
                dy = np.asarray(fam_row["y0"], dtype=np.float64) - np.asarray(
                    f0_row["y0"], dtype=np.float64
                )
                align_angles.append(principal_angle_deg(dy, f0_row["o2_jacobian"]))
            expected = {
                "parallel": 0.0,
                "orthogonal": 90.0,
                "mixed": 45.0,
            }[str(spec.synthetic_direction)]
            synthetic_alignment = {
                "direction": spec.synthetic_direction,
                "mean_observation_delta_angle_to_o2_jacobian_deg": float(np.mean(align_angles)),
                "max_observation_delta_angle_to_o2_jacobian_deg": float(np.max(align_angles)),
                "expected_angle_deg": expected,
                "n_points": len(align_angles),
            }

        angle_gate_value = None
        if angle_gate_stat == "max_deg":
            angle_gate_value = ang_stats["max_deg"]
        elif angle_gate_stat.startswith("p"):
            angle_gate_value = ang_stats["quantiles_deg"].get(angle_gate_stat)
        else:
            raise ValueError(f"unsupported principal_angle_gate_stat: {angle_gate_stat}")

        spearman = None
        top1_match = None
        relative_p90_change_vs_f0 = None

        if spec.family_id == "F0_mrs1_baseline":
            f0_ranking_ids = [r["design_id"] for r in ranking]
            f0_rank_by_id = rank_by_id
            f0_top_class = set(top_class)
            f0_baseline_summary = design_summaries[baseline_id]
            f0_ranking_meta = {
                "ranking_resolvable": ranking_resolvable,
                "ranking_span_relative": span_rel,
                "distinguishable_rank_levels": n_levels,
                "baseline_k4_metric": float(f0_baseline_summary["max_p90_o2_percent"]),
                "best_metric": float(ranking[0]["metric"]),
                "worst_metric": float(ranking[-1]["metric"]),
                "best_vs_baseline_improve_relative": (
                    float(f0_baseline_summary["max_p90_o2_percent"]) - float(ranking[0]["metric"])
                )
                / max(float(f0_baseline_summary["max_p90_o2_percent"]), 1e-30),
            }
            spearman = 1.0 if ranking_resolvable else float("nan")
            top1_match = True
            relative_p90_change_vs_f0 = 0.0
        else:
            assert f0_ranking_ids is not None and f0_top_class is not None
            if ranking_resolvable and f0_ranking_meta.get("ranking_resolvable"):
                spearman = spearman_rank_corr(
                    [f0_rank_by_id[d] for d in f0_ranking_ids],
                    [rank_by_id[d] for d in f0_ranking_ids],
                )
                top1_match = top_class == f0_top_class
            else:
                spearman = float("nan")
                top1_match = None
            relative_p90_change_vs_f0 = abs(
                float(design_summaries[baseline_id]["max_p90_o2_percent"])
                - float(f0_baseline_summary["max_p90_o2_percent"])
            ) / max(float(f0_baseline_summary["max_p90_o2_percent"]), 1e-30)

        family_reports[spec.family_id] = {
            "kind": spec.kind,
            "registry_family": spec.registry_family,
            "notes": spec.notes,
            "inert_for_flip_gate": spec.family_id in inert_ids,
            "baseline_k4_summary": design_summaries[baseline_id],
            "ranking": ranking,
            "top1_design_id": top1,
            "top_equivalence_class": sorted(top_class),
            "ranking_resolvable": ranking_resolvable,
            "ranking_span_relative": span_rel,
            "distinguishable_rank_levels": n_levels,
            "o2_jacobian_angle_vs_f0": ang_stats,
            "principal_angle_gate_stat": angle_gate_stat,
            "principal_angle_gate_value_deg": angle_gate_value,
            # retained for backward-compatible summary readers
            "mean_o2_jacobian_principal_angle_deg_vs_f0": ang_stats["mean_deg"],
            "spearman_vs_f0": spearman,
            "top1_matches_f0": top1_match,
            "mode_bottleneck": summarize_design(point_rows_baseline)["mode_bottleneck"],
            "point_bottlenecks": bottlenecks,
            "relative_max_p90_change_vs_f0_on_baseline_k4": relative_p90_change_vs_f0,
            "synthetic_alignment": synthetic_alignment,
            "n_designs": len(designs),
            "n_points": len(points),
        }

    f0_bots = family_reports["F0_mrs1_baseline"]["point_bottlenecks"]
    for fid, report in family_reports.items():
        flips = sum(
            1 for a, b in zip(f0_bots, report["point_bottlenecks"], strict=True) if a != b
        )
        frac = flips / max(len(f0_bots), 1)
        report["bottleneck_flip_fraction_vs_f0"] = float(frac)

    flip_events: list[dict[str, Any]] = []
    ranking_resolvable = bool(f0_ranking_meta.get("ranking_resolvable"))
    for fid, report in family_reports.items():
        if fid == "F0_mrs1_baseline" or report["inert_for_flip_gate"]:
            continue
        reasons: list[str] = []
        if ranking_resolvable:
            if gates.get("top1_must_match_f0") and report["top1_matches_f0"] is False:
                reasons.append("top_equivalence_class_changed")
            sp = report["spearman_vs_f0"]
            if sp == sp and float(sp) < float(gates["min_spearman_vs_f0"]):
                reasons.append(f"spearman={sp:.4f}<{gates['min_spearman_vs_f0']}")
        gate_angle = report["principal_angle_gate_value_deg"]
        if gate_angle is not None and float(gate_angle) > float(gates["max_principal_angle_deg"]):
            reasons.append(
                f"principal_angle_{angle_gate_stat}={float(gate_angle):.3f}"
                f">{gates['max_principal_angle_deg']}"
            )
        if float(report["bottleneck_flip_fraction_vs_f0"]) > float(
            gates["bottleneck_flip_fraction_max"]
        ):
            reasons.append(
                "bottleneck_flip_fraction="
                f"{report['bottleneck_flip_fraction_vs_f0']:.3f}"
                f">{gates['bottleneck_flip_fraction_max']}"
            )
        report["p90_change_exceeds_delta_num"] = bool(
            float(report["relative_max_p90_change_vs_f0_on_baseline_k4"]) > delta_num
        )
        if reasons:
            flip_events.append({"family_id": fid, "reasons": reasons})

    unrepresented_blocking = collect_unrepresented_blocking(model)
    decision = decide_mei1_verdict(
        issues=issues,
        flip_events=flip_events,
        unrepresented_blocking=unrepresented_blocking,
        ranking_resolvable=ranking_resolvable,
    )

    f1_status = {
        "registry_id": "F1_humid_air_c_eq",
        "mei1_realized_as": "F1_humid_air_c_eq_upper",
        "c_eq_relative_correction": 0.01,
        "implementation": "post_process_scale_on_shared_relaxation_spectrum",
    }

    return {
        "schema_version": "tunnel-ventilation-mrs-ei-1",
        "stage": "MEI-1",
        "passed": decision["passed"],
        "verdict": decision["verdict"],
        "allowed_next_stage": decision["allowed_next_stage"],
        "blockers": decision["blockers"],
        "issues": issues,
        "delta_num": delta_num,
        "n_points": len(points),
        "n_designs": len(designs),
        "baseline_design_id": baseline_id,
        "point_labels": [lab for lab, _ in labeled_points],
        "family_reports": family_reports,
        "flip_events": flip_events,
        "f0_ranking_meta": f0_ranking_meta,
        "comsol_holdout_status": comsol_status,
        "f1_realization": f1_status,
        "unrepresented_registry_families": unrepresented_blocking,
        "stability_gates": gates,
        "claim_scope": "registered_simulation_domain_only",
        "formal_waveform_generation": "forbidden_until_authorized",
        "registry_sha256": {
            "model_family_registry.json": sha256_file(cfg_dir / "model_family_registry.json"),
            "design_space.json": sha256_file(cfg_dir / "design_space.json"),
            "metric_registry.json": sha256_file(cfg_dir / "metric_registry.json"),
            "mei1_forward_envelope.json": sha256_file(cfg_dir / "mei1_forward_envelope.json"),
        },
    }


__all__ = [
    "EnvelopeSpec",
    "apply_envelope",
    "apply_frozen_delta_c",
    "build_aligned_delta_tof",
    "collect_unrepresented_blocking",
    "decide_mei1_verdict",
    "design_id",
    "enumerate_k4_designs",
    "evaluate_point_design",
    "make_spectrum_fn",
    "precompute_synthetic_delta_c",
    "principal_angle_deg",
    "rank_designs",
    "run_mei1_audit",
    "select_audit_points",
    "spearman_rank_corr",
]
