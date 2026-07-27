"""MRS-2 forward identifiability audit (multifreq relaxation spectroscopy).

Acoustic-only relative SVD rank (reuse v2 convention). Joint CRLB for O₂ uses
measurement Σ plus diagonal nuisance priors (T / L / RH). No waveform generation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from tv3.audit.error_budget import NORMAL_P90_Z
from tv3.audit.identifiability_v2 import DEFAULT_RANK_RELATIVE_TOL, _relative_svd_rank
from tv3.sim.generation.tunnel_ventilation.relaxation_spectrum import relaxation_spectrum

DERIVATIVE_PARAMETERS = (
    "o2_percent",
    "co2_percent",
    "t_c",
    "path_length_m",
    "h_rh",
)

ARM_IDS = (
    "obs-single-200k",
    "obs-cfreq",
    "obs-calpha",
    "obs-rh-diff",
    "obs-p-scan",
)

MULTIFREQ_RANK_ARMS = ("obs-cfreq", "obs-calpha", "obs-rh-diff")

DEFAULT_F_HZ = (
    10000.0,
    16000.0,
    25000.0,
    40000.0,
    63000.0,
    100000.0,
    160000.0,
    200000.0,
)

FREQ_SUBSETS: dict[str, tuple[float, ...]] = {
    "K8": DEFAULT_F_HZ,
    "K6": (25000.0, 40000.0, 63000.0, 100000.0, 160000.0, 200000.0),
    "K4": (25000.0, 63000.0, 100000.0, 200000.0),
}


@dataclass(frozen=True)
class MrsPoint:
    co2_percent: float
    o2_percent: float
    t_c: float
    path_length_m: float
    h_rh: float
    p_mpa: float

    @property
    def n2_percent(self) -> float:
        return 100.0 - self.co2_percent - self.o2_percent

    def validate(self) -> None:
        vals = (
            self.co2_percent,
            self.o2_percent,
            self.t_c,
            self.path_length_m,
            self.h_rh,
            self.p_mpa,
        )
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("MrsPoint values must be finite")
        if self.co2_percent < 0.0 or self.o2_percent < 0.0 or self.n2_percent < 0.0 - 1e-9:
            raise ValueError("composition must be non-negative and sum to 100 percent")
        if self.path_length_m <= 0.0:
            raise ValueError("path_length_m must be positive")
        if self.p_mpa <= 0.0:
            raise ValueError("p_mpa must be positive")


def build_mrs_points(grid: Mapping[str, Sequence[float]]) -> list[MrsPoint]:
    required = (
        "co2_percent",
        "o2_percent",
        "t_c",
        "path_length_m",
        "h_rh",
        "p_mpa",
    )
    if set(grid) != set(required):
        raise ValueError(f"grid keys must be {required}, got {tuple(grid)}")
    values = [grid[name] for name in required]
    if any(not entries for entries in values):
        raise ValueError("each grid dimension must contain at least one value")
    return [MrsPoint(*map(float, entries)) for entries in product(*values)]


def _spectrum(point: MrsPoint, f_hz: np.ndarray) -> dict[str, Any]:
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


def observe_tof_alpha(
    point: MrsPoint,
    *,
    f_hz: Sequence[float],
    fixed_delay_s: float,
) -> dict[str, np.ndarray]:
    """Forward acoustic observables at frequencies f_hz."""
    if not math.isfinite(fixed_delay_s) or fixed_delay_s < 0.0:
        raise ValueError("fixed_delay_s must be finite and >= 0")
    f = np.asarray(f_hz, dtype=np.float64)
    out = _spectrum(point, f)
    c = np.asarray(out["c_f"], dtype=np.float64)
    alpha = np.asarray(out["alpha_f"], dtype=np.float64)
    tof = point.path_length_m / c + fixed_delay_s
    return {"f_hz": f, "c_f": c, "alpha_f": alpha, "tof_s": tof}


def tof_std_s_for_frequency(
    f_hz: float,
    *,
    jitter_std_s: float,
    phase_std_s_at_anchor: float = 0.0,
    anchor_hz: float = 200000.0,
) -> float:
    """Per-frequency TOF σ.

    Trigger jitter is frequency-independent (v1/MRS-0). Optional phase term
    scales as 1/f relative to ``anchor_hz``. Total: sqrt(jitter² + phase(f)²).
    """
    if f_hz <= 0.0 or anchor_hz <= 0.0:
        raise ValueError("frequencies must be > 0")
    if not math.isfinite(jitter_std_s) or jitter_std_s <= 0.0:
        raise ValueError("jitter_std_s must be finite and > 0")
    if not math.isfinite(phase_std_s_at_anchor) or phase_std_s_at_anchor < 0.0:
        raise ValueError("phase_std_s_at_anchor must be finite and >= 0")
    phase = float(phase_std_s_at_anchor) * (anchor_hz / float(f_hz))
    return float(math.sqrt(jitter_std_s**2 + phase**2))


def alpha_std_npm(*, relative_amp_std: float, path_length_m: float) -> float:
    """Relative amplitude error → α uncertainty: |δα| ≈ |dA/A| / L."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    if not math.isfinite(relative_amp_std) or relative_amp_std < 0.0:
        raise ValueError("relative_amp_std must be finite and >= 0")
    return float(relative_amp_std) / float(path_length_m)


def _shift_point(point: MrsPoint, parameter: str, delta: float) -> MrsPoint:
    kwargs = {
        "co2_percent": point.co2_percent,
        "o2_percent": point.o2_percent,
        "t_c": point.t_c,
        "path_length_m": point.path_length_m,
        "h_rh": point.h_rh,
        "p_mpa": point.p_mpa,
    }
    if parameter not in kwargs:
        raise ValueError(f"unsupported derivative parameter: {parameter!r}")
    kwargs[parameter] = float(kwargs[parameter]) + float(delta)
    return MrsPoint(**kwargs)


def _within_bounds(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def _parameter_value(point: MrsPoint, parameter: str) -> float:
    return float(getattr(point, parameter))


def observation_vector(
    point: MrsPoint,
    *,
    arm: str,
    f_hz: Sequence[float],
    fixed_delay_s: float,
    rh_delta: float,
    p_scan_mpa: Sequence[float],
) -> tuple[np.ndarray, list[str]]:
    """Build acoustic observation vector and row labels for one arm."""
    if arm not in ARM_IDS:
        raise ValueError(f"unknown arm {arm!r}")

    def _pack(pt: MrsPoint, tag: str) -> tuple[list[float], list[str]]:
        obs = observe_tof_alpha(pt, f_hz=f_hz, fixed_delay_s=fixed_delay_s)
        values: list[float] = []
        labels: list[str] = []
        for i, fk in enumerate(obs["f_hz"]):
            values.append(float(obs["tof_s"][i]))
            labels.append(f"{tag}:tof@{int(fk)}Hz")
        if arm in ("obs-calpha", "obs-rh-diff", "obs-p-scan"):
            for i, fk in enumerate(obs["f_hz"]):
                values.append(float(obs["alpha_f"][i]))
                labels.append(f"{tag}:alpha@{int(fk)}Hz")
        return values, labels

    if arm == "obs-single-200k":
        obs = observe_tof_alpha(point, f_hz=(200000.0,), fixed_delay_s=fixed_delay_s)
        return np.asarray([float(obs["tof_s"][0])], dtype=np.float64), ["tof@200000Hz"]

    if arm == "obs-cfreq":
        vals, labs = _pack(point, "base")
        # cfreq: TOF only (strip alpha if any — _pack only adds alpha for calpha+)
        return np.asarray(vals, dtype=np.float64), labs

    if arm == "obs-calpha":
        vals, labs = _pack(point, "base")
        return np.asarray(vals, dtype=np.float64), labs

    if arm == "obs-rh-diff":
        if rh_delta <= 0.0:
            raise ValueError("rh_delta must be > 0")
        pt2 = MrsPoint(
            point.co2_percent,
            point.o2_percent,
            point.t_c,
            point.path_length_m,
            point.h_rh + rh_delta,
            point.p_mpa,
        )
        v1, l1 = _pack(point, "rh0")
        v2, l2 = _pack(pt2, "rh1")
        return np.asarray(v1 + v2, dtype=np.float64), l1 + l2

    # obs-p-scan
    if len(p_scan_mpa) != 2:
        raise ValueError("p_scan_mpa must have exactly 2 points")
    chunks_v: list[float] = []
    chunks_l: list[str] = []
    for p in p_scan_mpa:
        base = MrsPoint(
            point.co2_percent,
            point.o2_percent,
            point.t_c,
            point.path_length_m,
            point.h_rh,
            float(p),
        )
        pt2 = MrsPoint(
            point.co2_percent,
            point.o2_percent,
            point.t_c,
            point.path_length_m,
            point.h_rh + rh_delta,
            float(p),
        )
        for pt, tag in ((base, f"p{p:g}_rh0"), (pt2, f"p{p:g}_rh1")):
            v, lab = _pack(pt, tag)
            chunks_v.extend(v)
            chunks_l.extend(lab)
    return np.asarray(chunks_v, dtype=np.float64), chunks_l


def observation_noise_std(
    labels: Sequence[str],
    *,
    point: MrsPoint,
    jitter_std_s: float,
    relative_amp_std: float,
    phase_std_s_at_anchor: float = 0.0,
) -> np.ndarray:
    """Diagonal Σ^{1/2} matching row labels."""
    sigmas: list[float] = []
    for lab in labels:
        if ":tof@" in lab or lab.startswith("tof@"):
            hz_token = lab.split("tof@", 1)[1].replace("Hz", "")
            f_hz = float(hz_token)
            sigmas.append(
                tof_std_s_for_frequency(
                    f_hz,
                    jitter_std_s=jitter_std_s,
                    phase_std_s_at_anchor=phase_std_s_at_anchor,
                )
            )
        elif ":alpha@" in lab:
            sigmas.append(alpha_std_npm(relative_amp_std=relative_amp_std, path_length_m=point.path_length_m))
        else:
            raise ValueError(f"unrecognized observation label: {lab}")
    arr = np.asarray(sigmas, dtype=np.float64)
    if np.any(arr <= 0.0):
        raise ValueError("observation noise std must be > 0")
    return arr


def _finite_difference_vector(
    point: MrsPoint,
    *,
    parameter: str,
    step: float,
    bounds: tuple[float, float],
    arm: str,
    f_hz: Sequence[float],
    fixed_delay_s: float,
    rh_delta: float,
    p_scan_mpa: Sequence[float],
) -> tuple[np.ndarray, str]:
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step for {parameter} must be finite and > 0")
    current = _parameter_value(point, parameter)
    plus = current + step
    minus = current - step
    has_plus = _within_bounds(plus, bounds)
    has_minus = _within_bounds(minus, bounds)
    y0, _ = observation_vector(
        point,
        arm=arm,
        f_hz=f_hz,
        fixed_delay_s=fixed_delay_s,
        rh_delta=rh_delta,
        p_scan_mpa=p_scan_mpa,
    )
    if has_plus and has_minus:
        yp, _ = observation_vector(
            _shift_point(point, parameter, step),
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        ym, _ = observation_vector(
            _shift_point(point, parameter, -step),
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        return (yp - ym) / (2.0 * step), "central"
    if has_plus:
        yp, _ = observation_vector(
            _shift_point(point, parameter, step),
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        return (yp - y0) / step, "forward"
    if has_minus:
        ym, _ = observation_vector(
            _shift_point(point, parameter, -step),
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        return (y0 - ym) / step, "backward"
    raise ValueError(f"no valid FD stencil for {parameter} at {current} within {bounds}")


def local_mrs_jacobian(
    point: MrsPoint,
    *,
    arm: str,
    f_hz: Sequence[float],
    parameter_steps: Mapping[str, float],
    parameter_bounds: Mapping[str, Sequence[float]],
    fixed_delay_s: float,
    rh_delta: float,
    p_scan_mpa: Sequence[float],
    max_relative_step_disagreement: float = 0.01,
) -> dict[str, Any]:
    """Finite-difference Jacobian + stability check (half/double step)."""
    y0, labels = observation_vector(
        point,
        arm=arm,
        f_hz=f_hz,
        fixed_delay_s=fixed_delay_s,
        rh_delta=rh_delta,
        p_scan_mpa=p_scan_mpa,
    )
    n_obs = y0.size
    n_param = len(DERIVATIVE_PARAMETERS)
    jac = np.zeros((n_obs, n_param), dtype=np.float64)
    meta: dict[str, Any] = {}
    for idx, parameter in enumerate(DERIVATIVE_PARAMETERS):
        step = float(parameter_steps[parameter])
        bounds = (float(parameter_bounds[parameter][0]), float(parameter_bounds[parameter][1]))
        d, scheme = _finite_difference_vector(
            point,
            parameter=parameter,
            step=step,
            bounds=bounds,
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        d_half, _ = _finite_difference_vector(
            point,
            parameter=parameter,
            step=step * 0.5,
            bounds=bounds,
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        d_double, _ = _finite_difference_vector(
            point,
            parameter=parameter,
            step=step * 2.0,
            bounds=bounds,
            arm=arm,
            f_hz=f_hz,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
        )
        denom = np.maximum(np.maximum(np.abs(d_half), np.abs(d_double)), 1e-15)
        disagreement = float(np.max(np.abs(d_half - d_double) / denom))
        jac[:, idx] = d
        meta[parameter] = {
            "scheme": scheme,
            "step_disagreement": disagreement,
            "stable": disagreement <= max_relative_step_disagreement,
            "derivative_l2": float(np.linalg.norm(d)),
        }
    return {
        "jacobian": jac,
        "labels": labels,
        "y0": y0,
        "parameter_meta": meta,
        "all_stable": all(m["stable"] for m in meta.values()),
    }


def fisher_rank_crb(
    jacobian: np.ndarray,
    *,
    row_sigmas: np.ndarray,
    parameter_steps: Mapping[str, float],
    prior_std: Mapping[str, float],
    rank_relative_tol: float = DEFAULT_RANK_RELATIVE_TOL,
) -> dict[str, Any]:
    """Acoustic Fisher, relative SVD rank, and O₂ CRLB with nuisance priors.

    ``prior_std`` may include ``co2_percent`` (NDIR engineering prior). Rank uses
    acoustic Jacobian only (no direct T/RH/CO2 sensor rows), matching the
    obs-single-200k rank-1 negative control.
    """
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be 2-D")
    n_obs, n_param = jacobian.shape
    if n_param != len(DERIVATIVE_PARAMETERS):
        raise ValueError("jacobian column count must match DERIVATIVE_PARAMETERS")
    if row_sigmas.shape != (n_obs,):
        raise ValueError("row_sigmas shape mismatch")
    if not np.isfinite(jacobian).all():
        raise ValueError("jacobian must be finite")

    cov_inv = np.diag(1.0 / (row_sigmas**2))
    fisher = jacobian.T @ cov_inv @ jacobian

    scaled = jacobian / row_sigmas[:, None]
    for idx, parameter in enumerate(DERIVATIVE_PARAMETERS):
        step = float(parameter_steps[parameter])
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"parameter_steps[{parameter!r}] must be finite and > 0")
        scaled[:, idx] *= step
    rank = _relative_svd_rank(scaled, relative_tol=rank_relative_tol)

    # Priors: never on O₂ (target). Optional CO₂ (NDIR). Required T/L/RH.
    prior = np.zeros((n_param, n_param), dtype=np.float64)
    for idx, parameter in enumerate(DERIVATIVE_PARAMETERS):
        if parameter == "o2_percent":
            continue
        if parameter not in prior_std:
            if parameter == "co2_percent":
                continue
            raise ValueError(f"prior_std missing required key {parameter!r}")
        sigma = float(prior_std[parameter])
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"prior_std[{parameter!r}] must be finite and > 0")
        prior[idx, idx] = 1.0 / (sigma**2)
    fisher_aug = fisher + prior

    crlb_o2 = float("nan")
    p90_o2 = float("nan")
    invertible = False
    cond = None
    try:
        fa = 0.5 * (fisher_aug + fisher_aug.T)
        cond = float(np.linalg.cond(fa))
        if math.isfinite(cond) and cond < 1e16:
            cov = np.linalg.inv(fa)
            var = float(cov[0, 0])
            if var > 0.0 and math.isfinite(var):
                crlb_o2 = math.sqrt(var)
                p90_o2 = NORMAL_P90_Z * crlb_o2
                invertible = True
    except np.linalg.LinAlgError:
        invertible = False

    return {
        "joint_rank": int(rank),
        "joint_parameter_count": n_param,
        "joint_observation_count": n_obs,
        "rank_upgraded": int(rank) >= 2,
        "fisher": fisher,
        "fisher_aug": fisher_aug,
        "crlb_o2_std_percent": crlb_o2,
        "p90_o2_percent": p90_o2,
        "fisher_aug_invertible": invertible,
        "fisher_aug_condition_number": cond,
        "singular_values_scaled": np.linalg.svd(scaled, compute_uv=False).tolist(),
        "prior_keys": sorted(k for k in prior_std if k != "o2_percent"),
    }


def single_nuisance_equivalent_o2(
    jacobian: np.ndarray,
    *,
    row_sigmas: np.ndarray,
    parameter_steps: Mapping[str, float],
    prior_std: Mapping[str, float],
    window_width_percent: float,
) -> dict[str, Any]:
    """Per-nuisance O₂ σ and fraction of narrow-window width (≤50% gate)."""
    rows: dict[str, Any] = {}
    worst_frac = 0.0

    # Measurement-noise-only conditional CRLB (nuisance params treated as known).
    j_o2 = jacobian[:, 0:1]
    info = float((j_o2.T @ np.diag(1.0 / (row_sigmas**2)) @ j_o2)[0, 0])
    sigma_meas = math.sqrt(1.0 / info) if info > 0.0 else float("inf")
    frac_meas = NORMAL_P90_Z * sigma_meas / window_width_percent
    rows["measurement_noise"] = {
        "equivalent_o2_std_percent": sigma_meas,
        "p90_o2_percent": NORMAL_P90_Z * sigma_meas if math.isfinite(sigma_meas) else float("inf"),
        "fraction_of_window": frac_meas,
    }
    if math.isfinite(frac_meas):
        worst_frac = max(worst_frac, frac_meas)

    for nuisance in ("t_c", "path_length_m", "h_rh", "co2_percent"):
        if nuisance == "co2_percent" and "co2_percent" not in prior_std:
            continue
        # Only this nuisance uncertain; others hard-constrained via tiny prior σ.
        prior = {
            "t_c": 1e-9,
            "path_length_m": 1e-12,
            "h_rh": 1e-9,
        }
        if "co2_percent" in prior_std:
            prior["co2_percent"] = 1e-9
        prior[nuisance] = float(prior_std[nuisance])
        out = fisher_rank_crb(
            jacobian,
            row_sigmas=row_sigmas,
            parameter_steps=parameter_steps,
            prior_std=prior,
        )
        sigma = float(out["crlb_o2_std_percent"])
        frac = (
            NORMAL_P90_Z * sigma / window_width_percent
            if math.isfinite(sigma)
            else float("inf")
        )
        rows[nuisance] = {
            "equivalent_o2_std_percent": sigma,
            "p90_o2_percent": NORMAL_P90_Z * sigma if math.isfinite(sigma) else float("inf"),
            "fraction_of_window": frac,
            "fisher_aug_invertible": out["fisher_aug_invertible"],
        }
        if math.isfinite(frac):
            worst_frac = max(worst_frac, frac)
    return {"per_nuisance": rows, "worst_fraction_of_window": worst_frac}


def evaluate_point_arm(
    point: MrsPoint,
    *,
    arm: str,
    f_hz: Sequence[float],
    parameter_steps: Mapping[str, float],
    parameter_bounds: Mapping[str, Sequence[float]],
    fixed_delay_s: float,
    rh_delta: float,
    p_scan_mpa: Sequence[float],
    jitter_std_s: float,
    relative_amp_std: float,
    prior_std: Mapping[str, float],
    window_width_percent: float,
    phase_std_s_at_anchor: float = 0.0,
    max_relative_step_disagreement: float = 0.01,
) -> dict[str, Any]:
    loc = local_mrs_jacobian(
        point,
        arm=arm,
        f_hz=f_hz,
        parameter_steps=parameter_steps,
        parameter_bounds=parameter_bounds,
        fixed_delay_s=fixed_delay_s,
        rh_delta=rh_delta,
        p_scan_mpa=p_scan_mpa,
        max_relative_step_disagreement=max_relative_step_disagreement,
    )
    sigmas = observation_noise_std(
        loc["labels"],
        point=point,
        jitter_std_s=jitter_std_s,
        relative_amp_std=relative_amp_std,
        phase_std_s_at_anchor=phase_std_s_at_anchor,
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
    return {
        "arm": arm,
        "f_hz": list(map(float, f_hz)),
        "point": {
            "co2_percent": point.co2_percent,
            "o2_percent": point.o2_percent,
            "t_c": point.t_c,
            "path_length_m": point.path_length_m,
            "h_rh": point.h_rh,
            "p_mpa": point.p_mpa,
        },
        "all_stable": loc["all_stable"],
        "parameter_meta": loc["parameter_meta"],
        "n_obs": len(loc["labels"]),
        "labels": loc["labels"],
        **{k: fish[k] for k in (
            "joint_rank",
            "joint_parameter_count",
            "joint_observation_count",
            "rank_upgraded",
            "crlb_o2_std_percent",
            "p90_o2_percent",
            "fisher_aug_invertible",
            "fisher_aug_condition_number",
            "singular_values_scaled",
            "prior_keys",
        )},
        "nuisance": nuisance,
    }


def choose_mrs2_verdict(
    *,
    single_200k_min_rank: int,
    arm_summaries: Mapping[str, Mapping[str, Any]],
    target_p90: float,
    max_nuisance_fraction: float,
    max_rejection_rate: float,
    rejection_rate: float,
) -> dict[str, Any]:
    """Freeze MRS-2 verdict branching (no gate retuning)."""
    if int(single_200k_min_rank) != 1:
        return {
            "verdict": "audit_failed",
            "reason": (
                f"obs-single-200k min_rank={single_200k_min_rank} (expected 1); "
                "negative control failed — audit implementation error"
            ),
            "allow_mrs3": False,
        }

    multifreq_ranks = {
        arm: int(arm_summaries[arm]["min_joint_rank"]) for arm in MULTIFREQ_RANK_ARMS
    }
    rank_still_deficient = all(r < 2 for r in multifreq_ranks.values())
    if rank_still_deficient:
        return {
            "verdict": "mrs2_rank_still_deficient",
            "reason": (
                "obs-cfreq/calpha/rh-diff joint ranks still all < 2; "
                "problem is hardware bandwidth / humidity-control capability, not algorithms"
            ),
            "allow_mrs3": False,
            "multifreq_ranks": multifreq_ranks,
            "require_mrs6": True,
        }

    if rejection_rate > max_rejection_rate:
        return {
            "verdict": "mrs2_rank_upgraded_p90_fail",
            "reason": f"rejection_rate={rejection_rate} > {max_rejection_rate}",
            "allow_mrs3": False,
            "multifreq_ranks": multifreq_ranks,
            "require_mrs6": True,
        }

    passing_arms: list[str] = []
    for arm, summary in arm_summaries.items():
        if arm == "obs-single-200k":
            continue
        if int(summary["min_joint_rank"]) < 2:
            continue
        p90_ok = float(summary["max_p90_o2_percent"]) <= target_p90
        nuis_ok = float(summary["max_nuisance_fraction"]) <= max_nuisance_fraction
        inv_ok = bool(summary.get("all_crlb_invertible", False))
        if p90_ok and nuis_ok and inv_ok:
            passing_arms.append(arm)

    if not passing_arms:
        return {
            "verdict": "mrs2_rank_upgraded_p90_fail",
            "reason": (
                "rank upgraded on at least one multifreq arm, but every arm fails "
                f"P90≤{target_p90} and/or nuisance≤{max_nuisance_fraction} (incl. obs-p-scan)"
            ),
            "allow_mrs3": False,
            "multifreq_ranks": multifreq_ranks,
            "require_mrs6": True,
        }

    return {
        "verdict": "mrs2_rank_upgraded_p90_pass",
        "reason": f"rank upgraded; arms passing gates: {passing_arms}",
        "allow_mrs3": True,
        "multifreq_ranks": multifreq_ranks,
        "passing_arms": passing_arms,
        "require_mrs6": False,
    }


__all__ = [
    "ARM_IDS",
    "DEFAULT_F_HZ",
    "DEFAULT_RANK_RELATIVE_TOL",
    "DERIVATIVE_PARAMETERS",
    "FREQ_SUBSETS",
    "MULTIFREQ_RANK_ARMS",
    "MrsPoint",
    "alpha_std_npm",
    "build_mrs_points",
    "choose_mrs2_verdict",
    "evaluate_point_arm",
    "fisher_rank_crb",
    "local_mrs_jacobian",
    "observation_noise_std",
    "observation_vector",
    "observe_tof_alpha",
    "tof_std_s_for_frequency",
]
