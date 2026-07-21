"""Uniform axial-flow transit-time physics for tv3 bidirectional ultrasound (F 线).

Constants match F0 ``configs/tv3_bidir/parameter_registry.json``.
Propagation: ``t_ab = L/(c+v)``, ``t_ba = L/(c-v)``; reciprocal-sum estimators
are exact for spatially uniform path velocity.
"""
from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
from scipy.stats.qmc import LatinHypercube

from tv3.sim.core.tunnel_ventilation_bidir_schema import (
    PHYSICS_BACKEND,
    PHYSICS_BACKEND_FLOW,
    SCHEMA_VERSION,
    SIM_REVISION_TAG,
)

# F0-frozen defaults (do not scatter magic numbers in callers)
V_PATH_RANGE_M_PER_S: tuple[float, float] = (-4.0, 4.0)
ZERO_ANCHOR_FRACTION_MIN: float = 0.10
DEFAULT_PAIR_INTERVAL_S: float = 0.0025
MAX_PAIR_INTERVAL_S: float = 0.0025  # registry upper bound; distinct from default
DEFAULT_DELAY_ASYMMETRY_S: float = 0.0
DEFAULT_JITTER_CORRELATION: Literal["independent", "shared_trigger"] = "independent"
# 规程风速界（岩巷下限 0.25 m/s；上限 4 m/s），用于 flow_scenario 分类
REGULATION_V_PATH_MIN_M_PER_S: float = 0.25
REGULATION_V_PATH_MAX_M_PER_S: float = 4.0
FLOW_SCENARIO_ZERO = "zero_anchor"
FLOW_SCENARIO_REGULATION = "regulation_range"
FLOW_SCENARIO_SCAN = "scan"

JITTER_CORRELATIONS = frozenset({"independent", "shared_trigger"})


def assert_subsonic_path_velocity(*, sound_speed_m_per_s: float, v_path_m_per_s: float) -> None:
    if not math.isfinite(sound_speed_m_per_s) or sound_speed_m_per_s <= 0.0:
        raise ValueError(f"sound_speed_m_per_s must be finite and > 0, got {sound_speed_m_per_s}")
    if not math.isfinite(v_path_m_per_s):
        raise ValueError(f"v_path_m_per_s must be finite, got {v_path_m_per_s}")
    if abs(v_path_m_per_s) >= sound_speed_m_per_s:
        raise ValueError(
            f"|v_path|={abs(v_path_m_per_s)} must be < sound_speed={sound_speed_m_per_s}"
        )


def transit_time_s(
    path_length_m: float,
    sound_speed_m_per_s: float,
    v_path_m_per_s: float,
) -> float:
    """One-way transit time along the acoustic path with signed path velocity."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    assert_subsonic_path_velocity(
        sound_speed_m_per_s=sound_speed_m_per_s,
        v_path_m_per_s=v_path_m_per_s,
    )
    return float(path_length_m) / (float(sound_speed_m_per_s) + float(v_path_m_per_s))


def bidirectional_transit_times_s(
    path_length_m: float,
    sound_speed_m_per_s: float,
    v_path_m_per_s: float,
) -> tuple[float, float]:
    """Return ``(t_ab, t_ba)`` for AB = +v_path and BA = −v_path."""
    t_ab = transit_time_s(path_length_m, sound_speed_m_per_s, v_path_m_per_s)
    t_ba = transit_time_s(path_length_m, sound_speed_m_per_s, -v_path_m_per_s)
    return t_ab, t_ba


def reciprocal_sum_sound_speed_m_per_s(
    path_length_m: float,
    t_ab_s: float,
    t_ba_s: float,
) -> float:
    """ĉ = (L/2) * (1/t_ab + 1/t_ba). Exact for uniform axial flow."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    if t_ab_s <= 0.0 or t_ba_s <= 0.0:
        raise ValueError("transit times must be > 0")
    return 0.5 * float(path_length_m) * (1.0 / float(t_ab_s) + 1.0 / float(t_ba_s))


def reciprocal_sum_path_velocity_m_per_s(
    path_length_m: float,
    t_ab_s: float,
    t_ba_s: float,
) -> float:
    """v̂ = (L/2) * (1/t_ab − 1/t_ba). Exact for uniform axial flow."""
    if path_length_m <= 0.0:
        raise ValueError("path_length_m must be > 0")
    if t_ab_s <= 0.0 or t_ba_s <= 0.0:
        raise ValueError("transit times must be > 0")
    return 0.5 * float(path_length_m) * (1.0 / float(t_ab_s) - 1.0 / float(t_ba_s))


def sound_speed_bias_from_velocity_variance_m_per_s(
    sound_speed_m_per_s: float,
    velocity_variance_m2_per_s2: float,
) -> float:
    """Leading second-order reciprocal-sum bias under path velocity variance: −Var(v)/c.

    Uniform flow ⇒ Var(v)=0 ⇒ bias 0. Sign is negative (ĉ underestimates medium c).
    """
    if sound_speed_m_per_s <= 0.0:
        raise ValueError("sound_speed_m_per_s must be > 0")
    if velocity_variance_m2_per_s2 < 0.0:
        raise ValueError("velocity_variance_m2_per_s2 must be >= 0")
    return -float(velocity_variance_m2_per_s2) / float(sound_speed_m_per_s)


def classify_flow_scenario(v_path_m_per_s: float) -> str:
    if abs(v_path_m_per_s) <= 1e-15:
        return FLOW_SCENARIO_ZERO
    mag = abs(v_path_m_per_s)
    if REGULATION_V_PATH_MIN_M_PER_S <= mag <= REGULATION_V_PATH_MAX_M_PER_S:
        return FLOW_SCENARIO_REGULATION
    return FLOW_SCENARIO_SCAN


def sample_v_path_m_per_s(
    sequence_count: int,
    *,
    rng: random.Random,
    v_path_range: tuple[float, float] = V_PATH_RANGE_M_PER_S,
    zero_anchor_fraction_min: float = ZERO_ANCHOR_FRACTION_MIN,
) -> np.ndarray:
    """1D Latin-hypercube sample of path velocity with a zero-anchor fraction.

    Matches registry ``sampling.mode = constrained_lhs_with_zero_anchor``.
    When ``zero_anchor_fraction_min == 0``, no zero anchors are forced.
    """
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    lo, hi = v_path_range
    if lo >= hi:
        raise ValueError(f"v_path_range must have lo < hi, got {v_path_range}")
    if not (0.0 <= zero_anchor_fraction_min <= 1.0):
        raise ValueError("zero_anchor_fraction_min must be in [0, 1]")

    n_zero = int(math.ceil(sequence_count * zero_anchor_fraction_min))
    n_zero = min(n_zero, sequence_count)
    n_flow = sequence_count - n_zero
    values = np.zeros(sequence_count, dtype=np.float64)
    if n_flow > 0:
        lhs_seed = rng.randrange(1, 2**31 - 1)
        sampler = LatinHypercube(d=1, seed=lhs_seed)
        raw = sampler.random(n=n_flow)
        values[n_zero:] = lo + raw[:, 0] * (hi - lo)
    order = list(range(sequence_count))
    rng.shuffle(order)
    return values[order].copy().astype(np.float64)


def attach_flow_fields_to_conditions(
    conditions: list[dict[str, str]],
    *,
    seed: int,
    pair_interval_s: float = DEFAULT_PAIR_INTERVAL_S,
    delay_asymmetry_s: float = DEFAULT_DELAY_ASYMMETRY_S,
    jitter_correlation: str = DEFAULT_JITTER_CORRELATION,
    v_path_range: tuple[float, float] = V_PATH_RANGE_M_PER_S,
    zero_anchor_fraction_min: float = ZERO_ANCHOR_FRACTION_MIN,
) -> list[dict[str, str]]:
    """Copy condition rows and attach F-line flow metadata (sequence-constant)."""
    if jitter_correlation not in JITTER_CORRELATIONS:
        raise ValueError(
            f"jitter_correlation must be one of {sorted(JITTER_CORRELATIONS)}, got {jitter_correlation!r}"
        )
    if pair_interval_s <= 0.0 or pair_interval_s > MAX_PAIR_INTERVAL_S:
        raise ValueError(f"pair_interval_s must be in (0, {MAX_PAIR_INTERVAL_S}]")
    rng = random.Random(seed)
    v_paths = sample_v_path_m_per_s(
        len(conditions),
        rng=rng,
        v_path_range=v_path_range,
        zero_anchor_fraction_min=zero_anchor_fraction_min,
    )
    out: list[dict[str, str]] = []
    for row, v_path in zip(conditions, v_paths, strict=True):
        enriched = dict(row)
        enriched["v_path_m_per_s"] = f"{float(v_path):.6f}"
        enriched["flow_scenario"] = classify_flow_scenario(float(v_path))
        enriched["pair_interval_s"] = f"{float(pair_interval_s):.6f}"
        enriched["delay_asymmetry_s"] = f"{float(delay_asymmetry_s):.6e}"
        enriched["jitter_correlation"] = jitter_correlation
        out.append(enriched)
    return out


def bidir_sim_revision(
    *,
    jitter_scenario_ids: tuple[str, ...] = ("conservative_v1", "nominal_daq_half_sample"),
) -> dict[str, object]:
    """Manifest ``sim_revision`` payload for F-line datasets."""
    return {
        "tag": SIM_REVISION_TAG,
        "physics_backend": PHYSICS_BACKEND,
        "flow_model": PHYSICS_BACKEND_FLOW,
        "schema": SCHEMA_VERSION,
        "jitter_scenarios": list(jitter_scenario_ids),
        "claim_scope": "registered_simulation_domain_only",
        "bidirectional": True,
    }
