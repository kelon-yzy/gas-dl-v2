"""Registered ideal-observation operator for the MEI-3 MRS solver audit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tv3.sim.generation.tunnel_ventilation.relaxation_spectrum import (
    relaxation_spectrum,
)

RAW3_COMPONENTS = ("x_CO2", "x_O2", "x_N2")
RAW3_TOTAL_PERCENT = 100.0
RAW3_CLOSURE_ATOL = 1e-5
OBSERVATION_FIELDS = ("raw_tof_s", "log_amplitude", "unwrapped_phase_rad")
OBSERVATION_UNITS = ("s", "dimensionless", "rad")

# Columns span {delta x: sum(delta x) = 0}. The affine map updates all three
# components jointly and never derives N2 from the other two components.
RAW3_TANGENT_BASIS = np.asarray(
    [
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [0.0, -2.0 / np.sqrt(6.0)],
    ],
    dtype=np.float64,
)
RAW3_SIMPLEX_CENTER = np.full(3, RAW3_TOTAL_PERCENT / 3.0, dtype=np.float64)


@dataclass(frozen=True)
class MrsIdealObservation:
    frequencies_hz: np.ndarray
    raw_tof_s: np.ndarray
    log_amplitude: np.ndarray
    unwrapped_phase_rad: np.ndarray
    covariance: np.ndarray
    phase_branch_cycles: np.ndarray

    @property
    def vector(self) -> np.ndarray:
        return np.column_stack(
            (self.raw_tof_s, self.log_amplitude, self.unwrapped_phase_rad)
        ).reshape(-1)

    @property
    def row_fields(self) -> tuple[str, ...]:
        return OBSERVATION_FIELDS * self.frequencies_hz.size

    @property
    def row_units(self) -> tuple[str, ...]:
        return OBSERVATION_UNITS * self.frequencies_hz.size


def validate_raw3_percent(raw3_percent: Sequence[float]) -> np.ndarray:
    """Validate the registered dry-basis raw3 physical domain without projection."""
    raw3 = np.asarray(raw3_percent, dtype=np.float64)
    if raw3.shape != (3,) or not np.all(np.isfinite(raw3)):
        raise ValueError("raw3_percent must contain three finite values")
    if np.any(raw3 < 0.0):
        raise ValueError("raw3_percent components must be nonnegative")
    total = float(np.sum(raw3))
    if abs(total - RAW3_TOTAL_PERCENT) > RAW3_CLOSURE_ATOL:
        raise ValueError(
            "raw3_percent must satisfy the registered dry-basis sum=100 constraint; "
            f"got sum={total}"
        )
    return raw3


def raw3_tangent_coordinates(raw3_percent: Sequence[float]) -> np.ndarray:
    raw3 = validate_raw3_percent(raw3_percent)
    return RAW3_TANGENT_BASIS.T @ (raw3 - RAW3_SIMPLEX_CENTER)


def raw3_percent_from_tangent(coordinates: Sequence[float]) -> np.ndarray:
    z = np.asarray(coordinates, dtype=np.float64)
    if z.shape != (2,) or not np.all(np.isfinite(z)):
        raise ValueError("raw3 tangent coordinates must contain two finite values")
    return validate_raw3_percent(RAW3_SIMPLEX_CENTER + RAW3_TANGENT_BASIS @ z)


def ideal_mrs_observation(
    raw3_percent: Sequence[float],
    *,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    path_length_m: float,
    frequencies_hz: Sequence[float],
    phase_branch_cycles: Sequence[int],
    observation_std: Mapping[str, float],
) -> MrsIdealObservation:
    """Return ideal TOF, log-amplitude, phase, and covariance in fixed row order."""
    raw3 = validate_raw3_percent(raw3_percent)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    branches = np.asarray(phase_branch_cycles)
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be a non-empty positive 1-D array")
    if branches.shape != frequencies.shape or not np.issubdtype(branches.dtype, np.integer):
        raise ValueError("phase_branch_cycles must be an integer per frequency")
    if not np.isfinite(path_length_m) or path_length_m <= 0.0:
        raise ValueError("path_length_m must be finite and positive")

    std = np.asarray([float(observation_std[name]) for name in OBSERVATION_FIELDS])
    if np.any(~np.isfinite(std)) or np.any(std <= 0.0):
        raise ValueError("all registered observation standard deviations must be positive")

    spectrum = relaxation_spectrum(
        float(raw3[0]),
        float(raw3[1]),
        float(raw3[2]),
        float(t_c),
        float(p_mpa),
        float(h_rh),
        frequencies,
    )
    raw_tof = path_length_m / np.asarray(spectrum["c_f"], dtype=np.float64)
    log_amplitude = -np.asarray(spectrum["alpha_f"], dtype=np.float64) * path_length_m
    phase = -2.0 * np.pi * frequencies * raw_tof + 2.0 * np.pi * branches
    row_std = np.tile(std, frequencies.size)
    return MrsIdealObservation(
        frequencies_hz=frequencies,
        raw_tof_s=raw_tof,
        log_amplitude=log_amplitude,
        unwrapped_phase_rad=phase,
        covariance=np.diag(row_std**2),
        phase_branch_cycles=branches.astype(np.int64, copy=False),
    )


__all__ = [
    "MrsIdealObservation",
    "OBSERVATION_FIELDS",
    "OBSERVATION_UNITS",
    "RAW3_CLOSURE_ATOL",
    "RAW3_COMPONENTS",
    "RAW3_TANGENT_BASIS",
    "RAW3_TOTAL_PERCENT",
    "ideal_mrs_observation",
    "raw3_percent_from_tangent",
    "raw3_tangent_coordinates",
    "validate_raw3_percent",
]
