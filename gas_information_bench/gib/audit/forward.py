"""P2 pure-forward candidate audit.

This module owns a deterministic, data-free screening model. It re-derives the
small mathematical functions locally and never imports a historical scenario
package. The thermophysical profile is deliberately explicit so P2-10 can
replace or complete its source registry without changing the audit contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Iterable, Mapping

import numpy as np


COMPONENT_ORDER = ("N2", "CO2", "O2", "Ar", "CH4")
ETA_NAMES = (
    "T_K",
    "P_kPa",
    "RH_frac",
    "L_m",
    "gain",
    "baseline",
    "delay_s",
    "crosstalk",
    "q_flow",
)
SLOW_CHANNEL_NAMES = ("T_K", "P_kPa", "RH_frac", "q_flow")
ETA_DEFAULT = np.array(
    [298.15, 101.325, 0.50, 0.20, 1.0, 0.0, 0.0, 0.03, 1.0],
    dtype=float,
)
ETA_PRIOR_STD = np.array(
    [0.10, 0.50, 0.01, 0.005, 0.02, 0.005, 2.0e-6, 0.02, 0.05],
    dtype=float,
)
ETA_STEPS = np.array(
    [1.0e-3, 1.0e-3, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-8, 1.0e-5, 1.0e-4],
    dtype=float,
)

# Formula and hardware provenance used by this screening profile. These are
# metadata only; this module has no runtime dependency on the historical paths.
SOURCE_PROFILE = {
    "acoustic_formula": "hydrogen_ng/docs/物理模型严格化实施计划.md §2; ideal-gas c=sqrt(gamma*R*T/M)",
    "sensor_hardware": "hydrogen_ng/docs/references/传感器硬件资料整理.md §2–§8",
    "industrial_carrier": "docs/工业多传感器气体组分检测_工业应用方向调研报告.md §2.1",
    "oxygen_argon_profile": "P2-05 screening profile; formal source registry is required in P2-10",
}

# The values are a local screening profile, not a copied implementation. The
# O2/Ar entries are intentionally marked for P2-10 source completion above.
MOLAR_MASS = {
    "N2": 0.0280134,
    "CO2": 0.0440095,
    "O2": 0.0319988,
    "Ar": 0.0399480,
    "CH4": 0.0160430,
}
CP_MOLAR = {
    "N2": 29.124,
    "CO2": 37.135,
    "O2": 29.376,
    "Ar": 20.786,
    "CH4": 35.690,
}
THERMAL_CONDUCTIVITY = {
    "N2": 0.02583,
    "CO2": 0.01665,
    "O2": 0.02630,
    "Ar": 0.01772,
    "CH4": 0.03400,
}
ATTENUATION_WEIGHT = {
    "N2": 0.10,
    "CO2": 0.40,
    "O2": 0.18,
    "Ar": 0.28,
    "CH4": 0.55,
}
PHASE_WEIGHT = {
    "N2": 0.010,
    "CO2": 0.080,
    "O2": 0.030,
    "Ar": -0.050,
    "CH4": 0.120,
}


@dataclass(frozen=True)
class CandidateProfile:
    candidate_id: str
    components: tuple[str, ...]
    baseline_composition: tuple[float, ...]
    safety_gate_required: bool


CANDIDATES = {
    "GIB-C4-LR": CandidateProfile(
        candidate_id="GIB-C4-LR",
        components=("N2", "CO2", "O2", "Ar"),
        baseline_composition=(0.55, 0.20, 0.20, 0.05),
        safety_gate_required=False,
    ),
    "GIB-C4-CH4": CandidateProfile(
        candidate_id="GIB-C4-CH4",
        components=("N2", "CO2", "O2", "CH4"),
        baseline_composition=(0.55, 0.20, 0.20, 0.05),
        safety_gate_required=True,
    ),
}


@dataclass(frozen=True)
class AuditConfig:
    candidate_id: str = "GIB-C4-LR"
    modalities: tuple[str, ...] = ("ndir", "acoustic_raw", "thermal", "slow", "calibration")
    noise_scale: float = 1.0
    nuisance_prior_scale: float = 1.0
    coupling_strength: float = 1.0
    target_signal_scale: float = 1.0
    fast_waveform: bool = True
    nonlinear_nuisance_coupling: bool = True
    cross_sensitivity: bool = True


@dataclass(frozen=True)
class Observation:
    values: np.ndarray
    noise_std: np.ndarray
    labels: tuple[str, ...]
    blocks: dict[str, slice]


@dataclass(frozen=True)
class AuditResult:
    candidate_id: str
    theta_free: np.ndarray
    eta: np.ndarray
    labels: tuple[str, ...]
    modality_blocks: dict[str, slice]
    whitened_j_theta: np.ndarray
    whitened_j_eta: np.ndarray
    target_fisher: np.ndarray
    effective_fisher: np.ndarray
    crb: np.ndarray
    crb_p90: np.ndarray
    joint_rank: int
    ranks_by_tolerance: dict[float, int]
    condition_number: float
    minimum_principal_angle_deg: float
    modality_sensitivity: dict[str, tuple[float, ...]]
    modality_effective_information: dict[str, tuple[float, ...]]
    modality_effective_information_share: dict[str, tuple[float, ...]]
    component_pair_similarity: dict[str, float]


def _validate_config(config: AuditConfig) -> CandidateProfile:
    if config.candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate_id: {config.candidate_id}")
    if not config.modalities:
        raise ValueError("at least one modality is required")
    known = {"ndir", "acoustic_raw", "acoustic_dsp", "thermal", "slow", "calibration"}
    unknown = set(config.modalities) - known
    if unknown:
        raise ValueError(f"unknown modalities: {sorted(unknown)}")
    if len(set(config.modalities)) != len(config.modalities):
        raise ValueError("modalities must not be duplicated")
    if "acoustic_raw" in config.modalities and "acoustic_dsp" in config.modalities:
        raise ValueError("acoustic_raw and acoustic_dsp are alternate views of one Raw source")
    for name in ("noise_scale", "nuisance_prior_scale", "target_signal_scale"):
        if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(config.coupling_strength) or config.coupling_strength < 0:
        raise ValueError("coupling_strength must be finite and non-negative")
    for name in ("fast_waveform", "nonlinear_nuisance_coupling", "cross_sensitivity"):
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")
    return CANDIDATES[config.candidate_id]


def _resolved_modalities(config: AuditConfig) -> tuple[str, ...]:
    """Resolve the Raw/DSP view without counting both views as independent."""

    if config.fast_waveform or "acoustic_raw" not in config.modalities:
        return config.modalities
    return tuple("acoustic_dsp" if name == "acoustic_raw" else name for name in config.modalities)


def _validate_eta(eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if eta.shape != (len(ETA_NAMES),):
        raise ValueError(f"eta must have {len(ETA_NAMES)} values")
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta must be finite")
    if eta[0] <= 0 or eta[1] <= 0 or eta[3] <= 0 or eta[4] <= 0 or eta[8] <= 0:
        raise ValueError("T_K, P_kPa, L_m, gain and q_flow must be positive")
    if not 0 <= eta[2] <= 1:
        raise ValueError("RH_frac must be in [0, 1]")
    return eta


def _composition_from_free(theta_free: Iterable[float], profile: CandidateProfile) -> np.ndarray:
    theta_free = np.asarray(tuple(theta_free), dtype=float)
    expected = len(profile.components) - 1
    if theta_free.shape != (expected,):
        raise ValueError(f"theta_free must have {expected} values")
    if not np.all(np.isfinite(theta_free)):
        raise ValueError("theta_free must be finite")
    last = 1.0 - float(np.sum(theta_free))
    composition = np.concatenate([theta_free, np.array([last])])
    if np.any(composition < -1.0e-12):
        raise ValueError("composition violates the simplex")
    if not np.isclose(float(np.sum(composition)), 1.0, atol=1.0e-12):
        raise ValueError("composition does not close to one")
    return composition


def _effective_composition(composition: np.ndarray, target_signal_scale: float) -> np.ndarray:
    reference = np.full(composition.shape, 1.0 / composition.size)
    effective = reference + target_signal_scale * (composition - reference)
    if np.any(effective < 0):
        raise ValueError("target_signal_scale leaves the simplex")
    return effective


def _nuisance_interaction_factor(factors: Iterable[float], enabled: bool) -> float:
    factors_array = np.asarray(tuple(factors), dtype=float)
    if enabled:
        return float(np.prod(factors_array))
    return float(1.0 + np.sum(factors_array - 1.0))


def _mixture_properties(composition: np.ndarray, components: tuple[str, ...], temperature_k: float) -> tuple[float, float, float, float]:
    masses = np.array([MOLAR_MASS[name] for name in components], dtype=float)
    cps = np.array([CP_MOLAR[name] for name in components], dtype=float)
    conductivities = np.array([THERMAL_CONDUCTIVITY[name] for name in components], dtype=float)
    mass_mix = float(composition @ masses)
    cp_mix = float(composition @ cps)
    gamma_mix = cp_mix / (cp_mix - 8.314462618)
    speed = float(np.sqrt(gamma_mix * 8.314462618 * temperature_k / mass_mix))
    conductivity = float(composition @ conductivities) * (temperature_k / 298.15) ** 0.80
    return mass_mix, cp_mix, speed, conductivity


def forward_observation(
    theta_free: Iterable[float],
    eta: Iterable[float] = ETA_DEFAULT,
    config: AuditConfig = AuditConfig(),
) -> Observation:
    """Evaluate deterministic observations and their declared noise scales."""

    profile = _validate_config(config)
    modalities = _resolved_modalities(config)
    eta = _validate_eta(np.asarray(tuple(eta), dtype=float))
    composition = _composition_from_free(theta_free, profile)
    composition = _effective_composition(composition, config.target_signal_scale)
    fractions = dict(zip(profile.components, composition))
    temperature_k, pressure_kpa, rh_frac, path_m, gain, baseline, delay_s, crosstalk, q_flow = eta
    coupling = config.coupling_strength
    effective_temperature_k = 298.15 + coupling * (temperature_k - 298.15)
    effective_pressure_kpa = 101.325 + coupling * (pressure_kpa - 101.325)
    effective_rh_frac = 0.50 + coupling * (rh_frac - 0.50)
    effective_path_m = 0.20 + coupling * (path_m - 0.20)
    effective_gain = 1.0 + coupling * (gain - 1.0)
    effective_baseline = coupling * baseline
    effective_delay_s = coupling * delay_s
    _, _, speed, conductivity = _mixture_properties(
        composition,
        profile.components,
        effective_temperature_k,
    )
    pressure_ratio = effective_pressure_kpa / 101.325
    temperature_ratio = 298.15 / effective_temperature_k
    path_ratio = effective_path_m / 0.20
    humidity_factor = 1.0 + 0.05 * (effective_rh_frac - 0.50)
    optical_environment = _nuisance_interaction_factor(
        (pressure_ratio, path_ratio, temperature_ratio),
        config.nonlinear_nuisance_coupling,
    )
    acoustic_environment = _nuisance_interaction_factor(
        (pressure_ratio, temperature_ratio, humidity_factor, path_ratio),
        config.nonlinear_nuisance_coupling,
    )
    cross = coupling * crosstalk if config.cross_sensitivity else 0.0

    blocks: dict[str, tuple[np.ndarray, np.ndarray, tuple[str, ...]]] = {}
    if "ndir" in modalities:
        co2_fraction = fractions.get("CO2", 0.0)
        absorption_co2 = 1.40 * co2_fraction * optical_environment
        optical_co2 = np.exp(-absorption_co2)
        if "CH4" in fractions:
            absorption_ch4 = 1.00 * fractions["CH4"] * optical_environment
            optical_second = np.exp(-absorption_ch4)
            second_label = "ndir_ch4"
        else:
            optical_second = 0.02 * (fractions.get("O2", 0.0) + fractions.get("Ar", 0.0)) + 0.005 * fractions.get("N2", 0.0)
            second_label = "ndir_null"
        optical = np.array([optical_co2, optical_second], dtype=float)
        cross_matrix = np.array([[1.0, cross], [cross, 1.0]], dtype=float)
        ndir_values = effective_gain * (cross_matrix @ optical) + effective_baseline
        blocks["ndir"] = (
            ndir_values,
            np.array([0.03, 0.03], dtype=float) * config.noise_scale,
            ("ndir_co2", second_label),
        )

    if "acoustic_raw" in modalities or "acoustic_dsp" in modalities:
        attenuation_weights = np.array([ATTENUATION_WEIGHT[name] for name in profile.components], dtype=float)
        attenuation = acoustic_environment * float(composition @ attenuation_weights)
        attenuation *= 1.0 + 0.20 * cross * fractions.get("CO2", 0.0)
        amplitude = float(np.exp(-attenuation * effective_path_m / 0.20))
        phase = float(composition @ np.array([PHASE_WEIGHT[name] for name in profile.components]))
        phase += 0.25 * cross * fractions.get("CO2", 0.0) * fractions.get("O2", 0.0)
        tof_norm = (effective_path_m / speed + effective_delay_s) / 1.0e-3
        speed_norm = speed / 350.0
        if "acoustic_raw" in modalities:
            blocks["acoustic_raw"] = (
                np.array([tof_norm, amplitude, phase, speed_norm], dtype=float),
                np.array([0.003, 0.002, 0.004, 0.002], dtype=float) * config.noise_scale,
                ("us_tof_raw", "us_amplitude_raw", "us_phase_raw", "us_speed_raw"),
            )
        else:
            blocks["acoustic_dsp"] = (
                np.array([speed_norm, attenuation, phase], dtype=float),
                np.array([0.002, 0.004, 0.004], dtype=float) * config.noise_scale,
                ("acoustic_tof_dsp", "acoustic_attenuation_dsp", "acoustic_phase_dsp"),
            )

    if "thermal" in modalities:
        conductivity_norm = conductivity / 0.025
        thermal_primary = effective_gain * (
            conductivity_norm + 0.25 * cross * fractions.get("CO2", 0.0) * fractions.get("O2", 0.0)
        ) + effective_baseline
        thermal_auxiliary = effective_gain * (0.50 * conductivity_norm + 0.20 * fractions.get("N2", 0.0)) + effective_baseline
        blocks["thermal"] = (
            np.array([thermal_primary, thermal_auxiliary], dtype=float),
            np.array([0.01, 0.01], dtype=float) * config.noise_scale,
            ("thermal_primary", "thermal_auxiliary"),
        )

    if "slow" in modalities:
        blocks["slow"] = (
            np.array(
                [
                    temperature_k / 298.15,
                    pressure_kpa / 101.325,
                    rh_frac,
                    q_flow,
                ],
                dtype=float,
            ),
            np.array([0.001, 0.001, 0.01, 0.02], dtype=float) * config.noise_scale,
            tuple(f"slow_{name}" for name in SLOW_CHANNEL_NAMES),
        )

    if "calibration" in modalities:
        blocks["calibration"] = (
            np.array(
                [
                    path_m / 0.20,
                    gain,
                    baseline,
                    delay_s / 1.0e-3,
                    crosstalk,
                ],
                dtype=float,
            ),
            np.array([0.005, 0.002, 0.01, 0.002, 0.01], dtype=float) * config.noise_scale,
            (
                "calibration_L_m",
                "calibration_gain",
                "calibration_baseline",
                "calibration_delay_s",
                "calibration_crosstalk",
            ),
        )

    values: list[np.ndarray] = []
    noise_std: list[np.ndarray] = []
    labels: list[str] = []
    block_slices: dict[str, slice] = {}
    offset = 0
    for modality in modalities:
        block_values, block_noise, block_labels = blocks[modality]
        end = offset + block_values.size
        block_slices[modality] = slice(offset, end)
        values.append(block_values)
        noise_std.append(block_noise)
        labels.extend(block_labels)
        offset = end
    if not values:
        raise ValueError("no observation block was produced")
    return Observation(
        values=np.concatenate(values),
        noise_std=np.concatenate(noise_std),
        labels=tuple(labels),
        blocks=block_slices,
    )


def _finite_difference_jacobian(
    fn,
    values: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    jacobian = np.empty((fn(values).size, values.size), dtype=float)
    for index, step in enumerate(steps):
        plus = values.copy()
        minus = values.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (fn(plus) - fn(minus)) / (2.0 * step)
    return jacobian


def _effective_fisher(
    whitened_j_theta: np.ndarray,
    whitened_j_eta: np.ndarray,
    nuisance_prior_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_fisher = whitened_j_theta.T @ whitened_j_theta
    cross_fisher = whitened_j_theta.T @ whitened_j_eta
    nuisance_fisher = whitened_j_eta.T @ whitened_j_eta
    nuisance_fisher += np.diag(1.0 / np.square(nuisance_prior_std))
    effective = target_fisher - cross_fisher @ np.linalg.solve(nuisance_fisher, cross_fisher.T)
    effective = 0.5 * (effective + effective.T)
    return target_fisher, effective


def _rank(matrix: np.ndarray, relative_tolerance: float) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    return int(np.sum(singular_values > relative_tolerance * singular_values[0]))


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)
    keep = singular_values > 1.0e-10 * singular_values[0]
    return left[:, keep]


def subspace_minimum_angle_deg(whitened_j_theta: np.ndarray, whitened_j_eta: np.ndarray) -> float:
    """Return the smallest principal angle between whitened target/nuisance spaces."""

    target_basis = _orthonormal_basis(whitened_j_theta)
    nuisance_basis = _orthonormal_basis(whitened_j_eta)
    if target_basis.shape[1] == 0 or nuisance_basis.shape[1] == 0:
        raise ValueError("both Jacobian subspaces must be non-empty")
    singular_values = np.linalg.svd(target_basis.T @ nuisance_basis, compute_uv=False)
    cosine = float(np.clip(singular_values[0], -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _component_directions(profile: CandidateProfile) -> dict[str, np.ndarray]:
    free_count = len(profile.components) - 1
    directions: dict[str, np.ndarray] = {}
    for index, name in enumerate(profile.components[:-1]):
        direction = np.zeros(free_count, dtype=float)
        direction[index] = 1.0
        directions[name] = direction
    directions[profile.components[-1]] = -np.ones(free_count, dtype=float)
    return directions


def _modality_metrics(
    result_j_theta: np.ndarray,
    result_j_eta: np.ndarray,
    blocks: dict[str, slice],
    nuisance_prior_std: np.ndarray,
) -> tuple[dict[str, tuple[float, ...]], dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    sensitivity: dict[str, tuple[float, ...]] = {}
    effective_information: dict[str, tuple[float, ...]] = {}
    for modality, block in blocks.items():
        target_block = result_j_theta[block]
        nuisance_block = result_j_eta[block]
        sensitivity[modality] = tuple(np.sqrt(np.sum(np.square(target_block), axis=0)).tolist())
        _, effective = _effective_fisher(target_block, nuisance_block, nuisance_prior_std)
        diagonal = np.diag(effective)
        if np.any(diagonal < -1.0e-10):
            raise RuntimeError(f"negative effective information in modality {modality}")
        effective_information[modality] = tuple(np.maximum(diagonal, 0.0).tolist())
    total = np.sum(np.array(list(effective_information.values()), dtype=float), axis=0)
    if np.any(total <= 0):
        raise RuntimeError("effective information is zero for at least one target direction")
    shares = {
        modality: tuple((np.array(values) / total).tolist())
        for modality, values in effective_information.items()
    }
    return sensitivity, effective_information, shares


def _component_pair_similarity(
    whitened_j_theta: np.ndarray,
    profile: CandidateProfile,
) -> dict[str, float]:
    directions = _component_directions(profile)
    component_vectors = {
        name: whitened_j_theta @ direction for name, direction in directions.items()
    }
    similarities: dict[str, float] = {}
    for left, right in combinations(profile.components, 2):
        left_vector = component_vectors[left]
        right_vector = component_vectors[right]
        left_norm = float(np.linalg.norm(left_vector))
        right_norm = float(np.linalg.norm(right_vector))
        if left_norm == 0 or right_norm == 0:
            raise RuntimeError(f"zero sensitivity for component pair {left}/{right}")
        similarities[f"{left}/{right}"] = float(left_vector @ right_vector / (left_norm * right_norm))
    return similarities


def analyze_point(
    theta_free: Iterable[float],
    eta: Iterable[float],
    config: AuditConfig = AuditConfig(),
) -> AuditResult:
    """Analyze one explicit composition and nuisance point."""

    profile = _validate_config(config)
    theta_free = np.asarray(tuple(theta_free), dtype=float)
    eta = _validate_eta(np.asarray(tuple(eta), dtype=float))
    _composition_from_free(theta_free, profile)
    base_observation = forward_observation(theta_free, eta, config)
    theta_steps = np.full(theta_free.shape, 1.0e-6, dtype=float)
    theta_jacobian = _finite_difference_jacobian(
        lambda values: forward_observation(values, eta, config).values,
        theta_free,
        theta_steps,
    )
    eta_jacobian = _finite_difference_jacobian(
        lambda values: forward_observation(theta_free, values, config).values,
        eta,
        ETA_STEPS,
    )
    whitened_j_theta = theta_jacobian / base_observation.noise_std[:, None]
    whitened_j_eta = eta_jacobian / base_observation.noise_std[:, None]
    prior_std = ETA_PRIOR_STD * config.nuisance_prior_scale
    target_fisher, effective_fisher = _effective_fisher(
        whitened_j_theta,
        whitened_j_eta,
        prior_std,
    )
    crb = np.linalg.inv(effective_fisher)
    crb = 0.5 * (crb + crb.T)
    # Nuisance columns are expressed in their declared prior units for rank
    # testing; otherwise delay_s (seconds) dominates the singular spectrum by
    # its unit scale rather than by an identifiability property.
    rank_matrix = np.column_stack([whitened_j_theta, whitened_j_eta * prior_std[None, :]])
    ranks = {tol: _rank(rank_matrix, tol) for tol in (1.0e-7, 1.0e-6, 1.0e-5)}
    sensitivity, effective_information, shares = _modality_metrics(
        whitened_j_theta,
        whitened_j_eta,
        base_observation.blocks,
        prior_std,
    )
    return AuditResult(
        candidate_id=profile.candidate_id,
        theta_free=theta_free,
        eta=eta,
        labels=base_observation.labels,
        modality_blocks=base_observation.blocks,
        whitened_j_theta=whitened_j_theta,
        whitened_j_eta=whitened_j_eta,
        target_fisher=target_fisher,
        effective_fisher=effective_fisher,
        crb=crb,
        crb_p90=1.645 * np.sqrt(np.diag(crb)),
        joint_rank=ranks[1.0e-6],
        ranks_by_tolerance=ranks,
        condition_number=float(np.linalg.cond(effective_fisher)),
        minimum_principal_angle_deg=subspace_minimum_angle_deg(whitened_j_theta, whitened_j_eta),
        modality_sensitivity=sensitivity,
        modality_effective_information=effective_information,
        modality_effective_information_share=shares,
        component_pair_similarity=_component_pair_similarity(whitened_j_theta, profile),
    )


def analyze_candidate(config: AuditConfig = AuditConfig()) -> AuditResult:
    profile = _validate_config(config)
    return analyze_point(profile.baseline_composition[:-1], ETA_DEFAULT, config)


def negative_controls(config: AuditConfig = AuditConfig()) -> dict[str, dict[str, object]]:
    baseline = analyze_candidate(config)
    profile = CANDIDATES[config.candidate_id]
    component_norms = {
        name: float(np.linalg.norm(baseline.whitened_j_theta @ direction))
        for name, direction in _component_directions(profile).items()
    }
    single_component_pass = all(value > 1.0e-8 for value in component_norms.values())

    scaled = analyze_candidate(replace(config, target_signal_scale=0.5))
    total_scaling_pass = (
        scaled.joint_rank == baseline.joint_rank
        and float(np.linalg.norm(scaled.whitened_j_theta)) < float(np.linalg.norm(baseline.whitened_j_theta))
    )

    modality_actual: dict[str, float] = {}
    full_information = float(np.trace(baseline.effective_fisher))
    modality_off_pass = True
    active_modalities = _resolved_modalities(config)
    for modality in active_modalities:
        remaining = tuple(name for name in active_modalities if name != modality)
        if not remaining:
            continue
        without = analyze_candidate(replace(config, modalities=remaining))
        actual = float(np.trace(without.effective_fisher))
        modality_actual[modality] = actual
        modality_off_pass = modality_off_pass and actual <= full_information + 1.0e-8

    noisy = analyze_candidate(replace(config, noise_scale=config.noise_scale * 2.0))
    noise_pass = bool(np.all(np.diag(noisy.crb) >= np.diag(baseline.crb) - 1.0e-8))
    repeated = analyze_candidate(config)
    deterministic_pass = bool(
        np.allclose(baseline.effective_fisher, repeated.effective_fisher, rtol=0.0, atol=1.0e-12)
        and np.allclose(baseline.crb, repeated.crb, rtol=0.0, atol=1.0e-12)
    )
    return {
        "single_component_perturbation": {
            "expected": "every physical component has nonzero whitened sensitivity",
            "actual": component_norms,
            "passed": single_component_pass,
        },
        "total_scaling": {
            "expected": "target sensitivity decreases while joint rank is unchanged",
            "actual": {
                "baseline_target_norm": float(np.linalg.norm(baseline.whitened_j_theta)),
                "scaled_target_norm": float(np.linalg.norm(scaled.whitened_j_theta)),
                "baseline_joint_rank": baseline.joint_rank,
                "scaled_joint_rank": scaled.joint_rank,
            },
            "passed": total_scaling_pass,
        },
        "modality_off": {
            "expected": "removing an active modality does not increase effective information",
            "actual": {"full_trace": full_information, "without_modality_trace": modality_actual},
            "passed": modality_off_pass,
        },
        "noise_monotonicity": {
            "expected": "doubling observation noise does not improve CRB",
            "actual": {
                "baseline_crb": np.diag(baseline.crb).tolist(),
                "double_noise_crb": np.diag(noisy.crb).tolist(),
            },
            "passed": noise_pass,
        },
        "deterministic_repeat": {
            "expected": "same configuration returns identical effective Fisher and CRB",
            "actual": {"passed": deterministic_pass},
            "passed": deterministic_pass,
        },
    }


def screen_candidate(config: AuditConfig = AuditConfig()) -> dict[str, object]:
    result = analyze_candidate(config)
    controls = negative_controls(config)
    passed = all(bool(value["passed"]) for value in controls.values())
    return {
        "candidate_id": result.candidate_id,
        "candidate_verdict": "candidate_selected" if passed and result.joint_rank >= len(result.theta_free) else "redesign_required",
        "safety_gate_required": CANDIDATES[result.candidate_id].safety_gate_required,
        "result": result,
        "negative_controls": controls,
    }


def _observation_sha256(observation: Observation) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(observation.labels, separators=(",", ":")).encode("utf-8"))
    digest.update(np.asarray(observation.values, dtype="<f8").tobytes())
    digest.update(np.asarray(observation.noise_std, dtype="<f8").tobytes())
    return digest.hexdigest().upper()


def _component_perturbation_audit(
    config: AuditConfig,
    audit_config: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    profile = CANDIDATES[config.candidate_id]
    baseline_composition = np.asarray(profile.baseline_composition, dtype=float)
    baseline = forward_observation(baseline_composition[:-1], ETA_DEFAULT, config)
    delta = float(audit_config["component_delta"])
    minimum_norm = float(audit_config["minimum_whitened_response_norm"])
    maximum_invariant = float(audit_config["maximum_invariant_abs_response"])
    maximum_antisymmetry_error = float(audit_config["maximum_antisymmetry_relative_error"])
    component_index = {name: index for index, name in enumerate(profile.components)}
    label_index = {label: index for index, label in enumerate(baseline.labels)}
    results: dict[str, dict[str, object]] = {}

    for component in profile.components:
        reference = str(audit_config["exchange_reference"][component])
        direction = np.zeros(len(profile.components), dtype=float)
        direction[component_index[component]] = 1.0
        direction[component_index[reference]] = -1.0
        positive_composition = baseline_composition + delta * direction
        negative_composition = baseline_composition - delta * direction
        positive = forward_observation(positive_composition[:-1], ETA_DEFAULT, config)
        negative = forward_observation(negative_composition[:-1], ETA_DEFAULT, config)
        positive_delta = positive.values - baseline.values
        negative_delta = negative.values - baseline.values

        modality_norms = {
            modality: float(np.linalg.norm(positive_delta[block] / baseline.noise_std[block]))
            for modality, block in baseline.blocks.items()
        }
        modality_max_abs = {
            modality: float(np.max(np.abs(positive_delta[block])))
            for modality, block in baseline.blocks.items()
        }
        responsive = [str(item) for item in audit_config["responsive_modalities"][component]]
        invariant = [str(item) for item in audit_config["invariant_modalities"][component]]
        response_pass = all(modality_norms[name] >= minimum_norm for name in responsive)
        invariant_pass = all(modality_max_abs[name] <= maximum_invariant for name in invariant)

        primary = audit_config["primary_response"][component]
        primary_label = str(primary["label"])
        primary_delta = float(positive_delta[label_index[primary_label]])
        sign_pass = int(np.sign(primary_delta)) == int(primary["sign"])
        response_scale = max(float(np.linalg.norm(positive_delta)), float(np.linalg.norm(negative_delta)))
        antisymmetry_error = float(np.linalg.norm(positive_delta + negative_delta)) / response_scale
        antisymmetry_pass = antisymmetry_error <= maximum_antisymmetry_error
        passed = response_pass and invariant_pass and sign_pass and antisymmetry_pass
        results[component] = {
            "exchange_reference": reference,
            "delta_mol_per_mol": delta,
            "primary_label": primary_label,
            "expected_primary_sign": int(primary["sign"]),
            "actual_primary_delta": primary_delta,
            "responsive_modalities": responsive,
            "invariant_modalities": invariant,
            "whitened_response_norm_by_modality": modality_norms,
            "maximum_abs_response_by_modality": modality_max_abs,
            "antisymmetry_relative_error": antisymmetry_error,
            "response_magnitude_pass": response_pass,
            "invariant_modality_pass": invariant_pass,
            "response_sign_pass": sign_pass,
            "positive_negative_antisymmetry_pass": antisymmetry_pass,
            "passed": passed,
        }
    return results


def _target_signal_scaling_audit(
    config: AuditConfig,
    scales: Iterable[float],
) -> dict[str, object]:
    records = []
    for scale in scales:
        result = analyze_candidate(replace(config, target_signal_scale=float(scale)))
        records.append(
            {
                "target_signal_scale": float(scale),
                "whitened_target_norm": float(np.linalg.norm(result.whitened_j_theta)),
                "effective_fisher_trace": float(np.trace(result.effective_fisher)),
                "joint_rank": result.joint_rank,
            }
        )
    target_norms = [record["whitened_target_norm"] for record in records]
    fisher_traces = [record["effective_fisher_trace"] for record in records]
    rank_values = {record["joint_rank"] for record in records}
    passed = (
        all(left < right for left, right in zip(target_norms, target_norms[1:]))
        and all(left < right for left, right in zip(fisher_traces, fisher_traces[1:]))
        and len(rank_values) == 1
    )
    return {
        "expected": "target norm and effective information increase strictly without a hidden zero direction",
        "records": records,
        "passed": passed,
    }


def _modality_off_audit(
    config: AuditConfig,
    required_modalities: Iterable[str],
) -> dict[str, object]:
    baseline = analyze_candidate(config)
    full_trace = float(np.trace(baseline.effective_fisher))
    records: dict[str, dict[str, object]] = {}
    for modality in required_modalities:
        remaining = tuple(item for item in config.modalities if item != modality)
        without = analyze_candidate(replace(config, modalities=remaining))
        without_trace = float(np.trace(without.effective_fisher))
        removed_labels = sorted(set(baseline.labels) - set(without.labels))
        records[modality] = {
            "removed_labels": removed_labels,
            "remaining_effective_fisher_trace": without_trace,
            "information_did_not_increase": without_trace <= full_trace + 1.0e-8,
            "passed": bool(removed_labels) and without_trace <= full_trace + 1.0e-8,
        }
    return {
        "full_effective_fisher_trace": full_trace,
        "modalities": records,
        "passed": all(record["passed"] for record in records.values()),
    }


def _noise_monotonicity_audit(
    config: AuditConfig,
    scales: Iterable[float],
) -> dict[str, object]:
    records = []
    for scale in scales:
        result = analyze_candidate(replace(config, noise_scale=float(scale)))
        records.append(
            {
                "noise_scale": float(scale),
                "crb_diagonal": np.diag(result.crb).tolist(),
                "crb_p90": result.crb_p90.tolist(),
                "maximum_crb_p90": float(np.max(result.crb_p90)),
            }
        )
    maxima = [record["maximum_crb_p90"] for record in records]
    componentwise_pass = all(
        np.all(np.asarray(right["crb_p90"]) > np.asarray(left["crb_p90"]))
        for left, right in zip(records, records[1:])
    )
    return {
        "records": records,
        "strict_componentwise_monotonicity": componentwise_pass,
        "strict_maximum_monotonicity": all(left < right for left, right in zip(maxima, maxima[1:])),
        "passed": componentwise_pass and all(left < right for left, right in zip(maxima, maxima[1:])),
    }


def g3_1_forward_audit(audit_config: Mapping[str, Any]) -> dict[str, object]:
    """Run the preregistered P3 G3-1 forward and negative-control audit."""

    config = AuditConfig(
        candidate_id=str(audit_config["candidate_id"]),
        modalities=tuple(str(item) for item in audit_config["modalities"]),
    )
    profile = CANDIDATES[config.candidate_id]
    theta_free = np.asarray(profile.baseline_composition[:-1], dtype=float)
    first = forward_observation(theta_free, ETA_DEFAULT, config)
    second = forward_observation(theta_free, ETA_DEFAULT, config)
    first_hash = _observation_sha256(first)
    second_hash = _observation_sha256(second)
    deterministic = {
        "first_observation_sha256": first_hash,
        "second_observation_sha256": second_hash,
        "values_exactly_equal": bool(np.array_equal(first.values, second.values)),
        "noise_exactly_equal": bool(np.array_equal(first.noise_std, second.noise_std)),
        "passed": first_hash == second_hash and np.array_equal(first.values, second.values) and np.array_equal(first.noise_std, second.noise_std),
    }
    components = _component_perturbation_audit(config, audit_config)
    target_scaling = _target_signal_scaling_audit(config, audit_config["target_signal_scales"])
    modality_off = _modality_off_audit(config, audit_config["required_modality_off_checks"])
    noise = _noise_monotonicity_audit(config, audit_config["noise_scales"])

    from .s2_s3 import audit_summary

    s3 = audit_summary()["s3"]
    s3_switches = s3["switches"]
    all_off = s3["all_off_negative_control"]
    all_off["target_profile_eligible"] = False
    all_off_pass = bool(all_off["negative_control_only"] and all_off["passed"] and not all_off["target_profile_eligible"])
    checks = {
        "deterministic_repeat": deterministic,
        "component_perturbations": {
            "components": components,
            "passed": all(record["passed"] for record in components.values()),
        },
        "target_signal_scaling": target_scaling,
        "modality_off": modality_off,
        "s3_switches": {
            "switches": s3_switches,
            "passed": all(record["passed"] for record in s3_switches.values()),
        },
        "noise_monotonicity": noise,
        "all_off_negative_control": {**all_off, "passed": all_off_pass},
    }
    passed = all(bool(check["passed"]) for check in checks.values())
    return {
        "schema_version": "gib-benchmark-1",
        "audit_id": str(audit_config["audit_id"]),
        "task_id": "P3-01",
        "candidate_id": config.candidate_id,
        "claim_scope": "controlled_synthetic_relative_comparison_only",
        "checks": checks,
        "gate_verdict": "pass" if passed else "fail",
        "next_allowed_task": "P3-02" if passed else "P2-05_or_P2-07",
    }
