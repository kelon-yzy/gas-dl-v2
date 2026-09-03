"""Build the A2-DYN pair-virial draft asset from registered source snapshots.

This is an offline builder.  The formal simulation path does not import
CoolProp or TAPPS; it consumes the generated Chebyshev coefficients only.
The source roots are explicit arguments so a build can be reproduced against
the hashes recorded in the source manifest.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np


TEMPERATURE_RANGE_K = (278.15, 313.15)
PAIR_IDS = ("Ar-Ar", "Ar-He", "Ar-CO2", "He-He", "He-CO2", "CO2-CO2")
COOLPROP_FLUIDS = {"Ar": "Argon", "He": "Helium", "CO2": "CarbonDioxide"}
GENERATOR_REFERENCE = "general_fusion/tools/build_a2dyn_pair_virial_v2.py"


def _load_tapps(tapps_root: Path) -> Any:
    module_path = tapps_root / "thermophysicalPairProperties.py"
    spec = importlib.util.spec_from_file_location("a2dyn_tapps_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TAPPS source module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThermophysicalPairProperties


def _tapps_values(tapps_root: Path, data_name: str, temperatures: np.ndarray) -> np.ndarray:
    evaluator_type = _load_tapps(tapps_root)
    evaluator = evaluator_type(str(tapps_root / "data" / data_name))
    values = []
    for temperature in temperatures:
        b_cm3 = float(evaluator.B(float(temperature)))
        values.append(b_cm3 * 1.0e-6)
    return np.asarray(values, dtype=np.float64)


def _coolprop_values(gas_i: str, gas_j: str, temperatures: np.ndarray) -> np.ndarray:
    import CoolProp.CoolProp as coolprop

    if gas_i == gas_j:
        return np.asarray(
            [
                float(
                    coolprop.PropsSI(
                        "Bvirial",
                        "T",
                        float(temperature),
                        "P",
                        101325.0,
                        COOLPROP_FLUIDS[gas_i],
                    )
                )
                for temperature in temperatures
            ],
            dtype=np.float64,
        )

    state = coolprop.AbstractState(
        "HEOS", f"{COOLPROP_FLUIDS[gas_i]}&{COOLPROP_FLUIDS[gas_j]}"
    )
    state.set_mole_fractions([0.5, 0.5])
    pure_i = _coolprop_values(gas_i, gas_i, temperatures)
    pure_j = _coolprop_values(gas_j, gas_j, temperatures)
    mixture = []
    for temperature in temperatures:
        state.update(coolprop.PT_INPUTS, 101325.0, float(temperature))
        mixture.append(float(state.keyed_output(coolprop.iBvirial)))
    return 2.0 * np.asarray(mixture) - 0.5 * (pure_i + pure_j)


def _fortran_array(source: str, name: str, expected_size: int) -> np.ndarray:
    match = re.search(rf"\bDATA\s+{name}\s*/(.*?)/", source, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(f"Fortran source does not contain DATA array {name}")
    tokens = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?",
        match.group(1),
    )
    values = np.asarray([float(token.replace("D", "E").replace("d", "e")) for token in tokens])
    if values.size != expected_size:
        raise ValueError(f"Fortran DATA {name} has {values.size} values, expected {expected_size}")
    return values


def _load_rsc_potential(source_path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    source = source_path.read_text(encoding="ascii")
    coefficients: dict[int, np.ndarray] = {}
    powers: dict[int, np.ndarray] = {}
    for order in range(0, 24, 2):
        coefficients[order] = _fortran_array(source, f"pa{order}", 8)
        powers[order] = _fortran_array(source, f"Cn{order}", 3)
    return coefficients, powers


def _rsc_potential_values(
    radius_angstrom: np.ndarray,
    cosine_theta: np.ndarray,
    coefficients: dict[int, np.ndarray],
    powers: dict[int, np.ndarray],
) -> np.ndarray:
    legendre = np.polynomial.legendre.legvander(cosine_theta, 22)
    potential = np.zeros((radius_angstrom.size, cosine_theta.size), dtype=np.float64)
    for order in range(0, 24, 2):
        pa = coefficients[order]
        cn = powers[order]
        radius = radius_angstrom
        v_order = (
            -pa[0] * (1.0 - (1.0 - np.exp(-pa[2] * (radius - pa[1]))) ** 2.0)
            + pa[3] * np.exp(-pa[4] * radius**2.0)
            + pa[5] / radius**cn[0]
            + pa[6] / radius**cn[1]
            + pa[7] / radius**cn[2]
        )
        potential += v_order[:, None] * legendre[None, :, order]
    return potential


def _rsc_values(source_path: Path, temperatures: np.ndarray) -> np.ndarray:
    coefficients, powers = _load_rsc_potential(source_path)
    angle_nodes, angle_weights = np.polynomial.legendre.leggauss(48)
    radial_nodes: list[np.ndarray] = []
    radial_weights: list[np.ndarray] = []
    radial_edges = (2.5, 5.0, 8.0, 10.0, 15.0, 25.0, 50.0, 100.0)
    segment_nodes, segment_weights = np.polynomial.legendre.leggauss(64)
    for lower, upper in zip(radial_edges[:-1], radial_edges[1:]):
        radial_nodes.append(0.5 * (upper - lower) * segment_nodes + 0.5 * (upper + lower))
        radial_weights.append(0.5 * (upper - lower) * segment_weights)
    radius = np.concatenate(radial_nodes)
    weights = np.concatenate(radial_weights)
    potential_cm = _rsc_potential_values(radius, angle_nodes, coefficients, powers)
    potential_j = potential_cm * 100.0 * 6.62607015e-34 * 299792458.0
    radial_weight = weights * radius**2.0
    angular_weight = 0.5 * angle_weights
    scale = -2.0 * math.pi * 6.02214076e23 * 1.0e-30
    hard_core_integral = -(2.5**3) / 3.0
    values = []
    for temperature in temperatures:
        exponent = -potential_j / (1.380649e-23 * float(temperature))
        boltzmann_minus_one = np.expm1(exponent)
        angular_average = boltzmann_minus_one @ angular_weight
        integral = hard_core_integral + float(np.dot(radial_weight, angular_average))
        values.append(scale * integral)
    return np.asarray(values, dtype=np.float64)


def _chebyshev_coefficients(
    source_function: Callable[[np.ndarray], np.ndarray],
    degree: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    lower, upper = TEMPERATURE_RANGE_K
    sample_index = np.arange(sample_count, dtype=np.float64)
    z_samples = np.cos((2.0 * sample_index + 1.0) * math.pi / (2.0 * sample_count))
    temperatures = 0.5 * (lower + upper) + 0.5 * (upper - lower) * z_samples
    values = source_function(temperatures)
    coefficients = np.polynomial.chebyshev.chebfit(z_samples, values, degree)
    check_index = np.linspace(0.0, 1.0, 401)
    check_temperatures = lower + (upper - lower) * check_index
    check_z = (2.0 * check_temperatures - lower - upper) / (upper - lower)
    check_error = np.max(
        np.abs(np.polynomial.chebyshev.chebval(check_z, coefficients) - source_function(check_temperatures))
    )
    return temperatures, coefficients, float(check_error)


def _pair_source_metadata() -> dict[str, dict[str, Any]]:
    return {
        "Ar-Ar": {
            "source_snapshot_ids": ["tapps-github-main-d930ec74"],
            "source_evaluator": "TAPPS phase-shift integral for Ar40",
            "fit_dataset_ids": ["tapps-Ar40-phase-shift"],
            "validation_dataset_ids": ["ar-acoustic-virial-1989"],
        },
        "He-He": {
            "source_snapshot_ids": ["tapps-github-main-d930ec74"],
            "source_evaluator": "TAPPS phase-shift integral for He4",
            "fit_dataset_ids": ["tapps-He4-phase-shift"],
            "validation_dataset_ids": ["nist-he-eos-8474"],
        },
        "CO2-CO2": {
            "source_snapshot_ids": ["coolprop-source-8.0.0-61b616ed"],
            "source_evaluator": "Span-Wagner Helmholtz zero-density limit",
            "fit_dataset_ids": ["coolprop-SpanWagner-CO2-zero-density"],
            "validation_dataset_ids": ["co2-independent-virial-required"],
        },
        "Ar-He": {
            "source_snapshot_ids": ["coolprop-source-8.0.0-61b616ed"],
            "source_evaluator": "Tkaczuk Gaussian-exponential departure zero-density limit",
            "fit_dataset_ids": ["coolprop-Tkaczuk-ArHe-zero-density"],
            "validation_dataset_ids": ["brewer-vaughn-1969", "blancett-1970"],
        },
        "Ar-CO2": {
            "source_snapshot_ids": ["coolprop-source-8.0.0-61b616ed"],
            "source_evaluator": "Gernert GERG-2008 departure zero-density limit",
            "fit_dataset_ids": ["coolprop-Gernert-ArCO2-zero-density"],
            "validation_dataset_ids": ["nist-ArCO2-ThermoML", "arco2-acoustic-2016"],
        },
        "He-CO2": {
            "source_snapshot_ids": ["rsc-he-co2-si-zip-20260901"],
            "source_evaluator": "Nemati-Kande fitted PES classical anisotropic second-virial integral",
            "fit_dataset_ids": ["rsc-NematiKande-HeCO2-PES-classical-integral"],
            "validation_dataset_ids": ["holste-heco2-1980", "heco2-quantum-correction-required"],
        },
    }


def build_asset(tapps_root: Path, rsc_source: Path, generator_path: Path) -> dict[str, Any]:
    lower, upper = TEMPERATURE_RANGE_K
    metadata = _pair_source_metadata()
    temperatures = np.linspace(lower, upper, 25, dtype=np.float64)
    sources: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Ar-Ar": lambda values: _tapps_values(tapps_root, "Ar40_phase_shift_data.json.bz2", values),
        "He-He": lambda values: _tapps_values(tapps_root, "He4_phase_shift_data.json.bz2", values),
        "CO2-CO2": lambda values: _coolprop_values("CO2", "CO2", values),
        "Ar-He": lambda values: _coolprop_values("Ar", "He", values),
        "Ar-CO2": lambda values: _coolprop_values("Ar", "CO2", values),
        "He-CO2": lambda values: _rsc_values(rsc_source, values),
    }
    pairs: dict[str, Any] = {}
    for pair_id in PAIR_IDS:
        _, coefficients, max_fit_error = _chebyshev_coefficients(
            sources[pair_id], degree=12, sample_count=25
        )
        pairs[pair_id] = {
            "representation": "chebyshev_polynomial",
            "temperature_domain_k": [lower, upper],
            "normalization": "z=(2*T-T_min-T_max)/(T_max-T_min)",
            "B_m3_mol_coefficients": [float(value) for value in coefficients],
            "derivative_evaluation": "analytic_chebyshev_derivatives",
            "fit_max_abs_error_m3_mol": max_fit_error,
            **metadata[pair_id],
        }
    return {
        "schema_version": "gf-a2dyn-eos-coefficients-2",
        "model_id": "a2dyn_cp_t_pair_virial_v2",
        "coefficient_version": "a2dyn-eos-coefficients-20260901-r2-draft1",
        "status": "DRAFT_SOURCE_PARITY_PENDING_INDEPENDENT_VALIDATION",
        "temperature_range_k": [lower, upper],
        "pressure_range_pa": [90000.0, 112000.0],
        "composition_components": ["Ar", "He", "CO2"],
        "ideal_heat_capacity": {
            "source_asset": "configs/data/a2dyn_eos_coefficients_v1.json",
            "source_field": "ideal_heat_capacity",
            "reuse_policy": "single_registered_NASA7_source",
        },
        "virial": {
            "representation": "pair-specific-chebyshev-surrogate",
            "units": {
                "B_m3_mol": "m^3/mol",
                "dB_dT_m3_mol_k": "m^3/(mol*K)",
                "d2B_dT2_m3_mol_k2": "m^3/(mol*K^2)",
            },
            "pair_ids": list(PAIR_IDS),
            "pair_symmetry": "canonical_pair_id_only",
            "derivative_policy": "analytic_chebyshev_derivatives",
            "surrogate_gate": {
                "max_sound_speed_propagation_error": 0.00005,
                "status": "NOT_CERTIFIED",
            },
            "generator": {
                "path": str(generator_path).replace("\\", "/"),
                "temperature_fit_nodes": 25,
                "chebyshev_degree": 12,
                "source_evaluator_is_offline_only": True,
            },
            "pairs": pairs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapps-root", type=Path, required=True)
    parser.add_argument("--rsc-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asset = build_asset(args.tapps_root, args.rsc_source, Path(GENERATOR_REFERENCE))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
