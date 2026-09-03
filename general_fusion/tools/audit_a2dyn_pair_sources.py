"""Audit the v2 pair asset against its offline source evaluators.

The report is diagnostic until every pair has an independent validation
dataset.  TAPPS supplies its analytic temperature derivatives; the other
source snapshots are differentiated numerically here only as a read-only
source oracle.  The production evaluator remains the analytic Chebyshev
representation in ``gf.sim.a2dyn_pair_virial``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_BUILDER_PATH = PROJECT_ROOT / "tools" / "build_a2dyn_pair_virial_v2.py"
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gf.sim.a2dyn_pair_virial import (  # noqa: E402
    PAIR_IDS,
    PAIR_TEMPERATURE_RANGE_K,
    pair_virial_terms,
)


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("a2dyn_pair_asset_builder", SOURCE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source builder: {SOURCE_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _five_point_terms(
    source_function: Callable[[np.ndarray], np.ndarray],
    temperatures: np.ndarray,
    *,
    step_k: float = 1.0e-2,
) -> np.ndarray:
    center = source_function(temperatures)
    left_one = source_function(temperatures - step_k)
    right_one = source_function(temperatures + step_k)
    left_two = source_function(temperatures - 2.0 * step_k)
    right_two = source_function(temperatures + 2.0 * step_k)
    first = (left_two - 8.0 * left_one + 8.0 * right_one - right_two) / (12.0 * step_k)
    second = (
        -right_two
        + 16.0 * right_one
        - 30.0 * center
        + 16.0 * left_one
        - left_two
    ) / (12.0 * step_k**2)
    return np.column_stack((center, first, second))


def _tapps_terms(builder: Any, tapps_root: Path, data_name: str, temperatures: np.ndarray) -> np.ndarray:
    evaluator_type = builder._load_tapps(tapps_root)
    evaluator = evaluator_type(str(tapps_root / "data" / data_name))
    result = []
    for temperature in temperatures:
        temperature = float(temperature)
        b = float(evaluator.B(temperature)) * 1.0e-6
        first = float(evaluator.TdBdT(temperature)) * 1.0e-6 / temperature
        second = float(evaluator.T2d2BdT2(temperature)) * 1.0e-6 / temperature**2
        result.append((b, first, second))
    return np.asarray(result, dtype=np.float64)


def _source_terms(
    builder: Any,
    pair_id: str,
    tapps_root: Path,
    rsc_source: Path,
    temperatures: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    if pair_id == "Ar-Ar":
        return (
            _tapps_terms(builder, tapps_root, "Ar40_phase_shift_data.json.bz2", temperatures),
            "TAPPS Ar40 phase-shift integral",
            "TAPPS analytic TdBdT and T2d2BdT2",
        )
    if pair_id == "He-He":
        return (
            _tapps_terms(builder, tapps_root, "He4_phase_shift_data.json.bz2", temperatures),
            "TAPPS He4 phase-shift integral",
            "TAPPS analytic TdBdT and T2d2BdT2",
        )
    gas_i, gas_j = pair_id.split("-")
    if pair_id == "He-CO2":
        source_function = lambda values: builder._rsc_values(rsc_source, values)
        description = "RSC Nemati-Kande fitted PES classical anisotropic integral"
    else:
        source_function = lambda values: builder._coolprop_values(gas_i, gas_j, values)
        description = f"CoolProp zero-density source evaluator for {pair_id}"
    return (
        _five_point_terms(source_function, temperatures),
        description,
        "five-point derivative of the source B(T) oracle",
    )


def build_report(tapps_root: Path, rsc_source: Path, *, temperature_count: int = 71) -> dict[str, Any]:
    if temperature_count < 5:
        raise ValueError("temperature_count must be at least 5")
    lower, upper = PAIR_TEMPERATURE_RANGE_K
    temperatures = np.linspace(lower, upper, temperature_count, dtype=np.float64)
    builder = _load_builder()
    pair_reports: dict[str, Any] = {}
    for pair_id in PAIR_IDS:
        source_terms, source_description, derivative_oracle = _source_terms(
            builder, pair_id, tapps_root, rsc_source, temperatures
        )
        model_terms = np.asarray(
            [pair_virial_terms(*pair_id.split("-"), float(temperature)) for temperature in temperatures],
            dtype=np.float64,
        )
        difference = model_terms - source_terms
        relative_difference = np.abs(difference) / np.maximum(np.abs(source_terms), 1.0e-30)
        pair_reports[pair_id] = {
            "source_description": source_description,
            "derivative_oracle": derivative_oracle,
            "temperature_range_k": [lower, upper],
            "temperature_count": int(temperature_count),
            "max_abs_error": {
                "B_m3_mol": float(np.max(np.abs(difference[:, 0]))),
                "dB_dT_m3_mol_k": float(np.max(np.abs(difference[:, 1]))),
                "d2B_dT2_m3_mol_k2": float(np.max(np.abs(difference[:, 2]))),
            },
            "max_relative_error": {
                "B": float(np.max(relative_difference[:, 0])),
                "dB_dT": float(np.max(relative_difference[:, 1])),
                "d2B_dT2": float(np.max(relative_difference[:, 2])),
            },
        }
    return {
        "schema_version": "gf-a2dyn-pair-source-audit-1",
        "model_id": "a2dyn_cp_t_pair_virial_v2",
        "status": "COMPUTED_NOT_CERTIFIED",
        "temperature_range_k": list(PAIR_TEMPERATURE_RANGE_K),
        "tapps_root": str(tapps_root),
        "rsc_source": str(rsc_source),
        "independent_validation": "NOT_RUN",
        "pairs": pair_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapps-root", type=Path, required=True)
    parser.add_argument("--rsc-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temperature-count", type=int, default=71)
    args = parser.parse_args()
    report = build_report(args.tapps_root, args.rsc_source, temperature_count=args.temperature_count)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
