"""P3-06 orchestration from the frozen pilot into the solver owner."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..audit.solver import SeparableProblem, SolverTrial, run_solver_preflight
from ..common.io import atomic_promote_directory, atomic_write_json, remove_owned_staging
from ..sim.packaging.arrays import read_array_artifact
from .baseline import _partition_indices, load_pilot_metadata


def _basis(time_axis: np.ndarray):
    def evaluate(nonlinear: np.ndarray) -> np.ndarray:
        rate = float(nonlinear[0])
        return np.column_stack(
            [
                np.ones_like(time_axis),
                np.exp(-rate * time_axis),
                np.sin(2.0 * np.pi * rate * time_axis),
                np.cos(2.0 * np.pi * rate * time_axis),
            ]
        )

    return evaluate


def build_solver_trials(
    pilot_freeze: Path,
    config: dict[str, Any],
    *,
    partition: str = "test",
    mixtures_per_cell: int | None = None,
) -> list[SolverTrial]:
    metadata = load_pilot_metadata(pilot_freeze)
    time_axis = np.linspace(0.0, 1.0, 32)
    trials = []
    for split_id in config["split_ids"]:
        indices = _partition_indices(metadata, split_id)[partition]
        selected = [index for index in indices if int(metadata.records[int(index)]["sequence_index"]) == 0]
        if mixtures_per_cell is not None:
            limited: list[int] = []
            for cell_id in config["grid_cell_ids"]:
                local = [index for index in selected if metadata.records[int(index)]["grade"]["grid_cell_id"] == cell_id]
                limited.extend(local[:mixtures_per_cell])
            selected = limited
        for seed in config["seeds"]:
            for index in selected:
                record = metadata.records[int(index)]
                truth = read_array_artifact(metadata.root / record["arrays"]["labels"]["file_ref"])
                raw = read_array_artifact(metadata.root / record["arrays"]["raw_waveform"]["file_ref"])
                rate = float(record["nuisance"]["T"]) / 298.15
                basis = _basis(time_axis)
                noise = raw[0, :32] - np.mean(raw[0, :32])
                noise_scale = np.std(noise)
                if noise_scale == 0.0:
                    raise RuntimeError("solver trial raw channel has zero variance")
                observations = basis(np.asarray([rate])) @ truth + noise / noise_scale * 1.0e-5
                trials.append(
                    SolverTrial(
                        mixture_id=str(record["mixture_id"]),
                        sequence_id=str(record["sequence_id"]),
                        grid_cell_id=str(record["grade"]["grid_cell_id"]),
                        split_id=str(split_id),
                        seed=int(seed),
                        problem=SeparableProblem(
                            observations=observations,
                            basis=basis,
                            linear_initial=np.full(4, 0.25),
                            nonlinear_initial=np.asarray([1.0]),
                        ),
                        truth_linear=np.asarray(truth, dtype=np.float64),
                        information_band=str(record["grade"]["information_band"]),
                        hardware_fingerprint="GIB-HW-WIN-R9-8940HX-RTX5060L-20260825",
                    )
                )
    return trials


def _runtime_metadata(registry: dict[str, Any], git_commit: str) -> dict[str, Any]:
    runtime = registry["runtime"]
    hardware = registry["hardware"]
    return {
        "logical_cpu_count": int(hardware["cpu_logical_processors"]),
        "blas_threads": int(runtime["blas"]["threads"]),
        "omp_threads": int(runtime["openmp_threads"]),
        "mkl_threads": int(runtime["blas"]["threads"]),
        "framework_threads": int(runtime["torch_openmp_threads"]),
        "os_version": str(hardware["os"]),
        "python_version": str(runtime["python"]),
        "numpy_version": str(runtime["numpy"]),
        "framework_version": f"numpy-{runtime['numpy']}",
        "method_package_versions": f"numpy=={runtime['numpy']};scipy=={runtime['scipy']}",
        "git_commit": git_commit,
    }


def execute_solver_preflight(
    config: dict[str, Any],
    *,
    pilot_freeze: Path,
    execution_registry: dict[str, Any],
    git_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        result = run_solver_preflight(
            build_solver_trials(pilot_freeze, config),
            config,
            runtime_metadata=_runtime_metadata(execution_registry, git_commit),
        )
        result["claim_scope"] = config["claim_scope"]
        result["next_allowed_task"] = "P3-10" if result["activated_tasks"]["P3-10"] else "candidate_terminal_fail"
        atomic_write_json(staging / "solver_preflight_results.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": "P3-06",
                "status": "complete",
                "task_status": "completed",
                "c2_preflight": result["c2_preflight"],
                "claim_scope": result["claim_scope"],
                "next_allowed_task": result["next_allowed_task"],
            },
        )
        atomic_promote_directory(staging, target)
        return result
    except Exception:
        remove_owned_staging(staging)
        raise


__all__ = ["build_solver_trials", "execute_solver_preflight"]
