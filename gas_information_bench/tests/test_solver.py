from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gib.audit.solver import (
    SeparableProblem,
    SolverError,
    SolverOptions,
    SolverTrial,
    evaluate_c2_preflight,
    paired_group_bootstrap,
    run_solver_preflight,
    solve_classical_vp,
    solve_joint_lm,
    solve_projection_disabled_control,
    solve_tsvd_ridge_vp,
    solve_vplr,
)


ROOT = Path(__file__).resolve().parents[1]


RUNTIME_METADATA = {
    "logical_cpu_count": 1,
    "blas_threads": 1,
    "omp_threads": 1,
    "mkl_threads": 1,
    "framework_threads": 1,
    "os_version": "test-os",
    "python_version": "test-python",
    "numpy_version": np.__version__,
    "framework_version": f"numpy-{np.__version__}",
    "method_package_versions": f"numpy=={np.__version__}",
    "git_commit": "test-worktree",
}


def _problem() -> tuple[SeparableProblem, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 2.0, 31)

    def basis(nonlinear: np.ndarray) -> np.ndarray:
        rate = nonlinear[0]
        return np.column_stack([np.ones_like(time), np.exp(-rate * time)])

    truth_linear = np.array([0.3, 0.7])
    truth_nonlinear = np.array([0.8])
    observations = basis(truth_nonlinear) @ truth_linear
    return (
        SeparableProblem(
            observations=observations,
            basis=basis,
            linear_initial=np.array([0.25, 0.75]),
            nonlinear_initial=np.array([0.65]),
        ),
        truth_linear,
        truth_nonlinear,
    )


def _plan() -> dict[str, object]:
    return json.loads((ROOT / "configs" / "p3_c2_solver_plan.json").read_text(encoding="utf-8"))


def test_all_four_real_solvers_recover_the_same_separable_solution():
    problem, truth_linear, truth_nonlinear = _problem()
    options = SolverOptions(max_iterations=80)
    results = [
        solve_joint_lm(problem, options),
        solve_classical_vp(problem, options),
        solve_vplr(problem, options),
        solve_tsvd_ridge_vp(problem, options),
    ]
    assert all(result.convergence for result in results)
    for result in results:
        np.testing.assert_allclose(result.linear_parameters, truth_linear, atol=2e-5)
        np.testing.assert_allclose(result.nonlinear_parameters, truth_nonlinear, atol=2e-5)
        assert result.iterations <= options.max_iterations
        assert result.forward_calls > 0
        assert result.final_residual < 1e-6
        assert np.isfinite(result.condition_number)


def test_projection_disabled_control_is_exactly_the_joint_implementation():
    problem, _, _ = _problem()
    joint = solve_joint_lm(problem)
    control = solve_projection_disabled_control(problem)
    np.testing.assert_array_equal(control.linear_parameters, joint.linear_parameters)
    np.testing.assert_array_equal(control.nonlinear_parameters, joint.nonlinear_parameters)
    assert control.iterations == joint.iterations
    assert control.forward_calls == joint.forward_calls
    assert control.final_residual == joint.final_residual


def _paired_metric_rows(candidate_error: float = 0.01) -> list[dict[str, object]]:
    rows = []
    for cell in ("CELL-A", "CELL-B"):
        for seed in (101, 202):
            for split in ("GIB-SPLIT-01", "GIB-SPLIT-02"):
                for group in range(4):
                    identity = {
                        "mixture_id": f"{cell}-{split}-{group}",
                        "sequence_id": f"{cell}-{split}-{group}-Q",
                        "grid_cell_id": cell,
                        "split_id": split,
                        "seed": seed,
                        "information_band": "sufficient",
                        "condition_number": 10.0,
                        "final_residual": 0.01,
                        "hardware_fingerprint": "HW-1",
                        "convergence": True,
                        "repeat_index": 0,
                    }
                    rows.append(
                        {
                            **identity,
                            "method_id": "joint_lm",
                            "component_abs_errors": [0.02, 0.02],
                            "iterations": 20,
                            "forward_calls": 100,
                            "solver_wall_clock": 1000,
                        }
                    )
                    for method in ("classical_vp", "vplr", "tsvd_ridge_vp"):
                        rows.append(
                            {
                                **identity,
                                "method_id": method,
                                "component_abs_errors": [candidate_error, candidate_error],
                                "iterations": 10,
                                "forward_calls": 50,
                                "solver_wall_clock": 500,
                            }
                        )
    return rows


def _small_gate_plan() -> dict[str, object]:
    plan = _plan()
    plan["components"] = ["A", "B"]
    plan["gates"]["non_inferiority_bands"] = {"A": 0.001, "B": 0.001}
    plan["statistics"]["bootstrap_resamples"] = 200
    plan["robustness"]["minimum_distinct_passing_cells"] = 2
    plan["robustness"]["minimum_distinct_passing_seeds"] = 2
    return plan


def test_mixture_paired_bootstrap_and_c2_gate_clear_ni_e30_e20_nr5():
    rows = _paired_metric_rows()
    summary = paired_group_bootstrap(
        [row for row in rows if row["grid_cell_id"] == "CELL-A" and row["seed"] == 101],
        "classical_vp",
        component_count=2,
        resamples=200,
        seed=7,
    )
    assert all(item["ci"][1] < 0.0 for item in summary["precision_p90_difference"])
    assert summary["cost_relative_reduction"]["iterations"]["ci"][0] == pytest.approx(0.5)
    verdict = evaluate_c2_preflight(rows, _small_gate_plan())
    assert verdict["c2_preflight"] == "pass"
    assert all(item["passes"] for item in verdict["method_verdicts"])
    assert verdict["activated_tasks"] == {"P3-10": True, "P3-12": True}


def test_gate_fails_precision_regression_and_pairing_or_hardware_mismatch():
    plan = _small_gate_plan()
    verdict = evaluate_c2_preflight(_paired_metric_rows(candidate_error=0.04), plan)
    assert verdict["c2_preflight"] == "fail"

    broken = _paired_metric_rows()
    broken[1] = {**broken[1], "hardware_fingerprint": "HW-2"}
    with pytest.raises(SolverError, match="hardware"):
        paired_group_bootstrap(
            [row for row in broken if row["grid_cell_id"] == "CELL-A" and row["seed"] == 101],
            "classical_vp",
            component_count=2,
            resamples=10,
            seed=7,
        )


def test_technical_preflight_runs_controls_but_formal_mode_requires_9x5x3():
    problem, truth, _ = _problem()
    plan = _plan()
    plan["components"] = ["A", "B"]
    plan["gates"]["non_inferiority_bands"] = {"A": 1.0, "B": 1.0}
    plan["statistics"]["bootstrap_resamples"] = 20
    plan["robustness"]["minimum_distinct_passing_cells"] = 1
    plan["robustness"]["minimum_distinct_passing_seeds"] = 1
    plan["timing"]["formal_repeats"] = 1
    plan["timing"]["warmup_solver_runs"] = 0
    trial = SolverTrial(
        mixture_id="GIB-M-0000000000000001",
        sequence_id="GIB-Q-0000000000000001",
        grid_cell_id=plan["grid_cell_ids"][0],
        split_id=plan["split_ids"][0],
        seed=plan["seeds"][0],
        problem=problem,
        truth_linear=truth,
        information_band="sufficient",
        hardware_fingerprint="HW-TEST",
    )
    result = run_solver_preflight(
        [trial],
        plan,
        runtime_metadata=RUNTIME_METADATA,
        technical_test_mode=True,
    )
    assert len(result["solver_rows"]) == 4
    assert all(item["passed"] for item in result["negative_controls"].values())
    assert result["formal_run_started"] is False
    with pytest.raises(SolverError, match="coverage mismatch"):
        run_solver_preflight([trial], plan, runtime_metadata=RUNTIME_METADATA)


def test_invalid_basis_and_nonfinite_errors_fail_explicitly():
    problem, _, _ = _problem()
    broken = SeparableProblem(
        observations=problem.observations,
        basis=lambda nonlinear: np.ones((2, 2)),
        linear_initial=problem.linear_initial,
        nonlinear_initial=problem.nonlinear_initial,
    )
    with pytest.raises(SolverError, match="basis"):
        solve_joint_lm(broken)

    rows = _paired_metric_rows()
    bad = copy.deepcopy(rows)
    bad[1]["component_abs_errors"][0] = float("nan")
    with pytest.raises(SolverError, match="non-finite"):
        paired_group_bootstrap(
            [row for row in bad if row["grid_cell_id"] == "CELL-A" and row["seed"] == 101],
            "classical_vp",
            component_count=2,
            resamples=10,
            seed=7,
        )
