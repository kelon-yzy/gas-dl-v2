"""F5 unit tests: S-Flow selector, arm contracts, gate evaluation (no 6000 training)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.bidir_arm_features import (
    ORACLE_V_SCALAR,
    arm_specs,
    assemble_arm_feature_matrix,
)
from tv3.ml.bidir_features import assert_no_oracle_inputs
from tv3.ml.bidir_s_flow import (
    classify_mixtures_by_abs_v,
    derive_s_flow_split,
    mixture_median_abs_v_path,
    zero_anchor_sequence_ids,
)
from tv3.ml.bidir_f5_gates import evaluate_f5_gates
from tv3.pipeline.generate_tunnel_ventilation_benchmark import BIDIR_FORMAL_6000_PRESET


def test_formal_6000_preset():
    assert BIDIR_FORMAL_6000_PRESET["dataset"] == "tv3-bidir-6000"
    assert BIDIR_FORMAL_6000_PRESET["sequences"] == 6000
    assert BIDIR_FORMAL_6000_PRESET["timesteps"] == 512
    assert BIDIR_FORMAL_6000_PRESET["bidirectional"] is True
    assert BIDIR_FORMAL_6000_PRESET["skip_fiber_mic"] is True


def test_arm_specs_deploy_hygiene():
    specs = arm_specs()
    assert set(specs) == {"A1", "A2", "A3", "A4", "A5"}
    assert specs["A2"].deployable is False
    assert ORACLE_V_SCALAR in specs["A2"].sequence_scalars
    for arm_id in ("A1", "A3", "A4", "A5"):
        assert specs[arm_id].deployable is True
        assert_no_oracle_inputs(list(specs[arm_id].frame_arrays) + list(specs[arm_id].sequence_scalars))
    assert "ultrasonic_v_path_hat_raw_dsp_m_per_s" not in specs["A5"].frame_arrays
    assert "v_hat_seq_m_per_s" not in specs["A5"].sequence_scalars
    assert "ultrasonic_v_path_hat_raw_dsp_m_per_s" in specs["A3"].frame_arrays


def test_mixture_median_and_s_flow_classify():
    conditions = [
        {"mixture_id": "M1", "v_path_m_per_s": "1.0"},
        {"mixture_id": "M1", "v_path_m_per_s": "2.0"},
        {"mixture_id": "M2", "v_path_m_per_s": "3.0"},
        {"mixture_id": "M3", "v_path_m_per_s": "0.0"},
        {"mixture_id": "M4", "v_path_m_per_s": "4.5"},
    ]
    med = mixture_median_abs_v_path(conditions)
    assert med["M1"] == pytest.approx(1.5)
    assert med["M2"] == pytest.approx(3.0)
    labels = classify_mixtures_by_abs_v(med, train_abs_v_max=2.5, ood_abs_v_max=4.0)
    assert labels["M1"] == "in_domain"
    assert labels["M2"] == "ood"
    assert labels["M3"] == "in_domain"
    assert labels["M4"] == "excluded"


def _write_mini_dataset(root: Path, *, n: int = 12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir()
    (root / "metadata").mkdir()
    (root / "sequences").mkdir()
    (root / "splits").mkdir()
    mixture_ids = [f"M{i // 2 + 1:06d}" for i in range(n)]
    sequence_ids = [f"Q{i + 1:06d}" for i in range(n)]
    # Half low |v|, half high |v| by mixture pairs.
    v_paths = []
    for i in range(n):
        mid = i // 2
        v_paths.append(1.0 if mid % 2 == 0 else 3.5)
    with (root / "condition_grid_sequence.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sequence_id", "mixture_id", "v_path_m_per_s", "x_CO2", "x_O2", "x_N2"],
        )
        writer.writeheader()
        for sid, mid, v in zip(sequence_ids, mixture_ids, v_paths, strict=True):
            writer.writerow(
                {
                    "sequence_id": sid,
                    "mixture_id": mid,
                    "v_path_m_per_s": f"{v:.6f}",
                    "x_CO2": "1.0",
                    "x_O2": "20.0",
                    "x_N2": "79.0",
                }
            )
    np.save(root / "labels" / "y.npy", np.tile(np.array([1.0, 20.0, 79.0], dtype=np.float32), (n, 1)))
    np.save(root / "metadata" / "sequence_ids.npy", np.asarray(sequence_ids))
    np.save(root / "metadata" / "label_names.npy", np.asarray(["x_CO2", "x_O2", "x_N2"]))
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": "tunnel-ventilation-bidir-1", "composition_scheme": "tunnel_ventilation_bidir"}),
        encoding="utf-8",
    )
    # Dummy splits so hashing works; derive_s_flow overwrites.
    for name in ("train", "val", "test", "extrapolation"):
        with (root / "splits" / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence_id", "mixture_id"])
            writer.writeheader()
    return root


def test_derive_s_flow_split(tmp_path: Path):
    source = _write_mini_dataset(tmp_path / "src")
    # Touch files required by hash.
    (source / "labels" / "y.npy").write_bytes((source / "labels" / "y.npy").read_bytes())
    out = tmp_path / "s_flow"
    info = derive_s_flow_split(source, out, seed=0, train_abs_v_max=2.5, ood_abs_v_max=4.0)
    assert info["splits"]["extrapolation"] > 0
    assert info["splits"]["train"] > 0
    summary = json.loads((out / "splits" / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["split_policy"] == "s_flow_abs_v_path_mixture_median_v1"
    assert "pure OOD" in summary["extrapolation_note"]
    assert (out / "condition_grid_sequence.csv").is_file()
    # Extrapolation must contain only OOD mixtures (median |v| > 2.5).
    conditions = list(csv.DictReader((source / "condition_grid_sequence.csv").open(encoding="utf-8")))
    median_abs_v = mixture_median_abs_v_path(conditions)
    with (out / "splits" / "extrapolation.csv").open(encoding="utf-8", newline="") as handle:
        extrap_rows = list(csv.DictReader(handle))
    assert extrap_rows
    for row in extrap_rows:
        assert median_abs_v[row["mixture_id"]] > 2.5
    id_ids = set()
    for split_name in ("train", "val", "test"):
        with (out / "splits" / f"{split_name}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                id_ids.add(row["mixture_id"])
                assert median_abs_v[row["mixture_id"]] <= 2.5
    extrap_ids = {row["mixture_id"] for row in extrap_rows}
    assert id_ids.isdisjoint(extrap_ids)
    zeros = zero_anchor_sequence_ids(conditions)
    assert isinstance(zeros, tuple)


def test_assemble_a3_rejects_nothing_and_a2_keeps_oracle():
    n, t = 4, 8
    rng = np.random.default_rng(0)
    slow = rng.normal(size=(n, t, 7)).astype(np.float32)
    slow_names = (
        "V_NDIR_CO2",
        "V_TCS",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    )
    sequence_ids = tuple(f"Q{i}" for i in range(n))
    phase_lookup = {sid: tuple(["baseline", "exposure", "steady", "recovery"] * 2) for sid in sequence_ids}
    labels = np.tile(np.array([1.0, 20.0, 79.0], dtype=np.float32), (n, 1))
    specs = arm_specs()
    frames = {
        name: rng.normal(size=(n, t)).astype(np.float32) for name in specs["A3"].frame_arrays
    }
    scalars = {name: rng.normal(size=(n,)).astype(np.float32) for name in specs["A3"].sequence_scalars}
    matrix = assemble_arm_feature_matrix(
        slow=slow,
        slow_channel_names=slow_names,
        sequence_ids=sequence_ids,
        labels=labels,
        label_names=("x_CO2", "x_O2", "x_N2"),
        phase_lookup=phase_lookup,
        frame_arrays=frames,
        sequence_scalars=scalars,
        arm=specs["A3"],
    )
    assert matrix.x.shape[0] == n
    assert matrix.x.shape[1] > 10

    a2_frames = {name: frames[name] for name in specs["A2"].frame_arrays}
    a2_scalars = {
        "tau_ab_s": scalars["tau_ab_s"],
        ORACLE_V_SCALAR: np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
    }
    a2 = assemble_arm_feature_matrix(
        slow=slow,
        slow_channel_names=slow_names,
        sequence_ids=sequence_ids,
        labels=labels,
        label_names=("x_CO2", "x_O2", "x_N2"),
        phase_lookup=phase_lookup,
        frame_arrays=a2_frames,
        sequence_scalars=a2_scalars,
        arm=specs["A2"],
    )
    assert any(ORACLE_V_SCALAR in name for name in a2.feature_names)


def test_f5_gates_preregistered_logic():
    def _fake(arm: str, ood_mae: float, test_zero_mae: float) -> dict:
        return {
            "evaluations": {
                "extrapolation": {"component_metrics": {"x_O2": {"mae": ood_mae, "r2": 0.5}}},
                "test": {"component_metrics": {"x_O2": {"mae": ood_mae, "r2": 0.5}}},
            },
            "zero_anchor_metrics": {
                "test": {"o2_mae": test_zero_mae, "n": 10},
            },
            "sound_speed_audit": {
                "pair_sound_speed_mean_abs_seq_bias_m_per_s": 0.01,
                "ab_sound_speed_mean_abs_seq_bias_m_per_s": 0.5,
                "reciprocity_residual_p95_of_seq_p95_s": 5e-8,
            },
        }

    arm_metrics = {
        "A1:b1_ridge": _fake("A1", 1.2, 0.40),
        "A2:b1_ridge": _fake("A2", 0.45, 0.40),
        "A3:b1_ridge": _fake("A3", 0.50, 0.42),
    }
    gates = {
        "a3_minus_a1_o2_mae_min_vol_percent": 0.5,
        "a3_minus_a2_o2_mae_max_vol_percent": 0.25,
        "v_path_zero_anchor_delta_mae_max_vol_percent": 0.05,
        "s_line_b1_reference_o2_mae": 0.40,
        "reciprocity_p95_max_s": 1.0e-7,
    }
    result = evaluate_f5_gates(arm_metrics, gates=gates, head="b1_ridge")
    assert result["checks"]["a_a3_beats_a1_ood"]["passed"] is True  # 1.2-0.5=0.7 ≥ 0.5
    assert result["checks"]["b_a3_near_a2_ood"]["passed"] is True  # 0.50-0.45=0.05 ≤ 0.25
    # c: A3 zero 0.42 vs S-line B1 0.40 → Δ=0.02 ≤ 0.05
    assert result["checks"]["c_zero_anchor_noninferior_vs_s_line_b1"]["passed"] is True
    assert result["checks"]["e_sound_speed_bias_and_reciprocity"]["passed"] is True
    assert result["core_gates_passed"] is True


def test_f5_gates_require_s_line_reference():
    arm_metrics = {
        "A1:b1_ridge": {
            "evaluations": {
                "extrapolation": {"component_metrics": {"x_O2": {"mae": 1.0, "r2": 0.1}}},
                "test": {"component_metrics": {"x_O2": {"mae": 1.0, "r2": 0.1}}},
            },
            "zero_anchor_metrics": {"test": {"o2_mae": 0.4, "n": 1}},
            "sound_speed_audit": {},
        },
        "A2:b1_ridge": {
            "evaluations": {
                "extrapolation": {"component_metrics": {"x_O2": {"mae": 0.5, "r2": 0.1}}},
                "test": {"component_metrics": {"x_O2": {"mae": 0.5, "r2": 0.1}}},
            },
            "zero_anchor_metrics": {"test": {"o2_mae": 0.4, "n": 1}},
            "sound_speed_audit": {},
        },
        "A3:b1_ridge": {
            "evaluations": {
                "extrapolation": {"component_metrics": {"x_O2": {"mae": 0.4, "r2": 0.1}}},
                "test": {"component_metrics": {"x_O2": {"mae": 0.4, "r2": 0.1}}},
            },
            "zero_anchor_metrics": {"test": {"o2_mae": 0.4, "n": 1}},
            "sound_speed_audit": {
                "pair_sound_speed_mean_abs_seq_bias_m_per_s": 0.01,
                "ab_sound_speed_mean_abs_seq_bias_m_per_s": 0.5,
                "reciprocity_residual_p95_of_seq_p95_s": 5e-8,
            },
        },
    }
    with pytest.raises(ValueError, match="s_line_b1_reference_o2_mae"):
        evaluate_f5_gates(
            arm_metrics,
            gates={
                "a3_minus_a1_o2_mae_min_vol_percent": 0.5,
                "a3_minus_a2_o2_mae_max_vol_percent": 0.25,
                "v_path_zero_anchor_delta_mae_max_vol_percent": 0.05,
                "s_line_b1_reference_o2_mae": None,
                "reciprocity_p95_max_s": 1.0e-7,
            },
            head="b1_ridge",
        )


def test_f5_gates_wide_criterion_c_uses_in_domain_a1():
    def _fake(ood_mae: float, test_zero_mae: float) -> dict:
        return {
            "evaluations": {
                "extrapolation": {"component_metrics": {"x_O2": {"mae": ood_mae, "r2": 0.5}}},
                "test": {"component_metrics": {"x_O2": {"mae": ood_mae, "r2": 0.5}}},
            },
            "zero_anchor_metrics": {"test": {"o2_mae": test_zero_mae, "n": 10}},
            "sound_speed_audit": {
                "pair_sound_speed_mean_abs_seq_bias_m_per_s": 0.01,
                "ab_sound_speed_mean_abs_seq_bias_m_per_s": 0.5,
                "reciprocity_residual_p95_of_seq_p95_s": 5e-8,
            },
        }

    arm_metrics = {
        "A1:b1_ridge": _fake(1.2, 0.40),
        "A2:b1_ridge": _fake(0.45, 0.40),
        "A3:b1_ridge": _fake(0.50, 0.42),
    }
    gates = {
        "criterion_c_anchor": "in_domain_a1_v_path_zero",
        "a3_minus_a1_o2_mae_min_vol_percent": 0.5,
        "a3_minus_a2_o2_mae_max_vol_percent": 0.25,
        "v_path_zero_anchor_delta_mae_max_vol_percent": 0.05,
        "s_line_b1_reference_o2_mae": None,
        "reciprocity_p95_max_s": 1.0e-7,
    }
    result = evaluate_f5_gates(arm_metrics, gates=gates, head="b1_ridge")
    assert result["criterion_c_anchor"] == "in_domain_a1_v_path_zero"
    # A3 zero 0.42 − A1 zero 0.40 = 0.02 ≤ 0.05
    assert result["checks"]["c_zero_anchor_noninferior_vs_in_domain_a1"]["passed"] is True
    assert result["core_gates_passed"] is True


def test_wide_model_protocol_config_paths():
    from pathlib import Path
    import json

    path = Path(__file__).resolve().parents[1] / "configs" / "tv3_bidir_model_protocol_wide.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["composition_domain"] == "wide"
    assert cfg["source_dataset_dir"] == "data/tv3-bidir-6000-wide"
    assert cfg["output_dir"] == "outputs/tv3_bidir/model_protocol_wide"
    assert cfg["f5_amplitude_gates"]["criterion_c_anchor"] == "in_domain_a1_v_path_zero"
    assert cfg["f4_prerequisite"]["verdict_path"].endswith("identifiability_v2_wide/f4_verdict.json")
    assert "tv3-bidir-6000-wide" in cfg["source_dataset_dir"]
