"""Module C protocol matrix / verdict tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tv3.ml.grouped_bottleneck import EXPECTED_PARAMETER_COUNT


def _load_protocol_module():
    project_root = Path(__file__).resolve().parents[1]
    module_name = "test_module_c_protocol_runner_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        project_root / "scripts" / "run_module_c_grouped_bottleneck_protocol.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _base_row(
    *,
    variant: str,
    protocol_id: str,
    split_seed: int,
    training_seed: int,
    test_delta: float,
    ood_delta: float,
    c0_test: float = 0.65,
    c0_ood: float = 0.60,
) -> dict:
    return {
        "variant": variant,
        "protocol_id": protocol_id,
        "dataset_name": f"{protocol_id}_{split_seed}",
        "split_seed": split_seed,
        "training_seed": training_seed,
        "is_ood_evidence": protocol_id in {"S-Y", "S-L"},
        "status": "ok",
        "c0_status": "ok",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "c1c2_test_o2_r2": c0_test + test_delta,
        "c0_test_o2_r2": c0_test,
        "delta_vs_c0_test": test_delta,
        "c1c2_extrapolation_o2_r2": c0_ood + ood_delta,
        "c0_extrapolation_o2_r2": c0_ood,
        "delta_vs_c0_extrapolation": ood_delta,
        "c1c2_val_o2_r2": 0.7,
        "c0_val_o2_r2": 0.68,
        "delta_vs_c0_val": 0.02,
    }


def _full_matrix(
    *,
    physical_test_delta: float = 0.0,
    physical_ood_delta: float = 0.0,
    permuted_test_delta: float = -0.02,
    permuted_ood_delta: float = -0.03,
) -> tuple[list[dict], list[dict]]:
    mod = _load_protocol_module()
    physical: list[dict] = []
    permuted: list[dict] = []
    for protocol_id in mod.PROTOCOL_IDS:
        for split_seed in mod.SPLIT_SEEDS:
            for training_seed in mod.TRAINING_SEEDS:
                physical.append(
                    _base_row(
                        variant="physical",
                        protocol_id=protocol_id,
                        split_seed=split_seed,
                        training_seed=training_seed,
                        test_delta=physical_test_delta,
                        ood_delta=physical_ood_delta,
                    )
                )
                permuted.append(
                    _base_row(
                        variant="permuted",
                        protocol_id=protocol_id,
                        split_seed=split_seed,
                        training_seed=training_seed,
                        test_delta=permuted_test_delta,
                        ood_delta=permuted_ood_delta,
                    )
                )
    return physical, permuted


def test_protocol_matrix_is_single_training_seed_and_24_runs():
    mod = _load_protocol_module()
    specs = mod.build_protocol_matrix()
    assert len(specs) == 12
    assert {s.protocol_id for s in specs} == {"R", "L", "S-Y", "S-L"}
    assert {s.split_seed for s in specs} == {20260704, 20260712, 20260720}
    assert mod.TRAINING_SEEDS == (42,)
    assert mod.EXPECTED_ROWS_PER_VARIANT == 12
    # 12 split × 2 variants
    assert len(mod.PROTOCOL_IDS) * len(mod.SPLIT_SEEDS) * len(mod.TRAINING_SEEDS) * 2 == 24
    assert mod.PRE_REGISTERED_PERMUTATION_SEED == 20260712
    assert mod.EXPECTED_PARAMETER_COUNT == 28051


def test_b7_protocol_root_requires_complete_protocol_pass(tmp_path: Path):
    mod = _load_protocol_module()
    b7 = mod._load_b7_protocol_module()
    rows = [
        {
            "protocol_id": spec.protocol_id,
            "split_seed": spec.split_seed,
            "training_seed": seed,
        }
        for spec in b7.build_protocol_matrix()
        for seed in b7.TRAINING_SEEDS
    ]
    split_metrics_path = tmp_path / "split_metrics.json"
    split_metrics_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "verdict": {
                    "protocol_pass": False,
                    "matrix_complete": True,
                    "unexpected_row_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="protocol_pass"):
        mod._validate_b7_protocol_root(tmp_path)

    split_metrics_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "verdict": {
                    "protocol_pass": True,
                    "matrix_complete": True,
                    "unexpected_row_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    mod._validate_b7_protocol_root(tmp_path)


def test_bottleneck_pass_requires_non_inferior_c0_and_better_than_permuted():
    mod = _load_protocol_module()
    physical, permuted = _full_matrix(
        physical_test_delta=0.0,
        physical_ood_delta=0.0,
        permuted_test_delta=-0.02,
        permuted_ood_delta=-0.03,
    )
    assert len(physical) == 12
    assert len(permuted) == 12
    verdict = mod.evaluate_module_c_verdict(physical, permuted)
    assert verdict["verdict"] == "bottleneck_pass"
    assert verdict["c1_vs_c0_non_inferior"] is True
    assert verdict["c1_vs_c2_directional"] is True
    assert verdict["ood_gain_hit"] is True
    assert verdict["training_seeds"] == [42]


def test_grouped_failed_when_c1_below_c0_tolerance():
    mod = _load_protocol_module()
    physical, permuted = _full_matrix(
        physical_test_delta=-0.02,
        physical_ood_delta=0.0,
        permuted_test_delta=-0.05,
        permuted_ood_delta=-0.05,
    )
    verdict = mod.evaluate_module_c_verdict(physical, permuted)
    assert verdict["verdict"] == "grouped_failed"


def test_compression_only_when_not_better_than_permuted():
    mod = _load_protocol_module()
    physical, permuted = _full_matrix(
        physical_test_delta=0.0,
        physical_ood_delta=0.0,
        permuted_test_delta=0.0,
        permuted_ood_delta=0.0,
    )
    verdict = mod.evaluate_module_c_verdict(physical, permuted)
    assert verdict["verdict"] == "compression_only"


def test_mean_ood_drop_vs_c0_blocks_bottleneck_pass():
    mod = _load_protocol_module()
    physical, permuted = _full_matrix(
        physical_test_delta=0.0,
        physical_ood_delta=-0.02,
        permuted_test_delta=-0.05,
        permuted_ood_delta=-0.05,
    )
    verdict = mod.evaluate_module_c_verdict(physical, permuted)
    assert verdict["verdict"] == "grouped_failed"
    assert verdict["checks"]["S-Y"]["c1_vs_c0_extrapolation_non_inferior"] is False


def test_audit_failed_on_incomplete_matrix():
    mod = _load_protocol_module()
    physical, permuted = _full_matrix()
    physical = physical[:-1]
    verdict = mod.evaluate_module_c_verdict(physical, permuted)
    assert verdict["verdict"] == "audit_failed"


def test_load_c0_b7_matrix_filters_to_single_training_seed():
    project_root = Path(__file__).resolve().parents[1]
    matrix_path = project_root / "outputs" / "tv3_b7_protocol" / "result_matrix.csv"
    if not matrix_path.is_file():
        pytest.skip("formal B7 protocol matrix not present")
    mod = _load_protocol_module()
    rows = mod.load_c0_b7_matrix(project_root / "outputs" / "tv3_b7_protocol")
    assert len(rows) == 12
    assert all(key[2] == 42 for key in rows)
    assert all(row["c0_feature_names_digest"] for row in rows.values())
    assert ("R", 20260704, 42) in rows
    assert ("R", 20260704, 123) not in rows


def test_c1_c2_config_hashes_stable():
    project_root = Path(__file__).resolve().parents[1]
    mod = _load_protocol_module()
    c1 = project_root / "configs" / "tv3_module_c_grouped_bottleneck_physical.json"
    c2 = project_root / "configs" / "tv3_module_c_grouped_bottleneck_permuted.json"
    assert c1.is_file() and c2.is_file()
    assert mod._file_sha256(c1) != mod._file_sha256(c2)
    c1_payload = json.loads(c1.read_text(encoding="utf-8"))
    c2_payload = json.loads(c2.read_text(encoding="utf-8"))
    assert c1_payload["group_assignment"] == "physical"
    assert c2_payload["group_assignment"] == "permuted"
    assert c1_payload["output_dir"] == "outputs/tv3_module_c_grouped_bottleneck/physical"
    assert c2_payload["output_dir"] == "outputs/tv3_module_c_grouped_bottleneck/permuted"


def test_protocol_cli_rejects_unfrozen_training_seed_option():
    mod = _load_protocol_module()
    with pytest.raises(SystemExit):
        mod.build_parser().parse_args(["--training-seeds", "123"])


def test_train_stage_audits_before_model_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _load_protocol_module()
    spec = next(
        item
        for item in mod.build_protocol_matrix()
        if item.protocol_id == "R" and item.split_seed == 20260704
    )
    audit_calls: list[object] = []

    def fail_if_training_starts(*args, **kwargs):
        raise AssertionError("train stage must stop after a failed prerequisite audit")

    monkeypatch.setattr(mod, "build_protocol_matrix", lambda: [spec])
    monkeypatch.setattr(mod, "load_c0_b7_matrix", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "_audit_prerequisites",
        lambda *args, **kwargs: audit_calls.append(args[0]) or ["stale provenance"],
    )
    monkeypatch.setattr(mod, "run_module_c_seed", fail_if_training_starts)

    exit_code = mod.main(
        [
            "--stage",
            "train",
            "--protocol",
            "R",
            "--split-seeds",
            "20260704",
            "--runs-root",
            str(tmp_path / "runs"),
            "--summary-root",
            str(tmp_path / "summary"),
            "--reports-root",
            str(tmp_path / "reports"),
        ]
    )

    assert exit_code == 1
    assert audit_calls == [spec]
