from __future__ import annotations

import json
from pathlib import Path

import pytest

from gf.dl.adapters import AdapterError, ArHeCO2Adapter, XyleneENoseAdapter, parse_xylene_label
from gf.dl.splits import SplitLeakageError, validate_group_splits


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_group_split_validation_rejects_overlap() -> None:
    with pytest.raises(SplitLeakageError, match="group leakage"):
        validate_group_splits(
            {
                "train": ["group-a"],
                "val": ["group-b"],
                "test": ["group-a"],
            }
        )


def test_a0_configs_use_mixture_and_workbook_groups_without_leakage() -> None:
    ar_config = _load_json(PROJECT_ROOT / "configs/data/ar_he_co2_a0_smoke.json")
    xylene_config = _load_json(PROJECT_ROOT / "configs/data/xylene_e_nose_a0_smoke.json")
    samples_by_dataset = {
        "ar_he_co2": ArHeCO2Adapter.from_config(ar_config).load_samples(),
        "xylene_e_nose": XyleneENoseAdapter.from_config(xylene_config, project_root=PROJECT_ROOT).load_samples(),
    }

    for dataset_id, samples in samples_by_dataset.items():
        splits = {"train": [], "val": [], "test": []}
        for sample in samples:
            splits[str(sample.metadata["split"])].append(sample.group_id)
        validated = validate_group_splits(splits, known_group_ids=(sample.group_id for sample in samples))
        assert all(len(groups) == 1 for groups in validated.values())
        assert all(sample.dataset_id == dataset_id for sample in samples)

    for sample in samples_by_dataset["ar_he_co2"]:
        assert sample.group_id == sample.metadata["mixture_id"]
        assert "sequence_id" not in sample.metadata
        assert "base_condition_id" not in sample.metadata
        assert "noise_seed_index" not in sample.metadata
        assert "noise_seed" not in sample.metadata
    for sample in samples_by_dataset["xylene_e_nose"]:
        assert sample.group_id == sample.metadata["workbook_name"]
        assert sample.signals[0].shape == (64, 1)
        assert sample.time[0][1] - sample.time[0][0] > 1.0


def test_xylene_label_parser_and_quantitative_boundary_are_explicit() -> None:
    target, total, family = parse_xylene_label("100 ppm m-o-p=1-1-2.xlsx")
    assert target.tolist() == pytest.approx([25.0, 25.0, 50.0])
    assert total == 100.0
    assert family == "ternary"
    with pytest.raises(AdapterError, match="no quantitative ppm label"):
        parse_xylene_label("max concentration m-xylene.xlsx")


def test_xylene_adapter_rejects_ambiguous_qcm_header() -> None:
    config = {
        "dataset_id": "xylene_e_nose",
        "dataset_root": "../数据集/xylene-e-nose_三元二甲苯混合物",
        "window_rows": 8,
        "workbooks": [{"file": "10 ppm p-xylene.xlsx", "split": "train"}],
    }
    adapter = XyleneENoseAdapter.from_config(config, project_root=PROJECT_ROOT)
    with pytest.raises(AdapterError, match="QCM channel"):
        adapter.load_samples()


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value
