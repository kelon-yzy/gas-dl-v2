import hashlib
import json
import math
from pathlib import Path

from gib.common.io import sha256_file
from gib.pipeline.dataset import load_deployment_records, load_oracle_records
from gib.sim.pilot import build_pilot_dataset, validate_pilot_plan


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def _tree_hashes(root: Path):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_pilot_plan_freezes_all_nine_cells_and_generation_dimensions():
    plan = _load("p3_pilot_plan_v2.json")
    grid = _load("p2_s1_grid.json")
    validate_pilot_plan(plan, grid)
    assert plan["grid_cell_count"] == len(grid["cells"]) == 9
    assert plan["pilot"] == {"mixtures_per_cell": 20, "sequences_per_mixture": 2}
    assert plan["splits"]["stratify_by"] == "grid_cell_id"
    assert plan["raw"]["storage"] == "per_sequence_npy"
    assert plan["splits"]["nested_train_fractions"] == [10, 25, 50, 75, 100]


def test_two_dry_runs_have_identical_ids_splits_arrays_and_hashes(tmp_path: Path):
    plan = _load("p3_pilot_plan_v2.json")
    code_hash = sha256_file(ROOT / "gib" / "pipeline" / "raw_dsp.py")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = build_pilot_dataset(
        plan,
        config_root=ROOT / "configs",
        output_dir=first,
        dry_run=True,
        raw_dsp_code_sha256=code_hash,
    )
    second_summary = build_pilot_dataset(
        plan,
        config_root=ROOT / "configs",
        output_dir=second,
        dry_run=True,
        raw_dsp_code_sha256=code_hash,
    )
    assert first_summary == second_summary
    assert _tree_hashes(first) == _tree_hashes(second)
    assert first_summary["mixture_count"] == 27
    assert first_summary["sequence_count"] == 54
    assert first_summary["artifact_file_count"] == 54 * 11

    records = [json.loads(line) for line in (first / "sample_records.jsonl").read_text(encoding="utf-8").splitlines()]
    mixture_sequences = {}
    for record in records:
        mixture_sequences.setdefault(record["mixture_id"], set()).add(record["sequence_id"])
    assert len(mixture_sequences) == 27
    assert all(len(sequence_ids) == 2 for sequence_ids in mixture_sequences.values())
    assert len({record["sequence_id"] for record in records}) == 54

    split_rows = json.loads((first / "split_assignments.json").read_text(encoding="utf-8"))
    assert len(split_rows) == 54 * 5
    for split_id in plan["splits"]["split_ids"]:
        rows = [row for row in split_rows if row["split_id"] == split_id]
        partition_by_mixture = {}
        for row in rows:
            previous = partition_by_mixture.setdefault(row["mixture_id"], row["partition"])
            assert previous == row["partition"]
        assert set(partition_by_mixture.values()) == {"train", "val", "test"}

    nested = json.loads((first / "nested_train_groups.json").read_text(encoding="utf-8"))
    for split in nested.values():
        order = split["train_group_order"]
        previous = set()
        for fraction in (10, 25, 50, 75, 100):
            prefix = split["train_prefixes"][str(fraction)]
            expected_count = max(9, math.ceil(len(order) * fraction / 100.0))
            assert prefix == order[:expected_count]
            assert previous.issubset(prefix)
            previous = set(prefix)

    assert len(load_deployment_records(first / "deployment" / "records.jsonl")) == 54
    assert len(load_oracle_records(first / "oracle" / "records.jsonl")) == 54
