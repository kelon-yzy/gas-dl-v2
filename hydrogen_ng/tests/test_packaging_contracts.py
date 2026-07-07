from hg.sim.packaging.index import build_sequence_index_rows
from hg.sim.packaging.run_contract import REQUIRED_RUN_FILES, minimum_run_contract
from hg.sim.packaging.splits import build_split_rows_from_group_sets


def test_sequence_index_preserves_mixture_id_from_condition_rows():
    rows = build_sequence_index_rows(
        [
            {"sequence_id": "Q000001", "mixture_id": "M000010"},
            {"sequence_id": "Q000002", "mixture_id": "M000010"},
        ],
        stage_profile="standard_exposure",
        timesteps=128,
        dt_s=0.5,
    )

    assert rows == [
        {
            "sequence_id": "Q000001",
            "mixture_id": "M000010",
            "stage_profile": "standard_exposure",
            "status": "synthetic_measurement",
            "n_timesteps": "128",
            "dt_s": "0.5",
        },
        {
            "sequence_id": "Q000002",
            "mixture_id": "M000010",
            "stage_profile": "standard_exposure",
            "status": "synthetic_measurement",
            "n_timesteps": "128",
            "dt_s": "0.5",
        },
    ]


def test_split_rows_group_by_mixture_id_without_rewriting_it():
    conditions = [
        {"sequence_id": "Q000001", "mixture_id": "M000001"},
        {"sequence_id": "Q000002", "mixture_id": "M000001"},
        {"sequence_id": "Q000003", "mixture_id": "M000002"},
    ]

    rows = build_split_rows_from_group_sets(
        conditions,
        split_groups={
            "train": {"M000001"},
            "test": {"M000002"},
        },
    )

    assert rows == {
        "train": [
            {"sequence_id": "Q000001", "mixture_id": "M000001"},
            {"sequence_id": "Q000002", "mixture_id": "M000001"},
        ],
        "test": [
            {"sequence_id": "Q000003", "mixture_id": "M000002"},
        ],
    }


def test_minimum_run_contract_lists_required_outputs():
    contract = minimum_run_contract()

    assert contract.required_files == REQUIRED_RUN_FILES
    assert "summary.json" in contract.required_files
    assert "predictions.csv" in contract.required_files
    assert "report.md" in contract.required_files
