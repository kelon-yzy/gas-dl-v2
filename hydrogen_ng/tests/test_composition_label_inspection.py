from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pipeline.inspect_composition_labels import format_markdown_report, inspect_composition_labels, main
from sim.core.schema import COMPONENT_FIELDS


def _write_split(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sequence_id", "mixture_id"))
        writer.writeheader()
        for sequence_id, mixture_id in rows:
            writer.writerow({"sequence_id": sequence_id, "mixture_id": mixture_id})


def _write_dataset(root: Path) -> Path:
    dataset_dir = root / "labels-smoke"
    (dataset_dir / "labels").mkdir(parents=True)
    (dataset_dir / "metadata").mkdir()
    np.save(
        dataset_dir / "labels" / "y.npy",
        np.array(
            [
                [0.0, 80.0, 5.0, 15.0],
                [10.0, 70.0, 0.0, 20.0],
                [5.0, 90.0, 5.0, 0.0],
                [1.0, 99.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.save(dataset_dir / "metadata" / "label_names.npy", np.array(COMPONENT_FIELDS))
    np.save(dataset_dir / "metadata" / "sequence_ids.npy", np.array(["seq0", "seq1", "seq2", "seq3"]))
    _write_split(dataset_dir / "splits" / "train.csv", [("seq0", "mix0"), ("seq1", "mix1")])
    _write_split(dataset_dir / "splits" / "val.csv", [("seq2", "mix2")])
    _write_split(dataset_dir / "splits" / "test.csv", [("seq3", "mix3")])
    _write_split(dataset_dir / "splits" / "extrapolation.csv", [])
    return dataset_dir


def test_inspect_composition_labels_reports_zero_stats_and_reference_variance(tmp_path: Path):
    dataset_dir = _write_dataset(tmp_path)

    payload = inspect_composition_labels(dataset_dir)
    report = format_markdown_report(payload)

    assert payload["recommended_zero_replacement"]["epsilon"] == 0.025
    assert payload["splits"]["train"]["components"]["x_H2"]["zero_count"] == 1
    assert payload["splits"]["train"]["components"]["x_CO2"]["zero_ratio"] == 0.5
    assert payload["splits"]["train"]["components"]["x_N2"]["min_positive_percent"] == 15.0
    assert payload["splits"]["extrapolation"]["total_rows"] == 0
    assert payload["splits"]["train"]["alr_reference"]["log_variance"] >= 0.0
    assert "Composition Label Inspection" in report
    json.dumps(payload)


def test_inspect_composition_labels_cli_writes_report(tmp_path: Path, capsys):
    dataset_dir = _write_dataset(tmp_path)
    report_path = tmp_path / "reports" / "composition_labels.md"
    json_report_path = tmp_path / "reports" / "composition_labels.json"

    exit_code = main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-path",
            str(report_path),
            "--json-output-path",
            str(json_report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.is_file()
    report = report_path.read_text(encoding="utf-8")
    assert "# Composition Label Inspection" in report
    assert report == capsys.readouterr().out
    payload = json.loads(json_report_path.read_text(encoding="utf-8"))
    assert payload["recommended_zero_replacement"]["epsilon"] == 0.025
