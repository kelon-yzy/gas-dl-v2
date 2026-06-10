from __future__ import annotations

import json
from pathlib import Path

from pipeline.analyze_n2_improvement import analyze_n2_improvement, format_markdown_report, main


def _write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _split_eval(n2_r2: float, rmse: float, *, h2_r2: float = 0.9, ch4_r2: float = 0.8, co2_r2: float = 0.85):
    return {
        "metrics": {"mae": 1.0, "rmse": rmse, "r2": 0.8},
        "component_metrics": {
            "x_H2": {"mae": 1.0, "rmse": 1.0, "r2": h2_r2},
            "x_CH4": {"mae": 1.0, "rmse": 1.0, "r2": ch4_r2},
            "x_CO2": {"mae": 1.0, "rmse": 1.0, "r2": co2_r2},
            "x_N2": {"mae": 1.0, "rmse": 1.0, "r2": n2_r2},
        },
        "compositional_metrics": {"aitchison_mean": 0.5, "aitchison_rmse": 0.6},
        "conditional_metrics": {
            "n2_bins": {
                "component": "x_N2",
                "bins": {
                    "0_10": _bin_eval(n2_r2 - 0.05, rmse + 0.2, [0.0, 10.0]),
                    "10_20": _bin_eval(n2_r2 + 0.02, rmse - 0.1, [10.0, 20.0]),
                },
            },
            "ch4_bins": {
                "component": "x_CH4",
                "bins": {
                    "40_70": _bin_eval(n2_r2 - 0.02, rmse + 0.1, [40.0, 70.0]),
                    "70_100": _bin_eval(n2_r2 + 0.03, rmse - 0.2, [70.0, 100.0]),
                },
            },
        },
    }


def _bin_eval(n2_r2: float, rmse: float, value_range: list[float]) -> dict[str, object]:
    return {
        "range": value_range,
        "count": 2,
        "metrics": {"mae": 1.0, "rmse": rmse, "r2": 0.8},
        "component_metrics": {
            "x_H2": {"mae": 1.0, "rmse": 1.0, "r2": 0.9},
            "x_CH4": {"mae": 1.0, "rmse": 1.0, "r2": 0.8},
            "x_CO2": {"mae": 1.0, "rmse": 1.0, "r2": 0.85},
            "x_N2": {"mae": 1.0, "rmse": 1.0, "r2": n2_r2},
        },
    }


def test_analyze_n2_improvement_reads_protocol_and_plain_metrics(tmp_path: Path):
    run_root = tmp_path / "runs" / "formal_full"
    _write_metrics(
        run_root / "ridge_all_modalities" / "metrics.json",
        {
            "full": {"evaluations": {"test": _split_eval(0.22, 4.0)}},
            "per_phase": {"exposure": {"evaluations": {"test": _split_eval(0.30, 3.5)}}},
            "early": {"0.5": {"evaluations": {"test": _split_eval(0.24, 3.8)}}},
        },
    )
    _write_metrics(
        run_root / "ridge_ilr_n2_first_all_modalities" / "metrics.json",
        {
            "full": {"evaluations": {"test": _split_eval(0.35, 3.9, h2_r2=0.89, ch4_r2=0.79, co2_r2=0.84)}},
            "per_phase": {"exposure": {"evaluations": {"test": _split_eval(0.46, 3.2)}}},
            "early": {"0.5": {"evaluations": {"test": _split_eval(0.31, 3.7)}}},
        },
    )

    payload = analyze_n2_improvement(
        run_root,
        comparisons=(("ridge_all_modalities", "ridge_ilr_n2_first_all_modalities"),),
    )

    item = payload["comparisons"][0]
    assert round(item["n2_r2_gain"], 6) == 0.13
    assert round(item["conditional_bins"]["n2_bins"]["0_10"]["n2_r2_gain"], 6) == 0.13
    assert item["conditional_bins"]["n2_bins"]["0_10"]["passed_overall"] is True
    assert round(item["conditional_bins"]["ch4_bins"]["40_70"]["n2_r2_gain"], 6) == 0.13
    assert round(item["protocol_windows"]["per_phase"]["exposure"]["n2_r2_gain"], 6) == 0.16
    assert item["protocol_windows"]["per_phase"]["exposure"]["passed_overall"] is True
    assert round(item["protocol_windows"]["per_phase"]["exposure"]["conditional_bins"]["n2_bins"]["0_10"]["n2_r2_gain"], 6) == 0.16
    assert item["protocol_windows"]["per_phase"]["exposure"]["conditional_bins"]["n2_bins"]["0_10"]["passed_overall"] is True
    assert round(item["protocol_windows"]["early"]["0.5"]["n2_r2_gain"], 6) == 0.07
    assert item["protocol_windows"]["early"]["0.5"]["passed_overall"] is False
    assert item["candidate_aitchison_mean"] == 0.5
    assert item["passed_overall"] is True
    report = format_markdown_report(payload)
    assert "ridge_ilr_n2_first_all_modalities" in report
    assert "Protocol Windows" in report
    assert "Conditional Bins" in report
    assert "Protocol Conditional Bins" in report
    assert "| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | n2_bins | 0_10 |" in report
    assert "| baseline | candidate | group | window | N2 R2 gain | RMSE regression | max other R2 drop | Aitchison mean | pass |" in report
    assert "| baseline | candidate | group | bin | count | range | N2 R2 gain | RMSE regression | max other R2 drop | pass |" in report


def test_analyze_n2_improvement_flags_other_component_regression(tmp_path: Path):
    run_root = tmp_path / "runs" / "formal_full"
    _write_metrics(
        run_root / "cnn1d_tcn_fusion" / "metrics.json",
        {"evaluations": {"test": _split_eval(-0.01, 5.0)}},
    )
    _write_metrics(
        run_root / "cnn1d_tcn_fusion_ilr" / "metrics.json",
        {"evaluations": {"test": _split_eval(0.15, 4.8, h2_r2=0.5, ch4_r2=0.8, co2_r2=0.85)}},
    )

    payload = analyze_n2_improvement(
        run_root,
        comparisons=(("cnn1d_tcn_fusion", "cnn1d_tcn_fusion_ilr"),),
    )

    item = payload["comparisons"][0]
    assert item["passed_n2_gain"] is True
    assert item["passed_other_components"] is False
    assert item["passed_overall"] is False


def test_analyze_n2_improvement_cli_writes_markdown_and_json_reports(tmp_path: Path, capsys):
    run_root = tmp_path / "runs" / "formal_full"
    _write_metrics(
        run_root / "ridge_all_modalities" / "metrics.json",
        {"evaluations": {"test": _split_eval(0.22, 4.0)}},
    )
    _write_metrics(
        run_root / "ridge_ilr_n2_first_all_modalities" / "metrics.json",
        {"evaluations": {"test": _split_eval(0.35, 3.9)}},
    )
    _write_metrics(
        run_root / "ridge_alr_ch4_all_modalities" / "metrics.json",
        {"evaluations": {"test": _split_eval(0.34, 3.8)}},
    )
    _write_metrics(
        run_root / "cnn1d_tcn_fusion" / "metrics.json",
        {"evaluations": {"test": _split_eval(-0.01, 5.0)}},
    )
    _write_metrics(
        run_root / "cnn1d_tcn_fusion_ilr" / "metrics.json",
        {"evaluations": {"test": _split_eval(0.15, 4.7)}},
    )
    markdown_path = tmp_path / "reports" / "n2_improvement.md"
    json_path = tmp_path / "reports" / "n2_improvement.json"
    json_sidecar_path = tmp_path / "reports" / "n2_improvement_sidecar.json"

    assert main([
        "--run-root",
        str(run_root),
        "--output-path",
        str(markdown_path),
        "--json-output-path",
        str(json_sidecar_path),
    ]) == 0
    markdown_stdout = capsys.readouterr().out
    assert markdown_path.read_text(encoding="utf-8") == markdown_stdout
    assert "Conditional Bins" in markdown_stdout
    sidecar_payload = json.loads(json_sidecar_path.read_text(encoding="utf-8"))
    assert sidecar_payload["run_root"] == str(run_root)

    assert main([
        "--run-root",
        str(run_root),
        "--json",
        "--output-path",
        str(json_path),
    ]) == 0
    json_stdout = capsys.readouterr().out
    assert json_path.read_text(encoding="utf-8") == json_stdout
    payload = json.loads(json_stdout)
    assert {item["candidate_run"] for item in payload["comparisons"]} == {
        "ridge_alr_ch4_all_modalities",
        "ridge_ilr_n2_first_all_modalities",
        "cnn1d_tcn_fusion_ilr",
    }
