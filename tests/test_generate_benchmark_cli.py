import json

from pipeline.generate_benchmark import main


def test_generate_benchmark_cli_writes_summary(tmp_path, capsys):
    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--dataset",
            "wv4-cli",
            "--sequences",
            "6",
            "--seed",
            "13",
            "--storage",
            "npz",
            "--path-lms",
            "0.20,0.25,0.30",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["dataset_slug"] == "wv4-cli"
    assert payload["sequence_count"] == 6
    assert (tmp_path / "wv4-cli" / "manifest.json").is_file()
    assert (tmp_path / "wv4-cli" / "sequences" / "waveform_sequence.npz").is_file()

    manifest = json.loads((tmp_path / "wv4-cli" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["path_lms"] == [0.2, 0.25, 0.3]
