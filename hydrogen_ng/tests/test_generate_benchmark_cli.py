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
            "--workers",
            "1",
            "--path-lms",
            "0.20,0.25,0.30",
            "--optical-absorption-backend",
            "empirical_v1",
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
    assert manifest["optical_absorption_backend"] == "empirical_v1"


def test_generate_benchmark_cli_applies_time_axis_preset(tmp_path, capsys):
    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--dataset",
            "wv4-cli-standard",
            "--sequences",
            "4",
            "--seed",
            "17",
            "--storage",
            "npz",
            "--workers",
            "1",
            "--time-axis-preset",
            "standard",
            "--optical-absorption-backend",
            "empirical_v1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    manifest = json.loads((tmp_path / "wv4-cli-standard" / "manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["dataset_slug"] == "wv4-cli-standard"
    assert manifest["timesteps"] == 512
    assert manifest["dt_s"] == 0.5


def test_generate_benchmark_cli_formal_preset_sets_standard_defaults(tmp_path, capsys):
    exit_code = main(
        [
            "--experiment-preset",
            "formal-hitran-standard-6000",
            "--output-root",
            str(tmp_path),
            "--dataset",
            "wv4-cli-formal-small",
            "--sequences",
            "4",
            "--storage",
            "npz",
            "--workers",
            "1",
            "--optical-absorption-backend",
            "empirical_v1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    manifest = json.loads((tmp_path / "wv4-cli-formal-small" / "manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["dataset_slug"] == "wv4-cli-formal-small"
    assert manifest["timesteps"] == 512
    assert manifest["dt_s"] == 0.5
    assert manifest["seed"] == 20260603


def test_generate_benchmark_cli_applies_stage_profile_and_jitter(tmp_path, capsys):
    exit_code = main(
        [
            "--output-root",
            str(tmp_path),
            "--dataset",
            "wv4-cli-profile",
            "--sequences",
            "4",
            "--seed",
            "19",
            "--storage",
            "npz",
            "--workers",
            "1",
            "--timesteps",
            "32",
            "--stage-profile",
            "fast_transient",
            "--stage-jitter",
            "0.1",
            "--optical-absorption-backend",
            "empirical_v1",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    manifest = json.loads((tmp_path / "wv4-cli-profile" / "manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["dataset_slug"] == "wv4-cli-profile"
    assert manifest["stage_profile"] == "fast_transient"
    assert manifest["stage_jitter"] == 0.1
