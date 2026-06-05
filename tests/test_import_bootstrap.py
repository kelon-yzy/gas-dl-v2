from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_pipeline_modules_are_importable_from_repo_root_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "pipeline.generate_benchmark", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Generate a v4 benchmark dataset" in result.stdout


def test_waveform_bundle_module_is_importable_from_repo_root_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "pipeline.bundle_waveform_sequence", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Bundle generated waveform arrays" in result.stdout


def test_dl_cli_is_importable_from_repo_root_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "dl.cli", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Train a DL regressor" in result.stdout


def test_ml_cli_is_importable_from_repo_root_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "ml.cli", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Train a traditional ML regressor" in result.stdout


def test_run_experiment_is_importable_from_repo_root_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "pipeline.run_experiment", "--help"],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Run a configured v4 formal experiment suite" in result.stdout
