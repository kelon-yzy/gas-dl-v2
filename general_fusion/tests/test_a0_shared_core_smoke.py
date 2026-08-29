from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gf.dl.fusion_core import FusionCore
from gf.pipeline.a0_smoke import run_a0_smoke
from gf.sim.ar_he_co2 import (
    ideal_gas_sound_speed,
    ndir_co2_voltage,
    wms_thermal_conductivity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ar_he_co2_pilot_physics_has_expected_directions() -> None:
    pure_ar = {"Ar": 1.0, "He": 0.0, "CO2": 0.0}
    pure_he = {"Ar": 0.0, "He": 1.0, "CO2": 0.0}

    assert ideal_gas_sound_speed(pure_he, 298.15) > ideal_gas_sound_speed(pure_ar, 298.15)
    assert wms_thermal_conductivity(pure_he) > wms_thermal_conductivity(pure_ar)
    assert ndir_co2_voltage(30.0, 101_325.0, 298.15) < ndir_co2_voltage(0.0, 101_325.0, 298.15)


def test_two_real_source_batches_share_one_core_and_are_deterministic() -> None:
    first = run_a0_smoke(project_root=PROJECT_ROOT)
    second = run_a0_smoke(project_root=PROJECT_ROOT)

    assert first.core_class == "FusionCore"
    assert first.all_gradients_finite
    assert [dataset.dataset_id for dataset in first.datasets] == ["ar_he_co2", "xylene_e_nose"]
    assert [dataset.sensor_count for dataset in first.datasets] == [3, 6]
    assert all(dataset.output_shape == (3, 3) for dataset in first.datasets)
    assert [dataset.output_checksum for dataset in first.datasets] == pytest.approx(
        [dataset.output_checksum for dataset in second.datasets], abs=0.0
    )
    assert all(len(dataset.fitted_scaler_groups) == 1 for dataset in first.datasets)


def test_fusion_core_and_new_mainline_have_no_dataset_or_history_package_branch() -> None:
    source = inspect.getsource(FusionCore).lower()
    for forbidden in ("ar_he", "xylene", "dataset_id", "mixture_id", "workbook"):
        assert forbidden not in source

    forbidden_imports = (
        "import hydrogen_ng",
        "from hydrogen_ng",
        "import syngas",
        "from syngas",
        "import tunnel_ventilation",
        "from tunnel_ventilation",
        "import rcdw_mgda",
        "from rcdw_mgda",
        "import gas_information_bench",
        "from gas_information_bench",
    )
    for path in (PROJECT_ROOT / "src/gf").rglob("*.py"):
        module_source = path.read_text(encoding="utf-8").lower()
        assert not any(forbidden in module_source for forbidden in forbidden_imports), path
