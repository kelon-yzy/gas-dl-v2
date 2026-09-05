"""注册 HEOS 单纯形插值，用于 O-KIN 超声反演。"""

from __future__ import annotations

import math

import numpy as np

from gf.sim.a2_dynamic_physics import evaluate_shared_physics
from gf.sim.ar_he_co2 import SYSTEM_DELAY_S
from gf.sim.a2_dynamic_audit._shared import (
    HEOS_INTERPOLATION_COMPOSITIONS,
    HEOS_INTERPOLATION_GRID_SIZE,
    HEOS_INTERPOLATION_STEP_PCT,
    TARGET_TOTAL,
)


def _registered_heos_interpolated_tof(
    composition: np.ndarray,
    *,
    temperature_k: float,
    pressure_pa: float,
    path_length_m: float,
    sound_speed_model_id: str,
    cache: dict[tuple[float, float], np.ndarray],
) -> float:
    cache_key = (float(temperature_k), float(pressure_pa))
    grid = cache.get(cache_key)
    if grid is None:
        reference = evaluate_shared_physics(
            HEOS_INTERPOLATION_COMPOSITIONS,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            path_length_m=1.0,
            sound_speed_model_id=sound_speed_model_id,
        )
        inverse_speed = 1.0 / np.asarray(reference["sound_speed_m_s"], dtype=np.float64)
        grid = np.full(
            (HEOS_INTERPOLATION_GRID_SIZE + 1, HEOS_INTERPOLATION_GRID_SIZE + 1),
            np.nan,
            dtype=np.float64,
        )
        for sample, value in zip(HEOS_INTERPOLATION_COMPOSITIONS, inverse_speed):
            ar_index = int(round(float(sample[0]) / HEOS_INTERPOLATION_STEP_PCT))
            he_index = int(round(float(sample[1]) / HEOS_INTERPOLATION_STEP_PCT))
            grid[ar_index, he_index] = float(value)
        if not np.isfinite(grid[~np.isnan(grid)]).all():
            raise ValueError("registered HEOS interpolation table is not finite")
        cache[cache_key] = grid
    values = np.asarray(composition, dtype=np.float64)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > TARGET_TOTAL)
    ):
        raise ValueError("registered HEOS interpolation expects a finite composition in [0,100]")
    if not math.isclose(float(values.sum()), TARGET_TOTAL, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("registered HEOS interpolation composition must sum to 100 mol%")
    helium_units = float(values[1] / HEOS_INTERPOLATION_STEP_PCT)
    carbon_dioxide_units = float(values[2] / HEOS_INTERPOLATION_STEP_PCT)
    helium_index = int(math.floor(helium_units))
    carbon_dioxide_index = int(math.floor(carbon_dioxide_units))
    helium_fraction = helium_units - helium_index
    carbon_dioxide_fraction = carbon_dioxide_units - carbon_dioxide_index
    if helium_index + carbon_dioxide_index == HEOS_INTERPOLATION_GRID_SIZE:
        if not math.isclose(helium_fraction, 0.0, abs_tol=1.0e-10) or not math.isclose(
            carbon_dioxide_fraction, 0.0, abs_tol=1.0e-10
        ):
            raise ValueError("registered HEOS interpolation reached an invalid simplex boundary")
        inverse_speed_at_composition = float(grid[0, helium_index])
    elif helium_index + carbon_dioxide_index < HEOS_INTERPOLATION_GRID_SIZE:
        if helium_fraction + carbon_dioxide_fraction <= 1.0:
            vertices = (
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index + 1,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index,
                    ]
                ),
            )
            inverse_speed_at_composition = (
                vertices[0]
                + helium_fraction * (vertices[1] - vertices[0])
                + carbon_dioxide_fraction * (vertices[2] - vertices[0])
            )
        else:
            vertices = (
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index + 1,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 2
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index + 1,
                    ]
                ),
            )
            inverse_speed_at_composition = (
                (1.0 - carbon_dioxide_fraction) * vertices[0]
                + (1.0 - helium_fraction) * vertices[1]
                + (helium_fraction + carbon_dioxide_fraction - 1.0) * vertices[2]
            )
    else:
        raise ValueError("registered HEOS interpolation composition is outside the simplex")
    inverse_speed_at_composition = float(inverse_speed_at_composition)
    if not math.isfinite(inverse_speed_at_composition) or inverse_speed_at_composition <= 0.0:
        raise ValueError("registered HEOS interpolation produced an invalid inverse speed")
    tof = float(path_length_m) * inverse_speed_at_composition + SYSTEM_DELAY_S
    if not math.isfinite(tof) or tof <= 0.0:
        raise ValueError("registered HEOS interpolation produced an invalid ToF")
    return tof
