"""A2-DYN 审计共享常量与跨类别工具。

各审计子模块（schema / physics / dynamic / baselines / jacobian /
freeze）只从本模块取共享常量与工具，彼此不互相导入，避免循环依赖。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from gf.sim.a2_dynamic_dataset import DynamicDataset


AUDIT_SCHEMA_VERSION = "gf-a2-dynamic-audit-2"
OBSERVED_ADMISSION_SIGMA_FACTOR = 5.0
OBSERVED_ADMISSION_DRIFT_MINUTES = 4.0
DEVELOPMENT_SPLITS = ("train", "val", "stress_val")
FAMILIES = (
    "D-IID",
    "D-KINETICS",
    "D-PROTOCOL",
    "D-NOISE-DRIFT",
    "D-ENV-CAL",
    "D-JOINT",
)
EARLY_HORIZONS = ("P015", "P030", "P060")
AUDIT_HORIZONS = ("P005", "P015", "P030", "P060", "P120", "P150")
MIN_UNIQUE_QUANTIZED_LEVELS = 10
TARGET_TOTAL = 100.0
PURGE_COMPOSITION = np.asarray([100.0, 0.0, 0.0], dtype=np.float64)
HEOS_INTERPOLATION_STEP_PCT = 1.0
HEOS_INTERPOLATION_GRID_SIZE = int(round(TARGET_TOTAL / HEOS_INTERPOLATION_STEP_PCT))
HEOS_INTERPOLATION_TOF_TOLERANCE_S = 1.0e-7
TARGET_TANGENT_DIRECTIONS = np.asarray(
    [[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]],
    dtype=np.float64,
)
HEOS_INTERPOLATION_COMPOSITIONS = np.asarray(
    [
        [
            HEOS_INTERPOLATION_STEP_PCT * float(ar_index),
            HEOS_INTERPOLATION_STEP_PCT * float(he_index),
            TARGET_TOTAL
            - HEOS_INTERPOLATION_STEP_PCT * float(ar_index)
            - HEOS_INTERPOLATION_STEP_PCT * float(he_index),
        ]
        for ar_index in range(HEOS_INTERPOLATION_GRID_SIZE + 1)
        for he_index in range(HEOS_INTERPOLATION_GRID_SIZE + 1 - ar_index)
    ],
    dtype=np.float64,
)


def _dataset_arrays(dataset: DynamicDataset) -> dict[str, np.ndarray]:
    """导出数据集数组字典（content hash 与审计共用的稳定视图）。"""
    return {
        "signals": dataset.signals,
        "valid_mask": dataset.valid_mask,
        "quality": dataset.quality,
        "time_s": dataset.time_s,
        "target": dataset.target,
        "phase_id": dataset.phase_id,
        "observation_index": dataset.observation_index,
        "inlet_composition": dataset.inlet_composition,
        "inlet_coefficient": dataset.inlet_coefficient,
        "chamber_composition": dataset.chamber_composition,
        "equilibrium_reference_signals": dataset.equilibrium_reference_signals,
        "clean_device_signals": dataset.clean_device_signals,
        "device_states": dataset.device_states,
        "privileged_parameters": dataset.privileged_parameters,
        **dict(dataset.device_audit),
    }


def _horizon_indices(
    time_s: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    data_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """各前缀 horizon 在每条序列上的端点位置；cutoff 进入 recovery 的行置 -1。"""
    result: dict[str, np.ndarray] = {}
    for horizon in data_config["prefix_horizons"]:
        horizon_id = str(horizon["horizon_id"])
        exposure_after = float(horizon["exposure_after_s"])
        indices = np.empty(len(records), dtype=np.int64)
        for row, record in enumerate(records):
            if horizon_id == "FULL":
                cutoff = float(time_s[-1])
            else:
                cutoff = (
                    float(record["exposure_onset_s"])
                    + exposure_after
                    - float(data_config["dt_s"])
                )
            position = int(np.searchsorted(time_s, cutoff, side="right") - 1)
            if (
                position < 0
                or position >= len(time_s)
                or (
                    horizon_id != "FULL"
                    and cutoff >= float(records[row]["exposure_end_s"])
                )
            ):
                indices[row] = -1
            else:
                indices[row] = position
        result[horizon_id] = indices
    return result
