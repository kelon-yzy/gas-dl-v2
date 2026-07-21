"""掘进通风双向超声（F 线）schema 草案。

独立于基 schema ``tunnel-ventilation-1`` 与 COMSOL ``tunnel-ventilation-comsol-1``。
组分与 7 慢通道复用基 schema 常量；flow 不进 slow 通道。

F0 仅冻结字段名与数组契约；波形生成与 packaging 在 F1/F2。
"""
from __future__ import annotations

from tv3.sim.core.tunnel_ventilation_schema import (
    BACKGROUND_FIELDS,
    COMPONENT_FIELDS,
    COMPOSITION_SCHEME as BASE_COMPOSITION_SCHEME,
    SLOW_CHANNELS,
    SLOW_DYNAMIC_CHANNELS,
    SLOW_MODAL_GROUPS,
)

SCHEMA_VERSION = "tunnel-ventilation-bidir-1"
COMPOSITION_SCHEME = "tunnel_ventilation_bidir"
BASE_SCHEMA_VERSION = "tunnel-ventilation-1"
BASE_COMPOSITION_SCHEME_NAME = BASE_COMPOSITION_SCHEME

SIM_REVISION_TAG = "v7-bidir-flow-v1"
PHYSICS_BACKEND_BASE = "ideal_gas_wms_fracdelay"
PHYSICS_BACKEND_FLOW = "uniform_axial_flow_transit_time_v1"
PHYSICS_BACKEND = f"{PHYSICS_BACKEND_BASE}+{PHYSICS_BACKEND_FLOW}"
FORMAL_FEATURE_BUILDER = "raw_dsp_bidirectional_v1"

# 复用基场景：预测目标与慢通道不变
ALL_COMPONENT_FIELDS = (*COMPONENT_FIELDS, *BACKGROUND_FIELDS)

CONDITION_GRID_FLOW_FIELDS = (
    "v_path_m_per_s",
    "flow_scenario",
    "pair_interval_s",
    "delay_asymmetry_s",
    "jitter_correlation",
)

CONDITION_GRID_FIELDS = (
    "sequence_id",
    "mixture_id",
    *COMPONENT_FIELDS,
    "T_C_base",
    "P_MPa_base",
    "H_RH_base",
    "L_m_base",
    "status",
    *CONDITION_GRID_FLOW_FIELDS,
)

SEQUENCE_INDEX_FIELDS = ("sequence_id", "mixture_id", "stage_profile", "status", "n_timesteps", "dt_s")
SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)
SLOW_SEQUENCE_FIELDS = ("sequence_id", "timestep", "timestamp_s", "phase_id", *SLOW_CHANNELS)
SPLIT_FIELDS = ("sequence_id", "mixture_id")
SPLIT_NAMES = ("train", "val", "test", "extrapolation")

# 波形 / TOF 数组 stem（packaging 按 int16+scale 规则扩展）
BIDIR_WAVEFORM_STEMS = (
    "ultrasonic_ab",
    "ultrasonic_ba",
)

BIDIR_OBSERVED_ARRAYS = (
    "ultrasonic_tof_observed_ab_s",
    "ultrasonic_tof_observed_ba_s",
    "ultrasonic_peak_index_ab",
    "ultrasonic_peak_index_ba",
    "ultrasonic_tof_quality_ab",
    "ultrasonic_tof_quality_ba",
    "ultrasonic_tof_accepted_ab",
    "ultrasonic_tof_accepted_ba",
)

BIDIR_ORACLE_ARRAYS = (
    "ultrasonic_tof_true_ab_s",
    "ultrasonic_tof_true_ba_s",
    "ultrasonic_v_path_true_m_per_s",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_alpha_true_npm",
)

BIDIR_REQUIRED_ARRAYS = (
    *BIDIR_WAVEFORM_STEMS,
    *BIDIR_OBSERVED_ARRAYS,
    *BIDIR_ORACLE_ARRAYS,
)

SOURCE_TAGS = frozenset(
    {
        "implemented_physics",
        "literature_bound",
        "engineering_scenario",
        "not_represented",
    }
)

__all__ = [
    "ALL_COMPONENT_FIELDS",
    "BACKGROUND_FIELDS",
    "BASE_COMPOSITION_SCHEME_NAME",
    "BASE_SCHEMA_VERSION",
    "BIDIR_OBSERVED_ARRAYS",
    "BIDIR_ORACLE_ARRAYS",
    "BIDIR_REQUIRED_ARRAYS",
    "BIDIR_WAVEFORM_STEMS",
    "COMPONENT_FIELDS",
    "COMPOSITION_SCHEME",
    "CONDITION_GRID_FIELDS",
    "CONDITION_GRID_FLOW_FIELDS",
    "FORMAL_FEATURE_BUILDER",
    "PHYSICS_BACKEND",
    "PHYSICS_BACKEND_BASE",
    "PHYSICS_BACKEND_FLOW",
    "SCHEMA_VERSION",
    "SEQUENCE_INDEX_FIELDS",
    "SEQUENCE_LABEL_FIELDS",
    "SIM_REVISION_TAG",
    "SLOW_CHANNELS",
    "SLOW_DYNAMIC_CHANNELS",
    "SLOW_MODAL_GROUPS",
    "SLOW_SEQUENCE_FIELDS",
    "SOURCE_TAGS",
    "SPLIT_FIELDS",
    "SPLIT_NAMES",
]
