from __future__ import annotations


SCHEMA_VERSION = "v4-benchmark-1"

COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_N2")
SLOW_CHANNELS = (
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)
SLOW_DYNAMIC_CHANNELS = ("V_NDIR_CH4", "V_NDIR_CO2", "V_TCS")
SLOW_MODAL_GROUPS = {
    "optical": ("V_NDIR_CH4", "V_NDIR_CO2"),
    "thermal": ("V_TCS",),
    "environment": ("T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
}
PHASE_NAMES = ("baseline", "exposure", "steady", "recovery")
MULTI_PATH_PHASES = ("off", "baseline", "steady")
VALID_STORAGE_FORMATS = ("memmap", "npz", "both")
CONDITION_GRID_FIELDS = (
    "sequence_id",
    "mixture_id",
    *COMPONENT_FIELDS,
    "T_C_base",
    "P_MPa_base",
    "H_RH_base",
    "L_m_base",
    "status",
)
SEQUENCE_INDEX_FIELDS = ("sequence_id", "mixture_id", "stage_profile", "status", "n_timesteps", "dt_s")
SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)
SLOW_SEQUENCE_FIELDS = ("sequence_id", "timestep", "timestamp_s", "phase_id", *SLOW_CHANNELS)
SPLIT_FIELDS = ("sequence_id", "mixture_id")
SPLIT_NAMES = ("train", "val", "test", "extrapolation")
LEGACY_CONDITION_FIELDS = ("base_condition_id", "noise_seed_index", "noise_seed")
