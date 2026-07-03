"""RCDW 数据集 schema 常量。

独立于 ``src/sim/core/schema.py``，命名空间前缀为 ``rcdw-``。
对应方案 §4。
"""

from __future__ import annotations


SCHEMA_VERSION = "rcdw-benchmark-1"

# 三组分闭包：O2 / CO2 / N2，sum = 100%；自由度 d = 2。
COMPONENT_FIELDS = ("x_O2", "x_CO2", "x_N2")
GAS_DISPLAY_NAMES = tuple(name.removeprefix("x_") for name in COMPONENT_FIELDS)

# 7 个慢通道。与 HG 主线相比：删除 V_NDIR_CH4（RCDW 无 CH4）。
SLOW_CHANNELS = (
    "V_NDIR_CO2",         # NDIR CO2 通道电压 [V]
    "V_TCS",              # 热导传感器电压 [V]
    "T_C",                # 温度 [°C]
    "P_MPa",              # 压力 [MPa]
    "H_RH",               # 相对湿度 [%]
    "L_m",                # 声程 [m]
    "piston_position_m",  # 活塞位置 [m]
)

# 参与 RC 动力学 + drift + random walk + noise 的通道。
# 环境通道直接取值，不经过动力学。
SLOW_DYNAMIC_CHANNELS = ("V_NDIR_CO2", "V_TCS")

# 模态分组：用于 scaler modal_groups 输出。
SLOW_MODAL_GROUPS = {
    "optical": ("V_NDIR_CO2",),
    "thermal": ("V_TCS",),
    "environment": ("T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
}

# Phase 时间结构（与 HG 一致）。
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

SEQUENCE_INDEX_FIELDS = (
    "sequence_id",
    "mixture_id",
    "stage_profile",
    "status",
    "n_timesteps",
    "dt_s",
)

SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)

SLOW_SEQUENCE_FIELDS = (
    "sequence_id",
    "timestep",
    "timestamp_s",
    "phase_id",
    *SLOW_CHANNELS,
)

SPLIT_FIELDS = ("sequence_id", "mixture_id")

# RCDW 删除 extrapolation split。理由：三组分范围已覆盖全 simplex，
# 外推 holdout 无物理意义。详见方案 §2.6。
SPLIT_NAMES = ("train", "val", "test")

# 防御性黑名单：禁止重新引入 HG 主线已废弃的字段命名。
LEGACY_CONDITION_FIELDS = ("base_condition_id", "noise_seed_index", "noise_seed")
