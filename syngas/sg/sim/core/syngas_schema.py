"""合成气场景的 schema 定义。

与 hydrogen_ng 场景 (`sim.core.schema`) 并存。两者通过 `composition_scheme`
字段区分，不共享 `COMPONENT_FIELDS` 等全局常量。

字段约定：
- COMPONENT_FIELDS: 预测目标列（4 列，sum<100%）
- BACKGROUND_FIELDS: 背景气列（参与物理仿真，不入 labels）
- SLOW_CHANNELS: 慢通道，新增 V_NDIR_CO
"""
from __future__ import annotations


SCHEMA_VERSION = "v4-syngas-1"
COMPOSITION_SCHEME = "syngas"

# 预测目标：H2 / CH4 / CO2 / CO（sum<100%，N2 是背景）
COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_CO")

# 背景气：仅参与物理仿真，不入 labels；x_N2 = 100 - sum(COMPONENT_FIELDS)
BACKGROUND_FIELDS = ("x_N2",)

# 全部组分（用于声学/光学物理混合计算）
ALL_COMPONENT_FIELDS = (*COMPONENT_FIELDS, *BACKGROUND_FIELDS)

# 慢通道：在 hydrogen_ng 基础上新增 V_NDIR_CO
SLOW_CHANNELS = (
    "V_NDIR_CH4",
    "V_NDIR_CO2",
    "V_NDIR_CO",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)
SLOW_DYNAMIC_CHANNELS = ("V_NDIR_CH4", "V_NDIR_CO2", "V_NDIR_CO", "V_TCS")
SLOW_MODAL_GROUPS = {
    "optical": ("V_NDIR_CH4", "V_NDIR_CO2", "V_NDIR_CO"),
    "thermal": ("V_TCS",),
    "environment": ("T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
}

# Condition grid 显式包含 x_N2（不通过 *COMPONENT_FIELDS 自动添加，因为
# COMPONENT_FIELDS 不再含 N2，但仿真层仍需要它）
CONDITION_GRID_FIELDS = (
    "sequence_id",
    "mixture_id",
    *COMPONENT_FIELDS,
    *BACKGROUND_FIELDS,
    "T_C_base",
    "P_MPa_base",
    "H_RH_base",
    "L_m_base",
    "status",
)
SEQUENCE_INDEX_FIELDS = ("sequence_id", "mixture_id", "stage_profile", "status", "n_timesteps", "dt_s")

# labels 仅写入 4 列预测目标，不含 N2
SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)

SLOW_SEQUENCE_FIELDS = ("sequence_id", "timestep", "timestamp_s", "phase_id", *SLOW_CHANNELS)
SPLIT_FIELDS = ("sequence_id", "mixture_id")
SPLIT_NAMES = ("train", "val", "test", "extrapolation")
