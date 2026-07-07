"""掘进通风场景的 schema 定义。

与 hydrogen_ng (`sim.core.schema`) 和 syngas (`sim.core.syngas_schema`) 并存。
三者通过 `composition_scheme` 字段区分，不共享 `COMPONENT_FIELDS` 等全局常量。

字段约定：
- COMPONENT_FIELDS: 预测目标列（3 列，sum=100% 严格闭包）
- BACKGROUND_FIELDS: 空（N₂ 是显式预测目标，不是背景气）
- SLOW_CHANNELS: 7 通道（V_NDIR_CO2 / V_TCS / T_C / P_MPa / H_RH / L_m / piston_position_m）

与 syngas 的关键差异：
- N₂ 升格为显式预测目标，写入 labels
- 数据层 sum=100% 严格闭包，但模型层不使用闭包残差头
- O₂ 是新增组分（同核双原子，无红外活性）
- 无 V_NDIR_CH4（场景无 CH₄）
"""
from __future__ import annotations


SCHEMA_VERSION = "tunnel-ventilation-1"
COMPOSITION_SCHEME = "tunnel_ventilation"

# 预测目标：CO2 / O2 / N2（sum=100%，N2 是显式目标而非背景）
COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")

# 无背景气：N₂ 在本场景是预测目标
BACKGROUND_FIELDS: tuple[str, ...] = ()

# 全部组分（用于声学/热导物理混合计算）
ALL_COMPONENT_FIELDS = (*COMPONENT_FIELDS, *BACKGROUND_FIELDS)

# 慢通道：7 通道。tv3 场景无 CH₄，不设 V_NDIR_CH4。
SLOW_CHANNELS = (
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)
SLOW_DYNAMIC_CHANNELS = ("V_NDIR_CO2", "V_TCS")
SLOW_MODAL_GROUPS = {
    "optical": ("V_NDIR_CO2",),
    "thermal": ("V_TCS",),
    "environment": ("T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
}

# Condition grid 包含全部预测目标（N2 在本场景写入 grid 和 labels）
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

# labels 写入 3 列预测目标（含 N2）
SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)

SLOW_SEQUENCE_FIELDS = ("sequence_id", "timestep", "timestamp_s", "phase_id", *SLOW_CHANNELS)
SPLIT_FIELDS = ("sequence_id", "mixture_id")
SPLIT_NAMES = ("train", "val", "test", "extrapolation")
