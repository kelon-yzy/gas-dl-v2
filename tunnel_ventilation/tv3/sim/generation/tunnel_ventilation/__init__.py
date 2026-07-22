"""掘进通风场景物理仿真子包。

与 `sim.generation`（hydrogen_ng）和 `sim.generation.syngas` 并存，互不影响。

模块：
- conditions: 2D LHS 采样（CO2, O2），N2 = 100 - CO2 - O2
- acoustic_physics: 含 O2 项的声速/衰减/热导（移除 H2/CH4 项）
- slow: 7 通道慢通道生成（无 V_NDIR_CH4，场景无 CH₄）
- benchmark: tv3 benchmark dataset 生成
"""
from __future__ import annotations

from tv3.sim.generation.tunnel_ventilation.benchmark import (
    DEFAULT_WAVEFORM_PATH_LMS,
    TunnelVentilationBenchmarkGenerationSpec,
    default_worker_count,
    generate_tunnel_ventilation_benchmark_dataset,
)
from tv3.sim.generation.tunnel_ventilation.conditions import (
    L_M_BASE_RANGE,
    TUNNEL_VENTILATION_RANGES,
    WIDE_COMPOSITION_RANGES,
    TunnelVentilationRanges,
    build_tunnel_ventilation_label_rows,
    generate_tunnel_ventilation_condition_rows,
    resolve_composition_ranges,
)

__all__ = [
    "DEFAULT_WAVEFORM_PATH_LMS",
    "L_M_BASE_RANGE",
    "TUNNEL_VENTILATION_RANGES",
    "WIDE_COMPOSITION_RANGES",
    "TunnelVentilationBenchmarkGenerationSpec",
    "TunnelVentilationRanges",
    "build_tunnel_ventilation_label_rows",
    "default_worker_count",
    "generate_tunnel_ventilation_benchmark_dataset",
    "generate_tunnel_ventilation_condition_rows",
    "resolve_composition_ranges",
]
