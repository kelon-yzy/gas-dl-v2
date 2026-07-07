"""合成气场景物理仿真子包。

与 `sim.generation` 中的 hydrogen_ng 实现并存，互不影响。

模块：
- conditions: 4D LHS + 条件顺序采样
- acoustic_physics: 含 CO 项的声速/衰减/热导
- optical_crosstalk: 3×3 光学串扰矩阵
- slow: 含 V_NDIR_CO 的慢通道生成
- benchmark: syngas benchmark dataset 生成（与 hg benchmark 并存）
"""
from __future__ import annotations

from sg.sim.generation.syngas.benchmark import (
    SyngasBenchmarkGenerationSpec,
    generate_syngas_benchmark_dataset,
)
from sg.sim.generation.syngas.conditions import (
    SYNGAS_RANGES,
    build_syngas_label_rows,
    generate_syngas_condition_rows,
    is_feasible_syngas,
)

__all__ = [
    "SYNGAS_RANGES",
    "SyngasBenchmarkGenerationSpec",
    "build_syngas_label_rows",
    "generate_syngas_benchmark_dataset",
    "generate_syngas_condition_rows",
    "is_feasible_syngas",
]
