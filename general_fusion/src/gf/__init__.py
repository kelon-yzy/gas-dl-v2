"""general_fusion (gf): 通用多模态气体融合主线包。

模块职责见 general_fusion/README.md 与 项目总体规划.md v6 §1.3。
A0 阶段仅提供包入口，实现待接口契约冻结后落地。
"""

from gf import dl, ml, pipeline, sim

__all__ = ["dl", "ml", "pipeline", "sim"]
