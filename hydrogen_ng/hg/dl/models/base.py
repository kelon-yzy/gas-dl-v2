from __future__ import annotations

import torch
from torch import nn


class BaseRegressor(nn.Module):
    """v4 回归模型基类。

    所有正式模型继承此类，统一 input_format 和 out_dim 语义。
    子类只需实现 ``forward``。
    """

    input_format: str = "NCT"

    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Conv1d/Linear: Kaiming Normal, fan_out + ReLU gain。
        # 选 fan_out 保前向方差，对当前 Conv1d+BN+ReLU 主线网络合适；
        # Transformer/PatchTST 实际激活是 GELU，gain 差异约 √2 vs ~1.39，量级相近未严格匹配，
        # 若后续这两个模型出现训练异常再单独按 module 类型走 GELU gain。
        # Linear.bias 故意不动：GasHeadNormalize._init_prior 已经在 __init__ 内写入先验 logits，
        # self.apply(_init_weights) 在 __init__ 末尾递归只覆盖 weight 不覆盖 bias，保留先验。
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
