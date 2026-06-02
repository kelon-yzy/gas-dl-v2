from __future__ import annotations

from torch import nn


def build_regression_head(in_features: int, out_dim: int, dropout: float) -> nn.Sequential:
    """共享 MLP 回归 head：in_features→128→ReLU→Drop→64→ReLU→Drop→out_dim。

    CNN1D 与 TCN 编码器在 ``AdaptiveAvgPool1d`` 池化后接入完全相同的 head，
    抽取到此处避免逐行复制（见 KARPATHY_REVIEW 2.1）。两模型的 head 设计若需
    分化，应在此显式参数化，而不是各自维护一份副本。
    """
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )
