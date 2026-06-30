"""测试 13 维特征提取器（v1.2 §8.4 12 维通道布局）。"""
from __future__ import annotations

import pytest
import torch

from rcdw.models.feature import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor(window=8)


def test_output_shape(extractor):
    x = torch.randn(4, 8, 12)
    Y = torch.rand(4, 3, 3)
    Y = Y / Y.sum(dim=-1, keepdim=True)
    feat = extractor(x, Y)
    assert feat.shape == (4, 3, 13)


def test_rejects_legacy_6_channel_input(extractor):
    """v1.2: 拒收旧 6 维输入，必须用 12 维新布局。"""
    x = torch.randn(4, 8, 6)
    Y = torch.rand(4, 3, 3)
    with pytest.raises(ValueError, match="12-channel"):
        extractor(x, Y)


def test_features_not_all_zero(extractor):
    torch.manual_seed(0)
    x = torch.randn(8, 8, 12)
    Y = torch.rand(8, 3, 3)
    feat = extractor(x, Y)
    # CV, gradient, drift, snr_proxy 应有非零值
    for f_idx in [0, 2, 10, 11]:
        assert feat[:, :, f_idx].abs().sum() > 0, f"feature {f_idx} is all zero"


def test_cv_positive(extractor):
    x = torch.randn(4, 8, 12).abs() + 0.1
    Y = torch.rand(4, 3, 3)
    feat = extractor(x, Y)
    assert (feat[:, :, 0] >= 0).all()


def test_quality_ratio_sum_one(extractor):
    """Q_m (feature 3) 对所有模态 sum ≈ 1。"""
    x = torch.randn(8, 8, 12)
    Y = torch.rand(8, 3, 3)
    feat = extractor(x, Y)
    Q_sum = feat[:, :, 3].sum(dim=1)
    torch.testing.assert_close(Q_sum, torch.ones_like(Q_sum), atol=1e-4, rtol=1e-4)


def test_dt_constant(extractor):
    """dt (feature 12) 应恒为 1.0。"""
    x = torch.randn(4, 8, 12)
    Y = torch.rand(4, 3, 3)
    feat = extractor(x, Y)
    torch.testing.assert_close(
        feat[:, :, 12], torch.ones(4, 3), atol=1e-6, rtol=1e-6
    )


def test_group_bias_symmetric(extractor):
    x = torch.randn(4, 8, 12)
    Y = torch.rand(4, 1, 3).expand(4, 3, 3).clone()
    feat = extractor(x, Y)
    torch.testing.assert_close(
        feat[:, :, 4], torch.zeros(4, 3), atol=1e-6, rtol=1e-6
    )


def test_env_delta_reads_from_correct_indices(extractor):
    """v1.2: delta_T / delta_P / delta_RH 应读取 ENV_INDICES = [2, 3, 4]。"""
    x = torch.zeros(2, 8, 12)
    # 在最后两时刻分别在 IDX_T_C=2, IDX_P_MPa=3, IDX_H_RH=4 注入差异
    x[:, -2, 2] = 10.0
    x[:, -1, 2] = 13.0  # delta_T = 3
    x[:, -2, 3] = 1.0
    x[:, -1, 3] = 1.5   # delta_P = 0.5
    x[:, -2, 4] = 40.0
    x[:, -1, 4] = 50.0  # delta_RH = 10
    Y = torch.rand(2, 3, 3)
    feat = extractor(x, Y)
    # feature 5/6/7 对所有 modal 相同
    for m in range(3):
        torch.testing.assert_close(
            feat[:, m, 5], torch.tensor([3.0, 3.0]), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            feat[:, m, 6], torch.tensor([0.5, 0.5]), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            feat[:, m, 7], torch.tensor([10.0, 10.0]), atol=1e-5, rtol=1e-5
        )
