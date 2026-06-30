"""测试 13 维特征提取器。"""
import torch
import pytest
from rcdw.models.feature import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor(window=8)


def test_output_shape(extractor):
    x = torch.randn(4, 8, 6)
    Y = torch.rand(4, 3, 3)
    Y = Y / Y.sum(dim=-1, keepdim=True)
    feat = extractor(x, Y)
    assert feat.shape == (4, 3, 13)


def test_features_not_all_zero(extractor):
    """滑窗模式下，时序统计特征不应全部为零。"""
    torch.manual_seed(0)
    x = torch.randn(8, 8, 6)
    Y = torch.rand(8, 3, 3)
    feat = extractor(x, Y)
    # CV, gradient, drift, snr_proxy 应有非零值
    for f_idx in [0, 2, 10, 11]:
        assert feat[:, :, f_idx].abs().sum() > 0, f"feature {f_idx} is all zero"


def test_cv_positive(extractor):
    """变异系数应 >= 0。"""
    x = torch.randn(4, 8, 6).abs() + 0.1
    Y = torch.rand(4, 3, 3)
    feat = extractor(x, Y)
    assert (feat[:, :, 0] >= 0).all()


def test_quality_ratio_sum_one(extractor):
    """Q_m (feature 3) 对所有模态 sum ≈ 1。"""
    x = torch.randn(8, 8, 6)
    Y = torch.rand(8, 3, 3)
    feat = extractor(x, Y)
    Q_sum = feat[:, :, 3].sum(dim=1)  # (B,)
    torch.testing.assert_close(Q_sum, torch.ones_like(Q_sum), atol=1e-4, rtol=1e-4)


def test_dt_constant(extractor):
    """dt (feature 12) 应恒为 1.0。"""
    x = torch.randn(4, 8, 6)
    Y = torch.rand(4, 3, 3)
    feat = extractor(x, Y)
    torch.testing.assert_close(
        feat[:, :, 12], torch.ones(4, 3), atol=1e-6, rtol=1e-6
    )


def test_group_bias_symmetric(extractor):
    """当所有模态预测相同时，群体偏差 B (feature 4) 应为 0。"""
    x = torch.randn(4, 8, 6)
    Y = torch.rand(4, 1, 3).expand(4, 3, 3).clone()  # 三模态相同
    feat = extractor(x, Y)
    torch.testing.assert_close(
        feat[:, :, 4], torch.zeros(4, 3), atol=1e-6, rtol=1e-6
    )
