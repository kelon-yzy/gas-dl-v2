"""测试扰动注入。"""
import torch
import pytest
from rcdw.perturbation.inject import inject, PERTURBATION_KINDS


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    return torch.randn(8, 8, 6).abs() + 0.5


def test_all_kinds_valid(sample_input):
    """五类扰动均可正常执行。"""
    for kind in PERTURBATION_KINDS:
        out = inject(sample_input, kind, 0.05)
        assert out.shape == sample_input.shape


def test_zero_level_unchanged(sample_input):
    """level=0 时输出与输入相同（确定性扰动）。"""
    for kind in ["optical_atten", "temperature"]:
        out = inject(sample_input, kind, 0.0)
        torch.testing.assert_close(out, sample_input)


def test_optical_atten_decreases(sample_input):
    """光学衰减使 NDIR 信号减小。"""
    out = inject(sample_input, "optical_atten", 0.1)
    # NDIR 通道 (index 0) 应该变小
    assert out[..., 0].mean() < sample_input[..., 0].mean()


def test_temperature_shift(sample_input):
    """温度突变使 T 通道增大。"""
    out = inject(sample_input, "temperature", 0.1)
    expected_shift = 0.1 * 80.0
    diff = (out[..., 4] - sample_input[..., 4]).mean()
    assert diff.item() == pytest.approx(expected_shift, abs=0.01)


def test_input_not_mutated(sample_input):
    """inject 不应修改原始输入。"""
    original = sample_input.clone()
    inject(sample_input, "thermal", 0.1)
    torch.testing.assert_close(sample_input, original)


def test_invalid_kind_raises(sample_input):
    with pytest.raises(ValueError):
        inject(sample_input, "nonexistent", 0.1)
