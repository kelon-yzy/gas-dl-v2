"""测试 ErrorNet。"""
import torch
from rcdw.models.error_net import ErrorNet


def test_output_shape():
    net = ErrorNet(in_dim=13, n_gas=3, hidden=32)
    feat = torch.randn(8, 3, 13)
    E = net(feat)
    assert E.shape == (8, 3, 3)


def test_output_non_negative():
    """Softplus 保证输出恒正。"""
    net = ErrorNet(in_dim=13, n_gas=3, hidden=32)
    feat = torch.randn(16, 3, 13)
    E = net(feat)
    assert (E >= 0).all()


def test_heads_independent():
    """三个 head 应有不同的参数。"""
    net = ErrorNet(in_dim=13, n_gas=3, hidden=32)
    p0 = list(net.heads[0].parameters())
    p1 = list(net.heads[1].parameters())
    # 随机初始化下参数不应完全相同
    assert not torch.equal(p0[0].data, p1[0].data)
