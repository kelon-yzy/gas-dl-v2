"""A0 骨架自检：包可安装、模块入口可导入。

接口契约 smoke（两个数据集适配器通过同一 FusionCore 前向）待契约冻结后
在此目录新增测试，本文件只验证骨架完整性（总体规划 v6 §3.3 动作 4 前置）。
"""

import gf


def test_package_exposes_module_entrypoints() -> None:
    assert gf.__all__ == ["dl", "ml", "pipeline", "sim"]
    for name in gf.__all__:
        module = getattr(gf, name)
        assert module is not None


def test_fusion_core_has_no_dataset_hardcoding() -> None:
    """不变量：融合核心入口不得出现数据集名称分支（总体规划 §1.2）。"""
    import inspect

    from gf import dl

    source = inspect.getsource(dl)
    for forbidden in ("ar_he", "xylene", "dataset_id =="):
        assert forbidden not in source.lower()
