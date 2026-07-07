#!/usr/bin/env python
"""诊断脚本：检查 CNN1DTCNFusionRegressor 的实际签名"""
import inspect

from hg.dl.models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor

# 获取 __init__ 的签名
sig = inspect.signature(CNN1DTCNFusionRegressor.__init__)

print("CNN1DTCNFusionRegressor.__init__ signature:")
print(sig)
print()

print("Parameters:")
for param_name, param in sig.parameters.items():
    if param_name == "self":
        continue
    default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
    print(f"  {param_name}: {param.annotation} = {default}")
print()

# 检查 output_mode 是否存在
if "output_mode" in sig.parameters:
    print("✅ output_mode parameter EXISTS")
    output_mode_param = sig.parameters["output_mode"]
    print(f"   Default value: {output_mode_param.default}")
    print(f"   Annotation: {output_mode_param.annotation}")
else:
    print("❌ output_mode parameter MISSING")
    print("   This explains the TypeError!")
