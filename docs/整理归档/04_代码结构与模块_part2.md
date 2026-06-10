> 复核说明：本页按当前真实目录修正。`run_all_training.ps1` 位于项目根目录，不在 `experiments/` 下；部分结果目录仍处于“已有部分产物”状态

## 7. 实验脚本（experiments/）

### 7.1 PowerShell 脚本

| 脚本 | 功能 | 对应目标 |
|------|------|----------|
| `exp01_traditional.ps1` | 传统 ML 基线训练 | G1 |
| `exp02_deep_e2e.ps1` | DL 端到端基线训练 | G1 |
| `exp03_fusion_grid.ps1` | 模态与融合对比 | G2 |
| `exp04_adaptation.ps1` | 环境适应性评估（G3a+G3b+G3c）| G3 |
| `exp06_reproducibility.ps1` | 多 seed 重复性检测 | R1 |

**项目根目录脚本**：

| 脚本 | 功能 | 对应目标 |
|------|------|----------|
| `run_all_training.ps1` | DL 全量训练批处理（当前脚本枚举 12 个配置入口） | G1/G2 |

### 7.2 脚本示例

```powershell
# exp01_traditional.ps1
python src\pipeline\feature_extraction.py `
  --source-dir data\waveform_v3_seedpath_formal `
  --output-dir outputs\exp01_traditional_seedpath

python src\pipeline\train_traditional.py `
  --data-dir outputs\exp01_traditional_seedpath `
  --output-root outputs\exp01_traditional_seedpath `
  --tag formal_seedpath `
  --seed 42 `
  --split-dir data\waveform_v3_seedpath_formal\splits `
  --profiles v3_raw_no_env v3_raw_tph `
  --combo-list svr_ridge pls_ridge xgboost_ridge `
  --max-workers 4
```

## 8. 输出目录（outputs/）

### 8.1 目录结构

```
outputs/
├── STATUS.tsv                        # 状态表（当前内容不完整）
├── exp01_traditional/                # 传统 ML 结果
│   ├── data/                         # 历史特征表
│   └── runs/                         # 历史 + seedpath grid summary
├── exp01_traditional_seedpath/       # seedpath_formal 中间产物
│   ├── features/
│   ├── labels/
│   └── training/
├── exp02_deep_e2e/                   # DL 端到端基线
│   ├── v3_tcn_multimodal_seed42/
│   │   ├── train_log.csv
│   │   ├── predictions.csv
│   │   ├── summary.json
│   │   └── checkpoints/
│   └── ...
├── exp03_full_training/              # DL 全量训练（当前已有若干完成 run）
├── exp04_adaptation/                 # 环境适应性
│   └── G3a_sensitivity/              # 当前已见实际输出
├── archive/                          # 归档
│   ├── traditional_before_speed_opt_20260514_1800/
│   ├── traditional_smoke_20260516/
│   └── pre_adaptation_redesign_20260516/
├── deep_training_curves/             # 训练曲线图
└── summary/                          # 最终汇总
    ├── results.tsv                   # 全部实验结果
    ├── results_multiseed.tsv         # 多 seed 统计
    └── ...                           # 其它文件按运行结果补充
```

### 8.2 关键输出文件

| 文件 | 内容 | 生成方式 |
|------|------|----------|
| `STATUS.tsv` | 实验状态表 | `status_store.py` 维护 |
| `summary/results.tsv` | 全部实验结果 | `python src/pipeline/aggregate.py` |
| `summary/results_multiseed.tsv` | 多 seed 聚合结果 | `python src/pipeline/aggregate.py` |
| `train_log.csv` | 训练日志（每 epoch）| 训练时自动生成 |
| `predictions.csv` | 测试集预测 | 训练结束自动生成 |
| `summary.json` | 单次训练汇总 | 训练结束自动生成 |
| `grid_summary.csv` | 传统 ML 网格结果 | 网格搜索自动生成 |

## 9. 代码风格与约定

### 9.1 命名约定

| 类型 | 约定 | 示例 |
|------|------|------|
| 类名 | PascalCase | `WaveformSequenceDataset` |
| 函数/变量 | snake_case | `train_epoch()`, `batch_size` |
| 常量 | UPPER_SNAKE_CASE | `MAX_EPOCHS`, `DEFAULT_LR` |
| 私有 | 下划线前缀 | `_internal_function()` |
| 配置键 | snake_case | `learning_rate`, `num_layers` |

### 9.2 文档字符串

```python
def train_model(cfg: Config) -> Model:
    """Train the model.

    Args:
        cfg: Training configuration object.

    Returns:
        Trained model instance.

    Raises:
        ValueError: When configuration is invalid.
    """
    ...
```

### 9.3 类型提示

```python
from typing import Dict, List, Optional, Tuple

def process_data(
    data: List[Dict],
    config: Config
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    ...
```

### 9.4 Import 顺序

```python
# 1. 标准库
import os
from pathlib import Path

# 2. 第三方库
import torch
import numpy as np
from hydra import compose, initialize

# 3. 本地模块
from src.dl.data import WaveformSequenceDataset
from src.dl.models import MODEL_REGISTRY
```

## 10. 代码原则（研究型项目）

**不做防御性编程**：

```python
# ❌ 避免这样写
try:
    data = load_data(path)
except FileNotFoundError:
    logger.warning("File not found, using default")
    data = default_data

# ✅ 推荐这样写
data = load_data(path)  # 文件不存在就让它崩，定位最快
```

**具体规则**：
- ❌ 不写 try/except；路径错、参数错、文件缺失，让它崩出栈
- ❌ 不做默认值兜底；配置漏了就报错
- ❌ 不做参数边界检查；输入异常就让 numpy/torch 自己崩
- ✅ 文档与状态控制照常做，但代码只走 happy path

## 11. 模块依赖关系

```
pipeline/
  ↓ 调用
dl/         ml/         sim/
  ↓          ↓           ↓
data/    patent_model/ scripts/
  ↓          ↓           ↓
models/  scripts/    sim_common/
  ↓
training/
```

**说明**：
- `pipeline/` 是最上层，调用 `dl/`, `ml/`, `sim/`
- `dl/`, `ml/`, `sim/` 三者互不依赖
- 当前 `pipeline/` 下没有 `shim.py`，依赖关系主要通过项目根目录运行入口和包导入解决

## 12. 关键接口

### 12.1 数据集接口

```python
# src/dl/data/dataset_waveform.py
class WaveformSequenceDataset(Dataset):
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'ultrasonic': ...,     # [T, 1000]
            'fiber_mic': ...,      # [T, 2000]
            'slow': ...,           # [T, 8]
            'target': ...          # [4]
        }
```

### 12.2 模型接口

```python
# src/dl/models/
class BaseModel(nn.Module):
    def __init__(self, config: Dict):
        ...
    
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        # 返回 [batch_size, 4] 的预测
        ...
```

### 12.3 训练接口

```python
# src/dl/training/orchestrator.py
result = train_one(config_path)

# 关键入口：
# - train_one(...)
# - train_config(...)
# - TrainRunOptions / TrainDependencies / TrainExecutionContext
```

### 12.4 传统 ML 接口

```python
# src/ml/patent_model/modeling.py
model = TraditionalFusionModel(config)
model.fit(dataset)
pred = model.predict(dataset)

# pred.by_model 包含：
# - acoustic
# - optical
# - thermal
# - fused
```

## 13. 测试（tests/）

### 13.1 测试结构

```
tests/
├── test_acoustic_waveform_v3.py          # 通道 1 单测
├── test_acoustic_fiber_mic_v3.py         # 通道 2 单测
├── test_generate_waveform_dataset.py     # 数据生成单测
├── test_check_waveform_directionality.py # 质量检查单测
├── test_plot_env_robustness_compare_mixture.py
├── test_plot_traditional_env_comparison.py
└── test_plot_traditional_single_profile_sensitivity.py
```

### 13.2 运行测试

```powershell
# 运行全部测试
pytest

# 运行特定测试
pytest tests/test_acoustic_waveform_v3.py

# 带覆盖率
pytest --cov=src tests/
```

## 14. 版本控制（git）

### 14.1 分支策略

| 分支 | 用途 | 说明 |
|------|------|------|
| `main` | 主分支 | 稳定版本 |
| `codex/robustness-compare-plan` | 当前工作分支 | 环境适应性对比计划 |

### 14.2 Git 状态（2026-06-08）

```
Current branch: codex/robustness-compare-plan
Main branch: main
Git user: ai98pro

Untracked files:
- artifacts/
- data/
- docs/新项目目标架构说明.md
- docs/CNN1D-TCN当前模型说明.md
- src/pipeline/plot_*.py (3 个)
- tests/test_plot_*.py (3 个)
- 新项目PLAN.md
```

## 15. 依赖管理

### 15.1 主要依赖

| 包 | 版本 | 用途 |
|---|------|------|
| torch | ≥1.13 | 深度学习框架 |
| numpy | ≥1.23 | 数值计算 |
| pandas | ≥1.5 | 数据处理 |
| scikit-learn | ≥1.2 | 传统 ML |
| xgboost | ≥1.7 | 梯度提升树 |
| matplotlib | ≥3.6 | 绘图 |
| pyyaml | ≥6.0 | 配置解析 |

### 15.2 安装

```powershell
# 深度学习环境
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install -r requirements.txt
```

## 16. 开发工具

### 16.1 代码检查

```powershell
# 类型检查
mypy src/

# 代码风格
ruff check .

# 自动格式化
black src/
```

### 16.2 IDE 配置

**推荐 IDE**：
- VS Code + Python extension
- PyCharm Professional

**VS Code 配置**：
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

## 17. 性能优化

### 17.1 传统 ML 优化

| 优化项 | 配置 | 说明 |
|--------|------|------|
| CPU 并行 | `n_jobs=-1` | 使用全部核心 |
| XGBoost 线程 | `xgb_n_jobs=4` | 单模型线程数 |
| 多 combo 并行 | `max_workers=4` | profile/combo 并行数 |

### 17.2 深度学习优化

| 优化项 | 配置 | 说明 |
|--------|------|------|
| 数据加载 | `num_workers=4` | 多线程加载 |
| 混合精度 | `torch.cuda.amp` | 可选（需配置）|
| 梯度累积 | 可选 | 模拟大 batch |
| 分布式训练 | 暂未实现 | 未来可选 |

## 18. 调试技巧

### 18.1 快速验证

```powershell
# 深度学习：1 epoch smoke test
python src\pipeline\train_deep.py \
  --config configs\deep\slow_only_tcn_formal.yaml \
  --epochs 1 \
  --no-ui

# 传统 ML：单 profile + 单 combo
python src\pipeline\train_traditional.py \
  --data-dir outputs\exp01_traditional_seedpath \
  --output-root outputs\test_smoke \
  --tag smoke \
  --seed 42 \
  --profiles v3_raw_tph \
  --combo-list xgboost_ridge \
  --max-workers 1 \
  --no-ui
```

### 18.2 日志查看

```powershell
# 查看训练日志
cat outputs\exp02_deep_e2e\v3_tcn_multimodal_seed42\train.log

# 查看训练曲线
python src\pipeline\plot_deep_training_curves.py \
  --root outputs\exp02_deep_e2e\v3_tcn_multimodal_seed42
```

## 19. 常见问题

### 19.1 数据加载慢

**问题**：`num_workers > 0` 时 Windows 下卡住

**解决**：
```python
if __name__ == '__main__':
    # Windows 下需要这个保护
    main()
```

### 19.2 OOM（显存不足）

**解决**：
- 降低 `batch_size`
- 缩短输入序列长度
- 使用梯度累积

### 19.3 训练不收敛

**检查清单**：
- [ ] 学习率是否合适（建议 1e-3 ~ 1e-4）
- [ ] 数据是否标准化
- [ ] 损失函数是否合理
- [ ] 模型是否过大/过小

## 20. 参考文档

| 文档 | 内容 |
|------|------|
| `docs/design.md` | 实验设计 |
| `docs/dl优化器实现.md` | 深度学习优化器 |
| `docs/训练速度优化计划.md` | 性能优化 |
| `docs/code_changelog.md` | 代码修改记录 |
| `CLAUDE.md` | 项目协作规则 |
| `.claude/CLAUDE.md` | 全局协作规则 |
