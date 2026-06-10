> 复核说明：本页强调“如何跑”和“如何验收”。其中五同约束属于目标口径；当前仓库实际产物仍存在 `waveform_v3` 与 `seedpath_formal` 并存的情况

## 6. 公平对比五同约束

为确保 ML 与 DL 的公平对比，严格执行以下约束：

| 约束 | 目标实现 | 当前复核状态 |
|------|----------|--------------|
| **同数据源** | 主线统一使用 `data/waveform_v3_seedpath_formal/`（30000 序列双通道） | 传统 ML 已接入；深度学习 YAML 仍指向 `data/waveform_v3/` |
| **同划分** | 锁定 4 路 split（train/val/test/extrapolation），按 mixture_id 分组 | `seedpath_formal` 已具备；旧深度学习结果仍使用旧 split |
| **同标准化** | slow scaler 仅在 train 上拟合；waveform 用 int16 × scale 反量化后归一化 | 数据与代码都支持，但结果需按数据源解读 |
| **同随机基准** | 主线固定 seed=42；多 seed 仅在 exp06_reproducibility 中运行 | 设计如此，当前多 seed 输出尚未生成 |
| **同指标** | `macro_RMSE / macro_MAE / macro_MRE / per-component R² / sum_error` | 传统 ML 与 DL 汇总已入 `aggregate.py`，但 `exp03_full_training` 尚未纳入 |

## 7. 实验验收标准

### 7.1 G1 验收

- [ ] 所有 ML 与 DL 模型用同一 split、同一 slow scaler、同一 waveform 归一化规则、同一主线 seed=42 运行
- [ ] `outputs/summary/results.tsv` 中的结果已经按统一数据源整理，而不是混合旧 10k 与 `seedpath_formal` 30k
- [ ] R1 完成后，`outputs/summary/results_multiseed.tsv` 含 mean / std / min / max
- [ ] Paired bootstrap CI + Wilcoxon 检验：`outputs/summary/stat_tests.tsv`

### 7.2 G2 验收

- [ ] 三种 DL 融合策略 + 传统 ML 的 `acoustic / optical / thermal / fused` 对照，主线固定 seed=42
- [ ] `outputs/exp03_fusion/` 结果入 `outputs/summary/results.tsv`，含 macro_RMSE + 参数量 + 训练时长
- [ ] 出图：`outputs/summary/fig_fusion_strategy.png`

### 7.3 G3 验收

- [ ] Track A：至少 9 域 hold-out，每域每模型固定 seed=42
- [ ] Track A 结果含 domain_gap，入 `outputs/summary/results.tsv`
- [ ] Track B：σ_train × σ_test 完整矩阵
- [ ] Track B 输出 degradation_ratio 曲线表
- [ ] Track C 可选；若执行，输出 sample_efficiency 表与曲线图

### 7.4 数据集验收（P0 先决条件）

- [ ] 双通道数据文件齐全：`ultrasonic_*`、`fiber_mic_*`、`slow.npy`、`y.npy`
- [ ] 通道 1 方向性通过：L_m ↑ → peak_index ↑，H2 ↑ → peak_index ↓，CO2 ↑ → 峰幅度 ↓
- [ ] 通道 2 方向性通过：CO2 / H2O / H2 ↑ → tau ↓，L_m ↑ → T_round ↑
- [ ] `quality/waveform_quality_summary.json` 中关键项为 passed

## 8. 实验运行流程

### 8.1 完整流程（按顺序执行）

```powershell
cd V3_正式实验/

# 0. 生成并验收 V3.1 双通道数据集（如果未生成）
python src/sim/scripts/generate_waveform_dataset.py
python src/sim/scripts/check_waveform_directionality.py

# 1. 传统 ML 基线（✅ 已完成）
powershell -File experiments\exp01_traditional.ps1

# 2. DL 端到端基线（当前已有多组 run）
powershell -File experiments\exp02_deep_e2e.ps1

# 3. DL 全量训练（脚本在项目根目录）
.\run_all_training.ps1

# 4. 环境适应性评估（G3a敏感性 + G3b噪声鲁棒 + G3c工况泛化）
powershell -File experiments\exp04_adaptation.ps1

# 5. 模态与动态融合对比（⏳ 待启动）
powershell -File experiments\exp03_fusion_grid.ps1

# 6. 多 seed 重复性检测（脚本已备，当前未见输出目录）
powershell -File experiments\exp06_reproducibility.ps1

# 7. 查看状态（任意时刻）
python src/pipeline/status.py

# 8. 汇总结果（全部实验完成后）
python src/pipeline/aggregate.py
```

### 8.2 传统 ML 单独运行

```powershell
# 特征表生成（新数据集，当前中间产物落在 `outputs/exp01_traditional_seedpath/`）
python src\pipeline\feature_extraction.py \
  --source-dir data\waveform_v3_seedpath_formal \
  --output-dir outputs\exp01_traditional_seedpath

# 训练（paper_core 配置，带 CLI 进度）
python src\pipeline\train_traditional.py \
  --data-dir outputs\exp01_traditional_seedpath \
  --output-root outputs\exp01_traditional_seedpath \
  --tag formal_seedpath \
  --seed 42 \
  --split-dir data\waveform_v3_seedpath_formal\splits \
  --profiles v3_raw_no_env v3_raw_tph \
  --combo-list svr_ridge pls_ridge xgboost_ridge \
  --max-workers 4

# 关闭 CLI 进度界面
python src\pipeline\train_traditional.py ... --no-ui
```

### 8.3 深度学习单独运行

```powershell
# 单模型训练（带 CLI 进度；注意当前 YAML 默认仍指向 `data/waveform_v3/`）
python src\pipeline\train_deep.py \
  --config configs\deep\slow_only_tcn_formal.yaml \
  --epochs 200

# 关闭 CLI 进度界面
python src\pipeline\train_deep.py \
  --config configs\deep\slow_only_tcn_formal.yaml \
  --epochs 200 \
  --no-ui

# 多 seed 训练（覆盖配置中的 seed）
python src\pipeline\train_deep.py \
  --config configs\deep\slow_only_tcn_formal.yaml \
  --epochs 200 \
  --seed 52 \
  --output-root outputs\exp06_reproducibility\deep

# 训练曲线导出（批量扫描 outputs）
python src\pipeline\plot_deep_training_curves.py

# 训练曲线导出（只画单个 run）
python src\pipeline\plot_deep_training_curves.py \
  --root outputs\exp02_deep_e2e\v3_tcn_multimodal_seed42 \
  --output-dir outputs\deep_training_curves\v3_tcn_multimodal_seed42
```

### 8.4 环境敏感度分析

```powershell
# G3a 敏感度曲线分析（用已训练模型）
python src\pipeline\sensitivity_scan.py \
  --predictions outputs\exp01_traditional\runs\...\xgboost_ridge\predictions.csv \
  --condition-grid outputs\exp01_traditional\data\condition_grid_v1.csv \
  --output-dir outputs\exp04_adaptation\G3a_sensitivity
```

## 9. 控制与反馈机制

### 9.1 单一状态真源

**文件**：`outputs/STATUS.tsv`

**格式**：每实验每 seed 一行

| 字段 | 说明 | 可选值 |
|------|------|--------|
| exp_id | 实验编号 | exp01, exp02, exp03, ... |
| model | 模型名称 | xgboost_ridge, tcn_multimodal, ... |
| seed | 随机种子 | 42, 52, 62, ... |
| status | 状态 | running, success, failed |
| started | 开始时间 | ISO 8601 |
| finished | 结束时间 | ISO 8601 |
| macro_RMSE | 测试集 macro RMSE | 浮点数 |
| notes | 备注 | 文本 |

**当前说明**：
- 状态表结构有效
- 当前仓库里的 `STATUS.tsv` 内容偏少，不能单独代表完整进度
- 查看状态：`python src/pipeline/status.py`

### 9.2 每实验一行 stdout

**格式**：
```
# 成功
[exp_id.model.seedN] OK macro_RMSE=X.XXX took=Ys -> outputs/...

# 失败
[exp_id.model.seedN] FAIL reason=... -> log path
```

**示例**：
```
[exp01.xgboost_ridge.seed42] OK macro_RMSE=0.631 took=156s -> outputs/exp01_traditional/runs/...
[exp02.tcn_multimodal.seed42] FAIL reason=OOM -> outputs/exp02_deep_e2e/v3_tcn_multimodal_seed42/train.log
```

### 9.3 实验结果统一表

**文件**：`outputs/summary/results.tsv`

**内容**：由 `aggregate.py` 重新扫描结果目录后生成，不是手工 append

**字段**：
- `exp_id`, `result_family`, `result_group`, `source_file`
- 传统 ML 的 `macro_RMSE_pp`、深度学习的 `macro_RMSE`
- 各组分 RMSE / MAE / MAPE / R²
- `mean_abs_sum_error`、`max_abs_sum_error`

**生成方式**：
```powershell
python src/pipeline/aggregate.py
```

### 9.4 训练命令行可视化

**功能**：
- 传统训练：显示 `profile / combo / 阶段 / 已完成数 / 累计耗时 / 最近 macro_RMSE`
- 深度训练：显示 `run_name / epoch / train_loss / val_loss / val_macro_RMSE / early stopping 状态`

**启用方式**：
- 默认在交互式终端自动开启
- 强制开启：`--ui`
- 关闭：`--no-ui`

**兼容性**：
- Windows 下若宿主终端不支持 ANSI/VT，`--ui` 会自动退回普通日志
- 非交互环境自动退回普通日志

## 10. 时间预算

| Phase | 任务 | 估时 | 状态 |
|-------|------|------|------|
| P0 第二步 | 双通道仿真代码改造 + 数据生成 | 已完成 | ✅ |
| P1 | 传统 ML 基线（历史 + seedpath 结果）| 已有产物 | ✅ |
| P2 | DL 端到端基线 | 已有多组 run，但未统一到 `seedpath_formal` | 🟡 |
| P3 | 模态与动态融合对比 | 深度学习 Late fusion 仍未独立落地 | ⏳ |
| P4 | Track A 留一域（9 域 × 多模型）| 当前仅 G3a 敏感度已有输出 | 🟡 |
| P5 | Track B 环境扰动 | 脚本存在，输出待补齐 | ⏳ |
| P6 | R1 多 seed 重复性检测（复用 seed42，补跑 52/62）| 脚本存在，输出待补齐 | ⏳ |
| P7 | Track C 跨域微调（可选）| 1-2 天 + GPU | ⏳ 可选 |
| P8 | 结果汇总 + 论文图 | 1 天 | ⏳ 未启动 |

**合计**：9-14 天 + 多批 GPU 时长（Track C 可选不计）

## 11. 论文图清单（P8 产出）

| 图号 | 主题 | 来源数据 | 图表类型 |
|------|------|----------|----------|
| **F1** | ML vs DL 主对比 | summary/results.tsv | 条形图 |
| **F2** | per-component R² | summary/results.tsv | 雷达图 |
| **F3** | 模态与动态融合对比 | summary/results.tsv（exp03 子集）| 条形图 |
| **F4** | 留一域 domain_gap | summary/results.tsv（exp04 子集）| 热图 |
| **F5** | 环境扰动退化率曲线 | summary/results.tsv（exp05 子集）| 折线图 |
| **F6** | 多 seed 重复性误差棒 | summary/results_multiseed.tsv（exp06 子集）| 误差棒图 |
| **F7** | 跨域微调 sample efficiency（可选）| summary/results.tsv（exp07 子集）| 折线图 |

## 12. 关键约束与风险

| 风险 | 影响 | 应对策略 | 状态 |
|------|------|----------|------|
| 双通道链路物理参数未校准 | 不能声称等价真实采集 | 数据卡片标注 `calibration: pending` | 已标注 |
| 数据包体积大（~21.6 GB）| 生成、训练、存储成本高 | 必要时缩短 fiber_mic 窗口至 5 ms | 可选优化 |
| MRE 在低含量组分上不稳定 | CO2/N2 接近 0 时 MRE 爆炸 | 采用 SMAPE 与限定子集 MRE 双报 | 已实现 |
| 留一域样本失衡 | 极端工况域样本可能不足 1000 | 先看分布，合并样本 < 500 的稀疏域 | 待检查 |
| GPU 时长未估算 | 可能阻塞主线 | 主线先固定 seed=42 出结果 | 执行中 |
| RBF SVR 全量训练成本过高 | 单次训练可能超过 1 小时 | 不放入正式主网格 | 已排除 |

## 13. 参考文档

| 文档 | 内容 |
|------|------|
| `docs/design.md` | 实验设计详细说明（三目标 × 矩阵 × 验收）|
| `docs/训练速度优化计划.md` | ML + DL 训练加速路线图与 smoke benchmark |
| `docs/V3 Waveform 多噪声种子 + 序列内多 L_m_PLAN.md` | 数据生成计划 |
| `docs/waveform_v3_seedpath_formal_适配说明.md` | 新数据集适配说明 |
| `README.md` | 项目总体说明 |
