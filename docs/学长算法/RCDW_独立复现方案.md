# RCDW-MGDA 独立复现方案

> 严格按照 [学长算法框架.md](学长算法框架.md) 中给出的 **O₂ / CO₂ / N₂** 三组分检测算法做一次最小可运行 PyTorch 复现。**不与 gas-dl-v2 主项目对齐**——独立目录、独立数据接口、独立训练入口、独立测试。
>
> 复现目标：跑通 NDIR + 热导 + 超声 三模态在 O₂ / CO₂ / N₂ 上的动态加权融合，并复现框架第十五节中扰动实验（光学衰减 / 光学散射 / 热导扰动）的趋势曲线。

> **实施状态（2026-06-29）**：完整实施手册见 [RCDW_实施指南.md](RCDW_实施指南.md)；落地代码位于 [`rcdw_mgda/`](../../rcdw_mgda/)；完成情况见 [RCDW_实施完成情况.md](RCDW_实施完成情况.md)（35 tests pass + 数值对齐 + smoke 训练+ 扰动实验全部通过）。

---

## 一、与学长框架的目标对齐表

严格保留学长框架中的所有约定：

| 框架要点 | 复现要点 |
|---|---|
| 三模态：NDIR、热导(TCD)、超声(US) | 三条单模态前向分支，M=3 |
| 三目标气体：O₂ / CO₂ / N₂ | 输出维度 G=3，体积分数 sum≈1 |
| 输入：`X(t)=[S_ndir, S_tc, S_us, P, T, RH]` | 滑窗输入 `(B, L=8, 6)`，L=8 对齐框架推荐窗口 |
| 单模态反演 → 三气体候选 `Y_m ∈ R^3` | NDIRNet / TCDNet / USNet |
| 13 维扰动感知特征 | `FeatureExtractor` 输出 `(B, M, 13)` |
| 误差预测 → 误差 `Ê_{m,g}` | `ErrorNet`，每气体一个 head |
| 基线锚定 + 自适应 α + maxShift | `RCDWFusion` 中可微实现 |
| 退化硬抑制 (>4×min → 4%) | eval-only hook，非可微 |
| 训练 1400 / 验证 300 / 测试 300 切分 | 合成 2000 样本；切分比 70/15/15 |
| 评价：ARE / MRE / RMSE / MAE | 标准回归指标，按气体分别报告 + 整体 |
| 扰动实验：光学衰减 / 光学散射 / 热导扰动 / 超声异常 / 温度突变 | 测试集注入扰动 × 7 强度，绘制指标 + 权重曲线 |

---

## 二、独立目录结构

复现目录与 gas-dl-v2 主项目**完全隔离**，放在项目根下的 `rcdw_mgda/`：

```
rcdw_mgda/                           # 独立子工程（与 src/、tests/ 同级）
├── README.md
├── pyproject.toml                   # 独立依赖声明（torch, numpy, scipy, matplotlib, pyyaml）
├── configs/
│   └── default.yaml                 # 全部超参 + W_base + 数据切分
├── rcdw/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── synth.py                 # 2000 合成样本（1400/300/300）
│   │   ├── real_loader.py           # 真实标定数据接口（预留）
│   │   └── preprocess.py            # 滤波 / 校准 / 温压湿补偿
│   ├── models/
│   │   ├── __init__.py
│   │   ├── single_modal.py          # NDIRNet / TCDNet / USNet
│   │   ├── feature.py               # 13 维扰动感知特征
│   │   ├── error_net.py             # 误差预测器
│   │   └── rcdw.py                  # RCDWFusion + 整体 RCDW_MGDA wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py                # MSE / 组分约束 / 误差监督
│   │   ├── metrics.py               # MAE / RMSE / MRE / ARE
│   │   ├── stage_a.py               # 阶段 A：单模态预训练
│   │   └── stage_b.py               # 阶段 B：联合训练 ErrorNet + RCDW
│   ├── perturbation/
│   │   ├── __init__.py
│   │   └── inject.py                # 光学衰减 / 散射 / 热导扰动 / 超声异常 / 温度突变注入
│   └── utils/
│       ├── __init__.py
│       ├── normalize.py             # 组分归一化、湿/干基换算
│       └── degradation.py           # 退化硬抑制 hook（eval-only）
├── scripts/
│   ├── train.py                     # 入口：stage_a + stage_b
│   ├── eval.py                      # 评测：MAE/RMSE/MRE/ARE
│   ├── perturb.py                   # 三类扰动 × 7 强度，绘制曲线
│   └── numerical_check.py           # 与框架公式 hand-checked 数值对齐
└── tests/
    ├── test_synth.py
    ├── test_single_modal.py
    ├── test_feature.py
    ├── test_error_net.py
    ├── test_rcdw_fusion.py          # 含数值对齐
    ├── test_degradation.py
    └── test_perturbation.py
```

> **隔离策略**：`rcdw_mgda/` 自带 `pyproject.toml` 与依赖；不 import `src/`；不读 `configs/experiment/`；测试不进入主项目 `pytest` 主线；与 `gas-dl-v2` 共享 venv 但代码完全独立。

---

## 三、数据接口

### 3.1 输入 / 输出（严格按框架第四节）

```python
# 输入（一个样本）
X = [S_ndir, S_tc, S_us, P, T, RH]   # shape: (6,) 或时序 (L, 6)

# 标签
Y = [C_O2, C_CO2, C_N2]              # shape: (3,)，sum≈1

# 时间同步约定：所有传感器已按 (B, 6) 或 (B, L, 6) 对齐
```

### 3.2 合成数据（先跑通）

[rcdw/data/synth.py](../../rcdw_mgda/rcdw/data/synth.py)：

> **滑窗数据生成**：`synth()` 生成 N 个独立样本点后，由 `make_windowed_splits()` 按时序滑窗切分为 `(B, L=8, 6)` 输入张量。每个窗口的标签取窗口最后一个时刻的浓度值（对齐实际在线推理场景）。

```python
import numpy as np

def synth(n: int = 2000, seed: int = 0):
    """生成 n 条样本，对齐框架第三节中各模态的物理近似。

    n=2000 保证验证/测试集各 300 样本，指标有足够统计意义。
    """
    rng = np.random.default_rng(seed)
    # O2 / CO2 / N2 真值，Dirichlet 偏 N2 主导
    C = rng.dirichlet([2, 1, 6], size=n)
    T  = rng.uniform(280, 360, n)         # K
    P  = rng.uniform(0.95, 1.05, n)       # atm
    RH = rng.uniform(0.00, 0.05, n)

    # NDIR：Beer-Lambert 对 CO2 敏感
    S_ndir = (1 - np.exp(-3.0 * C[:, 1])) + 0.01 * rng.standard_normal(n)

    # 超声：v = sqrt(γRT / M_mix)；M_O2=32, M_CO2=44, M_N2=28
    M_mix = 32 * C[:, 0] + 44 * C[:, 1] + 28 * C[:, 2]
    S_us  = np.sqrt(1.4 * 8.314 * T / (M_mix * 1e-3)) + 0.5 * rng.standard_normal(n)

    # 热导：λ_mix ≈ Σ x_i λ_i（O2 ~0.026, CO2 ~0.017, N2 ~0.026 W/m·K @ 300K）
    S_tc  = 0.026 * C[:, 0] + 0.017 * C[:, 1] + 0.026 * C[:, 2] \
            + 1e-4 * rng.standard_normal(n)

    X = np.stack([S_ndir, S_tc, S_us, P, T, RH], axis=1).astype(np.float32)
    return X, C.astype(np.float32)


def make_splits(n_train: int = 1400, n_val: int = 300, n_test: int = 300, seed: int = 0):
    X, Y = synth(n_train + n_val + n_test, seed=seed)
    s1 = n_train
    s2 = n_train + n_val
    return {
        "train": (X[:s1], Y[:s1]),
        "val":   (X[s1:s2], Y[s1:s2]),
        "test":  (X[s2:],  Y[s2:]),
    }
```

### 3.3 预处理（框架第五节 §1–2）

[rcdw/data/preprocess.py](../../rcdw_mgda/rcdw/data/preprocess.py)：

- 时间戳对齐（合成数据已对齐，留接口）；
- 滑动窗口均值 / 中值滤波（`L=8` 默认）；
- 零点 + 跨度校准（线性 `y = a·x + b`，参数从配置注入）；
- 温压湿补偿：NDIR / TCD / US 各一个补偿函数（多项式形式，参数标定后填入）。

---

## 四、单模态反演网络（框架第五节 §3）

### 4.1 网络结构

每个模态独立估计三气体浓度（**用 clamp + normalize 代替 softmax**，避免 softmax 过度压缩导致模型对低浓度组分表达能力不足；组分约束仅在此处施加一次，融合层不再重复归一化）：

```python
class SingleModal(nn.Module):
    """模态信号 + 环境 → 三气体候选浓度。

    使用 clamp(0,1) + L1-normalize 代替 softmax：
    - softmax 的指数映射会把 logit 差异放大，低浓度组分（如 CO₂ ~0.1）
      容易被挤压到接近 0，反演精度差；
    - clamp + normalize 保持线性比例关系，低浓度组分更易拟合。
    """

    def __init__(self, in_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).clamp(min=0)
        return raw / (raw.sum(dim=-1, keepdim=True) + 1e-6)


class NDIRNet(SingleModal): ...      # 输入: [S_ndir, P, T, RH]
class TCDNet(SingleModal):  ...      # 输入: [S_tc,   P, T, RH]
class USNet(SingleModal):   ...      # 输入: [S_us,   P, T, RH]
```

### 4.2 训练策略（与框架第九节呼应）

- **NDIR**：CO₂ 主输出。训练时对 O₂/N₂ 输出 loss 权重 0.1，避免拟合无信息项；
- **TCD / US**：联合估计三气体，loss 权重均匀；
- 阶段 A 单独训练，输出 3 个 checkpoint。

---

## 五、扰动感知特征（框架第六节，13 维）

[rcdw/models/feature.py](../../rcdw_mgda/rcdw/models/feature.py)：

按框架第六节表格，对每个模态 `m` 和每个目标气体 `g` 抽特征，最小集合 13 维：

| # | 符号 | 定义 |
|--:|---|---|
| 1 | `CV_m` | `std/mean` 滑窗变异系数 |
| 2 | `D_{m,g}` | `|Ŷ_{m,g} − median_j(Ŷ_{j,g})|` 群体中位偏离 |
| 3 | `G_m` | `(S_k − S_{k-1})²` 一阶差分能量 |
| 4 | `Q_m` | `q_m / Σ_j q_j` 信号质量比 |
| 5 | `B_{m,g}` | `(1/(M-1)) Σ_{j≠m} |Ŷ_{m,g} − Ŷ_{j,g}|` 群体偏差 |
| 6 | `ΔT` | `|T_k − T_{k-1}|` |
| 7 | `ΔP` | `|P_k − P_{k-1}|` |
| 8 | `ΔRH` | `|RH_k − RH_{k-1}|` |
| 9 | `|Y_m − mean(Y)|` | 模态浓度对组均值偏差 |
| 10 | `mod_residual` | `|Ŷ_{m,g} − Σ_j Ŷ_{j,g}/M|` 组内残差 |
| 11 | `snr_proxy` | `|μ| / σ` 滑窗信噪比代理 |
| 12 | `drift` | 滑窗线性拟合斜率 |
| 13 | `dt` | 距上次采样的时间差，固定采样率下设常数 |

```python
class FeatureExtractor(nn.Module):
    """生成 (B, M=3, F=13) 的扰动感知特征。

    始终使用滑窗模式（L=8），保证 CV/G/ΔT/ΔP/ΔRH/drift/snr_proxy 等
    时序统计特征有效。若单步退化（7/13 维为 0），ErrorNet 无法学到
    有意义的模态可靠性判别信号。
    """

    def __init__(self, window: int = 8):
        super().__init__()
        self.L = window

    def forward(self, x, Y_modal, ctx=None):
        # x:       (B, L, 6)  — 始终为滑窗输入
        # Y_modal: (B, M=3, G=3)
        # 返回:    (B, M, 13)
        ...
```

**实现要点**：
- 输入始终为 `(B, L, 6)`，L=8；单模态网络取窗口末尾 `x[:, -1, :]` 做反演，特征提取用完整窗口；
- 滑窗统计用 `torch.unfold` 或缓存 buffer，**不破坏梯度图**；
- 群体偏差 `B`、中位偏离 `D` 跨模态计算，需在 `Y_modal` 维度上 reduce。

---

## 六、误差预测器 ErrorNet（框架第七节）

```python
class ErrorNet(nn.Module):
    """为每个目标气体训一个 head，输入 13 维特征，输出该模态对该气体的预测误差。"""

    def __init__(self, in_dim: int = 13, n_gas: int = 3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 32), nn.GELU(),
                nn.Linear(32, 1), nn.Softplus(),     # 保证 Ê ≥ 0
            )
            for _ in range(n_gas)
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, M=3, F=13)  →  E: (B, M, G=3)
        outs = [h(feat).squeeze(-1) for h in self.heads]   # 每个: (B, M)
        return torch.stack(outs, dim=-1)
```

**训练标签**：`E_true_{m,g} = |Ŷ_{m,g} − C_ref_g|`，标定阶段直接生成。

> **detach 策略**：阶段 B 默认**冻结单模态权重**（`requires_grad=False`），此时 `Ŷ` 自动不带梯度，无需显式 detach。若后续实验需要 `lr×0.1` 微调单模态，则**必须**在计算误差标签时 `|Ŷ.detach() − C_ref|`，防止误差监督 loss 的梯度回传到单模态网络导致训练目标冲突（单模态想减小 `|Ŷ−C_ref|`，误差监督想让 ErrorNet 跟踪 `|Ŷ−C_ref|` 的变化）。

---

## 七、RCDW 融合层（框架第八–十节，可微版）

[rcdw/models/rcdw.py](../../rcdw_mgda/rcdw/models/rcdw.py)：

```python
class RCDWFusion(nn.Module):
    """可靠性约束动态加权融合层。

    输入:
      Y_modal: (B, M=3, G=3)   单模态候选浓度
      E_pred : (B, M, G)       预测误差（恒正）
      W_base : (M, G)          气体相关基线权重

    输出:
      C_fused: (B, G)
      W_final: (B, M, G)
    """

    def __init__(self, W_base: torch.Tensor, *,
                 beta: float = 8.0,
                 alpha_min: float = 0.1, alpha_max: float = 0.9, tau_a: float = 0.05,
                 s_min: float = 0.05,   s_max: float = 0.40, tau_s: float = 0.05):
        super().__init__()
        assert W_base.shape == (3, 3), "expect (M=3, G=3) baseline weights"
        self.register_buffer("W_base", W_base)
        self.beta = beta
        self.a_min, self.a_max, self.tau_a = alpha_min, alpha_max, tau_a
        self.s_min, self.s_max, self.tau_s = s_min,   s_max,   tau_s

    def forward(self, Y_modal: torch.Tensor, E_pred: torch.Tensor):
        eps = 1e-6

        # 1. softmax 形式的 Wmix （框架第八节式 §2 第二种）
        Wmix = torch.softmax(-self.beta * E_pred, dim=1)             # 在模态维归一

        # 2. 自适应 α / shift（框架第十节）
        dE = E_pred.max(dim=1).values - E_pred.min(dim=1).values     # (B, G)
        alpha = self.a_min + (self.a_max - self.a_min) * dE / (dE + self.tau_a)
        shift = self.s_min + (self.s_max - self.s_min) * dE / (dE + self.tau_s)
        alpha = alpha.unsqueeze(1)                                   # (B, 1, G)
        shift = shift.unsqueeze(1)

        # 3. 基线锚定 + maxShift clamp
        W = (1 - alpha) * self.W_base + alpha * Wmix
        W = torch.clamp(W, self.W_base - shift, self.W_base + shift)
        W = W / (W.sum(dim=1, keepdim=True) + eps)                   # 重归一

        # 4. 融合输出
        C_fused = (W * Y_modal).sum(dim=1)                           # (B, G)
        return C_fused, W
```

**退化硬抑制**：训练期不启用（保持可微），仅在 `scripts/eval.py` 与 `scripts/perturb.py` 中按窗口中位误差触发，落点 [rcdw/utils/degradation.py](../../rcdw_mgda/rcdw/utils/degradation.py)：

```python
def hard_suppress(W, E_history, *, ratio: float = 4.0, cap: float = 0.04):
    """框架第十一节：median(E_m) > ratio·min(E_med) → w_m ≤ cap，然后重归一。"""
    E_med = E_history.median(dim=0).values        # (M, G)
    min_E = E_med.min(dim=0, keepdim=True).values # (1, G)
    degraded = E_med > ratio * min_E              # (M, G) bool
    W = torch.where(degraded.unsqueeze(0), W.clamp_max(cap), W)
    W = W / W.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return W, degraded
```

---

## 八、基线权重 `W_base`（框架第九节推荐表）

直接采用框架第九节表：

| 目标气体 | NDIR | TCD | US |
|---|---:|---:|---:|
| O₂ | 0.05 | 0.50 | 0.45 |
| CO₂ | 0.70 | 0.15 | 0.15 |
| N₂ | 0.05 | 0.45 | 0.50 |

PyTorch 张量形态 — **行：模态 (M)，列：气体 (G)**，是框架表格的转置：

> **维度约定**：`W_base[m, g]` = 模态 m 对气体 g 的基线权重。`RCDWFusion` 中 `softmax(dim=1)` 在模态维度(M)归一化，即对每种气体 g，各模态权重 sum=1。此维度必须与 `Y_modal` 的 dim=1（模态维）对齐。

```python
#                    O₂    CO₂   N₂     ← 列 = 气体 (G)
W_base = torch.tensor([
    [0.05, 0.70, 0.05],   # ← NDIR  (m=0)
    [0.50, 0.15, 0.45],   # ← TCD   (m=1)
    [0.45, 0.15, 0.50],   # ← US    (m=2)
], dtype=torch.float32)       # shape (M=3, G=3)
# 验证：每列 sum=1 → O₂: 0.05+0.50+0.45=1.0 ✓  CO₂: 0.70+0.15+0.15=1.0 ✓  N₂: 0.05+0.45+0.50=1.0 ✓
```

---

## 九、整体前向（框架第十三节伪代码 → PyTorch）

```python
class RCDW_MGDA(nn.Module):
    def __init__(self, W_base: torch.Tensor):
        super().__init__()
        self.ndir = NDIRNet(in_dim=4)
        self.tcd  = TCDNet(in_dim=4)
        self.usn  = USNet(in_dim=4)
        self.feat = FeatureExtractor(window=8)
        self.err  = ErrorNet(in_dim=13, n_gas=3)
        self.fuse = RCDWFusion(W_base)

    def forward(self, x: torch.Tensor) -> dict:
        # x: (B, L, 6) = 滑窗输入 [S_ndir, S_tc, S_us, P, T, RH]
        # 单模态反演取窗口最后时刻
        x_last = x[:, -1, :]                                     # (B, 6)
        S_nd, S_tc, S_us, P, T, RH = x_last.unbind(-1)
        env = torch.stack([P, T, RH], dim=-1)

        y_nd = self.ndir(torch.cat([S_nd[:, None], env], dim=-1))
        y_tc = self.tcd (torch.cat([S_tc[:, None], env], dim=-1))
        y_us = self.usn (torch.cat([S_us[:, None], env], dim=-1))
        Y    = torch.stack([y_nd, y_tc, y_us], dim=1)         # (B, M, G)

        feat = self.feat(x, Y)                                # (B, M, F=13)
        E    = self.err(feat)                                 # (B, M, G)
        C, W = self.fuse(Y, E)

        return {"C": C, "Y_modal": Y, "E_pred": E, "W": W}
```

---

## 十、训练流程（框架"先标定再融合"）

### 10.1 阶段 A — 单模态预训练（[scripts/stage_a.py](../../rcdw_mgda/scripts/train.py)）

```
L_single = Σ_m λ_m · MSE(Y_m, C_ref)
```

- NDIR：对 O₂/N₂ 输出 loss 权重 0.1，对 CO₂ 权重 1.0；
- TCD / US：三气体权重均匀；
- Optimizer = AdamW，lr = 1e-3，cosine 退火；
- batch = 16，epoch = 200，验证集早停。

输出 3 个 checkpoint：`runs/stage_a/{ndir,tcd,us}.pt`。

### 10.2 阶段 B — 联合训练 ErrorNet + RCDW

```
L = MSE(C_fused, C_ref)
  + λ_e · MSE(E_pred, |Y_modal - C_ref|)              # 监督误差预测
  + λ_s · | Σ_g C_fused_g − 1 |                       # 组分约束
```

- **默认冻结单模态权重**（`requires_grad=False`），ErrorNet + RCDW 独立优化，`Ŷ` 无梯度因此误差标签无需 detach；
- 若切换为微调模式（`lr × 0.1`），误差标签必须写为 `|Ŷ.detach() − C_ref|`；
- `λ_e = 1.0`, `λ_s = 0.1`；
- 其它超参同阶段 A；
- 输出：`runs/stage_b/rcdw.pt`。

### 10.3 入口脚本

```bash
# 一键两阶段
python -m scripts.train --config configs/default.yaml

# 单独评测
python -m scripts.eval  --ckpt runs/stage_b/rcdw.pt --split test

# 扰动实验
python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt
```

---

## 十一、评价指标（框架第十五节）

[rcdw/training/metrics.py](../../rcdw_mgda/rcdw/training/metrics.py)：

```python
def metrics(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> dict:
    """按气体分别计算，外加 overall 平均。"""
    e = (pred - ref).abs()
    return {
        "MAE":  e.mean().item(),
        "RMSE": ((pred - ref) ** 2).mean().sqrt().item(),
        "MRE":  (e / (ref.abs() + eps)).mean().item() * 100,    # %
        "ARE":  (e / (ref.abs() + eps)).max().item() * 100,     # %（最大相对误差）
    }
```

- 按气体（O₂ / CO₂ / N₂）分别报告；
- Overall 平均（三气体均值）；
- 与单模态最佳结果对比，记录提升幅度。

---

## 十二、扰动实验（框架第十五节 + 原稿第十节）

复现五组：**光学衰减**、**光学散射**、**热导扰动**、**超声异常**、**温度突变**，扫描强度 `level ∈ {0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11}`。

[rcdw/perturbation/inject.py](../../rcdw_mgda/rcdw/perturbation/inject.py)：

```python
def inject(x: torch.Tensor, kind: str, level: float, rng=None) -> torch.Tensor:
    x = x.clone()
    if kind == "optical_atten":            # NDIR 通道衰减（光源老化 / 光路污染）
        x[..., 0] *= (1 - level)
    elif kind == "optical_scat":           # NDIR 通道加性高斯噪声（散射干扰）
        x[..., 0] += level * torch.randn_like(x[..., 0])
    elif kind == "thermal":                # 热导通道乘性扰动（温控漂移）
        x[..., 1] *= (1 + level * torch.randn_like(x[..., 1]))
    elif kind == "ultrasonic":             # 超声通道加性噪声（换能器老化 / 耦合不良）
        x[..., 2] += level * torch.randn_like(x[..., 2]) * x[..., 2].abs().mean()
    elif kind == "temperature":            # 温度突变（T 通道阶跃偏移）
        x[..., 4] += level * 80           # 最大偏移 ~8.8K @ level=0.11
    else:
        raise ValueError(f"unknown perturbation kind: {kind}")
    return x
```

[scripts/perturb.py](../../rcdw_mgda/scripts/perturb.py) 输出两张图：

1. **指标曲线**：ARE / MRE / RMSE 随扰动强度变化，每气体一图；
2. **权重曲线**：`W[ndir, :]`、`W[tcd, :]`、`W[us, :]` 在 CO₂ 上的占比随 level 变化。

**预期趋势**（对齐框架结论）：
- 光学扰动下，NDIR 权重单调下降，TCD / US 补偿上升；
- 热导扰动下，TCD 权重下降，NDIR / US 补偿上升；
- 超声异常下，US 权重下降，NDIR / TCD 补偿上升；
- 温度突变下，所有模态误差上升，但超声（v ∝ √T）和热导（λ 温度依赖）受影响最大，NDIR 相对稳定；
- 退化硬抑制在 level ≥ 0.09 时触发，把异常模态压到 4%。

---

## 十三、组分归一化（框架第十二节）

[rcdw/utils/normalize.py](../../rcdw_mgda/rcdw/utils/normalize.py)：

```python
def normalize_composition(C: torch.Tensor, RH: torch.Tensor | None = None,
                          *, basis: str = "dry") -> torch.Tensor:
    """对 O2/CO2/N2 归一化到 sum=C_total。

    basis="dry" → C_total = 100%（湿度已剔除）
    basis="wet" → C_total = 100% − C_H2O（湿度由 RH 换算）
    """
    eps = 1e-6
    if basis == "wet" and RH is not None:
        C_total = 1.0 - rh_to_water_vol(RH)
    else:
        C_total = 1.0
    return C / (C.sum(dim=-1, keepdim=True) + eps) * C_total
```

> 默认 `basis="dry"`，与合成数据约定一致。

---

## 十四、数值对齐脚本（关键）

[scripts/numerical_check.py](../../rcdw_mgda/scripts/numerical_check.py)：与框架第十、八节公式 hand-checked 数值对齐，确保 PyTorch 实现与 MATLAB 版偏差 < 1e-5。

测试矩阵：
- 输入：`Y_modal` 随机张量（固定 seed）、`E_pred` 已知张量；
- 期望输出：手算的 `Wmix`、`alpha`、`shift`、`W_final`、`C_fused`；
- 通过条件：所有维度 max abs diff < 1e-5。

---

## 十五、测试矩阵（独立 pytest）

`rcdw_mgda/tests/`，**不进入主项目** `pytest` 主线：

| 测试文件 | 用例数 | 覆盖 |
|---|--:|---|
| `test_synth.py` | 4 | 合成数据维度、sum≈1、splits 不重叠、滑窗 shape (B,L,6) |
| `test_single_modal.py` | 4 | clamp+normalize 输出 sum≈1、维度、CO₂ NDIR 收敛 |
| `test_feature.py` | 6 | 13 维特征维度、滑窗统计特征非零、群体偏差正确性 |
| `test_error_net.py` | 3 | 输出恒正、维度、多 head 独立 |
| `test_rcdw_fusion.py` | 8 | 数值对齐（hand-checked）；α/shift 边界；权重 sum=1；可微性 |
| `test_degradation.py` | 4 | 硬抑制触发条件、cap=0.04、重归一 |
| `test_perturbation.py` | 6 | 五类扰动注入维度；权重曲线 NDIR 单调下降（光学扰动）；US 单调下降（超声异常） |

入口：

```bash
cd rcdw_mgda && python -m pytest
```

---

## 十六、与框架的差异说明

仅做以下工程化调整，**不改变算法逻辑**：

1. **softmax 形式的 Wmix**：选用框架第八节式 §2 的第二种 `Wmix ∝ exp(-β·E)`，避免 `1/(E+ε)` 的数值不稳定；
2. **始终滑窗 L=8**：框架推荐滑动窗口 `L≈8`，本复现直接以 `(B, L=8, 6)` 作为输入，单模态反演取窗口末尾，特征提取用完整窗口。不支持单步模式（单步下 13 维特征有 7 维退化为 0，ErrorNet 无法有效学习）；
3. **clamp + normalize 代替 softmax**：单模态网络末端用 `clamp(min=0)` + L1 归一化代替 softmax，避免指数映射对低浓度组分的过度压缩。组分约束仅施加一次，融合层不重复归一化；最终输出经 `normalize_composition` 做湿/干基换算；
4. **退化硬抑制**：训练期不可微，仅在 eval/perturb 启用，与框架第十一节判据完全一致；
5. **数据量 2000 样本**：合成数据从框架建议的 126 提升到 2000（1400/300/300），保证验证/测试集统计显著性；
6. **扰动实验 5 类**：在框架 3 类（光学衰减/散射/热导）基础上补充超声异常和温度突变，覆盖框架第十五节中更多验证场景；
7. **W_base 布局 (M, G)**：框架表格行=气体/列=模态，代码张量行=模态/列=气体（转置），与 `Y_modal (B, M, G)` 的 dim=1 对齐，数值不变；
8. **ErrorNet detach 策略**：阶段 B 默认冻结单模态权重，`Ŷ` 自动无梯度；若切换到微调模式则必须显式 detach 误差标签，防止梯度回传冲突；
9. **基线权重维度**：严格 3×3（NDIR/TCD/US × O₂/CO₂/N₂），如后续加电化学 O₂ 或顺磁 O₂ 模态，需把 `W_base` 扩为 4×3，重新设计 O₂ 行。

---

## 十七、里程碑

| 阶段 | 产出 | 验证标准 | 估时 |
|---|---|---|---|
| M0 仓库骨架 | `rcdw_mgda/` 全目录 + `pyproject.toml` + `configs/default.yaml` | `python -c "import rcdw"` 成功 | 0.5 天 |
| M1 合成数据 + 单模态 | `synth.py`（2000 样本 + 滑窗）+ `single_modal.py` + `stage_a.py` 收敛 | 单模态 MAE < 0.05（每气体） | 1 天 |
| M2 RCDW 数值对齐 | `rcdw.py` + `numerical_check.py` 通过 | 与手算差 < 1e-5 | 1 天 |
| M3 联合训练 | `error_net.py` + `feature.py`（滑窗 13 维）+ `stage_b.py` | 验证集 RMSE 优于最佳单模态 ≥ 10% | 1.5 天 |
| M4 扰动实验 | `perturb.py` 五类扰动 × 7 强度 | NDIR 权重在光学扰动下单调下降；US 权重在超声异常下单调下降 | 1 天 |
| M5 退化硬抑制 + 归一化 | `degradation.py` + `normalize.py` eval-only | level ≥ 0.09 时触发硬抑制 | 0.5 天 |
| M6 真实数据接入 | `real_loader.py` 替换 `synth.py`，复跑 M3–M4 | 真实标定集上指标稳定 | 1–2 天 |

---

## 十八、下一步动作清单

- [ ] 建立 `rcdw_mgda/` 目录与 `pyproject.toml`、`configs/default.yaml`
- [ ] 实现 `data/synth.py`（2000 样本 + 滑窗切分）并跑通 `stage_a.py`，确认单模态 MSE 收敛
- [ ] 实现 `models/rcdw.py`，跑通 `scripts/numerical_check.py`（误差 < 1e-5）
- [ ] 实现 `models/feature.py`（始终滑窗模式）+ `models/error_net.py`，完成阶段 B 联合训练
- [ ] 实现 `scripts/perturb.py`，复现五类扰动趋势曲线（光学衰减/散射/热导/超声异常/温度突变）
- [ ] 实现 `utils/degradation.py` 与 `utils/normalize.py` 的 eval-only hook
- [ ] 替换为真实标定数据，复跑 M3–M4，整理到学位论文 §4 章
