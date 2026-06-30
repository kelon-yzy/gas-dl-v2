# RCDW-MGDA 逐步实施指南

> 本文档是 [RCDW_独立复现方案.md](RCDW_独立复现方案.md) 的完整代码实施手册。
> 目标读者：**需要逐文件照抄的 AI 或初学者**。每个文件给出完整代码，不省略，不留 `...`。
> 所有路径相对于项目根 `gas-dl-v2/`。

> **实施状态（2026-06-29）**：本指南已被完整实施，端到端 smoke 测试通过。详见 [RCDW_实施完成情况.md](RCDW_实施完成情况.md)。
> 实施时对本指南中**测试代码**的 5 处修正记录在该完成情况文档第 3 节，核心实现完全遵循本指南。

> ⚠️ **数据集形态已升级（2026-06-30 v1.2）**：本指南描述 M0-M5 的 toy 阶段（`synth.py` 6 维玩具数据）。
> 数据生成管线已重构为 **benchmark 形态**（`scripts.generate_benchmark` + 12 维通道布局 + HITRAN 物理建模），
> 实施完成情况见 [RCDW_数据集主线对齐改动方案.md](RCDW_数据集主线对齐改动方案.md) v1.2 +
> [RCDW_数据集主线对齐_完成情况.md](RCDW_数据集主线对齐_完成情况.md)。
> 本指南 M1 §1.1 中的 `synth.py` 已被删除替换为 `BenchmarkDataset`，但 M1–M5 的模型/训练/扰动框架结构整体保留。
> 新流程命令见完成情况 §7.4。

---

## 全局规则（实施前必读）

1. **所有代码写在 `rcdw_mgda/` 目录下**，与主项目 `src/` 完全隔离。
2. **不得 import `src/` 中的任何模块**。
3. **不得修改主项目的任何文件**（`src/`、`tests/`、`configs/`）。
4. **张量维度约定**：
   - 输入 `x`: `(B, L=8, 6)` — B=batch, L=窗口长度, 6=传感器通道
   - 通道顺序: `[S_ndir, S_tc, S_us, P, T, RH]` → 索引 `[0, 1, 2, 3, 4, 5]`
   - 模态顺序: `NDIR=0, TCD=1, US=2`
   - 气体顺序: `O₂=0, CO₂=1, N₂=2`
   - `Y_modal`: `(B, M=3, G=3)` — M=模态, G=气体
   - `W_base`: `(M=3, G=3)` — 行=模态, 列=气体（**框架表格的转置**）
5. **softmax 归一方向**: `dim=1`（模态维），即对每种气体 g，各模态权重 sum=1。
6. **测试运行命令**: `cd rcdw_mgda && python -m pytest`，不进入主项目 pytest。
7. **Python 版本**: 3.10–3.13，与主项目一致。

---

## M0 — 仓库骨架

### Step 0.1: 创建目录结构

在 `gas-dl-v2/` 根下执行：

```bash
mkdir -p rcdw_mgda/configs
mkdir -p rcdw_mgda/rcdw/data
mkdir -p rcdw_mgda/rcdw/models
mkdir -p rcdw_mgda/rcdw/training
mkdir -p rcdw_mgda/rcdw/perturbation
mkdir -p rcdw_mgda/rcdw/utils
mkdir -p rcdw_mgda/scripts
mkdir -p rcdw_mgda/tests
mkdir -p rcdw_mgda/runs/stage_a
mkdir -p rcdw_mgda/runs/stage_b
```

### Step 0.2: `rcdw_mgda/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "rcdw-mgda"
version = "0.1.0"
description = "RCDW-MGDA: Reliability-Constrained Dynamic Weighting for Multi-Gas Detection"
requires-python = ">=3.10,<3.14"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "pyyaml>=6.0",
    "pytest>=7.0",
]

[tool.setuptools.packages.find]
include = ["rcdw*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Step 0.3: `rcdw_mgda/configs/default.yaml`

```yaml
data:
  n_train: 1400
  n_val: 300
  n_test: 300
  window: 8
  seed: 42

model:
  single_modal:
    hidden: 32
  error_net:
    hidden: 32
  fusion:
    beta: 8.0
    alpha_min: 0.1
    alpha_max: 0.9
    tau_alpha: 0.05
    s_min: 0.05
    s_max: 0.40
    tau_s: 0.05
  # W_base: 行=模态(NDIR/TCD/US), 列=气体(O2/CO2/N2)
  # 每列 sum=1.0
  W_base:
    - [0.05, 0.70, 0.05]
    - [0.50, 0.15, 0.45]
    - [0.45, 0.15, 0.50]

training:
  stage_a:
    epochs: 200
    batch_size: 16
    lr: 1.0e-3
    weight_decay: 1.0e-4
    patience: 30
    # NDIR 对 O2/N2 的 loss 降权
    ndir_loss_weights: [0.1, 1.0, 0.1]
  stage_b:
    epochs: 200
    batch_size: 16
    lr: 1.0e-3
    weight_decay: 1.0e-4
    patience: 30
    lambda_error: 1.0
    lambda_sum: 0.1
    freeze_single_modal: true

perturbation:
  kinds:
    - optical_atten
    - optical_scat
    - thermal
    - ultrasonic
    - temperature
  levels: [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]

degradation:
  ratio: 4.0
  cap: 0.04

eval:
  basis: dry
```

### Step 0.4: 所有 `__init__.py`

以下 7 个文件内容**全部相同**（空文件）：

- `rcdw_mgda/rcdw/__init__.py`
- `rcdw_mgda/rcdw/data/__init__.py`
- `rcdw_mgda/rcdw/models/__init__.py`
- `rcdw_mgda/rcdw/training/__init__.py`
- `rcdw_mgda/rcdw/perturbation/__init__.py`
- `rcdw_mgda/rcdw/utils/__init__.py`
- `rcdw_mgda/scripts/__init__.py` （**必须创建**，否则 `python -m scripts.train` 找不到）

每个文件内容：

```python
```

（空文件，无内容）

### Step 0.5: 验证

```bash
cd rcdw_mgda && python -c "import rcdw; print('OK')"
```

预期输出：`OK`

---

## M1 — 合成数据 + 单模态网络 + 阶段 A 训练

### Step 1.1: `rcdw_mgda/rcdw/data/synth.py`

```python
"""合成时序数据 + 滑窗切分。

生成平滑变化的三组分浓度时序，模拟真实标定过程中的连续采样。
输出滑窗张量 (N_windows, L, 6) 与标签 (N_windows, 3)。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def synth_timeseries(n_steps: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成 n_steps 个时间步的传感器读数与浓度真值。

    浓度沿时间轴平滑变化（线性插值 + 小噪声），保证滑窗内的
    CV / gradient / drift 等统计特征有非零值。

    Returns:
        X: (n_steps, 6)  [S_ndir, S_tc, S_us, P, T, RH]
        Y: (n_steps, 3)  [C_O2, C_CO2, C_N2], sum≈1
    """
    rng = np.random.default_rng(seed)

    # --- 平滑浓度轨迹 ---
    n_anchors = max(n_steps // 100, 5)
    anchors = rng.dirichlet([2.0, 1.0, 6.0], size=n_anchors + 1)
    seg_len = n_steps // n_anchors

    C = np.zeros((n_steps, 3), dtype=np.float64)
    for i in range(n_anchors):
        s = i * seg_len
        e = min((i + 1) * seg_len, n_steps)
        t = np.linspace(0.0, 1.0, e - s)
        for j in range(3):
            C[s:e, j] = (1.0 - t) * anchors[i, j] + t * anchors[i + 1, j]
    if n_anchors * seg_len < n_steps:
        C[n_anchors * seg_len :] = anchors[-1]

    C += 0.005 * rng.standard_normal(C.shape)
    C = np.clip(C, 0.01, None)
    C = C / C.sum(axis=1, keepdims=True)

    # --- 环境参数（缓慢漂移 + 小噪声） ---
    t_norm = np.linspace(0.0, 1.0, n_steps)
    T = 300.0 + 40.0 * np.sin(2.0 * np.pi * t_norm * 3.0) + 3.0 * rng.standard_normal(n_steps)
    P = 1.0 + 0.03 * np.sin(2.0 * np.pi * t_norm * 5.0) + 0.003 * rng.standard_normal(n_steps)
    RH = 0.025 + 0.015 * np.sin(2.0 * np.pi * t_norm * 2.0)
    RH = np.clip(RH + 0.002 * rng.standard_normal(n_steps), 0.0, 0.05)

    # --- 传感器信号（含物理近似） ---
    # NDIR: Beer-Lambert 对 CO2
    S_ndir = (1.0 - np.exp(-3.0 * C[:, 1])) + 0.01 * rng.standard_normal(n_steps)
    # 超声: v = sqrt(γRT / M_mix)
    M_mix = 32.0 * C[:, 0] + 44.0 * C[:, 1] + 28.0 * C[:, 2]
    S_us = np.sqrt(1.4 * 8.314 * T / (M_mix * 1e-3)) + 0.5 * rng.standard_normal(n_steps)
    # 热导: λ_mix ≈ Σ x_i λ_i
    S_tc = (0.026 * C[:, 0] + 0.017 * C[:, 1] + 0.026 * C[:, 2]
            + 1e-4 * rng.standard_normal(n_steps))

    X = np.stack([S_ndir, S_tc, S_us, P, T, RH], axis=1).astype(np.float32)
    Y = C.astype(np.float32)
    return X, Y


def make_windows(X: np.ndarray, Y: np.ndarray, L: int = 8
                 ) -> tuple[np.ndarray, np.ndarray]:
    """将 (N, 6) 时序切分为 (N-L+1, L, 6) 滑窗，标签取窗口最后时刻。"""
    N = len(X)
    assert N >= L, f"时序长度 {N} < 窗口 {L}"
    n_windows = N - L + 1
    X_w = np.zeros((n_windows, L, 6), dtype=np.float32)
    Y_w = np.zeros((n_windows, 3), dtype=np.float32)
    for i in range(n_windows):
        X_w[i] = X[i : i + L]
        Y_w[i] = Y[i + L - 1]
    return X_w, Y_w


def make_splits(
    n_train: int = 1400,
    n_val: int = 300,
    n_test: int = 300,
    L: int = 8,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """生成训练/验证/测试滑窗数据。

    Returns:
        {"train": (X_w, Y_w), "val": ..., "test": ...}
        X_w: (N, L, 6), Y_w: (N, 3)
    """
    n_total_windows = n_train + n_val + n_test
    n_raw = n_total_windows + L - 1
    X_raw, Y_raw = synth_timeseries(n_raw, seed=seed)
    X_w, Y_w = make_windows(X_raw, Y_raw, L=L)
    assert len(X_w) == n_total_windows

    s1 = n_train
    s2 = n_train + n_val
    return {
        "train": (X_w[:s1], Y_w[:s1]),
        "val": (X_w[s1:s2], Y_w[s1:s2]),
        "test": (X_w[s2:], Y_w[s2:]),
    }


class WindowedDataset(Dataset):
    """PyTorch Dataset，包装滑窗数据。"""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 3 and X.shape[1:] == (8, 6), f"X shape {X.shape} != (N, 8, 6)"
        assert Y.ndim == 2 and Y.shape[1] == 3, f"Y shape {Y.shape} != (N, 3)"
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
```

### Step 1.2: `rcdw_mgda/rcdw/data/preprocess.py`

```python
"""预处理占位模块。

合成数据已对齐，无需实际预处理。
真实数据接入时在此实现滤波/校准/补偿。
"""
from __future__ import annotations

import numpy as np


def sliding_mean_filter(x: np.ndarray, window: int = 8) -> np.ndarray:
    """滑动窗口均值滤波（按最后一个轴）。"""
    if x.shape[0] < window:
        return x
    kernel = np.ones(window) / window
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, x)
    return out.astype(x.dtype)


def calibrate_linear(x: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    """线性零点+跨度校准: y = a * x + b"""
    return (a * x + b).astype(x.dtype)
```

### Step 1.3: `rcdw_mgda/rcdw/models/single_modal.py`

```python
"""单模态浓度反演网络。

每个模态独立估计三气体浓度。
使用 clamp(min=0) + L1-normalize 保证 sum=1 且非负。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SingleModal(nn.Module):
    """基类：传感器信号(1) + 环境(3) → 三气体浓度(3)。"""

    def __init__(self, in_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4) = [sensor_value, P, T, RH]
        Returns:
            (B, 3) = [C_O2, C_CO2, C_N2], sum≈1, 非负
        """
        raw = self.net(x).clamp(min=0.0)
        return raw / (raw.sum(dim=-1, keepdim=True) + 1e-6)


class NDIRNet(SingleModal):
    """NDIR 模态。输入: [S_ndir, P, T, RH]。"""
    pass


class TCDNet(SingleModal):
    """热导 TCD 模态。输入: [S_tc, P, T, RH]。"""
    pass


class USNet(SingleModal):
    """超声 US 模态。输入: [S_us, P, T, RH]。"""
    pass


# ---- 辅助函数 ----

# 通道索引常量（对应 x 的 dim=-1）
SENSOR_INDICES = {"ndir": 0, "tcd": 1, "us": 2}
ENV_INDICES = [3, 4, 5]  # P, T, RH


def extract_modal_input(x_last: torch.Tensor, modality: str) -> torch.Tensor:
    """从最后时刻 (B, 6) 提取特定模态的 (B, 4) 输入。

    Args:
        x_last: (B, 6) = [S_ndir, S_tc, S_us, P, T, RH]
        modality: "ndir" | "tcd" | "us"
    Returns:
        (B, 4) = [sensor_value, P, T, RH]
    """
    sensor_idx = SENSOR_INDICES[modality]
    sensor_val = x_last[:, sensor_idx : sensor_idx + 1]  # (B, 1)
    env = x_last[:, ENV_INDICES]  # (B, 3)
    return torch.cat([sensor_val, env], dim=-1)  # (B, 4)
```

### Step 1.4: `rcdw_mgda/rcdw/training/metrics.py`

```python
"""评价指标：MAE / RMSE / MRE / ARE。"""
from __future__ import annotations

import torch


def compute_metrics(
    pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8
) -> dict[str, float]:
    """计算回归指标。

    Args:
        pred: (N, G=3) 预测浓度
        ref:  (N, G=3) 真值浓度
    Returns:
        {"MAE": ..., "RMSE": ..., "MRE": ..., "ARE": ...}
    """
    e = (pred - ref).abs()
    re = e / (ref.abs() + eps)
    return {
        "MAE": e.mean().item(),
        "RMSE": ((pred - ref) ** 2).mean().sqrt().item(),
        "MRE": re.mean().item() * 100.0,
        "ARE": re.max().item() * 100.0,
    }


def compute_per_gas_metrics(
    pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8
) -> dict[str, dict[str, float]]:
    """按气体分别计算指标。

    Returns:
        {"O2": {...}, "CO2": {...}, "N2": {...}, "overall": {...}}
    """
    gas_names = ["O2", "CO2", "N2"]
    result = {}
    for g, name in enumerate(gas_names):
        result[name] = compute_metrics(pred[:, g : g + 1], ref[:, g : g + 1], eps)
    result["overall"] = compute_metrics(pred, ref, eps)
    return result
```

### Step 1.5: `rcdw_mgda/rcdw/training/losses.py`

```python
"""损失函数。"""
from __future__ import annotations

import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """按气体加权的 MSE。

    Args:
        weights: (G=3,) 各气体的 loss 权重
    """

    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3)
            target: (B, 3)
        """
        mse_per_gas = ((pred - target) ** 2).mean(dim=0)  # (3,)
        return (mse_per_gas * self.w).sum() / self.w.sum()


class StageBLoss(nn.Module):
    """阶段 B 联合损失。

    L = MSE(C_fused, C_ref)
      + lambda_e * MSE(E_pred, E_true)
      + lambda_s * |sum(C_fused) - 1|
    """

    def __init__(self, lambda_e: float = 1.0, lambda_s: float = 0.1):
        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_s = lambda_s

    def forward(
        self,
        C_fused: torch.Tensor,
        C_ref: torch.Tensor,
        E_pred: torch.Tensor,
        Y_modal: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            C_fused: (B, G=3)   融合输出
            C_ref:   (B, G=3)   真值
            E_pred:  (B, M, G)  预测误差
            Y_modal: (B, M, G)  单模态候选（已 detach 或已冻结）
        """
        # 融合 MSE
        loss_fuse = ((C_fused - C_ref) ** 2).mean()

        # 误差监督: E_true = |Y_modal - C_ref|
        C_ref_expand = C_ref.unsqueeze(1).expand_as(Y_modal)  # (B, M, G)
        E_true = (Y_modal - C_ref_expand).abs()
        loss_error = ((E_pred - E_true) ** 2).mean()

        # 组分约束
        loss_sum = (C_fused.sum(dim=-1) - 1.0).abs().mean()

        total = loss_fuse + self.lambda_e * loss_error + self.lambda_s * loss_sum

        details = {
            "loss_total": total.item(),
            "loss_fuse": loss_fuse.item(),
            "loss_error": loss_error.item(),
            "loss_sum": loss_sum.item(),
        }
        return total, details
```

### Step 1.6: `rcdw_mgda/rcdw/training/stage_a.py`

```python
"""阶段 A：单模态独立预训练。

依次训练 NDIRNet / TCDNet / USNet，各自保存 checkpoint。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from rcdw.models.single_modal import NDIRNet, TCDNet, USNet, extract_modal_input
from rcdw.training.losses import WeightedMSE
from rcdw.training.metrics import compute_metrics


def train_single_modal(
    model: nn.Module,
    modality: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_weights: list[float],
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    save_path: Path | str = "runs/stage_a",
    device: str = "cpu",
) -> nn.Module:
    """训练单个模态网络。"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = WeightedMSE(loss_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- 训练 ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_w, y in train_loader:
            x_w, y = x_w.to(device), y.to(device)
            x_last = x_w[:, -1, :]  # (B, 6)
            inp = extract_modal_input(x_last, modality)  # (B, 4)
            pred = model(inp)  # (B, 3)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(y)
            train_count += len(y)
        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_w, y in val_loader:
                x_w, y = x_w.to(device), y.to(device)
                x_last = x_w[:, -1, :]
                inp = extract_modal_input(x_last, modality)
                pred = model(inp)
                loss = criterion(pred, y)
                val_loss_sum += loss.item() * len(y)
                val_count += len(y)
        val_loss = val_loss_sum / val_count

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{modality}] epoch {epoch:3d}  "
                  f"train={train_loss_sum / train_count:.6f}  val={val_loss:.6f}")

        # --- 早停 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / f"{modality}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{modality}] early stop at epoch {epoch}")
                break

    # 加载最佳权重
    model.load_state_dict(torch.load(save_path / f"{modality}.pt", weights_only=True))
    return model


def run_stage_a(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: str = "cpu",
) -> dict[str, nn.Module]:
    """运行完整的阶段 A：训练 3 个单模态网络。"""
    sa = cfg["training"]["stage_a"]
    hidden = cfg["model"]["single_modal"]["hidden"]

    models = {
        "ndir": NDIRNet(in_dim=4, hidden=hidden),
        "tcd": TCDNet(in_dim=4, hidden=hidden),
        "us": USNet(in_dim=4, hidden=hidden),
    }

    loss_weights_map = {
        "ndir": sa["ndir_loss_weights"],   # [0.1, 1.0, 0.1]
        "tcd": [1.0, 1.0, 1.0],
        "us": [1.0, 1.0, 1.0],
    }

    trained = {}
    for name, model in models.items():
        print(f"\n=== Stage A: training {name.upper()} ===")
        trained[name] = train_single_modal(
            model,
            modality=name,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_weights=loss_weights_map[name],
            epochs=sa["epochs"],
            lr=sa["lr"],
            weight_decay=sa["weight_decay"],
            patience=sa["patience"],
            device=device,
        )

    return trained
```

### Step 1.7: 验证 M1（在实现 train.py 之前先测试数据）

创建 `rcdw_mgda/tests/test_synth.py`：

```python
"""测试合成数据模块。"""
import numpy as np
import pytest
from rcdw.data.synth import synth_timeseries, make_windows, make_splits, WindowedDataset


def test_timeseries_shape_and_sum():
    X, Y = synth_timeseries(100, seed=0)
    assert X.shape == (100, 6)
    assert Y.shape == (100, 3)
    np.testing.assert_allclose(Y.sum(axis=1), 1.0, atol=0.02)


def test_window_shape():
    X, Y = synth_timeseries(20, seed=0)
    Xw, Yw = make_windows(X, Y, L=8)
    assert Xw.shape == (13, 8, 6)
    assert Yw.shape == (13, 3)


def test_splits_no_overlap():
    splits = make_splits(n_train=100, n_val=20, n_test=20, L=8, seed=0)
    assert splits["train"][0].shape[0] == 100
    assert splits["val"][0].shape[0] == 20
    assert splits["test"][0].shape[0] == 20


def test_windowed_dataset():
    splits = make_splits(n_train=50, n_val=10, n_test=10, L=8, seed=0)
    ds = WindowedDataset(*splits["train"])
    x, y = ds[0]
    assert x.shape == (8, 6)
    assert y.shape == (3,)
    assert y.sum().item() == pytest.approx(1.0, abs=0.02)
```

创建 `rcdw_mgda/tests/test_single_modal.py`：

```python
"""测试单模态网络。"""
import torch
import pytest
from rcdw.models.single_modal import NDIRNet, TCDNet, USNet, extract_modal_input


def test_output_shape():
    net = NDIRNet(in_dim=4, hidden=32)
    x = torch.randn(8, 4)
    y = net(x)
    assert y.shape == (8, 3)


def test_output_sum_one():
    net = TCDNet(in_dim=4, hidden=32)
    x = torch.randn(16, 4)
    y = net(x)
    sums = y.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones(16), atol=1e-4, rtol=1e-4)


def test_output_non_negative():
    net = USNet(in_dim=4, hidden=32)
    x = torch.randn(16, 4)
    y = net(x)
    assert (y >= 0).all()


def test_extract_modal_input():
    x_last = torch.randn(4, 6)  # [S_ndir, S_tc, S_us, P, T, RH]
    inp = extract_modal_input(x_last, "ndir")
    assert inp.shape == (4, 4)
    torch.testing.assert_close(inp[:, 0], x_last[:, 0])  # S_ndir
    torch.testing.assert_close(inp[:, 1], x_last[:, 3])  # P
    torch.testing.assert_close(inp[:, 2], x_last[:, 4])  # T
    torch.testing.assert_close(inp[:, 3], x_last[:, 5])  # RH
```

验证：

```bash
cd rcdw_mgda && python -m pytest tests/test_synth.py tests/test_single_modal.py -v
```

预期：4 + 4 = 8 passed。

---

## M2 — RCDW 融合层 + 数值对齐

### Step 2.1: `rcdw_mgda/rcdw/models/rcdw.py`

```python
"""RCDW 融合层 + 整体 RCDW_MGDA 模型。"""
from __future__ import annotations

import torch
import torch.nn as nn

from rcdw.models.single_modal import NDIRNet, TCDNet, USNet, extract_modal_input
from rcdw.models.feature import FeatureExtractor
from rcdw.models.error_net import ErrorNet


class RCDWFusion(nn.Module):
    """可靠性约束动态加权融合层（可微，无可学习参数）。

    维度约定:
      W_base[m, g]: 模态 m 对气体 g 的基线权重
      softmax 在 dim=1 (模态维) 归一化
      即对每种气体 g，各模态权重 sum=1
    """

    def __init__(
        self,
        W_base: torch.Tensor,
        *,
        beta: float = 8.0,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        tau_a: float = 0.05,
        s_min: float = 0.05,
        s_max: float = 0.40,
        tau_s: float = 0.05,
    ):
        super().__init__()
        assert W_base.shape == (3, 3), f"W_base shape {W_base.shape} != (M=3, G=3)"
        self.register_buffer("W_base", W_base.clone())
        self.beta = beta
        self.a_min = alpha_min
        self.a_max = alpha_max
        self.tau_a = tau_a
        self.s_min = s_min
        self.s_max = s_max
        self.tau_s = tau_s

    def forward(
        self, Y_modal: torch.Tensor, E_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Y_modal: (B, M=3, G=3)  单模态候选浓度
            E_pred:  (B, M=3, G=3)  预测误差（恒正）
        Returns:
            C_fused: (B, G=3)
            W_final: (B, M=3, G=3)
        """
        eps = 1e-6

        # Step 1: softmax Wmix — 误差小的模态权重大
        # dim=1 是模态维，对每种气体归一化
        Wmix = torch.softmax(-self.beta * E_pred, dim=1)  # (B, M, G)

        # Step 2: 自适应 alpha 和 shift
        # dE: 模态间误差差异 (B, G)
        E_max = E_pred.max(dim=1).values  # (B, G)
        E_min = E_pred.min(dim=1).values  # (B, G)
        dE = E_max - E_min  # (B, G)

        alpha = self.a_min + (self.a_max - self.a_min) * dE / (dE + self.tau_a)  # (B, G)
        shift = self.s_min + (self.s_max - self.s_min) * dE / (dE + self.tau_s)  # (B, G)

        # 扩展到 (B, 1, G) 以广播
        alpha = alpha.unsqueeze(1)  # (B, 1, G)
        shift = shift.unsqueeze(1)  # (B, 1, G)

        # Step 3: 基线锚定
        # W_base: (M, G) 广播到 (B, M, G)
        W = (1.0 - alpha) * self.W_base + alpha * Wmix  # (B, M, G)

        # Step 4: maxShift 约束
        W = torch.clamp(W, self.W_base - shift, self.W_base + shift)

        # Step 5: 重归一化（对每种气体，各模态权重 sum=1）
        W = W / (W.sum(dim=1, keepdim=True) + eps)

        # Step 6: 加权融合
        C_fused = (W * Y_modal).sum(dim=1)  # (B, G)

        return C_fused, W


class RCDW_MGDA(nn.Module):
    """完整的 RCDW-MGDA 模型。

    输入: (B, L=8, 6)
    输出: dict with C, Y_modal, E_pred, W
    """

    def __init__(self, W_base: torch.Tensor, hidden: int = 32):
        super().__init__()
        self.ndir = NDIRNet(in_dim=4, hidden=hidden)
        self.tcd = TCDNet(in_dim=4, hidden=hidden)
        self.usn = USNet(in_dim=4, hidden=hidden)
        self.feat = FeatureExtractor(window=8)
        self.err = ErrorNet(in_dim=13, n_gas=3, hidden=hidden)
        self.fuse = RCDWFusion(W_base)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, L=8, 6) 滑窗输入
        Returns:
            {"C": (B,3), "Y_modal": (B,3,3), "E_pred": (B,3,3), "W": (B,3,3)}
        """
        # 单模态反演：取窗口最后时刻
        x_last = x[:, -1, :]  # (B, 6)

        y_nd = self.ndir(extract_modal_input(x_last, "ndir"))  # (B, 3)
        y_tc = self.tcd(extract_modal_input(x_last, "tcd"))    # (B, 3)
        y_us = self.usn(extract_modal_input(x_last, "us"))     # (B, 3)
        Y_modal = torch.stack([y_nd, y_tc, y_us], dim=1)       # (B, M=3, G=3)

        # 特征提取：用完整滑窗
        feat = self.feat(x, Y_modal)  # (B, M=3, F=13)

        # 误差预测
        E_pred = self.err(feat)  # (B, M=3, G=3)

        # RCDW 融合
        C_fused, W = self.fuse(Y_modal, E_pred)

        return {"C": C_fused, "Y_modal": Y_modal, "E_pred": E_pred, "W": W}
```

### Step 2.2: `rcdw_mgda/scripts/numerical_check.py`

```python
"""数值对齐脚本：验证 RCDWFusion 的 PyTorch 实现与公式手算结果一致。

运行: cd rcdw_mgda && python -m scripts.numerical_check
通过条件: 所有维度 max abs diff < 1e-5
"""
from __future__ import annotations

import torch
import numpy as np
import sys


def hand_compute_rcdw(
    Y_modal: np.ndarray,
    E_pred: np.ndarray,
    W_base: np.ndarray,
    beta: float = 8.0,
    alpha_min: float = 0.1,
    alpha_max: float = 0.9,
    tau_a: float = 0.05,
    s_min: float = 0.05,
    s_max: float = 0.40,
    tau_s: float = 0.05,
) -> dict[str, np.ndarray]:
    """纯 NumPy 手算 RCDW 融合结果。"""
    eps = 1e-6
    B, M, G = Y_modal.shape

    # Wmix: softmax(-beta * E) over modality dim
    logits = -beta * E_pred  # (B, M, G)
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    Wmix = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # alpha and shift
    E_max = E_pred.max(axis=1)  # (B, G)
    E_min = E_pred.min(axis=1)  # (B, G)
    dE = E_max - E_min

    alpha = alpha_min + (alpha_max - alpha_min) * dE / (dE + tau_a)
    shift = s_min + (s_max - s_min) * dE / (dE + tau_s)

    alpha_3d = alpha[:, np.newaxis, :]  # (B, 1, G)
    shift_3d = shift[:, np.newaxis, :]

    # 基线锚定
    W = (1.0 - alpha_3d) * W_base + alpha_3d * Wmix

    # maxShift clamp
    W = np.clip(W, W_base - shift_3d, W_base + shift_3d)

    # 重归一
    W = W / (W.sum(axis=1, keepdims=True) + eps)

    # 融合
    C_fused = (W * Y_modal).sum(axis=1)

    return {"Wmix": Wmix, "alpha": alpha, "shift": shift, "W": W, "C_fused": C_fused}


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    B, M, G = 4, 3, 3

    # 固定输入
    Y_modal_np = np.random.rand(B, M, G).astype(np.float32)
    # 对每个模态归一化（模拟 sum=1 约束）
    Y_modal_np = Y_modal_np / Y_modal_np.sum(axis=2, keepdims=True)

    E_pred_np = np.abs(np.random.randn(B, M, G).astype(np.float32)) * 0.1

    W_base_np = np.array([
        [0.05, 0.70, 0.05],
        [0.50, 0.15, 0.45],
        [0.45, 0.15, 0.50],
    ], dtype=np.float32)

    # NumPy 手算
    expected = hand_compute_rcdw(Y_modal_np, E_pred_np, W_base_np)

    # PyTorch 计算
    from rcdw.models.rcdw import RCDWFusion

    W_base_t = torch.from_numpy(W_base_np)
    fusion = RCDWFusion(W_base_t)
    fusion.eval()

    Y_modal_t = torch.from_numpy(Y_modal_np)
    E_pred_t = torch.from_numpy(E_pred_np)

    with torch.no_grad():
        C_fused_t, W_t = fusion(Y_modal_t, E_pred_t)

    # 对比
    tol = 1e-5
    all_pass = True

    checks = [
        ("W", W_t.numpy(), expected["W"]),
        ("C_fused", C_fused_t.numpy(), expected["C_fused"]),
    ]

    for name, actual, exp in checks:
        diff = np.abs(actual - exp).max()
        status = "PASS" if diff < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: max_abs_diff = {diff:.2e}  [{status}]")

    # 检查权重 sum=1（对每种气体）
    W_sum = W_t.sum(dim=1).numpy()
    sum_diff = np.abs(W_sum - 1.0).max()
    sum_status = "PASS" if sum_diff < tol else "FAIL"
    if sum_status == "FAIL":
        all_pass = False
    print(f"  W_sum=1: max_abs_diff = {sum_diff:.2e}  [{sum_status}]")

    if all_pass:
        print("\n=== ALL CHECKS PASSED ===")
    else:
        print("\n=== SOME CHECKS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### Step 2.3: `rcdw_mgda/tests/test_rcdw_fusion.py`

```python
"""测试 RCDWFusion 数值对齐、维度、可微性。"""
import torch
import numpy as np
import pytest
from rcdw.models.rcdw import RCDWFusion


@pytest.fixture
def W_base():
    return torch.tensor([
        [0.05, 0.70, 0.05],
        [0.50, 0.15, 0.45],
        [0.45, 0.15, 0.50],
    ], dtype=torch.float32)


@pytest.fixture
def fusion(W_base):
    return RCDWFusion(W_base)


def test_output_shape(fusion):
    Y = torch.rand(8, 3, 3)
    E = torch.rand(8, 3, 3).abs() * 0.1
    C, W = fusion(Y, E)
    assert C.shape == (8, 3)
    assert W.shape == (8, 3, 3)


def test_weight_sum_one(fusion):
    Y = torch.rand(16, 3, 3)
    E = torch.rand(16, 3, 3).abs() * 0.1
    _, W = fusion(Y, E)
    W_sum = W.sum(dim=1)  # (B, G)
    torch.testing.assert_close(W_sum, torch.ones_like(W_sum), atol=1e-5, rtol=1e-5)


def test_differentiable(fusion):
    Y = torch.rand(4, 3, 3, requires_grad=True)
    E = torch.rand(4, 3, 3, requires_grad=True).abs() * 0.1 + 0.01
    C, W = fusion(Y, E)
    loss = C.sum()
    loss.backward()
    assert Y.grad is not None
    assert E.grad is not None


def test_alpha_boundary(W_base):
    fusion = RCDWFusion(W_base, alpha_min=0.0, alpha_max=1.0, tau_a=0.05)
    # 所有模态误差相同 → dE=0 → alpha=alpha_min=0 → W=W_base
    Y = torch.rand(4, 3, 3)
    E = torch.ones(4, 3, 3) * 0.05
    _, W = fusion(Y, E)
    for b in range(4):
        for g in range(3):
            torch.testing.assert_close(
                W[b, :, g], W_base[:, g], atol=1e-4, rtol=1e-4
            )


def test_shift_clamp(W_base):
    fusion = RCDWFusion(W_base, s_min=0.01, s_max=0.01, tau_s=0.05)
    Y = torch.rand(4, 3, 3)
    E = torch.rand(4, 3, 3).abs()
    _, W = fusion(Y, E)
    diff = (W - W_base.unsqueeze(0)).abs()
    # clamp 后差异应 <= s_max + 归一化微调
    assert diff.max().item() < 0.05


def test_zero_error_uses_baseline(W_base):
    fusion = RCDWFusion(W_base)
    Y = torch.rand(4, 3, 3)
    E = torch.zeros(4, 3, 3)
    _, W = fusion(Y, E)
    for b in range(4):
        torch.testing.assert_close(W[b], W_base, atol=1e-4, rtol=1e-4)


def test_numerical_alignment(W_base):
    """与手算结果对齐（简化版 numerical_check）。"""
    torch.manual_seed(123)
    B = 2
    Y = torch.rand(B, 3, 3)
    E = torch.rand(B, 3, 3).abs() * 0.1

    fusion = RCDWFusion(W_base)
    C, W = fusion(Y, E)

    # 手算 Wmix
    logits = -8.0 * E
    Wmix = torch.softmax(logits, dim=1)

    dE = E.max(dim=1).values - E.min(dim=1).values
    alpha = 0.1 + 0.8 * dE / (dE + 0.05)
    alpha = alpha.unsqueeze(1)

    W_expected = (1 - alpha) * W_base + alpha * Wmix
    shift = 0.05 + 0.35 * dE.unsqueeze(1) / (dE.unsqueeze(1) + 0.05)
    W_expected = W_expected.clamp(W_base - shift, W_base + shift)
    W_expected = W_expected / W_expected.sum(dim=1, keepdim=True)

    torch.testing.assert_close(W, W_expected, atol=1e-5, rtol=1e-5)


def test_w_base_shape_assertion():
    with pytest.raises(AssertionError):
        RCDWFusion(torch.rand(3, 4))  # 错误 shape
```

验证：

```bash
cd rcdw_mgda && python -m pytest tests/test_rcdw_fusion.py -v
cd rcdw_mgda && python -m scripts.numerical_check
```

预期：8 passed + numerical_check ALL CHECKS PASSED。

---

## M3 — 特征提取 + 误差预测 + 阶段 B 联合训练

### Step 3.1: `rcdw_mgda/rcdw/models/feature.py`

```python
"""13 维扰动感知特征提取器。

输入始终为滑窗 (B, L=8, 6)。
输出 (B, M=3, F=13)。

特征列表:
  0  CV_m        滑窗变异系数 std/mean
  1  D_m         群体中位偏离 (跨气体平均)
  2  G_m         一阶差分能量 mean((S_k - S_{k-1})^2)
  3  Q_m         信号质量比 snr_m / sum(snr)
  4  B_m         群体偏差 (跨气体平均)
  5  delta_T     |T_k - T_{k-1}|
  6  delta_P     |P_k - P_{k-1}|
  7  delta_RH    |RH_k - RH_{k-1}|
  8  dev_max     |Y_m - mean(Y)| 跨气体 max
  9  dev_mean    |Y_m - mean(Y)| 跨气体 mean
  10 snr_proxy   |mu| / sigma
  11 drift       滑窗线性拟合斜率
  12 dt          固定采样间隔 = 1.0
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """扰动感知特征提取（纯计算，无可学习参数）。"""

    def __init__(self, window: int = 8):
        super().__init__()
        self.L = window
        # 预计算线性拟合用的时间轴
        t = torch.arange(window, dtype=torch.float32)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        self.register_buffer("_t_centered", t - t_mean)  # (L,)
        self.register_buffer("_t_var", torch.tensor(t_var))

    def forward(
        self, x: torch.Tensor, Y_modal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, L, 6)   滑窗传感器数据
            Y_modal: (B, M=3, G=3) 单模态浓度候选
        Returns:
            (B, M=3, F=13) 扰动感知特征
        """
        B, L, _ = x.shape
        M, G = 3, 3
        eps = 1e-8
        device = x.device
        dtype = x.dtype

        # 传感器信号: (B, L, 3) = S_ndir, S_tc, S_us
        S = x[:, :, :3]
        # 环境: (B, L, 3) = P, T, RH
        env = x[:, :, 3:]

        feats = torch.zeros(B, M, 13, device=device, dtype=dtype)

        # --- 环境变化率（所有模态共享） ---
        # delta_T, delta_P, delta_RH: 最后两步之差
        delta_T = (env[:, -1, 1] - env[:, -2, 1]).abs()    # (B,)  T=env[:,1]
        delta_P = (env[:, -1, 0] - env[:, -2, 0]).abs()    # (B,)  P=env[:,0]
        delta_RH = (env[:, -1, 2] - env[:, -2, 2]).abs()   # (B,)  RH=env[:,2]

        # --- 跨模态统计 ---
        Y_median = Y_modal.median(dim=1).values  # (B, G)
        Y_mean = Y_modal.mean(dim=1)              # (B, G)

        # 信号质量比 SNR 分母（所有模态的 SNR 之和）
        snr_all = torch.zeros(B, M, device=device, dtype=dtype)
        for m in range(M):
            s_m = S[:, :, m]  # (B, L)
            mu_m = s_m.mean(dim=1)
            sigma_m = s_m.std(dim=1)
            snr_all[:, m] = mu_m.abs() / (sigma_m + eps)
        snr_total = snr_all.sum(dim=1, keepdim=True) + eps  # (B, 1)

        # --- 逐模态计算 ---
        for m in range(M):
            s_m = S[:, :, m]  # (B, L)
            Y_m = Y_modal[:, m, :]  # (B, G)

            # 0: CV_m = std / |mean|
            mu = s_m.mean(dim=1)         # (B,)
            sigma = s_m.std(dim=1)       # (B,)
            feats[:, m, 0] = sigma / (mu.abs() + eps)

            # 1: D_m = |Y_m - median(Y)|, 跨气体平均
            feats[:, m, 1] = (Y_m - Y_median).abs().mean(dim=-1)

            # 2: G_m = gradient energy
            diffs_sq = (s_m[:, 1:] - s_m[:, :-1]) ** 2  # (B, L-1)
            feats[:, m, 2] = diffs_sq.mean(dim=1)

            # 3: Q_m = snr_m / sum(snr)
            feats[:, m, 3] = snr_all[:, m] / snr_total.squeeze(1)

            # 4: B_m = group bias, 跨气体平均
            bias_sum = torch.zeros(B, device=device, dtype=dtype)
            for j in range(M):
                if j != m:
                    bias_sum += (Y_m - Y_modal[:, j, :]).abs().mean(dim=-1)
            feats[:, m, 4] = bias_sum / (M - 1)

            # 5,6,7: 环境变化率
            feats[:, m, 5] = delta_T
            feats[:, m, 6] = delta_P
            feats[:, m, 7] = delta_RH

            # 8: dev_max = |Y_m - mean(Y)| 跨气体 max
            feats[:, m, 8] = (Y_m - Y_mean).abs().max(dim=-1).values

            # 9: dev_mean = |Y_m - mean(Y)| 跨气体 mean
            feats[:, m, 9] = (Y_m - Y_mean).abs().mean(dim=-1)

            # 10: snr_proxy = |mu| / sigma
            feats[:, m, 10] = mu.abs() / (sigma + eps)

            # 11: drift = 线性拟合斜率
            s_centered = s_m - s_m.mean(dim=1, keepdim=True)  # (B, L)
            # slope = sum(t_centered * s_centered) / t_var
            slope = (self._t_centered.unsqueeze(0) * s_centered).sum(dim=1) / (self._t_var + eps)
            feats[:, m, 11] = slope

            # 12: dt = 固定常数
            feats[:, m, 12] = 1.0

        return feats
```

### Step 3.2: `rcdw_mgda/rcdw/models/error_net.py`

```python
"""误差预测器 ErrorNet。

每个气体一个独立 head，输入 13 维特征，输出预测误差（恒正）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ErrorNet(nn.Module):
    """误差预测网络，3 个独立 head 分别预测 O₂/CO₂/N₂ 的模态误差。

    输入: (B, M=3, F=13) 扰动特征
    输出: (B, M=3, G=3)  预测误差，恒正（Softplus 保证）
    """

    def __init__(self, in_dim: int = 13, n_gas: int = 3, hidden: int = 32):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )
            for _ in range(n_gas)
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, M=3, F=13)
        Returns:
            E_pred: (B, M=3, G=3)  恒正
        """
        outs = []
        for head in self.heads:
            out = head(feat).squeeze(-1)  # (B, M)
            outs.append(out)
        return torch.stack(outs, dim=-1)  # (B, M, G)
```

### Step 3.3: `rcdw_mgda/rcdw/training/stage_b.py`

```python
"""阶段 B：联合训练 ErrorNet + RCDW（冻结单模态网络）。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from rcdw.models.rcdw import RCDW_MGDA
from rcdw.training.losses import StageBLoss
from rcdw.training.metrics import compute_metrics


def run_stage_b(
    model: RCDW_MGDA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: str = "cpu",
) -> RCDW_MGDA:
    """运行阶段 B 联合训练。"""
    sb = cfg["training"]["stage_b"]
    save_path = Path("runs/stage_b")
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # 冻结单模态网络
    if sb.get("freeze_single_modal", True):
        for name in ["ndir", "tcd", "usn"]:
            sub = getattr(model, name)
            for param in sub.parameters():
                param.requires_grad = False
        print("  [stage_b] single modal networks frozen")

    # 只优化 ErrorNet 参数（FeatureExtractor 和 RCDWFusion 无可学习参数）
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [stage_b] trainable params: {sum(p.numel() for p in trainable)}")

    criterion = StageBLoss(
        lambda_e=sb["lambda_error"],
        lambda_s=sb["lambda_sum"],
    ).to(device)
    optimizer = torch.optim.AdamW(trainable, lr=sb["lr"], weight_decay=sb["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sb["epochs"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, sb["epochs"] + 1):
        # --- 训练 ---
        model.train()
        # FeatureExtractor 和 RCDWFusion 无 dropout/BN，train mode 不影响
        train_loss_sum = 0.0
        train_count = 0
        for x_w, y in train_loader:
            x_w, y = x_w.to(device), y.to(device)
            out = model(x_w)
            loss, _ = criterion(out["C"], y, out["E_pred"], out["Y_modal"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(y)
            train_count += len(y)
        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        all_pred = []
        all_ref = []
        with torch.no_grad():
            for x_w, y in val_loader:
                x_w, y = x_w.to(device), y.to(device)
                out = model(x_w)
                loss, _ = criterion(out["C"], y, out["E_pred"], out["Y_modal"])
                val_loss_sum += loss.item() * len(y)
                val_count += len(y)
                all_pred.append(out["C"].cpu())
                all_ref.append(y.cpu())
        val_loss = val_loss_sum / val_count

        if epoch % 20 == 0 or epoch == 1:
            pred_cat = torch.cat(all_pred)
            ref_cat = torch.cat(all_ref)
            m = compute_metrics(pred_cat, ref_cat)
            print(f"  [stage_b] epoch {epoch:3d}  "
                  f"train={train_loss_sum / train_count:.6f}  val={val_loss:.6f}  "
                  f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

        # --- 早停 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "rcdw.pt")
        else:
            patience_counter += 1
            if patience_counter >= sb["patience"]:
                print(f"  [stage_b] early stop at epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(save_path / "rcdw.pt", weights_only=True)
    )
    return model
```

### Step 3.4: `rcdw_mgda/scripts/train.py`

```python
"""一键两阶段训练入口。

用法: cd rcdw_mgda && python -m scripts.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from rcdw.data.synth import make_splits, WindowedDataset
from rcdw.training.stage_a import run_stage_a
from rcdw.training.stage_b import run_stage_b
from rcdw.models.rcdw import RCDW_MGDA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stage", type=str, default="both",
                        choices=["a", "b", "both"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- 数据 ---
    dc = cfg["data"]
    splits = make_splits(
        n_train=dc["n_train"], n_val=dc["n_val"], n_test=dc["n_test"],
        L=dc["window"], seed=dc["seed"],
    )
    bs = cfg["training"]["stage_a"]["batch_size"]
    train_loader = DataLoader(WindowedDataset(*splits["train"]), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WindowedDataset(*splits["val"]), batch_size=bs)

    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        print(f"Auto-selected device: {device}")

    # --- Stage A ---
    if args.stage in ("a", "both"):
        print("\n" + "=" * 60)
        print("STAGE A: Single-modal pretraining")
        print("=" * 60)
        trained_models = run_stage_a(train_loader, val_loader, cfg, device=device)

    # --- Stage B ---
    if args.stage in ("b", "both"):
        print("\n" + "=" * 60)
        print("STAGE B: Joint training ErrorNet + RCDW")
        print("=" * 60)

        W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
        hidden = cfg["model"]["single_modal"]["hidden"]
        model = RCDW_MGDA(W_base, hidden=hidden)

        # 加载 Stage A checkpoint
        for name, attr_name in [("ndir", "ndir"), ("tcd", "tcd"), ("us", "usn")]:
            ckpt_path = Path(f"runs/stage_a/{name}.pt")
            if ckpt_path.exists():
                getattr(model, attr_name).load_state_dict(
                    torch.load(ckpt_path, weights_only=True)
                )
                print(f"  Loaded {ckpt_path}")
            else:
                print(f"  WARNING: {ckpt_path} not found, using random init")

        model = run_stage_b(model, train_loader, val_loader, cfg, device=device)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
```

### Step 3.5: 测试文件

`rcdw_mgda/tests/test_feature.py`：

```python
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
```

`rcdw_mgda/tests/test_error_net.py`：

```python
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
```

验证：

```bash
cd rcdw_mgda && python -m pytest tests/ -v
```

预期：全部通过。

---

## M4 — 扰动实验

### Step 4.1: `rcdw_mgda/rcdw/perturbation/inject.py`

```python
"""扰动注入：五类传感器退化/环境突变模拟。"""
from __future__ import annotations

import torch

PERTURBATION_KINDS = [
    "optical_atten",
    "optical_scat",
    "thermal",
    "ultrasonic",
    "temperature",
]


def inject(
    x: torch.Tensor, kind: str, level: float
) -> torch.Tensor:
    """在输入张量上注入指定类型和强度的扰动。

    Args:
        x:     (B, L, 6) 或 (N, L, 6) 滑窗输入
        kind:  扰动类型
        level: 扰动强度，0=无扰动，0.11=最大
    Returns:
        扰动后的张量（原张量不变）
    """
    x = x.clone()

    if kind == "optical_atten":
        # NDIR 通道乘性衰减（光源老化 / 光路污染）
        x[..., 0] *= (1.0 - level)

    elif kind == "optical_scat":
        # NDIR 通道加性高斯噪声（散射干扰）
        x[..., 0] = x[..., 0] + level * torch.randn_like(x[..., 0])

    elif kind == "thermal":
        # 热导通道乘性扰动（温控漂移）
        x[..., 1] = x[..., 1] * (1.0 + level * torch.randn_like(x[..., 1]))

    elif kind == "ultrasonic":
        # 超声通道加性噪声（换能器老化 / 耦合不良）
        scale = x[..., 2].abs().mean()
        x[..., 2] = x[..., 2] + level * scale * torch.randn_like(x[..., 2])

    elif kind == "temperature":
        # 温度阶跃偏移（T 通道, 索引 4）
        x[..., 4] = x[..., 4] + level * 80.0

    else:
        raise ValueError(f"unknown perturbation kind: {kind}. "
                         f"valid: {PERTURBATION_KINDS}")
    return x
```

### Step 4.2: `rcdw_mgda/scripts/perturb.py`

```python
"""扰动实验脚本：五类扰动 × 7 强度，输出指标曲线 + 权重曲线。

用法: cd rcdw_mgda && python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt
"""
from __future__ import annotations

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rcdw.data.synth import make_splits, WindowedDataset
from rcdw.models.rcdw import RCDW_MGDA
from rcdw.perturbation.inject import inject, PERTURBATION_KINDS
from rcdw.training.metrics import compute_per_gas_metrics
from rcdw.utils.degradation import hard_suppress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/stage_b/rcdw.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/perturb")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    dc = cfg["data"]
    splits = make_splits(
        n_train=dc["n_train"], n_val=dc["n_val"], n_test=dc["n_test"],
        L=dc["window"], seed=dc["seed"],
    )
    X_test = torch.from_numpy(splits["test"][0])
    Y_test = torch.from_numpy(splits["test"][1])

    # 模型
    W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
    model = RCDW_MGDA(W_base, hidden=cfg["model"]["single_modal"]["hidden"])
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    kinds = cfg["perturbation"]["kinds"]
    levels = cfg["perturbation"]["levels"]
    deg_cfg = cfg["degradation"]
    gas_names = ["O2", "CO2", "N2"]

    for kind in kinds:
        print(f"\n=== Perturbation: {kind} ===")
        results_by_level = []
        weights_by_level = []

        for level in levels:
            X_perturbed = inject(X_test, kind, level)
            with torch.no_grad():
                out = model(X_perturbed)
                # 退化硬抑制
                W_final, degraded = hard_suppress(
                    out["W"], out["E_pred"],
                    ratio=deg_cfg["ratio"], cap=deg_cfg["cap"],
                )
                C_fused = (W_final * out["Y_modal"]).sum(dim=1)

            metrics = compute_per_gas_metrics(C_fused, Y_test)
            results_by_level.append(metrics)

            # 平均权重: (M=3, G=3)
            W_avg = W_final.mean(dim=0).numpy()
            weights_by_level.append(W_avg)

            print(f"  level={level:.2f}  "
                  f"MAE={metrics['overall']['MAE']:.4f}  "
                  f"RMSE={metrics['overall']['RMSE']:.4f}  "
                  f"degraded={degraded.any().item()}")

        # --- 绘制指标曲线 ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for g, gas in enumerate(gas_names):
            ax = axes[g]
            for metric_name in ["MAE", "RMSE", "MRE"]:
                vals = [r[gas][metric_name] for r in results_by_level]
                ax.plot(levels, vals, marker="o", label=metric_name)
            ax.set_xlabel("Perturbation level")
            ax.set_ylabel("Error")
            ax.set_title(f"{kind} → {gas}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{kind}_metrics.png", dpi=150)
        plt.close()

        # --- 绘制权重曲线（以 CO2 为例）---
        fig, ax = plt.subplots(figsize=(8, 5))
        modal_names = ["NDIR", "TCD", "US"]
        for m, modal in enumerate(modal_names):
            vals = [w[m, 1] for w in weights_by_level]  # gas=1 (CO2)
            ax.plot(levels, vals, marker="s", label=modal)
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel("Weight for CO₂")
        ax.set_title(f"{kind} → CO₂ modality weights")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / f"{kind}_weights_CO2.png", dpi=150)
        plt.close()

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
```

### Step 4.3: `rcdw_mgda/tests/test_perturbation.py`

```python
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
```

---

## M5 — 退化硬抑制 + 归一化 + 评测

### Step 5.1: `rcdw_mgda/rcdw/utils/degradation.py`

```python
"""退化模态硬抑制（eval-only）。

框架第十一节：当某模态的中位误差 > ratio × 最小中位误差时，
将其权重压到 cap，然后重归一化。
"""
from __future__ import annotations

import torch


def hard_suppress(
    W: torch.Tensor,
    E_pred: torch.Tensor,
    *,
    ratio: float = 4.0,
    cap: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """退化硬抑制。

    Args:
        W:      (B, M=3, G=3) 融合权重
        E_pred: (B, M=3, G=3) 预测误差
        ratio:  退化判定倍率阈值
        cap:    退化模态最大权重
    Returns:
        W_suppressed: (B, M, G) 抑制后的权重
        degraded:     (M, G)    bool，标记哪些模态-气体对被抑制
    """
    # 中位误差: 对 batch 维求中位数 → (M, G)
    E_med = E_pred.median(dim=0).values  # (M, G)

    # 每种气体的最小中位误差 → (1, G)
    min_E = E_med.min(dim=0, keepdim=True).values  # (1, G)

    # 退化判定: median(E_m) > ratio * min(E_med)
    degraded = E_med > ratio * min_E  # (M, G) bool

    # 抑制: 退化模态权重压到 cap
    W_out = W.clone()
    degraded_3d = degraded.unsqueeze(0).expand_as(W)  # (B, M, G)
    W_out = torch.where(degraded_3d, W_out.clamp(max=cap), W_out)

    # 重归一化 (dim=1 = 模态维)
    W_out = W_out / W_out.sum(dim=1, keepdim=True).clamp(min=1e-6)

    return W_out, degraded
```

### Step 5.2: `rcdw_mgda/rcdw/utils/normalize.py`

```python
"""组分归一化与湿/干基换算。"""
from __future__ import annotations

import torch


def rh_to_water_vol(RH: torch.Tensor, T: torch.Tensor | None = None) -> torch.Tensor:
    """将相对湿度（0~1）近似转换为水汽体积分数。

    简化模型：假设 T=300K, P=1atm 下饱和水汽压 ~3.6 kPa。
    C_H2O ≈ RH * 3.6 / 101.325 ≈ RH * 0.0355
    """
    return RH * 0.0355


def normalize_composition(
    C: torch.Tensor,
    RH: torch.Tensor | None = None,
    *,
    basis: str = "dry",
) -> torch.Tensor:
    """对 O₂/CO₂/N₂ 浓度归一化。

    Args:
        C:     (B, 3) 或 (N, 3) 浓度
        RH:    (B,) 相对湿度（仅 basis="wet" 时使用）
        basis: "dry" → sum=1.0; "wet" → sum=1.0-C_H2O
    Returns:
        归一化后的浓度，与 C 同 shape
    """
    eps = 1e-6
    if basis == "wet" and RH is not None:
        C_total = 1.0 - rh_to_water_vol(RH)
        if C_total.dim() == 1:
            C_total = C_total.unsqueeze(-1)  # (B, 1)
    else:
        C_total = 1.0
    return C / (C.sum(dim=-1, keepdim=True) + eps) * C_total
```

### Step 5.3: `rcdw_mgda/scripts/eval.py`

```python
"""评测脚本：加载 checkpoint，在测试集上计算指标。

用法: cd rcdw_mgda && python -m scripts.eval --ckpt runs/stage_b/rcdw.pt --split test
"""
from __future__ import annotations

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from rcdw.data.synth import make_splits, WindowedDataset
from rcdw.models.rcdw import RCDW_MGDA
from rcdw.training.metrics import compute_per_gas_metrics
from rcdw.utils.degradation import hard_suppress
from rcdw.utils.normalize import normalize_composition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    splits = make_splits(
        n_train=dc["n_train"], n_val=dc["n_val"], n_test=dc["n_test"],
        L=dc["window"], seed=dc["seed"],
    )
    ds = WindowedDataset(*splits[args.split])
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
    model = RCDW_MGDA(W_base, hidden=cfg["model"]["single_modal"]["hidden"])
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    deg = cfg["degradation"]
    all_pred = []
    all_ref = []

    with torch.no_grad():
        for x_w, y in loader:
            out = model(x_w)
            W, _ = hard_suppress(out["W"], out["E_pred"],
                                 ratio=deg["ratio"], cap=deg["cap"])
            C = (W * out["Y_modal"]).sum(dim=1)
            C = normalize_composition(C, basis=cfg["eval"]["basis"])
            all_pred.append(C)
            all_ref.append(y)

    pred = torch.cat(all_pred)
    ref = torch.cat(all_ref)
    results = compute_per_gas_metrics(pred, ref)

    print(f"\n=== Evaluation on {args.split} set ({len(pred)} samples) ===\n")
    print(f"{'Gas':<10} {'MAE':>8} {'RMSE':>8} {'MRE%':>8} {'ARE%':>8}")
    print("-" * 44)
    for gas in ["O2", "CO2", "N2", "overall"]:
        m = results[gas]
        print(f"{gas:<10} {m['MAE']:8.5f} {m['RMSE']:8.5f} "
              f"{m['MRE']:8.2f} {m['ARE']:8.2f}")


if __name__ == "__main__":
    main()
```

### Step 5.4: 测试文件

`rcdw_mgda/tests/test_degradation.py`：

```python
"""测试退化硬抑制。"""
import torch
import pytest
from rcdw.utils.degradation import hard_suppress


def test_no_degradation():
    """所有模态误差相同时，不触发抑制。"""
    W = torch.ones(4, 3, 3) / 3
    E = torch.ones(4, 3, 3) * 0.05
    W_out, degraded = hard_suppress(W, E)
    assert not degraded.any()
    torch.testing.assert_close(W_out, W, atol=1e-5, rtol=1e-5)


def test_degradation_trigger():
    """模态 0 的误差 > 4x 最小值时触发抑制。"""
    W = torch.ones(10, 3, 3) / 3
    E = torch.ones(10, 3, 3) * 0.01
    E[:, 0, :] = 0.05  # 模态 0 误差 5x
    W_out, degraded = hard_suppress(W, E, ratio=4.0, cap=0.04)
    assert degraded[0, :].all()  # 模态 0 对所有气体退化
    assert (W_out[:, 0, :] <= 0.04 + 1e-5).all()


def test_renormalization():
    """抑制后权重应重归一化到 sum=1。"""
    W = torch.ones(8, 3, 3) / 3
    E = torch.ones(8, 3, 3) * 0.01
    E[:, 2, :] = 0.1  # 模态 2 退化
    W_out, _ = hard_suppress(W, E, ratio=4.0, cap=0.04)
    W_sum = W_out.sum(dim=1)
    torch.testing.assert_close(W_sum, torch.ones_like(W_sum), atol=1e-5, rtol=1e-5)


def test_cap_value():
    """退化模态的权重不应超过 cap。"""
    W = torch.ones(10, 3, 3) * 0.5
    E = torch.ones(10, 3, 3) * 0.01
    E[:, 1, 0] = 0.2  # 模态 1 对 O2 退化
    W_out, degraded = hard_suppress(W, E, ratio=4.0, cap=0.04)
    if degraded[1, 0]:
        assert (W_out[:, 1, 0] <= 0.04 + 1e-5).all()
```

---

## 完整验证流程

按照以下顺序执行，确认每步通过后再进入下一步：

```bash
# 0. 进入工作目录
cd rcdw_mgda

# 1. 导入检查
python -c "import rcdw; print('import OK')"

# 2. 全量测试
python -m pytest tests/ -v
# 预期: 全部通过 (8+4+6+3+8+6+4 = 约 35 个用例)

# 3. 数值对齐
python -m scripts.numerical_check
# 预期: ALL CHECKS PASSED

# 4. 两阶段训练
python -m scripts.train --config configs/default.yaml
# 预期:
#   Stage A: 3 个模态各自 loss 下降并早停
#   Stage B: 融合 MAE 持续下降

# 5. 评测
python -m scripts.eval --ckpt runs/stage_b/rcdw.pt --split test
# 预期: MAE / RMSE 指标输出

# 6. 扰动实验
python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt
# 预期:
#   runs/perturb/ 下生成 10 张 png (5 类 × 指标/权重 各 1 张)
#   光学扰动下 NDIR 权重单调下降
```

---

## 常见错误清单（实施者必读）

| # | 错误 | 原因 | 正确做法 |
|---|------|------|----------|
| 1 | `softmax(dim=-1)` | 在气体维度归一化，导致每个模态的三气体权重 sum=1（错误） | 必须 `softmax(dim=1)` 在模态维归一化 |
| 2 | `W_base` 行列写反 | 框架表行=气体/列=模态，代码行=模态/列=气体 | 按本文档的 W_base 精确复制 |
| 3 | Stage B 漏加载 Stage A checkpoint | 单模态权重随机，ErrorNet 无法学到有效误差 | 必须在 train.py 中加载 3 个 `.pt` |
| 4 | Stage B 忘记冻结单模态 | ErrorNet 的误差标签与单模态输出耦合，同时训练导致震荡 | `requires_grad = False` |
| 5 | FeatureExtractor 输入 `(B, 6)` | 所有滑窗统计特征退化为 0 | 必须输入 `(B, L=8, 6)` |
| 6 | 单模态用 softmax | 低浓度组分被压缩 | 用 `clamp(min=0) + L1-normalize` |
| 7 | `extract_modal_input` 索引错误 | 通道顺序 `[S_ndir=0, S_tc=1, S_us=2, P=3, T=4, RH=5]` | 严格按 `single_modal.py` 中的常量 |
| 8 | E_pred 允许负值 | 误差不应为负 | ErrorNet 末端用 `Softplus` |
| 9 | `hard_suppress` 在训练时调用 | 不可微，会中断梯度 | 仅在 eval/perturb 脚本中调用 |
| 10 | 扰动注入修改原始张量 | 测试集被污染 | `inject()` 内部必须 `x.clone()` |
| 11 | `RCDW_MGDA.usn` 写成 `us` | Stage A 存的是 `us.pt`，加载时 attr_name 映射不一致 | 参考 `train.py` 中的映射表 |
| 12 | 环境变量索引混淆 | `env` 来自 `x[:, :, 3:]`，所以 `env` 中 `P=0, T=1, RH=2` | FeatureExtractor 中 `delta_T = env[:,-1,1]-env[:,-2,1]` |
