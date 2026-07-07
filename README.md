# 多模态气体组分检测 v4

基于 NDIR 光学 + 超声波 + 光纤麦克风 + TCS 热导四模态传感器仿真信号，使用 DL/ML 预测混合气体各组分浓度。

## 仓库结构

四个气体检测场景各自独立为自包含子工程，另有共享数据目录：

| 目录 | 场景 | 包名 | 说明 |
|---|---|---|---|
| `hydrogen_ng/` | 掺氢天然气（H₂/CH₄/CO₂/N₂） | `hg` | 闭包 sum=100% |
| `syngas/` | 合成气 / 煤气化制气（H₂/CH₄/CO₂/CO） | `sg` | N₂ 背景，sum<100% |
| `tunnel_ventilation/` | 掘进通风（CO₂/O₂/N₂） | `tv3` | 闭包 sum=100% |
| `rcdw_mgda/` | 学长 RCDW 算法 | `rcdw` | 独立复现 |
| `shared/` | — | — | 共享光谱缓存（hitran_cache）+ 归档 |

每个子工程有独立 `pyproject.toml`、独立 CLI、独立 `tests/`，可独立 `pip install -e .[dev]` 并运行。原单仓库结构（`src/`/`docs/`/`configs/`/`scripts/`/`tests/`）已废弃删除。

## 安装

各子工程独立安装（以 hydrogen_ng 为例）：

```powershell
cd hydrogen_ng
pip install -e .[dev]
```

建议 Python 3.10–3.13（排除 3.14）。核心依赖：numpy、scipy、scikit-learn、torch、hitran-api。

## HITRAN 光谱缓存

共享光谱缓存在 `shared/hitran_cache/`（hg/sg 共用 CH₄/CO₂/H₂O/CO 谱线，约 2 万文件）。子工程运行 HITRAN 后端时，用 `--hitran-cache-root ../shared/hitran_cache` 指向共享缓存；hg 也支持环境变量 `HG_HITRAN_CACHE_ROOT`。tv3 默认 empirical 后端，不依赖 hitran_cache。

## 协作规则

- `CLAUDE.md` / `AGENTS.md`：AI 协作规则与边界
- 各子工程 `docs/`：场景专属文档与实验方案

## 场景重构历史

本仓库由原单仓库结构重构为四独立子工程，重构过程与决策见 `场景隔离重构计划.md`。
