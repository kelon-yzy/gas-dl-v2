# TikUDA 文献精读整理

## 1. 论文信息

**论文题目：** *Efficient Unsupervised Domain Adaptation Regression for Spatial–Temporal Sensor Fusion*

**作者：** Keivan Faghih Niresi, Ismail Nejjar, Olga Fink

**年份：** 2024 预印本，后正式发表于 2025 年

**期刊：** IEEE Internet of Things Journal

**研究方向：**
- 无监督域适应（Unsupervised Domain Adaptation, UDA）
- 连续值回归
- 时空传感器融合
- 传感器漂移与跨部署泛化

**核心方法：** TikUDA

**代码：** 作者公开了 TikUDA 官方实现。

---

# 2. 论文解决的问题

传统监督学习通常假设训练域与测试域分布近似一致：

$$
P_{\mathrm{train}}(X,Y)
\approx
P_{\mathrm{test}}(X,Y)
$$

但实际工业传感器部署后经常存在明显域偏移：

$$
P_S(X,Y)\neq P_T(X,Y)
$$

典型原因包括：

- 传感器老化和漂移；
- 不同批次传感器灵敏度存在差异；
- 环境温度、湿度改变；
- 不同安装位置；
- 不同装置或不同部署地点；
- 校准条件与实际运行条件不同。

TikUDA 研究的基本任务是：

源域具有完整标签：

$$
D_S=\{X_S,Y_S\}
$$

目标域只有输入：

$$
D_T=\{X_T\}
$$

但没有：

$$
Y_T
$$

目标是在没有目标域标签的情况下，使模型能够完成：

$$
f(X_T)\rightarrow Y_T
$$

因此属于：

> **面向连续值回归任务的无监督域适应。**

---

# 3. TikUDA 的核心思想

模型可以写成：

$$
X
\xrightarrow{h_\theta}
Z
\xrightarrow{g_\phi}
\hat Y
$$

其中：

- $h_\theta$：特征提取器；
- $Z$：潜在特征；
- $g_\phi$：回归器。

普通域适应通常直接缩小：

$$
Z_S
$$

与：

$$
Z_T
$$

之间的分布差异，例如：

- MMD；
- CORAL；
- DANN。

TikUDA 则从**回归器几何结构**出发进行域适应。

考虑线性回归：

$$
Y=Zw
$$

最小二乘解为：

$$
w=
(Z^TZ)^{-1}Z^TY
$$

定义 Gram 矩阵：

$$
G=Z^TZ
$$

因此回归映射与：

$$
G^{-1}
$$

密切相关。

TikUDA 的核心思想是：

> 与其直接要求源域和目标域特征完全一致，不如让两者具有相似的回归几何结构。

即希望：

$$
G_S^{-1}
\approx
G_T^{-1}
$$

---

# 4. 为什么引入 Tikhonov 正则化

神经网络训练中通常：

$$
b<p
$$

其中：

- $b$：Batch Size；
- $p$：隐藏特征维数。

若：

$$
Z\in\mathbb R^{b\times p}
$$

则：

$$
G=Z^TZ
$$

可能是奇异矩阵。

TikUDA 使用：

$$
G_\alpha
=
Z^TZ+\alpha I
$$

其中：

$$
\alpha>0
$$

通过 Tikhonov 正则化改善矩阵的数值稳定性。

因此：

$$
G_S
=
Z_S^TZ_S+\alpha I
$$

$$
G_T
=
Z_T^TZ_T+\alpha I
$$

然后利用 Cholesky 分解高效求逆。

与依赖 SVD 的方法相比，该方案具有更好的计算效率和可扩展性。

---

# 5. TikUDA 的两个核心域适应项

TikUDA 将域偏移拆分为：

$$
\boxed{
\text{方向差异}
+
\text{尺度差异}
}
$$

---

## 5.1 方向对齐

计算：

$$
G_S^{-1}
$$

与：

$$
G_T^{-1}
$$

将每一列视为一个向量：

$$
g_{S,i}^{-1}
$$

和：

$$
g_{T,i}^{-1}
$$

计算两者夹角：

$$
\cos\phi_i
=
\frac{
g_{S,i}^{-1}\cdot g_{T,i}^{-1}
}{
\|g_{S,i}^{-1}\|
\|g_{T,i}^{-1}\|
}
$$

TikUDA 没有直接采用普通余弦距离，而是使用 Haversine-style similarity：

$$
HS_i
=
1-
\sqrt{
\frac{1-\cos\phi_i}{2}
}
$$

最终定义：

$$
L_{\mathrm{angle}}
=
\|\mathbf 1-\mathbf{HS}\|_1
$$

目标为：

$$
L_{\mathrm{angle}}\rightarrow0
$$

即：

$$
\phi_i\rightarrow0
$$

使源域和目标域的回归方向保持一致。

---

# 6. 尺度对齐

仅仅方向一致并不足以保证回归性能。

例如：

源域：

$$
x_S=[1,2,3,4]
$$

目标域：

$$
x_T=[2,4,6,8]
$$

两者方向完全一致：

$$
\cos\phi=1
$$

但绝对尺度不同。

对于分类任务，这种尺度变化可能影响较小。

但对于浓度回归：

$$
10\%\neq20\%
$$

尺度信息十分关键。

TikUDA 使用最大主特征值衡量源域和目标域主要尺度：

$$
\lambda_{\max}^S
$$

和：

$$
\lambda_{\max}^T
$$

定义尺度损失：

$$
L_{\mathrm{scale}}
=
\left(
\lambda_{\max}^S-
\lambda_{\max}^T
\right)^2
$$

最大特征值通过幂迭代 Power Iteration 近似求得，而不是完整特征值分解。

---

# 7. TikUDA 总损失函数

源域监督回归损失：

$$
L_{\mathrm{src}}
=
\operatorname{MSE}
(
\hat Y_S,Y_S
)
$$

TikUDA 总损失：

$$
L_{\mathrm{total}}
=
L_{\mathrm{src}}
+
\gamma_{\mathrm{angle}}
L_{\mathrm{angle}}
+
\gamma_{\mathrm{scale}}
L_{\mathrm{scale}}
$$

注意：

目标域没有标签：

$$
Y_T
$$

不会参与：

$$
L_{\mathrm{src}}
$$

目标域数据只参与：

$$
L_{\mathrm{angle}}
$$

和：

$$
L_{\mathrm{scale}}
$$

因此模型可以利用**无标签目标域数据完成域适应**。

---

# 8. 动态域适应权重

TikUDA 不会在训练开始时立即使用较强的域适应约束。

定义：

$$
\lambda(p)
=
\frac{2}{1+\exp(-10p)}-1
$$

其中：

$$
p\in[0,1]
$$

表示训练进度。

随着训练进行：

$$
\lambda
$$

逐渐增大。

论文中类似采用：

$$
\gamma_{\mathrm{angle}}
=
10^{-2}\lambda
$$

$$
\gamma_{\mathrm{scale}}
=
10^{-3}\lambda
$$

因此训练初期主要优化：

$$
X_S\rightarrow Y_S
$$

等模型已经学习到较稳定的基础表示后，再逐渐增强：

$$
Z_S\leftrightarrow Z_T
$$

的域对齐。

这一设计可以避免训练初期随机特征被强行对齐。

---

# 9. 原论文网络结构

论文采用时空图神经网络作为基础回归网络。

总体结构：

$$
\boxed{
Linear
\rightarrow
GRU
\rightarrow
GAT
\rightarrow
Flatten
\rightarrow
Regression
}
$$

---

## 9.1 传感器编码

不同传感器首先通过线性层编码：

$$
x_i\rightarrow h_i
$$

同时为每种传感器增加可学习 Sensor Embedding：

$$
e_i
$$

使网络能够区分不同传感器节点。

---

## 9.2 时间建模

采用 GRU：

$$
H_t
=
GRU(H_{t-1},X_t)
$$

提取传感器的动态时间特征。

---

## 9.3 空间传感器关系建模

使用 Graph Attention Network：

$$
h'_i
=
\sum_{j\in N(i)}
\alpha_{ij}Wh_j
$$

其中：

$$
\alpha_{ij}
$$

表示不同传感器之间的信息关联权重。

如果不存在明确的物理拓扑，则可建立完全连接图。

---

# 10. 原论文主要超参数

论文模型整体规模并不大：

- Linear Hidden Dimension：16
- GRU：4 层
- GAT：1 层
- Hidden Dimension：16
- 回归头：单层全连接
- Optimizer：Adam
- Learning Rate：

$$
3\times10^{-4}
$$

- Batch Size：

$$
64
$$

- Epoch：

$$
150
$$

- 时间窗口：

$$
16
$$

- 时间步长：

$$
1
$$

输入数据采用 Min-Max 标准化到：

$$
[0,1]
$$

需要注意：

> 标准化参数只使用源域数据拟合，然后应用到目标域。

这样可以减少利用目标域统计信息产生的数据泄漏。

---

# 11. 论文实验任务

TikUDA 在两类任务上验证：

## 11.1 空气质量传感器迁移

不同部署地点之间进行：

- O$_3$ 浓度回归；
- NO$_2$ 浓度回归。

源站点具有标签。

目标站点不提供标签参与训练。

---

## 11.2 EEG 信号重建

不同受试者之间进行跨域 EEG 回归。

说明 TikUDA 并不限定于气体传感器，而是一种通用连续值域适应方法。

---

# 12. 主要实验结果

以典型 O$_3$ 跨站点迁移为例。

源模型直接迁移：

$$
RMSE=0.255
$$

TikUDA：

$$
RMSE=0.087
$$

另一迁移方向：

$$
0.217
\rightarrow
0.097
$$

不同算法平均 RMSE 结果大致为：

| 方法 | 平均 RMSE |
|---|---:|
| Source-only | 0.193 |
| MMD | 0.130 |
| CORAL | 0.161 |
| ERM-NU | 0.139 |
| AdaGCN | 0.116 |
| DARE-GRAM | 0.107 |
| **TikUDA** | **0.105** |

说明针对回归问题设计专门域适应方式，能够明显优于直接使用通用分布对齐方法。

---

# 13. 消融实验

完整 TikUDA：

$$
L_{\mathrm{angle}}
+
L_{\mathrm{scale}}
$$

典型结果：

| 方法 | R-212 $\rightarrow$ R-69 | R-69 $\rightarrow$ R-212 |
|---|---:|---:|
| 无域适应 | 0.255 | 0.217 |
| 只有尺度对齐 | 0.096 | 0.106 |
| 只有方向对齐 | 0.202 | 0.159 |
| **方向 + 尺度** | **0.087** | **0.099** |

其中最值得关注的结论是：

$$
\boxed{
L_{\mathrm{scale}}
>
L_{\mathrm{angle}}
}
$$

即尺度对齐对回归性能的贡献明显更大。

---

# 14. 为什么尺度对齐特别适合气体传感器

不同设备之间经常出现：

$$
x_T
=
ax_S+b+\epsilon
$$

其中：

- $a$：灵敏度变化；
- $b$：零点偏移；
- $\epsilon$：随机噪声。

尤其是：

$$
a
$$

变化，本质上就是传感器响应尺度发生改变。

例如：

- 热导传感器灵敏度不同；
- NDIR 光源衰减；
- 超声增益改变；
- 信号放大器参数变化。

因此 TikUDA 中：

$$
L_{\mathrm{scale}}
$$

与气体传感器跨设备迁移问题具有非常明确的物理对应关系。

---

# 15. PCA 域对齐分析

论文还对隐藏表示进行了 PCA 分析。

源域和目标域之间的 Energy Distance：

无域适应：

$$
0.3877
$$

AdaGCN：

$$
0.0790
$$

DARE-GRAM：

$$
0.0128
$$

TikUDA：

$$
0.0031
$$

说明 TikUDA 能显著提高源域与目标域潜在表示的一致性。

后续气体检测跨设备实验也建议加入：

$$
PCA(Z_S,Z_T)
$$

以及：

$$
MMD
$$

或：

$$
Energy\ Distance
$$

用于可视化和定量验证域适应效果。

---

# 16. 对多组分气体检测课题的迁移方案

定义源域设备 A：

$$
D_S=
\{
X_{\mathrm{US}}^A,
X_{\mathrm{TCD}}^A,
X_{\mathrm{NDIR}}^A,
X_{\mathrm{env}}^A,
Y^A
\}
$$

其中：

$$
Y^A=
[
C_1,C_2,\ldots,C_K
]
$$

目标域设备 B：

$$
D_T=
\{
X_{\mathrm{US}}^B,
X_{\mathrm{TCD}}^B,
X_{\mathrm{NDIR}}^B,
X_{\mathrm{env}}^B
\}
$$

目标域没有浓度标签：

$$
Y^B=\varnothing
$$

---

# 17. 推荐气体检测结构

建议先不要直接复现原论文 GAT，而将 TikUDA 作为独立损失模块加入已有网络。

结构：

```text
超声原始波形
    ↓
1D-CNN
    ┐
热导率
    ↓
MLP / 1D-CNN
    ├──→ 多模态融合 → TCN → Latent Feature Z → 浓度回归
NDIR
    ↓
MLP / 1D-CNN
    ┘
```

源域产生：

$$
Z_S
$$

目标域产生：

$$
Z_T
$$

然后加入：

$$
L_{\mathrm{TikUDA}}
=
\gamma_aL_{\mathrm{angle}}
+
\gamma_sL_{\mathrm{scale}}
$$

最终：

$$
L=
L_{\mathrm{gas}}
+
L_{\mathrm{TikUDA}}
$$

其中：

$$
L_{\mathrm{gas}}
=
\frac1K
\sum_{k=1}^{K}
(\hat C_k-C_k)^2
$$

---

# 18. TikUDA 与现有模型的兼容性

TikUDA 本质上是作用于隐藏特征上的域适应损失，因此可以和多种模型组合：

- 1D-CNN；
- TCN；
- LSTM；
- GRU；
- Transformer；
- GMU；
- QMF；
- DMAW；
- MoNIG。

因此不需要重新设计整个网络。

可以直接执行：

$$
\boxed{
Baseline
+
TikUDA
}
$$

进行跨设备验证。

---

# 19. TikUDA 的重要局限

## 19.1 仍然要求源域和目标域结构一致

这是最重要的问题。

原始 TikUDA 更适合：

$$
\{S_1,S_2,S_3,S_4\}
\rightarrow
\{S'_1,S'_2,S'_3,S'_4\}
$$

即：

- 传感器数量相同；
- 节点结构相同；
- 只是响应分布变化。

它并不能直接处理：

$$
\{US,TCD,NDIR\}
\rightarrow
\{US,TCD\}
$$

或者：

$$
\{US,TCD,NDIR_A\}
\rightarrow
\{US',TCD',NDIR_B,MOX\}
$$

因此 TikUDA 本质上解决的是：

$$
\boxed{
\text{跨域}
}
$$

而不是完整意义上的：

$$
\boxed{
\text{跨传感体系结构}
}
$$

---

# 20. 针对该局限的可创新方向

可以首先将任意传感器编码成统一维度：

$$
z_i
=
E_i(x_i)
\in\mathbb R^d
$$

所有传感器最终进入统一潜空间：

$$
Z_{\mathrm{common}}
\in\mathbb R^{N\times d}
$$

这样：

源域：

$$
Z_S^{common}
$$

目标域：

$$
Z_T^{common}
$$

即使原始传感器结构不同，也可以在统一 latent space 中执行 TikUDA。

最终形成：

$$
\boxed{
\text{Heterogeneous Sensor Encoder}
+
\text{Configurable Fusion}
+
\text{TikUDA}
}
$$

核心思想是：

> 先解决传感器结构不同，再解决数据分布不同。

这一改进比单纯复现 TikUDA 更适合作为论文创新。

---

# 21. TikUDA 与 QMF、DMAW、M3AE 的关系

这些方法解决的问题并不重复。

| 方法 | 主要解决问题 |
|---|---|
| QMF | 当前样本不同模态质量不同 |
| MoNIG | 模态预测不确定性 |
| DMAW | 模态可靠性动态变化 |
| M3AE | 模态缺失 |
| TikUDA | 训练设备与部署设备存在域偏移 |

可以建立三级鲁棒框架：

## 样本级质量变化

$$
QMF/DMAW
$$

## 模态级缺失

$$
M3AE/Modality\ Dropout
$$

## 设备级分布变化

$$
TikUDA
$$

因此最终可以形成：

$$
\boxed{
\text{质量感知}
+
\text{缺失模态鲁棒}
+
\text{跨设备域适应}
}
$$

三层鲁棒学习体系。

---

# 22. 推荐的复现实验路线

## Experiment 1：Baseline

$$
1D\text{-}CNN
+
MLP
+
TCN
$$

设备 A 训练：

$$
D_A
$$

设备 B 直接测试：

$$
D_B
$$

记录：

- RMSE；
- MAE；
- $R^2$；
- 各组分误差。

---

## Experiment 2：Baseline + TikUDA

$$
Baseline
+
L_{\mathrm{angle}}
+
L_{\mathrm{scale}}
$$

验证跨设备性能是否提升。

---

## Experiment 3：TikUDA 消融

分别比较：

$$
L_{\mathrm{scale}}
$$

$$
L_{\mathrm{angle}}
$$

$$
L_{\mathrm{scale}}
+
L_{\mathrm{angle}}
$$

重点验证：

> 气体传感器跨设备迁移中，尺度对齐是否仍然占主导作用。

---

## Experiment 4：域偏移强度实验

人为模拟：

$$
x'=ax+b+\epsilon
$$

逐步增加：

- 增益变化；
- 零点漂移；
- Gaussian Noise；
- 温漂；
- 灵敏度衰减。

观察：

$$
RMSE
$$

随域偏移强度变化。

---

## Experiment 5：TikUDA + QMF

验证：

$$
\text{设备域偏移}
+
\text{单样本质量退化}
$$

同时存在时的鲁棒性。

---

## Experiment 6：TikUDA + Modality Dropout

进一步模拟：

- NDIR 缺失；
- 热导缺失；
- 超声缺失。

验证 TikUDA 在结构不完整时是否仍然有效。

---

# 23. 最值得借鉴的三点

## 第一：回归专用域适应

TikUDA 并不是直接照搬分类领域的 DANN/MMD，而是从：

$$
w=
(Z^TZ)^{-1}Z^TY
$$

出发设计回归域对齐机制。

这一点非常适合浓度预测。

---

## 第二：尺度对齐

对于工业传感器：

$$
x_T=ax_S+b
$$

极其常见。

因此：

$$
L_{\mathrm{scale}}
$$

具有很强的传感器物理解释性。

---

## 第三：可作为插件式损失加入现有架构

TikUDA 不要求使用某一种特定 Backbone。

因此可以直接加入现有：

$$
1D\text{-}CNN+TCN
$$

架构，不需要首先更换整个网络。

---

# 24. 论文局限总结

TikUDA 的主要不足包括：

1. 源域和目标域传感器结构基本需要一致；
2. 不能直接解决模态缺失；
3. 没有显式判断不同模态的质量；
4. 没有判断到底是哪一个传感器发生漂移；
5. 需要访问无标签目标域训练数据；
6. 主要解决分布偏移，而不是传感器集合变化；
7. Gram 矩阵计算随隐藏维数增加仍会增加计算成本；
8. 对跨传感器类型泛化仍需要额外结构设计。

---

# 25. 对课题的最终评价

TikUDA 与本课题匹配度：

$$
\boxed{\text{★★★★★}}
$$

尤其适合研究：

> 已在实验装置 A 上完成大量标定后，如何将模型迁移到新设备 B，而不重新采集完整浓度标定数据。

最推荐作为：

$$
\boxed{
\text{跨设备气体浓度回归模块}
}
$$

而不是直接承担：

$$
\boxed{
\text{可配置跨传感体系模块}
}
$$

后者应进一步与：

- 异构传感器编码；
- Sensor Token；
- Graph Fusion；
- Missing Modality Learning；

等方法结合。

---

# 26. 推荐后续改进方向

最有价值的组合是：

$$
\boxed{
\text{异构编码}
+
\text{TikUDA}
+
\text{质量感知融合}
}
$$

进一步发展：

$$
\boxed{
\text{异构编码}
+
TikUDA
+
QMF/DMAW
+
Modality\ Dropout
}
$$

形成同时处理：

- 模态结构差异；
- 传感器质量退化；
- 传感器缺失；
- 设备分布偏移；

的统一多模态气体检测框架。

---

# 27. 一句话总结

> TikUDA 的核心不是普通“特征分布对齐”，而是通过 Tikhonov 正则化 Gram 矩阵，在无目标域标签的情况下同时对齐源域与目标域回归表示的方向和尺度，使模型能够更稳定地迁移到发生传感器漂移、灵敏度变化和部署环境变化的新设备，是目前非常适合多组分气体浓度跨设备回归的无监督域适应方法。
