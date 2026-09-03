# KKT-hPINN：硬线性物理约束神经网络精读笔记

## 1. 论文信息

- **题目**：Physics-Informed Neural Networks with Hard Linear Equality Constraints
- **作者**：Hao Chen, Gonzalo E. Constante-Flores, Canzhou Li
- **年份**：2024
- **期刊**：*Computers & Chemical Engineering*
- **卷**：189
- **文章号**：108764
- **Consensus 当前引用数**：74
- **论文链接**：https://consensus.app/papers/physicsinformed-neural-networks-with-hard-linear-chen-constante-flores/2479dc9f89d659d8919331e3a359561d/?utm_source=chatgpt

---

## 2. 一句话总结

KKT-hPINN 的核心思想是：

> 不再通过损失函数“鼓励”神经网络满足物理约束，而是在神经网络输出端加入一个由 KKT 条件推导得到的可微投影层，将任意预测严格投影到线性等式约束的可行域中。

因此它可以保证：

$$
A\mathbf{x}+B\mathbf{y}=\mathbf{b}
$$

在训练和推理阶段都严格成立。

---

## 3. 论文解决的问题

普通神经网络：

$$
\hat{\mathbf y}=f_\theta(\mathbf x)
$$

主要通过数据拟合损失训练：

$$
L_{\mathrm{data}}
=
\|\hat{\mathbf y}-\mathbf y\|_2^2
$$

但即使训练数据全部满足物理规律，预测结果仍然可能违反：

$$
A\mathbf x+B\hat{\mathbf y}=\mathbf b
$$

传统 PINN 常采用软约束：

$$
L
=
L_{\mathrm{data}}
+
\lambda L_{\mathrm{physics}}
$$

其中：

$$
L_{\mathrm{physics}}
=
\|A\mathbf x+B\hat{\mathbf y}-\mathbf b\|_2^2
$$

问题在于：

1. 只能尽量减少约束违反，不能保证严格满足；
2. 需要人为设置权重 $\lambda$；
3. 数据损失和物理损失可能存在梯度竞争；
4. $\lambda$ 过小则物理约束不起作用；
5. $\lambda$ 过大则可能损害数据拟合精度。

KKT-hPINN 的目标就是消除这一问题。

---

# 4. KKT-hPINN 的核心思想

设神经网络先输出：

$$
\hat{\mathbf y}=f_\theta(\mathbf x)
$$

这个预测可以不满足物理约束。

随后寻找距离 $\hat{\mathbf y}$ 最近，同时严格满足物理约束的预测：

$$
\tilde{\mathbf y}
=
\arg\min_{\mathbf y}
\frac{1}{2}
\|\mathbf y-\hat{\mathbf y}\|_2^2
$$

约束为：

$$
A\mathbf x+B\mathbf y=\mathbf b
$$

本质上就是：

> 将神经网络预测点正交投影到物理可行域。

---

# 5. KKT 条件推导

构造拉格朗日函数：

$$
\mathcal L(\mathbf y,\boldsymbol\lambda)
=
\frac12
(\mathbf y-\hat{\mathbf y})^T
(\mathbf y-\hat{\mathbf y})
+
\boldsymbol\lambda^T
(A\mathbf x+B\mathbf y-\mathbf b)
$$

对 $\mathbf y$ 求导：

$$
\nabla_{\mathbf y}\mathcal L
=
\mathbf y-\hat{\mathbf y}
+
B^T\boldsymbol\lambda
=
0
$$

所以：

$$
\mathbf y
=
\hat{\mathbf y}
-
B^T\boldsymbol\lambda
$$

代回约束：

$$
A\mathbf x
+
B
(
\hat{\mathbf y}
-
B^T\boldsymbol\lambda
)
=
\mathbf b
$$

得到：

$$
BB^T\boldsymbol\lambda
=
A\mathbf x+B\hat{\mathbf y}-\mathbf b
$$

如果 $B$ 满行秩，则：

$$
\boldsymbol\lambda
=
(BB^T)^{-1}
(
A\mathbf x+B\hat{\mathbf y}-\mathbf b
)
$$

最终：

$$
\boxed{
\tilde{\mathbf y}
=
\hat{\mathbf y}
-
B^T(BB^T)^{-1}
(
A\mathbf x+B\hat{\mathbf y}-\mathbf b
)
}
$$

这就是 KKT-hPINN 最核心的公式。

---

# 6. 将 KKT 投影写成神经网络层

展开：

$$
\tilde{\mathbf y}
=
-B^T(BB^T)^{-1}A\mathbf x
+
\left[
I-B^T(BB^T)^{-1}B
\right]\hat{\mathbf y}
+
B^T(BB^T)^{-1}\mathbf b
$$

定义：

$$
A^*
=
-B^T(BB^T)^{-1}A
$$

$$
B^*
=
I-B^T(BB^T)^{-1}B
$$

$$
b^*
=
B^T(BB^T)^{-1}\mathbf b
$$

则：

$$
\boxed{
\tilde{\mathbf y}
=
A^*\mathbf x+B^*\hat{\mathbf y}+b^*
}
$$

其中 $A^*$、$B^*$ 和 $b^*$ 都可以提前计算。

因此整个模型可以写成：

```text
输入 x
  │
  ▼
任意深度学习主干
CNN / TCN / LSTM / Transformer
  │
  ▼
原始预测 ŷ
  │
  ▼
KKT Projection Layer
  │
  ▼
约束后预测 ỹ
```

KKT 投影层本身是线性、确定且可微的，因此可以参与端到端反向传播。

---

# 7. 为什么不能只在预测之后做投影

一种简单方法是：

```text
训练：
NN → ŷ → Loss

推理：
NN → ŷ → 物理投影 → ỹ
```

这种方法只是后处理。

KKT-hPINN 则是：

```text
训练：
NN → ŷ → KKT投影 → ỹ → Loss

推理：
NN → ŷ → KKT投影 → ỹ
```

训练目标实际上变成：

$$
L
=
\|
A^*\mathbf x
+
B^*f_\theta(\mathbf x)
+
b^*
-
\mathbf y
\|_2^2
$$

因此投影层直接影响梯度：

$$
\frac{\partial L}{\partial \theta}
$$

网络会主动学习：

> 什么样的原始表示经过物理投影以后可以得到更准确的最终结果。

所以“训练中硬约束”与“预测后修正”并不等价。

---

# 8. 应用于四组分气体浓度回归

假设预测四组分：

$$
\mathbf y
=
[
x_{H_2},
x_{CH_4},
x_{CO_2},
x_{N_2}
]^T
$$

要求：

$$
x_{H_2}
+
x_{CH_4}
+
x_{CO_2}
+
x_{N_2}
=
1
$$

则：

$$
A=0
$$

$$
B=
\begin{bmatrix}
1&1&1&1
\end{bmatrix}
$$

$$
b=1
$$

有：

$$
BB^T=4
$$

所以：

$$
(BB^T)^{-1}
=
\frac14
$$

得到：

$$
\boxed{
\tilde{\mathbf y}
=
\hat{\mathbf y}
+
\frac{
1-\sum_{i=1}^{4}\hat y_i
}{4}
\begin{bmatrix}
1\\
1\\
1\\
1
\end{bmatrix}
}
$$

例如原始预测：

$$
\hat{\mathbf y}
=
[0.20,0.30,0.25,0.21]
$$

总和：

$$
0.96
$$

缺少：

$$
1-0.96=0.04
$$

因此每个组分增加：

$$
0.04/4=0.01
$$

得到：

$$
\tilde{\mathbf y}
=
[0.21,0.31,0.26,0.22]
$$

严格满足：

$$
\sum_i\tilde y_i=1
$$

---

# 9. KKT-hPINN 的一个重要局限

原始 KKT-hPINN 解决的是：

$$
A\mathbf x+B\mathbf y=\mathbf b
$$

这种**线性等式约束**。

它不能直接保证：

$$
y_i\ge0
$$

例如：

$$
\hat{\mathbf y}
=
[-0.08,0.40,0.40,0.40]
$$

虽然经过 KKT 投影后总和可以严格变成 1，但第一个组分仍可能为负。

因此：

> KKT-hPINN 可以保证“组分总和正确”，但不能自动保证“每个组分浓度非负”。

这是迁移到气体组分预测时必须考虑的问题。

---

# 10. KKT 与 Softmax 的比较

如果目标只有：

$$
y_i\ge0
$$

以及：

$$
\sum_i y_i=1
$$

Softmax 输出：

$$
\hat y_i
=
\frac{\exp(z_i)}
{\sum_j\exp(z_j)}
$$

天然保证：

$$
\hat y_i>0
$$

以及：

$$
\sum_i\hat y_i=1
$$

因此对于单纯的四组分闭合问题：

> Softmax 往往比 KKT 更简单。

但是 KKT 的优势在于能够处理更加一般的线性守恒关系，例如：

$$
A\mathbf x+B\mathbf y=\mathbf b
$$

其中约束同时涉及输入和多个输出。

例如质量守恒：

$$
F_{\mathrm{in}}
=
\sum_iF_{\mathrm{out},i}
$$

或者：

$$
F_{\mathrm{in}}x_i^{in}
=
F_{\mathrm{out}}x_i^{out}
$$

此时 Softmax 无法表达，而 KKT 投影层可以一次处理多个线性约束。

---

# 11. KKT-hPINN 与软 PINN 的本质区别

### 软 PINN

$$
L
=
L_{\mathrm{data}}
+
\lambda L_{\mathrm{physics}}
$$

物理约束只是优化目标之一。

因此：

$$
L_{\mathrm{physics}}\approx0
$$

并不意味着：

$$
L_{\mathrm{physics}}=0
$$

### KKT-hPINN

网络结构本身保证：

$$
A\mathbf x+B\tilde{\mathbf y}=\mathbf b
$$

因此不需要通过额外损失权重来逼近这个约束。

核心思想：

$$
\boxed{
\text{能通过网络结构严格满足的物理规律，不必全部写进 Loss}
}
$$

---

# 12. 论文实验结论

论文在多个 Aspen 化工过程模型中比较：

- 普通神经网络 NN；
- 后处理投影 NNPost；
- 软约束 PINN；
- KKT-hPINN。

总体趋势表明：

1. KKT-hPINN 可以将线性约束误差降低到接近机器精度；
2. 普通 NN 即使预测误差较低，也可能明显违反守恒规律；
3. 仅在推理阶段增加投影通常不如训练阶段直接加入投影层；
4. 软 PINN 在部分实验中可能因为损失竞争而降低预测精度；
5. 硬约束能够缩小模型的有效搜索空间，在有限训练数据下具有改善泛化的潜力。

论文在 DME-DEE 整体流程案例中，KKT-hPINN 的整体测试 RMSE 优于普通 NN、NNPost 和软 PINN。

---

# 13. 对多模态气体检测课题的迁移价值

对于“超声 + NDIR + 热导率 + 环境变量”的多模态气体定量模型，可以将网络写成：

```text
超声原始波形
    ↓
1D-CNN Encoder
    │
    ├──────────────┐
    │              │
NDIR Encoder   TCD Encoder
    │              │
    └──────┬───────┘
           ↓
      多模态融合
           ↓
      TCN / Transformer
           ↓
       回归预测头
           ↓
      Softmax / KKT
           ↓
     多组分气体浓度
```

前面的多模态编码与时序建模结构基本不需要因为 KKT 而改变。

---

# 14. 推荐的物理约束分类

不建议把所有物理知识全部作为同一种 PINN Loss。

应区分两种物理知识。

## 14.1 精确物理约束

例如：

$$
\sum_i x_i=1
$$

以及：

$$
x_i\ge0
$$

属于必须严格满足的物理边界。

建议使用：

- Softmax；
- KKT 投影；
- 更一般的硬约束层。

## 14.2 近似物理关系

例如超声声速：

$$
c=f(\mathbf x,T,P)
$$

超声飞行时间：

$$
t_{\mathrm{TOF}}
=
\frac{L}{c}
$$

NDIR Beer-Lambert 关系：

$$
I
=
I_0
\exp
\left(
-\sum_i
\epsilon_i(\lambda)c_iL
\right)
$$

热导率：

$$
k_{\mathrm{mix}}
=
f(\mathbf x,T,P)
$$

实际传感器会受到噪声、非理想响应、温漂和器件误差，因此不宜要求神经网络绝对满足这些理想模型。

更加合理的是使用辅助物理损失：

$$
L
=
L_{\mathrm{conc}}
+
\lambda_{US}L_{US}
+
\lambda_{NDIR}L_{NDIR}
+
\lambda_{TCD}L_{TCD}
$$

因此推荐总体思想：

$$
\boxed{
\text{硬物理约束}
+
\text{软物理辅助监督}
}
$$

---

# 15. 推荐的模型结构

```text
US waveform ─→ 1D-CNN ─┐
                       │
NDIR ───────→ Encoder ─┼─→ Fusion
                       │
TCD ────────→ Encoder ─┤
                       │
T / P / H ──→ Encoder ─┘
                          │
                          ↓
                    TCN / Transformer
                          │
                  ┌───────┴────────┐
                  │                │
             物理辅助头        浓度预测头
                  │                │
      TOF / attenuation /       Softmax
      NDIR / conductivity      或 KKT
                  │                │
                  ↓                ↓
              辅助损失         四组分浓度
```

---

# 16. 推荐消融实验

至少比较以下三种输出方式：

### Model A：无硬约束

$$
\hat{\mathbf y}
=
f_\theta(\mathbf x)
$$

### Model B：Softmax 硬闭合

$$
\hat y_i
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

### Model C：KKT-hPINN

$$
\tilde{\mathbf y}
=
\hat{\mathbf y}
-
B^T(BB^T)^{-1}
(
B\hat{\mathbf y}-b
)
$$

评价指标除了：

- MAE
- RMSE
- $R^2$

还应增加：

### 闭合误差

$$
E_{\mathrm{closure}}
=
\left|
\sum_i\hat y_i-1
\right|
$$

### 负浓度比例

$$
R_{\mathrm{negative}}
=
\frac{
N(\hat y_i<0)
}{
N_{\mathrm{prediction}}
}
$$

### 极端组分区间误差

重点分析低浓度和高浓度区间是否存在：

$$
\text{向均值收缩}
$$

---

# 17. 对课题的最终判断

KKT-hPINN 对本课题最大的启发并不是简单地“采用 KKT”。

更重要的是建立以下设计原则：

$$
\boxed{
\text{物理知识应区分为可严格满足的约束和只能近似满足的约束}
}
$$

具体而言：

| 物理知识 | 类型 | 推荐方式 |
|---|---|---|
| 多组分总和为 100% | 精确线性约束 | Softmax / KKT |
| 浓度非负 | 精确不等式约束 | Softmax / Hard Constraint |
| 超声 TOF 与声速关系 | 近似物理模型 | 辅助损失 |
| 超声衰减关系 | 近似物理模型 | 辅助损失 |
| NDIR Beer-Lambert 关系 | 近似物理模型 | 辅助损失 |
| 热导率与组分关系 | 近似物理模型 | 辅助损失 |
| 动态过程连续性 | 过程先验 | 时序正则 |
| 传感器质量变化 | 数据质量问题 | 质量感知融合 |

---

# 18. 推荐使用方式

对于当前四组分浓度回归，建议优先采用：

$$
\boxed{
\text{多模态融合网络}
+
\text{Softmax 硬闭合}
+
\text{超声/NDIR/TCD 辅助物理损失}
}
$$

随后将 KKT-hPINN 作为重要对照实验。

如果未来模型加入：

- 流量；
- 入口与出口组成；
- 多个质量平衡方程；
- 反应前后组分；
- 多设备耦合守恒；

则 KKT-hPINN 的优势会明显高于单纯 Softmax。

---

# 19. 核心参考文献

[1] Hao Chen, Gonzalo E. Constante-Flores, Canzhou Li. **Physics-Informed Neural Networks with Hard Linear Equality Constraints**. *Computers & Chemical Engineering*, 2024, 189:108764.  
Consensus: https://consensus.app/papers/physicsinformed-neural-networks-with-hard-linear-chen-constante-flores/2479dc9f89d659d8919331e3a359561d/?utm_source=chatgpt
