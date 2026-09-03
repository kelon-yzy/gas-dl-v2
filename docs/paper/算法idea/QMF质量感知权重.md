# Zhang et al., 2023 — QMF 论文精读整理

## 1. 论文信息

**论文题目：** *Provable Dynamic Fusion for Low-Quality Multimodal Data*  
**作者：** Qingyang Zhang, Haitao Wu, Changqing Zhang, Qinghua Hu, Huazhu Fu, Joey Tianyi Zhou, Xi Peng  
**年份：** 2023  
**会议：** ICML 2023  
**核心方法：** Quality-aware Multimodal Fusion（QMF，质量感知多模态融合）

### 一句话总结

QMF 的核心不是设计复杂的融合网络，而是从泛化理论出发提出：

> **当某个模态越容易产生预测错误时，其融合权重就应该越低。**

因此，通过估计每个模态的预测不确定度，并让模态权重与实际预测误差保持负相关，可以提升低质量、多噪声多模态数据下的融合鲁棒性。

---

# 2. QMF 要解决的问题

传统多模态融合通常默认所有模态始终可靠，例如：

$$
z=[z_1,z_2,\dots,z_M]
$$

或者采用固定晚期融合：

$$
\hat y=\sum_{m=1}^{M}w_m\hat y_m
$$

其中：

$$
w_m=\mathrm{constant}
$$

但在实际工业场景中，不同传感器质量会随时间和样本变化。

例如气体检测系统中可能出现：

- 超声回波信噪比下降；
- NDIR 光源衰减或饱和；
- 热导传感器温漂；
- 环境传感器噪声；
- 某一模态部分或完全失效。

如果仍然直接执行：

$$
[z_{\mathrm{US}},z_{\mathrm{TCD}},z_{\mathrm{NDIR}}]
$$

坏模态中的错误信息会继续进入后续融合网络。

QMF 的基本思想是：

$$
\boxed{
\text{估计模态质量}
\rightarrow
\text{动态调整模态权重}
\rightarrow
\text{抑制低质量模态}
}
$$

---

# 3. 动态融合基本形式

对于第 $m$ 个模态：

$$
f^m(x^m)
$$

最终预测为：

$$
f(x)
=
\sum_{m=1}^{M}
w^m(x)f^m(x^m)
$$

其中：

$$
w^m=w^m(x)
$$

即融合权重根据当前样本动态变化。

例如：

$$
w_{\mathrm{US}}=0.7,\quad
w_{\mathrm{TCD}}=0.2,\quad
w_{\mathrm{NDIR}}=0.1
$$

在另一个样本中可能变为：

$$
w_{\mathrm{US}}=0.1,\quad
w_{\mathrm{TCD}}=0.3,\quad
w_{\mathrm{NDIR}}=0.6
$$

---

# 4. QMF 的核心理论

论文从泛化误差上界出发分析动态融合。

其核心可以简化理解为：

$$
\mathrm{GError}
\le
\sum_mE(w^m)\hat E(f^m)
+
\sum_mE(w^m)\mathfrak R_m
+
\sum_m\mathrm{Cov}(w^m,l^m)
+
C
$$

其中：

- $w^m$：第 $m$ 个模态的动态融合权重；
- $l^m$：该模态对应样本的预测损失；
- $\mathfrak R_m$：模型复杂度相关项。

最关键的是：

$$
\boxed{
\mathrm{Cov}(w^m,l^m)
}
$$

如果某模态误差越大时权重反而越高：

$$
l_m\uparrow
\Rightarrow
w_m\uparrow
$$

那么坏模态会进一步污染最终预测。

理想情况应为：

$$
l_m\uparrow
\Rightarrow
w_m\downarrow
$$

即：

$$
\boxed{
\mathrm{Cov}(w^m,l^m)<0
}
$$

或者从相关系数角度：

$$
\boxed{
r(w^m,l^m)\le0
}
$$

这就是 QMF 的核心理论依据。

---

# 5. 为什么需要“不确定度”

实际推理阶段无法知道真实标签，因此无法直接获得：

$$
l^m=L(y,\hat y_m)
$$

所以 QMF 使用不确定度估计：

$$
u^m(x)
$$

代替真实误差。

要求：

$$
r(u^m,l^m)\ge0
$$

即：

> 预测越容易出错，不确定度应该越高。

随后设置：

$$
w^m=\alpha^mu^m+\beta^m
$$

且：

$$
\alpha^m<0
$$

从而：

$$
u_m\uparrow
\Rightarrow
w_m\downarrow
$$

最终形成：

$$
\boxed{
l_m\uparrow
\Rightarrow
u_m\uparrow
\Rightarrow
w_m\downarrow
}
$$

---

# 6. 原版 QMF 的不确定度估计

原论文针对分类任务，使用 **Energy Score（能量分数）**。

对于第 $m$ 个模态的分类 Logits：

$$
f^m(x)
=
[f_1^m,\dots,f_K^m]
$$

定义能量：

$$
E_m(x)
=
-T_m
\log
\sum_{k=1}^{K}
\exp
\left(
\frac{f_k^m(x)}{T_m}
\right)
$$

其中：

$$
T_m
$$

为温度参数。

随后根据能量估计模态可靠性。

核心特点是：

> QMF 不需要额外训练一个大型质量预测网络，而是直接从单模态预测结果中推导质量分数。

因此算法开销较低。

---

# 7. 历史损失：QMF 的第二个关键机制

仅依靠能量分数并不能保证：

$$
u_m
$$

与实际预测误差高度相关。

因此 QMF 又引入了**训练历史损失**。

对于第 $i$ 个样本、第 $m$ 个模态：

$$
\kappa_i^m
=
\frac{1}{T}
\sum_{t=T_s}^{T_s+T}
L
\left(
y_i,
f_{\theta_t}^m(x_i^m)
\right)
$$

它表示某个样本在一段训练过程中的平均预测困难程度。

相比只使用最终训练损失，历史损失更能区分：

- 容易样本；
- 困难样本；
- 噪声样本；
- 低质量模态。

原因在于深度网络最终可能记忆训练数据，使所有样本训练误差都变小，而历史损失仍然能够保留学习难度信息。

---

# 8. 质量排序约束

QMF 不要求精确预测一个绝对质量值，而要求质量排序正确。

如果：

$$
\kappa_i^m>\kappa_j^m
$$

说明样本 $i$ 在该模态下更加困难或不可靠。

因此需要满足：

$$
w_i^m<w_j^m
$$

即：

$$
\boxed{
\kappa_i^m\ge\kappa_j^m
\Longleftrightarrow
w_i^m\le w_j^m
}
$$

实际实现中采用 Margin Ranking Loss。

这种设计的重要优势是：

> **学习“谁更可靠”通常比精确预测传感器质量数值更容易。**

---

# 9. QMF 的总损失

原论文的整体训练目标可以写成：

$$
L_{\mathrm{overall}}
=
L_{\mathrm{fusion}}
+
\sum_mL_m
+
\lambda L_{\mathrm{rank}}
$$

其中：

## 9.1 融合预测损失

$$
L_{\mathrm{fusion}}
$$

监督最终多模态融合结果。

## 9.2 单模态辅助监督

$$
\sum_mL_m
$$

每个模态都有独立预测头。

例如：

$$
US\rightarrow\hat y_{\mathrm{US}}
$$

$$
TCD\rightarrow\hat y_{\mathrm{TCD}}
$$

$$
NDIR\rightarrow\hat y_{\mathrm{NDIR}}
$$

这样能够避免融合层完全支配梯度，使每个模态编码器保持独立预测能力。

## 9.3 质量排序损失

$$
L_{\mathrm{rank}}
$$

用于保证：

$$
\kappa_i^m>\kappa_j^m
\Rightarrow
w_i^m<w_j^m
$$

---

# 10. 原始 QMF 网络结构

可以简化为：

```text
模态 A
  ↓
编码器 A
  ↓
预测头 A
  ├────────→ 单模态监督
  ↓
不确定度 A
  ↓
权重 A ──────────────┐

                        ├→ 动态加权融合 → 最终预测
                        │

模态 B                 │
  ↓                     │
编码器 B                │
  ↓                     │
预测头 B                │
  ├────────→ 单模态监督 │
  ↓                     │
不确定度 B              │
  ↓                     │
权重 B ─────────────────┘

历史训练损失
      ↓
质量排序监督
```

因此 QMF 更准确的定义是：

$$
\boxed{
\text{带可靠性监督的动态晚期融合}
}
$$

---

# 11. 实验结论

论文在多个多模态数据集上测试，包括：

- NYU Depth V2；
- SUN RGB-D；
- FOOD-101；
- MVSA。

并人为加入：

- 高斯噪声；
- 椒盐噪声；
- 空白文本；
- 不同程度模态质量下降。

主要结论是：

> QMF 在干净数据上的优势通常不是最大的，而在某一模态严重受污染时优势更加明显。

这与工业传感场景高度一致。

QMF 的消融实验也说明：

$$
\boxed{
\text{不确定度动态权重}
+
\text{历史损失排序监督}
}
$$

二者是互补的。

论文还验证了：

$$
r(\mathrm{uncertainty},\mathrm{loss})
$$

QMF 能够得到明显更高的相关性，说明其质量分数确实更能反映实际预测风险。

---

# 12. QMF 的主要优势

## 12.1 结构轻量

不需要额外加入大型 Transformer 或复杂生成模型。

核心增加部分只有：

- 单模态预测头；
- 不确定度计算；
- 动态融合；
- 排序损失。

## 12.2 权重具有明确含义

普通 Attention 的权重只代表：

> 模型认为某个特征重要。

QMF 的权重则希望明确表示：

> 当前模态是否可靠。

因此具有更好的工程解释性。

## 12.3 适合传感器质量退化实验

例如人为增加超声噪声：

$$
\sigma_{\mathrm{US}}
=
0,\;0.1,\;0.2,\;0.5,\;1.0
$$

期望：

$$
\sigma_{\mathrm{US}}\uparrow
\Rightarrow
w_{\mathrm{US}}\downarrow
$$

可以非常直观地验证模型是否真正学会质量感知。

---

# 13. QMF 的主要局限

QMF 原论文针对的是：

$$
\boxed{
\text{分类任务}
}
$$

其 Energy Score 来自多个类别 Logits。

而多组分气体检测目标是：

$$
\mathbf y=
[
C_{\mathrm{H_2}},
C_{\mathrm{CH_4}},
C_{\mathrm{CO_2}},
C_{\mathrm{N_2}}
]
$$

属于连续多输出回归。

因此原始：

$$
-\log\sum_ke^{f_k(x)}
$$

不能直接作为气体浓度回归的不确定度估计。

所以迁移 QMF 时，应保留：

$$
\boxed{
\text{误差风险}
\rightarrow
\text{不确定度}
\rightarrow
\text{质量权重}
}
$$

这一理论框架，而不是机械复制分类 Energy Score。

---

# 14. 面向气体浓度回归的 QMF 改造

建议构建：

# QMF-Reg：Quality-aware Multimodal Regression

对于每个模态：

$$
z_m=E_m(x_m)
$$

分别输出：

$$
\hat{\mathbf y}_m
$$

以及：

$$
u_m
$$

即：

$$
R_m(z_m)
\rightarrow
(
\hat{\mathbf y}_m,u_m
)
$$

其中：

- $\hat{\mathbf y}_m$：该模态独立预测的气体浓度；
- $u_m$：该模态当前预测不确定度。

动态权重：

$$
w_m
=
\frac{
\exp(-\gamma u_m)
}{
\sum_j\exp(-\gamma u_j)
}
$$

最终浓度：

$$
\boxed{
\hat{\mathbf y}
=
\sum_m
w_m\hat{\mathbf y}_m
}
$$

---

# 15. 第一版建议：异方差回归

可以让每个模态输出：

$$
\hat{\mathbf y}_m
$$

和：

$$
\log\sigma_m^2
$$

网络结构：

```text
超声编码器
    ↓
1D-CNN / TCN
    ↓
┌────────────────┐
│ 浓度预测头      │ → y_US
│ 不确定度预测头  │ → σ²_US
└────────────────┘
```

然后：

$$
u_m=\sigma_m^2
$$

并：

$$
w_m=\operatorname{Softmax}(-\gamma u_m)
$$

这样自然实现：

$$
u_m\uparrow
\Rightarrow
w_m\downarrow
$$

---

# 16. 更完整的回归方案：QMF + MoNIG

与 QMF 同一研究方向已有多模态回归方法：

**Trustworthy Multimodal Regression with Mixture of Normal-Inverse Gamma Distributions**

该方法使用 Normal-Inverse-Gamma（NIG）分布估计：

- 偶然不确定度（Aleatoric Uncertainty）；
- 认知不确定度（Epistemic Uncertainty）。

因此对于本课题，更合理的组合是：

$$
\boxed{
QMF\text{ 的动态融合理论}
+
MoNIG\text{ 的回归不确定度估计}
}
$$

而不是直接复制 QMF 的分类能量分数。

---

# 17. 推荐用于本课题的结构

```text
超声原始波形
    ↓
1D-CNN
    ↓
TCN
    ↓
浓度预测 US + 不确定度 US
                         ┐
                         │
热导动态响应             │
    ↓                    │
MLP / TCN                │
    ↓                    ├→ 质量感知动态融合
浓度预测 TCD + 不确定度 TCD│
                         │
NDIR                     │
    ↓                    │
MLP / 1D-CNN             │
    ↓                    │
浓度预测 NDIR + 不确定度 NDIR
                         ┘
                         ↓
                  多组分浓度输出
```

质量权重：

$$
w_m=
\operatorname{Softmax}(-\gamma u_m)
$$

同时保留 QMF 的历史损失排序：

$$
\kappa_i^m>\kappa_j^m
\Rightarrow
w_i^m<w_j^m
$$

---

# 18. 推荐损失函数

可以设计为：

$$
L_{\mathrm{total}}
=
L_{\mathrm{fusion}}
+
\lambda_{\mathrm{uni}}L_{\mathrm{uni}}
+
\lambda_{\mathrm{rank}}L_{\mathrm{rank}}
+
\lambda_{\mathrm{physics}}L_{\mathrm{physics}}
$$

其中：

## 融合浓度回归

$$
L_{\mathrm{fusion}}
=
L
(\mathbf y,\hat{\mathbf y})
$$

## 单模态辅助监督

$$
L_{\mathrm{uni}}
=
\sum_m
L
(\mathbf y,\hat{\mathbf y}_m)
$$

## 质量排序

$$
L_{\mathrm{rank}}
$$

保证：

$$
\kappa_i^m>\kappa_j^m
\Rightarrow
w_i^m<w_j^m
$$

## 物理约束

对于气体摩尔/体积分数，可采用硬约束：

$$
\sum_k\hat y_k=1
$$

例如使用 Softmax 输出层保证非负性和闭合性。

---

# 19. 动态时间序列进一步改造

原始 QMF 是样本级：

$$
w_m(x)
$$

对于动态气体检测，更适合改成：

$$
\boxed{
w_{m,t}
}
$$

即每个时间步都有独立质量权重。

例如：

| 时间    | $w_{\mathrm{US}}$ | $w_{\mathrm{NDIR}}$ | $w_{\mathrm{TCD}}$ |
| ----- | -----------------:| -------------------:| ------------------:|
| $t_1$ | 0.8               | 0.1                 | 0.1                |
| $t_2$ | 0.8               | 0.1                 | 0.1                |
| $t_3$ | 0.7               | 0.2                 | 0.1                |
| $t_4$ | 0.2               | 0.5                 | 0.3                |
| $t_5$ | 0.1               | 0.6                 | 0.3                |

如果超声从 $t_4$ 开始退化：

$$
w_{\mathrm{US},t}\downarrow
$$

模型自动提高 NDIR 和热导通道贡献。

这可以形成：

$$
\boxed{
\text{Temporal Quality-aware Multimodal Fusion}
}
$$

即时间级质量感知动态融合。

---

# 20. QMF 与普通 Attention 的区别

普通 Attention：

$$
w_m=f(z_1,z_2,\dots,z_M)
$$

它学的是：

> 哪个模态当前更“重要”。

但并不能保证：

$$
w_m
$$

对应真实传感器质量。

QMF：

$$
w_m=f(u_m)
$$

而：

$$
u_m
$$

通过历史预测风险进行监督，使其尽量满足：

$$
u_m
\sim
\text{prediction error}
$$

因此：

$$
\boxed{
QMF权重具有明确的可靠性语义
}
$$

这是 QMF 相比普通 Attention 最重要的差异。

---

# 21. 对本课题的最终评价

| 维度          | 评价    |
| ----------- | ----- |
| 理论创新        | ★★★★★ |
| 网络复杂度       | ★★☆☆☆ |
| 可复现性        | ★★★★★ |
| 分类任务适配度     | ★★★★★ |
| 气体浓度回归直接适配度 | ★★★☆☆ |
| 传感器质量退化适配度  | ★★★★★ |
| 动态工业场景价值    | ★★★★★ |
| 与现有多模态模型兼容性 | ★★★★★ |

## 最值得继承的不是 Energy Score，而是：

$$
\boxed{
\text{预测风险}
\rightarrow
\text{不确定度}
\rightarrow
\text{质量权重}
}
$$

以及：

$$
\boxed{
\text{历史预测损失}
\rightarrow
\text{质量排序监督}
}
$$

## 推荐后续算法方向

最值得继续研究的是：

$$
\boxed{
QMF
+
MoNIG
+
时间级质量权重
}
$$

最终形成：

> **面向动态多组分气体浓度回归的时间质量感知多模态融合网络**

该方向能够直接解决：

- 模态质量动态变化；
- 传感器噪声；
- 灵敏度衰减；
- 局部异常；
- 不同模态可靠程度变化；
- 多组分连续浓度回归。
