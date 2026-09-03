# Ma et al., 2021 — MoNIG 论文精读整理

## 1. 论文信息

**论文题目：** *Trustworthy Multimodal Regression with Mixture of Normal-Inverse Gamma Distributions*  
**作者：** Huan Ma, Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou, Qinghua Hu  
**年份：** 2021  
**会议：** NeurIPS 2021  
**核心方法：** Mixture of Normal-Inverse Gamma Distributions（MoNIG）

### 一句话总结

MoNIG 面向多模态连续值回归任务，通过让每个模态输出一个 Normal-Inverse-Gamma（NIG）概率分布，同时估计预测值、偶然不确定度和认知不确定度，并利用证据强度和模态间预测冲突进行可信融合，从而提高低质量或受污染模态下的回归鲁棒性。

其核心可以概括为：

$$
\boxed{
\text{Evidential Regression}
+
\text{Uncertainty-aware Multimodal Fusion}
}
$$

---

# 2. 为什么普通多模态回归不够

传统回归通常只输出：

$$
\hat y=f_\theta(x)
$$

对于多组分气体检测，例如：

$$
\hat{\mathbf y}
=
[
\hat C_{\mathrm{H_2}},
\hat C_{\mathrm{CH_4}},
\hat C_{\mathrm{CO_2}},
\hat C_{\mathrm{N_2}}
]
$$

模型只能告诉我们预测结果是多少，却不能直接说明：

> 当前预测到底有多可靠。

例如两次 CO$_2$ 都预测：

$$
\hat C_{\mathrm{CO_2}}=5\%
$$

但实际可能分别对应：

- 传感器状态良好、数据处于训练分布内部；
- NDIR 光源退化、超声噪声严重、数据位于训练分布外。

普通回归无法显式区分这两种情况。

MoNIG 的目标则是同时得到：

$$
\boxed{
\text{预测值}
+
\text{预测不确定度}
}
$$

---

# 3. 两种不确定度

MoNIG 将回归不确定度分为两类。

## 3.1 Aleatoric Uncertainty

偶然不确定度：

$$
AU
$$

用于描述：

> 数据本身有多嘈杂。

例如：

- 超声电噪声；
- NDIR 光强抖动；
- 热导传感器短期波动；
- 环境扰动；
- 测量随机误差。

这类不确定度来自观测过程本身，即使增加训练数据，也未必可以完全消除。

---

## 3.2 Epistemic Uncertainty

认知不确定度：

$$
EU
$$

用于描述：

> 模型因为没有见过类似数据而有多不确定。

例如：

训练数据中：

$$
H_2\in[0,20\%]
$$

测试时：

$$
H_2\approx45\%
$$

或者训练温度范围：

$$
20^\circ C\sim30^\circ C
$$

测试时：

$$
50^\circ C
$$

即使传感器噪声不大，模型也应表现出：

$$
EU\uparrow
$$

因此：

- $AU$ 更偏向传感器噪声；
- $EU$ 更偏向分布外和模型认知不足。

---

# 4. 为什么使用 NIG 分布

普通高斯回归假设：

$$
y\sim\mathcal N(\mu,\sigma^2)
$$

如果网络输出：

$$
\mu,\sigma^2
$$

就可以得到预测均值和测量噪声。

但：

$$
\mu
$$

和：

$$
\sigma^2
$$

本身也是网络预测出来的。

因此 MoNIG 对它们进一步建模：

$$
y\sim\mathcal N(\mu,\sigma^2)
$$

$$
\mu
\sim
\mathcal N
\left(
\delta,
\frac{\sigma^2}{\gamma}
\right)
$$

$$
\sigma^2
\sim
\mathrm{InvGamma}
(\alpha,\beta)
$$

合并得到：

$$
\boxed{
(\mu,\sigma^2)
\sim
\mathrm{NIG}
(\delta,\gamma,\alpha,\beta)
}
$$

所以每个模态不再只输出一个预测值，而是输出四个 NIG 参数。

---

# 5. 四个 NIG 参数的含义

## 5.1 $\delta$：预测均值

$$
\boxed{
\delta=\hat y
}
$$

即回归任务的最终预测值。

对于气体检测：

$$
\delta_{\mathrm{CO_2}}
=
\hat C_{\mathrm{CO_2}}
$$

---

## 5.2 $\gamma$：均值证据强度

$\gamma$ 反映模型对预测均值 $\delta$ 的置信程度。

直观上：

$$
\gamma\uparrow
\Rightarrow
\text{更相信 }\delta
$$

$$
\gamma\downarrow
\Rightarrow
\text{对 }\delta\text{ 缺乏证据}
$$

在多模态融合中，$\gamma$ 直接参与不同模态预测均值的加权。

---

## 5.3 $\alpha$：方差证据参数

$\alpha$ 与方差分布的证据强度有关。

需要满足：

$$
\alpha>1
$$

以保证相关方差期望能够正常定义。

---

## 5.4 $\beta$：不确定度尺度参数

$\beta$ 控制整体方差尺度。

在多模态融合过程中，$\beta$ 还会吸收：

$$
\boxed{
\text{不同模态预测之间的冲突}
}
$$

因此它不仅反映单模态自身不确定度，也包含跨模态分歧信息。

---

# 6. AU 与 EU 的核心公式

对于：

$$
\mathrm{NIG}
(\delta,\gamma,\alpha,\beta)
$$

偶然不确定度为：

$$
\boxed{
AU
=
\mathbb E[\sigma^2]
=
\frac{\beta}{\alpha-1}
}
$$

认知不确定度为：

$$
\boxed{
EU
=
\mathrm{Var}[\mu]
=
\frac{\beta}
{\gamma(\alpha-1)}
}
$$

因此：

$$
EU
=
\frac{AU}{\gamma}
$$

意味着：

$$
\gamma\uparrow
\Rightarrow
EU\downarrow
$$

即证据越充分，对预测均值越确定。

---

# 7. 单模态网络输出形式

对于一个模态编码器：

$$
z_m=E_m(x_m)
$$

普通回归头：

$$
z_m\rightarrow\hat y_m
$$

MoNIG 改为：

$$
z_m
\rightarrow
\begin{cases}
\delta_m\\
\gamma_m\\
\alpha_m\\
\beta_m
\end{cases}
$$

也就是：

$$
\boxed{
R_m(z_m)
\rightarrow
NIG_m
}
$$

官方实现中通常通过 Softplus 保证参数合法：

$$
\gamma>0
$$

$$
\beta>0
$$

$$
\alpha>1
$$

例如：

$$
\alpha=\mathrm{Softplus}(a)+1
$$

---

# 8. 多输出气体组分的输出维度

如果目标为四种气体：

$$
[
H_2,
CH_4,
CO_2,
N_2
]
$$

最简单做法是为每个组分独立输出一个 NIG：

$$
4\times
(\delta,\gamma,\alpha,\beta)
$$

因此每个模态最终需要输出：

$$
4\times4=16
$$

个参数。

即：

$$
\delta\in\mathbb R^4
$$

$$
\gamma\in\mathbb R_+^4
$$

$$
\alpha\in(1,\infty)^4
$$

$$
\beta\in\mathbb R_+^4
$$

---

# 9. 为什么不能简单平均多个模态

假设：

$$
US\rightarrow NIG_1
$$

$$
NDIR\rightarrow NIG_2
$$

$$
TCD\rightarrow NIG_3
$$

如果直接平均：

$$
\frac13NIG_1+
\frac13NIG_2+
\frac13NIG_3
$$

会忽略：

$$
U_{\mathrm{US}}
\neq
U_{\mathrm{NDIR}}
\neq
U_{\mathrm{TCD}}
$$

也就是不同模态质量不同。

因此 MoNIG 提出：

$$
\boxed{
NIG\ Summation
}
$$

通过证据强度和预测冲突进行概率融合。

---

# 10. 两个 NIG 的融合公式

假设：

$$
NIG_1
=
NIG
(\delta_1,\gamma_1,\alpha_1,\beta_1)
$$

$$
NIG_2
=
NIG
(\delta_2,\gamma_2,\alpha_2,\beta_2)
$$

定义：

$$
NIG=NIG_1\oplus NIG_2
$$

融合后的均值：

$$
\boxed{
\delta
=
\frac{
\gamma_1\delta_1+
\gamma_2\delta_2
}{
\gamma_1+\gamma_2
}
}
$$

因此：

$$
\gamma_1\gg\gamma_2
$$

时：

$$
\delta\approx\delta_1
$$

说明模态 1 的证据更强，最终预测更依赖模态 1。

---

融合后的证据参数：

$$
\boxed{
\gamma=\gamma_1+\gamma_2
}
$$

$$
\boxed{
\alpha=
\alpha_1+\alpha_2+\frac12
}
$$

以及：

$$
\boxed{
\beta
=
\beta_1+\beta_2
+
\frac12\gamma_1(\delta_1-\delta)^2
+
\frac12\gamma_2(\delta_2-\delta)^2
}
$$

---

# 11. 模态冲突如何转化为不确定度

$\beta$ 的最后两项为：

$$
\frac12\gamma_1(\delta_1-\delta)^2
+
\frac12\gamma_2(\delta_2-\delta)^2
$$

它们代表：

$$
\boxed{
\text{模态预测之间的冲突程度}
}
$$

如果：

$$
\delta_{\mathrm{US}}
\approx
\delta_{\mathrm{NDIR}}
$$

则：

$$
(\delta_i-\delta)^2
$$

较小，

最终不确定度也较低。

如果：

$$
\delta_{\mathrm{US}}=5
$$

$$
\delta_{\mathrm{NDIR}}=15
$$

模态意见严重冲突，则：

$$
\beta\uparrow
$$

进一步导致：

$$
AU\uparrow
$$

和：

$$
EU\uparrow
$$

因此 MoNIG 自然实现：

$$
\boxed{
\text{模态冲突}
\rightarrow
\text{全局不确定度上升}
}
$$

---

# 12. 多个模态的递归融合

NIG Summation 满足交换律和结合律，因此：

$$
NIG_{\mathrm{US}}
\oplus
NIG_{\mathrm{TCD}}
\oplus
NIG_{\mathrm{NDIR}}
$$

可以逐步融合。

这使模型天然支持不同数量的模态。

从工程角度看，它与“可配置传感体系”的目标非常匹配。

---

# 13. Pseudo Modality

MoNIG 不仅采用单模态决策融合，还额外引入：

$$
\boxed{
Pseudo\ Modality
}
$$

假设各模态特征为：

$$
z_1,z_2,z_3
$$

构造：

$$
z_P=
[z_1;z_2;z_3]
$$

再利用一个融合网络预测：

$$
NIG_P
$$

最终：

$$
NIG_F
=
NIG_1
\oplus
NIG_2
\oplus
NIG_3
\oplus
NIG_P
$$

Pseudo Modality 的目的在于补充纯决策级融合无法直接建模跨模态特征交互的问题。

所以 MoNIG 实际上结合了：

$$
\boxed{
\text{中间特征融合}
+
\text{不确定度感知决策融合}
}
$$

---

# 14. 面向气体检测的网络结构

建议改造为：

```text
超声原始波形
    ↓
1D-CNN
    ↓
TCN
    ↓
z_US
    ├────────→ NIG_US
    │
    │
热导动态信号
    ↓
MLP / TCN
    ↓
z_TCD
    ├────────→ NIG_TCD
    │
    │
NDIR
    ↓
MLP / 1D-CNN
    ↓
z_NDIR
    ├────────→ NIG_NDIR
    │
    └───────────────┐
                    ↓
          [z_US,z_TCD,z_NDIR]
                    ↓
              Fusion Network
                    ↓
                NIG_P
                    │
                    ↓
NIG_US ⊕ NIG_TCD ⊕ NIG_NDIR ⊕ NIG_P
                    ↓
            浓度预测 + AU + EU
```

相比简单：

$$
Concat\rightarrow TCN\rightarrow MLP
$$

最大的区别是：

> 每个传感器都保留独立预测能力和独立不确定度。

---

# 15. MoNIG 的训练损失

MoNIG 不直接采用普通 MSE，而是通过 NIG 的边缘分布构造概率回归目标。

边缘化后预测分布对应 Student-$t$ 分布，因此可以最小化：

$$
\boxed{
L_{\mathrm{NLL}}
}
$$

即负对数似然。

定义：

$$
\Omega=2\beta(1+\gamma)
$$

则：

$$
L_{\mathrm{NLL}}
=
\frac12\log\frac{\pi}{\gamma}
-\alpha\log\Omega
+
\left(\alpha+\frac12\right)
\log
\left[
\gamma(y-\delta)^2+\Omega
\right]
+
\log\Gamma(\alpha)
-
\log\Gamma
\left(\alpha+\frac12\right)
$$

它同时优化：

$$
\boxed{
\text{预测值}
+
\text{概率分布}
+
\text{不确定度}
}
$$

---

# 16. Evidence Regularization

仅依靠 NLL 可能出现：

$$
\boxed{
\text{Confident but Wrong}
}
$$

即：

$$
|y-\delta|\gg0
$$

但模型仍然输出很高证据。

论文定义证据强度：

$$
\Phi
=
\gamma+2\alpha
$$

并加入：

$$
\boxed{
L_R
=
|y-\delta|
(\gamma+2\alpha)
}
$$

其作用是：

如果预测错误较大，同时模型仍然声称证据很强，则受到更大惩罚。

即迫使网络满足：

$$
\boxed{
\text{预测错误时应降低置信度}
}
$$

---

# 17. 单模态损失

单模态训练目标为：

$$
\boxed{
L_m
=
L_{\mathrm{NLL},m}
+
\lambda L_{R,m}
}
$$

其中：

$$
\lambda
$$

控制证据正则项强度。

---

# 18. 整体 MoNIG 损失

如果有：

$$
M
$$

个真实模态：

$$
m=1,\dots,M
$$

一个 Pseudo Modality：

$$
P
$$

以及融合结果：

$$
F
$$

则：

$$
\boxed{
L
=
\sum_{m=1}^{M}L_m
+
L_P
+
L_F
}
$$

对于气体检测可写为：

$$
L
=
L_{\mathrm{US}}
+
L_{\mathrm{TCD}}
+
L_{\mathrm{NDIR}}
+
L_P
+
L_{\mathrm{fusion}}
$$

这种方式可以保证每个独立模态编码器都能直接得到监督信号。

---

# 19. 实验结论

论文在多种连续回归任务上验证了 MoNIG，包括：

- 超导材料临界温度预测；
- CT 切片相对位置预测；
- 多模态情感回归等。

总体结果显示：

- MoNIG 优于单模态回归；
- 加入 Pseudo Modality 后性能进一步提高；
- 在某一模态被人工加入噪声后，MoNIG 相比普通特征融合具有明显更高鲁棒性。

核心价值不只是降低 RMSE，而是：

$$
\boxed{
\text{随着模态质量下降，模型能够提高不确定度并降低坏模态影响}
}
$$

---

# 20. 不确定度与真实误差的一致性

论文进一步验证：

$$
RMSE\uparrow
\Rightarrow
AU\uparrow
$$

以及：

$$
RMSE\uparrow
\Rightarrow
EU\uparrow
$$

说明不确定度能够一定程度反映实际预测风险。

论文还定义：

$$
UEIR
=
\frac{
N_{\mathrm{inconsistent}}
}{
N_{\mathrm{all}}
}
\times100\%
$$

即：

**Uncertainty-Error Inconsistency Rate**

如果：

$$
RMSE_i>RMSE_j
$$

但：

$$
U_i<U_j
$$

则认为不确定度排序与真实误差不一致。

因此：

$$
\boxed{
UEIR越低越好
}
$$

这一指标可以直接迁移到传感器质量实验中。

---

# 21. QMF 与 MoNIG 的区别

| 特征 | QMF | MoNIG |
|---|---|---|
| 主要任务 | 分类 | 回归 |
| 质量依据 | Energy / 不确定度 | NIG 证据 |
| 模态权重 | 显式动态权重 | 概率证据融合 |
| Aleatoric / Epistemic | 不显式区分 | 显式区分 |
| 单模态辅助监督 | 有 | 有 |
| Pseudo Modality | 无 | 有 |
| 历史 Loss Ranking | 有 | 无 |
| 理论重点 | 泛化误差 | 概率证据融合 |
| 与气体浓度回归适配 | 需改造 | 高 |

因此：

$$
\boxed{
MoNIG更适合作为气体浓度回归基础骨架
}
$$

而：

$$
\boxed{
QMF更适合作为质量校准增强模块
}
$$

---

# 22. QMF + MoNIG 的组合方式

MoNIG 可以产生：

$$
AU_m
$$

和：

$$
EU_m
$$

定义综合不确定度：

$$
U_m
=
\lambda_AAU_m
+
\lambda_EEU_m
$$

进一步定义质量：

$$
Q_m
=
\frac{1}{U_m+\epsilon}
$$

因此：

$$
U_m\uparrow
\Rightarrow
Q_m\downarrow
$$

再加入 QMF 的历史损失：

$$
\kappa_i^m
$$

要求：

$$
\kappa_i^m>\kappa_j^m
\Rightarrow
U_i^m>U_j^m
$$

或者：

$$
\kappa_i^m>\kappa_j^m
\Rightarrow
Q_i^m<Q_j^m
$$

这样形成：

$$
\boxed{
MoNIG负责产生概率不确定度
+
QMF负责校准不确定度与真实错误的关系
}
$$

---

# 23. 多组分气体回归的关键改造

原始 MoNIG 主要针对：

$$
\boxed{
\text{标量回归}
}
$$

而气体检测属于：

$$
\mathbf y
=
[
C_{\mathrm{H_2}},
C_{\mathrm{CH_4}},
C_{\mathrm{CO_2}},
C_{\mathrm{N_2}}
]
$$

最简单做法：

$$
4\times NIG
$$

即每个组分独立预测一个 NIG。

但四种气体满足：

$$
C_{\mathrm{H_2}}
+
C_{\mathrm{CH_4}}
+
C_{\mathrm{CO_2}}
+
C_{\mathrm{N_2}}
=1
$$

因此各组分并不独立。

如果直接独立预测四个 NIG，会忽略组分之间的闭合关系。

---

# 24. 推荐的组分闭合改造

建议将最终浓度预测改为：

```text
融合特征
   ↓
Composition Logits
   ↓
Softmax
   ↓
四组分浓度
```

保证：

$$
C_k\ge0
$$

以及：

$$
\sum_kC_k=1
$$

同时保留独立 Evidential Head，用于估计：

$$
AU_k
$$

和：

$$
EU_k
$$

形成：

$$
\boxed{
\text{组分闭合预测}
+
\text{证据不确定度估计}
}
$$

---

# 25. 动态时间序列扩展

原始 MoNIG 主要采用：

$$
\text{sample-level uncertainty}
$$

但动态气体检测具有：

$$
t=1,\dots,T
$$

因此可以扩展为：

$$
AU_{m,t}
$$

和：

$$
EU_{m,t}
$$

例如：

| 时间 | US-EU | NDIR-EU | TCD-EU |
|---|---:|---:|---:|
| $t_1$ | 0.1 | 0.2 | 0.3 |
| $t_2$ | 0.1 | 0.2 | 0.3 |
| $t_3$ | 0.2 | 0.2 | 0.3 |
| $t_4$ | 0.8 | 0.3 | 0.3 |
| $t_5$ | 0.9 | 0.3 | 0.3 |

如果超声从：

$$
t_4
$$

开始失真：

$$
EU_{\mathrm{US},t}\uparrow
$$

系统可以自动降低超声贡献。

这可以进一步形成：

$$
\boxed{
Temporal\ Evidential\ Multimodal\ Regression
}
$$

---

# 26. AU 与 EU 对不同退化类型的解释

## 随机噪声

$$
x'=x+\epsilon
$$

更可能体现为：

$$
AU\uparrow
$$

---

## 未见过的传感器漂移

$$
x'=ax+b
$$

若训练时未覆盖该漂移，则更可能：

$$
EU\uparrow
$$

---

## 跨设备

$$
Sensor_A
\rightarrow
Sensor_B
$$

可能出现：

$$
EU\uparrow
$$

---

## 训练阶段见过的固定噪声

模型学会其统计规律后，可能主要表现为：

$$
AU\uparrow
$$

而：

$$
EU
$$

不一定明显增大。

因此 AU/EU 分离不仅用于融合，还可以用于解释：

> 当前性能下降到底来自测量噪声还是模型认知不足。

---

# 27. 对低可辨识组分的解释价值

如果某一组分缺少直接敏感传感通道，即使传感器没有明显随机噪声：

$$
AU
$$

未必很大，

但模型因为缺乏足够信息确定该组分：

$$
EU
$$

可能持续较高。

因此 MoNIG 可以帮助区分：

$$
\boxed{
\text{传感器噪声问题}
}
$$

和：

$$
\boxed{
\text{组分可辨识性不足}
}
$$

这比仅观察 RMSE 更有解释力。

---

# 28. MoNIG 的主要局限

## 28.1 Confident but Wrong

MoNIG 本质上仍然让网络自己预测：

$$
\delta_m
$$

和：

$$
U_m
$$

即可能发生：

$$
Error\uparrow
$$

但：

$$
U\downarrow
$$

Evidence Regularization 可以缓解，但不能绝对保证消失。

这也是 QMF 可以进一步补强 MoNIG 的原因。

---

## 28.2 Pseudo Modality 仍可能受到坏模态污染

原始：

$$
z_P
=
[z_{\mathrm{US}},z_{\mathrm{TCD}},z_{\mathrm{NDIR}}]
$$

如果：

$$
z_{\mathrm{US}}
$$

已经严重污染，

Pseudo Modality 仍然直接吃入坏特征。

可以进一步改成：

$$
z_P
=
[
Q_{\mathrm{US}}z_{\mathrm{US}},
Q_{\mathrm{TCD}}z_{\mathrm{TCD}},
Q_{\mathrm{NDIR}}z_{\mathrm{NDIR}}
]
$$

形成：

$$
\boxed{
Quality-aware\ Pseudo\ Modality
}
$$

即先抑制低质量模态，再做跨模态特征融合。

---

# 29. 推荐第一阶段模型

第一阶段不要一次加入过多模块。

建议先实现：

$$
E_{\mathrm{US}},
E_{\mathrm{TCD}},
E_{\mathrm{NDIR}}
$$

分别得到：

$$
NIG_{\mathrm{US}},
NIG_{\mathrm{TCD}},
NIG_{\mathrm{NDIR}}
$$

同时：

$$
z_P=
Concat(z_{\mathrm{US}},z_{\mathrm{TCD}},z_{\mathrm{NDIR}})
$$

得到：

$$
NIG_P
$$

最后：

$$
\boxed{
NIG_F=
NIG_{\mathrm{US}}
\oplus
NIG_{\mathrm{TCD}}
\oplus
NIG_{\mathrm{NDIR}}
\oplus
NIG_P
}
$$

输出：

$$
\hat{\mathbf C}
$$

$$
AU
$$

$$
EU
$$

---

# 30. 第二阶段加入 QMF 质量校准

维护每个样本或时间步的历史误差：

$$
\kappa_{i,t}^m
$$

加入：

$$
L_{\mathrm{rank}}
$$

要求：

$$
\kappa_{i,t}^m\uparrow
\Rightarrow
U_{i,t}^m\uparrow
$$

最终：

$$
L=
L_{\mathrm{MoNIG}}
+
\lambda_Q L_{\mathrm{rank}}
$$

可以形成：

> **Quality-aware Evidential Multimodal Fusion Network**

核心流程：

$$
\boxed{
\text{异构编码}
\rightarrow
\text{证据回归}
\rightarrow
\text{不确定度估计}
\rightarrow
\text{质量校准}
\rightarrow
\text{可信动态融合}
}
$$

---

# 31. 推荐退化实验

至少应测试：

- Gaussian Noise；
- Bias Drift；
- Sensitivity Decay；
- Saturation；
- Missing Segment；
- Complete Modality Dropout；
- Sensor Delay；
- Cross-device Shift。

除传统：

$$
RMSE
$$

$$
MAE
$$

$$
R^2
$$

外，还应增加：

$$
Corr(U,Error)
$$

和：

$$
UEIR
$$

以及以下曲线：

$$
Noise\ Level
\rightarrow
AU
$$

$$
OOD\ Distance
\rightarrow
EU
$$

$$
Degradation
\rightarrow
Modality\ Weight
$$

这些指标能直接验证模型是否真正学会了传感器质量。

---

# 32. 复现时需要注意的公式/代码差异

论文补充材料中的 Evidence Regularization 为：

$$
L_R
=
|y-\delta|
(\gamma+2\alpha)
$$

但官方 GitHub 实现中出现了：

$$
|y-\delta|
(2\gamma+\alpha)
$$

即两者并不完全一致。

因此复现时建议分别测试：

### 论文版本

$$
\gamma+2\alpha
$$

### 官方代码版本

$$
2\gamma+\alpha
$$

进行小规模对照实验，再固定最终实现。

官方代码通过 Softplus 和 $+1$ 保证 NIG 参数合法，这一部分应保留。

---

# 33. 对本课题的最终评价

| 维度 | 评价 |
|---|---|
| 与连续浓度回归匹配 | ★★★★★ |
| 不确定度建模 | ★★★★★ |
| 抗传感器噪声 | ★★★★★ |
| 可解释性 | ★★★★★ |
| 可复现性 | ★★★★☆ |
| 多组分直接适配 | ★★★☆☆ |
| 动态时间序列直接适配 | ★★★☆☆ |
| 与 QMF 组合潜力 | ★★★★★ |
| 形成算法创新潜力 | ★★★★★ |

## 核心判断

MoNIG 更适合作为：

$$
\boxed{
\text{可信多模态回归主体}
}
$$

QMF 更适合作为：

$$
\boxed{
\text{质量校准增强模块}
}
$$

因此目前最值得继续发展的主线为：

$$
\boxed{
\text{独立物理模态编码}
\rightarrow
\text{MoNIG证据回归}
\rightarrow
\text{AU/EU质量估计}
\rightarrow
\text{QMF质量排序校准}
\rightarrow
\text{时间级动态融合}
\rightarrow
\text{闭合约束多组分输出}
}
$$

该结构可以同时解决：

- 多模态连续浓度回归；
- 传感器随机噪声；
- 质量动态变化；
- 模态间预测冲突；
- 分布外输入；
- 跨设备不确定性；
- 多组分闭合约束；
- 低可辨识组分的解释问题。
