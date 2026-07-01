<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-07-01 -->
<!-- file-state: revision=90 | updated-at=2026-07-01T15:16:27+08:00 | writer-id=Codex | base-workspace-revision=178 -->

<!-- section: current_state -->
- **合成气 Stage Ⅱ ablation 全部完成（27/27 runs，2026-06-27）**。三组消融在 sg4-formal（Ⅱ-1 与 Ⅱ-3）与 sg4-formal-crosstalk（Ⅱ-2）上展开，结果已写入 docs/syngas/stage_ii_ablation_results.md，相关文档已同步更新。全量测试 462 passed（hg 零回归）。
- **Ⅱ-1 物理证据**：移除 V_NDIR_CO 通道，x_CO R² 从 0.954 到 0.484（TCN 与 Ridge 同向 0.470），损失约 50%；仅保留 V_NDIR_CO 加环境可恢复 0.93 到 0.94。结论修正：CO 主导依赖光学通道，残留可学性来自闭包约束、V_TCS 弱热导差与可能的 CO₂ NDIR 弱串扰。
- **Ⅱ-2 鲁棒性**：3 x 3 CO₂ 与 CO 光学串扰为确定性线性变换，模型可学到逆映射，所有组分 R² 与无串扰持平（|Δ| ≤ 0.006）。sim-to-real gap 不在串扰矩阵层面，应在 Stage Ⅲ 硬件层面验证。
- **Ⅱ-3 方法学**：weighted_component_mse(inverse_train_var) 切换至任一未加权 loss 后，x_CH4 R² 从 0.827 跌至 0.39 到 0.44，其他组分几乎不变。逆方差加权对低浓度组分是论文核心方法学贡献。
- **掺氢天然气 P3 状态保留**：gaussian_noise 单独使用 Test R²=+0.5907（P0-P3 历史最高）但 CO₂ 毁灭（-0.23），noise_std=0.005 调优待跑。两条工程线（hg 噪声调优与 syngas Stage Ⅲ）独立推进。
- **RCDW Phase 6E 已完成（2026-07-01）**：rcdw_mgda 现为独立 benchmark 子工程，schema 为 rcdw-benchmark-1，12 维通道与 HITRAN smoke 证据链已建立。Phase 6A-6D 已完成；Phase 6E pressure_drift 已完成代码、配置、测试与 perturb 实验，runs/phase6e_pressure/perturb 下 12 张 PNG 已生成，定向测试 23 passed，代码测试基线 222 passed。

<!-- section: active_judgments -->
- **syngas 阶段 Ⅱ 三大结论可写入论文**：(1) V_NDIR_CO 支配 CO 检测；(2) 线性串扰对模型可学性无影响；(3) inverse_train_var 加权让 CH₄ R² 翻倍。论文叙事重心：Ⅱ-1 与 Ⅱ-3 是科学与方法学贡献，Ⅱ-2 作为 informative 负结果反衬 Stage Ⅲ 硬件验证的必要性。
- **syngas baseline 排序确认**：TCN 约等于 Ridge，约 0.96；PatchTST、CNN1D、LSTM 约 0.93。慢通道加手工统计量在该规模（6000 序列）接近性能上限，深度模型边际收益有限。
- **gaussian_noise CO₂ 毁灭仍未解决（hg）**：+0.5907 整体最佳以 CO₂ -0.23 为代价。noise_std=0.005 与 apply_prob 扫描尚未启动，与 syngas Stage Ⅱ 并行不冲突。
- **N₂ 不可学本质未变（hg）**：所有 P0-P3 实验 N₂ R² 均为 0 或负。
- **RCDW 定位为独立验证线**：三组分 O2、CO2、N2 不兼容主线 hg 四组分闭包与 syngas 四组分加背景气体体系。若要复用 RCDW 思想到主线需重新设计 W_base 矩阵（声学、光学、TCS、光纤麦克风四模态）。不构成主线 blocker。
- **RCDW pressure_drift 结论边界**：当前 pressure_drift 只移动输入空间 P_MPa，不重新生成压力派生的声学、光学或慢响应观测；因此 level 0.11 的 MAE/RMSE 0.0576/0.0678 只能说明当前输入扰动未造成可观测退化，不能解释为压力物理链路鲁棒。temperature 仍是 scaler-on 128-seq smoke 中主导扰动，level 0.11 为 0.1129/0.1463。

<!-- section: risks_open_questions -->
- **Ⅱ-1 残留 50% CO 可学性的物理来源未量化**：闭包约束、V_TCS 热导与可能的 CO₂ NDIR 弱串扰，三因素相对贡献需要后续 ablation。
- **Ⅱ-2 与 sim-to-real 关系待落地**：仿真层面证明线性串扰不构成学习难度，但真实硬件的非线性、时变与标定漂移因素需要 Stage Ⅲ 实测数据验证。
- **gaussian_noise CO₂ 毁灭问题（hg）**：noise_std=0.005 加 apply_prob 扫描待跑，预计 6 到 12 小时 GPU。
- **DL vs Ridge 差距（hg）**：缩小但仍在 0.12，超声与光纤麦克风模态尚未接入 syngas 训练，未验证多模态能否拉开 DL 与 Ridge。
- **PatchTST AMP 敏感性与 LSTM seed 不收敛**：已记录，不阻塞主结论。
- **RCDW h2o_cross 尚未决策**：进入正式扰动集前需先决定它是输入空间湿度扰动，还是 HITRAN 光学后端重生成扰动。
- **RCDW ErrorNet 判定偏敏感**：Phase 6B、6D、6E 中所有扰动 level 均打印 degraded=True，不能单独作为扰动严重性证据。
- **RecallLoom helper Windows 兼容性**：unlock_write_lock 对 stale PID 即便 --force 仍可能抛 WinError 87；遇到 damaged_sidecar 报错先查 .recallloom.write.lock 是否残留。

<!-- section: next_step -->
- **Stage Ⅲ 真实硬件 sim-to-real 测试规划**（优先级 P0，论文工程闭环）：需要先确认硬件可用性、数据采集协议、对标传感器型号。Ⅱ-2 持平结论说明仿真已不是 gap 主因。
- **Ⅱ 后续可选 ablation**（优先级 P1）：B 组在 sg4-formal-crosstalk 上重复以验证 CO₂ NDIR 弱串扰假设；B 组再去 V_TCS 以量化热导贡献；接入 ultrasonic 与 fiber_mic 后重做 Ⅱ-1。
- **hg P3 噪声调优**（优先级 P1，GPU 6 到 12 小时）：先跑 gaussian_noise_std=0.005 和 gaussian_noise_only + apply_prob=0.7，观察 CO₂ 是否转正。
- **论文初稿启动**（优先级 P2）：以 Ⅰ-3 加 Ⅱ 的 27 runs 数据为基础，结构对齐 IMRaD，重点突出 Ⅱ-1 物理证据与 Ⅱ-3 方法学贡献。
- **RCDW 后续**（优先级 P2）：先做 h2o_cross 语义决策；若继续物理校核，再进入 O2 弛豫参数校核与多 stage_profile 后段扩展。

<!-- section: recent_pivots -->
- 2026-07-01：**RCDW Phase 6E pressure_drift 完成**。完成 6 类 perturb 实验，runs/phase6e_pressure/perturb 下生成 12 张 PNG；temperature 仍是主导扰动，pressure_drift 在当前输入空间实现下未造成可观测退化。三份 RCDW 文档已同步，git diff --check 通过，定向测试 23 passed。
- 2026-07-01：**RCDW Phase 6A-6D 完成并同步**。HITRAN cache 预热、64 与 128 sequence HITRAN smoke、可选并行 benchmark、12 维 input scaler ablation 均已形成文档证据；scaler-on 明显优于 scaler-off。
- 2026-06-29：**学长 RCDW-MGDA 算法独立复现完整落地**。rcdw_mgda 子工程 M0-M5 全里程碑，35 静态测试 pass 加 numerical_check ALL PASS 加 smoke 端到端流水线通过，与主线 src 完全隔离。
- 2026-06-27：**syngas Stage Ⅱ ablation 全部完成（27 runs）**。三组消融结论清晰，文档全栈同步。下一步主线转向 Stage Ⅲ 硬件 sim-to-real 与论文写作。
- 2026-06-26：syngas Stage Ⅰ-3 基线训练完成（5 模型 x 3 seeds = 15 runs），TCN 约等于 Ridge，约 0.96；CNN1D、LSTM、PatchTST 约 0.93。
- 2026-06-25：P3 消融实验结果颠覆性发现。gaussian_noise 单独使用 R²=+0.5907（P0-P3 历史最高），CO₂ 毁灭 -0.23，time_jitter +0.31，amplitude_scale -0.64。
