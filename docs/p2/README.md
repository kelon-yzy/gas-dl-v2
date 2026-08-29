# P2 专用 bench 规格设计

> 状态：已完成并关闭（H3 `pass`；正式证据 `GIB-FREEZE-P2-20260825-01`；允许启动 P3 G3-1）
> 计划依据：[P2执行计划](./P2执行计划.md)
> 输入核验日期：2026-08-24

## 输入冻结登记

本阶段的方向与阶段门事实源是[项目总体规划](../../general_fusion/项目总体规划.md)；P1 冻结候选集是原门值事实源；两份调研报告均为派生参考，不构成授权文件。

| 输入 | 路径 | 工作区 SHA256 | 职责 |
|---|---|---|---|
| 项目总体规划 | [项目总体规划.md](../../general_fusion/项目总体规划.md) | `49E674945540D0A9E077C48A2011D29A6A0C6E1DEF62B3C5B33D72ED062CA7E8` | 方向、阶段门、P2 约束与原始范围事实源 |
| P1 冻结候选集 | [创新点候选集.md](../p1/创新点候选集.md) | `DE10F12F5A6D6F87B86983B5901CC6DE9F57ABB8C3CD63768E05FC8DE02C0263` | P1 候选、原门值和领地约束事实源 |
| P1 关闭后审查 | [09_P1补充检索与正式关闭审查.md](../p1/09_P1补充检索与正式关闭审查.md) | `E8D90B6D032F43C08E853A0B3543765BC688FE1B3972B239AF5060E9C105368E` | P1 关闭后排序变化与审查结论参考 |
| 算法创新深度调研 | [多传感器多组分气体检测算法创新深度调研报告.md](../多传感器多组分气体检测算法创新深度调研报告.md) | `5D7BC2A2F4D9816084BFC892D2C804505620FE13D95CEE7B17975544F6355BF6` | 算法角色、候选机制和检索证据参考，不提供授权 |
| 工业应用方向调研 | [工业多传感器气体组分检测_工业应用方向调研报告.md](../工业多传感器气体组分检测_工业应用方向调研报告.md) | `2BB8C20E4767E42541B786D1CC13CCDA15906DCAEEFF0F6362F7B9CB224AFA88` | 工业工作域、传感器和应用约束参考，不提供授权 |

## P2 任务状态

建立时所有任务初始状态均为 `not_started`；完成任务后按计划写入对应状态与 verdict。

| 任务 | 状态 | verdict |
|---|---|---|
| P2-00 | completed | `ready` |
| P2-01 | completed | `pass` |
| P2-02 | completed | `authorized` |
| P2-03 | completed | `approved` |
| P2-04 | completed | `ready_for_forward_screen` |
| P2-05 | completed | `candidate_selected` |
| P2-06 | completed | `grid_frozen` |
| P2-07 | completed | `pass` |
| P2-08 | completed | `protocol_frozen` |
| P2-09 | completed | `contract_frozen` |
| P2-10 | completed | `source_complete`（`controlled_synthetic`） |
| P2-11 | completed | `ownership_frozen` |
| P2-12 | completed | `matrix_ready` |
| P2-13 | completed | `ready_for_review` |
| P2-14 | completed | `pass` |

## P2-00 执行记录

```text
task_id: P2-00
input_versions:
  - 项目总体规划.md | workspace_sha256=49E674945540D0A9E077C48A2011D29A6A0C6E1DEF62B3C5B33D72ED062CA7E8
  - docs/p1/创新点候选集.md | workspace_sha256=DE10F12F5A6D6F87B86983B5901CC6DE9F57ABB8C3CD63768E05FC8DE02C0263
  - docs/p1/09_P1补充检索与正式关闭审查.md | workspace_sha256=E8D90B6D032F43C08E853A0B3543765BC688FE1B3972B239AF5060E9C105368E
  - docs/多传感器多组分气体检测算法创新深度调研报告.md | workspace_sha256=5D7BC2A2F4D9816084BFC892D2C804505620FE13D95CEE7B17975544F6355BF6
  - docs/工业多传感器气体组分检测_工业应用方向调研报告.md | workspace_sha256=2BB8C20E4767E42541B786D1CC13CCDA15906DCAEEFF0F6362F7B9CB224AFA88
changed_files:
  - docs/p2/README.md
commands:
  - rg --files -g '项目总体规划.md' -g '创新点候选集.md' -g '09_P1补充检索与正式关闭审查.md' -g '多传感器多组分气体检测算法创新深度调研报告.md' -g '工业多传感器气体组分检测_工业应用方向调研报告.md'
  - Get-FileHash -Algorithm SHA256 -LiteralPath <五份输入>
  - Test-Path -LiteralPath docs/p2/README.md
exit_codes:
  - rg: 0
  - Get-FileHash: 0
  - Test-Path: 0
generated_artifacts:
  - docs/p2/README.md
artifact_sha256: append-only 状态页；本记录不对自身内容建立循环哈希
failed_checks: []
verdict: ready
next_allowed_task: P2-01
```

## P2-01 执行记录

```text
task_id: P2-01
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=F33DAC6939845C753220CB484DED99556155707ABA13C436006ADCC298FFE8EA
  - tunnel_ventilation/docs/掘进通风代码契约事实源.md | workspace_sha256=3601D677F174E48D5AB3C7B575EBED70D466A707AB4F43CB69C0E3609E560DC9
  - 计划列出的 HG/SG packaging、TV3 audit/ml/pipeline/scripts 工作区来源 | 具体符号与行号见审计文档
changed_files:
  - docs/p2/P2能力复用审计.md
  - docs/p2/README.md
commands:
  - rg --files <计划列出的来源目录与文件>
  - rg -n <计划列出的来源符号、字段与 import 关键词>
  - rg -n "^(from|import) (hg|sg|tv3|rcdw)" <计划列出的来源代码>
  - Test-Path -LiteralPath gas_information_bench
  - 新子工程跨场景 import 验收：not_applicable_before_H2
  - rg -n "base_condition_id|noise_seed_index|noise_seed" docs/p2/P2能力复用审计.md
  - git diff --check -- docs/p2/P2能力复用审计.md
exit_codes:
  - source file discovery rg: 0
  - source symbol/import rg: 0
  - source import rg: 0
  - Test-Path gas_information_bench: 0（结果 False）
  - new-subproject import rg: not_applicable_before_H2
  - forbidden-field rg: 0（仅命中审计文档“禁止迁移字段说明”）
  - git diff --check: 0
generated_artifacts:
  - docs/p2/P2能力复用审计.md
artifact_sha256:
  - docs/p2/P2能力复用审计.md=9C6B27438B28535B8534A6C50CA4CDC7327F2BCDF6173B4CBA97D7AD7B506AFC
failed_checks: []
verdict: pass
next_allowed_task: P2-02
```

## P2-02 执行记录

```text
task_id: P2-02
input_versions:
  - 项目总体规划.md §5.4、§7.4
  - docs/p1/创新点候选集.md §3、§6
  - tunnel_ventilation/docs/paper/artifacts/table6_solver_efficiency.csv
  - tunnel_ventilation/docs/paper/artifacts/table7_structural_verification.csv
  - docs/多传感器多组分气体检测算法创新深度调研报告.md §3、§5–§7
changed_files:
  - docs/p2/P1候选修订记录.md
  - docs/p2/tools/render_c2_frozen_evidence.py
  - docs/p2/generated/c2_tv3_frozen_evidence.md
  - docs/p2/README.md
commands:
  - python docs/p2/tools/render_c2_frozen_evidence.py
  - Get-Content docs/p2/generated/c2_tv3_frozen_evidence.md -Raw
  - rg -n "table6|table7|clears_band|wall-clock|授权" docs/p2/generated/c2_tv3_frozen_evidence.md
  - git diff --check -- docs/p2/P1候选修订记录.md docs/p2/tools/render_c2_frozen_evidence.py docs/p2/generated/c2_tv3_frozen_evidence.md
exit_codes:
  - render_c2_frozen_evidence.py: 0（重复运行一致）
  - generated evidence read: 0
  - generated evidence rg: 0
  - git diff --check: 0
generated_artifacts:
  - docs/p2/P1候选修订记录.md
  - docs/p2/tools/render_c2_frozen_evidence.py
  - docs/p2/generated/c2_tv3_frozen_evidence.md
artifact_sha256:
  - docs/p2/tools/render_c2_frozen_evidence.py=9B63F829F590205B01D596469C65856ACE7441251F0B65CA1ADE10008FC0DD00
  - docs/p2/generated/c2_tv3_frozen_evidence.md=C3F9848938897F641C4B6C07668456CF81670C5AF1BFB3E2FAC32BC7BD8DD681
  - docs/p2/P1候选修订记录.md=append-only 任务记录，不对自身内容建立循环哈希
failed_checks: []
verdict: pending_authorization
next_allowed_task: P2-03
```

## P2-03 执行记录

```text
task_id: P2-03
input_versions:
  - README.md | workspace_sha256=47D3092AF2FA3C6F3732E27B01107B984FEF4E8D899C5E69E1AABCAEC0358FC7
  - AGENTS.md | workspace_sha256=4E39B21127DC98DFE0AC6CF81D8E3C0E8981A61011966FDAC6580BF502F74540
  - docs/p2/P2能力复用审计.md | workspace_sha256=9C6B27438B28535B8534A6C50CA4CDC7327F2BCDF6173B4CBA97D7AD7B506AFC
changed_files:
  - docs/p2/P2_bench规格书.md
  - gas_information_bench/README.md
  - gas_information_bench/pyproject.toml
  - gas_information_bench/gib/__init__.py
  - gas_information_bench/sim/core/.gitkeep
  - gas_information_bench/sim/packaging/.gitkeep
  - gas_information_bench/sim/validation/.gitkeep
  - gas_information_bench/audit/.gitkeep
  - gas_information_bench/configs/.gitkeep
  - gas_information_bench/docs/.gitkeep
  - gas_information_bench/tests/.gitkeep
  - gas_information_bench/outputs/runs/.gitkeep
  - gas_information_bench/outputs/summary/.gitkeep
  - gas_information_bench/outputs/reports/.gitkeep
  - gas_information_bench/outputs/archive/.gitkeep
  - docs/p2/README.md
commands:
  - Get-FileHash -Algorithm SHA256 -LiteralPath README.md, AGENTS.md, docs/p2/P2能力复用审计.md
  - New-Item -ItemType Directory -Force -Path gas_information_bench/sim/core, gas_information_bench/sim/packaging, gas_information_bench/sim/validation, gas_information_bench/audit, gas_information_bench/configs, gas_information_bench/docs, gas_information_bench/tests, gas_information_bench/outputs/runs, gas_information_bench/outputs/summary, gas_information_bench/outputs/reports, gas_information_bench/outputs/archive
  - rg -n "gas_information_bench|gas-information-bench|gib-benchmark-1|GIB-M|GIB-Q|mixture_id|sequence_id|review_gate|review_verdict" docs/p2/P2_bench规格书.md gas_information_bench
  - rg -n "base_condition_id|noise_seed_index|noise_seed" gas_information_bench
  - rg -n "^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)" gas_information_bench
  - Test-Path -LiteralPath gas_information_bench, gas_information_bench/README.md, gas_information_bench/pyproject.toml, gas_information_bench/gib/__init__.py, gas_information_bench/sim/core, gas_information_bench/sim/packaging, gas_information_bench/sim/validation, gas_information_bench/audit, gas_information_bench/configs, gas_information_bench/docs, gas_information_bench/tests, gas_information_bench/outputs/runs, gas_information_bench/outputs/summary, gas_information_bench/outputs/reports, gas_information_bench/outputs/archive, docs/p2/P2_bench规格书.md
  - git diff --check -- docs/p2/P2_bench规格书.md gas_information_bench docs/p2/README.md
exit_codes:
  - input hash: 0
  - skeleton creation: 0
  - namespace marker rg: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - skeleton path Test-Path: 0（全部为 True）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/ P2-03 空骨架
artifact_sha256:
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - gas_information_bench/README.md=190F8A84E56C821521B75175723404FFE127F92FF046C110969EDB8F465817FD
  - gas_information_bench/pyproject.toml=90B34254ECB807C545467D8331870EA292CDAC6E492C0B4A3134AA18F01D1F29
  - gas_information_bench/gib/__init__.py=1623E2D8B7AC8174FF50E12392228ABBADA23AE2628A115C31A25D20954824DD
  - gas_information_bench/<eleven .gitkeep files>=E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
failed_checks: []
verdict: approved
next_allowed_task: P2-04
```

## P2-04 执行记录

```text
task_id: P2-04
input_versions:
  - 项目总体规划.md §6.1–§6.3 | workspace_sha256=49E674945540D0A9E077C48A2011D29A6A0C6E1DEF62B3C5B33D72ED062CA7E8
  - docs/工业多传感器气体组分检测_工业应用方向调研报告.md §2.1 | workspace_sha256=2BB8C20E4767E42541B786D1CC13CCDA15906DCAEEFF0F6362F7B9CB224AFA88
  - docs/多传感器多组分气体检测算法创新深度调研报告.md §6 | workspace_sha256=5D7BC2A2F4D9816084BFC892D2C804505620FE13D95CEE7B17975544F6355BF6
  - P2-03 前置 | README status=completed/approved；H2 review_verdict=approved
changed_files:
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - Get-FileHash -Algorithm SHA256 -LiteralPath 项目总体规划.md, docs/工业多传感器气体组分检测_工业应用方向调研报告.md, docs/多传感器多组分气体检测算法创新深度调研报告.md
  - rg -n 'P2-03 \\| completed \\| `approved`|review_verdict: approved' docs/p2/README.md docs/p2/P2_bench规格书.md
  - rg -n 'N2/CO2/O2/Ar|c_N2|c_CO2|c_O2|c_Ar|theta|eta|T|P|RH|声程|gain|baseline|delay|crosstalk|NDIR|超声 Raw|声学 DSP|热导|flow|CH4|闭包|GIB-C4-LR' docs/p2/P2_bench规格书.md
  - rg -n 'y_ndir|y_us_raw|y_ac_dsp|y_tc|y_slow|T/P/RH/q_flow|Raw 与 DSP' docs/p2/P2_bench规格书.md
  - rg -n '草案|提案值|提案前缀|输出根目录提案|待人工填写|在 H2 人工批准前' docs/p2/P2_bench规格书.md
  - rg -n '[ \\t]+$' docs/p2/P2_bench规格书.md
  - git diff --check -- docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - input hash: 0
  - P2-03 prerequisite rg: 0
  - parameter/candidate marker rg: 0
  - modality mapping rg: 0
  - stale approval marker rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - docs/p2/P2_bench规格书.md
artifact_sha256:
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
failed_checks: []
verdict: ready_for_forward_screen
next_allowed_task: P2-05
```

## P2-05 执行记录

```text
task_id: P2-05
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=F33DAC6939845C753220CB484DED99556155707ABA13C436006ADCC298FFE8EA
  - docs/p2/P2_bench规格书.md | P2-04 verdict=ready_for_forward_screen；append-only 文档不对自身内容建立循环哈希
  - hydrogen_ng/docs/references/传感器硬件资料整理.md | workspace_sha256=D1C44BEF8E463809985570BA40E54CE25B2DCA182D41EBB0FC5BCDD66A438AC9
  - tunnel_ventilation/docs/references/传感器硬件资料整理.md | workspace_sha256=5D7FAF30639AB9F5E91324FF7B9B321E1244AB4EEF2044D5232227E7E1C0860E
  - hydrogen_ng/docs/物理模型严格化实施计划.md | workspace_sha256=953887D913D0FF7D9032F0C1D90BFEECA676047D0B206BE0725B56DBA2EE1514
  - syngas/docs/references/co_acoustic_constants.md | workspace_sha256=A2913A114F72DBE9C123F8C2ECD404D4DE9C67CCC5BA129EC55F7B224359304B
changed_files:
  - gas_information_bench/pyproject.toml
  - gas_information_bench/gib/audit/__init__.py
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/tests/test_forward_audit.py
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - rg -n -A 75 -B 5 'P2-05|P2-06' docs/p2/P2执行计划.md
  - python -m pytest -q tests/test_forward_audit.py
  - python -c "from gib.audit.forward import AuditConfig, screen_candidate; results={key: screen_candidate(AuditConfig(candidate_id=key)) for key in ('GIB-C4-LR','GIB-C4-CH4')}; print([(key, item['candidate_verdict'], item['result'].joint_rank, all(value['passed'] for value in item['negative_controls'].values()), item['safety_gate_required']) for key, item in results.items()])"
  - rg -n 'base_condition_id|noise_seed_index|noise_seed' gas_information_bench
  - rg -n '^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)' gas_information_bench
  - rg -n '[ \t]+$' gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - task card rg: 0
  - forward audit tests: 0（7 passed）
  - candidate screen: 0；GIB-C4-LR 与 GIB-C4-CH4 均为 candidate_selected，joint_rank=11，四类负对照均通过；CH4 safety_gate_required=True
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/tests/test_forward_audit.py
  - docs/p2/P2_bench规格书.md §4.4 候选筛选表
artifact_sha256:
  - gas_information_bench/pyproject.toml=71CE666711ECC8758A67D5B35114C229E5B8DDAED3D14B5C413ABF4DE5EE63D0
  - gas_information_bench/gib/audit/__init__.py=C0A8793A19E84E889E08D8A1BD3639438278E36B4BC7CF81F9E188BBA48B7763
  - gas_information_bench/gib/audit/forward.py=12650B418B5027A4FC2EC1540619A2E319FE4D22EDE14156AAD3332416DB81EF
  - gas_information_bench/tests/test_forward_audit.py=DB4194194ADC9CB83A8920A8F94A13D6B67F2326CC6080CE085B08B4872A1A8C
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 初轮联合秩直接使用未缩放 nuisance 列时，1e-5 的 SVD 秩与其他容差不一致；已改为按声明先验尺度缩放 nuisance 列，最终三档容差均为 joint_rank=11。
  - 初轮候选判定错误要求 joint_rank 等于目标维度；已修正为不少于目标维度，最终两个候选均通过纯前向筛选。
  - 两次临时候选探针分别误传 CandidateProfile、误将 AuditResult 直接 JSON 序列化而失败；未改变实现，改用 AuditConfig 并只读取 verdict/审计字段后 exit 0。
verdict: candidate_selected
next_allowed_task: P2-06
```

## P2-06 执行记录

```text
task_id: P2-06
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=F33DAC6939845C753220CB484DED99556155707ABA13C436006ADCC298FFE8EA
  - docs/p2/P2_bench规格书.md | P2-05 primary candidate=GIB-C4-LR；append-only 文档不对自身内容建立循环哈希
  - gas_information_bench/gib/audit/forward.py | workspace_sha256=12650B418B5027A4FC2EC1540619A2E319FE4D22EDE14156AAD3332416DB81EF
  - gas_information_bench/tests/test_forward_audit.py | workspace_sha256=DB4194194ADC9CB83A8920A8F94A13D6B67F2326CC6080CE085B08B4872A1A8C
changed_files:
  - gas_information_bench/gib/audit/__init__.py
  - gas_information_bench/gib/audit/grid.py
  - gas_information_bench/tests/test_grid.py
  - gas_information_bench/configs/p2_s1_grid.json
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -m pytest -q tests/test_grid.py
  - python -m gib.audit.grid
  - python -c "import json; json.load(open('configs/p2_s1_grid.json', encoding='utf-8')); print('valid')"
  - python -c "import json; from gib.audit.grid import grid_summary; actual=json.load(open('configs/p2_s1_grid.json', encoding='utf-8')); expected=json.loads(json.dumps(grid_summary(), ensure_ascii=False, sort_keys=True)); assert actual == expected; print('generated_config_matches_grid_summary')"
  - rg -n 'GIB-S1-(SUF|CRI|INS)' gas_information_bench/configs/p2_s1_grid.json docs/p2/P2_bench规格书.md
  - rg -n 'base_condition_id|noise_seed_index|noise_seed' gas_information_bench
  - rg -n '^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)' gas_information_bench
  - rg -n '[ \t]+$' gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - grid tests: 0（4 passed）
  - grid generation: 0；9 格均生成
  - generated JSON parse: 0（valid）
  - generated config comparison: 0（JSON 规范化后与 grid_summary 完全一致）
  - 9-cell marker rg: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/gib/audit/grid.py
  - gas_information_bench/tests/test_grid.py
  - gas_information_bench/configs/p2_s1_grid.json
  - docs/p2/P2_bench规格书.md §5 S1 信息量与夹角刻度
artifact_sha256:
  - gas_information_bench/gib/audit/__init__.py=C0A8793A19E84E889E08D8A1BD3639438278E36B4BC7CF81F9E188BBA48B7763
  - gas_information_bench/gib/audit/grid.py=F5B43CE11BE8179FAD295B3FCE77A9959CD96A5E7615AE7821B76D16DBC7E14E
  - gas_information_bench/tests/test_grid.py=732E98B1EBC175933FF4474CF482A233017449CE059670E9F554EE8F8968EA1A
  - gas_information_bench/configs/p2_s1_grid.json=F7D834DB51B13F47A8A7716252346B425876ECE3A5195DD61103B41196CB765F
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 初轮 sufficient/high 单元 information_ratio=0.517，超过 0.5 门值；已调整确定性 sufficient noise profile 到 0.35，门值本身未改变，最终该单元 ratio=0.4618。
  - 初轮测试断言误检查固定噪声跨角度，而计划要求固定信息档只改变 coupling；已修正断言轴，最终 9 格全部可达且测试通过。
  - 初次生成时 `gib.audit.__init__` 预先导入 grid 触发 runpy 警告，旧捕获文件将该警告带入 JSON 顶部；已移除 package-level grid import 并清理旧生成前缀，最终生成命令无该警告且 `json.load` exit 0。
  - 初次用 Python 对象直接比较落盘 JSON 与 `grid_summary()` 时，tuple/list 序列化差异导致断言失败；改用 JSON 规范化等价比较后通过，逐格数值与配置 ID 一致。
verdict: grid_frozen
next_allowed_task: P2-07
```

## P2-07 执行记录

```text
task_id: P2-07
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - docs/p2/P2_bench规格书.md §5 S1 3 × 3 | P2-06 verdict=grid_frozen；append-only 文档不对自身内容建立循环哈希
  - gas_information_bench/configs/p2_s1_grid.json | workspace_sha256=F7D834DB51B13F47A8A7716252346B425876ECE3A5195DD61103B41196CB765F
changed_files:
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/gib/audit/s2_s3.py
  - gas_information_bench/tests/test_s2_s3.py
  - gas_information_bench/configs/p2_s2_s3_audit.json
  - gas_information_bench/configs/p2_s2_s3_frozen_evidence.json
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -m gib.audit.s2_s3
  - python -m pytest -q tests/test_s2_s3.py
  - python -m pytest -q
  - python -c "import json; from gib.audit.s2_s3 import audit_summary; actual=json.load(open('configs/p2_s2_s3_frozen_evidence.json',encoding='utf-8')); expected=audit_summary(); assert actual == expected; print('frozen_evidence_matches; verdict=' + expected['verdict'] + '; c4=' + expected['s2']['c4_pre_verdict'])"
  - rg -n 'base_condition_id|noise_seed_index|noise_seed' gas_information_bench
  - rg -n '^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)' gas_information_bench
  - rg -n '[ \t]+$' gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- docs/p2/P2_bench规格书.md gas_information_bench docs/p2/README.md
exit_codes:
  - S2/S3 audit: 0；S2 五项量化检查、S3 三个独立关闭检查和同时关闭负对照均通过；`verdict=pass`，`c4_pre_verdict=eligible_for_P3_test`
  - S2/S3 tests: 0（3 passed）
  - full new-subproject tests: 0（14 passed）
  - frozen evidence comparison: 0（generated evidence matches audit_summary）
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/gib/audit/forward.py | AuditConfig 三个显式 headroom 开关及其前向语义
  - gas_information_bench/gib/audit/s2_s3.py | S2/S3 纯前向审计与证据生成器
  - gas_information_bench/configs/p2_s2_s3_audit.json | 阈值、probe nuisance 和独立开关 profile
  - gas_information_bench/configs/p2_s2_s3_frozen_evidence.json | S2/S3 生成数值与 verdict
  - gas_information_bench/tests/test_s2_s3.py | S2/S3 阈值、独立关闭和冻结证据测试
  - docs/p2/P2_bench规格书.md §6 S2、§7 S3
artifact_sha256:
  - gas_information_bench/gib/audit/forward.py=51DE91424E3E607AB4A10A02FDE1AE2C646725136A1B608BA5E8288C89E68DFD
  - gas_information_bench/gib/audit/s2_s3.py=92815A5271157EBB1BDAD298D3030C69244C963A8FDDF25E3D5A9FDB6EC013C4
  - gas_information_bench/tests/test_s2_s3.py=52E924BA7E16DEA8F437EAEB894867E7FC0CCB5CFADDDEC3B69B0BF8782C5C06
  - gas_information_bench/configs/p2_s2_s3_audit.json=01C113230635075A1159EE87E6A13F4ACA5E9A40A59FABAF0A97232C151C4FAA
  - gas_information_bench/configs/p2_s2_s3_frozen_evidence.json=EB240B0C7EE34875C5D4F163B507301FDEF2E5066D10C072F46BDC775C47B474
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 首轮冻结证据严格比较因线性代数末位漂移失败，差异约 1e-9；已由审计脚本统一数值精度并保留小于 1e-6 的阈值量，未改变任何门值或 verdict。
  - 8 位小数仍有一个值跨舍入边界；已收敛到 6 位小数并重新生成证据，最终严格比较通过。
verdict: pass
next_allowed_task: P2-08
```

## P2-08 执行记录

```text
task_id: P2-08
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - docs/p1/创新点候选集.md §3 | workspace_sha256=DE10F12F5A6D6F87B86983B5901CC6DE9F57ABB8C3CD63768E05FC8DE02C0263
  - docs/p2/P1候选修订记录.md §1–§4 | workspace_sha256=6724D9343C44D37D57E47C4DD99A84A7694C05F5A29C3A6CBF0D6877F1D7F38F
  - docs/多传感器多组分气体检测算法创新深度调研报告.md §3 | workspace_sha256=5D7BC2A2F4D9816084BFC892D2C804505620FE13D95CEE7B17975544F6355BF6
  - 项目总体规划.md §6.4 | workspace_sha256=49E674945540D0A9E077C48A2011D29A6A0C6E1DEF62B3C5B33D72ED062CA7E8
  - P2-07 前置 | README status=completed/pass
changed_files:
  - gas_information_bench/configs/p2_s4_metric_registry.json
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -c "load JSON and assert S4 layers, methods, paired splits/seeds, timing, solver, nested efficiency and pending authorization fields"
  - Test-Path all P2-08 input documents and generated registry
  - rg -n "base_condition_id|noise_seed_index|noise_seed" gas_information_bench
  - rg -n "^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)" gas_information_bench
  - rg -n "[ \t]+$" gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - registry schema and field validation: 0（四层、5 split × 3 seed、嵌套五档、计时、solver 和联合 verdict 均通过）
  - input path validation: 0（6 个输入/产物路径均存在）
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/configs/p2_s4_metric_registry.json | S4 精度、效率、solver 和联合 verdict registry
  - docs/p2/P2_bench规格书.md §8 | S4 CRB、oracle、测量级、强基线和效率协议
  - docs/p2/README.md | P2-08 状态与执行记录
artifact_sha256:
  - gas_information_bench/configs/p2_s4_metric_registry.json=F68817A8C781A8A6DE17FA01BCBAD5842A9570F613406320C6FA6169AD898B66
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 初次辅助路径核对误写为 docs/p1/候选集.md，结果为 False；未进入验收结论，随后按仓库实际路径 docs/p1/创新点候选集.md 复核通过。
  - P1 修订中的非劣带、效率门、硬件 profile 和精确重复次数仍为 pending_authorization/requires_human_value；本任务未填默认值，也未据此运行比较。
verdict: protocol_frozen
next_allowed_task: P2-09
```

## P2-09 执行记录

```text
task_id: P2-09
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - P2-03 前置 | README status=completed/approved；gib-benchmark-1；当前 README sha256=9D4C05038E9D199917F4425D0C3ABD3D7712F7206D452BEDC8FF6FB6DEC7DD6B；pyproject sha256=71CE666711ECC8758A67D5B35114C229E5B8DDAED3D14B5C413ABF4DE5EE63D0
  - P2-06 前置 | README status=completed/grid_frozen；configs/p2_s1_grid.json sha256=F7D834DB51B13F47A8A7716252346B425876ECE3A5195DD61103B41196CB765F
  - P2-08 前置 | README status=completed/protocol_frozen；configs/p2_s4_metric_registry.json sha256=F68817A8C781A8A6DE17FA01BCBAD5842A9570F613406320C6FA6169AD898B66
changed_files:
  - gas_information_bench/configs/p2_data_schema.json
  - gas_information_bench/configs/p2_manifest_schema.json
  - gas_information_bench/configs/p2_split_contract.json
  - gas_information_bench/gib/contract.py
  - gas_information_bench/tests/test_contract.py
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -m pytest -q tests/test_contract.py
  - python -m py_compile gib/contract.py
  - python -m pytest -q
  - python -c "parse p2_data_schema.json, p2_manifest_schema.json and p2_split_contract.json"
  - rg -n "base_condition_id|noise_seed_index|noise_seed" .
  - rg -n "from (hg|sg|tv3|rcdw)|import (hg|sg|tv3|rcdw)" .
  - rg -n "[ \t]+$" gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - contract tests: 0（8 passed）
  - contract module compile: 0
  - full new-subproject tests: 0（22 passed）
  - contract JSON parse: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/configs/p2_data_schema.json | 样本记录与数组层 schema
  - gas_information_bench/configs/p2_manifest_schema.json | manifest 字段与 SHA256 规则；仅契约，不是数据 manifest
  - gas_information_bench/configs/p2_split_contract.json | 5 split、3 partition、group isolation 规则
  - gas_information_bench/gib/contract.py | ID、样本、manifest、DSP provenance、solver、split 和 deployment/oracle 校验
  - gas_information_bench/tests/test_contract.py | P2-09 契约与泄漏拒绝测试
  - docs/p2/P2_bench规格书.md §9 | 数据契约、manifest、ID 与 split
artifact_sha256:
  - gas_information_bench/configs/p2_data_schema.json=6E2A2C95F025913D48911263047B2EB2863320985A6110F69305A3914637CDFA
  - gas_information_bench/configs/p2_manifest_schema.json=A1320C2BD0E808740BB6E9DBF293A8A13FB3ACC497E60B61AB863B360E795D20
  - gas_information_bench/configs/p2_split_contract.json=AED3A8F9BA037F581B0176D50D42382C2ED3E0C09DD39C75AFD208D83D595588
  - gas_information_bench/gib/contract.py=E42C875716E5CA683FD338B9A6B1B6223649D598B3AA6F3CFB601394E5033A17
  - gas_information_bench/tests/test_contract.py=333201701ED70B50DA98D96AAE935FFF99048C946F6D13E94F96B995620EEB6A
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks: []
verdict: contract_frozen
next_allowed_task: P2-10
```

## P2-10 执行记录

```text
task_id: P2-10
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - P2-04 前置 | README status=completed/ready_for_forward_screen；参数表 §3.2–§3.3；append-only 规格书不对自身内容建立循环哈希
  - P2-05 前置 | README status=completed/candidate_selected；forward.py workspace_sha256=51DE91424E3E607AB4A10A02FDE1AE2C646725136A1B608BA5E8288C89E68DFD
  - hydrogen_ng/docs/物理模型严格化实施计划.md | workspace_sha256=953887D913D0FF7D9032F0C1D90BFEECA676047D0B206BE0725B56DBA2EE1514
  - hydrogen_ng/docs/references/传感器硬件资料整理.md | workspace_sha256=D1C44BEF8E463809985570BA40E54CE25B2DCA182D41EBB0FC5BCDD66A438AC9
  - tunnel_ventilation/docs/references/传感器硬件资料整理.md | workspace_sha256=5D7FAF30639AB9F5E91324FF7B9B321E1244AB4EEF2044D5232227E7E1C0860E
  - tunnel_ventilation/docs/references/co2_o2_n2_gas_properties.md | workspace_sha256=1F347722388298F6660FDDB0B8EFF61AD34F4594348EABE733D61AF519391C5E
  - syngas/docs/references/co_acoustic_constants.md | workspace_sha256=A2913A114F72DBE9C123F8C2ECD404D4DE9C67CCC5BA129EC55F7B224359304B
changed_files:
  - gas_information_bench/configs/p2_s5_source_registry.json
  - gas_information_bench/configs/p2_s5_discrepancy_contract.json
  - gas_information_bench/gib/s5_contract.py
  - gas_information_bench/tests/test_s5_contract.py
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
  - gas_information_bench/gib/audit/forward.py | 未修改，仅登记代码 hash 与参数绑定
commands:
  - Get-FileHash -Algorithm SHA256 -LiteralPath <P2-10 registry、contract、module、test、forward.py、P2执行计划>
  - python -m pytest -q tests/test_s5_contract.py
  - python -m py_compile gib/s5_contract.py
  - python -c "parse p2_s5_source_registry.json and p2_s5_discrepancy_contract.json"
  - python -m pytest -q
  - rg -n "base_condition_id|noise_seed_index|noise_seed" .
  - rg -n "from (hg|sg|tv3|rcdw)|import (hg|sg|tv3|rcdw)" .
  - rg -n "[ \\t]+$" gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- gas_information_bench/configs/p2_s5_source_registry.json gas_information_bench/configs/p2_s5_discrepancy_contract.json gas_information_bench/gib/s5_contract.py gas_information_bench/tests/test_s5_contract.py docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - input hash: 0
  - S5 pure interface tests: 0（12 passed）
  - S5 module compile: 0
  - S5 JSON parse: 0
  - full new-subproject tests: 0（34 passed）
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/configs/p2_s5_source_registry.json | 前向数量、来源分类、代码绑定、缺失来源和 verdict registry
  - gas_information_bench/configs/p2_s5_discrepancy_contract.json | delta 签名、off/P5 profile、字段与单位契约
  - gas_information_bench/gib/s5_contract.py | registry 校验、显式单位换算和 discrepancy 接口
  - gas_information_bench/tests/test_s5_contract.py | 来源阻塞、单位换算、off 不变和 P5 禁止注入测试
  - docs/p2/P2_bench规格书.md §10 | S5 来源与 discrepancy 接口规范
artifact_sha256:
  - gas_information_bench/configs/p2_s5_source_registry.json=6FAFFAEBCB91EAC5855C705A153E1F6B5DF53EF8D6D744F9370F3A53E7A64946
  - gas_information_bench/configs/p2_s5_discrepancy_contract.json=1507F9F77301445F3CE46B2A3F90829CDD4ABF41AFAA15BC6FC1E4658338A3E3
  - gas_information_bench/gib/s5_contract.py=3A91013F0B60777BE0F71AEE43DFCD818E15A7B487B7F2E6D568C0A8B6CA364F
  - gas_information_bench/tests/test_s5_contract.py=DB22E46FD64640441AFDF50BBEA7BC25C28B5737B0FAFB2C1BE3C796CB34794F
  - gas_information_bench/gib/audit/forward.py=51DE91424E3E607AB4A10A02FDE1AE2C646725136A1B608BA5E8288C89E68DFD
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - source_complete 未满足：Ar 三项物性、目标 TraceGas filter response、NDIR proxy 标定、声学频率绑定和逐通道噪声标定证据仍缺失；按计划给出 blocked_source_missing，未做邻近数值替代。
verdict: blocked_source_missing
next_allowed_task: P2-11
```

## 2026-08-25 契约一致性修复记录

```text
scope:
  - mixture_id 只由 candidate_id + composition 生成并在样本校验时复算
  - sequence_id 由 mixture_id + sequence_index + sequence_profile_id 生成并复算
  - 同一 split/partition 允许一个 mixture_id 对应多个 sequence_id，跨 partition 仍显式失败
  - slow_channels 固定为 T/P/RH/q_flow；L/gain/baseline/delay/crosstalk 迁至独立 calibration_channels
  - S1 3×3 与 S2/S3 冻结证据由修复后的纯前向实现重新生成
  - P1 候选修订记录删除手工运行数值，只引用自动生成证据
validation:
  - targeted contract/forward/grid/S2-S3/S5 tests: 36 passed
  - S1 grid: 9/9 accessible；信息档与 10/45/80 deg 夹角档均保持原门值
  - S2/S3: pass；C4 前置仍为 eligible_for_P3_test
artifact_sha256:
  - gib/audit/forward.py=D7E2B1F44F9AFD97FDAE9450B13B822BC6210C6BD6E63491C372179AFFF0B7CE
  - gib/contract.py=3638DB011FCD761B4EFE4B288418C2448BA35D85EBAECB82209D7B3F66F4E84E
  - configs/p2_data_schema.json=1FA14AEDA5A384D2EFE0F28682910E304719367352A31221BAD78A8654F3D90E
  - configs/p2_s1_grid.json=D193585A12932C78A8383A73058AFB989318DAB83EAE4953E4C4B74F1C682310
  - configs/p2_s2_s3_frozen_evidence.json=357874A6121DB1770B238BDCDF22B7D5A6A8E78B06D5EDC7C1C312D456FD1E4F
status_effect:
  - P2-05 candidate_selected、P2-06 grid_frozen、P2-07 pass、P2-09 contract_frozen 经重算保持不变
  - P2-10 仍为 blocked_source_missing；本修复不补造物理来源
```

## P2-11 执行记录

```text
task_id: P2-11
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=698864EF3664E75C2D36E540FEA2B091F89401800F946FCA10315F68FE91AB46
  - docs/p2/P2能力复用审计.md | workspace_sha256=9C6B27438B28535B8534A6C50CA4CDC7327F2BCDF6173B4CBA97D7AD7B506AFC
  - P2-03 前置 | README status=completed/approved；独立命名空间 gib-benchmark-1
  - P2-09 前置 | README status=completed/contract_frozen；gib/contract.py workspace_sha256=3638DB011FCD761B4EFE4B288418C2448BA35D85EBAECB82209D7B3F66F4E84E
changed_files:
  - gas_information_bench/pyproject.toml
  - gas_information_bench/gib/freeze.py
  - gas_information_bench/gib/cli.py
  - gas_information_bench/configs/p2_s6_ownership_registry.json
  - gas_information_bench/tests/test_freeze_contract.py
  - gas_information_bench/tests/test_ownership_registry.py
  - gas_information_bench/README.md
  - gas_information_bench/docs/README.md
  - gas_information_bench/outputs/README.md
  - gas_information_bench/outputs/runs/attempts/README.md
  - gas_information_bench/outputs/archive/freezes/README.md
  - README.md
  - docs/p2/P2_bench规格书.md
  - docs/p2/P2执行计划.md
  - docs/p2/README.md
commands:
  - python -m pytest -q tests/test_freeze_contract.py
  - python -m pytest -q
  - python -m pytest --collect-only -q
  - python -m py_compile gib/freeze.py gib/cli.py
  - python -m pip install -e . --no-deps
  - gib --help
  - rg 禁止历史字段、历史私有包 import 和行尾空白
  - git diff --check -- <P2-11 影响文件>
exit_codes:
  - freeze contract tests: 0（5 passed）
  - full new-subproject tests: 0（43 passed）
  - independent test collection: 0（43 tests collected）
  - module compile: 0
  - independent editable install: 0
  - installed CLI help: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - configs/p2_s6_ownership_registry.json | P2-01 全部能力的唯一 owner 与 active/reserved 状态
  - gib/freeze.py | append-only promotion、五类输入 hash、source snapshots 和 manifest 重算
  - gib/cli.py | gib freeze 与 gib verify-freeze
  - outputs/README.md | attempts、freezes、summary 和 reports 的目录纪律
  - docs/p2/P2_bench规格书.md §11 | S6 单一所有者、独立入口与冻结不变量
artifact_sha256:
  - gas_information_bench/pyproject.toml=C46B0A65DE21A4DB3D5B41AD03699D5035A5F187BFD22A09F9DBB217CD56FC84
  - gas_information_bench/gib/freeze.py=22E61F1521513D685E3A633983114E577B94132509BE4993D6D520B7DDE91896
  - gas_information_bench/gib/cli.py=D216A36DB6ECC905DCE3E95C4610597D3C203A389F0DC346CABC256C64EE02E4
  - gas_information_bench/configs/p2_s6_ownership_registry.json=66A5522B7015C1E6AE6F517494467F9B78BC9E4A5640B7AEF9B4C356E5C6DBA5
  - gas_information_bench/tests/test_freeze_contract.py=6B2B62FC4DF69C1977B2DB547BE899A59E47DD60EEDFDD0C15DE2E634587B87E
  - gas_information_bench/tests/test_ownership_registry.py=975D4EC9C9B652C185A2A40531A903CD2421385EB7A644035736F598821DC607
  - gas_information_bench/README.md=FAEF2B023F9658C773FFE024A8C4F0952727516496AD944CAE20F0A4B0AEBC1D
  - gas_information_bench/docs/README.md=B0DE6693B6D5CBDC682FAD4644CDE5A75E1D4D696E781EFA260848240DCCBDC9
  - gas_information_bench/outputs/README.md=A53CF7025B6ECA95C5387690E210989CFBE85CCEAF5C3C100908224B933EE93D
  - README.md=4017D07D14C4D881C5433F9CF92D483B1C4133F3760348FD6992A9A225862924
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks: []
verdict: ownership_frozen
next_allowed_task: P2-12_after_H1_authorized
```

## P2-12 至 P2-14 终态

- P2-12：H1 已授权，候选矩阵为 `matrix_ready`。
- P2-13：S5 已冻结为 `controlled_synthetic` profile，规格和 G3 消费路径为 `ready_for_review`。
- P2-14：设计评审为 `pass`，P2 关闭并允许启动 P3 G3-1。

`source_complete` 只覆盖同条件合成算法比较；目标器件硬件保真和绝对性能仍不属于 P2/P3 声明范围。

## H1 授权与 S5 来源补充记录

```text
date: 2026-08-25
authority: project_owner_delegated_evidence_based_freeze
authorization_verdict: authorized
authorized_values:
  - component P90 non-inferiority bands: N2=0.008, CO2=0.003, O2=0.010, Ar=0.005 mol/mol
  - efficiency reductions: iterations=30%, forward_calls=30%, solver_wall_clock=20%, single_sample_latency=20%
  - maximum regression of other primary costs: 5%
  - independent repeats: 30
  - hardware profile: GIB-HW-WIN-R9-8940HX-RTX5060L-20260825
s5_source_update:
  - Ar molar mass, molar heat capacity and thermal conductivity now have peer-reviewed traceability
  - source blockers reduced from 7 to 4
  - verdict remains blocked_source_missing
validation:
  - targeted S4/S5 tests: 15 passed
  - full new-subproject tests: 46 passed
  - source registry: 14 inventory entries, 15 sources, 4 blockers
artifact_sha256:
  - gas_information_bench/configs/p2_s4_metric_registry.json=C287C9B48A74C420614E617859458E1D750967EF9542AFBB5FBE7918F02A1D5E
  - gas_information_bench/configs/p2_s5_source_registry.json=B53FA053809A6C5ED15A830D649F6CB2F3A3ED611C46C3D71C255F626D95A5F8
  - gas_information_bench/tests/test_s4_authorization.py=109FCB8E1C7A3E3098ABDCD4CD27164242079488B3652376CF11AB23C0685F5D
  - docs/p2/S5来源检索记录.md=3BBB7B996823C8F74027643EB4964ADA6A5555BC56E2073E2F2E2191CF13996B
```

## P2-12 执行记录

```text
task_id: P2-12
hard_gate:
  - P2-02 authorized: passed
inputs:
  - H1 authorized C2/C5 revision
  - P2-06 grid_frozen
  - P2-07 pass / C4 eligible_for_P3_test
  - P2-08 protocol_frozen
  - P2-09 contract_frozen
changed_files:
  - docs/p2/P2_bench规格书.md §12
matrix:
  active: C2-S0, C5-A-CRB-ADAPT, C5-B-FO-MPLSELM, C4-ID-MULTIVIEW
  conditional: C2-IC-RDU-VP, C5-C-CR-PKD, C5-D-FIGS
  non_candidate_framework: C1
  deferred: C3_to_P5
validation:
  - every row has mechanism, control, 3x3 cells, primary/secondary endpoint, gate, negative control, stop path and one next step
  - authorization and matrix direct-judgeability audit: passed
  - full new-subproject tests: 46 passed
  - forbidden historical fields and private imports: no match
  - git diff --check: passed
failed_checks: []
verdict: matrix_ready
next_allowed_task: P2-13_after_P2-10_source_complete
```

## P2-10 合成 profile 授权修订记录

```text
date: 2026-08-25
scope: controlled_synthetic relative algorithm benchmark
authorization: project owner permits continuation without target-device fidelity
claim_boundary:
  - identical profile is mandatory for every candidate and baseline
  - target-hardware absolute performance claims are forbidden
  - real hardware validation remains P5-only
validation:
  - S5 tests: 13 passed
  - full new-subproject tests: 47 passed
  - source registry: source_complete, 14 inventory entries, 16 sources, 0 blockers
artifact_sha256:
  - gas_information_bench/configs/p2_s5_source_registry.json=7573FC247B490BAC2DB5613E534A37814C558DD7EF2B75E6100F5ABC0BFC6995
  - gas_information_bench/gib/s5_contract.py=A66F005101A04662558A64EAC2BA0D2EA6B901CDA2F0419CEC2C53774F8D039A
  - gas_information_bench/tests/test_s5_contract.py=C36D9D25E66CA2A1D58C0288462FF2D92F967C3A0896BCA158DC6E2E6A030C77
  - docs/p2/S5来源检索记录.md=A67787702B4263A4D3B6C6CEEFDEFA9CDE85ABCAF7409B3A5502994B17B109B6
verdict: source_complete
next_allowed_task: P2-13
```

## P2-13 执行记录

```text
task_id: P2-13
changed_files:
  - docs/p2/P2_bench规格书.md §5.3, §13
  - docs/p2/tools/render_s1_grid_table.py
  - docs/p2/generated/s1_grid_table.md
validation:
  - generated S1 table matches frozen grid
  - local links resolve
  - active terminology and TODO/TBD audits pass
  - G3-1 through G3-5 pass/fail paths are reachable
  - full new-subproject tests: 47 passed
artifact_sha256:
  - docs/p2/tools/render_s1_grid_table.py=46E81011C36B6548FF5CCBAEF185000DE4D407CBBB016001AB5E9AA842F56D0E
  - docs/p2/generated/s1_grid_table.md=B89AC664D3EAC68C3AE67EC6ECC0177A048A8337F2C5A45AE1E359C27C5AFAB2
failed_checks: []
verdict: ready_for_review
next_allowed_task: P2-14
```

## P2-14 执行记录

```text
task_id: P2-14
review_file: docs/p2/P2设计评审记录.md
review_scope: controlled_synthetic benchmark specification
checklist:
  - S1 through S6: pass
  - data contract: pass
  - candidate boundaries: pass
  - G3 reachability: pass
  - no pilot generation, training or algorithm-advantage claim: pass
validation:
  - full new-subproject tests: 47 passed
  - generated table and local links: passed
  - forbidden runtime fields and private imports: no match
  - git diff --check: passed
artifact_sha256:
  - docs/p2/P2设计评审记录.md=7DB1A641E0DB6D9B42728AD890AA9DCC81547EB1FEA0D012A191C9DB04D93627
failed_checks:
  - combined link-check command was parsed by PowerShell because it embedded newlines; per-file one-line checks passed
  - initial status assertion used Markdown backticks that PowerShell interpreted; ASCII-token rerun passed
  - placeholder scan first matched the execution record itself; wording was corrected and rerun had no match
verdict: pass
next_allowed_task: P3-G3-1
```

## P2 关闭审查修复与重新验收

```text
task_id: P2-close-review-remediation
changed_files:
  - gas_information_bench/pyproject.toml
  - gas_information_bench/configs/__init__.py
  - gas_information_bench/configs/p2_data_schema.json
  - gas_information_bench/gib/contract.py
  - gas_information_bench/gib/freeze.py
  - gas_information_bench/gib/s5_contract.py
  - gas_information_bench/gib/audit/s2_s3.py
  - gas_information_bench/tests/test_contract.py
  - gas_information_bench/tests/test_freeze_contract.py
  - gas_information_bench/tests/test_package_resources.py
  - gas_information_bench/tests/installed_package_smoke.py
  - gas_information_bench/README.md
  - docs/p2/P2_bench规格书.md
  - docs/p2/P2设计评审记录.md
validation:
  - targeted contract/freeze/package/S5 tests: 31 passed
  - full new-subproject tests: 52 passed in 73.97s
  - non-editable wheel installation smoke: passed outside source tree
  - forbidden historical fields: no match
  - private historical package imports: no match
  - GIB-FREEZE-P2-20260825-01 verification: 23 inputs, 3 source snapshots, 2 evidence files
failed_checks:
  - initial resource-package alias was importable only after installation and caused 10 contract-test failures; replaced with the existing configs directory as the sole source/install resource package, then rerun passed
verdict: pass
next_allowed_task: P3-G3-1
```

## P2 最终交接

1. P2 verdict 与日期：`pass`，2026-08-25。
2. 关键文档与 SHA256：`P2_bench规格书.md=BC3583EA6EAB7AD358214B36B2AFED0B52806DAE50D79C5B9B78ACFCEF237CC7`；`P2能力复用审计.md=9C6B27438B28535B8534A6C50CA4CDC7327F2BCDF6173B4CBA97D7AD7B506AFC`；`P1候选修订记录.md=606E664AC87E3B723165D6E73B42EEDAF608FFF38000DAF5779EDBE8CC183C61`；`P2设计评审记录.md=A9B4A53556ABA9639C4429B1E724462502D92C9B11B41CE19C68F2F6C3B2F5AE`。
3. 新子工程身份：目录 `gas_information_bench/`；包 `gib`；schema `gib-benchmark-1`；ID 前缀 `GIB-M`、`GIB-Q`。
4. 3 × 3 刻度：配置 ID 为 `GIB-S1-{SUF,CRI,INS}-{HIG,MED,LOW}`；CRB/angle 产物 `gas_information_bench/configs/p2_s1_grid.json`，SHA256 `D193585A12932C78A8383A73058AFB989318DAB83EAE4953E4C4B74F1C682310`；正式 evidence manifest SHA256 `B7B8176B523DAC0F2BCC5CCDF8FF4F0874773C3772A200CEEEEF3C1EFA197321`。
5. 允许进入 P3 的第一个动作：仅 G3-1 物理前向单测与负对照。
6. 风险或未通过项：`[]`。
