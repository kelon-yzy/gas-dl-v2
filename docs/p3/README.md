# P3 执行状态

> 当前阶段：W6 已完成；G3-4 已冻结但 `gate_verdict=fail`，P3 按预注册路径返回 P1/P2，不进入 P4 或正式数据集阶段。
> 计划依据：[P3 执行计划](./P3执行计划.md)
> 最终评审：[P3 评审记录](./P3评审记录.md)
> 声明边界：仅限 `controlled_synthetic` 同条件相对比较。

## 任务状态

| 任务 | 状态 | verdict | 下一任务 |
|---|---|---|---|
| P3-00 | completed | `handoff_status=valid` | P3-01 |
| P3-01 | completed | `gate_verdict=pass` | P3-02 |
| P3-02 | completed | `gate_verdict=pass` | P3-03 |
| P3-03 | completed | `pilot_plan_status=frozen` | P3-04 |
| P3-04 | completed | `gate_verdict=pass` | P3-05 |
| P3-05 | completed | `gate_verdict=pass` | P3-06 至 P3-09 |
| P3-06 | completed | `c2_preflight=pass` | P3-10、P3-12 |
| P3-07 | completed | `candidate_verdict=reject` | P3-13 |
| P3-08 | completed | `candidate_verdict=reject` | P3-13 |
| P3-09 | completed | `candidate_verdict=reject` | candidate terminal |
| P3-10 | completed | `candidate_verdict=reject` | P3-13 |
| P3-11 | completed | `candidate_verdict=not_activated` | P3-13 |
| P3-12 | completed | `candidate_verdict=reject` | P3-13 |
| P3-13 | completed | `gate_verdict=fail; p3_verdict=return_to_P1_P2` | stop P3-14 至 P3-16 |

## P3-00 执行记录

```text
task_id: P3-00
status: completed
input_freezes:
  - GIB-FREEZE-P2-20260825-01 | evidence_manifest_sha256=B7B8176B523DAC0F2BCC5CCDF8FF4F0874773C3772A200CEEEEF3C1EFA197321
changed_files:
  - gas_information_bench/configs/p3_execution_registry.json
  - gas_information_bench/pyproject.toml
  - docs/p3/README.md
commands:
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P2-20260825-01 | exit=0
  - P2 evidence manifest 23 个输入逐项重算工作区 SHA256 | exit=0
  - PYTHONNOUSERSITE=1 python -c <dependency and CUDA fingerprint> | exit=0
  - Get-CimInstance <OS CPU GPU memory fingerprint> | exit=0
  - PYTHONNOUSERSITE=1 python -m pytest -q | exit=0
evidence:
  - h3_verdict=pass
  - p2_next_allowed_task=P3-G3-1
  - p2_freeze_verify=23 inputs, 3 source snapshots, 2 evidence files
  - p2_workspace_hashes=23/23 match
  - hardware_profile=GIB-HW-WIN-R9-8940HX-RTX5060L-20260825
  - full_subproject_tests=52 passed
  - pilot_data_generated=false
  - model_training_run=false
  - candidate_result_generated=false
verdict:
  gate_verdict: not_applicable
  candidate_verdict: not_applicable
  handoff_status: valid
scope:
  - controlled_synthetic_relative_comparison_only
failed_checks:
  - 初次全量测试递归收集 freeze 内同名测试快照并产生 3 个 collection error；将 pytest testpaths 锁定为 tests 后通过
  - 初次依赖指纹受用户 site-packages 与沙箱读取权限影响；锁定 PYTHONNOUSERSITE=1 后科学计算栈可完整导入
next_allowed_task: P3-01
```

## P3-01 执行记录

```text
task_id: P3-01
status: completed
input_freezes:
  - GIB-FREEZE-P2-20260825-01 | evidence_manifest_sha256=B7B8176B523DAC0F2BCC5CCDF8FF4F0874773C3772A200CEEEEF3C1EFA197321
changed_files:
  - gas_information_bench/configs/p3_g3_1_forward.json
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/gib/cli.py
  - gas_information_bench/tests/test_forward_audit.py
  - gas_information_bench/tests/test_package_resources.py
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-G3-1-20260825-01/
  - gas_information_bench/outputs/archive/freezes/GIB-FREEZE-P3-G3-1-20260825-01/
commands:
  - python -m pytest -q tests/test_forward_audit.py tests/test_s2_s3.py tests/test_package_resources.py | exit=0
  - python -m gib.cli audit-forward --config configs/p3_g3_1_forward.json --attempt-dir outputs/runs/attempts/GIB-ATTEMPT-P3-G3-1-20260825-01 | exit=0
  - 重复执行同一 audit-forward attempt 路径 | exit=1，按 append-only 预期拒绝
  - python -m pytest -q | exit=0
  - python -m gib.cli freeze <P3-01 完整输入> | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-G3-1-20260825-01 | exit=0
evidence:
  - deterministic_repeat=pass
  - deterministic_observation_sha256=B645D4F5DF928ADD0E3A4612EBC7DDAFA77B2126AB31335209C267D4FDE37C0D
  - component_perturbations=N2/CO2/O2/Ar 全部 pass
  - target_signal_scaling=pass
  - modality_off=pass
  - s3_switches=pass
  - noise_monotonicity=pass
  - all_off_negative_control=pass, target_profile_eligible=false
  - target_tests=16 passed
  - full_subproject_tests=55 passed
  - freeze_verify=16 inputs, 3 source snapshots, 2 evidence files
  - evidence_manifest_sha256=D31D76FF9672A07797A253C39652E9168A3AE50B163B3A079A694E8833AEAA26
verdict:
  gate_verdict: pass
  candidate_verdict: not_applicable
scope:
  - controlled_synthetic_relative_comparison_only
failed_checks:
  - 初始 target_signal_scale=1.25 使 Ar 基线位于 simplex 边界，中心差分负扰动越界；在正式 attempt 前将技术审计点改为 1.2，未改变 P2 门值、profile 或网格
next_allowed_task: P3-02
```

## P3-02 执行记录

```text
task_id: P3-02
status: completed
input_freezes:
  - GIB-FREEZE-P3-G3-1-20260825-01 | evidence_manifest_sha256=D31D76FF9672A07797A253C39652E9168A3AE50B163B3A079A694E8833AEAA26
changed_files:
  - gas_information_bench/gib/audit/grid.py
  - gas_information_bench/gib/cli.py
  - gas_information_bench/tests/test_grid.py
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-G3-2-20260825-01/
  - gas_information_bench/outputs/archive/freezes/GIB-FREEZE-P3-G3-2-20260825-01/
commands:
  - python -m pytest -q tests/test_grid.py | exit=0
  - python -m gib.cli audit-grid --config configs/p2_s1_grid.json --g3-1-freeze outputs/archive/freezes/GIB-FREEZE-P3-G3-1-20260825-01 --attempt-dir outputs/runs/attempts/GIB-ATTEMPT-P3-G3-2-20260825-01 | exit=0
  - 重复执行同一 audit-grid attempt 路径 | exit=1，按 append-only 预期拒绝
  - python docs/p2/tools/render_s1_grid_table.py --check | exit=0
  - python -m pytest -q | exit=0
  - python -m gib.cli freeze <P3-02 完整输入> | 首次 exit=1，修正长路径输入后 exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-G3-2-20260825-01 | exit=0
evidence:
  - unique_reachable_cells=9/9
  - information_bands=pass
  - angle_bands=10/45/80 deg within 5 deg tolerance
  - noise_information_monotonicity=pass
  - frozen_json_value_match=pass
  - generated_markdown_exact_match=pass
  - full_subproject_tests=57 passed
  - forbidden_legacy_fields=no runtime match
  - historical_private_imports=no match
  - freeze_verify=14 inputs, 3 source snapshots, 2 evidence files
  - evidence_manifest_sha256=461A720CCD1952FEB3C252CFEF0D7DEF01019B52DB4192C657BC82EA105FC9F8
verdict:
  gate_verdict: pass
  candidate_verdict: not_applicable
scope:
  - controlled_synthetic_relative_comparison_only
failed_checks:
  - 多次 BLAS 复算可产生约 1e-10 量级浮点差；审计采用命名容差 rtol=1e-9、atol=1e-12，分类、ID 与结构仍严格相等；正式 attempt 的最大误差为 0
  - 首次 freeze 将上一 freeze manifest 作为嵌套输入快照，触发 Windows 长路径限制；事务 staging 已清理，G3-1 hash 保留在 grid report provenance，重试未重复嵌套 freeze 文件
next_allowed_task: P3-03
```

## P3-03 执行记录

```text
task_id: P3-03
status: completed
input_freezes:
  - GIB-FREEZE-P3-G3-1-20260825-01 | evidence_manifest_sha256=D31D76FF9672A07797A253C39652E9168A3AE50B163B3A079A694E8833AEAA26
  - GIB-FREEZE-P3-G3-2-20260825-01 | evidence_manifest_sha256=461A720CCD1952FEB3C252CFEF0D7DEF01019B52DB4192C657BC82EA105FC9F8
changed_files:
  - gas_information_bench/configs/p3_pilot_plan.json
  - gas_information_bench/gib/common/io.py
  - gas_information_bench/gib/sim/pilot.py
  - gas_information_bench/gib/sim/packaging/arrays.py
  - gas_information_bench/gib/pipeline/raw_dsp.py
  - gas_information_bench/gib/pipeline/dataset.py
  - gas_information_bench/gib/contract.py
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/gib/cli.py
  - gas_information_bench/pyproject.toml
  - gas_information_bench/tests/test_array_io.py
  - gas_information_bench/tests/test_raw_dsp.py
  - gas_information_bench/tests/test_pilot_generation.py
  - gas_information_bench/tests/test_package_resources.py
  - gas_information_bench/tests/installed_package_smoke.py
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-DRYRUN-20260825-01/
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-DRYRUN-20260825-02/
commands:
  - python -m gib.cli pilot-generate <G3-1/G3-2 inputs> --dry-run <attempt-01> | exit=0
  - python -m gib.cli pilot-generate <G3-1/G3-2 inputs> --dry-run <attempt-02> | exit=0
  - 重复执行 attempt-01 路径 | exit=1，按 append-only 预期拒绝
  - python -m pytest -q | exit=0
  - python tests/installed_package_smoke.py | exit=0
evidence:
  - pilot_plan_status=frozen
  - grid_cells=9/9
  - dry_run_mixtures=27
  - dry_run_sequences=54
  - dry_run_artifacts=594
  - deterministic_files=602/602 paths and SHA256 match across two attempts
  - mixture_to_sequence_cardinality=2 for every mixture
  - split_rows=270, five splits, no group overlap
  - nested_train_prefixes=10/25/50/75/100 pass
  - raw_dsp_provenance=three hashes independently enforced
  - deployment_oracle_physical_separation=pass
  - atomic_write_positive_and_failure_paths=pass
  - full_subproject_tests=65 passed
  - non_editable_wheel_smoke=pass
verdict:
  gate_verdict: not_applicable
  candidate_verdict: not_applicable
  pilot_plan_status: frozen
scope:
  - technical_generation_validation_only
failed_checks: []
next_allowed_task: P3-04
```

## P3-04 执行记录

```text
task_id: P3-04
status: completed
input_freezes:
  - GIB-FREEZE-P3-G3-1-20260825-01 | evidence_manifest_sha256=D31D76FF9672A07797A253C39652E9168A3AE50B163B3A079A694E8833AEAA26
  - GIB-FREEZE-P3-G3-2-20260825-01 | evidence_manifest_sha256=461A720CCD1952FEB3C252CFEF0D7DEF01019B52DB4192C657BC82EA105FC9F8
changed_files:
  - gas_information_bench/configs/p3_pilot_plan_v2.json
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-PILOT-20260825-01/
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-PILOT-20260825-02/
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-DRYRUN-V2-20260825-01/
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-PILOT-V2-20260825-01/
  - gas_information_bench/outputs/archive/freezes/GIB-FREEZE-P3-PILOT-20260825-01/
  - gas_information_bench/outputs/archive/freezes/GIB-FREEZE-P3-PILOT-20260825-02/
commands:
  - python -m gib.cli pilot-generate <v1 pilot> | exit=0
  - python -m gib.cli freeze <v1 pilot> | exit=0
  - P3-05 前逐 split、逐 grid cell 覆盖审计 | exit=1，v1 不作为下游输入
  - python -m gib.cli pilot-generate <v2 dry-run> --dry-run | exit=0
  - python -m gib.cli pilot-generate <v2 pilot> | exit=0
  - python -m pytest -q | exit=0
  - python -m gib.cli freeze <v2 pilot> | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-PILOT-20260825-02 | exit=0
evidence:
  - pilot_integrity=pass
  - v2_mixtures=180
  - v2_sequences=360
  - v2_artifacts=3960
  - grid_cells=9/9
  - per_split_partition_groups=126/27/27 train/val/test
  - per_cell_partition_groups=14/3/3 train/val/test
  - freeze_verify=23 inputs, 3 source snapshots, 3969 evidence files
  - evidence_manifest_sha256=123B2BBA153A64D6E2E3814830F1EBDDBB1588918EB2B4B5AFB2DA5909642B29
verdict:
  gate_verdict: pass
  candidate_verdict: not_applicable
scope:
  - controlled_synthetic_pilot
failed_checks:
  - v1 pilot 虽通过文件完整性，但部分 split/grid cell 的 val 或 test 组数为 0；保留其 attempt 与 freeze 作为非下游证据
  - 在任何模型拟合前冻结 v2 分层 split 契约；未修改 P2 门值、grid、split ID 或 seed
next_allowed_task: P3-05
```

## P3-05 执行记录

```text
task_id: P3-05
status: completed
input_freezes:
  - GIB-FREEZE-P3-PILOT-20260825-02 | evidence_manifest_sha256=123B2BBA153A64D6E2E3814830F1EBDDBB1588918EB2B4B5AFB2DA5909642B29
changed_files:
  - gas_information_bench/configs/p3_baseline_plan.json
  - gas_information_bench/gib/pipeline/baseline.py
  - gas_information_bench/gib/cli.py
  - gas_information_bench/tests/test_baseline.py
  - gas_information_bench/tests/test_package_resources.py
  - gas_information_bench/outputs/runs/attempts/GIB-ATTEMPT-P3-G3-3-20260825-01/
  - gas_information_bench/outputs/archive/freezes/GIB-FREEZE-P3-G3-3-20260825-01/
commands:
  - PYTHONNOUSERSITE=1 python -m gib.cli run-baselines <frozen v2 pilot> | exit=0
  - python -m pytest -q --basetemp=outputs/test-tmp-p305-final -p no:cacheprovider | exit=0
  - python -m gib.cli freeze <P3-05 inputs> | 首次 exit=1，修正长路径输入后 exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-G3-3-20260825-01 | exit=0
evidence:
  - exact_runtime_lock=Python 3.13.11, NumPy 2.5.0, scikit-learn 1.7.1, XGBoost 3.2.0, PyTorch 2.11.0+cu128
  - formal_runs=75
  - metric_rows=3240
  - crb_rows=540
  - negative_controls=4/4 pass
  - critical_cells_passing=3/3
  - GIB-S1-CRI-HIG oracle_r2_gap=4.340509, p90_gap_branch=pass
  - GIB-S1-CRI-MED oracle_r2_gap=1.534248, p90_gap_branch=pass
  - GIB-S1-CRI-LOW oracle_r2_gap=1.487886, p90_gap_branch=pass
  - full_subproject_tests=67 passed
  - freeze_verify=13 inputs, 3 source snapshots, 2 evidence files
  - evidence_manifest_sha256=82B84A6E4FAEF77EB56D7E6B57797270FD9C91C5C384198E83DAF7D4C7711E69
verdict:
  gate_verdict: pass
  baseline_sufficient: true
  candidate_verdict: not_applicable
scope:
  - controlled_synthetic_relative_comparison_only
failed_checks:
  - 首次 runtime lock 使用 distribution metadata，环境内多份 NumPy 元数据与实际导入版本不一致；在任何数据读取和 fit 前失败，改为锁定实际导入模块版本
  - 首次全量测试无法访问系统 pytest 临时目录；改用仓库内独立 basetemp 后 67 项全部通过
  - 首次 freeze 嵌套复制 pilot freeze manifest 触发 Windows 长路径限制；baseline 结果已绑定 pilot freeze ID 与 dataset manifest hash，冻结输入改为 v2 pilot 配置
  - Ridge 报告病态矩阵警告；结果未删除、未替换模型，G3-3 由预注册门值直接裁决
next_allowed_task: P3-06_to_P3-09
```

## P3-08 暂停记录

```text
task_id: P3-08
task_status: in_progress
implementation_status: completed
formal_attempt_status: interrupted_not_promoted
interrupted_run:
  - requested_attempt_id=GIB-ATTEMPT-P3-C5B-20260825-01
  - command_exit=1
  - valid_attempt_directory_created=false
  - freeze_created=false
  - resumable=false
cleanup:
  - interrupted staging directory removed
reason:
  - frozen execution profile requires about 15750 repeated model fits and about 1890000 predict calls
  - a single uninterrupted run exceeded one hour and had no checkpoint or resume support
required_before_retry:
  - apply P3-08 v2 execution profile: 1620 fits and 108000 timed predict calls
  - retain 30 independent repeats, three batch sizes and paired randomized order for FO-MPLSELM versus the validation-selected reference at 100% fraction
  - bind training total cost to preprocess + fit + validation and record exact execution fingerprint
  - require append-only checkpoint/resume with completed-unit hashes, progress counters and ETA
next_allowed_task: P3-08_execution_optimization
```

## W4 证据闭环状态

```text
task_id: P3-06
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C2-S0-20260825-01 | status=complete | c2_preflight=pass
freeze:
  - GIB-FREEZE-P3-C2-S0-20260826-01 | evidence_manifest_sha256=B6450C29EFAA58E82DA0B668503D1B6239AB388C405FC7B1503D8A748304817F
activated_tasks:
  - P3-10
  - P3-12
evidence_closure:
  - freeze_verify=7 inputs, 3 source snapshots, 2 evidence files
  - status_page_synced=true
failed_checks:
  - 初次 freeze 未设置 PYTHONNOUSERSITE=1，CLI 导入用户 site-packages 时权限失败；未创建 freeze，按相同输入和 freeze ID 在锁定环境中重试成功
next_allowed_task: P3-10_and_P3-12
```

```text
task_id: P3-09
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C4-20260825-02 | status=complete | candidate_verdict=reject
freeze:
  - GIB-FREEZE-P3-C4-20260826-01 | evidence_manifest_sha256=1BD106955344A3912BA5CA45EBF814679D4A74275EFBAEDE3C4C27E00F44F159
superseded_attempt:
  - GIB-ATTEMPT-P3-C4-20260825-01 | 保留但不作为下游结论；第二次 attempt 修正逐单元 OOD 统计
evidence_closure:
  - freeze_verify=5 inputs, 3 source snapshots, 2 evidence files
  - status_page_synced=true
next_allowed_task: candidate_terminal_reject
```

## P3-08 v2 执行器修正记录

```text
task_id: P3-08_execution_optimization
task_status: completed
implementation_status: completed
execution_contract:
  - scientific_model_fits=525
  - formal_timing_additional_fits=870
  - negative_control_fits=225
  - total_fits=1620
  - formal_predict_calls=108000
  - timing_scope=100_percent_fraction_FO-MPLSELM_vs_validation_selected_reference
  - training_wall_clock=preprocess_plus_fit_plus_validation
  - checkpoint_unit=split_id_seed_training_fraction
  - resume_requires_exact_identity_and_payload_hash=true
  - paired_thread_limit=1
  - device_track=cpu
  - bootstrap=mixture_id_vectorized_equal_and_variable_group_paths
validation:
  - target_tests=10 passed
  - full_subproject_tests=92 passed
  - installed_package_smoke=pass
formal_attempt_created=false
next_allowed_task: P3-08_formal_retry
```

## W4-W6 正式候选执行与 G3-4 记录

```text
task_id: P3-07
task_status: completed
implementation_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C5A-20260826-02 | status=complete | candidate_verdict=reject
freeze:
  - GIB-FREEZE-P3-C5A-20260826-03 | evidence_manifest_sha256=7281DEABAE41BF4F9C7164ECC7F06C5EA991B2455184C41B0BA21EC02C2BAA58
evidence:
  - observation_count=810
  - coverage=9 cells, 135 grid-cell/split/seed combinations
  - candidate=crb_dynamic_modality
  - raw_cost_pass=true
  - ni_pass=false
  - nr5_pass=true
  - stage_b=status=not_run, reason=stage_a_gate_failed
negative_controls:
  - random_stop=joint_gate_failed
  - equal_length_fixed=joint_gate_failed
  - crb_rank_shuffle=joint_gate_failed
commands:
  - python -m pytest -q tests/test_adaptive_sampling.py tests/test_package_resources.py | exit=0
  - python -m gib.cli run-adaptive-sampling ... | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-C5A-20260826-03 | exit=0
next_allowed_task: P3-13
```

```text
task_id: P3-08
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C5B-20260826-01 | status=complete | candidate_verdict=reject
freeze:
  - GIB-FREEZE-P3-C5B-20260826-01 | evidence_manifest_sha256=C7CDA5AF4364279A5199922DCECD403DC27357CA0A6360C3E43BE98845971F46
execution_counts:
  - checkpoint_units=75
  - scientific_model_fits=525
  - formal_timing_additional_fits=870
  - negative_control_fits=225
  - formal_predict_calls=108000
evidence:
  - timing_training_wall_clock_reduction=0.998260; 95% CI=[0.998239, 0.998275]
  - timing_batch_size_1_latency_reduction=0.909604; 95% CI=[0.907968, 0.911093]
  - precision_non_inferiority=false
  - first_target_fraction=null
  - negative_controls_pass=false
  - selected_reference_distribution=12 split-seed use xgboost_strong_table; 3 use PLS，效率结论按配对总体报告
reason: precision non-inferiority failed
commands:
  - python -m gib.cli run-data-efficiency ... | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-C5B-20260826-01 | exit=0
next_allowed_task: P3-13
```

```text
task_id: P3-10
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C2-IC-RDU-VP-20260826-02 | status=complete | candidate_verdict=reject
freeze:
  - GIB-FREEZE-P3-C2-IC-RDU-VP-20260826-03 | evidence_manifest_sha256=351B2034C73C3E464AC8E8CDA183FDD8D9BCE0FAA6C9DFE90B3A6ADFEA73CC7F
evidence:
  - trial_count=405
  - grid_split_seed_count=135
  - component_N2_CO2_O2_Ar=NI pass; E20/E30/NR5 fail
  - cell_pass_count=0
  - negative_control_cell_label_shuffle=failed
  - equivalent_core_ablations=[]
  - learned_update_scale=0.5×3; 1.0×6; 1.25×6 split-seed
activation:
  - source=GIB-FREEZE-P3-C2-S0-20260826-01
  - activation_gate=c2_preflight=pass
commands:
  - python -m gib.cli run-ic-rdu-vp ... | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-C2-IC-RDU-VP-20260826-03 | exit=0
next_allowed_task: P3-13
```

```text
task_id: P3-11
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C5C-20260826-01 | status=complete | candidate_verdict=not_activated
freeze:
  - GIB-FREEZE-P3-C5C-20260826-01 | evidence_manifest_sha256=90D4A1AD5BF00CBDB635B42D452A72C88730B46F08DF0B48B3117F4E10663D38
evidence:
  - row_count=810
  - coverage=9 cells, 5 splits, 3 seeds
  - teacher_NI=CO2 failed; N2/O2/Ar passed
  - teacher_relative_improvement_95_percent_lower_ci= N2 -0.128857; CO2 -0.260268; O2 -0.027115; Ar -0.045561
  - teacher_gate_pass=false
  - student_status=not_run
reason: teacher activation gate failed; student was not trained
commands:
  - python -m gib.cli run-teacher-preflight ... | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-C5C-20260826-01 | exit=0
next_allowed_task: P3-13
```

```text
task_id: P3-12
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-C5D-FIGS-20260826-01 | status=complete | candidate_verdict=reject
freeze:
  - GIB-FREEZE-P3-C5D-FIGS-20260826-01 | evidence_manifest_sha256=F25DDB7E7F0C1A78303399ECF63E4AC8E111C36FEB2337712271E985F92BD098
evidence:
  - trial_count=405
  - grid_split_seed_count=135
  - route_regions=4
  - route_coverage=all regions have 5 splits and seeds 101/202/303
  - physical_rule_route=NI pass; E20 fail
  - negative_control_region_label_shuffle=failed
  - fixed_strongest_solver_control=failed
commands:
  - python -m gib.cli run-figs ... | exit=0
  - python -m gib.cli verify-freeze outputs/archive/freezes/GIB-FREEZE-P3-C5D-FIGS-20260826-01 | exit=0
next_allowed_task: P3-13
```

## P3-13 G3-4 候选晋级评审

```text
task_id: P3-13
task_status: completed
formal_attempt:
  - GIB-ATTEMPT-P3-G3-4-20260826-02 | status=complete
freeze:
  - GIB-FREEZE-P3-G3-4-20260826-03 | evidence_manifest_sha256=3CAF2F9D40CC0E29C6D429D01DA3AC3BFF2328C53954612220B8D928F50D4010
evidence_closure:
  - all_active_paths_terminal=true
  - all_triggered_conditional_paths_terminal=true
  - enter_P4_count=0
  - freeze_verify=14 inputs（含 7 个候选 evidence manifest）, 4 source snapshots, 2 evidence files
  - candidate_record=docs/p3/P3候选晋级记录.md
verdict:
  gate_verdict: fail
  candidate_verdict: not_applicable
  p3_verdict: return_to_P1_P2
failed_checks:
  - C5-A candidate NI failed
  - C5-B precision NI failed despite timing reductions
  - C4 OOD/shared representation gates failed
  - C2-IC-RDU-VP did not clear E20/E30/NR5; C5-D did not clear E20
  - C5-C teacher activation failed, so student remained not_run
next_allowed_task: return_to_P1_P2
```

P3-13 由 `review-candidates` 校验并读取 7 个候选 freeze 后派生 summary，G3-4 freeze 同时绑定全部上游 evidence manifest。未调整门值、未新增 conditional 行。根据 G3-4 失败路径，P3-14 至 P3-16 不得启动。
