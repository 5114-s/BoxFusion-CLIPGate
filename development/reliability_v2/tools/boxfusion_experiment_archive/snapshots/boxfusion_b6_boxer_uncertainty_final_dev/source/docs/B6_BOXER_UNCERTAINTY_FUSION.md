# B6 + Selective Boxer 不确定性感知多视角融合

## 目标与边界

这条隔离路线只验证一个新增模块：在冻结的 B6 + Selective Boxer G0
基础上，用 Boxer 的逐 proposal 标量 aleatoric confidence 调整已有 Top-K
可靠视角融合权重。其他 proposal、关联、Selective Boxer 门控、B6 质量分数、
Soft-NMS、`score_thresh=0.4`、`minimum_extent=0.4` 和 Top-K 数量均保持不变。

隔离代码目录为：

```text
/data/ZhaoX/OVM3D-Dett/boxfusion_b6_boxer_uncertainty_dev
```

当前冻结对照为：

| 对照 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| Selective Boxer G0，固定10场 | 44.6302 | 40.8154 | 17.1297 |
| Selective Boxer G0，完整100场 | 40.2787 | 35.4508 | 15.2181 |

Boxer AleHead 对每个 proposal 只预测一个共享的 `log(sigma^2)`，因此本模块
不是 7-DoF 协方差融合，也不是 Kalman 融合。准确名称是：

```text
Boxer scalar aleatoric-confidence-aware Top-K fusion
```

## 融合公式

Boxer 提供：

```text
q = 1 / (1 + exp(logvar))
```

已有可靠视角权重记为 `w_base`。对真正通过 Selective Boxer G0 门控并采用
Boxer 几何的行：

```text
w_uncertainty = w_base * clip(q, 0.05, 1.0)
```

CuTR fallback 行、不合法或缺失的 confidence 使用中性因子 `1.0`。调整后的
权重同时用于稳定 Top-K 排序、加权框初始化、旋转视角选择以及后续 CUDA
融合目标。最终仍按均值归一化，使优化器中的总权重尺度与原路径一致。
不确定性因子有意在原 `minimum_weight=0.05` floor 之后相乘，因此理论最小
原始权重可到 `0.0025`；随后只做均值归一化，不再次截断，以免抹掉低质量视角
之间的 Boxer 置信度差异。

## 严格消融

| 阶段 | 相比上一阶段唯一变化 | 输出预期 |
|---|---|---|
| U0 `u0_control` | 无，重新运行冻结 G0 | 与历史 G0 数值一致 |
| U1 `u1_observer` | 计算并记录不确定性反事实 | 与 U0 数值一致 |
| U2 `u2_active` | 将不确定性权重用于融合 | 允许几何和下游 B6 分数变化 |

这里的 U1 observer 只属于融合模块；`lifting.boxer.mode` 在三组中始终为
`active`，否则就不再是 G0 对照。

## 固定10场运行

当前有其他 GPU 实验时，只保留代码，不要同时启动以下命令。GPU 空闲后运行：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_boxer_uncertainty_dev
bash scripts/run_scannet_b6_boxer_uncertainty_paired.sh 0,1
```

也可以逐项运行，便于断点续跑：

```bash
bash scripts/run_scannet_b6_boxer_uncertainty.sh u0_control 0,1
bash scripts/run_scannet_b6_boxer_uncertainty.sh u1_observer 0,1
bash scripts/run_scannet_b6_boxer_uncertainty.sh u2_active 0,1
bash scripts/audit_scannet_b6_boxer_uncertainty.sh
bash scripts/report_scannet_b6_boxer_uncertainty.sh
```

每个 profile 使用独立的预测、日志、诊断和评估目录，运行指纹也包含模式、
场景列表、配置与新增源码，因此不会错误复用其他消融结果。

## 进入100场的门槛

只有固定10场同时满足下列条件才运行完整100场：

1. U0 与历史 G0 的10个场景全部通过数值恒等审计（框最大绝对漂移
   `<=1e-4`，score 漂移 `<=1e-6`，用于容纳独立 GPU 运行的原子归约漂移）。
2. U1 与 U0 的10个场景全部通过相同数值审计，且
   `applied_to_fusion=false`。
3. confidence 非法计数为0；CuTR fallback 的 uncertainty factor 全部为1。
4. uncertainty 至少影响5%的可融合实例，否则覆盖率不足。
5. U2 相比 U0：`Delta AP50 >= +0.5`、`Delta AP25 >= 0`、
   `Delta AP15 >= -0.3`。
6. 新增融合计算耗时低于端到端时间的2%。Boxer 本体原本已运行，因此这里只
   统计新增标量加权耗时。

`report_scannet_b6_boxer_uncertainty.sh` 会自动计算前五项并明确输出
`promote_to_full100=true/false`；端到端耗时仍需结合驱动日志人工核对。

通过后，完整100场只运行 U0/U2 并比较两份标准评估结果：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_boxer_uncertainty_dev
BOXFUSION_B6_BOXER_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
  bash scripts/run_scannet_b6_boxer_uncertainty.sh u0_control 0,1
BOXFUSION_B6_BOXER_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
  bash scripts/run_scannet_b6_boxer_uncertainty.sh u2_active 0,1
```

完整100场不必重跑 U1；U1 的作用是固定10场验证 observer 恒等契约。当前完整
配对审计工具要求三组产物，因此不要在仅有100场 U0/U2 时调用它。如需做100场
三路完整审计，再补跑100场 U1 后调用同一审计命令。

## 结果解释

该模块降低坏 Boxer 视角对融合框的影响，主要可能改善 AP50，而不会增加 proposal
召回。由于当前 `Top-K=3`，且 confidence 在 G0 中分布较集中，合理预期是小幅、
低风险增益，不应承诺大幅提升。如果 U1 诊断显示 Top-K/权重几乎不变，或 U2
未通过固定10场门槛，应记录为负结果并停止，不在验证集上扫描 `confidence_power`。
