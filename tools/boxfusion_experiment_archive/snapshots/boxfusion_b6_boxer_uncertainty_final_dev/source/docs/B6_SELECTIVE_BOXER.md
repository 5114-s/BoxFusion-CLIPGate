# B6 + Selective Boxer 实验协议

本路线在完全隔离的开发目录
`/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev` 中验证。它不会修改或覆盖
`/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev`、
`/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev` 或正在运行的实验。

## 目标与边界

Selective Boxer 不是把全部 CuTR lifting 无条件替换成 Boxer。它仅在 Boxer 几何相对
CuTR 足够保守时逐框采用 Boxer，否则该框回退到原始 CuTR 几何。实验的目标是验证这种
选择性替换能否保留 B6 的 AP15/排序优势，同时吸收 Boxer 在 AP25/AP50 上的潜在收益。

本路线目前只是待验证的消融方案，**不能预先承诺超过 B6**。是否有效只能由同一场景列表、
同一冻结配置下的配对实验决定。

## 冻结 B6 合同

三个实验阶段必须共同固定以下设置：

- proposal `score_thresh = 0.40`；
- ScanNet 最小边长过滤 `minimum_extent = 0.40 m`；
- 可靠视角融合 `Top-K = 3`；
- B6 质量模型 `iou_mlp`；
- detector/quality score blend `0.40`；
- online ablation profile 为 `quality_only`；
- 推理种子和评估种子均为 `0`；
- 使用同一冻结 CuTR proposal replay cache、同一场景顺序及同一模型权重。

`quality_only` 必须继续关闭 supplemental proposal 输出、几何 refit、Soft-NMS 等其他会改变
结果的路径。这样，`s0_control`、`s0_observer` 与 `s1_selective` 之间唯一允许改变的是
Selective Boxer 是否观察或应用。

## Selective 门控

门控在单个 proposal 上比较 CuTR 和 Boxer 的相机坐标系 3D 框：

```text
center_shift = ||center_boxer - center_cutr||2
volume_ratio = volume_boxer / volume_cutr

accept Boxer iff:
    center_shift <= 0.10 m
    and 0.50 <= volume_ratio <= 2.00
```

边界值按闭区间接受。任一几何值非有限、框无效、中心漂移过大或体积比越界时，只对该框
使用 CuTR 回退；不得让一条坏 proposal 导致整帧失败。`observer` 模式只计算并记录门控，
实际输出必须仍为 CuTR；`active` 模式只替换通过门控的行，其余行保持 CuTR。

## 固定 10 场配对实验

默认场景表是 `scannetv2_val_ablation10_even.txt`。在隔离目录执行：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev

# S0：冻结 B6 + CuTR replay 控制组
bash scripts/run_scannet_b6_selective_boxer.sh s0_control 0,1

# S0-observer：执行 Boxer 和门控，但禁止修改任何预测
bash scripts/run_scannet_b6_selective_boxer.sh s0_observer 0,1

# S1：只对通过门控的 proposal 激活 Boxer，其他逐框回退 CuTR
bash scripts/run_scannet_b6_selective_boxer.sh s1_selective 0,1
```

三个命令必须使用完全相同的场景表。输出、日志及 diagnostics 会按 profile 和场景表 SHA
分别写入 `results/b6_selective_boxer/`、`logs/b6_selective_boxer/` 和
`diagnostics/b6_selective_boxer/`，不会复用不同 profile 的预测文件。

也可以用一个命令顺序完成三组实验并自动执行强制审计：

```bash
bash scripts/run_scannet_b6_selective_boxer_paired.sh 0,1
```

## 进入 100 场前的强制审计

固定 10 场完成后，必须先完成配对审计，至少确认：

1. 三个 profile 的场景 ID、处理顺序、proposal 数和 replay cache manifest 完全对应；
2. `s0_observer` 的 `applied = 0`，其预测框、分数、类别和顺序与 `s0_control` 一致；
3. observer 与 active 的门控判定、接受掩码和拒绝原因一致；
4. `s1_selective` 中每个接受行等于 Boxer 几何，每个拒绝行等于 CuTR 几何；
5. `eligible + fallback = proposal_count`，没有缺失、重复或非有限框；
6. score、类别、proposal 顺序、B6 Top-K 与 B6 质量模型没有被 Boxer 路径改写；
7. 所有场景的运行指纹、冻结 checkpoint SHA 和 diagnostics 均齐全；
8. 标准评估使用相同 evaluator，并同时报告 AP15/AP25/AP50、运行时间、显存和门控接受率。

只要 identity/safety 审计失败，就必须先修复并重新跑固定 10 场，不能用其 AP 判断模块效果，
也不能进入 100 场。即使审计通过，固定 10 场结果也只用于发现明显退化，不能据此声称最终提升。

若三组已分别完成，可单独运行审计：

```bash
bash scripts/audit_scannet_b6_selective_boxer.sh
```

## 100 场运行

仅当上述审计全部通过后，显式指定完整验证集运行 active profile：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev

BOXFUSION_B6_BOXER_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
  bash scripts/run_scannet_b6_selective_boxer.sh s1_selective 0,1
```

最终应与冻结 B6 `40.0434 / 33.5492 / 12.1613` 做同协议比较，并同时给出相对变化。
若没有超过 B6，应如实将其记录为负结果或消融结果，而不是继续依据验证集结果反复调门控阈值。
