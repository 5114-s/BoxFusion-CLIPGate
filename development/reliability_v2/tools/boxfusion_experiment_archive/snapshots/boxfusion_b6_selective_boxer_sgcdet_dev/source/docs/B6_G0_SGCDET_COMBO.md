# B6 + Selective Boxer G0 + SGCDet 局部稀疏 Refiner

本目录是隔离实验快照，不修改以下已完成路线：

- `/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev`
- `/data/ZhaoX/OVM3D-Dett/boxfusion_b6_sgcdet_local_refiner_dev`

## 数据流与唯一变量

```text
冻结 CuTR proposal replay
→ Selective Boxer G0（0.10 m，[0.50, 2.00]）
→ 原 BoxFusion 关联/融合
→ 冻结 B6 IoU MLP（detector blend=0.40）
→ SGCDet K=5、P=128 局部稀疏 Refiner
→ score=0.4 / minimum extent=0.4 导出
```

Boxer 只在逐 proposal lifting 阶段替换通过门控的相机坐标3D框，拒绝行回退
CuTR；SGCDet 只在最终输出前修改通过门控的全局框几何。类别、分数、数量、
顺序和 stable ID 都不得由 SGCDet 改写。

第一版直接迁移已训练的 SGCDet checkpoint，不在 Boxer 分布上重训，以保证相对
G0 唯一新增因素是局部几何 Refiner。

## 固定10场协议

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_sgcdet_dev

bash scripts/run_scannet_b6_g0_sgcdet_combo.sh g0 0,1
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh observer 0,1
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh identity 0,1
bash scripts/audit_scannet_b6_g0_sgcdet_combo.sh
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh active 0,1
bash scripts/evaluate_scannet_b6_g0_sgcdet_same_run.sh 0
```

也可以一次执行完整固定10场协议：

```bash
bash scripts/run_scannet_b6_g0_sgcdet_combo_paired.sh 0,1
```

必须先通过 G0、observer 和 identity 控制审计，再解释 active 结果。独立GPU运行
之间的框数量、浮点值和 AP 漂移仅作报告；因果增益以 active 同一次运行记录的
`output_pre_geometry_corners` 与 `output_post_geometry_corners` 为准。

固定10场建议放行条件：配对 `ΔAP50 >= +0.5` 点、`ΔAP25 >= 0`、
`ΔAP15 >= -0.3` 点，并且至少3个场景存在最终几何修改。

## 100场协议

固定10场通过后，冻结所有参数并运行：

```bash
BOXFUSION_COMBO_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_COMBO_RUN_TAG=g0_sgcdet_active_full100_v1 \
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh active 0,1

BOXFUSION_COMBO_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_COMBO_ACTIVE_TAG=g0_sgcdet_active_full100_v1 \
BOXFUSION_COMBO_COUNTERFACTUAL_TAG=g0_sgcdet_active_full100_identity_v1 \
bash scripts/evaluate_scannet_b6_g0_sgcdet_same_run.sh 0
```

100场最终报告需要同时列出：

1. 组合 active 的标准 AP；
2. 同运行 pre-geometry identity AP；
3. 三个阈值的严格配对差值；
4. 修改框数/场数、中心偏移、体积比和 pre/post IoU；
5. 与历史 G0 `40.2787 / 35.4508 / 15.2181` 的非配对参考比较。

历史 G0 只作为参考锚点。由于原 Selective Boxer 工作树在该 full100 实验后又发生
过修改，新组合目录必须先重新跑自己的 G0 控制，不能把历史结果当作字节级基线。

## 候选几何上限审计

当 active 未通过固定10场放行条件时，不允许直接放宽门控。先执行：

```bash
bash scripts/evaluate_scannet_b6_g0_sgcdet_candidate_oracle.sh 0
```

该工具从冻结 checkpoint 读取残差边界，用运行时相同的 box-local 解码公式重建
全部 raw candidate。它不会误用拒绝后已回退成原框的 `sparse_active_corners`，并
保留原标签、分数、数量与行顺序。随后由仓库原始 `eval_scannet.py` 复核 identity、
active、all-candidate、逐框 GT oracle 与逐阈值 forward oracle。

当前 frozen checkpoint 的 fixed10 结果为：

| 几何选择 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| same-run identity | 44.6302 | 40.8154 | 17.1297 |
| runtime active | 45.1152 | 40.8154 | 17.1297 |
| 全部77个有效候选 | 45.1152 | 40.8154 | 17.1297 |
| 逐框 best-IoU GT oracle | 45.1152 | 40.8154 | 17.1297 |
| AP25/AP50 forward oracle | 44.6302 | 40.8154 | 17.1297 |

77个有效候选中，37个提高逐框 best-GT IoU、34个降低、6个不变；但只有1个候选
跨过 IoU 0.15，跨过0.25和0.50的候选均为0。当前门控接受17个，其中7个提高、
9个降低、1个不变。因此结论不是“门控太保守”，而是当前迁移 checkpoint 的候选
在 G0 分布上没有 AP25/AP50 跨阈值能力。禁止继续调 `improvement_threshold`、
uncertainty 或 support 门控，也禁止据此跑100场 active。

完整报告位于
`reports/b6_g0_sgcdet_candidate_oracle/g0_sgcdet_candidate_oracle_fixed10_v1/`。

## G0 同分布重训（v1）

候选 oracle 证明旧 checkpoint 在 fixed10 只有一个 IoU@0.15 上跨，
IoU@0.25/0.50 上跨均为零。进一步审计发现旧训练采集使用
`b5v2_memory_observer`：它既没有 Selective Boxer G0 lifting，也关闭了 B6
quality，因此不能继续通过调 active gate 修复。

新协议只修复这一项分布错配，不改变验证集推理结构：

```text
train-only fresh CuTR
→ Selective Boxer G0 (0.10 m, [0.50, 2.00])
→ frozen B6 score (blend=0.40)
→ sgcdet_sparse_observer K=5/P=128 diagnostics
→ AP50 target-potential preflight
→ AP50-primary sparse-head training
→ frozen replay fixed10 paired validation
```

### 1. GPU 收集 train-only diagnostics

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_sgcdet_dev
bash scripts/collect_scannet_b6_g0_sgcdet_train.sh 0,1
```

采集脚本显式禁用只覆盖 validation scenes 的 CuTR replay cache，并跳过
对 train scenes 的 ScanNet evaluation；prediction、online diagnostics、Boxer
JSONL 和 YOLOE cache 均写入独立 train namespace。结束时会生成并复核
不可变 collection manifest。

### 2. CPU 构建、预检并训练

```bash
bash scripts/train_scannet_b6_g0_sgcdet_refiner.sh
```

硬停止条件包括：train/held-out-train 的 eligible、geometry 正负样本、
AP50 crossing 数量/场景覆盖、preserve 数量和 oracle net rate。checkpoint
选择顺序为 `AP50 proxy → drop50 rate → validation loss`；若没有任何 epoch
在 held-out train scenes 实现正 AP50 crossing，脚本拒绝写 active checkpoint。

### 3. fixed10 严格配对验证

```bash
bash scripts/run_scannet_b6_g0_sgcdet_retrained_paired.sh 0,1
```

该命令依次运行 G0、observer、identity 控制审计、active 和 same-run identity
反事实评估。只有 fixed10 的 AP25/AP50 净增益为正，且无 count/order/B6-score
旁路变化时，才允许进入 full100；否则本路线应停止，而不是继续调 gate。
GT oracle 仅用于离线诊断，不能作为可部署模型结果。
