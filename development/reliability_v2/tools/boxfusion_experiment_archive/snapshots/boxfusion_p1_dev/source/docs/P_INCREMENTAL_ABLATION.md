# BoxFusion P 路线：从冻结 B6 到残差 Proposal

本目录是独立实验副本，不读取或写入正在运行的
`/data/ZhaoX/BoxFusion` 代码状态。当前只实现并放行：

```text
P0  冻结 B6 quality-only
P1  + 类别无关 residual RGB-D sparse proposal observer
```

P2（SGCDet 式 occupancy Top-K）、P3（SPGroup3D 式轻量 grouping）、
P4（多视角 Mask-RGBD 确认）和 P5（source-aware score/output gate）仍未
激活。这样可以保证相邻消融只新增一个模块。

## P1 数据流

```text
当前关键帧 RGB + 真实 depth + pose
          │
          ├── 当前 BoxFusion/B6 OBB：仅用于标记 explained points
          │
          ▼
未解释的 residual depth points
          ▼
确定性稀疏体素 + 14-D 类别无关特征
          ▼
每体素 1 objectness + 6-D box residual
          ▼
稳定 Top-K + class-agnostic 3D NMS
          ▼
P1 diagnostics（不进入正式输出）
```

6-D 回归编码为：

```text
delta_center_m = gt_center - voxel_center
log_size_m     = log(gt_size_in_metres)
```

在线接口不接受 GT、类别标签或 CLIP 标签。最终类别仍由 BoxFusion 的
CLIP 开放词汇分支负责。

## 参考实现边界

P1 clean-room 借鉴 OpenMMLab TR3D 的稀疏逐点预测、中心/尺寸残差、
Top-K 和 NMS 思路。审计时参考的 MMDetection3D commit 为
`fe25f7a51d36e3702f961e198894580d83c4387b`，其许可证为 Apache-2.0。
本实现未复制 TR3D 源码，也不依赖 MinkowskiEngine/MMDetection3D。

SGCDet commit `eb4ba52a711ab30302569ce7329aca9be28aa39d`
的 occupancy no-grad Top-K 仅作为 P2 设计参考；仓库缺少顶层许可证
文件，因此没有复制代码。SPGroup3D commit
`181283547323d3bd54d0e9f58baf0cd413ccc107` 为 CC BY-NC 4.0，
P3 也必须 clean-room 实现，不能直接搬运源码。

P1 的 14-D 输入中虽包含固定的邻域占用统计，用来替代当前环境中不可用的
MinkowskiEngine 稀疏卷积局部上下文，但它没有 occupancy prediction、
occupancy loss 或 occupancy-guided Top-K；这三项仍严格属于 P2。

需要明确：当前 P1 头使用 ScanNet train GT 构造残差监督，因此方法从
原始 BoxFusion 的 training-free 路线变为“类别无关监督 proposal +
开放词汇 CLIP 语义”的 supervised hybrid。它仍是开放词汇检测，但不能
再宣称“无目标数据集训练”。如果论文必须保留 training-free 属性，就只能
把 P1 保留为 observer，或换成与 ScanNet 无关的通用预训练 proposal 头。

## 运行顺序

### 1. 收集 train-only 稀疏输入

先用两个 train scene 做端到端 smoke（不会读 validation GT）：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev
BOXFUSION_P1_TRAIN_SCENES="$PWD/evaluation/data_util/meta_data/scannetv2_train_p1_smoke2.txt" \
BOXFUSION_P1_TRAIN_RUN_TAG=p1_residual_inputs_train_smoke2_v1 \
  bash scripts/collect_scannet_p1_train.sh 0,1
```

smoke 通过后再收集固定 100 个 train scenes：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p1_dev
bash scripts/collect_scannet_p1_train.sh 0,1
```

收集脚本拒绝 train/val scene 交集，并且强制跳过评测；GT 只在下一步
CPU 离线构造训练 target 时读取。

### 2. 训练 P1 类别无关头

```bash
bash scripts/train_scannet_p1.sh
```

默认输出：

```text
models/scannet_p1_residual.pt
reports/p1_training_summary.json
```

checkpoint 包含 feature schema、网络结构、train/forbidden scene-list
哈希和训练配置；运行时会同时检查：

- 固定 14-D 特征顺序；
- train scene ID 合法且唯一；
- 完整 ScanNet validation 列表已作为 forbidden split；
- P1 checkpoint 内的 B6 SHA 与本次冻结 B6 checkpoint 完全一致。

任一项不一致都会在推理前终止。

### 3. 固定 10 场 P0/P1

```bash
bash scripts/run_scannet_p_ablation.sh P0 0,1

BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
  bash scripts/run_scannet_p_ablation.sh P1 0,1
```

默认冻结当前最强 B6 协议：

```text
quality detector blend = 0.40
minimum extent         = 0.40
proposal interval      = 5 keyframes
```

如果你要对照另一个 B6 参数，必须给 P0/P1 同时设置同一组
`BOXFUSION_P_B6_*` 环境变量。

每次 P0/P1 启动都会在对应 log 目录写入不可变
`run_manifest.json`，绑定场景列表、配置、B6/P1/YOLOE/Cubify/CLIP
权重、关键数据资产、代码树哈希和运行参数。断点续跑时任一项改变都会
拒绝混跑，必须换新的 `BOXFUSION_P_RUN_TAG`。

### 4. 恒等和召回审计

```bash
bash scripts/audit_scannet_p1.sh
```

必须同时满足：

1. P0 与 P1 prediction pickle 逐结构、dtype、shape、数值字节一致；
2. `p1_mutation_enabled=false`；
3. `p1_applied_count=0`；
4. report 使用按 objectness 排序的一对一匹配，不把重复框重复算 TP；
5. 报告 B6/P1/union Recall@0.15/0.25/0.50、novel precision、与 B6
   重复率、候选数及额外耗时。

建议 P1 放行到 P2 的最低门槛：

```text
train-only held-out ΔRecall@0.25 >= 3 个百分点
或
train-only held-out ΔRecall@0.50 >= 1 个百分点
```

未达到门槛就停止 P 路线，不能靠继续叠 P2/P3 掩盖 proposal 本身无效。

### 5. 100 场（仅在固定 10 场与 held-out 通过后）

```bash
BOXFUSION_P_FULL100=1 \
BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
  bash scripts/run_scannet_p_ablation.sh P1 0,1
```

P1 observer 的标准 AP 必须与 P0 完全一致；此阶段判断的是 proposal
召回上限，不是直接声称 AP 增益。只有 P4/P5 完成多视角确认和安全输出
门控后，残差候选才可以作为正式检测参加 AP 评测。
