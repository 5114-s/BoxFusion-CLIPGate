# BoxFusion P 路线：P2 Occupancy Top-K Observer

本目录是独立实验副本，不读取或写入正在运行的
`/data/ZhaoX/BoxFusion`、`boxfusion_p1_dev` 的代码状态。当前实现为：

```text
P0  冻结 B6 quality-only
P1  + 类别无关 residual RGB-D sparse proposal observer
P2  + 类别无关 foreground occupancy + 确定性 Top-K observer
```

P0/P1/P2 的正式 prediction 均走同一条冻结 B6 输出路径。P2 只选择
P1 residual voxels 并产生诊断候选，不能修改框、分数、类别、数量或顺序。
P3（SPGroup3D 式轻量 grouping）、P4（多视角 Mask-RGBD 确认）和
P5（source-aware score/output gate）仍未实现或激活。这样保证相邻消融
只新增一个模块。

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

## P2 数据流

```text
冻结 P1 的同一批 residual voxel 14-D features
          ▼
14→32→32→1 类别无关 occupancy MLP
          ▼
sigmoid occupancy probability
          ▼
按 (-score, voxel_x, voxel_y, voxel_z) 稳定排序的 Top-K
          ▼
冻结 P1 box head 解码 + class-agnostic 3D NMS
          ▼
P2 diagnostics（不进入正式输出）
```

P2 只有一个训练目标：

```text
target = voxel center 是否位于 B6 未覆盖的 train-only GT AABB 内
loss   = weighted binary cross entropy
```

在线 P2 接口不接收 GT。模型加载和推理默认放在 CPU；P1/P2 模型构造均
保存并恢复 PyTorch RNG 状态，以尽量不扰动冻结 B6。但原融合路径存在
已测得的 CUDA 数值非确定性，因此正式身份判断必须结合 P0 repeat 漂移
审计，不能只比较两个 pickle 的字节。

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
的 occupancy no-grad Top-K 作为 P2 设计参考；仓库缺少顶层许可证
文件，因此 P2 为 clean-room 实现，没有复制其源码。SPGroup3D commit
`181283547323d3bd54d0e9f58baf0cd413ccc107` 为 CC BY-NC 4.0，
P3 也必须 clean-room 实现，不能直接搬运源码。

P1 的 14-D 输入中包含固定邻域占用统计；P2 新增的唯一模块是可学习
foreground occupancy prediction、occupancy BCE 和 occupancy-guided
Top-K。它没有 grouping、mask/depth 多视角确认和输出门控。

需要明确：P1/P2 使用 ScanNet train GT 构造残差监督，因此方法从
原始 BoxFusion 的 training-free 路线变为“类别无关监督 proposal +
开放词汇 CLIP 语义”的 supervised hybrid。它仍是开放词汇检测，但不能
再宣称“无目标数据集训练”。如果论文必须保留 training-free 属性，就只能
把 P1/P2 保留为 observer，或换成与 ScanNet 无关的通用预训练 proposal
头。

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

### 3. 审计冻结基线的非确定性

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p2_dev
bash scripts/audit_scannet_p1_nondeterminism.sh \
  reports/p_ablation/p1_nondeterminism_audit.json
```

当前实际 10 场审计确认：P0 与 P1 的标准 AP 完全相同，但部分框的浮点
值不是 bit-exact；一次 P0 repeat 本身也发生了框数和 AP 漂移。因此
observer 的安全合同是 `mutation=false/applied=0`，并同时报告结构、
Hungarian 3D-IoU 数值漂移与 AP 漂移。

### 4. 训练 P2 occupancy 头

该命令默认只读复用 `boxfusion_p1_dev` 已完成的 100 个 train scene
P1 diagnostics/predictions，输出只写本目录：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p2_dev
bash scripts/train_scannet_p2.sh
```

默认输出：

```text
models/scannet_p2_occupancy_topk.pt
reports/p2_training_summary.json
```

checkpoint 会绑定 exact P1/B6 SHA、train/forbidden scene-list SHA、
`forbidden_overlap=[]`、14-D feature schema 和训练配置。训练器采用
scene-disjoint 80/20 切分，不会读取 validation GT。

### 5. 固定 10 场 P0/P1/P2

```bash
bash scripts/run_scannet_p_ablation.sh P0 0,1

BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
  bash scripts/run_scannet_p_ablation.sh P1 0,1

BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
BOXFUSION_P2_OCCUPANCY_CHECKPOINT="$PWD/models/scannet_p2_occupancy_topk.pt" \
  bash scripts/run_scannet_p_ablation.sh P2 0,1
```

默认冻结当前最强 B6 协议：

```text
quality detector blend = 0.40
minimum extent         = 0.40
proposal interval      = 5 keyframes
```

如果要对照另一个 B6 参数，必须给 P0/P1/P2 同时设置同一组
`BOXFUSION_P_B6_*` 环境变量。

每次 P0/P1/P2 启动都会在对应 log 目录写入不可变
`run_manifest.json`，绑定场景列表、配置、B6/P1/P2/YOLOE/Cubify/CLIP
权重、关键数据资产、代码树哈希和运行参数。断点续跑时任一项改变都会
拒绝混跑，必须换新的 `BOXFUSION_P_RUN_TAG`。

### 6. P2 安全与召回审计

```bash
bash scripts/audit_scannet_p2.sh
```

该审计默认同时读取本目录、同一代码树生成的
`results/p_ablation/p1_ablation10_b6frozen_v1`。如果尚未运行第 5 步
的 P1，审计会直接拒绝；不会拿历史 `boxfusion_p1_dev` 的不同代码状态
冒充 identity baseline。默认数值包络为 corner max `0.02 m`、score max
`0.02`、matched-IoU loss max `0.05`，略宽于已观测的固定 10 场 P0-repeat
漂移。更换硬件/软件环境后必须先重跑 P0 repeat 审计再调整包络。

必须同时满足：

1. P1/P2 均为 `observer_only=true`、`uses_ground_truth=false`；
2. `p1/p2_mutation_enabled=false` 且 `p1/p2_applied_count=0`；
3. P1/P2 checkpoint SHA 在所有场景一致且与命令指定文件一致；
4. report 使用稳定分数排序的一对一 strict-IoU 匹配，不重复计算 TP；
5. 报告 B6、P1-only、P2-only 及各 union 的
   Recall@0.15/0.25/0.50、novel precision、候选数和分项耗时；
6. 同 candidate ID 的 P1/P2 框和 frozen objectness 必须一致，否则
   fail closed。

P2 是否值得推进 P3/P4 的建议门槛：

```text
相对 B6∪P1：
ΔRecall@0.25 >= 3 个百分点
且
ΔRecall@0.50 >= 1 个百分点
```

未达到门槛就停止 P2/P3，不靠堆模块掩盖 proposal/selector 无效。

### 7. 100 场（仅在固定 10 场与 train-only held-out 通过后）

```bash
# 先在同一 P2 代码树生成 frozen-P1 formal-output 对照
BOXFUSION_P_FULL100=1 \
BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
  bash scripts/run_scannet_p_ablation.sh P1 0,1

BOXFUSION_P_FULL100=1 \
BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
BOXFUSION_P2_OCCUPANCY_CHECKPOINT="$PWD/models/scannet_p2_occupancy_topk.pt" \
  bash scripts/run_scannet_p_ablation.sh P2 0,1

BOXFUSION_P_FULL100=1 \
BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
BOXFUSION_P2_OCCUPANCY_CHECKPOINT="$PWD/models/scannet_p2_occupancy_topk.pt" \
  bash scripts/audit_scannet_p2.sh
```

P2 observer 的标准 AP 理论上应与冻结 B6 一致；若出现小漂移，必须先
与 P0-repeat 数值包络比较。此阶段判断的是 proposal 召回上限和
occupancy selector 的增量价值，不是直接声称 AP 增益。只有未来 P4/P5
完成多视角确认和安全输出门控后，残差候选才可作为正式检测参加 AP。
