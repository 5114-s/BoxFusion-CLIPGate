# B5-v2：保留朝向的局部 BoxRefiner

B5-v2 是 B3-v2 之后的学习式几何分支。它不再用固定分位数规则修改少量框，
而是从 Top-K Mask-RGBD 实例点中学习“如何调整局部六个边界”以及“本次调整
是否可能改善 IoU”。这条路线受到 SGCDet 粗到细局部精修思想启发，但不是
SGCDet 网络的逐层复刻。

## 为什么不能直接启用旧 BoxRefiner

旧实现以世界坐标 AABB 为输入和输出，会丢失 BoxFusion 上游框的 yaw。旧训练
数据还存在三个问题：

- 训练和验证按样本随机划分，同一场景可能同时出现在两边；
- 大量低质量匹配样本被当作 identity 几何目标；
- 负样本也参与几何回归，网络容易学会不稳定的平均残差。

B5-v2 明确修复这些问题：

```text
BoxFusion 原始 OBB + K=5 Mask-RGBD 多视角实例点
  -> 恢复原框的中心、尺寸与局部正交 basis
  -> 点变换到原框局部坐标并按原尺寸归一化
  -> 轻量 PointNet 编码
  -> 局部中心/对数尺寸残差 + 改善概率
  -> 支持率、尺寸、中心偏移和有朝向重投影门控
  -> 恢复原 basis/yaw 并导出 8 角点框
```

第一版只修正局部中心和三轴尺寸，保持原 yaw。这样可以先验证学习式几何是否
真正提升 AP50，再把 yaw residual 作为独立消融加入。

## 固定的安全边界

`b5v2_refiner_only` 和 `b5v2_b6` profile 均锁定：

- Top-K memory：`K=5`，候选池最多 12 个视角；
- `coordinate_frame=box_local` 且 `preserve_orientation=true`；
- 中心残差上限为原尺寸的 15%；
- 单轴尺寸比例由网络结构限制在 `[0.8, 1.25]` 附近；
- 改善概率阈值为 `0.50`；
- 至少 2 个视角和 128 个实例点；
- 原框与候选框的点支持率均至少为 `0.55`；
- 候选支持率最多下降 `0.08`；
- 候选有朝向框的 2D 重投影 IoU 至少为 `0.20`，且不得下降。

手工 B3 refit、supplemental output 和 Soft-NMS 全部关闭。因此纯 B5-v2 只允许
学习式局部几何改变输出，detector score、检测数量和输出顺序保持不变。

## 训练监督

训练只允许使用 ScanNet train 场景。离线数据构建器会读取：

- no-op K=5 memory diagnostics；
- 同一次 no-op 推理导出的原始 8 角点框；
- ScanNet train GT 和每场 `axisAlignment`。

每个原框先在评测坐标中匹配 GT，再把可达到的目标中心和尺寸投影回原 OBB
局部坐标。只有裁剪后的目标确实提高评测 IoU 时，样本才获得几何回归监督；
其余样本只训练拒绝/改善概率头。训练器按 `scene_ids` 做场景级划分，并在
训练批次中平衡正负样本，避免场景泄漏和负样本支配。

首先准备 train 场景帧链接：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b3_dev
bash scripts/prepare_scannet_b6_train_data.sh
```

然后采集不改变输出的 K=5 memory。该命令和后续训练使用同一 train-only
scene list：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b3_dev

BOXFUSION_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt" \
BOXFUSION_SCANNET_FRAMES_ROOT="$PWD/data/scannet_train" \
BOXFUSION_ONLINE_ABLATION_PROFILE="b3v2_memory_observer" \
BOXFUSION_ONLINE_PRED_ROOT="$PWD/results/b5v2_memory_observer_train" \
BOXFUSION_ONLINE_LOG_ROOT="$PWD/logs/b5v2_memory_observer_train" \
BOXFUSION_DIAGNOSTICS_ROOT="$PWD/diagnostics/b5v2_memory_observer_train" \
BOXFUSION_EVAL_ROOT="$PWD/evaluation/b5v2_memory_observer_train" \
BOXFUSION_SCANNET_MIN_EXTENT=0.0 \
bash scripts/run_scannet_online_refinement.sh 0,1
```

训练 B5-v2：

```bash
bash scripts/train_scannet_b5v2_refiner.sh
```

默认产物为：

- `datasets/scannet_b5v2_oriented_refiner_train.npz`；
- `models/scannet_b5v2_oriented_refiner.pt`。

训练脚本会拒绝文件名包含 `val` 的 scene list，也会在 diagnostics、原始预测、
GT 或场景列表缺失时直接停止。

## 严格消融

默认先运行固定 10 场：

```bash
# 纯 B5-v2：真实 detector score，只改几何
bash scripts/run_scannet_b5v2_refiner.sh 0,1

# B5-v2 + B6：B6 使用 original geometry，detector blend 默认 0.40
bash scripts/run_scannet_b5v2_b6.sh 0,1
```

评估顺序应固定为：

1. 同配置 identity；
2. B5-v2 only；
3. B6 only；
4. B5-v2 + B6。

只有 B5-v2 固定 10 场结果为正，才运行纯 B5-v2 的完整 100 场：

```bash
BOXFUSION_B5V2_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_B5V2_RUN_TAG="b5v2_refiner_only_extent040_full100" \
bash scripts/run_scannet_b5v2_refiner.sh 0,1
```

随后再运行 B5-v2+B6：

```bash
BOXFUSION_B5V2_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_B5V2_RUN_TAG="b5v2_b6_blend040_extent040_full100" \
BOXFUSION_B5V2_B6_DETECTOR_BLEND=0.40 \
bash scripts/run_scannet_b5v2_b6.sh 0,1
```

B5-v2 的主要目标是 AP50。实现完成并不等同于已有精度提升；必须用独立
ScanNet train 监督训练权重，再由固定 10 场和完整 100 场逐级验证。
