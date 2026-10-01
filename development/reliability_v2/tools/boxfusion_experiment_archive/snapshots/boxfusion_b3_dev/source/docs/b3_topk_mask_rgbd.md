# B3-v2: 保留朝向的 Top-K Mask-RGBD 多视角几何精修

本目录中的 B3-v2 是真正的逐视角几何记忆和局部坐标框精修，不是对一个已经
聚合的点云做 Top-K 计数。每次实例观测都会独立保存：

- 帧号和相机位置；
- 该帧实例 mask 内由真实深度反投影得到的世界坐标点；
- proposal confidence、有效深度比例和重投影 mask IoU；
- 视角质量
  `confidence * valid_depth_ratio * projection_mask_iou`。

## 为什么旧 B3 无效

旧 B3 把 BoxFusion 的 8 角点框转换成世界坐标轴对齐框（AABB），在 AABB 上
做分位数收缩，最后再输出 AABB 角点。该过程丢弃了上游框的 yaw。ScanNet
评测还会应用场景 `axisAlignment`，因此世界 AABB 中看似保守的收缩，在评测
坐标中可能成为明显的尺寸扩张或位置偏移。

固定 10 场中，旧 B3 的 AP15/AP25/AP50 为
`41.1799 / 32.7179 / 13.0992`，低于同设置的 identity 控制组。这说明问题
不是简单地把阈值放宽或增大 blend 就能解决。

## B3-v2 数据流

```text
实例 mask + ScanNet 真实 depth + pose
  -> 每视角独立点集与质量
  -> 有界候选池（最多 12 个视角）
  -> 质量与相机方向多样性联合选择 Top-K（B3-v2 为 K=5）
  -> 恢复原 BoxFusion 框的中心、尺寸和局部正交坐标系
  -> 把 Top-K 点和相机位置变换到原框局部坐标系
  -> 选择边界一致性最佳的视角对
  -> 在有双侧/轮廓证据的局部轴上做保守边界更新
  -> 点支持率与真实有朝向框的 2D 重投影门控
  -> 变换回世界坐标，并恢复原框朝向
```

第一个视角按质量选择，后续视角使用确定性的贪心分数：

```text
(1 - diversity_weight) * quality
  + diversity_weight * angular_diversity
```

同一帧只保留质量更高的观测。候选轨迹被全局框吸收时，逐视角点会先裁剪到
扩展后的匹配全局框，再并入全局 Top-K 池，避免邻近物体和背景点污染记忆。

局部框更新还受到以下硬约束：

- 单个边界最多移动原尺寸的 3%；
- 候选尺寸只能处于原尺寸的 `[0.92, 1.00]`；
- 中心移动不超过原框尺度的 8%；
- 原框和候选框内点支持率都至少为 0.70；
- 候选支持率相对原框最多下降 0.03；
- 2D 重投影 IoU 不得下降；
- 输出轴方向与原框一致，因此不会再次丢失 yaw。

轮廓边界采用确定性的最佳一致视角对。第三个噪声视角不会再把所有候选视角
一起否决，也不会参与最终边界测量。

## 与 B6 隔离

B3 使用双轨内存：

- `points` / `aabb`：原始全视角聚合路径，继续供 B6 特征、关联和补充框使用；
- `geometry_points` / `geometry_aabb`：Top-K 路径，只供 B3 refit 和后续
  BoxRefiner 使用。

因此启用 B3 memory observer 不会改变导出的框、score 或检测数量。B6 profile
会自动把 `top_k_views` 设为 0，原 B6 推理不承担 Top-K 开销。

在 B3-v2+B6 中，B6 仍从原始几何分支提取特征，以保持训练时的特征分布；
B3-v2 只改变最终输出几何。纯 B3-v2 保留 detector score、检测数量和输出顺序。

## 固定 10 场结果

所有结果均使用相同 10 场、`minimum_extent=0.40`、proposal interval 5 和随机
种子 0。

| 实验 | AP15 | AP25 | AP50 | 相对 identity |
| --- | ---: | ---: | ---: | ---: |
| B3-v2 identity | 41.5741 | 34.9987 | 15.3455 | — |
| B3-v2 oriented paired-view refit | 41.5741 | **36.2884** | 15.3455 | **AP25 +1.2897** |

AP25 precision 从 `0.500000` 提升至 `0.506944`，recall 从 `0.483221`
提升至 `0.489933`。最终预测中保留 11 个有效 refit；运行时有 15 个候选通过
refit 门控，其中 4 个随后未进入最终导出结果。检测数量和 score 与 identity
逐元素一致，输出框的最小轴方向点积为 `0.99999988`。

这是 B3 在固定 10 场上的第一个无退化正向结果，但仍属于开发集证据。是否能
泛化以及完整提升幅度，必须由 100 场固定配置评估确认。

## 可复现实验

以下命令默认使用固定 10 场、`minimum_extent=0.40`、随机种子 0。

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b3_dev

# B3-v2 identity：构建完全相同的 K=5 记忆，但不修改输出
bash scripts/run_scannet_b3v2_identity.sh 0,1

# 纯 B3-v2：局部坐标、最佳一致视角对、保留 yaw 的几何 refit
bash scripts/run_scannet_b3v2_visibility_refit.sh 0,1

# B3-v2 + 已训练 B6；应在纯 B3-v2 的 100 场结果确认后再运行
bash scripts/run_scannet_b3v2_b6.sh 0,1
```

纯 B3-v2 的完整 100 场命令：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b3_dev

BOXFUSION_B3V2_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_B3V2_RUN_TAG="b3v2_oriented_pair_refit_extent040_full100" \
bash scripts/run_scannet_b3v2_visibility_refit.sh 0,1
```

## 诊断

每场 `.npz` 保留旧 B6 字段，并新增：

- `geometry_points` / `geometry_point_mask`；
- `selected_view_frame_ids`、候选数和入选数；
- 每个 Top-K 视角的点、quality、confidence、有效深度率和重投影 IoU；
- `refit_original_corners` / `refit_candidate_corners`；
- `refit_local_original_boxes` / `refit_local_candidate_boxes`；
- `refit_local_basis` / `refit_local_frame_valid`；
- `b3_schema=visibility_aware_oriented_v3`。

日志中的 `topk=candidates/selected views` 能确认真实逐视角记忆正在工作。
`refit_frame=box_local` 能确认走的是保留朝向的新路径。
`refits=accepted/attempted` 为 `0/N` 不代表 memory 未运行，而表示保守门控拒绝
了所有候选；此时应先检查 `refit_rejections`，不要直接放宽全部阈值。
