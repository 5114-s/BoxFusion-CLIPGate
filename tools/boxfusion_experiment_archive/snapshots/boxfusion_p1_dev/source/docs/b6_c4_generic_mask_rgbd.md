# B6 + C4 通用 SAM3 Mask-RGBD Oriented 局部几何路线（v2）

## 实验身份

C4 是历史最佳 B6 `40.0434 / 33.5492 / 12.1613` 的只读子实验：

- 主 proposal 流仍为 YOLOE；
- B6 使用 `scannet_b6_iou_mlp.npz`，detector blend 为 `0.40`；
- ScanNet minimum extent 为 `0.40`；
- proposal interval 为 5，TTL 时钟为 `provider_call`，track TTL 为 3；
- C4 从冻结的 SAM3 cache 读取第二条 proposal 流；
- C4 只观察已有全局框，不增加 supplemental detection；
- C4 不修改框、角点、score、数量、顺序、stable ID 或 B6 quality feature。
- v2 直接保存原框与候选框的 `[8,3]` oriented corners，并在
  axis alignment 前保留原始 yaw；`[center, size]` 六维框仅用于诊断，
  不再用于重建评估角点。

因此 observer 在同一次运行内不得改写 B6 的正式输出。候选收益只通过诊断文件和
离线替换模拟评估，held-out 判据冻结前禁止开启 mutation。需要注意，YOLOE/CUDA
本身在两个独立进程间不是逐字节确定的：即使连续运行两次纯 B6，角点和 score
也可能有毫米级/千分位浮点漂移。因此不能把两个独立进程的 pickle 字节相等作为
observer 恒等判据；应同时检查 `c4_applied == 0`、同次运行诊断原角点与导出角点
逐点一致、框数/顺序/score 一致，以及正式 AP 与冻结 B6 对照一致。

## C4 几何流程

每个已匹配的全局实例维护一份与 B6 完全隔离的 Top-5 SAM3 Mask-RGBD 记忆：

1. 按 mask 提取真实 depth 世界坐标点；
2. 选取最多 5 个可靠且有视角差异的观察；
3. 保留至少两个视角共同支持的细体素；
4. 用 26 邻域连通域选择与原框相交的主体，并保守合并邻近组件；
5. 按相机可见方向估计六个边界；
6. 用 point support、重投影、多视角、尺度、中心漂移及邻框重叠门控；
7. 只记录 `attempted/proposed/verified`，固定 `applied=0`。

诊断 schema 固定为 `generic_mask_rgbd_local_geometry_v2`。离线报告会先逐点
核对 `c4_original_corners` 与 B6 导出的角点，并再检查 AABB IoU 为 1；
任一校验失败都直接终止，避免用错误的角点顺序或旧版 6D AABB 悄悄评估。
旧的 v1 目录不会被删除或复用，v2 runner 使用独立默认 tag。

## 推荐执行顺序

先跑单场：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev
BOXFUSION_C4_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val_c4_smoke1.txt" \
BOXFUSION_C4_RUN_TAG=b6_c4_mask_rgbd_oriented_observer_smoke1_v2 \
bash scripts/run_scannet_b6_c4_geometry_observer.sh 0
```

单场通过后跑固定 10 场：

```bash
bash scripts/run_scannet_b6_c4_geometry_observer.sh 0,1
```

生成 10 场离线报告：

```bash
/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  tools/report_c4_geometry_ablation.py \
  --pred-root results/b6_c4_mask_rgbd_oriented_observer_ablation10_v2 \
  --diagnostics-root diagnostics/b6_c4_mask_rgbd_oriented_observer_ablation10_v2 \
  --scene-list evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scan-root /extra/ZhaoX/scannet_data/scans \
  --output logs/b6_c4_mask_rgbd_oriented_observer_ablation10_v2/c4_geometry_report.json
```

只有固定 10 场满足下面条件，才跑 100 场 observer：

- `c4_applied == 0`；
- 输出 AP 与同配置 B6 一致；
- 没有 fail-open 或 cache miss；
- verified 候选不是零；
- verified 替换模拟在 AP25 与 AP50 均非负；
- AP50 跨阈值向上数量高于向下数量。

100 场命令：

```bash
BOXFUSION_C4_FULL100=1 \
BOXFUSION_C4_RUN_TAG=b6_c4_mask_rgbd_oriented_observer_full100_v2 \
bash scripts/run_scannet_b6_c4_geometry_observer.sh 0,1
```

100 场完成后，除全量报告外，必须用固定 10 场作为排除集再生成 held-out 90
报告。只有 held-out 90 仍满足安全判据，才设计单独命名的 active profile。
