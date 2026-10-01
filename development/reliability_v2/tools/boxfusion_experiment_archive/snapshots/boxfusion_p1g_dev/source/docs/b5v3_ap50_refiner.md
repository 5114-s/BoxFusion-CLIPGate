# B5-v3：严格 K=5 与 AP50-aware 局部 BoxRefiner

## 目的

B5-v2 原型在固定 10 场上使 AP25 提高约 1.29，但 AP50 没有变化。本路线
同时修复两个问题：

1. 原型权重实际使用旧的 `quality_observer` 数据训练；旧文件没有 K=5
   逐视角记忆，训练输入与推理输入不一致。
2. 原型只学习“候选框是否带来任意 IoU 改善”，没有重点学习接近或跨越
   IoU=0.50 的框。

B5-v3 仍然保持原 OBB basis/yaw，只预测 box-local 中心和尺寸残差。它不改变
detector score、检测数量和顺序，也不启用 B6、supplemental proposals 或
Soft-NMS。

本目录完全隔离：

```text
/data/ZhaoX/OVM3D-Dett/boxfusion_b5_ap50_dev
```

所有新数据、权重和实验 tag 都使用 `b5v2_k5` 或 `b5v3_ap50` 命名，不覆盖
旧的 `scannet_b5v2_oriented_refiner.pt`。不要在仍有实验使用同一 GPU 时启动
下面的采集或评估命令。

## 第一步：采集真正的 K=5 train diagnostics

默认使用 100 个 ScanNet train-only 场景和 `minimum_extent=0.40`：

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b5_ap50_dev
bash scripts/collect_scannet_b5v3_k5_train.sh 0,1
```

采集配置固定为：

- profile：`b5v2_memory_observer`，输出框、score、数量和顺序不变；
- `top_k_views=5`，候选视角池至少为 12；
- YOLOE 每 5 个 BoxFusion keyframe 调用一次；
- TTL 使用 `provider_call`，track TTL 为 3，archive 关闭；
- inference/evaluation seed 均为 0；
- runtime/output minimum extent 为 0.40；
- 模型输入固定为 box-local `512×3` 点和对应 mask；
- 运行时支撑门控保存完整的 box-local `8192×3` 点；
- 真正被 memory 选中的 5 个 frame ID 与重投影 evidence 分开保存；
- train scene list 与正式 val100 必须零交集。

默认产物：

```text
results/b5v3_k5_gatealigned_train_extent040_v2/
diagnostics/b5v3_k5_gatealigned_train_extent040_v2/
logs/b5v3_k5_gatealigned_train_extent040_v2/
```

双 RTX 3090 的预计耗时约 1.5–2.2 小时。`minimum_extent=0.40` 同时用于
finalize 输出过滤和 `demo.py` post-process，使 diagnostics 的
`result_indices` 与同次导出的 pkl 保持一致，并让训练分布匹配固定 10 场推理
分布。默认拒绝复用任何已有场景对；若确认是同一次、同代码和同配置的中断
任务，才可显式设置 `BOXFUSION_B5V3_ALLOW_RESUME=1`。任意 prediction 与
diagnostic 半对状态都会被拒绝，避免不同代码状态或采集配置被混合。

## 第二步：训练配对控制 B5-v2-K5

先用相同严格 K5 数据训练原 B5-v2 objective，作为判断提升究竟来自 K5 数据
还是 AP50 objective 的配对控制：

```bash
bash scripts/train_scannet_b5v2_k5_refiner.sh
```

默认输出：

```text
datasets/scannet_b5v2_k5_gatealigned_extent040_train_v2.npz
models/scannet_b5v2_k5_gatealigned_extent040_refiner_v2.pt
```

这是 CPU-only 步骤。数据构建器必须逐场验证：

- runtime schema、profile、K、TTL、archive、输出 mutation 和所有 refit
  gate 参数都与 `b5v2_refiner_only` 一致；
- 模型实际输入、完整支撑点、OBB frame 和 padding sentinel 合法；
- `selected_view_counts == top_k_view_valid.sum()`，选中 frame ID 唯一；
- 重投影 evidence 只使用真正选中的 Top-K frame，而不是把全部可见记录冒充
  Top-K；
- train scene 不出现在禁止的 val list 中；
- runtime minimum extent provenance 为 0.40；
- scene list 的数量和 SHA-256 与数据集内实际 `scene_ids` 一致。

如果传入旧 `b6_quality_observer_train`，严格构建应直接失败，而不是退回普通
`points/point_mask`。B5-v2-K5 控制显式传入 `--strict-k5-diagnostics`；
AP50 模式本身也强制相同检查。

## 第三步：训练 B5-v3 AP50-aware

```bash
bash scripts/train_scannet_b5v3_ap50_refiner.sh
```

默认输出使用独立名称：

```text
datasets/scannet_b5v3_k5_ap50_gatealigned_extent040_train_v2.npz
models/scannet_b5v3_k5_ap50_gatealigned_refiner_v2.pt
```

AP50-aware objective 不再只判断“任意 IoU 改善”。构建器先沿可达残差的
`α={0.25,0.50,0.75,1.00}` 做线搜索，并对每个候选重放与推理一致的完整门控：
视角数、点数、输出 extent、生存状态、中心/尺寸变化、点支撑和多视角重投影。
只把真正能在推理时通过门控的候选作为目标。

随后使用真实 detector score，在每个场景内按 ScanNet/VOC 的排序和一对一
占用规则分别计算 identity TP50 与 candidate-oracle TP50。训练重点包括：

- 可达目标带来的 IoU 增益更大；
- candidate 获得 identity 没有的场景级 TP50；
- 原框或可达目标位于 IoU=0.50 附近。

默认近阈值带宽为 0.15、IoU 增益截断为 0.25；数据采样的 gain/cross/near
权重分别为 2.0/4.0/2.0，训练损失的 IoU-gain/crossing 权重为 2.0/4.0。
正负样本仍做固定 50/50 平衡采样，AP50 权重只在 loss 中使用一次，避免重复
加权。验证模型按同一分母报告
`(新增 TP50 - 丢失 TP50) / eligible_matched`。由于该轻量 proxy 不保存
8192 点和相机证据、不能对模型预测框重放完整 gate，checkpoint 以 scene-held-
out AP50-aware validation loss 最小值选择，proxy 仅作诊断；最终结论只看固定
10 场的真实配对评估。crossing loss 使用 0.5001 margin，事件统计仍遵循官方
evaluator 的严格 `IoU > 0.50`。
这些参数可以通过 `BOXFUSION_B5V3_NEAR_IOU50_BAND`、
`BOXFUSION_B5V3_GAIN_CAP`、`BOXFUSION_B5V3_GAIN_SAMPLE_WEIGHT`、
`BOXFUSION_B5V3_CROSS_SAMPLE_WEIGHT`、`BOXFUSION_B5V3_NEAR_SAMPLE_WEIGHT`、
`BOXFUSION_B5V3_IOU_GAIN_WEIGHT` 和
`BOXFUSION_B5V3_CROSS_IOU50_WEIGHT` 覆盖。训练/验证仍按完整 scene 划分，
不能对单个框随机切分。

## 第四步：固定 10 场

先运行同配置 K5 identity：

```bash
bash scripts/run_scannet_b5v3_gatealigned_identity.sh 0,1
```

再运行使用新 K5 train diagnostics 重训的 B5-v2 配对控制：

```bash
bash scripts/run_scannet_b5v2_k5_refiner.sh 0,1
```

最后运行纯 B5-v3：

```bash
bash scripts/run_scannet_b5v3_ap50_refiner.sh 0,1
```

默认 tag：

```text
b5v3_ap50_gatealigned_refiner_only_extent040_ablation10_v2
```

必须与下列配对实验使用完全相同的固定 10 场、extent 和种子：

1. K5 identity；
2. B5-v2 原型；
3. B5-v2-K5；
4. B5-v3-AP50。

重点报告 AP15/AP25/AP50、accepted/attempted、quality/gate rejection，以及
保留方向轴的最小 cosine。B5-v3 进入 val100 的最低条件是：

- AP50 高于 K5 identity 和 B5-v2-K5；
- AP25 没有明显退化；
- score、检测数量与输出顺序保持不变；
- OBB/yaw 方向轴保持不变；
- 增益不能仅来自固定 10 场中的一个异常场景。

若 AP50 仍未提高，应先查看接近 0.50 的可达样本数量、crossing 样本的
validation recall，以及候选被 quality gate/reprojection gate 拒绝的比例，
而不是直接运行 100 场或放宽全部安全门。

## 按条件自动执行

当前 GPU 实验结束后，可以执行完整决策协议：

```bash
bash scripts/run_scannet_b5_k5_then_ap50_protocol.sh 0,1
```

该脚本启动前会检查 CUDA compute process；只要仍有任何 GPU 任务就直接拒绝
运行。它先采集、训练并评估 K5 improvement 对照，再生成 identity-score
locked 的配对报告。只有固定 10 场 AP50 没有严格高于 identity 时，才训练和
评估 AP50-aware fallback。无论哪条路线，都不会自动进入 val100。

若需要在另一个 full100 正在运行时安全排队，可用：

```bash
tmux new-session -d -s b5_k5_ap50_protocol \
  -c /data/ZhaoX/OVM3D-Dett/boxfusion_b5_ap50_dev \
  "bash scripts/wait_then_run_scannet_b5_k5_ap50_protocol.sh 0,1"
```

watcher 只轮询上游成功标志；上游失败时退出，上游成功后仍会等待 CUDA process
列表为空，才复制稳定缓存并交给上述协议。
