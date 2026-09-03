# CuTR + Boxer：AP 与端到端实时性核验

核验日期：2026-08-24（Asia/Shanghai）

## 最终判定

| 要求 | 判定 | 证据 |
|---|---:|---|
| official100 constant-score AP 有提升 | 通过 | AP15/AP25/AP50 分别 `+0.9353/+1.7203/+3.3854` 点 |
| 因果、在线、无 proposal replay 的联合执行 | 通过 | live 配置没有 `proposal_cache`；CuTR 与 Boxer 在同一进程执行 |
| 单卡 batch=1、25 Hz 流实时 | 当前场景通过 | `25.88 FPS > 25 FPS` |
| 单卡 batch=1、严格 30 Hz 不掉帧 | 不通过 | `25.88 FPS < 30 FPS` |
| 与 CuTR-only 原版吞吐相同 | 不通过 | `33.37 -> 25.88 FPS`，下降 `22.45%` |

因此可以证明 **Boxer 几何替换提高 AP 且保持因果在线**，但不能证明当前同步实现满足严格 30 Hz 或与原版实时吞吐相同。

## 1. AP：official100、constant score

协议与旧 `x0/x2` Boxer 几何消融完全一致：

- `score_thresh=0.5` 只在在线推理中筛选真实 CuTR score；
- 评测器在读取预测后把每个框的 confidence 固定为 `1.0`；
- 100 个 official ScanNet 场景；
- appearance gate 关闭；
- Reliable-View Top-K 关闭；
- active 只用 Boxer 替换相同 CuTR proposal 行的 `pred_boxes_3d`。

| full100 | 框数 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| CuTR replay control | 1,794 | 29.7915 | 25.3389 | 7.8230 |
| CuTR proposals + Boxer geometry | 1,777 | 30.7268 | 27.0592 | 11.2084 |
| active - control | -17 | **+0.9353** | **+1.7203** | **+3.3854** |

评测器 SHA256：

`aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27`

评测代码中的固定分数位置：`eval_scannet.py:199`。

AP 日志 SHA256：

- control：`8b60771f73eeb4439885fd61a07324ca55659f613dd84d819342888ce5af7d59`
- active：`7d766aec8c1cbaa190a5593b73c433a89a5b57769d88c7e1513cb0ad9a62be4f`

## 2. live 端到端配对测试

### 协议

- 场景：official100/fixed10 中的 `scene0277_00`；
- GPU：单张 NVIDIA GeForce RTX 3090 24 GB；
- batch size：1；
- `gap=25`；
- 同一 GPU、同一场景、顺序运行，避免双进程 CPU/I/O 争用；
- steady-state `Cost` 从模型加载完成后开始；
- 包含流读取、预处理、真实 CuTR forward、真实 Boxer forward、关联和融合；
- 不含一次性模型冷启动及最终 pickle 写盘；外层 wall clock 另外报告；
- 两臂除输出目录和 `lifting.backend` 外配置相同；
- 两个配置都没有 `proposal_cache`，因此不是 replay 计时。

### 结果

| 指标 | CuTR-only control | CuTR + Boxer active | 变化 |
|---|---:|---:|---:|
| steady-state Cost | 33.71 s | 43.47 s | +9.76 s / +28.95% |
| stream FPS | 33.37 | 25.88 | -7.49 / -22.45% |
| 平均每流帧 | 29.97 ms | 38.64 ms | +8.67 ms |
| 外层 wall clock（含冷启动） | 44.69 s | 54.12 s | +9.43 s |
| host peak RSS | 8,958,752 KiB | 8,957,112 KiB | 近似不变 |

active 内部统计：

- 45 个 Boxer keyframe calls；
- 57 个 CuTR proposals，57 个 geometry replacements；
- Boxer 已同步 forward p50/p95：`32.355/33.106 ms`；
- 根据成对总 Cost 计算的完整增量为 `216.89 ms/keyframe`。

后一个数包含 Boxer forward 计时窗口之外的联合路径开销，实时判定必须使用完整 `Cost/FPS`，不能只引用约 33 ms 的 Boxer forward。

### 30 Hz 判定

- 30 Hz 帧预算：`33.33 ms/frame`；active 为 `38.64 ms/frame`，未通过。
- 25 Hz 帧预算：`40.00 ms/frame`；active 为 `38.64 ms/frame`，在该场景通过，但余量只有约 `1.36 ms/frame`。

这是一条正式场景上的反例，足以否定“当前同步实现保证 30 Hz”的表述；它不是 fixed10/full100 的延迟分布，不能代替后续 p50/p95 场景统计。

## 3. replay AP 与 live 路径的连接检查

把 exact live-active 的 `scene0277_00` 输出与旧 full100 replay-active 对应输出逐行比较：

- 行数：`8 == 8`；
- class/order：完全相同；
- 3D corners 最大绝对差：`0.0 m`；
- 真实 score 最大绝对差：`1.9681453704833984e-4`；
- constant-score 评测会把这点 score 漂移统一替换为 `1.0`。

因此该场景确认 proposal replay 没有制造 Boxer 几何增益；live 与 replay 得到了逐元素相同的最终几何。full100 AP 仍应表述为“冻结 CuTR proposal 的几何消融”，不能冒充 full100 live-runtime 测试。

## 4. 可复核文件与哈希

运行配置：

- control SHA256：`7111bde3d1f2048c4cc5b4e7239bab8e7dc4cf3760bd7aa94654df8e7f5503e0`
- active SHA256：`97178da4504a387f9c6b237584fc1244173b1c818533df5e918eb6da2832d26f`

live 日志：

- control SHA256：`10edfd094b51b4840f65168490b3d42f94da8b5cdfd0908a5570e670144c9017`
- active SHA256：`59c55c5103f1e45fa20bdadd6fec98a0ceb4497e424563218036fa6d35722bb7`

冻结运行代码：

- `demo.py`：`e397856d70419e57856d59c73e6461765e38f8bc5625201cbb397d14d328f69b`
- `boxer_lifter.py`：`00d29b6e05adecc07f6ec956b9375e8d44174908b4e20591a7b9b39f5b85246c`
- `box_fusion.py`：`76a1be9d2202527e50fc8e0d2c598367309812a45ff6cd0ca6405bfe19bcea23`

官方 Boxer 资产：

- Boxer commit：`1f86542dc342a4b1d474c87c97c5d1d6566d9148`
- BoxerNet checkpoint SHA256：`d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f`
- DINOv3 checkpoint SHA256：`4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea`

## 5. 不应混淆的结论

1. 上述 AP 是 **CuTR + Boxer、Top-K 关闭** 的旧纯几何协议。
2. 不能把 `+0.9353/+1.7203/+3.3854` 直接写成 `T05 + Boxer` 的增益；T05 组合仍需单独跑 official100 paired AP。
3. 当前结果离用户目标 `+10 AP points` 仍很远，最高只在 AP50 提升 `3.3854` 点。
4. 若必须维持严格 30 Hz，下一版本需要异步/选择性 Boxer；同步逐 keyframe 全量替换目前不满足要求。
