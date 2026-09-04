# WeDetect-Uni 换源判定：2D/3D 召回头部空间 preflight

核验日期：2026-09-03（Asia/Shanghai）

## 判定：通过——WeDetect-Uni + 冻结 BoxerNet 达到并超过 OWLv2+Boxer 参照

三密封场景（scene0568_00 / scene0606_01 / scene0377_02，gap-25 关键帧，共 28 GT、
11 个 Cbest 漏检），proposal 源为 WeDetect-Base-Uni（冻结），3D 提升为冻结 BoxerNet
（与 Cbest 同一 checkpoint，SHA 钉扎一致）：

| 设置 | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| score>0.05 | **39.3** | **32.1** | 14.3 |
| score>0.1 | 32.1 | 25.0 | 10.7 |
| score>0.2 | 25.0 | 21.4 | 10.7 |
| 参照：OWLv2+Boxer 门前（3D） | 25.0 | 21.4 | 17.9 |

- IoU 0.15/0.25（birth 路径的目标阈值，对应 AP15/25）：**每个操作点都 ≥ 参照**；
  score>0.05 时超出参照 +14.3/+10.7 点。
- IoU 0.50 低于参照（14.3 vs 17.9）：类无关 proposal 的提升几何更粗，符合预期——
  AP50 精修是下游多视角融合 + selective Boxer 的职责（Stream3Dv2-lite 的经验：
  追加框增益也集中在 AP15/25）。
- 2D 层（IoU2D≥0.3）：11/11 漏检全覆盖（39.3 点），score>0.2 仍 10/11——
  室内低分不构成障碍。

## 工程事实

- 代码/权重：`third_party/WeDetect/`（vendored 推理 + 413MB Base-Uni，GPL-v3）。
- 接口：每帧 ≤300 proposal + 768 维 embedding；WeDetect 单帧 ~110ms（含 I/O），
  "每 2 关键帧一次"摊销 ≈ 2.2ms/帧。
- Boxer 提升复用 `BoxerLiftingAdapter._make_datum + forward_raw_with_feature_cache`
  （需要 `cache_image_features: true`）；**约束：Boxer 进程内不可导入 evaluation/
  的顶层 `utils`**（命名空间冲突，fail-closed 检查）——preflight 因此拆成
  脚本 A（评测器环境算漏检 GT → npz，已归档 `gt_inputs/`）和脚本 B（干净环境提升）。
- 脚本：`third_party/WeDetect/preflight_2d_headroom.py`、`preflight_3d_headroom.py`。
- 投影约定：ScanNet 标准约定 A（逐场景自动校准在杂乱房间不可靠，曾致 scene0568 误判 0/3）。

## 结论与下一步

WeDetect-Uni 作为 Stage 2 proposal 源的必要条件（2D 可见）与同口径判据（3D 头部
空间 ≥ OWLv2 参照）均满足。下一步进入运行时集成：在 F4/Stream3Dv2-lite 的 birth
机器中以 WeDetect-Uni 替换 FastSM 残差源（每 2 关键帧第二条前向 → 孤儿流 → 多帧
确认 → selective Boxer → 低分追加），机制复刻已被反事实重评证明承载 +3.21/+2.27
的路径；集成时带上三条硬前提（real-score 协议、birth 分数模型、词表审计）与
管线内 FPS 实测（目标 ≥15）。
