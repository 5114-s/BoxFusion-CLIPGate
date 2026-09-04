# WeDetect-Uni birth：离线因果确认 + 低分追加的 AP 兑现（sealed replay）

核验日期：2026-09-03（Asia/Shanghai）

## 结果

| official100 real-score | 框数 | AP15 | AP25 | AP50 | Δ vs Cbest |
|---|---:|---:|---:|---:|---|
| Cbest（Top-K+Boxer+real-score） | 1,788 | 34.9863 | 31.4140 | 15.6662 | — |
| **Cbest + WeDetect birth @0.05/0.10** | +593 | **38.0114** | **33.9979** | **16.5168** | **+3.03 / +2.58 / +0.85** |
| Cbest + WeDetect birth @0.20 | +593 | 38.0096 | 33.9964 | 16.5152 | +3.02 / +2.58 / +0.85 |
| 参照：F4+Stream3Dv2-lite（FastSAM，14.38 FPS） | +600 | 38.1936 | 33.6868 | 15.7113 | +3.21 / +2.27 / +0.05 |

- AP15 兑现 Stream3Dv2-lite 的 94%（+3.03 vs +3.21），AP25 反超（+2.58 vs +2.27），
  **AP50 = +0.85，是 lite（+0.05）的 17 倍**——质量审计（44 vs 24 个 IoU≥0.5 有效框）的预言在 AP 层面兑现。
- 相对论文原始（29.21/24.64/8.04）：+8.80/+9.35/+8.48——**全项目最好的 AP25/AP50**。
- 追加分数 0.05/0.10/0.20 三档结果到第三位小数才分开：再次确认"低位即可、精确取值不敏感"。

## 方法（birth-v2 契约 + Stream3Dv2-lite 教训）

1. 候选生成：全 100 场景 gap-25 关键帧，WeDetect-Base-Uni（冻结，score≥0.05，每帧 top-150）
   → 冻结 BoxerNet 提升（同 Cbest checkpoint，feature cache）→ 世界系 OBB。
   资产：`results/wedetect_lifted_cache/`（100 场，~47k per-view 候选）。
2. 因果确认（past-only）：按帧序处理；AABB IoU≥0.10 且中心距≤0.50m 关联 receipt；
   TTL=10 个关键帧序数；≥3 个不同历史帧确认；medoid 几何。
3. 追加：与 native 终框 AABB IoU≥0.10 去重；birth 间自 NMS 0.5；每场上限 6；
   低分追加（0.05–0.2 带）；native 前缀逐字节保持。
4. 评测：SHA 密封 real-score 评测器，与 Cbest 同协议。

## 593 个追加框质量（对照 Stream3Dv2-lite 的 601 框）

| | 本方法 | Stream3Dv2-lite |
|---|---|---|
| best-IoU≥0.15 | 110（18.5%） | 107（17.8%） |
| ≥0.25 | 93（15.7%） | 77（12.8%） |
| ≥0.5 | **44（7.4%）** | 24（4.0%） |

同容量、更高几何质量——WeDetect-Uni 换源判定的最终兑现。

## 诚实边界

- 这是 **sealed replay**（离线追加），不是在线管线：FPS 主张需在运行时集成后实测
  （成本结构有利：WeDetect ~110ms/帧含 I/O、每 2 关键帧摊销 ≈2.2ms/帧；Boxer 批量+缓存）。
- native 去重用的是终态 native 框（离线近似；在线版对当时地图去重，略宽松）。
- 无词表门：483/593 追加框无 GT 匹配（靠低分排序保护，无害但可清理）。
- CA-1M 未验证。

## 过程记录（两个已修复的坑）

1. TTL 用了原始 frame id：gap-25 下相邻关键帧差 25 > 10，receipt 全部立即过期（首跑 0 追加）；
   改用关键帧序数后 593 追加。
2. 追加行写成 pkl 第二个场景条目：评测器只读第一个场景，首轮评测静默等于 Cbest；
   改为 extend 进第一个场景。
（另一个低级错误：后台任务相对路径因 shell cwd 漂移找不到脚本，改绝对路径。）

## 结论与下一步

流②（WeDetect-Uni proposal 源 + 因果确认 + 低分追加）**在 AP 层面兑现**：
+3.03/+2.58/+0.85，替代被 FPS 判死的 F4/Stream3Dv2-lite 且 AP50 大幅更好。
剩余：运行时集成（Instances3D 造形 + 契约）、管线内 FPS 实测（≥15）、词表门消融、
CA-1M 验证；以及与流①（PVQ-AR@NMS，88 被吞 GT 桶）的叠加实验——两者目标桶不相交。
