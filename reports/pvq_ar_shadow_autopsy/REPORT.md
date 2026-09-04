# PVQ-AR CA 歧义事件验尸报告（shadow full100，95 事件）

核验日期：2026-09-03（Asia/Shanghai）

## 结论（一句话）

**95 个歧义事件几乎全部发生在"垃圾对垃圾"的决策上：95/95 的 proposal 和 186/190 的候选
track 与任何 GT 的 3D IoU 都 < 0.1。对垃圾框之间的选择做任何仲裁都不可能移动 AP——
这就是 active ≈ 噪声（35.03/31.46/15.70 vs 基线 34.99/31.41/15.67）的确定性解释。
门 3（oracle 上限）按构造 ≈ 0，无需管线运行即可判死该触发器。**

## 判定门结果

| 门 | 结果 | 说明 |
|---|---|---|
| 门 1（事件数 ≥100） | 95/100，未过 | 31/100 场景有事件，高度集中 |
| 门 2（正确答案在 Top-3 ≥95%） | 形式上 100% | 但空洞：choice set 恒为 2 个候选 + null；FP proposal 的正确答案是 null，恒在集合内 |
| 门 3（oracle 上限 ≥+2/+2/+1） | **按构造 ≈ 0** | 唯一能影响 AP 的事件至多 4 个（候选 ≥0.1 IoU 的），影响 ≤4/1433 GT ≈ 0.3 recall 点 |

## 核心数字

- proposal（95 个）对全场 GT 的 best-IoU：**median 0.0012，p75 0.0030，max 0.0205**；≥0.1 的 **0 个**。
  变换后离最近 GT 中心 0.5–1.4 m（在场景内、无重叠——单视角小物体深度误差的典型形态）。
- 候选 track 框（190 个）：≥0.1 IoU 的仅 **4 个**（≥0.25 的 2 个，max 0.83）；
  离最近最终输出框中心 median 0.53 m——事件聚集在不存活/不被 GT 匹配的 CA 尾部 track 上。
- reason 分布：native_better 63 / rearrange 22 / abstain_missing_prototype 8 / abstain_low_similarity 2。
  PVQ 假想决策（chosen）与 native 在 GT 上无差别可比——两侧都是垃圾。

## 方法与验证

- 数据：`diagnostics/pvq_ar_shadow_score05/` 的 95 条 `ambiguity_event`（含 proposal 与候选的
  8 角点世界坐标、margin、reason、假想 chosen）。
- GT：复用评测器 `ScannetDetectionDataset` + `parse_groundtruths`（31 个有事件场景）。
- 坐标变换：完整复刻评测器链条（axisAlignment → flip_axis_to_camera → OBB→AABB）。
  **帧守卫**：同变换下最终输出框对 GT 的 match@0.15 = 45%（scene0025_00），与 AP 水平一致，
  证明变换链正确。
- IoU：凸包 footprint + z 重叠的旋转框 IoU；合成框单元测试通过
  （identical=1.0 / disjoint=0 / half=0.3333 / yaw90=1.0）。
- 日志中 `iou` 字段实为 CA 的 **2D 投影 IoU**（margin = iou − 0.10 逐行成立），非 3D IoU。

## 对路线的判定

1. **该触发器（CA top1/top2 margin ≤ 0.10 的歧义拦截）作为 AP 杠杆判死**——证据充分、
   成本一天离线。PVQ-AR active 的无效不是实现问题，是拦截到的东西不值钱。
2. **关联身份问题未被本实验关死**：堆叠同款物体的身份错乱主要发生在 SA/3D-NMS 路径
   （同帧 IoU>0.35 去重、好 track 之间的混淆），当前触发器不覆盖。但按本报告的证据，
   继续在关联侧寻找 AP 杠杆的优先级下调——错误显然更多集中在"垃圾尾部"而非"好框错配"。
3. **下一步转向蓝色 (a) 召回侧（Stage 2）**，前置条件已齐：
   - 通用源召回头部空间 25.0/21.4/17.9（unexplained-depth preflight，门前值）；
   - +3.21/+2.27 可达参照（Stream3Dv2-lite：600 框、0.05–0.2 低分追加）；
   - 硬前提三条：real-score 协议、birth 分数模型（分数位置值 ~1.6 AP，见
     `reports/cbest_birth_v2_m50/REAL_SCORE_REEVAL.md`）、词表对齐门（v2/v3 教训）；
   - WeDetect-Uni 入场走同一套三场景 preflight 协议，不直接集成。
