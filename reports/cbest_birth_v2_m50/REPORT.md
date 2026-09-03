# Cbest + Raw Boxer Past3 birth-v2-M50：official100 结果

核验日期：2026-08-25（Asia/Shanghai）

## 结论

该版本 **无效，拒绝累计进 Cbest**。在完全相同的 official100、
`score_thresh=0.5`、最终评测分数统一为 `1.0` 的协议下，birth-v2-M50
提高了召回，但三个 AP 均下降：

| official100 | 框数 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| Cbest（Top-K + Boxer active） | 1,788 | 31.0130 | 26.7911 | 12.0669 |
| Cbest + birth-v2-M50 | 1,838 | 30.8308 | 26.4901 | 12.0122 |
| birth-v2 − Cbest | +50 | **−0.1823** | **−0.3010** | **−0.0547** |

对应 evaluator 的平均 precision / recall 变化（百分点）为：

| 指标 | AP15阈值 | AP25阈值 | AP50阈值 |
|---|---:|---:|---:|
| precision 变化 | −0.8869 | −0.8500 | −0.5617 |
| recall 变化 | +0.5583 | +0.4885 | +0.3489 |

这说明分支确实发现了少量 Cbest 漏检，但新增误检更多；在 constant-score
协议下无法靠“低分追加”保护排序。

## 新增框质量审计

100 场共有 1,433 个 GT。50 个新增框相对 Cbest 的一对一独立新增匹配为：

| IoU | 新增匹配 | birth precision | 最大新增 recall |
|---|---:|---:|---:|
| 0.15 | 8 / 50 | 16% | +0.5583 点 |
| 0.25 | 7 / 50 | 14% | +0.4885 点 |
| 0.50 | 5 / 50 | 10% | +0.3489 点 |

主要失败模式不是三视角轨迹不稳定，而是冻结的 open-vocabulary OWLv2
会稳定检测许多 ScanNet official18 未标注物体，例如显示器、塑料袋、枕头、
笔记本电脑和书。这些在真实场景中可能是物体，但在当前 class-agnostic official18
评测中被计作 FP。深度一致性只能验证“那里确实有物体”，不能单独解决该标注词表
不一致问题。

## 实际执行契约

- 输入：封存的 100 场 Raw OWLv2 + Boxer 资产，共 69,012 个 per-view proposal。
- 每个有效关键帧按冻结 source score 取 Top-K8。
- 关联：只查询已提交历史，`AABB IoU >= 0.10 AND center <= 0.50 m`，TTL=10。
- receipt：首三个不同历史帧冻结，几何为三框 AABB-IoU medoid。
- v2 admission：三个 `sem_id` 完全相同、三对几何一致、视角跨度、双向
  containment novelty、自 NMS、每场最多两个。
- 漏斗：5,773 receipts → 50 births，40/100 场非空。
- Cbest 的 1,788 行逐行保持为 byte-equivalent prefix；只追加 50 行；原 CLIP
  路径不变。
- 无 GT、annotation、evaluator、optimizer 或目标数据训练参与 proposal 生成和
  v2 筛选；GT 只在输出冻结后用于 official100 评测及上面的事后质量审计。

## 实时性边界

本次 official100 AP 使用封存 proposal replay，因此只证明精度，不是整链实时
证据。独立严格因果 H10 provider 的 warm p50/p95/max 为
147.0/227.0/340.9 ms/关键帧，tracker p95 为 1.36 ms；它们满足 gap25 的
833 ms 周期，但当前仍没有同 GPU 的 Cbest + provider + birth-v2 集成 full100
延迟 receipt，不能表述为已证明端到端实时。

## 复核入口

- 实现：`tools/materialize_scannet_raw_boxer_past3_birth_full100.py`
- 单元测试：`tests/test_materialize_scannet_raw_boxer_past3_birth_full100.py`
- active 结果：`results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05`
- v2 审计清单：
  `results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05/RAW_BOXER_PAST3_BIRTH_FULL100.json`
- official evaluator 日志：
  `logs/cgf_paper100_constant_score/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05_constant.log`
- Cbest 对照日志：
  `logs/cgf_paper100_constant_score/scannet_t05_boxer_replay_active_score05_constant.log`

