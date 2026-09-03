# F3 FastSAM OpenBox projection self-validation：paper100

核验日期：2026-08-29（Asia/Shanghai）

## 结论

F3 的在线、实时、无目标数据训练约束全部通过，但精度容量失败，按预登记规则
`discard_f3_shadow`。不得启用 active birth，也不得把 F3 累计进 Cbest。

F3 是 shadow 模块，因此正式输出仍是冻结 Cbest 基线。下表中的“全部追加”是冻结
selector 的反事实 AP；“grouped GT 上限”是只用于测容量的非部署 oracle。

| official100 / score=1.0 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| 冻结 Cbest 基线 | 31.0130 | 26.7911 | 12.0669 |
| F3 fixed selector 全部追加 | 10.7660 | 8.4730 | 3.3109 |
| 相对基线 | **-20.2470** | **-18.3181** | **-8.7560** |
| F3 grouped GT-selected 上限 | 39.1695 | 31.7612 | 12.8054 |
| 理论上限增量 | **+8.1565** | **+4.9700** | **+0.7385** |

固定 selector 共追加 5,404 个 track。在 AP15/AP25/AP50 下分别只增加
126/75/15 个 greedy TP，却增加 5,278/5,329/5,389 个 FP，因此 constant-score
AP 显著下降。

## 容量门

| IoU | fixed-selector 新增 union matches | B/C grouped 新增 union matches | 最终 +10 所需 |
|---|---:|---:|---:|
| 0.15 | 134 | 157 | 144 |
| 0.25 | 78 | 98 | 144 |
| 0.50 | 15 | **18** | 144 |

F1/H0 在 AP50 已有 63 个新增 union matches；F3 grouped 只剩 18 个，相对 F1
减少 45。预登记的 F3 保留门是 AP50 至少 78（F1 的 63 再增加 15），因此失败。
即使使用 GT 在每条 track 的 B/C 中选择几何，AP50 也只能增加 0.7385 点，无法支持
+10 AP 点目标。

## 约束与实时性

- 100 场、6,817 个关键帧、52,299 个 H0 source 全部通过身份复现。
- prefix-invariance、query-before-commit、one-source/one-track、past-only
  最大逻辑访问帧全部通过。
- F3 mean/p95：11.823/32.873 ms/keyframe，门限 25/40 ms。
- composed p95/max：228.331/377.419 ms/keyframe，门限 250/833.33 ms。
- gap-25 amortized F3/composed：0.473/5.971 ms/source frame。
- 新 GPU allocation 为 0；继承峰值 632,576,000 bytes，低于 4 GiB。
- 无训练、无 online learning、无未来帧访问；shadow 阶段不读 GT、不改原预测，
  GT 仅在全部 create-only receipt 封存后由独立 oracle 读取。

## 完整证据

- 合并 receipt：
  `logs/scannet_fastsam_f3_openbox_paper100_score05/final/F3_FASTSAM_OPENBOX_PAPER100.json`
  (`sha256=59fd76600496f9feb465b96b34697d166d18347a2474078b082606123b8bc06d`)
- Oracle：`F3_FASTSAM_OPENBOX_PAPER100_ORACLE.json`
  (`sha256=055770ece9cb7bfa1704429202466cb05ba6ae400ce84f8a09f20cdb507c7ec3`)
- 实现与测试：48 项 focused tests 全部通过。

下一步不应激活或继续调 F3 selector。瓶颈已由“筛选”转为 AP50 候选几何本身；
需要新的冻结通用几何提议/修正器先把 source-level AP50 容量显著提高，再讨论
past-only 确认和 birth。
