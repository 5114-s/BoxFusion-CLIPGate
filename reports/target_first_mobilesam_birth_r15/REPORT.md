# Target-first MobileSAM birth R15：ScanNet official100 核验

## 结论

本次 100 场验证已完成，但 **R15 不应累计进 Cbest**。相对冻结基线，它只在低 IoU 阈值取得小幅收益：AP15 `+1.1912`、AP25 `+0.7279` 个百分点；AP50 反而 `-0.1354` 个百分点。shadow 只有 160 个候选并覆盖 64 场，未通过预注册的 `至少 200 candidates / 至少 70 scenes` 容量门；active 又缩减为 132 个框。

## 固定协议与产物

- 评测集合：`evaluation/data_util/meta_data/scannetv2_val.txt` 的 official100，共 100 场、1433 个 GT 框。
- 基线预测：`results/scannet_t05_boxer_replay_active_score05`，1788 框。
- active 预测：`results/scannet_cbest_target_first_mobilesam_birth_r15_score05`，1920 框，即追加 132 框。
- 上游 proposal 门槛：`score_thresh=0.5`；官方复现评测器在加载后把所有预测置信度强制设为 `1.0`，见 `scripts/eval_scannet_cgf_paper100_constant_score.sh` 和 `upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py:199`。
- Shadow sidecar：`logs/scannet_target_first_mobilesam_masklift_full100_score05/TARGET_FIRST_MOBILESAM_MASKLIFT_FULL100.json`。
- Active manifest：`results/scannet_cbest_target_first_mobilesam_birth_r15_score05/TARGET_FIRST_MOBILESAM_BIRTH_FULL100.json`。
- Official logs：
  - baseline：`logs/cgf_paper100_constant_score/scannet_t05_boxer_replay_active_score05_constant.log`
  - active：`logs/cgf_paper100_constant_score/scannet_cbest_target_first_mobilesam_birth_r15_score05_constant.log`

Manifest 声明且本地逐行复核确认：100/100 场的 1788 个原生框均为 active 输出的完全相同前缀；只追加 132 个 birth，不改变原生框几何、类别或持久化分数。模块未访问 GT/评测器，不做目标数据集训练或在线学习，冻结 MobileSAM；原生 CLIP 不变。

## Official100 AP

| IoU | baseline AP | R15 AP | AP 差值（百分点） | baseline precision | R15 precision | precision 差值（百分点） | baseline recall | R15 recall | recall 差值（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 31.0130 | 32.2042 | **+1.1912** | 48.6018 | 47.5521 | -1.0497 | 60.6420 | 63.7125 | +3.0705 |
| 0.25 | 26.7911 | 27.5190 | **+0.7279** | 45.2461 | 44.0104 | -1.2357 | 56.4550 | 58.9672 | +2.5122 |
| 0.50 | 12.0669 | 11.9315 | **-0.1354** | 30.6488 | 29.3229 | -1.3259 | 38.2415 | 39.2882 | +1.0467 |

由 1433 个 GT 和 recall 反算，132 个新增框分别只增加 44、36、15 个匹配 TP；对应边际 precision 为：

| IoU | 新增 TP | 132 个 birth 的边际 precision |
| --- | ---: | ---: |
| 0.15 | 44 | 33.33% |
| 0.25 | 36 | 27.27% |
| 0.50 | 15 | 11.36% |

因此 AP50 虽增加 15 个匹配和 `+1.0467` 点 recall，但大量不够精确的追加框令整体 precision 下降，最终 AP50 净退化。

### AP50 事后错误画像

132 个 birth 中有 117 个未形成新的 IoU>0.50 匹配：32 个与任意 GT
零重叠，54 个最大 IoU 位于 `(0,0.15]`，10 个位于 `(0.15,0.25]`，
9 个位于 `(0.25,0.35]`，只有 12 个位于 `(0.35,0.50]`。因此多数错误
不是小幅移动框即可修复；仅最后 12 个属于值得优先尝试几何修正的 near-miss。

按冻结 target group 看，`garbage_bin` 的 19 个 birth 贡献 `11/11/8` 个
TP（IoU 0.15/0.25/0.50），是 AP50 最可靠的来源；`window` 的 15 个 birth
贡献 `8/6/0`，说明它提升低阈值召回、但框几何不够精确。该统计只作事后诊断，
不能据此在同一 official100 上重新挑类别或阈值并宣称独立改进。

### Paired scene bootstrap

`reports/target_first_mobilesam_birth_r15_bootstrap/PAIRED_SCENE_BOOTSTRAP_2000_SEED20260822.json` 使用同一组 100 场进行 2000 次 paired scene bootstrap（seed `20260822`）：

| 指标 | observed 差值（百分点） | 95% percentile CI（百分点） | one-sided bootstrap p | 判定 |
| --- | ---: | ---: | ---: | --- |
| AP15 | +1.1912 | `[+0.0290, +1.7572]` | 0.0225 | 仅此项显著为正 |
| AP25 | +0.7279 | `[-0.3053, +1.1951]` | 0.1189 | 不显著 |
| AP50 | -0.1354 | `[-0.6700, +0.2298]` | 0.8341 | 不显著且点估计退化 |

所以不能把这次结果概括为稳定的三阈值精度提升。

## 容量与决策漏斗

| 阶段 | 数量 | 相对比例 |
| --- | ---: | ---: |
| target-first box prompts | 15,313 | 100% |
| MobileSAM 成功 lift | 15,230 | 99.46% prompts |
| past-only 三视角 receipt | 1,818 | 100% receipts |
| 通过几何/视角/分数的 pre-novelty | 950 | 52.26% receipts |
| 通过原生框 novelty 的 shadow candidates | 160 | 8.80% receipts |
| self-NMS 与每场 cap 后 active birth | 132 | 7.26% receipts；覆盖 64 场 |

Active 首个失败原因合计：view diversity 527、score 60、too small 81、voxel support 2、raw-medoid distance 12、mask center distance 88、mask IoU/R15 98、native overlap 645、native containment 145、self-NMS 18、scene cap 10；最终 accepted 132，合计 1818 receipts。

容量结论：

- shadow 160 `< 200`；active 132 `< 200`，容量 gate 未通过。
- shadow 只覆盖 64 场 `< 70`，场景覆盖 gate 未通过。
- active 的实测边际 precision 为 33.33% / 27.27% / 11.36%；这是冻结输出后的质量结论，不是预注册 continuation gate。
- official100 有 1433 个 GT。即使 active 的 132 个新增框全部是 TP，其 recall/AP 增量容量也至多约 `132/1433 = 9.21` 个百分点，仍达不到 `+10`；按 70% precision 仅相当于约 92 个 TP、`+6.45` 点 recall 容量。

## 实时性解释

Shadow 在 RTX 3090 上测了 5650 个实际触发帧：

- provider mean / p50 / p95：`26.22 / 26.46 / 29.50 ms`
- 包含 RGB-D 解码、MobileSAM、mask lifting 的 incremental total mean / p50 / p95：`115.48 / 109.07 / 188.47 ms`
- peak allocated GPU memory：约 `291.61 MiB`
- p95 通过预定 `<200 ms` 模块门槛。

当前输入按 gap25 触发；若以 30 Hz ScanNet 流计，两个触发点相隔约 `25/30 = 833.33 ms`。`188.47 ms` p95 只占该预算的 22.62%，余量约 `644.86 ms`，约有 `4.42x` 调度裕量；摊到原始流约为 mean `4.62 ms/frame`、p95 `7.54 ms/frame`。

这证明的是 **该冻结增量模块在 gap25 事件调度下可在线运行**，不是完整 BoxFusion + OWLv2 + MobileSAM 同机端到端 FPS 证明。当前 full100 还是 past-only 因果 replay，不能把 188.47 ms 直接宣称为整条系统延迟。

## 最终判定

R15 是一个低 IoU recall 分支，但不满足 `+10` 目标：容量不足、边际 precision 不足，并导致 AP50 退化。保留其 shadow 产物用于错误分析；active 结果不进入 Cbest。下一轮应优先解决 birth 几何精度与高置信筛选，而不是放宽门槛继续增加同类候选。
