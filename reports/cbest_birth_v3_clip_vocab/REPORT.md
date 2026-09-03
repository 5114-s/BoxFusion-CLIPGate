# Cbest + frozen CLIP-vocab birth-v3：official100 结果

核验日期：2026-08-25（Asia/Shanghai）

## 结论

该模块在数值上有效，但增益很小，不能承担“+10 AP 点”的主目标。它适合保留为
高精度 birth 的语义保护门，前提是后续先提供一个容量足够的高召回 proposal 分支；
当前分支本身不应被描述成主要精度模块。

所有结果使用相同的 official100、`score_thresh=0.5`，并由 evaluator 将每个最终
预测分数统一设为 `1.0`：

| official100 | 框数 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| Cbest（Reliable Top-K + Boxer active） | 1,788 | 31.0130 | 26.7911 | 12.0669 |
| Cbest + CLIP-vocab birth-v3 | 1,790 | **31.0848** | **26.8546** | **12.0859** |
| v3 − Cbest | +2 | **+0.0718** | **+0.0635** | **+0.0191** |

相对未加词表门的 birth-v2-M50，v3 分别改善 AP15/AP25/AP50
`+0.2541 / +0.3645 / +0.0737` 点；也就是说，语义门成功消除了上一版的负增益，
但没有产生足够召回。

## 候选漏斗与质量

- 5,773 个 Past3 receipt；
- 62 个通过 birth-v2 的语义一致、分数、几何、视角和 native-novelty 门，进入
  CLIP gate；
- 186 个历史/当前 RGB crop（每条 receipt 三视角）；
- 60 条被 CLIP-vocab gate 拒绝；
- 2 条通过：`scene0207_02/track211/bathtub` 和
  `scene0696_00/track37/window`；
- aggregate recall 在三个 IoU 阈值均增加 `1 / 1433 = 0.0698` 点，说明两个
  suffix 中一个形成新增 TP、一个为 FP，suffix precision 为 50%。

这个容量在数学上不可能支持 +10 点：即使 62 个候选全部为 TP，对 1,433 个 GT
也只对应约 +4.33 recall 点；实际 gate 只追加两个框。

## 冻结门与训练条件

- 复用原生冻结 OpenCLIP `ViT-H-14` checkpoint；没有训练、微调、optimizer、
  ScanNet train split 或 official100 GT 参与 gate；
- 复用原生 473×1024 `class_features.pt`，没有生成新 prompt embedding，也没有
  改原 BoxFusion 的 CLIP 分类、类别、词表或排序；
- 只把原 473 词表中与 ScanNet18 兼容的 18 行索引作为只读 target subset；
- Raw Boxer receipt 通过 `(time_ns, instance)` 精确映射到 OWL 2D 框；RGB 先按
  原 provider 规则 resize 到 960×960，再 clamp/crop；
- gate 阈值在 official100 evaluator 前冻结：三帧 OWL alias 同组、至少 2/3
  CLIP 全词表 top-1 属于 target 且与 OWL 同组、median target cosine ≥0.20、
  median target-vs-nontarget margin ≥−0.01；
- active selector 的顺序是 v2 scalar/native gates → CLIP gate → 原 self-NMS/cap，
  因此被语义门拒绝的候选不会压制后续通过候选；
- 原 Cbest 的 1,788 行保持逐行不变前缀，只追加两个 class-agnostic、score=1.0
  的框。

这里的“无训练”是指没有在目标数据上训练或在线学习；OpenCLIP、OWLv2 和 Boxer
本身都是外部大规模预训练模型。

## 在线与实时性边界

三个语义证据帧均来自 receipt 确认时刻及过去帧，因此语义证据是 past-only 的。
但本次 AP 使用 sealed proposal 的 terminal replay，并用最终 Cbest 框执行
native novelty、全场排序/NMS/cap；所以它不是严格端到端在线运行证明。严格在线版
仍需把相同 gate 接入 live runner，按 `confirmation_frame_id` 即时处理，并只访问
当时的 native state 和已接受 births。

RTX 3090 上、复用已加载模型的隔离 warm benchmark（真实三 crop、batch=3）为：

- ViT-H image encode + normalize + 473-way score：p50/p95/max
  `68.76 / 69.76 / 70.30 ms`；
- 已有 960×960 RGB 时的 crop + preprocess：p50/p95/max
  `2.95 / 5.19 / 6.40 ms`；
- 若包含磁盘 JPEG decode + 960 resize：p50/p95/max
  `37.82 / 45.13 / 48.07 ms`；
- model + text feature cold load：7.33 s，只应在进程启动时支付一次。

因此在 RGB 已驻留内存时，孤立门控约 71.7 ms/receipt，低于 gap25 的 833 ms
关键帧周期；但这不是 Cbest + OWLv2 + Boxer + gate 的端到端 FPS 证明，也没有包含
每次 H2D。在线实现必须复用原生 CLIP 实例，不能额外加载第二份 ViT-H。

## 复核入口

- shadow runner：`tools/run_scannet_raw_boxer_clip_vocab_shadow_full100.py`
- active materializer：`tools/materialize_scannet_raw_boxer_past3_birth_full100.py`
- 单元测试：
  `tests/test_run_scannet_raw_boxer_clip_vocab_shadow_full100.py`、
  `tests/test_materialize_scannet_raw_boxer_past3_birth_full100.py`
- no-GT sidecar：
  `logs/scannet_cbest_raw_boxer_clip_vocab_shadow_score05/CLIP_VOCAB_SHADOW_FULL100.json`
- active 预测：
  `results/scannet_cbest_raw_boxer_past3_birth_v3_clip_vocab_score05`
- active 审计清单：
  `results/scannet_cbest_raw_boxer_past3_birth_v3_clip_vocab_score05/RAW_BOXER_PAST3_BIRTH_FULL100.json`
- official100 evaluator 日志：
  `logs/cgf_paper100_constant_score/scannet_cbest_raw_boxer_past3_birth_v3_clip_vocab_score05_constant.log`
- Cbest 对照日志：
  `logs/cgf_paper100_constant_score/scannet_t05_boxer_replay_active_score05_constant.log`
- runtime benchmark：`tools/benchmark_scannet_clip_vocab_gate_runtime.py`
- runtime 报告：`reports/clip_vocab_gate_runtime/REPORT.md`

测试状态：26 项 focused tests 全部通过。
