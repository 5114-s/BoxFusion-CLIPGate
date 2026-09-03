# Reliable-View Top-K + Boxer active：official100 配对测试

核验日期：2026-08-25（Asia/Shanghai）

## 结论

在 `score_thresh=0.5`、最终评测 `score=1.0`、appearance gate 关闭的协议下，
**Boxer active 相对同路径 Boxer observer 有正向 AP 增益**，其中 AP50 增益最大：

| official100 | 最终框数 | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| Top-K + Boxer observer | 1,799 | 30.0728 | 25.4458 | 8.5488 |
| Top-K + Boxer active | 1,788 | 31.0130 | 26.7911 | 12.0669 |
| active - observer | -11 | **+0.9402** | **+1.3453** | **+3.5181** |

所以该组合是有效的，但离 `+10 AP points` 目标仍明显不足；当前最大提升为
AP50 的 `+3.5181` 点。

## 严格配对条件

- 场景：同一 official100。
- 输入：同一 sealed CuTR proposal replay。
- 帧数：两臂均为 6,817。
- proposal 数：两臂均为 23,651。
- Boxer observer：0 次 geometry replacement。
- Boxer active：23,651 次 geometry replacement。
- protected contract：类别、分数、2D 框、embedding、proposal 顺序等异常数为 0。
- 两臂共同启用 Reliable-View Top-K：`top_k=3`、`min_views=3`。
- 评测时所有最终框 confidence 统一设为 `1.0`。

observer 是严格因果分母。旧的原生 T05 结果可作背景参考，但不能替代这个同缓存、
同执行路径的 observer 分母。

## 配对 bootstrap（补充证据）

对 100 个场景做 2,000 次 paired scene bootstrap，active-observer AP 差值的
95% percentile 区间（AP points）为：

| 指标 | 95% CI | 单侧 bootstrap p |
|---|---:|---:|
| AP15 | [-1.3425, 2.7875] | 0.2589 |
| AP25 | [-0.9892, 3.9943] | 0.1389 |
| AP50 | **[2.1123, 5.7970]** | **0.0005** |

因此，AP50 的正向增益在该检验下稳定；AP15/AP25 的 full100 点估计为正，但区间跨 0，
不应表述为统计显著。由于 constant score 存在同分排序敏感性，正式主结果仍以上表的
官方 evaluator 输出为准，bootstrap 只作稳健性补充。

## 在线实时性边界

full100 AP 实验使用 proposal replay，只用于隔离几何模块的精度影响，不能作为端到端
实时性证据。另一个无 replay 的 `scene0277_00` 单卡 live smoke 中：

| live 路径 | FPS |
|---|---:|
| Top-K + CuTR control | 33.92 |
| Top-K + CuTR + Boxer active | 26.34 |

该场景满足用户接受的 20 FPS 门槛，但不代表 full100 延迟分布；而且该场景的 Top-K
更新数为 0，因此它主要证明联合代码路径和 Boxer active 的在线吞吐，不证明 Top-K
实际触发时的最坏延迟。

## 复核入口

- observer 配置：`config/scannet_t05_boxer_replay_observer_score05.yaml`
- active 配置：`config/scannet_t05_boxer_replay_active_score05.yaml`
- full100 runner：`scripts/run_scannet_t05_boxer_replay_full100.sh`
- 配对汇总器：`tools/summarize_t05_boxer_paired.py`
- observer constant-score 日志：
  `logs/cgf_paper100_constant_score/scannet_t05_boxer_replay_observer_score05_constant.log`
- active constant-score 日志：
  `logs/cgf_paper100_constant_score/scannet_t05_boxer_replay_active_score05_constant.log`
- live active 日志：`logs/t05_boxer_live_smoke/scene0277_00_solo_active.log`
- live control 日志：`logs/t05_boxer_live_smoke/scene0277_00_solo_control.log`

