# ScanNet-100 key controls with final no-visibility MVSR

All five arms use the final three-factor MVSR (proposal score, 2D overlap,
and 3D consistency; visibility removed) with historical-maximum temporal
aggregation. PLR controls share online CALR-v1, and CALR controls share online
PLR. Candidate budgets are matched per scene without reading ground truth.

| Branch | Recovery strategy | Temporal protocol | Added boxes | AP15 | AP25 | AP50 |
|---|---|---|---:|---:|---:|---:|
| PLR | Single-view direct admission | Scene-end matched-budget control | 1,101 | 46.5842 | 41.9827 | 21.8613 |
| PLR | Cross-keyframe verification | Strictly causal online | 1,101 | **47.0867** | **42.6051** | **22.0350** |
| CALR | Directly lower anchor threshold | Per-frame direct release | 23,172 | 29.3262 | 27.4738 | 15.6024 |
| CALR | Scene-end raw-score Top-K | Scene-end matched-budget control | 23,172 | **47.2605** | **42.7209** | 22.0292 |
| CALR | Three-keyframe voxel consensus | Strictly causal online | 23,172 | 47.0867 | 42.6051 | **22.0350** |

Every materialized arm contains 26,058 predictions over the official ScanNet
validation 100 scenes. The two proposed online rows are the same full system,
viewed once as the PLR comparison target and once as the CALR comparison target.

The scene-end Top-K CALR control remains marginally higher at AP15/AP25, whereas
the causal CALR route is higher at AP50 by 0.0058. Therefore this table supports
the online/causal construction and matched-budget comparability, but it does not
support claiming that CALR voxel consensus dominates the non-causal Top-K control
at every IoU threshold.
