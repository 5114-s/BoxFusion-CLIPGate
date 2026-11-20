# ScanNet-100 full-system MVSR visibility ablation

Both arms contain exactly 26,058 predictions and share identical native-box
geometry, labels, PLR recoveries, and replayed CALR-v1 recoveries. The only
difference is whether MVSR composite support includes the visibility factor.

| Full-system variant | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| Four-factor MVSR | 46.5123 | 42.2285 | 21.6135 |
| Three-factor MVSR (without visibility) | **47.0867** | **42.6051** | **22.0350** |
| Paired improvement | **+0.5744** | **+0.3766** | **+0.4215** |

Relative to the previously reported full ReCaR-3D result
46.4262/42.2240/21.6788, the new default improves AP15/AP25/AP50 by
+0.6605/+0.3811/+0.3562.

Protocol: official ScanNet validation 100 scenes, category-agnostic 3D AABB AP,
strictly causal cached online evidence, CALR-v1, and unchanged PLR/CALR boxes.
