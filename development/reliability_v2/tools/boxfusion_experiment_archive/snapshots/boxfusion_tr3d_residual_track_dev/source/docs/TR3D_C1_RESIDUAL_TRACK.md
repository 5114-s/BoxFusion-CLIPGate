# C1 unmatched TR3D multi-view evidence-track observer

C1 targets AP15/AP25 recall that geometry replacement cannot recover.  It
starts from the frozen R3-active output, selects only class-agnostic TR3D
proposals with maximum AABB IoU `<= 0.15` to every active prediction, and
binds each proposal to its existing Top-5 causal RGB-D observations.

The observer records, without ground truth or CLIP access:

- per-view depth support, invalid depth and free-space contradiction;
- the number and temporal span of supporting views;
- DINO multi-view feature availability and pairwise consistency;
- fixed `visible2`, `depth2`, `depth3_strict`, and `depth_feature2` gates;
- deterministic depth and depth-feature ranking scores.

The available cache contains only terminal `p100` proposals.  Therefore this
is a **cross-view evidence track**, not cross-prefix temporal association.
`cross_prefix_tracking=false` is stored in every sidecar and report.

Run the fixed 10-scene observer/audit with:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_residual_track_dev
bash scripts/run_tr3d_c1_track_fixed10.sh c1_r3active_fixed10_v1
```

The exporter cannot write a prediction file.  The GT audit is a separate
process and reports oracle recall headroom, independent candidate hit
precision, within-source duplicate rate, and fixed per-scene ranking budgets.
A PASS only authorizes development of a separate C2 confirmation observer; it
does not authorize active candidate output.
