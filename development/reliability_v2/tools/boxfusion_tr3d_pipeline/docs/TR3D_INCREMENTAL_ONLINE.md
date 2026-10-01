# Causal incremental TR3D observer

This route replaces the terminal-only TR3D input assumption with a bounded
causal RGB-D voxel memory. Every keyframe is backprojected with its current
depth, intrinsics and pose. A persistent official-TR3D worker runs every five
keyframes and proposals are associated across prefixes. No future frame, GT,
or terminal p100 cache is used by this observer.

The first implementation is deliberately observer-only:

- `mutation_enabled=false` and `applied_count=0`;
- BoxFusion predictions and scores remain unchanged;
- proposal geometry, confidence and cross-prefix hit counts are stored for a
  separate offline GT audit;
- activation is forbidden until a multi-scene recall/precision frontier is
  stable.

## Verified smoke result

On `scene0277_00` (45 keyframes), the live worker made nine TR3D calls over
62,527 voxels. Memory plus model inference cost 37.49 ms per keyframe, or about
1.50 ms per raw RGB-D frame at BoxFusion's gap of 25. At score >= 0.50 and at
least two prefix hits, 12 candidates gave an oracle union gain of 4/5/5 true
positives at IoU 0.15/0.25/0.50. This is a one-scene diagnostic, not an AP
claim and not authorization to append those candidates.

Run the fixed ten-scene observer and audit with:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/run_scannet_tr3d_incremental_observer.sh 0,1
```

For 100 scenes, set a fresh run tag and the full validation scene list. Do so
only after the fixed-ten frontier passes:

```bash
BOXFUSION_INCREMENTAL_TR3D_RUN_TAG=incremental_tr3d_observer_full100_v1 \
BOXFUSION_INCREMENTAL_TR3D_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
bash scripts/run_scannet_tr3d_incremental_observer.sh 0,1
```

The current overall runner still replays the original Cubify/CuTR proposal
cache and the old terminal R3 output for paired evaluation. Therefore its
reported stream FPS is not yet an end-to-end live-system measurement. The new
incremental observer itself is live; replacing terminal R3 output is a later,
gated step.
