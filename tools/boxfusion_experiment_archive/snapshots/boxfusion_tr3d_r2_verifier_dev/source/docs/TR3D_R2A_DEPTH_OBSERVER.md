# TR3D R2a depth/free-space observer

This isolated route evaluates the next module after frozen G0 Selective
Boxer and the trained one-class TR3D residual-proposal observer. It does not
change BoxFusion predictions, scores, ordering, CLIP labels, or AP.

## Causal input correction

The retired `tr3d_prefix_val10_causal_p100_v1` export appended the source
tail frame and dropped frames whose raw pose contained infinity. That does
not reproduce frozen G0. The replacement namespace is:

```text
data/tr3d_prefix_val10_boxfusion_causal_p100_v2
cache/tr3d_prefix_boxfusion_causal_p100_fixed10_v2
```

The exporter now reproduces the `demo.py` post-frame tail guard and the
`ScannetDataset.load_poses()` previous-valid-pose policy. Examples:

| scene | strict G0 keyframes | last timestamp |
|---|---:|---:|
| scene0568_00 | 66 | 1625 |
| scene0435_00 | 130 | 3225 |
| scene0277_00 | 45 | 1100 |

Each manifest row records the selected source timestamps, raw and resolved
pose lineage, paths, and hashes. R2a additionally hashes all selected RGB,
depth, input/resolved pose, and calibration artifacts.

## R2a computation

For every immutable TR3D OBB:

1. project it into every causal depth keyframe without decoding depth;
2. select a stable Top-K by projected area, then frame id;
3. decode only the union of selected depth frames;
4. intersect sampled metric-depth rays with the world yaw OBB;
5. count `support`, `occluded`, `free_space`, and `invalid` pixels;
6. write a separate immutable sidecar bound to the parent TR3D NPZ SHA.

This is a real-depth consistency/free-space module inspired by SMOV3D's
multi-view geometric consistency principle. It is not a claim that SMOV3D
itself contains this exact free-space classifier.

## Fixed10 observer result

The strict R1 parent has 1,688 proposals. Relative to frozen G0 on the fixed
10 scenes, its raw union oracle changes recall from `50.34%` to `80.54%` at
IoU 0.25 and from `29.53%` to `67.11%` at IoU 0.50. Raw AP50 candidate
precision upper bound is only `3.32%`, so direct append remains forbidden.

R2a processed all 1,688 proposals in 24.57 seconds summed CPU scene runtime.
The fixed validation diagnostics are:

| fixed gate | candidates | independent P50 upper | delta oracle R50 | novel TP50 |
|---|---:|---:|---:|---:|
| visible | 1,345 | 1.93% | +14.77 pp | 22 |
| depth loose | 791 | 3.29% | +14.77 pp | 22 |
| depth medium | 586 | 3.58% | +12.08 pp | 18 |
| depth strict | 382 | 4.45% | +10.07 pp | 15 |
| depth very strict | 206 | 4.37% | +6.04 pp | 9 |

Positive residual proposals have median support `0.378`, versus `0.121` for
clear negatives, and median occlusion `0.043`, versus `0.205`. Thus R2a has
real discriminatory signal and removes many negatives, but depth-only gating
is not precise enough to activate. These rows are validation diagnostics, not
deployment-threshold selection.

## Reproduction

The parent causal cache must already exist. Run a new immutable R2a attempt:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev
bash scripts/run_tr3d_r2a_depth_observer10.sh r2a_depth_fixed10_v3
```

Current audited artifacts are:

```text
cache/tr3d_r2a_depth_fixed10_v2
reports/tr3d_r2a/depth_fixed10_v2/export_report.json
reports/tr3d_r2a/depth_fixed10_v2/depth_audit.json
```

## Decision

Do not activate R2a alone and do not tune these gates on validation. The next
increment is R2b multi-view appearance/feature consistency, calibrated on a
ScanNet-train-only split together with the frozen TR3D score and R2a evidence.
CLIP text features and the final open-vocabulary semantic assignment remain
unchanged.
