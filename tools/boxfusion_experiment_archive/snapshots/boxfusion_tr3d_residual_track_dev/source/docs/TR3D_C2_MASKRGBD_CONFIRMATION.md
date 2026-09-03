# C2: multi-view Mask-RGBD residual confirmation

C2 is an isolated observer after C1. It does not modify BoxFusion/R3
predictions and therefore cannot change standard AP by itself.

## Frozen route

```text
frozen R3-active predictions
  + immutable C1 unmatched TR3D tracks
  -> stable C1 depth-feature Top-10 per scene
  -> cached SAM3 instance masks on scheduled RGB frames
  -> exact ScanNet depth + intrinsics + camera pose
  -> class-agnostic 2D overlap
  -> mask-depth backprojection in unaligned ScanNet world
  -> box-local 3D voxel connected component
  -> multi-view confirmation gates
  -> immutable C2 sidecar only
```

The SAM3 label string is stored only for diagnostics. It is not read by any
match, evidence score, or gate. CLIP semantics remain unchanged. C2 has no GT
input; GT is used only by the separate audit process after the sidecars have
been frozen.

Invalid/non-finite ScanNet poses are skipped fail-closed. No pose interpolation
or nearest-frame substitution is allowed.

## Fixed gates

- `mask_any`: at least one geometrically overlapping cached mask.
- `mask1`: at least one strong Mask-RGBD observation.
- `mask2`: at least two strong observations.
- `mask2_depth`: `mask2`, at least 64 connected component points, and mean
  expanded-box point support at least 0.25.
- `mask3_strict`: at least three strong observations, at least 96 connected
  component points, and mean expanded-box point support at least 0.30.

All thresholds are serialized into each sidecar and re-derived by the loader.

## Run

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_residual_track_dev
bash scripts/run_tr3d_c2_maskrgbd_fixed10.sh c2_c1top10_fixed10_v3
```

The command is CPU-only because it strictly replays the existing immutable
SAM3 teacher cache. It snapshots and hashes every frozen R3-active prediction
before and after both export and audit.

## Fixed-10 development result

The first valid run used 126 valid-pose cached views, 100 C1 candidates, and
did not write a prediction file.

| Route | Candidates | hit precision @0.15/0.25/0.50 | novel oracle TP @0.15/0.25/0.50 |
|---|---:|---:|---:|
| C1 Top-10 | 100 | 41.0 / 33.0 / 19.0 | 29 / 25 / 19 |
| C2 `mask1` | 60 | 63.3 / 53.3 / 30.0 | 28 / 24 / 18 |
| C2 `mask2_depth` | 43 | 69.8 / 58.1 / 32.6 | 21 / 19 / 14 |
| Top-5 and `mask2_depth` | 26 | 80.8 / 73.1 / 50.0 | 19 / 18 / 13 |
| C2 `mask3_strict` | 16 | 87.5 / 81.2 / 50.0 | 10 / 10 / 8 |

These are candidate hit/oracle statistics, not standard AP. The result only
authorizes a separate source-aware C3 shadow materializer and an independent
100-scene audit. It does not authorize validation-tuned active output.

## Artifacts

The current immutable result is under:

```text
artifacts/tr3d_c2_maskrgbd/c2_c1top10_fixed10_v3/
  cache/<scene>/p100.c2-maskrgbd.npz
  reports/export_report.json
  reports/gt_audit.json
  reports/summary.txt
```
