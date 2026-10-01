# TR3D R2b: causal multi-view DINO feature observer

## Status

R2b is implemented and has run on the fixed ten-scene validation subset.  It
is strictly **observer-only**:

- it does not append, remove, move, or rescore any BoxFusion box;
- it does not read CLIP text features and leaves the CLIP semantic branch
  byte-for-byte outside this code path;
- it does not read ground truth while exporting feature sidecars;
- no validation result is permitted to select a deployment threshold.

The frozen active anchor remains G0 Selective Boxer:

`AP15/AP25/AP50 = 40.2787 / 35.4508 / 15.2181`.

## Implemented pipeline

```text
exact epoch-12 TR3D proposal row
  -> exact causal R2a Top-K frame IDs
  -> recompute metric-depth support pixels
  -> depth point -> RGB camera projection
     (pose @ extrinsic_color, intrinsic_color)
  -> official Boxer DINOv3-S/16+ dense map
  -> unique support-cell mean + L2 normalization
  -> multi-view pairwise cosine evidence
  -> immutable R2b sidecar
```

The standalone observer matches Selective Boxer's 960x960 bilinear stretch.
The intended online version reuses `BoxerNet.encode(...)["dino0"]`; it must
not run a second image backbone.

## Provenance and safety contract

Every R2b sidecar is bound to:

1. the exact R2a NPZ bytes;
2. the exact TR3D parent cache and proposal/lineage/Top-K row identity;
3. the strict frozen-G0 prefix row and selected RGB/depth/pose/calibration
   artifact tree;
4. epoch-12 TR3D checkpoint/config hashes;
5. the official Boxer commit, DINO checkpoint, feature config, and feature
   code hashes.

Invalid feature slots use a strict false/zero sentinel. Aggregate vectors and
cosine statistics are recomputed by the loader. Files are atomically created,
made read-only, and never overwritten.

## Reproduction

First create the hardened R2a parent:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev
bash scripts/run_tr3d_r2a_depth_observer10.sh r2a_depth_fixed10_v3
```

Then run R2b on one GPU:

```bash
bash scripts/run_tr3d_r2b_feature_observer10.sh \
  0 r2b_dino_fixed10_v1
```

Both commands require fresh output namespaces. Cache resume is allowed only
with a new report namespace; reports and sidecars are immutable.

## Fixed-ten export measurements

- proposals: 1,688;
- valid feature views: 6,291;
- pairwise feature pairs: 11,607;
- standalone DINO feature compute: 13.00 s summed over ten scenes;
- standalone geometry compute: 9.38 s;
- end-to-end sidecar wall time: 35.48 s.

These are offline observer measurements, not online FPS. Online latency and
memory must be measured again after reusing Selective Boxer's existing
`dino0`; no active claim may use the standalone timing as its speed result.

## Decision rule

R2b may proceed to ScanNet-train-only source-aware calibration only if, on the
pre-registered held-out audit, `score + depth + feature` improves over
TR3D-score-only at the same candidate budget by either:

- at least five percentage points of candidate precision with the same novel
  TP50 count; or
- at least two additional novel TP50 without reducing scene coverage.

Otherwise R2b remains a documented negative/weak ablation and is not
activated.

## Fixed-ten decision

The pre-registered audit **did not pass**. On the clear residual population
(24 IoU50 positives and 851 IoU15 negatives):

| ranking signal | AUC | AP |
|---|---:|---:|
| TR3D score | 0.9560 | 0.6452 |
| depth quality | 0.7778 | 0.1206 |
| DINO consistency | 0.6322 | 0.0407 |
| fixed score+depth+DINO formula | 0.9611 | 0.6611 |

The joint score is marginally better as a global ranking diagnostic, but at
all four fixed budgets (25/50/100/200) it produced exactly the same number of
independent and novel IoU50 true positives as TR3D score-only. The precision
gain was zero percentage points and novel TP50 gain was zero. Therefore:

- do not activate R2a/R2b;
- do not tune a validation feature threshold;
- do not proceed to a source-aware calibrator merely to rescue this result;
- retain R2b as a weak/negative ablation and keep G0 frozen.

The immutable audit is
`reports/tr3d_r2b/r2b_dino_fixed10_v1/feature_audit.json`.
