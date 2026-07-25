# Stage 2: Top-K reliable-view weighted fusion

This ablation extends the score-preserving CLIP appearance-gate experiment.
It does not change Cubify proposals, association thresholds, CLIP gating, or
the confidence score exported for evaluation.

## Experiment lineage

```text
Stage 0  Real Cubify score baseline
Stage 1  + CLIP instance-appearance soft gate
Stage 2  + Top-K reliable-view weighted fusion
```

Stage 2 must be compared with the completed Stage-1 result produced by:

```text
config/scannet_clip_gate_scorefix.yaml
results/scannet_clip_gate_scorefix
```

Its outputs are isolated under:

```text
config/scannet_clip_gate_topk_fusion_scorefix.yaml
results/scannet_clip_gate_topk_fusion_scorefix
logs/scannet_clip_gate_topk_fusion_scorefix
evaluation/scannet_clip_gate_topk_fusion_scorefix
```

## Reliable-view score

For each observation associated with one global object, the implementation
computes:

```text
reliability =
    real Cubify confidence
  * projected visible-area quality
  * 2D detector / projected-3D agreement
  * cross-view 3D center-and-size consistency
```

The terms are soft rather than binary:

- confidence keeps the real proposal evidence;
- projected area gently discounts very small or distant views;
- projection IoU checks whether the detected 2D rectangle agrees with the
  image footprint of its predicted 3D box;
- robust median center/size consistency discounts a geometrically abnormal
  observation without allowing semantics alone to create a fusion.

The BoxManager currently stores at most five observations per object. The
first conservative configuration retains the three highest-reliability
observations.

## Weighted fusion

Selected observations affect both parts of the original optimizer:

1. The initial 3D center and dimensions are reliability-weighted. Dimension
   axes are aligned to the most reliable observation before averaging, and
   rotation still comes from that best observation.
2. The CUDA reprojection objective changes from:

   ```text
   mean(1 - IoU_view)
   ```

   to:

   ```text
   sum(weight_view * (1 - IoU_view)) / sum(weight_view)
   ```

The final class-agnostic detection confidence remains the original real
Cubify score. Stage 2 therefore measures fusion geometry rather than changing
AP through a new score aggregation rule.

## Backward compatibility

`reliable_views` is disabled by default. When it is absent or disabled:

- no Top-K selector runs;
- the full fusion list and its order are preserved;
- the released unweighted initialization is used;
- the CUDA kernel executes the original unweighted atomic-add branch.

This allows Stage 0 and Stage 1 configurations to keep their original
behavior after the Stage-2 patch is merged.

## Conservative first configuration

```yaml
box_fusion:
  reliable_views:
    enabled: True
    top_k: 3
    min_views: 3
    confidence_power: 1.0
    area_power: 0.25
    area_reference_ratio: 0.02
    projection_iou_power: 0.50
    geometry_consistency_power: 0.50
    center_sigma: 0.75
    size_sigma: 0.50
    minimum_box_diagonal: 0.10
    minimum_weight: 0.05
```

The exponents are intentionally conservative. A view is softly downweighted;
only the Top-K ranking is discrete.

## Validation

CPU unit tests:

```bash
cd /data/ZhaoX/BoxFusion
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/admin1/miniconda3/envs/boxfusion2/bin/python \
  -m pytest -q \
  tests/test_clip_appearance_gate.py \
  tests/test_reliable_view_fusion.py
```

Single GPU:

```bash
bash scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh 0
```

Dual GPU:

```bash
bash scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh 0,1
```

The runner supports safe resume and writes unbuffered per-scene logs. Each
scene prints a `Reliable-view fusion summary` with view counts and reliability
quantiles.

Do not start Stage 2 until all Stage-1 scenes have finished and Stage-1 AP has
been recorded. Otherwise the ablation lineage cannot be audited cleanly.
