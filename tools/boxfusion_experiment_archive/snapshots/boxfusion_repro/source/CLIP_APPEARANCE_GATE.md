# BoxFusion CLIP instance-appearance soft gate

This is the first, deliberately conservative semantic-association ablation.
It reuses BoxFusion's existing CLIP crop feature as an instance-appearance
descriptor. It does not add an MLLM, change Cubify Anything, or replace
BoxFusion's geometric association/fusion.

## Data flow

For every new keyframe:

1. Cubify Anything proposes 2D/3D boxes.
2. Each proposal crop is encoded once by CLIP and stored as a normalized
   `appearance_features` vector.
3. Original 3D OBB overlap or small-object 2D correspondence generates
   geometrically plausible pairs.
4. CLIP cosine similarity continuously adjusts the IoU required to merge each
   pair.
5. Accepted pairs follow the original BoxManager recording and box-fusion
   path.

The representative feature is currently the surviving observation's CLIP
feature. Multi-view feature memory/prototypes are intentionally deferred to
the next ablation.

## Gate

For a pair with geometry overlap `g`, CLIP cosine `a`, and reliability `r`,
the effective merge threshold is:

```text
tau = tau_base + r * appearance_penalty(a)
                 - r * appearance_bonus(a)
```

`r` is derived from the lower of the two Cubify detection scores. Low
confidence therefore falls back to original geometry. `geometry_min_iou`
remains a hard lower bound and `hard_geometry_iou` lets sufficiently strong
geometry override a bad crop.

The initial ScanNet config sets `max_iou_bonus: 0.0`. Thus appearance can
protect against a likely false merge but cannot create a merge that original
geometry rejected. This is safer for adjacent visually similar instances.

## Run

Unit tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  LD_LIBRARY_PATH=/home/admin1/miniconda3/envs/boxfusion_official_cu118/lib \
  conda run -n boxfusion_official_cu118 \
  python -m pytest -q boxfusion_repro/tests/test_clip_appearance_gate.py
```

Full 100-scene ScanNet ablation:

```bash
bash boxfusion_repro/run_scannet_sens_rgb_clip_gate.sh 0
```

Predictions and logs are isolated from the reproduced baseline:

- baseline: `boxfusion_repro/results/scannet_sens_rgb_fixedk_score`
- CLIP gate: `boxfusion_repro/results/scannet_sens_rgb_clip_gate`
- gate logs: `boxfusion_repro/logs/scannet_sens_rgb_clip_gate`

Each scene log ends with `Appearance gate summary`, including geometric
candidates, accepted pairs, baseline merges protected by CLIP, optional
promotions, hard-geometry overrides, adjusted-pair counts, and candidate CLIP
similarity q10/q50/q90. The `spatial` and `correspondence` sub-mappings can
override shared thresholds because their similarity distributions differ.

## First tuning sweep

Keep `max_iou_bonus=0.0` and sweep:

- `low_similarity`: 0.35, 0.45, 0.55
- `max_iou_penalty`: 0.05, 0.10, 0.15
- `hard_geometry_iou`: 0.35, 0.45, 0.55

Compare class-agnostic AP, false-merge rate, fragmentation rate, runtime, and
the number of `protected` associations. Only enable a positive appearance
bonus after verifying that cross-view positive pairs are separable from
adjacent same-category objects.
