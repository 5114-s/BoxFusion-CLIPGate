# Frozen-B6 residual mask-track observer

## Scope

This stage adds exactly one observer to the frozen B6 anchor:

```text
frozen B6 globals
  + unmatched SAM3 and/or YOLOE masks
  + aligned real sensor depth
  -> depth-aware mask components
  -> one multi-view residual track graph
  -> diagnostic candidates only
```

It does not enable C4 per-global box refinement, TriFusion occupancy/MSR,
BoxRefiner, supplemental output, Soft-NMS, or a score write-back.  Formal B6
boxes, scores, count, stable IDs, labels, and order are protected by an
in-process byte-level zero-write audit.

`graph_contract_valid` and `graph_confirmed` mean only that a candidate passed
the deterministic graph contract.  They do not mean that its IoU with hidden
ground truth is high.

## Source modes

- `sam3`: immutable SAM3 teacher-cache masks only.  This is the recommended
  first ablation because it changes one evidence source.
- `yoloe`: online YOLOE masks only.
- `dual`: YOLOE and SAM3 observations are concatenated and passed to one graph
  update per scheduled provider call.  Same-view cross-provider duplicates
  therefore cannot create a false multi-view confirmation.

The default graph requires two distinct frames, uses a provider-call TTL of
10, and keeps metric geometry as a hard association condition.

Its semantic compatibility threshold remains `0.50` by default.  A
single-variable diagnostic can disable only that semantic gate while retaining
all geometry/global-overlap gates and the zero-write output contract:

```bash
BOXFUSION_RESIDUAL_MIN_SEMANTIC_SCORE=0.0 \
BOXFUSION_RESIDUAL_RUN_TAG=residual_track_sam3_semantic0_train_smoke2_v1 \
bash scripts/run_scannet_residual_track_observer.sh 0,1
```

This environment/CLI override is rejected for every profile other than the
exact `residual_track_observer`.  Values must be finite and lie in `[0, 1]`.

## Engineering smoke test

The default command uses two ScanNet train scenes and does not touch validation
thresholds:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_yidu_dev

BOXFUSION_RESIDUAL_DRY_RUN=1 \
bash scripts/run_scannet_residual_track_observer.sh 0,1

bash scripts/run_scannet_residual_track_observer.sh 0,1
```

Use a fresh run tag for every code/config/source change:

```bash
BOXFUSION_RESIDUAL_RUN_TAG=residual_track_sam3_train_smoke2_v2 \
bash scripts/run_scannet_residual_track_observer.sh 0,1
```

Other source modes:

```bash
BOXFUSION_RESIDUAL_SOURCE_MODE=yoloe \
BOXFUSION_RESIDUAL_RUN_TAG=residual_track_yoloe_train_smoke2_v1 \
bash scripts/run_scannet_residual_track_observer.sh 0,1

BOXFUSION_RESIDUAL_SOURCE_MODE=dual \
BOXFUSION_RESIDUAL_RUN_TAG=residual_track_dual_train_smoke2_v1 \
bash scripts/run_scannet_residual_track_observer.sh 0,1
```

The SAM3 cache remains read-only.  A cache failure in `sam3` or `dual` mode
clears the whole scene's residual graph, while B6 output continues unchanged.

## Export candidates and measure the recall ceiling

Observer candidates can be exported without ground truth:

```bash
python tools/export_residual_track_candidates.py \
  --diagnostics-root diagnostics/residual_track/residual_track_sam3_train_smoke2_v1 \
  --scene-list /data/ZhaoX/OVM3D-Dett/boxfusion_p1g_dev/evaluation/data_util/meta_data/scannetv2_train_p1_smoke2.txt \
  --output-root datasets/residual_track/residual_track_sam3_train_smoke2_v1
```

Then run the existing one-to-one ScanNet oracle reporter:

```bash
python tools/report_trifusion_oracles.py \
  --pred-root results/residual_track/residual_track_sam3_train_smoke2_v1 \
  --scene-list /data/ZhaoX/OVM3D-Dett/boxfusion_p1g_dev/evaluation/data_util/meta_data/scannetv2_train_p1_smoke2.txt \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scan-root /extra/ZhaoX/scannet_data/scans \
  --supplemental-root datasets/residual_track/residual_track_sam3_train_smoke2_v1 \
  --output reports/residual_track/residual_track_sam3_train_smoke2_v1_oracle.json
```

The standard AP from this observer run must remain B6-equivalent.  Its useful
measurement is `B6 union residual-candidates` novel recall.

## Progression rule

After the two-scene engineering test, use a frozen train-only 20-scene list.
Proceed to a geometry-completion/refiner stage only if all conditions hold:

- union `Delta Recall@0.25 >= 3` percentage points;
- union `Delta Recall@0.50 >= 1` percentage point;
- at least three new IoU-0.50 true positives;
- at least three different scenes contribute those true positives;
- duplicate candidate rate is below 50%;
- no scene violates the zero-write or cache provenance contract.

If the SAM3 teacher passes but YOLOE does not, the next experiment should be
teacher-to-student proposal distillation.  If the teacher itself has no
IoU-0.50 recall ceiling, tuning graph thresholds or running 100 validation
scenes is not justified; the bottleneck is proposal geometry rather than the
tracking parameters.
