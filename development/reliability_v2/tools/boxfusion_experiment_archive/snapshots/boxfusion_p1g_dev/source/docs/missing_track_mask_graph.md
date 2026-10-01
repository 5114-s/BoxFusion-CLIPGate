# Missing-track Identity + Incremental Mask Graph

This isolated branch tests the proposal-recall route before combining more
geometry heads. The online path remains:

```text
lightweight YOLOE-compatible proposal/mask
  -> weak/unmatched proposal identity
  -> incremental cross-view Mask Graph confirmation
  -> optional confirmed supplemental output
  -> optional C2 depth/occupancy geometry verification
  -> optional C3 cross-lifecycle stitch observer
  -> optional frozen B6 score
  -> optional B5 on confirmed supplemental boxes only
```

SAM3 is an offline teacher/cache source, not an online dependency. The online
run still uses the lightweight proposal provider. Mask Graph confirmation
requires repeated, compatible evidence from different frames; one frame alone
cannot create an exported supplemental instance.

Build the fixed-ten ScanNet teacher cache on two GPUs:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev

BOXFUSION_SAM3_TEACHER_NAMESPACE=sam3-scannet18-val10-c050-v1 \
BOXFUSION_SAM3_TEACHER_RUN_TAG=sam3_teacher_ablation10_c050_v1 \
  bash scripts/build_scannet_sam3_teacher_cache.sh 0,1
```

The builder uses the exact 128 provider frames observed by the corresponding
BoxFusion run, ScanNet object prompts, BF16 inference, immutable provenance,
atomic NPZ writes, and a strict key replay check. For teacher-quality
diagnostics, replay that cache without loading YOLOE or SAM3 and without
falling back on a cache miss:

```bash
BOXFUSION_MASK_GRAPH_PROVIDER=cache_only \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_DIRECTORY="$PWD/cache/sam3_teacher/sam3_teacher_ablation10_c050_v1" \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_NAMESPACE=sam3-scannet18-val10-c050-v1 \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_MISSING_POLICY=error \
BOXFUSION_MASK_GRAPH_RUN_TAG=maskgraph_sam3_observer_ablation10_c050_v1 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 observer
```

Cache keys bind namespace, logical frame ID, RGB shape/dtype, and RGB content.
The cache reader is strictly read-only. The normal online/FPS experiment keeps
`BOXFUSION_MASK_GRAPH_PROVIDER=yoloe` (the default).

## Eight strict ablations

| Variant | Profile | Export new tracks | C1 safety gates | C2 geometry | C3 stitch | B6 | B5 |
|---|---|---:|---:|---:|---:|---:|---:|
| `observer` | `missing_mask_graph_observer` | no | no | no | no | no | no |
| `supplemental` | `missing_mask_graph_supplemental` | yes, confirmed only | no | no | no | no | no |
| `c1` | `missing_mask_graph_c1_recovery` | yes, including eligible absorbed-confirmed tracks | yes | no | no | no | no |
| `c2_observer` | `missing_mask_graph_c2_geometry_observer` | same as C1 | yes | diagnostics only | no | no | no |
| `c2` | `missing_mask_graph_c2_geometry` | same as C1 | yes | verified mutation | no | no | no |
| `c3_observer` | `missing_mask_graph_c3_stitch_observer` | exactly the C2 rows | yes | verified mutation | diagnostics only | no | no |
| `b6` | `missing_mask_graph_b6` | yes, confirmed only | no | no | no | frozen | no |
| `b5_b6` | `missing_mask_graph_b5_b6` | yes, confirmed only | no | no | no | frozen | confirmed supplemental only |

The observer must preserve exported predictions exactly and is the first
contract check. Compare `supplemental` against observer to measure the proposal
and graph contribution. Compare `b6` against supplemental for score
calibration, then `b5_b6` against `b6` for geometry refinement. Do not attribute
the combined result to B5 or B6 without those preceding runs.

The B6 checkpoint was trained on the established BoxFusion rows. Applying it
to supplemental rows is therefore deliberately isolated and reported as a
source-domain ablation; it is not assumed to improve accuracy in advance.

`c1` is independent of B5/B6. It fixes the lifecycle failure where a
graph-confirmed missing track is deleted after a transient strong global
association. Recovered/live supplemental rows must still pass graph,
projection, score, and global-overlap gates. C1 additionally:

- applies a narrow semantic extent allow-list for `sink`, `door`, and
  `window`, while unknown classes retain the normal extent threshold;
- rejects footprint duplicates with a joint BEV-IoU/containment gate;
- maps supplemental rows into a fixed `0.25–0.399` score band below the
  score-preserving `0.40` global anchor; detector confidence and multi-view
  projection agreement provide a deterministic cross-scene internal rank.

`c2_observer` and `c2` are strict children of C1. C1 first freezes the
supplemental row set, representative, label, stable ID, score, and order. C2
does **not** treat Top-K as the complete point memory. It supplies the
shape-adaptive proposer with two separately bounded point sets:

- the cleaner, view-diverse Top-K `geometry_points` are the classification and
  minimum-evidence points; the C2 semantic allow-list fixes `sink` to the
  solid branch and `door`/`window` to the planar branch;
- bounded all-observation `memory.points` form the active envelope and planar
  occupancy components, retaining surfaces outside the selected Top-K;
- `sink` uses the raw bounded-full-memory envelope;
- `door` and `window` use dense 4 cm voxels, at least five points per voxel,
  26-connected components, and density-first component selection;
- center/extent change, point support, per-view support, reprojection, density,
  global overlap, and supplemental overlap are all checked before mutation;
- a rejected or invalid proposal falls back to the exact C1 row and can never
  delete it or alter its score.

`c2_observer` executes the identical proposal and verification path but leaves
all exported predictions unchanged. Its aligned NPZ fields use the
`c2_depth_occupancy_v1` diagnostics schema.

## C3 fragment-stitch observer

`c3_observer` is a strict observer-only child of active C2. It runs verified
C2 geometry, then partitions stored Mask Graph lifecycle snapshots into
immutable stitch candidates. It never adds a candidate to the detection list,
changes a C2 box, or changes count, order, stable ID, label, or score. There
is intentionally no active `c3` variant.

Two snapshots receive a compatible edge only when:

- their normalized, non-empty labels are exactly equal;
- their event frames differ by at least five keyframes;
- their 3D AABB IoU is at least `0.40`, **or** their
  intersection-over-smaller-volume is at least `0.60` and their center
  distance is at most `0.25 m`.

The implementation uses an anchor-clique partition, not union-find:

1. Within each label, choose the anchor by descending sum of direct-neighbor
   IoU, then distinct neighbor frames, view count, bounded geometry-point
   count, detector score, and finally stable track ID.
2. Consider direct neighbors by pair IoU, containment, center distance, view
   and point evidence.
3. Admit a neighbor only if it is directly compatible with the anchor and
   every member already admitted. An `A-B-C` chain therefore cannot merge
   incompatible endpoints.
4. Keep only clusters with at least two tracks, two event frames, and two
   total views; reject any cluster containing an already graph-confirmed
   fragment.
5. Require at least one `active` or `archived` member, maximum detector score
   at least `0.85`, and mean detector score at least `0.70`.

The representative is the partition anchor and its box is only copied into
diagnostics. The NPZ contract is `mask_graph_fragment_stitch_v2`; it records
aligned track IDs/event frames, representative ID and box, label, states,
view/edge counts, pair geometry, and cluster detector scores. The run summary
prints `c3_stitch(candidates/tracks)` and `c3_s`. Invalid sparse-memory
snapshots are skipped individually; any remaining clustering exception fails
open to zero C3 candidates, so observer diagnostics cannot abort or suppress
an otherwise valid C2 result.

### Fixed-ten offline ceiling, not a reported C3 result

Replaying the implemented anchor-clique rule over the frozen C1-v3 ten-scene
diagnostics yields eight raw observer candidates built from sixteen fragment
tracks. This is a candidate ceiling, not eight accepted detections.

As an explicitly offline, GT-audited thought experiment, appending all eight
at score `0.051` to the frozen C2 predictions changes:

| Source | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| C2 fixed-ten reference | 42.3280 | 35.6790 | 15.7684 |
| C2 + eight hypothetical low-score rows | 43.8554 | 36.6992 | 16.6701 |
| optimistic difference | +1.5274 | +1.0202 | +0.9017 |

This table is an offline upper-bound/ranking audit using the same validation
ten scenes that motivated the thresholds. It is **not** the output of
`c3_observer`, whose evaluation must equal C2, and it is not evidence that an
active exporter generalizes. The eight raw candidates have not yet passed
C1-style class-aware extent, global/BEV duplicate, projection, or an unseen
precision audit.

Applying the implemented runtime anchor-clique first and then the existing C1
extent, frozen-global duplicate, and already-output-track gates leaves four
structurally admissible candidates. The runtime-aligned offline report gives:

| Source | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| C2 fixed-ten reference | 42.3280 | 35.6790 | 15.7684 |
| C2 + four hypothetical score-0.051 rows | 43.0876 | 36.3680 | 16.4490 |
| simulated difference | +0.7596 | +0.6890 | +0.6806 |

Those four rows are `sofa` in `scene0568_00`, `sink` in `scene0574_00`,
`table` in `scene0081_02`, and `table` in `scene0187_01`; three of four reach
IoU 0.50. This remains a same-split simulation, not an active C3 result.

Accordingly:

- always run `c3_observer` first and verify paired C2/C3 evaluation, row
  counts, labels, and observer mutation flags;
- do not tune thresholds repeatedly on `scannetv2_val_ablation10_even.txt`;
- freeze the rule before inspecting the other validation scenes, and report
  the remaining-scene result separately from this ten-scene development set;
- do not add an active C3 export until an unseen audit shows useful
  incremental recall at acceptable candidate precision.

## Fixed ten-scene commands

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev

bash scripts/run_scannet_missing_mask_graph.sh 0,1 observer
bash scripts/run_scannet_missing_mask_graph.sh 0,1 supplemental
bash scripts/run_scannet_missing_mask_graph.sh 0,1 c1
bash scripts/run_scannet_missing_mask_graph.sh 0,1 c2_observer
bash scripts/run_scannet_missing_mask_graph.sh 0,1 c2
bash scripts/run_scannet_missing_mask_graph.sh 0,1 c3_observer
bash scripts/run_scannet_missing_mask_graph.sh 0,1 b6
bash scripts/run_scannet_missing_mask_graph.sh 0,1 b5_b6
```

Defaults:

- scene list: `scannetv2_val_ablation10_even.txt`;
- proposal interval: every 5 keyframes;
- candidate TTL clock: provider call;
- confirmed-track archive: enabled;
- online global filter: identity (`0.0`);
- graph supplemental pre/post-B5 filter: `0.30`;
- final ScanNet post-process: `0.40` by default, matching the
  `b6_iou_mlp_blend040_extent040_full100` anchor;
- each variant has separate results, logs, diagnostics, and evaluation roots.

For the cache-controlled SAM3 replay used by the fixed-ten C1/C2/C3 audit,
run a fresh paired C2 reference and C3 observer with identical cache,
configuration, thresholds, and seeds:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev

BOXFUSION_MASK_GRAPH_PROVIDER=cache_only \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_DIRECTORY="$PWD/cache/sam3_teacher/sam3_teacher_ablation10_c050_v3" \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_NAMESPACE=sam3-scannet18-val10-c050-v3 \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_MISSING_POLICY=error \
BOXFUSION_MASK_GRAPH_RUN_TAG=maskgraph_c2_c3_pair_reference_ablation10_v1 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 c2

BOXFUSION_MASK_GRAPH_PROVIDER=cache_only \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_DIRECTORY="$PWD/cache/sam3_teacher/sam3_teacher_ablation10_c050_v3" \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_NAMESPACE=sam3-scannet18-val10-c050-v3 \
BOXFUSION_MASK_GRAPH_TEACHER_CACHE_MISSING_POLICY=error \
BOXFUSION_MASK_GRAPH_RUN_TAG=maskgraph_c3_stitch_observer_ablation10_v1 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 c3_observer
```

The second run must report C3 candidates while retaining the same C2
evaluation and per-scene detection count/label order. Do **not** require
cross-process pickle SHA256 equality: BoxFusion's random box optimization has
small run-to-run numerical drift even between repeated C2 runs. In the frozen
paired run, C2 and C3 both produced `42.3280 / 35.6790 / 15.7684`; all ten
scene counts and label orders matched, while the maximum cross-run score and
corner differences were `3.08e-4` and `1.43 cm`. A repeated C2 comparison
showed the same approximately `1.38 cm` corner drift.

The true no-mutation contract is therefore enforced in-process by
`test_c3_stitch_observer_is_c2_output_identity_and_dumps_candidates`, which
passes identical frozen arrays through C2 and C3 and requires exact
`numpy.array_equal` for boxes, corners, scores, order, and stable IDs.

`b6` requires `models/scannet_b6_iou_mlp.npz`. `b5_b6` additionally requires
`models/scannet_b5v2_oriented_refiner_prototype.pt`. Override local paths with:

```bash
BOXFUSION_MASK_GRAPH_B6_CHECKPOINT=/path/scannet_b6_iou_mlp.npz \
BOXFUSION_MASK_GRAPH_B5_CHECKPOINT=/path/box_refiner.pt \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 b5_b6
```

To run a paired final `0.30` extent protocol instead, change only the final
export threshold:

```bash
BOXFUSION_MASK_GRAPH_POST_MIN_EXTENT=0.30 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 observer
```

This does not change the online/global identity filter or the supplemental
source-specific gate.

Use a fresh run tag when code, configuration, or checkpoints change:

```bash
BOXFUSION_MASK_GRAPH_RUN_TAG=maskgraph_supplemental_ablation10_v2 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 supplemental
```

Only resume an unchanged interrupted run:

```bash
BOXFUSION_MASK_GRAPH_ALLOW_RESUME=1 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 supplemental
```

## Full 100-scene run

Promote a variant only after the fixed-ten observer identity contract and the
supplemental precision/recall diagnostics pass:

```bash
BOXFUSION_MASK_GRAPH_FULL100=1 \
BOXFUSION_MASK_GRAPH_RUN_TAG=maskgraph_supplemental_full100_v1 \
  bash scripts/run_scannet_missing_mask_graph.sh 0,1 supplemental
```

The same switch works for `c1`, `c2_observer`, `c2`, `c3_observer`, `b6`, and
`b5_b6`. A custom list can instead be set with
`BOXFUSION_MASK_GRAPH_SCENE_LIST=/path/scenes.txt`.

No accuracy gain is claimed until paired runs use the same scene list,
proposal checkpoint, seeds, and evaluation protocol.

## Proposal-recall report

Run this immediately after `observer`, before enabling supplemental output:

```bash
python tools/report_mask_graph_recall.py \
  --diagnostics-root diagnostics/maskgraph_observer_ablation10_v1 \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scans-root /extra/ZhaoX/scannet_data/scans \
  --scene-list evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt \
  --pred-root results/maskgraph_observer_ablation10_v1 \
  --output-json logs/maskgraph_observer_ablation10_v1/mask_graph_recall.json
```

The report separates all graph components, confirmed components, and
confirmed live (`active|archived`) components, then reports class-agnostic
Recall@0.15/0.25/0.50 and the incremental recall over the observer baseline.
Promote `supplemental` only when confirmed-live components add recall without
an excessive component count.

## C3 runtime-aligned offline report

After `c3_observer`, audit the frozen candidates without changing predictions:

```bash
python tools/report_fragment_stitch_ablation.py \
  --diagnostics-root diagnostics/maskgraph_c3_frozen_ablation10_c050_v3 \
  --prediction-root results/maskgraph_c3_frozen_ablation10_c050_v3 \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scans-root /extra/ZhaoX/scannet_data/scans \
  --scene-list evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt \
  --output logs/fragment_stitch_c3_frozen_report_v3.json
```

The `fragment_stitch_offline_ablation_v3` report reads the effective C3
configuration from each diagnostic NPZ, rejects cross-scene configuration
drift, reuses the runtime anchor-clique implementation, and only then applies
the C1 structural gates. Legacy diagnostics without configuration provenance
are explicitly marked `legacy_default`.
