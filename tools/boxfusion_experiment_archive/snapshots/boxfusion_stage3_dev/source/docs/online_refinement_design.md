# Online accuracy refinement

This branch implements an opt-in accuracy path for BoxFusion while preserving
the legacy path byte-for-byte when the feature is disabled.

## Runtime data flow

```text
CuTR RGB-D proposals -------------------------------+
                                                     |
RGB -> supplemental provider -> boxes + masks       |
                         |                           |
depth + K + pose -> masked world points              |
                         |                           |
                         +-> bounded object memory <-+
                                     |
                         robust AABB / BoxRefiner
                                     |
                  multi-view quality score + Soft-NMS
```

The supplemental provider may be:

- `yoloe`: an in-process, lazily imported YOLOE segmentation model;
- `disabled`: exact legacy behavior.

The YOLOE provider can be wrapped by a deterministic, pickle-free NPZ cache.
Cached masks are a debugging/reproducibility mechanism. Runs that use cached
masks must not report cache generation as zero-cost online inference.

The controller does not rewrite the live `Instances3D` objects used by
BoxFusion association. It observes each fused keyframe and refines only the
final exported boxes. This avoids feeding a mask or learned-refiner error back
into later geometric association.

## Safety and benchmark invariants

1. `online_refinement.enabled: false` does not load an external model, read a
   cache, mutate boxes, or change scores.
2. ScanNet refinement is axis-aligned. It changes center and dimensions, never
   optimizes yaw.
3. Sensor depth remains the primary metric depth. Learned depth is not used in
   this branch.
4. Object memory is bounded by `max_points_per_object`; no dense scene map is
   constructed.
5. A supplemental detection is exported only after multi-view confirmation.
6. A refined box replaces the CuTR/BoxFusion box only after passing finite,
   size, shift, point-support, and reprojection checks.
7. Ground-truth boxes are never read by online inference. Oracle and training
   tools are separate executables.
8. End-to-end timing must include supplemental inference, lifting, memory,
   refinement, and score calibration.

## Track identities

For an existing BoxFusion object, the stable key is the minimum original
proposal id in the corresponding `BoxManager.fusion_list` entry. Supplemental
tracks use negative ids owned by the refinement controller. The controller
re-matches by geometry when a BoxFusion representative changes.

The minimum-id rule is preserved exactly while those minima are unique.
Rare overlapping fusion groups can share the same minimum and would otherwise
alias two global boxes to one memory track. In that collision-only case, a
deterministic resolver keeps the minimum for one canonical group and assigns
an unused member id to the other group. A high-range deterministic id is a
last resort when no group member remains. This repair does not change box
geometry, confidence, association, or the normal unique-minimum path.

Candidate TTL has a separate clock from the stored keyframe id.  The legacy
clock advances on every BoxFusion keyframe.  The Stage-3 clock advances only
after a successful supplemental-provider call, so `track_ttl` means missed
opportunities to observe the instance rather than skipped YOLOE frames.
Confirmed tracks that leave the active association window are frozen in a
scene-local archive; unconfirmed tracks are discarded.  Active and archived
confirmed tracks are both eligible for final supplemental output, and an
archived track can still be absorbed when a later proposal matches a BoxFusion
global object.

The conservative B1 ablation disables that archive.  Before exporting an
active confirmed supplemental track, it requires the final 3D AABB's
proposal-score-weighted mean 2D projection IoU to reach 0.30, then rejects it
if its 3D IoU with any valid (all extents at least 0.30 m) BoxFusion global box
reaches 0.30.  The detector-score floor is 0.25.  These gates affect only the
new `supplemental_conservative` profile; the historical
`supplemental_only` profile retains its original behavior.

## Quality feature schema

The quality model consumes this immutable 12-column feature vector:

1. `detector_score`;
2. `mask_confidence`;
3. `valid_depth_ratio`;
4. `depth_support`;
5. `projection_iou`;
6. `geometry_consistency`;
7. `appearance_consistency`;
8. `view_count_quality`;
9. `box_stability`;
10. `source_agreement`;
11. `area_quality`;
12. `refiner_quality`.

The default scorer is deterministic and training-free. A learned linear or
MLP checkpoint is optional and must record its feature schema version.

## Recommended ablations

```text
B0  score_thresh=0.4 protocol baseline
B1  + provider-call candidate TTL
B2  + confirmed-track archive
B1c + B1 conservative projection/global-IoU output gates (archive off)
B3  + Top-K Mask-RGBD object memory
B4  + object-local BoxRefiner
B5  + learned multi-view quality score
B6  + track-aware Soft-NMS
```

Report real-score and public-code constant-score protocols separately. Tune on
a development split; do not repeatedly tune on the final 100 validation scenes.

## Online latency policy

The measured Stage-1 path was about 11.71 FPS, or 85.4 ms per processed frame.
A 10 FPS lower bound leaves only about 14.6 ms for every new component. The
accuracy-first full configuration is therefore not automatically a strict
single-GPU real-time configuration.

For the strict online result:

- run mask proposals sparsely with `inference_every_keyframes: 5` or higher;
- trigger the neural refiner only after multi-view confirmation;
- include cache misses and provider inference in latency;
- report one-GPU and two-GPU latency separately;
- do not describe an NPZ-cache replay as online model speed.

SAM3, Depth Anything 3, and an MLLM are intentionally absent from the online
critical path. They can be offline teachers or oracle tools, but their latency
must not be hidden in the reported online number.
