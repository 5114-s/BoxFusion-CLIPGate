# S3 frozen proposal-source audit and MobileSAM mask-lifting plan

Date: 2026-08-23

Status: local asset audit complete; S3a is authorized only as an output-inert
dev3 shadow.  This document does not authorize H10 ground-truth access, active
birth, or a full100 accuracy claim.

## Decision

Do not replace the frozen OWLv2 + Boxer source yet.  Its raw per-view pool has
already demonstrated enough complementary geometry to make the requested
`+10` absolute-point target possible.  The loss occurs after proposal
generation: the hard unexplained-depth gate, the old Past3 association, and
terminal geometry collapse that raw ceiling.

The next experiment is therefore:

```text
T05 native prefix (unchanged)
  + sealed gap-25 frozen Boxer Top-4 rows
  -> exact paired OWLv2 2D box
  -> MobileSAM batched box-prompt mask refinement
  -> current sensor-depth masked lifting
  -> q02/q98 AABB per view
  -> per-view shadow/oracle only; no tracking or birth in S3a
```

MobileSAM is the best first component because the image embedding is shared by
all four prompts, the exact local checkpoint is runnable, and it changes the
part that failed without discarding the part with measured recall: Boxer
localizes the object; MobileSAM supplies a cleaner instance boundary; metric
depth supplies the 3D points.  FastSAM is the best automatic fallback if the
Boxer-conditioned branch still lacks recall.  SAM3 is a later asynchronous
verifier, not the first S3 source.

## Audit boundary

This audit was read-only apart from this new document.  It searched local
source trees, checkpoints, caches, logs and prior result receipts.  It did not
open H10 ground truth.  It excludes:

- any detector or geometry head trained on the target ScanNet split;
- any future-frame, bidirectional-video, or offline scene-fusion dependency;
- any result whose score/evaluation protocol is silently treated as comparable
  to T05 constant-score evaluation;
- any unavailable checkpoint inferred merely from the presence of wrapper
  source code.

Externally pretrained frozen models are admissible under the user's clarified
constraint.  “Training-free” below means no target-dataset fitting or mutable
model state in this experiment; it does not mean that the foundation model was
never pretrained.

## Exact local checkpoint ledger

| Candidate | Exact local checkpoint | Bytes | SHA-256 | Local status |
|---|---|---:|---|---|
| YOLOE-11s-seg-PF comparator | `/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt` | 27,948,751 | `292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d` | Runnable prompt-free automatic instance masks |
| MobileSAM `vit_t` | `/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/MobileSAM/weights/mobile_sam.pt` | 40,728,226 | `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f` | Runnable; box/point prompted, or expensive automatic-mask grid |
| FastSAM-x | `/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/checkpoints/FastSAM.pt` | 144,943,063 | `c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7` | Runnable automatic masks; checkpoint identifies a 72,234,149-parameter YOLOv8x-seg model |
| SAM1 ViT-B | `/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/checkpoints/sam_vit_b.pth` | 375,042,383 | `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912` | Checkpoint present; prompt or automatic grid |
| SAM1 ViT-H | `/data/ZhaoX/ovmono3d/checkpoints/sam_vit_h_4b8939.pth` | 2,564,550,879 | `a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e` | Checkpoint present; too heavy for the first realtime branch |
| SAM2.1 Hiera-L | `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt` | 898,083,611 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | Code and checkpoint present; prompt/automatic image mode, online video mode possible only with past state |
| GroundingDINO Swin-T | `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth` | 693,997,677 | `3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799` | Text-prompt proposal source; heavier two-model route with SAM2 |
| GroundingDINO Swin-B | `/data/ZhaoX/ovmono3d/checkpoints/groundingdino_swinb_cogcoor.pth` | 938,057,991 | `46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7` | Text-prompt proposal source; not preferred for class-agnostic residual discovery |
| SAM3 | `/data/ZhaoX/Group3D/checkpoints/sam3/sam3.pt` | 3,450,062,241 | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` | Runnable; text and geometric prompts; existing full100 current-frame cache |

Relevant source roots are:

- MobileSAM:
  `/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/MobileSAM/mobile_sam`;
- generic Grounded-SAM light-model wrappers:
  `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-Segment-Anything/EfficientSAM`;
- FastSAM wrapper:
  `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-Segment-Anything/EfficientSAM/grounded_fast_sam.py`;
- SAM2/Grounded-SAM2:
  `/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2`;
- SAM3:
  `/data/ZhaoX/Group3D/third_party/sam3`.

There is wrapper/source material for EfficientSAM, EdgeSAM, RepViTSAM and
LightHQSAM, but no corresponding local checkpoint was found.  No runnable
generic SEEM or OpenSeeD source-plus-checkpoint pair was found.  Mask2Former is
represented only by installed library code/configuration fragments, not a
verified generic checkpoint.  Target-dataset Mask2Former heads would also
violate this experiment's constraint.  No relevant model-name artifact was
found in the local Hugging Face or Torch caches.

The resulting capability decision is:

| Family | Automatic masks without an external prompt? | Local speed evidence | T05 constant-score AP evidence | S3 decision |
|---|---|---|---|---|
| MobileSAM | Not in the realtime profile; it needs a box/point, while its automatic grid would be a different expensive route | Formal Top-4 GPU receipt below | None | **Use now** as category-agnostic refinement of proven Boxer locations |
| FastSAM-x | Yes | One-frame CPU smoke; upstream README says 64 ms GPU | None | Keep as automatic S3b residual source; first establish precision on sealed data |
| SAM1 ViT-B/H | Prompted or automatic grid | No valid local GPU receipt | None | Dominated by MobileSAM for the first bounded prompt route |
| EfficientSAM/EdgeSAM/RepViTSAM/LightHQSAM | Normally prompted; wrapper-dependent | No checkpoint, hence no runnable timing | None | Unavailable locally |
| SAM2.1 | Prompted or automatic image masks; video propagation must be constrained to past state | SUN cache logs about 235 ms/item for box masks | None | Feasible later, but slower and not better evidenced than MobileSAM |
| Grounded-SAM2 | Text detector plus prompted masks | SUN pipeline logs about 382 ms/image | None | Not class-agnostic and unnecessarily heavy for Boxer-conditioned refinement |
| SAM3 | Text or geometric prompts | Existing current-frame full100 cache about 0.735 s/trigger | Incomparable fixed-ten receipt, negative at AP25/AP50 | Async S4 verifier only |
| Mask2Former | Model/config dependent automatic instance/panoptic output | No generic local checkpoint or timing | None | Unavailable; reject any ScanNet-supervised head |
| SEEM | Model/prompt dependent | No local source-plus-checkpoint pair | None | Unavailable |
| OpenSeeD | Open-vocabulary prompt/taxonomy dependent | No local source-plus-checkpoint pair | None | Unavailable |

Frozen OWLv2 + LVIS+ is open-vocabulary rather than strictly class-agnostic.
S3 nevertheless never consumes or emits its names: OWLv2 supplies only a
location seed and MobileSAM refinement is category-agnostic.  If the location
source itself must also be class-agnostic, FastSAM is the only verified local
automatic candidate, but it currently lacks precision/AP evidence.

## Import and runtime evidence

### MobileSAM

In the local `pcdet` environment, with writes disabled, the source imports
`vit_t`, `SamPredictor`, and `SamAutomaticMaskGenerator`.  The exact weight
instantiates on CPU as a `Sam` model with 10,130,092 parameters and image size
1024.  Environment versions observed were PyTorch `2.2.2+cu121` and
Ultralytics `8.3.226`; CUDA is hidden inside the read-only audit sandbox.

A current-frame CPU smoke on the non-H10 image
`/extra/ZhaoX/scannet_data/scans/scene0377_02/color/650.jpg` succeeded.  It
returned three prompt hypotheses of shape `(3, 968, 1296)`; measured CPU time
was 0.4698 s for the shared encoder and 0.0476 s for the prompt decoder.  This
is viability evidence, not a realtime GPU claim.

The formal local runtime receipt uses exactly the S3 profile: depth-aligned
`640x480`, four batched box prompts, `multimask_output=true`, and selection by
maximum frozen predicted IoU.  On an RTX 3090 with PyTorch `2.6.0+cu124`, after
three warm-ups, ten runs measured 27.2948 ms mean, 28.0012 ms p95 and
28.0863 ms maximum.  The mean encoder time was 21.9203 ms, mean decoder plus
host mask handling was 5.3745 ms, and peak allocated memory was 312,496,128
bytes.  Every measurement was below the 833.33 ms gap-25 budget.

The reproducible benchmark is `tools/benchmark_mobilesam_boxprompt.py`,
SHA-256
`062d29814df5123fe207d3be7a0862d42327a605e4f2e8b5e0c530914c993eb0`.
Its receipt is
`logs/scannet_s3_mobilesam_runtime/mobilesam_top4_rtx3090_receipt.json`,
SHA-256
`a1769f0186d2cabcfcfea9330a508cc8701d1b07b6ba4a66c35e9922ba489c14`.
This is isolated provider evidence.  It does not yet prove same-GPU co-run
throughput with OWLv2/Boxer plus CuTR/BoxFusion.

The bundled MobileSAM README reports 8 ms for the encoder and 4 ms for one
prompt decoder, 12 ms total on its reference GPU.  It reports FastSAM at 64 ms
and two-point prompt alignment mIoU of `0.71--0.74` for MobileSAM versus
`0.27--0.41` for FastSAM.  Those are upstream measurements, not ScanNet AP and
not an automatic-mask-grid latency.  For multiple prompts, one must share the
single embedding rather than rerun the encoder.

### FastSAM

The exact FastSAM checkpoint loads through local Ultralytics `8.3.226` as a
segmentation model with 72,234,149 parameters.  On the same non-H10 frame at
`imgsz=1024`, `conf=0.25`, IoU `0.90`, maximum 100, a CPU smoke produced 66
boxes/masks of shape `(66, 968, 1296)` in 1.518 s wall time.  Reported component
times were 3.84 ms preprocessing, 1091.55 ms inference and 188.26 ms
postprocessing.  The earlier YOLOE-PF engineering smoke on this frame produced
22 masks.  Thus FastSAM is plausibly more complete on this one frame, but the
66-versus-22 observation is not a recall or precision metric and predicts a
substantial false-positive burden.

### SAM2/Grounded-SAM2 and SAM3

Existing local SAM2 pipeline logs are slower and use a different dataset and
protocol:

- `/data/ZhaoX/OVM3D-Dett/build_detic_gsam_cache_sun.log`: 355 box-prompted
  SAM2 items in 83 s, about 235 ms/item;
- `/data/ZhaoX/OVM3D-Dett/build_gsam2_unidepthv2_cache_calibrated_sun.log`:
  4,929 train images in 31:20 and 356 validation images in 2:15, about
  382 ms/image.

They are SUN RGB-D/offline cache timings, not valid T05 accuracy evidence.

The exact SAM3 full100 cache is at
`/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/cache/sam3_teacher/sam3_teacher_full100_c050_frozen_v1`.
Its metadata receipts are
`/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/logs/sam3_teacher_full100_c050_frozen_v1/metadata/shard0.json`
and `shard1.json`.  They record 100 scenes, 1,401 current-only provider calls,
6,280 proposals, gap 25 with provider interval 5, BF16 and resolution 1008.
The two shards took 508.598 s for 671 frames and 520.848 s for 730 frames:
about 0.735 s per provider trigger overall.  It is causally usable as an
asynchronous interval-5 branch, but it is far more expensive than MobileSAM
refinement.

That existing SAM3 cache used 18 ScanNet text prompts.  It therefore must not
be reused as evidence for a class-agnostic S3 source.  The local SAM3 API does
support geometric boxes through `Sam3Processor.add_geometric_prompt`; a later
S4 can use only residual boxes and ignore text/class output.

## Existing accuracy evidence and protocol compatibility

| Source/route | Local evidence | What it establishes |
|---|---|---|
| YOLOE-PF S2 terminal | `logs/scannet_s2_yoloe_direct_oracle_score05_dev3_v2.json`: 17 appended candidates recover 0 new maximum matches; fixed-suffix AP delta `-6.3267/-6.3267/-2.6632` | The existing YOLOE terminal selection is rejected; it does not prove that every YOLOE raw mask is useless |
| YOLOE-PF raw S2 geometry | `logs/scannet_s2_yoloe_direct_raw_ceiling_score05_dev3_v2.json`: best reported raw union adds `+4/+1/+2` matches | Fails the +10 necessary ceiling at AP25 and AP50 on dev3 |
| Raw frozen Boxer, all rows | `docs/BOXER_UNEXPLAINED_DEPTH_ORACLE_PREFLIGHT.md`: adds `+7/+6/+5` matches, or `+25.0000/+21.4286/+17.8571` recall points | The proposal source has sufficient necessary headroom; association/geometry is the bottleneck |
| Boxer hard depth gate | Same receipt: `+6/+5/+1`, or `+21.4286/+17.8571/+3.5714` | Hard unexplained-depth admission destroys AP50 headroom |
| Boxer old terminal tracks | Same receipt: `+1/+0/+2` | Past3/terminal geometry collapses the source; do not reuse it unchanged |
| Boxer score-only Top-4 | `logs/scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json`: `+3/+3/+4`, or `+10.7143/+10.7143/+14.2857` | Smallest tested per-frame cap retaining +10 necessary headroom at all IoUs |
| SAM3 fixed-ten supplemental | Observer `41.5741/34.9987/15.3455`, supplemental `41.8258/34.4462/15.1451`, delta `+0.2517/-0.5525/-0.2004`; ARecall unchanged | No positive AP25/AP50 evidence; fixed-ten, real-score, `conf_thresh=0.05` protocol is not T05 constant-score dev3/full100 |
| MobileSAM, FastSAM, SAM1, SAM2/Grounded-SAM2, EfficientSAM, SEEM, OpenSeeD, generic Mask2Former | No valid local ScanNet T05 constant-score AP receipt found | Model choice must be treated as a new shadow experiment, not as an expected AP guarantee |

The Boxer GT-selected fixed suffix reported `+12.5744/+14.4172/+9.9254` AP
points for the all-row pool, but this is a constructive oracle, not a formal
upper bound: constant score `1.0` creates unstable global tie ordering when a
suffix length changes.  Maximum-cardinality matching is the primary necessary
ceiling diagnostic at this stage.

## Why MobileSAM can recover the raw Boxer ceiling

The raw Boxer result proves that many missed GT objects are localized in the
candidate pool.  It does not prove that the old terminal geometry is usable.
The failure pattern distinguishes those two questions:

1. Raw Boxer union adds 5 matches at IoU 0.50.
2. The hard unexplained-depth gate leaves only 1 at IoU 0.50.
3. The old terminal route leaves only 2 at IoU 0.50 and none at IoU 0.25.

MobileSAM preserves the frozen candidate's current-view 2D location but
replaces the learned Boxer 3D box as the lifting support.  A prompt-conditioned
mask can exclude wall, floor and adjacent-object pixels inside the loose 2D
rectangle.  Intersecting that mask with current metric depth then yields an
object-specific point fragment; fixed q02/q98 bounds reduce isolated depth
outliers.  This mechanism can improve center and extent errors, especially at
AP50, while avoiding a hard unexplained-depth rejection of otherwise useful
Boxer rows.

“Can” is deliberately conditional.  SAM may choose the wrong object inside a
box, thin masks may lose depth support, and raw Boxer may already have better
geometry for some rows.  S3a therefore records and compares identical Top-4
membership under raw Boxer OBB and MobileSAM-lifted AABB before association is
allowed to confound the diagnosis.  Tracking and birth remain disabled.

## Frozen S3a dev3 contract

### Inputs and selection

- Native prefix remains T05: `score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K3, frozen CuTR/CLIP, formal score exactly `1.0`.
- Scene order is `scene0568_00`, `scene0606_01`, `scene0377_02` from
  `evaluation/data_util/meta_data/scannetv2_graw_e2_preflight3.txt`, SHA-256
  `117b5bea04c557f52d4c2a9435c3961bbaae66e420fb5bb849a278f89fe454fc`.
- The sealed Boxer source is
  `logs/scannet_boxer_unexplained_shadow_clean_in2_v5_score05/sealed/boxer_shadow_candidates.npz`,
  SHA-256
  `c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c`;
  its JSON receipt SHA-256 is
  `84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e`.
- Input cadence is the exact sealed gap-25 T05 schedule.  Invalid poses are
  abstentions; a future valid frame is never substituted.
- Per current frame retain exactly the four Boxer rows with highest frozen
  source score, breaking ties by ascending source row then sealed NPZ row.
  The fixed selection SHA-256 is
  `68049b78dba86441a6b691d1687b9fd2c90fc22f9f6e4c7c78548cc64384b306`.
- Top-4 was selected because it is the smallest tested cap that passes the
  predeclared +10 necessary raw-matching ceiling.  The post-hoc dev3 ceiling
  receipt is
  `logs/scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json`, SHA-256
  `d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f`.
  K=2 failed; K=4 passed with `+3/+3/+4` matches.  This is dev-set protocol
  selection and must remain unchanged on a later untouched set.
- Each selected Boxer row is paired to its exact OWLv2 rectangle without
  semantic inference.  `boxernet.py` assigns `inst_id=arange(M)` before the 3D
  confidence subset; `ObbTW` preserves it; the Boxer CSV writer preserves it;
  the sealed NPZ verifies `per_view_source_instance_id`.  The exact 2D row is
  therefore `(time_ns, zero-based OWL row == per_view_source_instance_id)`.
  The runner must fail closed on an out-of-range ID, timestamp mismatch, frame
  ordinal mismatch, or source-hash mismatch.  Names/labels are neither needed
  nor emitted.

The clean Boxer source commit is
`1f86542dc342a4b1d474c87c97c5d1d6566d9148`.  Pairing-critical source hashes
are:

- `boxernet/boxernet.py`:
  `a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130`;
- `utils/file_io.py`:
  `72b140e7e235571e734e70c4f8c682de133cf1a16615f8c250a046df93ae1ee9`.

The exact OWL CSV SHA-256 values, in fixed scene order, are
`7ce2ca477e84d7430eebd30df706970e69ccbf23c843d862a3e91aaf4bfefe22`,
`f45e8d4ff6183a1f38fa9743a1e05b51235b9d2c16e21ff05f3c3720e9c99609`,
and `d7cd5ac56637efae8a36049566006cdce1188c50bcecc38a1d70049c7978632e`.

### Current-frame mask and lifting

For every frame having at least one selected row:

1. Use the depth-aligned `640x480` current RGB and the current native
   `640x480` metric depth/intrinsics.  Map the frozen square-detector OWLv2 box
   explicitly with `x_640 = x_960 * 2/3` and
   `y_480 = y_960 * 1/2`, then clip it to the image.  There is no inferred
   letterbox transform.  This is the exact invertible unwarp of Boxer's
   direct `960x960` resize; it avoids feeding MobileSAM an unnatural square
   scene and aligns masks directly with sensor depth and the formal benchmark.
2. Run the exact frozen MobileSAM `vit_t` checkpoint once to create the shared
   image embedding.  Submit the current frame's selected boxes as one batched
   box-only prompt call.  Extra positive/negative points are not authorized.
3. Use `multimask_output=true`; for each box, select the returned boolean mask
   with highest frozen predicted IoU, breaking a tie by lowest returned index.
   The predictor's boolean threshold is logit `0`, equivalent to probability
   `0.50`.  Use no text, class, CLIP feature or GT.
4. Remove a one-pixel mask boundary and valid pixels whose four-neighbour depth
   jump exceeds `0.15 m`.
5. Keep only finite depth in `[0.10 m, 6.00 m]`, back-project with the current
   intrinsics and pose, and signed-floor voxelize at `0.02 m`.  Fewer than 16
   unique cleaned voxels is a recorded abstention, not a replacement or
   backfill request.  Replace each occupied voxel by its centroid in sorted
   voxel order.  If more than 2,048 centroids remain, lexicographically sort
   XYZ and retain 2,048 evenly spaced indices using integer `linspace`.
6. Fit the primary per-view world-axis-aligned box with per-axis q02/q98.
   Also export the q00/q100 min/max box over the identical cleaned points as a
   prespecified diagnostic.  It cannot be chosen per row using GT.

The depth range, one-pixel boundary, 0.15 m depth edge, 2 cm voxel,
2,048-point observation cap and q02/q98 rule are transferred unchanged from
the frozen S2 masked-lifting contract.  The 16-voxel floor is the existing
no-GT geometric safety floor.  The exact deterministic geometry reference is
`tools/boxfusion_tr3d_pipeline/boxfusion/object_memory.py`, SHA-256
`c2f3f0e0753a34430f0d9d03c65039aa6eee80114a1337676ec4b5f1eaa60938`.
None is fitted to the S3a GT result.

### Output and isolation

The recommended schema is
`boxfusion.boxer_mobilesam_masklift_shadow.v1`, under a new S3-only root such
as `logs/scannet_s3_boxer_mobilesam_masklift_shadow_dev3_score05`.  It must not
reuse or mutate the frozen S2 YOLOE namespace.

Every output row records at least: scene, timestamp, schedule ordinal, sealed
Boxer source row, source instance ID, exact OWL rectangle, MobileSAM predicted
IoU, selected hypothesis index, mask pixel count/hash, valid-depth and voxel
counts, retained bounded-point count/hash, per-view q02/q98 AABB, q00/q100
diagnostic AABB, latency components, and a reason for every abstention.  The
sidecar records:

- `birth=false`;
- `active_authorized=false`;
- `H10_not_authorized=true`;
- `gt_access=false` in the inference process;
- `future_frame_access=false`;
- `native_mutation_applied=false`;
- native T05 hashes before and after, which must be identical;
- checkpoint, source, schedule, selection and preregistration hashes.

S3a has no tracking, terminal fusion, active native-overlap rejection,
semantic gate, unexplained-depth gate, terminal suffix NMS or low-score append.
It exports every exact Top-4 source row, including masks, bounded points,
primary/diagnostic boxes and explicit abstentions, so later association cannot
hide whether MobileSAM lifting itself preserved the proven ceiling.

## S3a evaluation and stopping rule

After the no-GT sidecar is complete and sealed, a separate read-only dev3
oracle may compare, on identical selected per-view membership:

- raw Boxer OBB maximum matching and native-union additional matches;
- MobileSAM q02/q98 AABB matching and native-union additional matches;
- the prespecified q00/q100 diagnostic on identical cleaned points;
- per-row change in best IoU and the counts crossing `0.15/0.25/0.50`;
- mask/valid-depth abstention rate; and
- isolated and integrated latency.

The necessary per-view geometry gate for continuing this branch is at least
three additional native-union matches at each IoU, equivalent to at least
`10.0` recall points over 28 dev3 GT objects.  A route failing any per-view
threshold is rejected as the primary masked lifter.  Passing dev3
authorizes only a new frozen, no-GT H10 shadow; it does not authorize opening
H10 GT or active birth without a separate decision.

Realtime qualification is separate from accuracy.  The isolated MobileSAM
Top-4 receipt is comfortably below the 833.3 ms gap-25 budget, but the required
proof is an integrated same-GPU run with frozen OWLv2/Boxer and T05.  It must
show a causal bounded queue, no future-frame substitution, no sustained
backlog, and native online throughput at least 10 FPS.  Cold-start time and
peak memory must be reported separately.

## Later modules, conditional on S3a

Only if S3a preserves the +10 necessary geometry ceiling should the route add,
one at a time:

1. past-only association over MobileSAM-lifted fragments using the transferred
   geometry-only `IoU>=0.05 OR (center<=0.75m AND containment>=0.25)`, minimum
   three-keyframe, TTL-10, 2,048-points/observation and 8,192-points/track
   bounded memory contract;
2. soft unexplained-depth reliability as a rank/diagnostic, never the rejected
   hard 0.50 admission gate;
3. an output-inert raw-Boxer versus mask-lifted dual-geometry verifier;
4. past-only third-view confirmation and then a fixed low-score birth
   counterfactual;
5. SAM3 geometric-prompt verification as an asynchronous S4 only if
   MobileSAM ambiguity remains material;
6. FastSAM automatic residual masks as S3b only if the Boxer-conditioned
   Top-4 pool itself is shown insufficient on a larger untouched split.

No later module may change native CuTR/BoxFusion rows, CLIP vocabulary,
category, embedding, or formal constant-score protocol.
