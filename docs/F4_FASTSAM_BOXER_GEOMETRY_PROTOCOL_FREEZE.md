# F4 FastSAM/F2 + frozen BoxerNet geometry shadow — paper100 freeze

Frozen: 2026-08-29 (Asia/Shanghai), before any F4 GT/evaluator access.

## Status and question

F4 is a prospective one-shot geometry-capacity experiment selected after the
aggregate F1/F2/F3 results were known; it is therefore not a blind benchmark.
No F4 Boxer geometry has been compared with ScanNet GT at freeze time.  F4
asks only whether the released, frozen BoxerNet can add enough high-IoU
geometry to the exact 52,299 sealed FastSAM sources to make a +10 AP-point
birth branch mathematically possible.

F4 is shadow-only.  It is not a detector, selector, association module or
birth path and cannot modify the 1,788-row Cbest native prefix, CLIP, labels,
embeddings, ordering or formal evaluation score 1.0.

Frozen protocol ID:
`F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100`.

## Frozen inputs and model

- paper100 scene list SHA-256:
  `4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5`.
- F0 receipt SHA-256:
  `07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1`.
- F2 receipt SHA-256:
  `455c0e36e35a30c7ba5915384e4d159a730a47b3368bf4b3fb6a5f6064f25603`.
- F2 oracle SHA-256 (historical comparison anchor only; forbidden to the
  shadow runner):
  `2c3d73f777331617c798aca5e6fdcf819a0267b7d698bdab88f70f7b72dbaff5`.
- source universe: exactly 52,299 selected F0/F2 sources in fixed
  scene/frame/rank order.  Identity is
  `(scene_index, frame_ordinal, frame_id, rank, raw_index, mask_sha256,
  points_and_voxel_keys_sha256, source_id)`.
- Boxer repository commit:
  `1f86542dc342a4b1d474c87c97c5d1d6566d9148` and clean worktree.
- BoxerNet checkpoint SHA-256:
  `d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f`.
- DINOv3 checkpoint SHA-256:
  `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea`.
- official `boxernet.py` SHA-256:
  `a8009c1c0932aaab98bb074a2a4c50e55a3fbdfc3c6cb1afc9e1aef0e5324130`.
- frozen BoxFusion Boxer adapter SHA-256:
  `3e82d49512de4abe61d033c2cca903993a83587d2ea56080ff71e42c2c7372a4`.

The checkpoint metadata names `SpaceportWDS`, not ScanNet.  BoxerNet and
DINOv3 are external general pretrained models.  All parameters are eval-only
and frozen; no ScanNet training, fine-tuning, calibration, optimizer or online
learning is allowed.

## Exact current-frame Boxer input

F4 reuses the validated BoxFusion Boxer adapter path, not the unrelated raw
OWLv2+Boxer cache.  For every successful scheduled F0 frame:

1. Open only the current frame RGB, depth and pose paths recorded in the F0
   sidecar and re-hash their bytes.  Load the sealed 640x480 depth intrinsic.
   No directory enumeration, prefetch, future frame, native prediction, GT or
   evaluator access is allowed.
2. Reproduce the BoxFusion ScanNet input: RGB is converted BGR->RGB and
   bilinearly resized to 640x480; metric depth is 640x480; image and depth K
   are the same sealed depth-camera K used by F0.  Pose is the exact F0 pose.
3. Fail-closed join each F2 source to the F0 candidate and mask diagnostic by
   `(rank, raw_index, mask_sha256)`, then verify the canonical source ID.
4. The only Boxer 2D input is the sealed FastSAM mask
   `tight_box_xyxy=[x0,y0,x1,y1]`.  The existing adapter deterministically
   maps the 640x480 XYXY box to Boxer's 960x960 X X Y Y convention.  No
   padding, provider-box alternative or box sweep is permitted.
5. One batch forward processes all current-frame source boxes in rank order
   (0--16).  DINO is encoded once per non-empty frame.  Released SDP is enabled
   with 10,000 samples and the adapter's stable `(seed=0, scene, frame)` seed.
   Empty or F0-abstained frames do not invoke Boxer.
6. Boxer confidence, probability, log-variance and raw parameters may be
   recorded as diagnostics only.  They cannot filter a source, change its
   score or choose geometry.

The Boxer output row count must exactly equal the input source count.  Row
order is the only binding from Boxer to source identity; class/name/OWL/CLIP
matching is forbidden.

## Geometry hypotheses

Each source remains one proposal identity with four geometry hypotheses:

- `H0`, `HL`, `HLG`: byte-equivalent sealed F2 geometries.
- `HB`: the frozen BoxerNet OBB for that source's FastSAM tight box.

HB is valid only if center, extent and rotation are finite, all extents are
positive, the rotation is right-handed/orthonormal after the adapter's fixed
SO(3) projection, and the camera-frame center has `z>1e-4 m`.  Invalid HB
abstains; H0/HL/HLG remain intact.  No Boxer probability threshold is used.

Seal the eight world OBB corners, world center, local extent, world rotation,
camera depth, confidence diagnostics, input tight box, source identity and
hashes.  The oracle applies the scene axis-alignment matrix to the eight HB
corners before taking min/max.  It must not first axis-align an expanded world
AABB.

## Shadow, integrity and runtime

All per-scene receipts and shard manifests are create-only.  Required
contracts are `shadow_only=true`, `birth_enabled=false`,
`native_output_mutation=false`, `gt_access=false`, `prediction_access=false`,
`evaluator_access=false`, `future_frame_access=false`, `training=false` and
`online_learning=false`.

Before an oracle is allowed, the merge must prove 100 scenes, 6,817 scheduled
keyframes, 6,726 successful F0 frames, 52,299 source identities, exact source
partition/order, current-frame-only access, frozen input before/after hashes
and zero output mutation.

Cold model load is reported but excluded.  The first three non-empty forwards
per shard are reported as warm-up and excluded only from warm distributions.
Timed online work includes RGB/depth decode, 640x480 reconstruction, datum/SDP,
Boxer forward with CUDA synchronization and OBB conversion.  Hashing and JSON
serialization are audit overhead.

Frozen gates:

- incremental F4 warm p95 `<=100 ms/keyframe`;
- replay-composed F0+F2+F4 p95 `<=350 ms/keyframe` and max `<833.33 ms`;
- replay-composed mean/25 `<=14 ms/source frame`;
- no missed warm gap-25 deadline and total CUDA peak `<=4 GiB`.

### Pre-paper100/no-GT runtime amendment

Before paper100 execution and without opening GT, a complete single-scene
preflight on `scene0568_00` exposed a mismatch between the declared warm-up
policy and the deadline counter: a first/cold composed forward took
`1745.28 ms`, while the warm composed maximum was `336.863 ms`.  The first
three non-empty forwards per shard were already frozen as warm-up and excluded
from every other formal warm runtime statistic.  Counting those same forwards
in the formal gap-25 deadline gate would therefore mix cold-start and steady
online regimes.

The formal gate is consequently clarified, before paper100 and without GT, as
`gap25_warm_deadline_miss_count == 0`.  The all-forward value is retained and
reported transparently as `gap25_all_deadline_miss_count`, but is diagnostic
only.  Frame receipts retain the all-forward miss flag and also identify
whether a miss belongs to the warm subset.  No numerical threshold, geometry
rule, oracle stopping rule, source set or output contract changes.

Replay timing is necessary but not sufficient for active promotion.  Any
later active experiment also needs a separately frozen, same-GPU live
Cbest+FastSAM+Boxer benchmark of at least 15 FPS.

## Post-seal GT oracle

Only after a complete merge may a separate oracle open the frozen Cbest
predictions, GT and evaluator.  Strict IoU comparisons are `>0.15`, `>0.25`
and `>0.50`.  It must first reproduce the constant-score Cbest AP
`31.0130259031 / 26.7911284298 / 12.0668518301`, F1 H0, and the complete F2
H0/HL/HLG grouped results.

Report H0-, HL-, HLG- and HB-only capacity, historical
`Gbase={H0,HL,HLG}`, and `G4={H0,HL,HLG,HB}`.  A source is the left graph node:
even if several hypotheses cross a threshold, one source can match at most
one GT and one GT can match at most once.  Report `G4-Gbase` separately so F2
capacity is never attributed to HB.

For a threshold-specific constructive suffix, first restrict to official
native-greedy-unmatched GT, maximum-match sources, then choose the greatest-IoU
hypothesis with exact ties `H0 > HL > HLG > HB`.  Append in frozen
scene/frame/rank order with score 1.0.  This is explicitly GT-selected,
nondeployable and oracle-only.

## Fixed stopping rule

F4 passes only if integrity, causality and runtime pass and, at all three IoU
thresholds independently:

1. `G4` adds at least 144 native-union matches; and
2. the constructive suffix improves official native AP by at least 10.0
   points.

Failure means `discard_f4_shadow`: no tight/provider-box switch, probability
threshold, SDP rule, validity rule or selector may be tuned and rerun on this
paper100, and no active birth is authorized.  Passing authorizes only a new
pre-registered GT-free selector experiment; it does not itself authorize
birth.
