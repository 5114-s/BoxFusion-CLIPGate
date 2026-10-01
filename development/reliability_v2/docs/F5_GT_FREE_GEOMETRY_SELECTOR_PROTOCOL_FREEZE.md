# F5 GT-free past-only geometry selector — protocol freeze

Frozen: 2026-08-29 (Asia/Shanghai), before any F5 execution, GT access or F5
evaluation.

Protocol ID:
`F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100`.

## Question and scope

F5 asks whether one deterministic, training-free and causal rule can select
exactly one geometry from `H0/HL/HLG/HB` for every sealed F4 source.  This
document was prepared from the frozen F2/F4 protocols and their no-GT sidecar
schemas only.  No annotation, GT box, evaluator output, oracle source match or
F5 result was inspected to choose these rules.

F5 is a geometry-selector shadow, not a detector.  It cannot add or remove a
source, create a proposal, enable birth, read native predictions, change CLIP
or semantics, or alter confidence, rank, class, embedding, source order or the
formal score `1.0`.  Its only output is a selected hypothesis name and a
deep-copied geometry for each of the same 52,299 source identities.  Native
BoxFusion output remains byte-identical.

The runner and merge schemas are frozen as:

- `boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.scene.v1`;
- `boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.shard.v1`;
- `boxfusion.scannet_fastsam_f5_gtfree_selector_paper100.merge.v1`.

## Sealed inputs and forbidden inputs

The input is one complete, passing F4 merge and its exact scene sidecars under
protocol `F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100`.  The F5 manifest records
and re-hashes the F4 receipt, every F4 scene sidecar, its upstream F2/F0
sidecars and evidence NPZ, the scene list, schedule, intrinsics, RGB, depth and
pose files that it opens.  The F4 receipt SHA-256 is pinned in the create-only
F5 run manifest before processing.

Permitted data are the current source's sealed F2 evidence points and mask,
`H0/HL/HLG/HB`, current RGB-D/intrinsics/pose, F0/F2 diagnostics, Boxer validity
and Boxer confidence, plus the bounded past-only geometry buffer defined
below.  CLIP, category names and all semantic fields are ignored.

GT, annotations, native/terminal predictions, evaluator code or output,
oracle reports, oracle match ledgers, future frames, ScanNet-specific fitting,
training, fine-tuning, calibration, optimizer state and online learning are
forbidden.  No directory enumeration or implicit nearest-frame lookup is
allowed.  An attempted forbidden read aborts the shard.

## Geometry conventions

All online comparisons use the sealed world coordinate system; scene
axis-alignment is not opened by F5.  For `H0/HL/HLG`, the geometry is the
sealed world AABB.  For valid `HB`, point containment is evaluated in its OBB:

`u = R^T (p-c)`, with `abs(u_i) <= extent_i/2`.

HB-to-AABB comparisons use the world AABB obtained by taking min/max of the
eight sealed HB corners.  This envelope is diagnostic only; the selected HB
retains its original center, local extent, rotation and eight corners.

For two AABBs `A,B`, define:

- `IoU3D = volume(A intersection B) / volume(A union B)`;
- `SC = intersection volume / min(volume(A), volume(B))`;
- `ND = center_distance / max(diagonal(A), diagonal(B), 0.02 m)`.

All threshold comparisons below are inclusive.  Computation is float64.

## Step 1: deterministic base hypothesis

Let `n0` and `V0` be H0's sealed point count and volume.  HL or HLG is
base-eligible only when all of the following hold:

1. `valid=true`, `diagnostics.applied=true`, and
   `diagnostics.fallback=false`;
2. retained count is at least `max(16, ceil(0.55*n0))`;
3. its volume ratio to H0 is in `[0.25, 1.05]`;
4. every world-axis extent ratio to H0 is in `[0.35, 1.05]`;
5. its center shift from H0 is at most `0.20*diagonal(H0) + 0.05 m`.

Test HLG first, then HL; the first eligible hypothesis is `Hbase`.  Otherwise
`Hbase=H0`.  Thus the fixed base tie/order is `HLG > HL > H0`.  Non-finite or
missing diagnostics fail the affected hypothesis closed, not the source.

These constants are tied to the frozen 5 cm robustness scale and require a
majority of the original bounded depth evidence; they are not fitted from
paper100 outcomes.

## Step 2: current-frame HB gates

HB can proceed only if its frozen F4 validity is true and all sealed geometry
checks reproduce.  Boxer confidence must be one scalar finite JSON number in
`[0,1]` and be at least `0.55`; vector, missing or out-of-domain confidence
fails HB closed.  Log-variance and raw parameters remain diagnostics and do
not enter the decision.

Using all original H0 evidence points for the source:

- at least `60%` must lie in the exact HB OBB;
- at least `80%` must lie in the HB OBB expanded by exactly `0.05 m` on every
  local face;
- there must be at least 16 finite evidence points.

Project all eight HB corners through the exact current pose and intrinsic.
Every corner must have camera depth greater than `1e-4 m`.  Take the enclosing
XYXY rectangle, clip it to `[0,640] x [0,480]`, and require 2D IoU at least
`0.50` with the sealed FastSAM `tight_box_xyxy`.

HB must also agree coarsely with Hbase: `ND <= 0.50`, HB/Hbase volume ratio in
`[0.25,4.00]`, and either `IoU3D >= 0.20` or `SC >= 0.70`.  These are safety
gates, not a score; Boxer confidence cannot compensate for a failed geometry
or depth gate.

## Step 3: bounded past-only confirmation

F5 retains only the selected source rows from the previous three successful
scheduled frames, and only while
`current_frame_ordinal - past_frame_ordinal <= 3`.  The buffer stores identity,
frame ordinal, rank and selected geometry; it stores no image feature,
semantic label or learned state.  A source never obtains evidence from its
current frame after the decision or from a future frame.

For each buffered frame, form eligible pairs between the current Hbase AABB
and each past selected AABB when `ND <= 0.50` and either `IoU3D >= 0.15` or
`SC >= 0.60`.  Pair affinity is the exact lexicographic tuple
`(IoU3D, SC, -ND)`.  Keep only mutual-best pairs.  Exact affinity ties prefer
the lower past rank, then the lexicographically smaller past `source_id`;
current ties prefer the lower current rank, then current `source_id`.  This is
observer matching only and never creates an output proposal or persistent
object ID.

HB needs confirming matches from at least two distinct past frames.  Against
each matched past selected AABB, HB must satisfy `ND <= 0.50` and either
`IoU3D >= 0.20` or `SC >= 0.60`.  At least two past matches must pass.  A
repeated source in one past frame counts once.  Insufficient history always
abstains from HB; there is no single-view high-confidence exception.

## Exact selection, ties and abstention

For each source in frozen scene/frame/rank order:

1. compute Hbase by Step 1;
2. if every Step 2 gate and Step 3 confirmation passes, select `HB`;
3. otherwise select Hbase.

Therefore the complete deterministic priority is
`eligible-and-confirmed HB > eligible HLG > eligible HL > H0`.  There is no
weighted score and no tunable equality epsilon.  Threshold equality passes.

`HB_abstention_reason` records the first failed gate in this frozen order:
validity, confidence domain, confidence threshold, point count, exact depth
support, expanded depth support, projection depth, projection IoU, center,
volume, base overlap, history count, past consistency.  HB abstention never
drops a source.  Failure to validate H0 or source lineage is a shard-fatal
integrity error; failure of optional HL/HLG/HB evidence falls back as above.

Each output row seals the source identity, all four input hypothesis hashes,
Hbase, selected hypothesis, selected geometry, every decision scalar, matched
past source IDs/frame ordinals, abstention reason and a canonical result hash.

## Causality and determinism audit

The runner must expose `maximum_lookahead_frames=0`, maximum accessed ordinal,
buffer membership before/after, and zero forbidden reads.  For every scene it
also replays the prefix ending at
`floor(successful_scheduled_frame_count/2)` from an empty buffer; all prefix
row hashes must equal the corresponding full-run prefix hashes.  A separate
synthetic future-perturbation test must leave every earlier row hash unchanged.

Two independent CPU selector replays over the sealed F4 sidecars must produce
the same ordered source/result hash ledger.  Any mismatch aborts F5.  Input
hashes are checked before and after, and all receipts are create-only.

## Runtime and memory gates

Receipt serialization, hashing and the second determinism replay are audit
overhead.  Timed F5 online work includes evidence decode, base checks,
projection, bounded past matching and buffer update.  The first three
non-empty F5 frames per shard are warm-up and excluded only from warm gates;
all-frame measurements remain transparent diagnostics.

### Pre-paper100/no-GT replay timing clarification

The sealed F2 evidence is stored as one compressed audit NPZ per scene, while
the real online pipeline already holds the current source points produced by
F2.  Before paper100 and without opening GT, the replay timer is therefore
clarified as follows: hashing and physically inflating the complete scene NPZ
are sealed-replay I/O overhead and remain outside the incremental timer;
copying, validating and decoding only the current frame's authenticated point
offsets into F5 evidence are inside it.  No whole-scene statistic, future
offset or prefetched source may enter a decision.  This preserves the declared
"evidence decode" work while avoiding an artificial whole-scene disk cost
that cannot exist in the live path.  No selector rule, threshold, source set
or runtime gate changes.

Formal paper100 gates are:

- F5 incremental warm p95 `<=25 ms/keyframe`;
- replay-composed F0+F2+F4+F5 warm p95 `<=375 ms/keyframe`;
- replay-composed warm maximum `<833.33 ms`;
- replay-composed warm mean divided by 25 `<=15 ms/source frame`;
- `gap25_warm_deadline_miss_count == 0`, while the all-frame miss count is
  diagnostic only;
- total CUDA peak `<=4 GiB` (F5 itself allocates no CUDA tensor);
- at most three prior frames and at most 16 source geometries per buffered
  frame; no unbounded history.

Any later active experiment still requires a separately frozen same-GPU live
pipeline measurement of at least 15 FPS.  F5 shadow timing alone is not that
measurement.

## GT-free pass and stopping rule

F5 passes its no-GT shadow only when all of the following are true:

1. the F4 receipt passes and F5 reproduces exactly 100 scenes, 6,817 scheduled
   frames, 6,726 successful frames, 52,299 unique sources and their order;
2. native-output mutation, source addition/removal, score/class/semantic
   mutation, future access, training and online learning counts are zero;
3. source/input/result hashes, prefix causality and the two deterministic
   replays all pass;
4. every selected HB has a complete current-depth, projection and two-past-
   frame proof, with zero rule violations;
5. HB is selected for at least 128 sources spanning at least 20 scenes, but
   for no more than 20% of all sources;
6. every runtime and bounded-memory gate passes.

If any integrity, causality, determinism or runtime gate fails, the result is
`discard_f5_selector`.  If only the minimum HB coverage fails, the result is
`stop_f5_insufficient_confirmed_hb`; thresholds must not be relaxed.  If HB
coverage exceeds 20%, the result is `stop_f5_overbroad_hb`; thresholds must not
be tightened after inspection.  A passing result is
`retain_f5_for_one_separately_sealed_evaluation_only`.

Passing does not authorize birth, terminal-output modification or deployment.
No threshold, tie-break, buffer length, confidence rule or fallback may be
changed after paper100 F5 receipts are observed.
