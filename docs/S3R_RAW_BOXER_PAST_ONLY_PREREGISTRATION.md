# S3R0 raw-Boxer K8 past-only receipt preregistration

Date: 2026-08-23

Status: **frozen after independent review for Stage 0 implementation and one
no-GT dev3 receipt-only shadow after code/test review.  This does not authorize
GT access, AP, birth, H10, C87, or full100.**

## Decision

S3a MobileSAM mask lifting stops.  Its formally sealed primary q02/q98
geometry retained only `+4/+3/+2` native-union matches at IoU
`0.15/0.25/0.50`; the preregistered all-threshold `+3` continuation gate
therefore failed at IoU 0.50.  S3R0 returns to the frozen raw Boxer OBBs and
asks one isolated question:

> Does strictly causal multi-frame association and a first-three-distinct-
> provider-frame receipt preserve the already measured raw-OBB recall ceiling?

S3R0 is a receipt-only, output-inert shadow.  It does not append a box, compute
AP, inspect native predictions during tracking, or perform terminal selection.
It is not a repair of the rejected S0/S1 terminal/depth branches.

The frozen source budget is **K8 per valid provider frame**, not S3a's K4.
The rule remains score-only and deployable: descending frozen source score,
then ascending source row, then ascending sealed NPZ row.  The K8 choice uses
only the already completed K2/K4/K6/K8 ceiling receipt and the user's stated
`+10` absolute-AP target.  No S3R0 output or S3R0/GT overlap existed when this
choice was made.

## Why K8, not K4

The raw ceiling is a necessary recall condition, not an AP prediction.  On the
28 already-open development GT objects, K4 has almost no margin for a selector
that can miss a true object or retain a false positive:

| frozen score-only budget | rows | native-union additions at .15/.25/.50 | recall-point headroom |
|---|---:|---:|---:|
| K2 | 411 | `+2/+2/+2` | `+7.14/+7.14/+7.14` |
| K4 | 814 | `+3/+3/+4` | `+10.71/+10.71/+14.29` |
| K6 | 1,203 | `+5/+4/+5` | `+17.86/+14.29/+17.86` |
| **K8 (S3R0)** | **1,571** | **`+7/+5/+5`** | **`+25.00/+17.86/+17.86`** |

At AP15 and AP25, K4 leaves only 0.71 recall point above the requested
10-point AP gain.  Losing one of the 28 GT matches already costs 3.57 points;
false positives and constant-score ordering can reduce AP further.  K8 retains
substantially more necessary headroom while remaining bounded at eight raw
OBBs per sampled frame.  This does not establish that K8 can deliver `+10` AP.

The formal K8 dev3 membership contains 501/854/216 rows for
`scene0568_00`/`scene0606_01`/`scene0377_02`.  Its selection SHA-256 is
`34ee638d51b3bc137253b3e361a60d84e110e114d2b46c487651550e708aa638`.
K4 remains a historical minimal-compute comparator only; it must not be tried
alongside K8 on H10.

## Development-informed status

This is explicitly a **dev3-informed new branch**:

- dev3 GT was already used by the raw K2/K4/K6/K8 ceiling audit;
- K8 was selected after seeing that ceiling;
- exploratory dev3 structural checks were consulted while defining this
  conservative three-frame association experiment.  Those checks are not a
  sealed trust anchor, and no numerical result from them is used as an S3R0
  pass condition or validation claim.

Consequently, any later dev3 oracle is diagnostic and cannot validate S3R0.
No active counterfactual and no full100 run are allowed before an independently
sealed, one-shot H10 shadow and oracle.  Passing H10 would still authorize only
a new preregistration for a later selector; it would not activate birth.

## Exact frozen trust anchors

### Raw proposal source

- JSON:
  `logs/scannet_boxer_unexplained_shadow_clean_in2_v5_score05/sealed/boxer_shadow_candidates.json`
  (`84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e`)
- NPZ:
  `logs/scannet_boxer_unexplained_shadow_clean_in2_v5_score05/sealed/boxer_shadow_candidates.npz`
  (`c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c`)
- candidate-content SHA-256:
  `8b2362cc11517a58f2a05b371698cf3a45db6805b27c4c1dd10a3c9b899ab529`
- source schema: `boxfusion.owl_boxer_shadow_candidates.v1`
- frozen source profile: OWLv2 Base Patch16 Ensemble with the fixed 1,220
  LVIS+ prompts, `thresh2d=0.25`, frozen BoxerNet, `thresh3d=0.5`, gap 25.

The source is externally pretrained and frozen.  “Training-free” here means
no ScanNet-target fitting, fine-tuning, calibration, optimizer, online
learning, or mutable model state.  It does not mean that OWLv2 or BoxerNet was
never pretrained.

The raw source JSON contains an old Graw native-prediction ledger.  That ledger
is **not trusted** and must never be copied into an S3R0 receipt.  The source is
trusted only for its sealed numeric candidate arrays and hashes.

### Ceiling and stopping receipts

- K2/K4/K6/K8 raw ceiling:
  `logs/scannet_boxer_per_view_topk_raw_ceiling_score05_dev3_v5.json`, SHA-256
  `d4ba67b37d362842333ac525abe32f6807c4fba90af83b699bbfc1494aa5ea1f`.
- formal S3a oracle:
  `logs/scannet_s3a_mobilesam_masklift_oracle_score05_dev3_v2.json`, SHA-256
  `1df78cb8e0a8211949ca58c2c6af475ff421116f2176796bbc77a53191c30309`.
- S3 proposal-source preregistration:
  `docs/S3_FROZEN_PROPOSAL_SOURCE_AUDIT.md`, SHA-256
  `ee742d4b0b9d3e26208ed8b59e587ed6de046ed850a22b80314fd8f939cad191`.

### Native T05 non-interference anchor

The tracker receives no native prediction path or native box.  An outer
read-only sealer may hash the formal T05 files before and after the shadow run
without deserializing them.  For dev3 the only valid root is
`results/scannet_topk_fusion_score05`, with:

| scene | expected SHA-256 |
|---|---|
| `scene0568_00` | `b55ce48fb6eb4dad9ee5bfe7007c3dbc9898b3f72ddbc5ad428b8be6414bcd2d` |
| `scene0606_01` | `d4e8d6dc85c917ac1634b81a45adb3866279d3e02f470c43b23bd71f5bb3ef1c` |
| `scene0377_02` | `ed7f849a33d45eebe846559a90aeb7de1a97f2eb169c3a7c0cb5de61d3dab35b` |

The outer ledger proves byte identity only.  Native geometry, scores, row
order, labels, CLIP vocabulary, CLIP embeddings, and fusion history are not
inputs to S3R0.

## Read-only implementation audit

The audit was completed before this document was written:

| component | SHA-256 | S3R0 decision |
|---|---|---|
| `boxfusion/boxer_past3_receipt.py` | `635c32eaaf61be461fc9a1e570f0e0f28b877737678af35295327515062410e6` | Reuse immutable observations, query-before-commit transactions, one observation per track/frame, TTL accounting, bounded histories and medoid/tie patterns. Do **not** reuse its old within-frame deduplication or stability/extent admission gates. |
| `boxfusion/observer_track_registry.py` | `346d207d27211fcde0f66674857f209ad6df43f8749c524271434297ecab60e7` | Reuse only fail-closed bounds and transaction-audit ideas. It mirrors native BoxFusion row lifecycle and is therefore the wrong identity engine for independent raw Boxer OBBs. |
| `boxfusion/observer_track_adapter.py` | `181ec0b425da83ab1fec02c5fc0f0e9d85a71b9153b9f7b3b39e73ace2364798` | Do not attach it to native BoxManager. Its output-inert digest patterns may be copied into an outer sealer. |
| `tools/materialize_boxer_past3_shadow.py` | `30372efdebadd75bebdd1ded075f9b55deeff2ac6d63c1f737c540464dc8234f` | Do not reuse terminal `close(native...)`, novelty, view gate, NMS or six-box cap. Reuse sealed-schedule replay and create-only serialization patterns only. |
| `tools/materialize_boxer_past3_depth_shadow.py` | `c90250afc2d56e81f8ef2c23a3b65bde69303f54b5a633e98ee46a05a3fb4874` | Do not use depth, mask, graph or terminal filters. Reuse only strict sequential-access assertions if useful. |

No audited component is silently declared S3R0-ready.  A dedicated adapter and
tests must implement the exact deltas below and must be hash-sealed before a
formal run.

## S3R0 data boundary

The no-GT tracker may decode only these numeric source fields:

- scene index and frame ID;
- sealed NPZ row and source row/instance ID;
- frozen source score;
- `center_world`, `extent_xyz`, and `quaternion_wxyz`.

It reconstructs the exact eight world-space OBB corners from those three
geometry arrays in float64.  The quaternion is Hamilton `(w,x,y,z)`.  For
squared norm `n=q dot q`, the standard Hamilton rotation matrix uses scale
`2/n`; zero, non-finite, or `n<=1e-12` fails closed.  Local corners use this
exact sign order:

```text
(-1,-1,-1), (-1,-1,+1), (-1,+1,-1), (-1,+1,+1),
(+1,-1,-1), (+1,-1,+1), (+1,+1,-1), (+1,+1,+1)
```

With `local = signs * extent/2`, world corners are
`local @ R.T + center`.  The source arrays remain untouched, and a receipt
copies one resulting raw OBB without averaging or later mutation.  Association
metrics use the world-axis AABB enclosing those corners.

It must not decode or compare source names.  It must not load detector text,
class labels, RGB, depth, masks, point clouds, CLIP data, native BoxFusion
objects, annotations, axis alignment, or any evaluator module.  There is no
label field in its output schema.

Per valid provider frame, membership is selected exactly as
`(-source_score, source_row, sealed_npz_row)[:8]`.  All selected rows are sent
to association; **there is no within-frame deduplication**.  An input with more
than eight selected rows, a selection-hash mismatch, an off-schedule row, or a
duplicate source-row identity fails closed.

## Frozen causal replay

Frames are replayed in the exact valid gap-25 provider schedule.  Empty valid
provider frames execute an empty query and commit so that TTL advances.
Invalid-pose frames are excluded exactly as recorded by the sealed source and
are never replaced by a later frame.  No RGB/depth/pose payload from a future
frame may be opened or enumerated by the tracker.

For the dev3 engineering replay, the schedule root is
`cache/cutr_postfilter_v3/scannet-graw-e2-score05-preflight3-v3-r1` and the
manifest ledger is:

| scene | manifest SHA-256 | valid provider frames | candidate-bearing frames | excluded invalid pose |
|---|---|---:|---:|---|
| `scene0568_00` | `1ee049e9ad8263e8d7c19838a1038445129a1ae7265434f042ea0c438f3ab19a` | 66 | 66 | none |
| `scene0606_01` | `aedfe2f230c252fb9aaad10b678e3264b8855cfe1150f8b36b291d48e5032753` | 112 | 110 | `1325` |
| `scene0377_02` | `9a8c127b09c36140494a8288425d6b23087b5865d3789b295ed55744d6edf80e` | 30 | 30 | none |

Each frame is a two-phase transaction:

1. `query(frame, current_rows)` snapshots only committed tracks whose maximum
   evidence frame is strictly less than `frame`;
2. all current-to-prior pair metrics and assignments are fixed against that
   snapshot;
3. `commit(exact_query_token)` makes current observations available to later
   frames.

New tracks created by current rows are not inserted into the current query's
association matrix.  Therefore same-frame proposals cannot confirm one
another.  At most one current observation can update a track, and each current
observation can update at most one track.

## Frozen association, confirmation and retirement policy

The S3R0 primary association is deliberately the conservative audited
Boxer-Past3 geometry rule.  This branch is dev3-informed, so H10 remains
mandatory regardless of the dev3 result.

For a current observation AABB `O` and the last committed observation AABB
`T` of a track, define:

- `IoU = volume(O intersection T) / volume(O union T)`;
- `center_m = L2(center(O), center(T))`.

The pair is eligible if and only if:

```text
IoU >= 0.10 AND center_m <= 0.50
```

There is no containment branch in the S3R0 primary.  The broad S3 containment
diagnostic is not exported as a second tunable association universe.

Current rows are processed in exact K8 source order
`(-source_score, source_row, sealed_npz_row)`.  Among unused eligible prior
tracks, one row chooses the track with the lexicographically smallest key:

```text
(-IoU, center_m, track_id)
```

An unmatched row creates the next monotonically increasing track ID.  Frozen
source score is used only for K8 selection and current-row precedence; it is
not an association threshold, confirmation score, learned calibration, or
output confidence.

The remaining fixed policy is:

| item | fixed value |
|---|---:|
| minimum distinct evidence frames | 3 |
| receipt time | immediately on the first third distinct-provider-frame observation |
| active-track TTL | 10 valid provider keyframes |
| expiry | before association on the 11th missed valid provider keyframe |
| active tracks | hard maximum 1,024 |
| immutable receipts | hard maximum 1,024 |
| immutable receipt evidence | exactly the first 3 observations |
| mutable post-receipt association state | one last committed observation plus TTL only |
| observations per input frame | exactly the frozen K8 membership, at most 8 |
| one observation per track per frame | required |

There is **no** median-IoU stability threshold, center-RMS threshold, minimum
extent threshold, camera-baseline/view-angle gate, depth gate, native novelty
gate, side NMS, score ceiling, semantic supporter, or terminal Top-N output
cap.  The 1,024 limits are safety-memory bounds, not candidate-selection
rules.  Reaching a safety cap invalidates the complete scene audit; it must not
silently truncate a supposedly complete receipt universe.

The immutable receipt contains exactly the first three associated distinct
provider frames.  Before confirmation, the live track holds at most those
three observations.  After confirmation, later matches replace only one
`last_association_observation` and advance TTL.  They are not appended to a
rolling evidence FIFO and cannot change receipt membership, confirmation time,
primary geometry, provenance, or any exported receipt metric.

An expired track is never re-identified or merged across its TTL boundary.  A
later unmatched observation may create a new monotonically increasing track ID;
S3R0 performs no cross-TTL NMS or receipt merge.

## Receipt geometry and diagnostic evidence ceiling

The primary receipt geometry is one unmodified raw Boxer OBB from its three
evidence rows.  Compute all pairwise enclosing-AABB IoUs, select the IoU medoid
that minimizes `sum(1 - IoU)`, and break an exact tie by
`(frame_id, source_row, sealed_npz_row)`.  Do not average centers, extents,
quaternions, corners, or scores.

Every receipt must also export all three evidence OBBs and identities.  A
separate post-hoc oracle may report two fixed geometries:

1. **primary medoid**: one deployable raw OBB per receipt;
2. **track-any-evidence ceiling**: for each receipt/GT pair, the maximum IoU
   among that receipt's three evidence OBBs, followed by maximum-cardinality
   matching over receipt IDs and GT IDs.

The second geometry is oracle-only.  It may diagnose whether medoid selection,
rather than association, loses ceiling, but it may never choose a deployable
box, alter a receipt, or enter AP evaluation.

## Bounded state and realtime budget

The online state is bounded independently of scene length:

- at most 8 current observations;
- at most 1,024 active tracks and 1,024 immutable receipts;
- exactly 3 immutable raw OBB observations per confirmed receipt, plus one
  last-association observation per live track;
- at most 8,192 current-to-track eligibility checks per provider frame;
- at most 4,096 valid provider frames and 32,768 selected rows per scene;
- trace records are streamed to a create-only sidecar, with a hard 32 MiB
  uncompressed diagnostic cap rather than retained indefinitely in RAM.

Any bound violation sets `audit_complete=false`, writes no valid seal, and
cannot be interpreted as a negative scientific result.  If a failure receipt
is retained, it uses a separate failure schema and cannot be consumed by the
S3R0 oracle.

The preregistered tracker-only CPU budget is p95 at most 2.0 ms and maximum at
most 10.0 ms per valid provider frame, including empty transactions, after
cold initialization.  Incremental tracker memory must be at most 64 MiB and
tracker GPU allocation must be zero.  The earlier all-row Past3 observer
measured at most 1.08 ms p95, but that historical receipt is only feasibility
evidence, not an S3R0 measurement.

The full frozen OWLv2+Boxer+S3R0 provider must also be measured on the same GPU
as native T05 before any promotion discussion.  After warm-up, it must have no
queue backlog or offline catch-up, must finish each gap-25 provider call before
the next 833.33 ms deadline, and must retain native online throughput of at
least 10 FPS.  Cold-start time, provider p50/p95/max, peak CPU/GPU memory and
same-GPU native-throughput delta are reported separately.  A precomputed
sidecar replay cannot satisfy this integrated runtime gate by itself.

## Create-only shadow output

Proposed schema: `boxfusion.s3r_raw_boxer_past_only_shadow.v1`.

The JSON seal must state at least:

```text
mode=shadow
output_inert=true
birth=false
active_authorized=false
native_mutation_applied=false
ap_evaluation=false
gt_access=false
semantics_access=false
clip_access=false
depth_access=false
training=false
online_learning=false
past_only=true
same_frame_confirmation=false
H10_not_authorized=true       # dev3 engineering artifact
full100_not_authorized=true
```

It binds this preregistration hash, source JSON/NPZ/content hashes, K8 receipt
and K8 selection hashes, schedule hashes, tracker/adapter/test hashes, all caps,
runtime statistics, every cap/rejection count, and input hashes before/after.
If an outer formal-T05 identity ledger is present, it records the formal root
and exact before/after hashes and explicitly records that the source JSON's
Graw ledger was not trusted.

The NPZ sidecar must contain, with fixed dtypes and shapes documented by the
implementation tests:

- scene IDs and complete selected-K8 row identities/order;
- per-frame assignments, creations, retirements and cap events;
- receipt scene/track/confirmation IDs and primary raw OBB corners;
- ragged evidence offsets plus frame ID, source row, source instance ID,
  sealed NPZ row, source score and raw OBB corners for every receipt member;
- primary medoid index and deterministic association metrics;
- a content SHA-256 independent of ZIP timestamps.

There are no label, class, CLIP, native-overlap, appended-score, AP, GT-match,
or terminal-selection arrays.  Output is create-only.  The final output root
must not already exist and must not be a symlink.  The runner writes JSON and
NPZ into a same-filesystem private staging directory, fsyncs both files and the
staging directory, verifies both hashes and the content hash, then atomically
renames the complete staging directory to the final root and fsyncs its parent.
A failure may retain an explicitly invalid staging/failure receipt for
debugging, but it must never publish a valid-schema partial pair.  An existing
root, symlink, hash mismatch, cap event, or changed input fails closed.

## Staged evaluation and stopping rules

### Stage 0: implementation review

Implement the exact receipt-only delta, add synthetic tests for query/commit,
same-frame isolation, empty-frame TTL, K8 order, one-to-one assignment, first
third-distinct-provider-frame freezing, medoid ties, capacity failure and input
immutability.  An independent reviewer must hash the code/tests and verify that GT/evaluation,
depth, semantics, CLIP and native BoxManager cannot be imported by the runner.

### Stage 1: dev3 engineering shadow and diagnostic oracle

Run a new no-GT, create-only dev3 shadow only after Stage 0.  Seal it before a
separate oracle opens the already-used dev3 GT.  The oracle reports no AP.  It
loads ScanNet `axisAlignment`, transforms every raw OBB corner into the aligned
frame, and then takes the enclosing aligned AABB.  GT boxes use the same
aligned coordinate convention.  IoU eligibility is strict `IoU > threshold`
at `0.15/0.25/0.50`.

For primary-medoid evaluation, matrix entry `(receipt, GT)` is the aligned-AABB
IoU of the receipt's one frozen medoid OBB.  For track-any-evidence diagnosis,
entry `(receipt, GT)` is the maximum of the three aligned-AABB IoUs from that
same receipt; maximum-cardinality matching is then performed over receipt IDs
and GT IDs.  The oracle must report raw K8, both confirmed matrices, and their
native-union additional matches.  It cannot expose the evidence argmax as a
deployable geometry.

S3R0 stops before H10 unless **both** confirmed geometries retain at least
three additional native-union matches at every threshold:

```text
track-any-evidence >= +3/+3/+3
primary-medoid     >= +3/+3/+3
```

The any-evidence check diagnoses association/confirmation loss; the medoid
check diagnoses deployable single-geometry loss.  There is no post-hoc row,
track, threshold or geometry selection.

### Stage 2: one-shot independent H10

Even a dev3 pass is not validation because S3R0 is dev3-informed.  Before any
H10 inference, freeze and hash:

- this document and the reviewed runner/tests;
- the fixed list
  `evaluation/data_util/meta_data/scannetv2_boxer_past3_s1_holdout10.txt`,
  SHA-256
  `8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90`;
- a freshly generated no-GT frozen Boxer source contract with
  `annotation_path=None` and `track=false`; it must neither create nor read
  `boxer_3dbbs_tracked.csv`;
- the score-only K8 rule and the resulting H10 membership hash;
- H10 formal-T05 before hashes and sealed schedule hashes.

The old preliminary S1 H10 pool must not be reused: its runner enumerated
future pose metadata and ran an unused inline tracker.  S3R0 H10 must iterate
only the exact valid frame IDs already present in each sealed manifest, never
enumerate a directory through `skip_n`/`max_n`, and finish and persist one
frame's output before advancing to the next schedule ID.

Run and seal the H10 shadow with no H10 annotations mounted or accepted.  Only
after sealing may a separately authorized, read-only oracle open H10 GT.  It
reports the same matching-only metrics, not AP.  With `N` H10 GT objects, the
necessary continuation count is `ceil(0.10*N)`.  Raw K8,
track-any-evidence, and primary medoid must each add at least that count over
native at all three IoUs, and every causality/hash/cap/runtime invariant must
pass.

Failure of any threshold or invariant rejects S3R0.  Passing H10 does **not**
authorize birth, suffix evaluation, AP computation, C87, or full100.  It only
permits writing a separate preregistration for a no-GT novelty/precision
selector.  That later selector must be frozen without H10 tuning and validated
on untouched scenes before any active or full100 claim.

## Explicitly forbidden transitions

Until the independent H10 sequence above passes and a new plan is approved:

- do not append a receipt to native T05;
- do not assign an output confidence or low-score suffix;
- do not compute or report S3R0 AP;
- do not suppress against native boxes or CLIP;
- do not select a receipt using GT, class names, embeddings, depth, future
  frames, all-scene statistics, or post-hoc best geometry;
- do not tune K, association gates, min views, TTL, medoid, or any cap;
- do not access H10 GT, run an active counterfactual, run C87, or run full100.

## Interpretation guardrail

K8 raw matching proves only that useful OBBs exist in a bounded frozen pool.
An exploratory three-distinct-provider-frame structural diagnostic proves only
a possible receipt ceiling.
Neither proves precision, AP, generalization, or realtime integration.  S3R0
is designed to separate association/confirmation loss from the old S0/S1
depth, novelty, NMS and terminal-cap losses; it cannot by itself meet the
user's `+10` absolute-AP goal.
