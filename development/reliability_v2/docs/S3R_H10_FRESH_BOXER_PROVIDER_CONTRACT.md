# S3R Stage-2 H10 fresh frozen-Boxer provider contract

Date: 2026-08-24

Status: **pre-inference frozen candidate, subject to final independent code/hash
review.  It authorizes exactly one create-only, no-GT H10 raw-provider shadow
after that review.  It does not authorize H10 GT, matching, AP, birth, native
mutation, C87, or full100.**

## Question isolated by this run

This run asks only whether the frozen OWLv2 + Boxer proposal model can produce
a fresh, strictly scheduled, numeric raw-OBB pool on the independent H10
scenes while preserving the online and no-target-training constraints.  It is
not an accuracy result.  Association, K8 receipt tracking, matching-only
oracle work, and any precision selector are later create-only stages.

“Training-free” means no ScanNet fitting, fine-tuning, calibration, optimizer,
online learning, or mutable target-scene model state.  The externally
pretrained OWLv2, DINOv3, and BoxerNet weights remain frozen.

## Superseded preliminary H10 path

The old `BOXER_PAST3_S1_H10_PROPOSAL_CONTRACT.md`, its runner, and
`logs/scannet_boxer_past3_s1_h10_v1_score05` are forbidden inputs.  That path
enumerated all future color/pose metadata through `skip_n/max_n`, used
background prefetch, enabled Boxer's unused inline tracker, did not durably
persist empty frames, and could create `boxer_3dbbs_tracked.csv`.

The first generated exact-schedule file
`docs/data/S3R_H10_EXACT_SCHEDULE_V1.json` (SHA-256
`45c1ab41e732d1e66c43d244ec976dfd2d96c5a64420a6849834b0063ff41b67`)
is also rejected before inference because one source-manifest field used an
absolute path.  It is retained only as an invalid preparation receipt and
must never be supplied to the provider.

## Frozen H10 schedule

The only valid schedule is
`docs/data/S3R_H10_EXACT_SCHEDULE_V2.json`, SHA-256
`1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9`.
It binds:

- the fixed H10 list, SHA-256
  `8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90`;
- ten scenes in the exact listed order;
- 770 raw gap-25 schedule IDs and 769 valid IDs;
- the sole exclusion `scene0412_00/2325`, whose non-finite pose SHA-256 is
  `8981acaa5e7d946d6031737ac0d55d4fe29ceda0a7fb10241ad4bae2e84bf467`;
- exact relative RGB, depth, pose, and intrinsic paths and hashes for every
  valid frame;
- the ten original schedule-manifest hashes and the independent formal-T05
  opaque hashes.

Frame 2325 is omitted without replacement.  The provider must not call
`listdir`, `scandir`, `glob`, `iterdir`, `skip_n`, or `max_n` to reconstruct a
schedule.  It may open only the current manifest-named frame payload.  All
2,317 named frame/intrinsic files are re-hashed after the stream, when they are
all in the past.

Frozen schedule implementation:

| file | SHA-256 |
|---|---|
| `tools/build_scannet_s3r_h10_exact_schedule.py` | `55edff91021329311e2f0920f5f846004aa53049a45906e7544740f1a5ea4ee0` |
| `tests/test_build_scannet_s3r_h10_exact_schedule.py` | `1306c3490801b1cc94feed2c25c42c9e4dc511b4e9221cf912e02b82d79e3efd` |

The schedule was built without annotations, GT boxes, axis alignment, an
evaluator, native prediction deserialization, or model inference.

## Frozen provider implementation

| file | SHA-256 |
|---|---|
| `boxfusion/s3r_h10_provider_core.py` | `c70e114dabe1ef1081967027e4b5a15955ac16bab745652984dfe981100f21dd` |
| `tests/test_s3r_h10_provider_core.py` | `40f75dac98e5774e9b1637a7c51c4ab5676df38a074e3c3b97a0d3a40a305ce2` |
| `tools/run_scannet_s3r_h10_fresh_boxer_provider.py` | `72e42f3a3865ee9f52687d2a5a5a40ecabe189864c4d7d2cce18daf6be056403` |
| `tests/test_run_scannet_s3r_h10_fresh_boxer_provider.py` | `89595cf544e60efdde5637f7315f42ce8d59b3a0088d50d7913c3a442d000a6e` |
| `docs/S3R_RAW_BOXER_PAST_ONLY_PREREGISTRATION.md` | `14f29a50dd65ee791be2df519e0000cf22bfc94a0209880f3539159acf4f7df3` |

The runner does not import or instantiate Boxer's `ScanNetLoader`,
`BaseLoader`, `run_boxer.main`, or tracker.  Its synchronous reader constructs
one current-frame datum directly.  OWLv2 and BoxerNet are each instantiated
once for the complete ten-scene process.  A scene reset changes only fixed
random seeds; no model parameter or online learned state changes.

The transaction order is:

1. authorize exactly the next `(scene, frame)` token;
2. hash and open only that frame's RGB/depth/pose;
3. run frozen OWLv2 then frozen BoxerNet;
4. publish a numeric per-frame NPZ using create-only no-replace semantics;
5. `fsync` the frame file and frame directory;
6. append and `fsync` the frame journal;
7. only then authorize the next scheduled frame.

An empty proposal frame follows the same durable transaction.  A pending,
duplicate, missing, off-order, out-of-schedule, non-finite, over-cap, symlink,
existing-output, hard-link race, output-root or output-parent name-swap race,
or fsync failure poisons the run and prevents a final seal.  The transaction
holds both the output parent and its grandparent and continuously verifies the
parent entry's inode.  The raw cap is 2,048 rows per frame; it is a safety
bound, not a selection rule.

## Frozen model and inference profile

External Boxer repository commit:
`1f86542dc342a4b1d474c87c97c5d1d6566d9148`, clean worktree.

| asset | SHA-256 |
|---|---|
| OWLv2 Base Patch16 Ensemble | `14aa78ffe7b13e5b3ebf55845bc9a07e339a095cfd88f4c4e8f726b38ce1ebbf` |
| frozen 1,220-prompt text cache | `59193fc014d381b2200edf1c1e6dc86324edb55a067189d3e84226a184185283` |
| BoxerNet | `d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f` |
| DINOv3 backbone | `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea` |

Fixed inference values:

- taxonomy: the frozen 1,220 LVIS+ prompts;
- image size: `960 x 960`;
- OWLv2 threshold: `0.25`;
- per-class 2D NMS IoU: `0.50`;
- Boxer 3D threshold: `0.50`;
- retained raw score: `mean(OWL_2D_score, Boxer_3D_score)` after the 3D gate;
- precision: bfloat16;
- seed: 0 at each scene boundary;
- `annotation_path=None`, `track=false`, no fusion, no BoxFusion downstream
  CLIP semantic classifier/deserialization/export, and no native prediction
  deserialization.  OWLv2's internal frozen CLIP tokenizer, text encoder, and
  text embeddings are part of the frozen provider and are explicitly allowed.

The emitted geometry uses absolute ScanNet world coordinates:

```text
center_world = center_boxer_recentered
             + translation_of_first_valid_exact_schedule_pose
```

Extent is unchanged.  Hamilton quaternion `(w,x,y,z)` is normalized per row.
No name, label, class, text, CLIP embedding, or semantic ID is exported.

## Native non-interference and score protocol

The formal native prefix is the already completed T05 root
`results/scannet_topk_fusion_score05`: native `score_thresh=0.5`, appearance
gate disabled, Reliable-View Top-K3.  The runner hashes each of the ten formal
prediction files before and after the provider stream but never unpickles
them.  Any byte change invalidates the run.

This provider does not append predictions or assign the eventual evaluation
score.  If a later active suffix is separately authorized, the formal native
evaluation remains constant score `1.0`; this provider run itself performs no
evaluation.

## Runtime and bounded-state receipt

The formal environment must pin these values before Python starts:

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
```

The runner records model cold-start time, the first real frame separately,
all-frame and warm-frame end-to-end p50/p95/max, peak CUDA allocated/reserved
memory, and peak process RSS.  End-to-end time includes current-frame reads,
datum construction, OWLv2, BoxerNet, CUDA synchronization, per-frame NPZ
fsync, directory fsync, and journal fsync.  The warm gap-25 deadline is
833.33 ms per provider call with no backlog.

This standalone shadow cannot by itself claim the complete integrated runtime
condition.  Its provenance must keep `integrated_realtime_qualified=false`
until a separately bound same-GPU native-throughput check shows at least
10 FPS with the provider active.  Failure of either runtime condition blocks
H10 GT/oracle promotion; it is an implementation/runtime failure, not an
accuracy result.

## Create-only output and binding

The only planned first formal output root is:

`logs/scannet_s3r_h10_fresh_boxer_provider_score05_v1`

It must not exist before launch.  The runner requires this contract path and
an externally computed SHA-256 of these exact bytes as command arguments.  It
hashes the contract before model construction, records it in
`RUN_PROVENANCE.json`, re-hashes it after the stream, and binds that provenance
hash into `FINAL_SEAL.json`.

A valid root contains exactly 769 committed frame artifacts, a complete
journal, `RUN_PROVENANCE.json`, and `FINAL_SEAL.json`.  A partial root without
the final seal is invalid and cannot be resumed, sealed by hand, or consumed
by the source sealer.  The runner must neither create nor read any file named
`boxer_3dbbs_tracked.csv`.

## Stopping rule after this provider

After a valid provider seal, a separately reviewed numeric source sealer may:

1. flatten all raw rows without semantic fields;
2. derive per-frame `source_row=0..N-1` and a fixed numeric instance identity;
3. freeze the only allowed K8 membership using
   `(-source_score, source_row, sealed_npz_row)[:8]`;
4. run the already frozen three-distinct-past-frame S3R receipt tracker in
   another output-inert shadow.

No H10 GT may be opened until the provider, source, K8 membership, receipt
shadow, input-identity checks, caps, and required runtime checks are all
sealed and independently reviewed.  Even a later H10 matching-only pass would
not authorize AP or birth; it would authorize only a new precision-selector
preregistration.
