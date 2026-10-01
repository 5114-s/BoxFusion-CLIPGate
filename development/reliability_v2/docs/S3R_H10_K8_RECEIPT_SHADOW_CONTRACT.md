# S3R H10 sealed-K8 receipt-shadow contract

Date: 2026-08-24

Status: **implementation-frozen, pre-replay, no-GT shadow only.**  These exact
bytes authorize one create-only replay of the already sealed numeric H10
source.  They do not authorize H10 ground truth, an oracle, AP, birth, native
BoxFusion mutation, C87, or full100.  No currently authorized receipt-shadow
root exists at the time this revised contract is sealed; the only planned
fresh root is the `v3` root frozen below.

One superseded `v1` invocation stopped during pre-publication numeric-source
validation because the runner/contract had miscopied the frozen array-content
hash as `a5efdb8d6f824b5006f79ee09333f89f0be5d9eb24fb9b4aa12813910382b862`.
The source manifest's actual hash is the value frozen below.  The tracker
replay and publication were never entered, and
`logs/scannet_s3r_h10_k8_receipt_shadow_score05_v1` is absent.  The `v1`
output name and the superseded contract bytes are permanently invalid; they
must not be resumed or recreated.

A later `v2` invocation completed the full no-GT replay and create-only
publication successfully.  Its runner/content/cap/runtime checks passed; it
is not a numerical or scientific failure.  It bound runner SHA-256
`aaafd5c3cf1305309eecfc76a56f30dd8977601c0b0966e442581a540c8fb27b`,
then-current focused-test SHA-256
`752dbdaea76fbce3b2258f0a24e58d56c2f11305c0efdef91003183dcfb14ff1`,
and then-current contract SHA-256
`84c1862b9a3f6803e0e92fd99600898bfa6e676c96e628c8441985c296b145bd`.
After publication, the focused test was tightened so the formal numeric-source
preflight receives no autouse monkeypatch fixture, producing the current test
and contract bytes below.  The old `752d...`/`84c...` bytes are no longer
present at their bound workspace paths, so current-path provenance cannot
reproduce v2 exactly.  Consequently
`logs/scannet_s3r_h10_k8_receipt_shadow_score05_v2` is retained unchanged but
is superseded and cannot be promoted, resumed, resealed, deleted, or
overwritten.

## Isolated question

This stage asks only whether the frozen K8 raw-Boxer rows form causal,
three-distinct-frame receipt tracks under the already preregistered S3R
association rule.  It produces a complete diagnostic assignment/receipt
trace.  It does not estimate accuracy or add a proposal to T05.

The runner may consume only the sealed numeric source JSON/NPZ pair.  It must
not open ScanNet annotations, labels, GT boxes, an evaluator, CLIP/semantic
artifacts, RGB/depth/pose files, or native T05 predictions.  It must not use
pickle or any other native-prediction deserializer.  It performs no training,
calibration, optimization, online learning, AP computation, or birth.

## Exact frozen H10 source

The only source root is:

`logs/scannet_s3r_h10_raw_boxer_source_score05_v1`

| identity | frozen value |
|---|---|
| source schema | `boxfusion.s3r_h10_raw_boxer_source.v1` |
| `S3R_H10_RAW_BOXER_SOURCE.json` SHA-256 | `ca65214f3e6327cea66ec8cb700ab3501572be9325af4366beaffa2b7cc2859e` |
| `S3R_H10_RAW_BOXER_SOURCE.npz` SHA-256 | `fdb688cc1372985f2ffaf3d257ed470cd4de28ff42f7a2d04a5f72311a1225f2` |
| numeric array-content SHA-256 | `a5efdb8d0d2c7b95f63368a3249229659a1052c400539321ce461da32732b862` |
| K8 membership SHA-256 | `a2a94b11461e8c1bdd15d6a4ad99d058f42db6fd73690c69269ff1b89deb6391` |
| K8 membership count | `4557` |
| exact valid-frame count | `769` |
| raw schedule count | `770` |
| scene count | `10` |

The exact scene order is:

1. `scene0304_00`
2. `scene0412_00`
3. `scene0019_00`
4. `scene0575_00`
5. `scene0426_00`
6. `scene0426_03`
7. `scene0578_00`
8. `scene0665_00`
9. `scene0050_01`
10. `scene0025_00`

The underlying exact schedule is
`docs/data/S3R_H10_EXACT_SCHEDULE_V2.json`, SHA-256
`1ce565a65510b80d69a0402fe7a40ea89920625f6a81147d42f9232f7a7761e9`.
The 769-frame ledger includes every valid empty-proposal frame.  The runner
does not enumerate a dataset or reconstruct a schedule from filenames.

The supplied K8 identity matrix is the experiment membership.  The runner
independently recomputes the sealer's deterministic ordering only to reject a
mismatch; it does not create a substitute membership.  Each identity has the
columns `(scene_index, frame_id, source_row, source_instance_id,
sealed_npz_row)`.  The frozen order is
`(-source_score, source_row, sealed_npz_row)[:8]` within each exact frame.

## Exact implementation bytes

| file | SHA-256 |
|---|---|
| `docs/S3R_RAW_BOXER_PAST_ONLY_PREREGISTRATION.md` | `14f29a50dd65ee791be2df519e0000cf22bfc94a0209880f3539159acf4f7df3` |
| `tools/seal_scannet_s3r_h10_raw_boxer_source.py` | `46642862d78ebc10f88e23b869607e4d0fbd3f61f9644fe0df7122983dc7fea7` |
| `tests/test_seal_scannet_s3r_h10_raw_boxer_source.py` | `e395cf820d6ffa3a9dd607d23c5286d1f9939b3ae98b63c2fa9a4f52acc1ffaf` |
| `boxfusion/s3r_receipt_tracker.py` | `277316c36b7a7fcb8005a24e907e0f232e41f6b5874411293eb26b0744df9628` |
| `tests/test_s3r_receipt_tracker.py` | `f08fd59ee2888c936e5b783de668fd789ba6b676bc4864e001b000ea287b1e3c` |
| `tools/run_scannet_s3r_h10_k8_receipt_shadow.py` | `aaafd5c3cf1305309eecfc76a56f30dd8977601c0b0966e442581a540c8fb27b` |
| `tests/test_run_scannet_s3r_h10_k8_receipt_shadow.py` | `a730f82e58b4418c56ab8c3e844ab33d8792a73703662f625832ee43f8dd00ec` |

The source sealer/test and tracker/test hashes are hard-coded and verified by
the runner.  To avoid a self-hash cycle, the runner and focused-test hashes in
this contract are passed explicitly on the command line.  They are checked
before source replay and again in the before/after input snapshots.  The
contract itself is opaque to the runner: its exact externally computed hash
is passed separately, checked before replay, rechecked afterward, and written
into the output seal.

## Causal tracker semantics

The exact replay order is scene order, then the sealed valid-frame order,
including empty frames.  A fresh `S3RReceiptTracker` is created once per
scene.  For each frame the runner must:

1. construct at most eight numeric observations in the frozen membership
   order;
2. call `query(frame_id, observations)` against committed history only;
3. call `commit(query)` on that exact pending query exactly once;
4. verify the complete supplied K8 order, assignment provenance, audit state,
   and every cap/rejection ledger, failing without publication on any
   mismatch;
5. only after the exact commit and verification advance to the next frame.

Current-frame rows cannot match or confirm one another.  A receipt freezes at
the first third distinct, causally committed provider frame.  Its evidence
contains exactly three distinct increasing frame IDs; the third is its
confirmation frame.  On every later frame the entire frozen receipt remains
immutable.  The exported receipt geometry is the AABB-IoU medoid of those
first three raw OBBs.  Association uses world-axis enclosing-AABB IoU at least
`0.10` and center distance at most `0.50 m`; active-track TTL is ten valid
keyframes.  No within-frame deduplication is performed.

The phrase “past-only” refers to query-before-commit association.  When a
downstream stage sees a sealed receipt, all three evidence frames are already
historical.  This contract does not authorize any downstream stage.

## Bounded state and runtime gates

Frozen safety bounds are:

- at most 8 observations per valid frame;
- at most 1,024 live tracks and 1,024 immutable receipts per scene;
- exactly 3 immutable evidence OBBs per receipt;
- at most 8,192 current-to-prior-track eligibility checks per frame;
- at most 4,096 valid frames and 32,768 selected rows per scene;
- at most 32 MiB of uncompressed exported diagnostic arrays;
- at most 64 MiB sampled incremental process RSS during tracker replay.

Any cap, incomplete audit, input mutation, non-finite geometry, malformed
identity, symlink/path identity failure, resource-gate failure, or fsync/
publication failure prevents a valid output seal.  It is an implementation
failure, not a negative scientific result.

The frozen tracker-only CPU gate is p95 at most `2.0 ms` and maximum at most
`10.0 ms` per valid frame, including empty transactions after cold tracker
initialization.  Tracker initialization, tracker CPU/wall, adapter CPU/wall,
and sampled RSS are all recorded.  Adapter timing is diagnostic only and has
no separate acceptance threshold in this contract; the preregistered
acceptance threshold remains tracker-only.

The tracker and adapter are NumPy/CPU-only.  A static AST import audit rejects
CUDA-capable runtime imports in the runner or tracker.  The seal records
tracker execution device `cpu`, GPU execution/API access `false`, and tracker
GPU allocation `0` by construction.  It explicitly records that no GPU-memory
measurement is claimed.

This is replay of a precomputed sealed sidecar.  Therefore
`integrated_provider_runtime_qualified=false` and
`native_online_fps_claimed=false` must remain in the seal.  This stage cannot
satisfy the independent same-GPU integrated runtime gate: the full frozen
provider plus tracker must still meet every warm 833.33 ms provider deadline
without backlog and preserve at least 10 FPS native throughput.  Until that
separate gate is sealed and independently reviewed, H10 GT and every H10
oracle remain forbidden.

## Exact environment and invocation

The formal process environment is:

```text
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONHASHSEED=0
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
```

The runner directly enforces the four numeric thread values.  The launch
receipt must preserve all seven values above.  Let `CONTRACT_SHA256` be the
externally computed SHA-256 of this exact file.  The only authorized command
is logically equivalent to:

```bash
python tools/run_scannet_s3r_h10_k8_receipt_shadow.py \
  --source-root logs/scannet_s3r_h10_raw_boxer_source_score05_v1 \
  --receipt-contract docs/S3R_H10_K8_RECEIPT_SHADOW_CONTRACT.md \
  --expected-receipt-contract-sha256 CONTRACT_SHA256 \
  --expected-runner-sha256 aaafd5c3cf1305309eecfc76a56f30dd8977601c0b0966e442581a540c8fb27b \
  --expected-runner-test-sha256 a730f82e58b4418c56ab8c3e844ab33d8792a73703662f625832ee43f8dd00ec \
  --output-root logs/scannet_s3r_h10_k8_receipt_shadow_score05_v3
```

No shell redirection into an existing output is allowed.  The planned output
root must not exist before launch and must not overlap the sealed source.

## Create-only output

The output schema is `boxfusion.s3r_h10_k8_receipt_shadow.v1`.  A valid root
contains exactly:

- `S3R_H10_K8_RECEIPT_SHADOW.json`
- `S3R_H10_K8_RECEIPT_SHADOW.npz`

The output JSON binds source, membership, contract, runner/test, tracker/test,
runtime, caps, complete before/after input identities, deterministic NPZ byte
hash, and numeric trace-content hash.  The NPZ contains all 769 frame
transactions, exact selected K8 identities/order, assignments, retirements,
cap counters, receipt geometry/metrics, and all three evidence identities and
OBBs per receipt.

Publication holds the output-parent and staging directory descriptors, binds
their device/inode identities, verifies the staging and published directories
contain exactly the two allowed files, and uses
`renameat2(RENAME_NOREPLACE)` relative to the held parent descriptor.  Parent,
staging, or published-name replacement fails closed.  Cleanup deletes only
the two runner-created files when the still-named staging inode equals the
held staging inode; it never recursively follows or deletes a replacement.

## Mandatory stopping rule

A valid seal must state at least:

```text
mode=shadow
output_inert=true
receipt_only=true
birth=false
active_authorized=false
native_mutation_applied=false
ap_evaluation=false
gt_access=false
gt_access_authorized=false
oracle_access=false
H10_oracle_not_run=true
H10_oracle_authorized=false
semantics_access=false
clip_access=false
native_prediction_access=false
pickle_deserialization=false
training=false
online_learning=false
past_only=true
query_before_commit=true
same_frame_confirmation=false
within_frame_deduplication=false
C87_not_authorized=true
full100_not_authorized=true
```

After a valid no-GT receipt shadow, stop.  Do not open H10 GT, run a matching
oracle, compute AP, enable birth, append boxes, alter T05, enter C87/full100,
or claim deployable/integrated online performance.  A separate, newly frozen
runtime-gate receipt and explicit independent authorization are required
before even a no-AP H10 oracle may be designed or run.
