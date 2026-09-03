# F2 FastSAM DFU-LGF-lite paper100 protocol freeze

## Scope

F2 is a deterministic, training-free, current-frame-only geometry shadow over
the sealed F0 FastSAM candidate universe. It does not read GT, native terminal
predictions, CLIP, semantics, tracking state, or future frames. It cannot emit
a birth or modify BoxFusion output. A separate, explicitly GT-assisted oracle
may read only the completed F2 receipt.

F2 addresses the failure isolated by F1: F0 already has enough low-IoU
candidate capacity, but the AP50 union adds only 63 matches, below the frozen
144-match requirement. F2 is not assumed to improve this result.

This protocol is frozen before any F2 candidate is compared with ScanNet GT.
No parameter below may be selected or changed using paper100 oracle results.

## Frozen inputs

- paper100 scene order:
  `evaluation/data_util/meta_data/scannetv2_val.txt`, SHA-256
  `4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5`;
- F0 full200 final receipt:
  `logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json`,
  SHA-256
  `07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1`;
- F1 paper100 receipt:
  `reports/fastsam_f1_paper100_oracle/F1_FASTSAM_PAPER100_ORACLE.json`,
  SHA-256
  `05fc3b740126fcc8ac83ac335cf62df85b8ebd99b9033d0fc452e52229105304`;
- frozen F0 core SHA-256
  `a7cf6e3ae4777ee62ca5a1aa9dbc9a38e91cacc8ce77ab15cea940c838686e48`;
- frozen FastSAM provider SHA-256
  `1e48f6676300dead2e77fad2a95be377d7d650980ba2377ab17cbb10c1f69f05`;
- frozen F0 runner SHA-256
  `638eb8670513aa03e3d20dbf47604fb0777a5487c9600143c7d3317ae6d5bf83`.

The checkpoint, RGB, registered depth, intrinsics, current pose, CuTR cache,
software versions, GPU identity, and producer orientation are those sealed by
the F0 receipt. F2 processes exactly the first 100 F0 scenes: 6,817 scheduled
keyframes, 6,726 successful frames, and 52,299 selected candidates.

## F2a deterministic evidence replay

The F0 sidecars intentionally omit point coordinates, voxel keys, and mask
bitmaps. They cannot be reconstructed from q02/q98 or hashes. F2 therefore
replays FastSAM and the unmodified F0 core on paper100.

Before applying F2, every frame must agree with its sealed F0 sidecar:

1. RGB, depth, pose, intrinsics, CuTR cache, and schedule hashes agree;
2. every provider mask agrees in source order on raw index, mask SHA-256,
   confidence, provider box, and F0 disposition;
3. every selected candidate agrees on rank, raw index, mask SHA-256, counts,
   q02/q98 geometry, and the joint points/voxel-keys SHA-256;
4. no extra or missing frame, mask, or candidate is permitted.

Any disagreement aborts the affected shard and publishes no completed F2
receipt. Ordinary invalid-pose/non-upright frames must reproduce the sealed F0
abstention rather than being forward-filled.

The replay may seal per-scene compressed evidence for F3: selected-mask
packbits, bounded point coordinates/voxel keys, frame/source identity, and
their hashes. Serialization is an offline receipt cost and is reported
separately from online inference timing.

## Frozen F2 geometry

F2 consumes the exact bounded F0 candidate points: at most 2,048
lexicographically sampled representatives of the original 2 cm occupied
voxels. A source candidate has one stable identity and three hypotheses:

- `H0`: the sealed F0 q02/q98 AABB;
- `HL`: q02/q98 after the local filter;
- `HLG`: q02/q98 after the local and global filters.

### Local filter

- Euclidean radius: `0.06 m`, exactly three F0 voxels;
- retain a point only when at least three *other* input representatives lie
  within the closed radius;
- use `scipy.spatial.cKDTree.query_pairs(eps=0)` only to enumerate a
  conservative pair superset, then apply the original exact squared-distance
  predicate; a signed-floor fixed-cell implementation is retained only as the
  output-equivalence audit reference;
- ties and boundary comparisons use `distance <= 0.06 m`.

### Global robust filter

For the retained local points:

1. compute the coordinate-wise median `c`;
2. compute radial distances `rho = ||p-c||`;
3. let `m = median(rho)` and
   `s = 1.4826 * median(|rho-m|)`;
4. retain `rho <= m + 3.5 * max(s, 0.02 m)`.

This is a bounded robust statistic, not the quadratic kNN implementation in an
older archived ablation.

Each stage is fail-open: if it produces fewer than 16 finite points or invalid
geometry, its points and geometry are identical to the preceding stage. Every
valid hypothesis uses the unchanged F0 q02/q98 estimator and minimum AABB
extent of 0.02 m. No hypothesis may affect candidate membership, confidence,
rank, semantics, or native predictions.

## Runtime contract

Record provider, F0 core, F2 local, F2 global, F2 total, and complete online
latencies independently. Exclude receipt compression from online latency.

The completed paper100 shadow must satisfy the unchanged online envelope:

- provider p95 no greater than 200 ms per processed keyframe;
- provider + F0 + F2 complete p95 no greater than 250 ms;
- complete maximum strictly below 833.33 ms;
- gap-25 amortized complete time no greater than 10 ms per source frame;
- F2 incremental gap-25 amortized time no greater than 2 ms per source frame;
- peak GPU allocation no greater than 4 GiB.

F2 has fixed memory and candidate caps and performs no online learning.

### Pre-GT implementation amendment: exact-query acceleration

The first real one-scene replay was run without GT on `scene0568_00`. All
574 F0 sources reproduced exactly, but the Python fixed-cell traversal took
75.91 ms per keyframe on average (3.04 ms per source frame at gap 25), above
the frozen 2 ms incremental envelope. Before any F2 oracle or AP computation,
the traversal implementation was replaced by SciPy cKDTree candidate-pair
enumeration followed by the unchanged exact squared-distance predicate. The
mathematical rule, radius, inclusive boundary, point set, retained-index order,
and all downstream geometry remain frozen.

On all 574 authenticated candidates (381,961 input points), the accelerated
query and the frozen fixed-cell reference produced bit-identical HL retained
indices (zero mismatches). This amendment is permitted solely to meet the
pre-registered runtime contract; no threshold was changed and no GT, AP, or
native prediction was consulted.

The same no-GT scene was then replayed afresh with the amended implementation.
All 574 H0/HL/HLG geometries and result SHA-256 values again matched the first
replay. Measured F2 mean latency fell from 75.91 to 13.27 ms per keyframe
(gap-25 amortized 0.531 ms per source frame); complete p95 fell from 330.85 to
245.59 ms and complete maximum was 259.23 ms. Provider p95 was 83.58 ms and
peak GPU allocation was 607,840,768 bytes. Thus every frozen runtime gate
passed before paper100 execution and before any F2 GT/oracle access.

## Identity-constrained F2 oracle

Only the completed, sealed F2 receipt may be presented to GT. H0 must first
reproduce the F1 counts, official constant-score baseline AP, and H0 matching
results exactly.

For strict IoU `>` 0.15, 0.25, and 0.50, report:

- H0-only, HL-only, and HLG-only candidate and native-union matching;
- raw-to-clean threshold wins and losses;
- a grouped candidate graph in which one source candidate contributes at most
  one edge/match even if multiple hypotheses cross the threshold;
- grouped native-union maximum matching and its increment over native;
- a threshold-specific GT-selected constructive suffix with at most one
  hypothesis per source, unchanged native scene prefix, and all scores 1.0.

For a matched `(source, GT)` pair, the diagnostic oracle chooses the hypothesis
with maximum IoU; exact ties prefer `H0`, then `HL`, then `HLG`. Source rows are
appended in frozen scene/frame/candidate order. This is non-deployable and is
not actual AP.

## Decisions

F2 can never authorize birth by itself.

- Retain F2 for the later F3 shadow only if it adds at least 15 AP50 grouped
  native-union matches over the F1 value of 63, does not corrupt H0 identity,
  and passes the runtime contract.
- If it adds fewer than 15, discard the F2 geometry but F3 may still be tested
  from H0; this prevents one weak cleaning stage from blocking the distinct
  multi-view hypothesis.
- The final active gate remains unchanged and is applied only after the
  F2/F3 joint oracle: at every threshold, native-union must add at least 144
  matches and the threshold-specific constructive suffix must improve official
  AP by at least 10.0 points.
- Until that joint gate passes, a GT-free selector, active birth, and active
  paper100 evaluation are not authorized.
