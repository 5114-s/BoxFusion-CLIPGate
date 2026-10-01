# F5 independent no-GT validation checklist

Purpose: validate the frozen F5 geometry selector without importing or opening
ground truth, annotations, evaluator code/output, oracle reports, or native
predictions.  These checks are independent of the F5 implementation and do
not authorize evaluation, birth, or output mutation.

The checklist targets protocol
`F5-GT-FREE-PAST-ONLY-GEOMETRY-SELECTOR-PAPER100`.  The final test must pin the
SHA-256 of `docs/F5_GT_FREE_GEOMETRY_SELECTOR_PROTOCOL_FREEZE.md` as
`2a6d62fa9d5912dc3871bbc485f44987565bda61b818722b3a4e6577d34a6afc`.
A draft hash must never be silently updated after an F5 receipt has been
produced.

## 1. Protocol and executable provenance

- Require the exact protocol ID and exact scene/shard/merge schemas in every
  receipt.
- Re-hash the final protocol, selector core, runner, merge, and any helper
  source before and after each run.  Record their paths and hashes in the run
  signature.
- Negative test: change one byte of a copied protocol or source.  Planning or
  replay must fail before the first source is processed.
- Require create-only scene, shard, merge, and determinism receipts.  A second
  write to any existing output path must fail.

## 2. Sealed F4 identity and geometry lineage

- Authenticate the exact, passing F4 merge receipt before source processing.
- Reproduce exactly 100 scenes, 6,817 scheduled frames, 6,726 successful
  frames, and 52,299 unique sources in frozen scene/frame/rank order.
- For every source, compare all identity fields exactly:
  `scene_index`, `frame_ordinal`, `frame_id`, `rank`, `raw_index`,
  `mask_sha256`, `points_and_voxel_keys_sha256`, `source_id`, and the sealed F4
  source-lineage hash.
- Re-hash all four input hypotheses independently.  The selector may read them
  but may not alter them.
- Negative tests: duplicate, remove, reorder, or re-key a source; change one
  hypothesis scalar; change a mask/evidence/hash field.  Every case is fatal,
  never a selector fallback.

## 3. No annotation, oracle, evaluator, or prediction dependency

Use three independent guards:

1. Parse the F5 production modules with `ast` and reject imports of annotation,
   GT, evaluator, oracle/audit, or native-prediction loaders.
2. Assert the production CLI has no GT, annotation, evaluator, oracle,
   baseline/native prediction, label, class, CLIP, training, calibration, or
   optimizer path option.
3. Run a synthetic replay with guarded `open`, `Path.open/read_text`, NumPy
   load, and directory-enumeration calls.  Permit only an explicit sealed-input
   allow-list; any unlisted read fails the test.

- Enforce a decision-feature allow-list.  Scene/frame/source IDs, ranks and
  hashes may bind identity or break the explicitly frozen mutual-best ties,
  but must never become a threshold, score, pseudo-random seed, or selector
  feature.  Log-variance and raw Boxer parameters are diagnostics only.
- Evidence NPZ reads must be restricted by the current source's sealed
  `offsets`; reject whole-scene percentiles, aggregates over later rows, or any
  source slice whose frame ordinal is in the future.

The independent test module itself must not import an oracle or deserialize a
GT/native prediction file.

## 4. Past/current-only causality

- Require monotonically increasing scheduled frame ordinals and expose the
  maximum accessed ordinal for every decision.
- Prove that a decision reads only current sealed source/evidence data and the
  buffer snapshot from before the current frame update.
- The buffer must contain at most three prior successful frames, each with at
  most 16 selected source geometries, and entries with ordinal distance greater
  than three must be evicted.
- Replay the midpoint prefix from an empty buffer and require byte-identical
  ordered result hashes to the full-run prefix.
- Synthetic future-perturbation test: change, append, delete, and reorder all
  future-frame contents while keeping the prefix fixed.  Every prefix result
  hash must remain identical.
- Negative test: expose a future source or update the buffer before deciding
  the current source.  The causality validator must fail.
- Present the same geometries under different non-tie identity strings and
  hashes.  Decisions and decision scalars must remain equal; only explicitly
  documented tie outcomes may follow rank/source-ID order.

## 5. One source, one selected hypothesis

- Every sealed source produces exactly one output row and one selected name in
  `{H0, HL, HLG, HB}`.  Selection must not add, remove, split, stack, merge, or
  persist an object identity.
- The selected geometry must be an exact deep copy of the named input
  hypothesis, including HB center, local extent, rotation, and eight corners.
- Enforce deterministic base priority `HLG > HL > H0` and final priority
  `eligible-and-confirmed HB > eligible HLG > eligible HL > H0`.
- Exercise mutual-best ties: lower past rank, then past `source_id`; lower
  current rank, then current `source_id`.  Repeated matches from one past frame
  count once.
- Negative tests: two selections for one source, selected geometry from a
  different source, a hybrid geometry assembled from two hypotheses, or one
  source emitted twice.  All are fatal.

## 6. Exact H0 fallback and threshold boundaries

- Malformed H0 or broken H0/source lineage is fatal.
- Missing, non-finite, invalid, or ineligible optional HL/HLG/HB evidence must
  retain the source and fall back to the already selected base; if no cleaned
  hypothesis is eligible this is exact H0.
- For every HB failure stage, assert the selected row is exact Hbase and that
  `HB_abstention_reason` is the first failure in the frozen order.
- Cover equality at every inclusive threshold, plus one representable float
  immediately below and above it.  No hidden epsilon is allowed.
- Explicitly cover zero history, one distinct confirming frame, two matches
  from the same past frame, invalid HB, malformed confidence, and insufficient
  depth evidence.  None may drop a source.

## 7. Determinism

- Run two independent CPU replays from fresh selector/buffer instances.
- Remove only explicitly declared runtime/environment diagnostics and compare
  canonical ordered per-row hashes, per-scene ledgers, and the global ledger.
- Repeat with shuffled JSON mapping key order and independent input array
  copies; results must remain identical.
- Verify no process-global random state, wall clock, hash randomization, thread
  scheduling, GPU state, or insertion order enters a decision.
- Negative test: inject a nonce/random tie-breaker and require the determinism
  validator to reject the mismatched ledger.

## 8. Native output and semantic immutability

- The F5 production CLI must not accept a native/terminal prediction root.
- Hash every in-memory protected native field before and after any integration
  adapter call, and re-hash any sealed native prefix file before and after the
  complete replay.
- Require `native_output_mutation_count == 0`, `source_addition_count == 0`,
  and `source_removal_count == 0` in scene, shard, and merge receipts.
- Mutation tests must cover geometry, order, class, CLIP embedding, object
  description, confidence, and score.

## 9. Constant formal score

- Every selected-source row must seal `formal_score` as the numeric value
  exactly equal to `1.0`.
- Boxer confidence is allowed only as the frozen scalar HB gate; it may not be
  copied into the formal score, rank, class, or semantic fields.
- FastSAM/F2 diagnostic confidence must likewise never become an output score.
- Negative tests: `0.999999`, string `"1.0"`, NaN/Inf, missing score, or a
  copied Boxer/FastSAM confidence must fail the receipt validator.

## 10. Runtime and bounded-memory accounting

- Time current-source offset slicing/copy/validation/decode, base checks,
  projection, bounded past matching, and buffer update.  Per the final frozen
  clarification, exclude one-time whole-scene NPZ inflate/hash, serialization,
  hashing, and the second determinism replay from online time.
- Exclude exactly the first three non-empty F5 frames per shard only from warm
  distributions; preserve all-frame diagnostics.
- Independently recompute count/mean/p50/p95/max from frame rows and require
  exact agreement with scene, shard, and merge summaries.
- Recompute and test the frozen gates: F5 p95 `<=25 ms`, composed p95
  `<=375 ms`, composed warm max `<833.33 ms`, composed warm mean/25 `<=15 ms`,
  zero warm gap-25 misses, CUDA peak `<=4 GiB`, no F5 CUDA allocation, and the
  3-frame/16-source buffer bounds.
- Boundary tests must distinguish inclusive from exclusive gates.  Inject a
  slow frame, a fourth buffered frame, a seventeenth source, non-finite timing,
  and inconsistent composed arithmetic; each must fail closed.

## 11. No-GT stopping decision

- Recompute selected-HB count, selected-HB scene coverage, and selection ratio
  from rows rather than trusting summaries.
- Passing requires at least 128 selected HB sources across at least 20 scenes,
  no more than 20% of all sources, and all preceding integrity, causality,
  determinism, immutability, and runtime checks.
- Test all three mutually exclusive outcomes without GT:
  `discard_f5_selector`, `stop_f5_insufficient_confirmed_hb`, and
  `stop_f5_overbroad_hb`; only a complete pass may return
  `retain_f5_for_one_separately_sealed_evaluation_only`.
- No outcome may set birth, terminal mutation, or deployment authorization.

## Minimum independent test-file layout

Once the production API is frozen, implement these tests outside the F5 main
files, preferably as:

- `tests/test_f5_gtfree_selector_no_gt_contract.py`: pure selector geometry,
  boundaries, unique selection, H0 fallback, mutual-best ties, causality, and
  determinism;
- `tests/test_run_scannet_fastsam_f5_gtfree_selector_paper100.py`: sealed
  joins, forbidden I/O/CLI, source and score immutability, create-only behavior;
- `tests/test_merge_scannet_fastsam_f5_gtfree_selector_paper100.py`: complete
  census/ledger recomputation, runtime gates, HB coverage, and stopping result.

All fixtures must be synthetic or copied from no-GT F4 receipts.  These tests
must never reference a GT root, annotation file, oracle report, evaluator, or
native prediction pickle.
