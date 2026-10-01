# S3 YOLOE complete-confirmed-universe dev3 preregistration

Status: frozen before the S3 exporter is run and before the S3 auditor is
allowed to open any ScanNet ground-truth file. This is a diagnostic ceiling,
not an active birth experiment.

## Scope and immutable candidate membership

- Scenes, in fixed order: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Exact gap-25 stream seal SHA-256:
  `cf363f9d92bd5b0c1aaa51ee6c200744fbf60404d671320c75df96ae20128655`.
- Frozen S2 manifest SHA-256:
  `0f15ee414003139a6b59e2092d8a0d73897acecba06132213fd7394f93cd5017`.
- Candidate membership is every causal active or archived YOLOE-direct track
  whose past/current-only view count reached three before terminal close.
- Expected no-GT counts, taken from the already frozen S2 producer summaries:
  88, 155, and 41 tracks respectively (284 total).
- No extent, projection-IoU, native-overlap, self-NMS, score, label, CLIP, or
  output-cap gate may remove a confirmed track from this diagnostic universe.
- The replay may be sealed only if its normal terminal arrays and deterministic
  512-point samples are identical to the frozen S2 diagnostics.

## Frozen producer

| Input | SHA-256 |
|---|---|
| `config/scannet_s2_yoloe_direct_shadow_score05.yaml` | `4f3e9739b296197d41c0d322c0a1e30230385ccb8c1384a36615ffa413e83441` |
| YOLOE-11s prompt-free segmentation checkpoint | `292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d` |
| frozen `demo.py` | `57fb58596401324785ee9696d16ebc15eed082df00dd6afede9e6d440b217423` |
| frozen `online_refinement.py` | `0faf3d7d6242facdd9300a942fe1e2bf2364f5f9ebc17e8f8f278382a0102f61` |
| frozen `object_memory.py` | `c2f3f0e0753a34430f0d9d03c65039aa6eee80114a1337676ec4b5f1eaa60938` |
| S3 isolated exporter | `25e1444c70613161054684789601c154787f6b1075bace02a114db7ed448aa9c` |

The S3 exporter is an in-memory observer wrapper. It must not edit any frozen
S2 producer source. It exports geometry, bounded points, causal lifecycle,
scores and per-view box records, but never reads or exports detector labels.

## Post-hoc ground-truth audit

The auditor may run only after all three no-GT scene artifacts have been
create-only sealed and that aggregate seal binds this preregistration file.
It is restricted to the three scenes above. H10 and full100 are forbidden.

Fixed strict IoU thresholds are 0.15, 0.25 and 0.50. Fixed geometries are the
producer q02/q98 AABB, point q00/q100, q01/q99, q02/q98, q05/q95, q10/q90,
the bounded per-view-box oracle, and their post-hoc per-track geometry oracle.
The oracle geometries are ceiling diagnostics and cannot become deployable
gates.

For each geometry and threshold the auditor reports candidate-only maximum
matching, frozen-native maximum matching, native-plus-candidate union maximum
matching, recovery of official-baseline-unmatched GT, and union recall points.
Two preregistered +10 checks are reported:

1. whether additional union recall over native maximum matching is at least
   10.0 points at all three thresholds; and
2. whether optimistic union recall minus frozen T05 official AP is at least
   10.0 points at all three thresholds.

Both are necessary optimistic ceilings, not evidence that a real-time,
training-free selector can attain a +10 AP gain.
