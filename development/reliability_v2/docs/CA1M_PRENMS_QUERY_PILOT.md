# CA-1M pre-NMS historical-query diagnostic

This is an isolated development pilot, not a production change, held-out AP
test, or end-to-end FPS benchmark. It does not alter the dynamic75 experiment.

## Frozen scope

- First three scenes in the existing full107 list: 42446540, 42897501,
  42897521. Full available sequences, gap 20; a separate one-frame integration
  check is excluded from reported results.
- Frozen WeDetect-Uni and Boxer weights; one ordinary WeDetect forward per
  frame. Observe the existing head before NMS without changing its output.
- Normal stream: original post-NMS proposals, score >= 0.05, at most 150/frame.
- Past-only normal-stream pending memory: 64 tracks, TTL 10 keyframes, at most
  three observations and two feature prototypes per track. Query only tracks
  with one or two past observations. Current ordinary support takes priority.
- At most eight targets/frame. Identical local candidate pool and top-one
  budget for historical-feature query versus detector-score control. Neither
  arm feeds observations back into memory. Extra Boxer queries must reuse the
  current frame's cached encoder output.
- Runtime never reads GT or final predictions. Final-map misses and seed
  identity are determined only by a separate offline evaluator.

## Questions and limits

1. Does historical query retrieve more correct additional 3D observations than
   same-budget score ranking? Report all candidates and the identical geometry
   acceptance gate separately, including same-frame normal-stream duplicates.
2. Do these observations concern GT missed by the existing final map, and can
   they provide third-frame support? Offline GT-union counts are diagnostic
   opportunities, not actual online births or measured AP.
3. Separately sample up to three local raw boxes ranked using projected GT,
   then run frozen Boxer. This is explicitly GT-assisted sampling, NOT an
   exhaustive 3D oracle or an upper bound for all pre-NMS proposals.

Report seed purity, correct retrieval, new support and confirmation opportunities
at the evaluator's IoU thresholds. Do not launch full107 merely because raw
candidate count is large. A negative pilot rejects this implementation on these
scenes, not every possible use of dense features or historical queries.

Timing includes diagnostic copies, two arms, and artifact writing; it excludes
the complete native BoxFusion pipeline. It cannot establish 15 FPS compliance.
