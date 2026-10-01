# PLR-v2 cumulative matched protocol

This protocol was written before the PLR-v2 official100 run. The frozen strong
prefix is the within-run composition of CALR-v1 and MVSR-v2 with max aggregation.
PLR receives post-NMS proposals only, never NMS-suppressed children. Every PLR
arm has at most 12 outputs per scene and remains outside native association.

## Cumulative arms

1. **Current**: frozen revisable PLR-v1, greedy association to the latest
   observation, raw mean proposal score for budget ranking, and size pricing.
2. **+Assoc**: replace only association with deterministic query-before-commit
   one-to-one matching. Tracks are represented by their current geometric
   medoid. Edges require AABB IoU >= 0.10, normalized centre distance <= 0.75,
   and mean absolute log-scale error <= 0.70. The edge cost is
   `(1-IoU) + 0.50*normalized_distance + 0.25*scale_error`.
3. **+Reliability**: retain +Assoc and rank confirmed candidates by
   `median_proposal_score * geometric_stability * n_eff/(n_eff+2)`, where
   geometric stability is the geometric mean of medoid overlap, normalized
   centre stability and scale stability. Effective views must differ by at
   least 20 degrees. Output scores remain the frozen size prices.
4. **+Score**: retain +Reliability and replace size pricing with a deterministic
   reliability-rank mapping into the fixed open interval (0.05, 0.50). For N
   selected rows and zero-based rank r, the score is
   `0.05 + 0.45*(N-r)/(N+1)`.

All constants are frozen before official100. No threshold is changed after
reading ScanNet results.

## Sequential gate

Starting from Current, a later cumulative stage is retained only when it:

- improves strong-prefix AP25 or IoU-0.25 recall at 5, 10 or 20 false
  positives per 100 GT; and
- does not reduce AP50 by more than 0.10 point relative to the immediately
  preceding retained stage.

The same online pass exports all four states. This removes front-end and native
PFO cross-run variation. Geometry/count differences between PLR arms are part
of the tested PLR mechanism; the CALR/MVSR prefix is identical for all arms.

The audit also reports output count, newly covered prefix-missed GT, overlap
with CALR, duplicate rate, per-keyframe latency and output-set Jaccard stability.

