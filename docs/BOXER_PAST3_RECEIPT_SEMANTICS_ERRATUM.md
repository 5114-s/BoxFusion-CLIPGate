# Boxer-Past3 receipt wording erratum

Date: 2026-08-23

The sealed `boxfusion.boxer_past3_receipt.v1` source is intentionally retained
byte-for-byte because its SHA-256 is embedded in the S1 development artifact.
Two descriptive phrases in that source are narrower than the actual API:

- `query` accepts current observations and exposes their planned assignments,
  but matches them only against the committed prior-track snapshot.  Current
  observations do not enter history until the exact query is committed, so
  same-frame self-confirmation remains impossible.
- Association thresholds and costs are geometric.  The frozen detector score
  controls deterministic row ordering and therefore can affect bounded cap,
  within-frame deduplication, and greedy assignment precedence.  The summary
  key `geometry_only_association=true` should be read as
  `geometric_gate_and_cost=true`, not as score-order invariance.

This is a documentation correction only; no candidate, threshold, association,
receipt, or result is changed.

