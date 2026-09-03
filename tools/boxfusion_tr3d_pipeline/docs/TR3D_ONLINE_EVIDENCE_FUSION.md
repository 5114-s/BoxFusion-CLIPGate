# Online evidence-fusion decision head

This isolated experiment tests a zero-extra-backbone `keep / replace / append`
head.  Runtime features reuse the existing online YOLOE mask, metric-depth
connected-component evidence, TR3D score, and nearest B6/R3 anchor geometry.
It does not execute a second DINO model, SPGroup3D mesh encoder, CLIP model, or
TR3D model.

The official mesh-based SPGroup3D observer remains an offline teacher.  Calling
that observer directly in the live path would not be an online or realtime
implementation.

## Train-only result

The policy was fitted on 100 ScanNet train scenes with five-fold scene-grouped
OOF evaluation.  The append gate passed its preregistered safety gate: 26
selected rows, 88.46% IoU25 precision, and 21 net IoU50 crossings.  The replace
gate failed: its best train-only operating point selected 130 rows at 74.62%
improvement precision, below the required 80%.

Consequently
`models/tr3d_online_evidence_fusion_train100_v3.json` has
`activation_authorized=false`.  The runtime loader refuses active use but
allows identity-preserving observer use.

## Validation shadow result

The frozen, non-authorized train-only threshold was evaluated on the fixed ten
validation scenes solely as a shadow experiment:

| route | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| existing R3 + online C3 | 46.1355 | 44.9020 | 32.2149 |
| evidence-fusion shadow | 46.1292 | 44.8960 | 32.2111 |

Only one replacement and two appends fired.  The replacement did not improve
the metrics, so the full-100 active run is intentionally stopped.

The observer was byte-identical over all 100 validation scenes.  Decision-head
runtime was 0.259 ms/scene mean and 0.429 ms p95; this only measures the small
decision head, not the cache-assisted BoxFusion/TR3D pipeline end-to-end.

## Files

- `boxfusion/tr3d_online_evidence_fusion.py`: fail-closed policy/runtime.
- `tools/train_tr3d_online_evidence_fusion.py`: train-only OOF trainer.
- `tools/audit_tr3d_online_evidence_fusion_observer.py`: identity and latency audit.
- `tools/materialize_tr3d_online_evidence_fusion_shadow.py`: explicitly non-authorized shadow materializer.

The next accuracy experiment should change the candidate geometry or proposal
recall.  Re-ranking the same five terminal TR3D candidates with the currently
available online evidence has reached its observed limit.
