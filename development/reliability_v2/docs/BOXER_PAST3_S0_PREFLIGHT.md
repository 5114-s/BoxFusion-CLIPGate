# Boxer-Past3 S0 shadow/oracle preflight

Date: 2026-08-23

## Decision

Reject S0 active birth.  The fixed no-GT three-view confirmer produced seven
terminal candidates on the sealed three-scene preflight, but none matched a GT
box at IoU 0.15, 0.25 or 0.50.  Appending those seven rows after the native T05
prefix would reduce constant-score AP at every threshold.

The native T05 predictions remain byte-identical.  No birth was enabled and a
full100 S0 run is not authorized.

## Frozen protocol

- Native prefix: T05, `score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K3.
- Formal score mode: every prediction is evaluated at score `1.0`.
- Shadow input: sealed frozen OWLv2 + Boxer per-view candidates, no GT access.
- Scene order: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Association: chronological query/commit replay of the complete sealed
  schedule.  Zero-candidate keyframes advance TTL; invalid pose frame 1325 is
  excluded without future-frame substitution.
- Three-view and terminal thresholds were frozen in
  `BOXER_PAST3_SHADOW_PREREGISTRATION.md` before the S0 oracle.
- Native T05 geometry is used only after confirmation for terminal duplicate
  suppression.  It does not participate in per-frame association.

Protocol erratum: the preregistration prose describes the transferred match as
`IoU >= 0.10 OR center <= 0.50 m`.  The sealed tracker source actually requires
both conditions (`AND`), and that source hash is what was executed.  All S0
numbers in this report therefore use `IoU >= 0.10 AND center <= 0.50 m`.  The
sealed preregistration file is intentionally not rewritten after the oracle;
this erratum records the discrepancy.  Because S0 is rejected, the correction
does not promote or rescue an experiment.

## Candidate funnel

| Scene | Raw Boxer rows | Scheduled keyframes | Confirmed tracks | Pre-view terminal | Final shadow candidates |
|---|---:|---:|---:|---:|---:|
| scene0568_00 | 941 | 66 | 124 | 5 | 4 |
| scene0606_01 | 1,802 | 112 | 201 | 6 | 3 |
| scene0377_02 | 342 | 30 | 39 | 1 | 0 |
| **Total** | **3,085** | **208** | **364** | **12** | **7** |

All seven final candidates satisfy the frozen 0.15 m camera-baseline and
10-degree viewing-ray gates.  The confirmer itself is inexpensive: per-scene
mean keyframe time is 0.37--0.57 ms, p95 is at most 1.08 ms, and terminal close
is at most 30.4 ms.  These times exclude frozen OWLv2 + Boxer inference.

## Fixed-candidate oracle

| Metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| T05 AP, score=1.0 | 27.7670 | 27.7670 | 16.2060 |
| T05 + fixed seven-candidate suffix AP | 24.1195 | 24.1195 | 13.6694 |
| AP delta | **-3.6475** | **-3.6475** | **-2.5366** |
| Candidate maximum-matching TP | 0 | 0 | 0 |
| Native-union additional maximum matches | 0 | 0 | 0 |
| Added false positives under formal evaluator | 7 | 7 | 7 |

The candidates were fixed before GT evaluation.  GT was used only by the
separate oracle and did not select a subset.  Therefore this is a valid
rejection of S0, not a tuned negative result.

## Diagnosis and next experiment

S0 demonstrates that repeated OBB geometry alone is not object correctness.
The transferred tracker also keeps a rolling last-five-observation geometry;
after a track has reached its third view, later association can change the
terminal medoid.  That is undesirable for a causal birth receipt.

S1 should therefore remain shadow-only and make two structural changes:

1. freeze the candidate geometry and provenance immediately when the third
   distinct-view stability condition is first met; and
2. require past-guide-to-current-depth consistency before issuing that frozen
   receipt, using only current/past RGB-D and pose.

The three S0 scenes have now been inspected with GT and are development-only.
S1 thresholds must be fixed without further GT tuning and evaluated first on a
disjoint held-out scene set before any full100 claim.

## Artifacts

- Preregistration: `docs/BOXER_PAST3_SHADOW_PREREGISTRATION.md`
- Shadow materializer: `tools/materialize_boxer_past3_shadow.py`
- Independent oracle: `tools/audit_scannet_boxer_past3_oracle.py`
- Shadow JSON SHA-256:
  `55acc5e3ecb7885ac205564531e4757f0cb5dc220736f8b64f9576ee91b33d9b`
- Shadow NPZ SHA-256:
  `7e54edfb737bba81050db8998a0f0a9b3c91cbf42abfda0e3874c5d767e33496`
- Oracle JSON SHA-256:
  `38df31de3477d962a764ac0e55fc44c4e583250e252fa66435584a4fb96a1ddc`
