# C1→C2 full-100 observer result

This run evaluates unmatched TR3D residual proposals without changing the
frozen R3 prediction tree.  C1 ranks proposals with R2a metric-depth/free-space
evidence and R2b multi-view DINO features.  C2 confirms them with cached SAM3
masks and real ScanNet depth/poses.

## Frozen inputs

- Scene list: `evaluation/data_util/meta_data/scannetv2_val.txt` (100 scenes)
- Development scenes excluded from the authoritative decision:
  `evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt`
- TR3D parent: `tr3d_prefix_boxfusion_causal_p100_full100_v3`
- Frozen active predictions: R3 full-100 `scannetv2_val-4b18fc586f7a`
- SAM3 namespace: `sam3-scannet18-val100-c050-frozen-v1`
- DINO checkpoint SHA256:
  `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea`

## C1 result

C1 observed 12,549 unmatched candidates.  The per-scene Top-10 route retained
1,000 candidates and provided 209/194/106 novel oracle matches at IoU
0.15/0.25/0.50.  C1 is observer-only; standard AP is unchanged.

## C2 result

Candidate precision below means independent probability of hitting any GT at
the specified IoU.  Novel TP is the oracle increase over the frozen active
prediction set, not measured AP gain.

| Partition / route | Candidates | P(hit)@0.25 | P(hit)@0.50 | Novel TP@0.25 | Novel TP@0.50 |
|---|---:|---:|---:|---:|---:|
| all100 / source Top-10 | 1000 | 23.9% | 10.9% | 194 | 106 |
| all100 / mask2+depth | 294 | 46.9% | 24.8% | 113 | 70 |
| all100 / Top-5 ∩ mask2+depth | 170 | 60.0% | 38.8% | 97 | 65 |
| heldout90 / source Top-10 | 900 | 22.9% | 10.0% | 169 | 87 |
| heldout90 / mask2+depth | 251 | 45.0% | 23.5% | 94 | 56 |
| heldout90 / Top-5 ∩ mask2+depth | 145 | 57.2% | 36.6% | 79 | 52 |

The pre-registered primary `mask2_depth` gate **fails** on heldout90 because
P(hit)@0.25 is below 50% and P(hit)@0.50 is below 25%.  Therefore this result
does not authorize active C2 materialization.  `Top-5 ∩ mask2_depth` is a
promising pre-recorded diagnostic route, but any score/materialization study
must remain a separate C3 shadow experiment and must not be reported as the
pre-registered C2 result.

## Safety and provenance

- `observer_only=true`
- `mutation_enabled=false`
- `applied_count=0`
- inference-side GT access: false
- inference-side CLIP access: false; CLIP semantics unchanged
- frozen prediction tree before/after SHA256:
  `6418bb137463ee946112631d672c0b25ce7f47a3195000017a68b79d418bc1f8`

Reports:

- `artifacts/tr3d_c1_track/c1_r3active_full100_v1/reports/gt_audit.json`
- `artifacts/tr3d_c2_maskrgbd/c2_c1top10_full100_v1/reports/gt_audit.json`

Reproduction entry points:

```bash
bash scripts/run_tr3d_c1_track_full100.sh c1_r3active_full100_v1
bash scripts/run_tr3d_c2_maskrgbd_full100.sh c2_c1top10_full100_v1
```

## Exploratory C3 shadow (no prediction files written)

After the strict C2 decision, a separate GT-only counterfactual measured the
already-recorded `Top-5 ∩ mask2_depth` route.  Every candidate was constrained
to rank below every frozen anchor, so this tests added low-score recall without
perturbing the original score order.

| heldout90 policy | AP15 | AP25 | AP50 | Delta vs same-run anchor |
|---|---:|---:|---:|---:|
| frozen anchor | 41.1340 | 36.1489 | 22.4782 | — |
| fixed-low append | 44.1668 | 38.8692 | 23.7779 | +3.0328 / +2.7203 / +1.2997 |
| C1-ranked low append | 44.2210 | 38.9197 | 23.8068 | +3.0870 / +2.7708 / +1.3286 |
| GT oracle ordering upper bound | 44.3354 | 39.0286 | 23.8584 | +3.2014 / +2.8797 / +1.3802 |

The all100 C1-ranked counterfactual is 44.9866/40.0373/24.7644, a paired
increase of +3.4997/+3.1456/+1.5542 over its same-protocol anchor.  The small
gap between C1 ranking and the GT ordering upper bound shows that the main
remaining limitation is candidate geometry/coverage, not ranking among these
145 held-out candidates.

This C3 result is explicitly exploratory and post-hoc.  It does not authorize
active materialization, and it did not change the frozen prediction tree.  Its
immutable report is:

- `artifacts/tr3d_c3_shadow/c3_top5_mask2_full100_v1/report.json`

## C3 engineering active replay (official evaluator)

For engineering verification only, the frozen C3 route was materialized in a
new immutable prediction tree and evaluated with the unmodified ScanNet
evaluator.  The original 1,759 R3 rows remain first and are value/type/byte
identical after loading; 170 class-agnostic candidates are appended.  Candidate
scores are positive, preserve the global C1 order, and are all below the frozen
anchor score floor.

| full100 official evaluation | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| frozen R3 anchor | 41.4869 | 36.8917 | 23.2102 |
| C3 Top-5 + mask2-depth | **44.9866** | **40.0373** | **24.7644** |
| paired delta | **+3.4997** | **+3.1456** | **+1.5542** |

The official mAP values match the frozen in-memory shadow values exactly at
six decimal places.  The GT-free identity audit also verifies all 1,759 anchor
rows, all 170 appended rows, the input/output tree hashes, and the global score
ordering.  Materialization itself took 0.997 seconds from cached evidence;
this is not an end-to-end online runtime measurement because TR3D, R2a/R2b and
SAM3 evidence had already been cached.

This remains a post-hoc engineering replay (`formal_active_authorized=false`),
not an independently calibrated validation result or a formal training-free
claim.  Immutable outputs:

- `reports/tr3d_c3_active/c3_top5_mask2_active_full100_v1/materialize_manifest.json`
- `reports/tr3d_c3_active/c3_top5_mask2_active_full100_v1/identity_audit.json`
- `reports/tr3d_c3_active/c3_top5_mask2_active_full100_v1/standard_eval_verification.json`
- `logs/tr3d_c3_active/c3_top5_mask2_active_full100_v1/eval_stdout.log`

Reproduction entry point (a new tag is required because outputs are immutable):

```bash
bash scripts/run_tr3d_c3_active_full100.sh <new-unique-tag>
```
