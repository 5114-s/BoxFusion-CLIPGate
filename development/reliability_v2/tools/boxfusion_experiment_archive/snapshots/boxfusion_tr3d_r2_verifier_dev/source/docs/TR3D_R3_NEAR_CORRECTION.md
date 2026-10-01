# TR3D R3 anchor-near correction

R3 tests a different use of the trained one-class TR3D proposals.  R1/R2
treated proposals with `IoU(TR3D, G0) <= 0.15` as missing-object candidates.
The fixed10 audit showed that this far-residual population is mostly false
geometry: only 26 of 1,352 rows overlap a ground-truth box at IoU 0.50.

The 336 proposals with `IoU(TR3D, G0) > 0.15` are much more useful.  They are
associated one-to-one with the frozen G0 boxes and observed as candidate
geometry corrections.  R3 never changes a prediction during observation.

## Frozen primary rule

For each G0 anchor:

1. collect TR3D candidates whose best axis-aligned anchor IoU is above 0.15;
2. choose the candidate with the highest TR3D foreground score;
3. replace geometry only if that score is strictly greater than the frozen G0
   score;
4. preserve the G0 label, score, row order, and CLIP semantic assignment.

The rule was frozen before the held-out90 run.  R2a depth and R2b DINO
features are optional evidence and are disabled for the authoritative run,
because their fixed-budget R2 audit did not improve AP50 selection.

## Fixed10 development result

| output | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| frozen G0 | 44.6302 | 40.8154 | 17.1297 |
| R3 counterfactual | 45.2499 | 44.0374 | 31.4913 |
| delta | +0.6197 | +3.2220 | +14.3617 |

There were 24 positive and 2 negative IoU-0.50 threshold crossings, for a
net gain of 22 and 92.31% crossing precision across seven scenes.  This is a
development signal only.  The report is explicitly veto-only and cannot
authorize active output.

## Held-out gate

The exact 10 development scenes are subtracted from the ordered 100-scene
list.  Only the remaining 90 scenes decide whether R3 continues.  The frozen
primary rule must satisfy all of:

- IoU-0.50 crossing gain minus loss at least 5;
- positive crossing coverage in at least 3 scenes;
- crossing precision at least 70%;
- paired AP50 gain at least 3 points;
- AP15 and AP25 may each fall by no more than 0.5 point.

Passing authorizes implementation of the active replacement and a paired
100-scene evaluation; it does not authorize changing the rule on validation.

## Reproduction

Generate and verify the strict p100 input, then the two-GPU parent cache:

```bash
bash scripts/export_tr3d_strict_val100_p100.sh
BOXFUSION_TR3D_RESUME=0 bash scripts/run_tr3d_strict_val100_parent.sh 0,1
```

Run the CPU-only R3 observer and authoritative held-out audit:

```bash
bash scripts/run_tr3d_r3_near_full100.sh r3_near_full100_v1
```

All caches and reports are create-only and content-bound to the frozen G0,
TR3D checkpoint/config, causal prefix artifacts, axisAlignment metadata, and
R3 code/config hashes.  CLIP semantics remain unchanged.

## Shadow-active engineering replay

The held-out90 gate passed for the frozen primary rule.  A separate replay can
therefore test whether an ordinary BoxFusion prediction tree exactly realizes
the counterfactual geometry.  This replay does not tune a threshold and does
not read ground truth while materializing predictions:

```bash
bash scripts/run_tr3d_r3_shadow_active_full100.sh r3_shadow_active_full100_v1
```

The driver is fail-closed and runs three ordered stages:

1. create a new prediction tree by replacing only selected geometry;
2. independently audit every label, score, row, and geometry byte and require
   its AP to equal the frozen all100 counterfactual;
3. run the unmodified ScanNet evaluator and require all printed metrics to
   equal the paired audit at six decimal places.

Expected all100 counterfactual metrics are `41.4870 / 36.8920 / 23.1078`,
versus frozen G0 `40.2787 / 35.4508 / 15.2181`.  Until a train-only calibration
and independent authorization protocol is completed, all generated manifests
remain explicitly marked `shadow_only=true` and
`formal_active_authorized=false`.

## Train-only veto calibration

The next experiment is a risk calibrator, not a new proposal selector.  It
first runs the unchanged primary rule, then a three-class linear model may
veto a replacement predicted as `harm`.  It can never introduce a replacement
that failed the primary rule.  Labels, scores, order, output count, and CLIP
semantics remain immutable.

The six fixed inputs are `logit(TR3D score)`, `logit(G0 score)`, anchor IoU,
normalized centre distance, absolute log volume ratio, and log point density.
Ground truth is used only by the offline train dataset builder.  Labels are
computed from the complete raw-primary train100 output: each candidate is
removed once, and its contribution to globally score-ranked AP and TP at
0.15/0.25/0.50 is recomputed.  Thus a candidate is harmful even when TP count
is unchanged but it moves a true positive to a worse score rank.  Whole-output
OOF AP, rather than candidate-level labels, is the final authority because
multiple veto decisions can still interact.  Inference has no GT access.
Scene assignment to five OOF folds is fixed by
`int(sha256(scene_id), 16) % 5`; the gate and class order have no validation-set
hyperparameters.

The gate fails closed on probability ties, requires at least four of five
folds to avoid AP50 regression against raw primary, bounds worst-fold AP15 and
AP25 loss, and rechecks all labels/AP deltas from stored geometry and GT when
loading the immutable dataset.  Train100, val100, and the official ScanNet
train list are pinned by canonical SHA256 values.

The create-only sequence is:

```bash
bash scripts/build_g0_train_anchor_manifest.sh
bash scripts/export_tr3d_strict_train100_p100.sh

BOXFUSION_TR3D_RESUME=0 \
bash scripts/run_tr3d_strict_train100_parent.sh 0,1

bash scripts/run_tr3d_r3_train100_observer.sh r3_train100_v1
bash scripts/build_tr3d_r3_train_calibration_dataset.sh r3_train100_v1
bash scripts/train_tr3d_r3_veto_calibrator.sh r3_train100_v1
```

If parent inference stops after writing a partial cache, first prove that its
process exited, then set `BOXFUSION_TR3D_RESUME=1`.  Prefix export itself is
not resumable: a partial namespace is quarantined rather than overwritten.

There is an important scope limitation.  The epoch-12 foreground TR3D model
was trained on official ScanNet train, which includes this train100 subset.
Scene-grouped OOF isolates calibration fitting, but does not make the proposal
model out-of-fold.  Every dataset/model/report therefore records
`tr3d_checkpoint_training_overlap=true`,
`independent_calibration_proof=false`, and
`formal_independent_activation_authorized=false`.  A passing gate permits one
explicitly shadow-only validation experiment; it is not evidence for a
training-free or independent-generalization claim.  Strict independent proof
would require reserved scenes excluded from TR3D training or OOF TR3D
retraining.

If the train gate passes, the one-shot shadow validation command is:

```bash
bash scripts/run_tr3d_r3_calibrated_shadow_full100.sh \
  r3_calibrated_shadow_full100_v1 \
  models/tr3d_r3_veto/r3_train100_v1.json
```

Before any prediction namespace is claimed, the materializer verifies the
TR3D checkpoint/config, R3 code/config, frozen primary policy, `p100` scope,
and the G0 score/extent/quality/Selective-Boxer distribution contract.  This
experiment remains end-of-scene `p100`; it does not establish real streaming
latency or causal intermediate-prefix quality.

Only when the frozen model contains both `activation_authorized=true` and a
matching positive train-gate attestation may the isolated validation replay be
started.  The model path is explicit; no "latest" checkpoint is discovered:

```bash
bash scripts/run_tr3d_r3_calibrated_shadow_full100.sh \
  r3_calibrated_shadow_full100_v1 \
  models/tr3d_r3_veto/r3_train100_v1.json
```

The calibrated materializer revalidates the frozen G0, R3 export, every R3
sidecar, and their parent checkpoint/config/prefix lineage before applying the
veto.  Its create-only manifest binds both the calibrator file SHA and its
canonical model SHA.  The result remains a one-shot `shadow_only=true`
validation experiment; a train-gate pass does not change the overlap
disclosure above or authorize validation-driven retuning.
