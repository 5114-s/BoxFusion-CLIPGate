# CA-1M E961 xfit-R2 → terminal gate v5 input chain

Status: **static contract complete; operational execution blocked**.

This namespace prepares the GT-free inputs required by the exploratory
terminal gate v5.  It does not authorize detector inference, candidate
materialization, GT joining, gate fitting, fold-0 diagnosis, fold 1, official
validation, or policy activation.

## Detector cross-fit topology

| Producer | CA-only detector training set | Candidate scenes | Use |
|---|---:|---:|---|
| `outer_dev` | E941 + folds 2/3/4 = 1001 | fold 0, exact 20 | reused-dev diagnostic only |
| `inner_holdout2` | E961 + folds 3/4 = 1001 | fold 2, exact 20 | gate-fit detector OOF |
| `inner_holdout3` | E961 + folds 2/4 = 1001 | fold 3, exact 20 | gate-fit detector OOF |
| `inner_holdout4` | E961 + folds 2/3 = 1001 | fold 4, exact 20 | gate-fit detector OOF |

Every detector is fixed-iteration (`11268`) random-scratch, FP32, global
batch 16, CA-only training.  A scene's producer training folds must exclude
that scene's fold.  No checkpoint is selected using fold 0.

The three inner receipts are accepted only after the sealed outer
continuation receipt authorizes all three roles.  A receipt is not sufficient
merely because a checkpoint file exists: the producing R2 verifier must prove
the create-only success receipt, final `iter_11268.pth`, effective config,
terminal optimizer/scheduler state, checkpoint SHA256, and absence of a
ScanNet training-weight load.

## P/O/E/M stages

1. **P — role-local anchor-free proposals (GPU).** Each authoritative role
   checkpoint produces a create-only exact-20 cache from processed official
   CA train RGB-D/pose/intrinsics.  Anchors, B6, GT, fold 1, and validation are
   not reachable.  Four role directories have disjoint scene inventories.
2. **O — CPU overlay.** Geometry and canonical row indices come only from the
   sealed final-base train100 predictions.  Stacked scores come only from the
   CA-native B6-v2 all-fold OOF sidecar, and the sidecar must state that each
   row's model excluded its scene.  Deployment or in-sample B6 scores are
   rejected.  The overlay associates candidates without changing anchor row
   order or score identity.
3. **E — fresh candidate-native evidence (CPU).** Candidate visibility/depth
   evidence is recomputed against the processed CA train RGB-D frames.  The
   sealed native-B6-v2 observer rows supply anchor-native evidence.  The
   terminal-v5 40-D feature construction is then recomputed; no GT is used.
4. **M — exact80 manifest (CPU).** Role manifests are cross-checked by scene,
   fold, producer receipt, checkpoint SHA, final-base row identity, OOF score
   identity, candidate/evidence hashes, finite values, and exact directory
   inventory.  The combined result is fit60 (folds 2/3/4) plus reused-dev20
   (fold 0).

All publications are create-only, non-symlink, hash-bound, and require stable
inode/size/mtime observations while being read.  Partial output is not a
resume authority.  A later driver may validate an already complete immutable
artifact and skip it, but it may never overwrite or merge a partial namespace.

## Isolation

Formal inputs exclude every terminal v1–v4 proposal, overlay, candidate
evidence, dataset, and rejected gate policy.  ScanNet weights and artifacts
are forbidden.  Reusing implementation primitives does not authorize reusing
their old artifacts.

There is no fold-1 scene-list, GT, prediction, or output path in this config.
There is likewise no official-validation path.  Fold 0 is not available for
gate fitting or threshold selection.

## Current commands

Static validation is safe and writes nothing:

```bash
python tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5.py --static-contract
```

Operational preflight is expected to return exit code 3 until a separately
sealed ready revision binds all four authoritative successful receipts and
the passing outer continuation receipt:

```bash
python tools/preflight_ca1m_tr3d_e961_terminal_inputs_v5.py --operational-preflight
```

The runner skeleton exposes only `--run-role` for the four permitted roles
and `--seal-exact80`.  In the pending revision both stop before receipt or
checkpoint access, output creation, or GPU startup.

## Required unblock sequence

1. Complete and independently verify the expanded `outer_dev` R2 receipt.
2. Pass the preregistered fold-0 continuation gate and seal its receipt.
3. Train and independently verify the three E961 inner R2 receipts.
4. Seal a new ready/run authorization that binds the exact receipt paths and
   SHA256 values plus the then-current runner implementation hashes.
5. Run P for all four roles, then O, E, and M in order.

Until step 4, static PASS is the only valid success state.
