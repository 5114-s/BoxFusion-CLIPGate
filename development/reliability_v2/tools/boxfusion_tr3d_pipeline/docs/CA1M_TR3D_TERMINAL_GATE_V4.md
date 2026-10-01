# CA-1M final-base terminal benefit gate v4

This route reuses only the old terminal gate's pure 40-D feature math and
dual-logistic optimizer.  It does not reuse a ScanNet checkpoint/config, CA
terminal v1/v2/v3 cache, old CA native-B6 diagnostic/checkpoint, old candidate
evidence, benefit dataset, thresholds, or policy.

## Why the OOF sidecar is mandatory

The deployable native-B6 v2 model is trained on folds 1--4.  Its scores on
folds 2/3/4 therefore cannot be inputs to another model trained on folds
2/3/4.  `tools/train_ca1m_native_b6_quality.py` now has optional, backward-
compatible OOF outputs, and the final-base v2 wrapper always supplies them:

- `models/ca1m_native_b6_final_base_oof_row_scores_v2.npz`
- `models/ca1m_native_b6_final_base_oof_row_scores_v2.manifest.json`

Every row is identified by `(scene_id, fold_id, source_row_index)` and is
scored by the model trained on all folds except that scene's fold.  The
sidecar includes raw/monotonic/quality/deployment-blend scores, five stable
fold-model hashes, the full fitting recipe, and bidirectional checkpoint
provenance.  The checkpoint, OOF NPZ, OOF manifest, and checkpoint manifest
are constructed before first publication and published as one all-owned
rollback transaction.

The terminal benefit split is fixed:

- folds 2/3/4: fit, 60 scenes;
- fold 0: threshold development, 20 scenes;
- fold 1: locked internal check, 20 scenes.

The same scene-fold namespace as native-B6 v2 is required.  A deployed B6
score is forbidden as a stacked-training anchor score.

## Fail-closed state

`config/ca1m_tr3d_benefit_gate_train100_v4.json` is intentionally pending.
The following succeeds because it checks only the static contract:

```bash
bash scripts/train_ca1m_tr3d_benefit_gate_v4.sh --static-contract
```

The operational modes currently fail before creating a directory or file:

```bash
bash scripts/train_ca1m_tr3d_benefit_gate_v4.sh --preflight
bash scripts/train_ca1m_tr3d_benefit_gate_v4.sh --run
```

`--run` does not fit a model.  Once all sources are sealed, it creates the one
immutable v4 training binding that later dataset/training/materialization
tools must consume.  This separates source authorization from scientific gate
training and prevents a missing prerequisite from leaving partial outputs.

## Exact continuation order

Run these stages in order.  Do not skip a failed train-only scientific gate.

1. Finish and seal exact100 final-base train collection and same-run identity.
2. Collect final-base B6-v2 evidence, build its train-only dataset, and train:

   ```bash
   BOXFUSION_CA1M_FINAL_BASE_FIXED10_ACCEPTED=1 \
     bash scripts/collect_ca1m_native_b6_final_base_train100_v2.sh --run 2
   bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --build-dataset
   bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --preflight
   bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --train
   ```

   Exit 3 is a scientific failure.  Do not activate B6 or continue the gate
   route.  A successful run must seal the two OOF files above and list them in
   the B6 checkpoint manifest.

3. Finish the CA-scratch TR3D stage-P exact100 proposal cache:

   ```bash
   bash scripts/collect_ca1m_tr3d_terminal_train100_v4.sh --run-proposals
   ```

4. Populate only the final-base/B6-v2 fields in the terminal-v4 overlay
   config, seal its authorization, and create exact100 CPU overlays:

   ```bash
   python tools/overlay_ca1m_tr3d_terminal_v4.py \
     --collection-config config/ca1m_tr3d_terminal_train100_v4.json
   ```

5. Collect candidate TopK5 depth evidence into the new v4 namespace and seal
   a `boxfusion.ca1m_tr3d_candidate_evidence_collection.v4` manifest.  Its
   per-scene records must bind both the stage-P proposal SHA and stage-O overlay
   SHA.  This collector is GT-free; only after its manifest is sealed may a
   dataset builder open CA train-derived GT.
6. Fill the null prerequisite records in
   `config/ca1m_tr3d_benefit_gate_train100_v4.json` with immutable
   path/SHA/schema records, change state to
   `ready_after_all_train100_seals`, set `run_authorized=true`, and chmod the
   config read-only.
7. Validate and seal the source binding:

   ```bash
   bash scripts/train_ca1m_tr3d_benefit_gate_v4.sh --preflight
   bash scripts/train_ca1m_tr3d_benefit_gate_v4.sh --run
   ```

8. Build the isolated v4 dataset from that binding.  Candidate collection is
   already sealed, so this is the first stage allowed to read train-derived
   GT.  Fit folds 2/3/4, choose both thresholds only on fold0, and reveal fold1
   once for the locked internal decision.  A policy is activatable only if all
   frozen AP/harm gates pass.
9. Materialize train100 evidence with OOF anchor scores and geometry-only
   replacements.  Scores, row order/count, and CLIP semantics remain byte-
   identical.  Canonical103 remains unauthorized until this paired train-only
   audit passes.

Steps 5, 8, and 9 are deliberately schema boundaries in the current static
skeleton; they must be implemented against the sealed v4 binding rather than
by pointing old v1/v2 tools at new directories.

