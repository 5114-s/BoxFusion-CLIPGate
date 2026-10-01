# CA-1M final-base native B6 v2

This stage starts only after the fixed-10 G0+CLIP+reliable-TopK3 result is
accepted and `ca1m_native_final_base_train100_v1` has sealed all 100 scenes.
The authoritative `paired_eval_report.json` is mandatory: it must authorize
train100 collection, require CA-native B6 retraining, forbid canonical active
use, and show positive mAP deltas at AP15/AP25/AP50.  Its path and SHA256 are
sealed into the v2 contract, source audit, collection, dataset, and checkpoint
provenance.  The run environment flag is only an explicit operator action and
cannot substitute for that report.

The sealed final-base `*_boxes.pkl` files are the sole geometry and score
authority.  V2 does **not** independently replay CuTR, Selective Boxer, CLIP,
or BoxFusion: independent CA-1M fusion replays can differ at a small number of
corners because of PyCUDA numerical drift, so cross-run byte identity is not a
valid correctness gate.  Instead, the offline collector reads the same CA
train depth/K/pose keyframes and calls `CA1MNativeB6Observer` directly on each
sealed row.  It inventories RGB filenames for frame alignment but never
decodes RGB pixels.

The keyframe lineage reproduces the established `demo.py` loop, including its
record-before-increment and early-finalize order.  It does not force the
physical terminal frame.  This policy is checked against all 100 sealed v1
train diagnostics; the frame-ID arrays match 100/100.  Real processed
landscape, portrait, and raw-mixed-orientation scenes are also checked against
`CA1MDataset` for exact depth, intrinsics, and pose parity.

The two Top-K values retain separate roles:

- final-base reliable-view Top-K=3 is provenance of the already sealed boxes;
- offline native-B6 Top-K=5 selects depth evidence for those boxes.

Read-only readiness check (the second argument is a CPU worker count):

```bash
bash scripts/collect_ca1m_native_b6_final_base_train100_v2.sh --preflight 2
```

Explicit offline collection after accepting fixed-10:

```bash
BOXFUSION_CA1M_FINAL_BASE_FIXED10_ACCEPTED=1 \
  bash scripts/collect_ca1m_native_b6_final_base_train100_v2.sh --run 2
```

The runner is resume-safe for the one recognized interrupted state.  If an
observer NPZ exists without its receipt, it recomputes the diagnostic from the
sealed anchor and depth lineage in a temporary directory, requires every NPZ
field to be exact except wall-clock `observer_seconds`, and only then creates
the missing receipt.  A receipt without its diagnostic or any unsafe/symlinked
artifact fails closed; the runner never deletes unknown output.

Build the isolated CA-train dataset, audit it, and fit B6:

```bash
bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --build-dataset
bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --preflight
bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --train
```

The established native-B6 split remains deterministic scene-grouped five-fold:
the deployable model is trained on folds 1--4 (80 scenes), fold 0 remains an
untouched 20-scene development split, and all-fold out-of-fold models are used
only by the train-only activation gate.  This is separate from the TR3D 60/20/20
protocol.  Collection does not read train GT; GT is joined only during the
later train-dataset build.  No official CA validation GT, predictions, or
evaluator are read by collection, dataset construction, or training.
