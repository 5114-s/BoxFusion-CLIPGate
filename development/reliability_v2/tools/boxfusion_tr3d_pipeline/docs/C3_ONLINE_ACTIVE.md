# C3 online active branch

This branch appends online YOLOE + real-depth confirmed terminal TR3D
candidates to the paired R3 output.  Existing R3 boxes, scores, and ordering
are byte-identical.  CLIP semantics are unchanged.

Activation is fail-closed.  The runtime accepts only an immutable
`boxfusion.tr3d_c3_source_gate.v1` checkpoint that:

- was fitted on ScanNet train scenes only;
- enumerates the complete forbidden validation partition;
- has zero train/validation scene overlap;
- passed five-fold scene-grouped out-of-fold precision gates;
- was not calibrated with validation predictions.

## Train-only preparation

The current machine has terminal TR3D train caches and extracted ScanNet
RGB-D under `/extra/ZhaoX/scannet_data/scans.sens`, but it does not yet have
the 100-scene train C2 sidecars or online C3 identity diagnostics.  Generate
those train-only observer artifacts first.  Never substitute validation C2
or validation identity JSON files.

After the train diagnostics exist:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
BOXFUSION_PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python \
BOXFUSION_C3_TRAIN_DIAGNOSTICS_ROOT="$PWD/diagnostics/tr3d_c3_online_train100_v1" \
bash scripts/train_scannet_tr3d_c3_source_gate.sh
```

If `activation_authorized` is false, stop.  The policy is intentionally
rejected by inference.

## Validation smoke test

Only after the train gate passes:

```bash
BOXFUSION_C3_ACTIVE_POLICY="$PWD/models/tr3d_c3_source_gate_train100_v1.json" \
BOXFUSION_C3_ACTIVE_RUN_TAG=c3_online_active_smoke1_v1 \
bash scripts/run_scannet_tr3d_c3_online_active.sh 0
```

Use a new run tag for every run.  The launcher performs a GT-free identity
audit before standard ScanNet evaluation.

## Realtime limitation

This is an online C3 confirmation/activation path, not yet an end-to-end live
BoxFusion claim.  Terminal TR3D proposals are still replayed from immutable
`p100` caches.  A valid realtime measurement must also run CuTR/TR3D live and
measure one input stream; two-GPU scene sharding is throughput, not lower
single-stream latency.
