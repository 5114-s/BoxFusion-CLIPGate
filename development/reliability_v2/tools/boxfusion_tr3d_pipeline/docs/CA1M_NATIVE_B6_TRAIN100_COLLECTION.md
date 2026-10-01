# CA-1M native-B6 train100 collection

This is an isolated train-only diagnostic collection stage. It does not train a
quality model, change validation predictions, call the CA-1M evaluator, or load
validation ground truth.

The frozen input contract is
`manifests/ca1m_native_b6_train100_v1/subset_manifest.json` (100 official
training scene IDs, zero intersection with the 107 official validation IDs).
Future processed inputs belong under:

```text
/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1/<scene_id>/
```

The collection has two frozen phases:

1. CuTR runs live at real `score_thresh=0.4` and records post-filter proposals
   into the independent immutable cache namespace
   `ca1m-native-b6-train100-score04-gap20-cutr-v1`.
2. The exact cache is replayed through Selective Boxer G0, followed by the
   observer-only CA-1M native-B6 final-OBB feature collector. Predictions and
   same-run anchors must be byte-identical.

Static preflight (safe now, even while all 100 processed scenes are absent):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/collect_ca1m_native_b6_train100.sh --preflight 0,1
```

Only after the processed train root is complete:

```bash
bash scripts/collect_ca1m_native_b6_train100.sh 0,1
```

The run is resume-safe at scene boundaries. Every completed CuTR-record and
G0-observer scene has a create-only completion JSON containing artifact hashes;
partially promoted permanent artifacts fail closed instead of being overwritten.
The final ordered collection hash is written to
`reports/ca1m_native_b6_train100_v1/collection_manifest.json`.

The current CA1M loader consumes a scene-static `K_depth.txt`. If optional
per-frame K sidecars exist and differ from that static K, the input audit rejects
the scene. It never silently averages varying per-frame intrinsics. A later
loader change can add true per-frame support as a separate audited modification.

The full input audit reads RGB, depth, camera-to-world poses, gravity, and
intrinsics. It deliberately does not open `after_filter_boxes.npy`; GT joining
and model fitting are separate stages.
