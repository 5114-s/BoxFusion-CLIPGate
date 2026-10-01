# TR3D residual observer contract

This checkout uses an independent `tr3d_residual` namespace. It does not
reuse the historical P1/P2 cache or diagnostic schema.

Each immutable cache file is stored at:

```text
<cache_root>/<scene_id>/<prefix_id>.npz
```

Its schema is `boxfusion.tr3d_residual_cache.v1`. The authoritative geometry
is `corners_world[N,8,3]` in unaligned ScanNet world coordinates. Corner order
has no semantic meaning; consumers may use only min/max or a convex hull.
`boxes_world[N,7]` preserves the center/size/yaw tensor for provenance.
`aligned_to_unaligned[4,4]` and its canonical SHA256 bind the inverse
`axisAlignment` used during export.

The following fields are hard safety invariants:

```text
observer_only = true
mutation_enabled = false
applied_count = 0
class_agnostic = true
labels_3d[:] = 0
coordinate_frame = scannet_unaligned_world
```

Cache creation is exclusive and refuses to overwrite an existing file.
Loading rejects unknown fields, object arrays, malformed dtypes, invalid
geometry, duplicate proposal IDs, or model/config provenance mismatches.

The frozen B6 reference is content-addressed by
`manifests/frozen_b6_full100.json`, including all 100 prediction SHA256s,
checkpoint SHA256, validation-list SHA256 and a tree hash. The union-oracle
audit verifies that manifest both before and after reading TR3D candidates.
It does not copy, reorder, rescore, or modify B6 predictions.

Use:

```bash
python tools/validate_tr3d_residual_cache.py \
  --cache-root /path/to/cache \
  --scene-list /path/to/scenes.txt \
  --checkpoint-sha256 "$TR3D_CHECKPOINT_SHA256" \
  --config-sha256 "$TR3D_CONFIG_SHA256"

python tools/audit_tr3d_residual_observer.py \
  --cache-root /path/to/cache \
  --scene-list /path/to/scenes.txt \
  --checkpoint-sha256 "$TR3D_CHECKPOINT_SHA256" \
  --config-sha256 "$TR3D_CONFIG_SHA256" \
  --report reports/tr3d_residual/audit.json
```

## Official inference export

Do not call MMDetection3D's generic `inference_detector` for these files: it
injects an identity `axis_align_matrix`, while this experiment stores ScanNet
XYZ in the unaligned world frame. The official adapter in
`boxfusion/tr3d_inference.py` passes the real matrix through the inherited test
pipeline so `GlobalAlignment` runs exactly once.

Run a no-write synthetic conversion check:

```bash
python tools/run_tr3d_cache_inference.py \
  --config config/tr3d/tr3d_scannet_foreground_official_val.py \
  --checkpoint /path/to/checkpoint.pth \
  --cache-root /tmp/tr3d-cache-dry-run \
  --input-manifest data/tr3d_scannet/scene_manifest.jsonl \
  --scene-list evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt \
  --prefix-id full \
  --dry-run-synthetic
```

Run genuine one-class TR3D on one or two GPUs:

```bash
BOXFUSION_TR3D_CHECKPOINT=/path/to/frozen_one_class_tr3d.pth \
bash scripts/run_tr3d_cache_inference.sh 0,1
```

The launcher shards samples deterministically and uses validated resume.
Existing cache files are loaded and checked against the point-file,
checkpoint, and config hashes; they are never overwritten. The exporter
refuses an 18-class checkpoint/output. The published 18-class weight may be
used for a separate environment smoke test, but it cannot populate this
class-agnostic residual cache.
