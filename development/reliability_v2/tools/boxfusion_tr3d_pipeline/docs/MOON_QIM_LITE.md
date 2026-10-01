# Moon-QIM-lite causal observer

This first-stage experiment borrows only MoonSeg3R's sparse 3D-key query-memory
idea.  It adds no network, learned parameter, class score, or CLIP update.
Native BoxFusion remains the only association path.

## Causal contract

For each real CuTR keyframe:

1. CuTR boxes are transformed to metric world coordinates.
2. Every proposal queries an index committed through the previous keyframe.
3. Unmodified BoxFusion spatial and correspondence association runs.
4. The observer compares its shortlist with the native stable track ID.
5. Only then are current global tracks committed to the sparse index.

The legacy terminal-frame branch can reuse the preceding `pred_instances`.
QIM therefore runs only when `count % gap == 0`; a terminal replay is never
treated as a new observation.

The implementation is bounded by `max_tracks`, at most
`samples_per_axis ** 3 + 1` sampled keys per track (the extra key is the box
center and is often a duplicate), `max_postings_per_key`,
`max_candidates_per_query`, and `track_ttl_keyframes`.  It stores stable track
IDs rather than mutable global row indices.

## Observer metrics

The end-of-scene summary reports:

- native matched, birth, and unresolved proposal counts;
- QIM Recall@1, Recall@3, and Recall@K against native association;
- mean/max query and commit time;
- retained and maximum track/key/posting counts;
- explicit `training_free`, `causal`, and no-semantic-access contracts.

An unresolved proposal is excluded from recall.  This occurs when native NMS
removes its source ID without retaining enough identity information; the
observer does not modify BoxManager to manufacture a target.

## Run

Use `config/scannet_qim_observer.yaml` with a new output directory.  The
control is the same command with `config/scannet_eval.yaml`.  Both runs must
use the same model, seed, frame stream, device, and visualization setting.

The sequence runner accepts `--output-dir` and `--seed`, so a paired smoke can
be launched without editing either frozen config.

Unit tests:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/admin1/miniconda3/envs/boxfusion2/bin/python -m pytest -q \
  tests/test_moon_qim_lite.py
```

QIM remains observer-only until paired predictions are identical, raw
Recall@8 is at least 0.995, and the same-device median FPS ratio is at least
0.95.  The next active experiment will feed this shortlist to PUF-lite with an
exhaustive fallback; QIM alone is not expected to improve AP because native
BoxFusion already scans all global boxes.

That PUF-lite shadow is now implemented and documented in
`docs/PUF_LITE_SHADOW.md`.  Its first real-stream smoke passes identity,
probability, support, and runtime checks, but its same-track conflict rate is
still too high to authorize active association.

## Validated real-stream smoke

On `scene0277_00` with seed 0, GPU 1, real CuTR inference, and no proposal
cache, the warm control ran at 33.83 FPS and the observer at 33.68 FPS
(`0.9956x`).  The final eight boxes were byte-identical.  Across 47 native
history matches, Recall@1/3/8 was `0.8511/1.0000/1.0000`, with zero unresolved
targets.  End-to-end QIM wrapper work (including tensor copies, causal-ID
assignment, query, scoring, and commit) totaled 47.90 ms, or about 0.043 ms
per processed input frame.

The machine-readable paired audit is produced by
`tools/audit_moon_qim_paired.py`.  A cold first run is not used for the FPS
ratio because model loading, filesystem cache, and PyCUDA compilation dominate
it.
