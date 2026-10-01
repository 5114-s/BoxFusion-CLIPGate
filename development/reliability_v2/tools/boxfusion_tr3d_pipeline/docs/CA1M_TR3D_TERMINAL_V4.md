# CA-1M terminal TR3D v4: split proposal and overlay

The v4 route separates expensive TR3D inference from anchor-dependent
association.

## P: anchor-free proposal cache

For each frozen CA train100 scene, P reads only processed `rgb/*.png`,
`depth/*.png`, `all_poses.npy`, and depth intrinsics.  It derives the exact
reachable final-base lineage by simulating `demo.py`: after incrementing the
frame counter, the loop finalizes when `count == N-1` or
`count + 20 > N-1`.  Thus `N=326` yields `0,20,...,300`; the apparent last-frame
branch is unreachable.  P then builds a world-frame XYZRGB cloud and runs the
CA-1M scratch-trained class-agnostic
TR3D checkpoint sealed by the v3 checkpoint binding.

P has no anchor or B6 argument and does not import an evaluator.  Each output
is a read-only, create-only
`boxfusion.ca1m_tr3d_anchor_free_proposal_cache.v4` NPZ.  A rerun validates a
complete existing cache and skips it; malformed or partial artifacts fail
instead of being overwritten.  The cache contains candidate geometry,
scores, point support, local/world transform, exact frame IDs, and hashes for
the point cloud, code, checkpoint binding, checkpoint, and config.  These
fields are sufficient for a later overlay without rerunning TR3D.

The lineage simulator was checked against all 100 sealed old native-B6
`used_frame_ids` arrays as a one-time protocol oracle: 100/100 match, 3,010
reachable keyframes in 61,189 frames.  The old diagnostics are not a proposal
runtime input.  P's raw-resolution RGB/depth backprojection was also rebuilt
for all 100 scenes and compared with the exact float32 `.bin` arrays used to
train CA-native TR3D: 100/100 array and byte identity over 24,382,287 points.
The training converter and P do not pre-resize or reorient these inputs; the
shared backprojection performs nearest-neighbour RGB lookup at sampled depth
coordinates.  The immutable proof is sealed in
`manifests/ca1m_tr3d_terminal_ca_native_train100_v4/lineage_training_point_parity_v2.json`.

The v3 path is allowed only as the immutable CA-only checkpoint binding.  No
v1/v2/v3 terminal cache is an input.

## O: CPU-only final-base/B6-v2 overlay

O will run only after all of these are sealed together:

1. exact train100 G0 + CLIP + reliable-TopK3 final-base predictions and
   identity manifest;
2. final-base native-B6 v2 diagnostics, per-scene completion receipts, and
   exact100 collection manifest;
3. an activation-authorized B6 v2 checkpoint trained from that collection,
   plus its manifest.

O recomputes the B6 v2 active anchor scores and associates immutable P
candidates on CPU.  Its output is observer evidence only; it does not mutate
predictions or activate the old ScanNet terminal heuristic.

## Current state

Static validation is available without GPU work:

```bash
bash scripts/collect_ca1m_tr3d_terminal_train100_v4.sh --static-preflight
```

P now has an immutable stage-only authorization receipt.  When explicitly
requested later, this command first validates the receipt/code/checkpoint and
then recomputes all pending scene point hashes on CPU before a GPU worker can
start:

```bash
bash scripts/collect_ca1m_tr3d_terminal_train100_v4.sh --run-proposals
```

This task did not invoke that command.  All final-anchor/B6-v2 bindings remain
null, so the full route must still fail before starting a worker:

```bash
bash scripts/collect_ca1m_tr3d_terminal_train100_v4.sh --run
```

No official validation prediction, validation annotation, evaluator, legacy
ScanNet TR3D checkpoint, old terminal cache, or old B6 artifact is part of
either stage.
