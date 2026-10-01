# CA-1M-native B6 train-scene builder

This route builds one **train-only** CA-1M scene at a time under the isolated
root `/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1`.  It never reads or
writes the CA-1M validation/live root, never consumes `after_filter_boxes.npy`
from validation, and has no batch mode.

The source must be an official `ca1m-train-ID.tar` selected by the immutable
`ca1m_native_b6_train100_v1` manifest.  The builder fails closed if the scene
appears in Apple's validation URL list.  It reconstructs BoxFusion-readable
`rgb/`, `depth/`, `all_poses.npy`, `T_gravity.npy`, `K_depth.txt`, and
`K_rgb.txt`.  Raw per-frame RGB/depth intrinsics, raw frame IDs, cardinal image
rotations, shapes, and the compatibility intrinsics are preserved in
`per_frame_intrinsics.npz` plus a JSON policy manifest.  The loader-facing
`K_depth_per_frame.npy` sidecar contains the per-frame depth K normalized to
the processed unified image orientation; the legacy scene-mean
`K_depth.txt` remains available for backward compatibility.

The label is explicitly derived rather than author-published validation GT:

1. read `world.gt/instances.json` from the train tar;
2. back-project real depth with stride 4 and `depth < 10 m`;
3. retain the first point in each 0.02 m voxel;
4. apply the author's six-visible-corner frustum rule;
5. retain a box only when at least four corners have strict nearest-surface
   distance `< 0.10 m`.

Both frustum projection and depth back-projection use the processed **per-frame
K**, never the legacy scene mean.

The primary name is `derived_train_gt_boxes.npy`.  A byte-identical hard link
named `after_filter_boxes.npy` is provided only for BoxFusion dataset-loader
compatibility.  `derived_train_gt_manifest.json` records source and artifact
SHA256 hashes, frame mapping, parameters, frustum/kept indices, and the frozen
train/validation non-overlap contract.

Artifact immutability is enforced by create-only paths, regular-file/no-symlink
checks, and frozen SHA256 values.  On POSIX filesystems the auditor additionally
requires all write bits to be absent.  `/extra` is currently a `fuseblk` volume
that reports mode `0777` even after a successful `chmod`; the audit records this
filesystem limitation instead of pretending those mode bits are meaningful.

Preflight (default, no output):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/build_ca1m_native_b6_train_scene.sh --preflight 48018894
```

One explicit scene build and full audit:

```bash
bash scripts/build_ca1m_native_b6_train_scene.sh --single-scene 48018894
```

There is intentionally no command here for converting all 100 scenes.  Batch
conversion should be authorized only after the single-scene artifact and label
statistics have been reviewed.

## Resume-safe exact100 driver

The exact100 driver is now available, but it never starts automatically. Its
default mode only verifies the frozen 100-scene contract and local tar
readiness:

```bash
bash scripts/build_ca1m_native_b6_train100.sh --preflight
```

After all frozen tars have downloaded and the single-scene result has been
accepted, explicit execution is:

```bash
bash scripts/build_ca1m_native_b6_train100.sh --run
```

Before any build, the driver requires exactly 100 complete canonical train
tars. Readiness is fixed-cost: canonical filename, regular/no-symlink file,
512-byte alignment, matching first-member scene prefix, and the canonical two
zero-block tar terminator. The official downloader already runs full `tar -tf`
before atomically renaming `.part` to `.tar`; each single-scene builder again
validates all members and requires `world.gt` before publishing. A single
non-blocking process lock prevents concurrent drivers. For every
frozen ID, an existing numeric scene receives a fresh full geometry audit and
is skipped; only an absent numeric scene is built and then fully audited.
`.part`, `.building`, `.failed`, and quarantine paths remain untouched and do
not count as completion. Success requires the numeric scene directories to be
exactly the frozen 100 and writes `exact100_completion.json` plus
`latest_run.json` under `reports/ca1m_native_b6_train100_v1`.
