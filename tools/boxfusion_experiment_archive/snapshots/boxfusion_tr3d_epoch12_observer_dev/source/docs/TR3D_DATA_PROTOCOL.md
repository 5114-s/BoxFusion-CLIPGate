# Genuine class-agnostic TR3D data protocol

## Frozen split

`prepare_tr3d_scannet_data.sh` deterministically partitions the 1,201 official
ScanNet train scenes using SHA-256 and the seed
`boxfusion-genuine-tr3d-v1`:

| partition | scenes | permitted use |
|---|---:|---|
| train | 1,001 | gradient updates |
| calibration | 100 | checkpoint/threshold selection |
| audit | 100 | frozen observer audit |
| official val | 312 | final evaluation only |

The command refuses any official-val scene in train/calibration/audit. Exact
scene-list hashes and paths are recorded in
`data/tr3d_scannet/DATASET_CONTRACT.json`.

Prepare links and one-class annotation files:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev
bash scripts/prepare_tr3d_scannet_data.sh
```

The existing `/extra/ZhaoX/scannet_data` arrays remain read-only. The prepared
root uses `points/full` and mask directory symlinks plus new annotation
pickles whose 18 detection labels are all mapped to foreground `0`.

## Coordinate frame contract

- Full ScanNet points and exported trajectory-prefix points are stored as
  float32 `N x 6` XYZRGB in `world_unaligned`.
- ScanNet detection boxes in annotation info are already
  `scannet_axis_aligned`.
- Every info row retains its 4x4 `axis_align_matrix`.
- The inherited official pipeline applies `GlobalAlignment` exactly once.
- TR3D predictions are therefore aligned. Before BoxFusion association, the
  inference adapter must transform box corners with the inverse axis-alignment
  matrix and refit the world-unaligned AABB. Never compare aligned TR3D boxes
  directly with world-unaligned B6 boxes.

## Trajectory-prefix training data

First inspect the schedule without decoding RGB-D:

```bash
bash scripts/export_tr3d_prefix_train.sh --manifest-only
```

Export points and filtered annotations:

```bash
bash scripts/export_tr3d_prefix_train.sh
```

For a cheap real-data smoke test:

```bash
bash scripts/export_tr3d_prefix_train.sh \
  --max-scenes 1 \
  --fractions 0.25 1.0 \
  --frame-stride 50 \
  --pixel-stride 8 \
  --output-info-name scannet_infos_prefix_smoke_foreground.pkl \
  --manifest-name trajectory_prefix_smoke.jsonl
```

Default export uses 25/50/75/100% temporal prefixes, every 25th RGB-D frame,
1 cm world voxels, and RGB projected using the ScanNet depth/color
calibrations.

Full-scene boxes are **not** copied blindly into a prefix. Prefix XYZ is first
axis-aligned and each aligned GT must contain at least 20 observed points.
`--min-visibility-fraction` may additionally require an observed/full support
ratio. Each decision and support count is recorded in the prefix JSONL
manifest.

## Configs

- `config/tr3d/tr3d_scannet_foreground.py`: full-scene one-class training;
  calibration and audit remain train-only.
- `config/tr3d/tr3d_scannet_foreground_prefix.py`: optional prefix
  fine-tuning after prefix export.
- `config/tr3d/tr3d_scannet_foreground_official_val.py`: final-only official
  validation.

The main config inherits the pinned official TR3D model and ScanNet dataset
bases by source-relative paths. It uses `TR3DClassAgnosticHead` so a collapsed
foreground label does not disable one TR3D feature level, and
`TR3DForegroundScanNetDataset` so MMDetection3D does not reinterpret
`foreground` as an invalid named subset of the original 18 classes.
