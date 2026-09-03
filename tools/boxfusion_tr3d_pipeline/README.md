# BoxFusion TR3D pipeline migration

This directory is a source-only snapshot of the cumulative BoxFusion work that
was developed in:

`/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_residual_track_dev`

It lives below the main BoxFusion repository so the implementation can be
versioned together with BoxFusion without overwriting the original
`boxfusion/`, `tools/`, `scripts/`, or `config/` packages.

## Migration status

- Source snapshot date: 2026-08-10 (Asia/Shanghai)
- Parent OVM3D-Dett revision: `374029e87ba82420387d5e6ef6fc3ceea934f0fc`
- Migration mode: non-destructive copy
- Original source is intentionally retained while the full-100 C3 run is active
- Large model weights, datasets, caches, predictions, diagnostics, and logs are
  not included; the 17 KiB hash-frozen B6 quality calibrator is included so its
  regression contract remains testable

The original repository README is preserved as `UPSTREAM_README.md`.

## Included implementation

The snapshot preserves the project-relative layout needed by the launchers:

- score-preserving CLIP appearance gate and Top-K view fusion;
- B6 quality calibration and online refinement infrastructure;
- Selective Boxer lifting (G0);
- terminal TR3D residual proposals;
- SMOV-style depth/free-space verification;
- SPGroup3D observer adapters;
- C1/C2/C3 residual-track, Mask-RGBD, identity and shadow tooling;
- training-free Moon-QIM/PUF arbitration observers and the bounded
  MV3DIS-Depth-Lite S0 real-stream diagnostic;
- deterministic ScanNet evaluator, audit tools, scripts, tests, and docs;
- the autograd-safe Boxer `AleHead` overlay and its standalone patch.

`external_overlays/` contains only modified third-party source. It does not
contain the Boxer repository or checkpoint.

## Deliberately excluded

The following stay outside Git because they are large, generated, private, or
currently being written:

- large checkpoints, ScanNet RGB-D frames, and ground truth;
- `artifacts/`, `cache/`, `diagnostics/`, `logs/`, `reports/`, and `results/`;
- CuTR/YOLOE replay caches and frozen prediction pickles;
- Conda environments and complete third-party repositories;
- historical failed experiment clones in sibling `boxfusion_*_dev` folders.

Those paths remain configurable external inputs. Copy `.env.example` to `.env`,
edit it for the machine, and load it before running. Boxer checkout/checkpoint
paths live under `lifting.boxer` in `config/scannet_b6_selective_boxer.yaml`;
use a machine-local YAML through `BOXFUSION_ONLINE_CONFIG` when relocating
them.

## Use this tree

Always change into this directory first. This makes the inner `tools` package
resolve to this snapshot instead of `/data/ZhaoX/BoxFusion/tools`.

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
cp .env.example .env
# Edit .env if the external assets were moved.
set -a
source .env
set +a
python tools/verify_migration.py
```

One-scene C3 identity smoke test:

```bash
BOXFUSION_C3_ONLINE_RUN_TAG=migrated_c3_smoke_v1 \
  bash scripts/run_scannet_tr3d_c3_online_identity.sh 0
```

Do not start this while the existing full-100 job is using the same GPUs or
artifact tag. Choose a fresh run tag and artifact root.

## External assets still required

The pipeline uses the main BoxFusion repository for CuTR/CLIP weights, class
features, ScanNet frames, and evaluation GT. The B6 quality calibrator is
included; YOLOE, terminal-TR3D and Boxer assets remain external. Their exact
variables are documented in `.env.example`.

The Boxer source change can be applied to an official Boxer checkout with:

```bash
git -C /path/to/boxer apply \
  "$PWD/external_overlays/third_party/boxer/boxernet/alehead_autograd_safe.patch"
```

## Final cut-over

This snapshot does not delete its source. After the running full-100 experiment
finishes, run the verifier again, perform one final source-to-destination sync,
and only then consider archiving/removing the old directory. Deletion is a
separate destructive operation and is not part of this migration.
