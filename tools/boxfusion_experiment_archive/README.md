# BoxFusion historical experiment archive

This archive preserves the source modifications from **all 30** historical
`/data/ZhaoX/OVM3D-Dett/boxfusion*` experiment directories.

Each route is stored as a complete, directly browseable **source-only** snapshot
under `snapshots/<route>/source/`. It is not a 40 GB copy of the experiment
workspace: datasets, model weights, Conda environments, caches, predictions,
diagnostics and logs remain external.

## Coverage

- early RGB/CLIP and Top-K reproduction;
- B3 memory, B5 refiner, B6 quality and joint B3/B5/B6 routes;
- YOLOE/SAM3 MaskGraph, TriFusion and YIDU A-series routes;
- P1/P2 residual-proposal families;
- Boxer, uncertainty, Selective Boxer and SGCDet refiners;
- trained TR3D, R2/R3, SMOV, SPGroup3D and residual-track C1/C2/C3 routes;
- the two original Stage-2 patch-only directories.

The authoritative list is [CATALOG.json](CATALOG.json). Every snapshot contains:

- `MANIFEST.json`: path, byte size, executable mode and SHA256 for every file;
- `EXCLUDED.json`: excluded runtime roots and recorded external symlinks;
- `source/`: the original project-relative source layout.

Capture performs two independent source scans around the copy. A route fails if
its code changes while being archived.

## Verify

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_experiment_archive
/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  scripts/verify_archive.py --syntax
```

This checks all hashes and modes, compiles every Python source in memory and
runs `bash -n` on every shell script without importing heavyweight models.
Syntax checking requires Python 3.10 or newer; Python 3.8 cannot parse syntax
legitimately used by several preserved routes.

## Browse, compare and materialize

```bash
# Compare two historical implementations.
python scripts/diff_snapshots.py boxfusion_b6_dev boxfusion_b3_dev

# Create a new source-only working copy. The output must not already exist.
python scripts/materialize_snapshot.py boxfusion_maskgraph_dev \
  "$PWD/materialized/maskgraph"
```

A materialized snapshot still needs the external assets recorded in its
configuration and `EXCLUDED.json`.

Historical files intentionally retain their original `/data/ZhaoX` and
`/home/admin1` paths so the archive remains byte-faithful. Override or rewrite
those paths in a separate materialized working copy; do not edit the snapshot.

## Safety boundary

This migration is deliberately non-destructive. The original OVM3D-Dett
directories are retained because active/full100 jobs and many absolute symlinks
still depend on them. Deleting those directories is not part of this archive.

Third-party repositories are represented by pinned metadata and local patches
under `vendors/` and `vendor_patches/`; their full source and checkpoints are
not duplicated here.
