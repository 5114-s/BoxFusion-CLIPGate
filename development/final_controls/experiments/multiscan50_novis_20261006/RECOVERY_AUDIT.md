# MultiScan-50 recovery audit

On 2026-10-07, repeated runs stopped at `scene9005_00` for two reproducible
reasons recorded in the archived scene logs:

1. the RGB-D iterator stepped beyond an aligned modality list at scene end;
2. the unrelated PVQ-AR shadow observer aborted when a keyframe had no
   `appearance_features` field.

The recovery keeps the ReCaR evaluation route unchanged:

- the stream now iterates over the common RGB/depth/pose prefix;
- PVQ-AR shadow diagnostics are disabled because they are not part of the
  evaluated ReCaR method and do not alter predictions in shadow mode;
- completed scenes `scene9000_00` through `scene9004_00` are reused;
- all partial `scene9005_00` artifacts are archived under `interrupted/`;
- the corrected run restarted at 2026-10-07 14:11:39 Asia/Shanghai.

The queued FPS reproduction remains blocked on the submission queue `DONE`
marker and therefore does not compete for the GPUs during this run.
