# Exact final paired efficiency benchmark

This benchmark compares the frozen released BoxFusion source at commit
`06ea629` against the snapshotted final ReCaR-3D route used by
`final_novis_calrv1_recar_fps_reproduction100_20261008`.

For every ScanNet validation scene, the two arms run in separate processes on
one RTX 3090.  Arm order alternates by scene.  The complete scene directory is
read immediately before each timed arm, outside the timing interval, so both
arms receive the same explicit page-cache treatment.

The ReCaR arm is locked to PLR-v1, CALR-v1 and three-factor MVSR
(`no_visibility`) with historical-max aggregation.  Proposal replay,
visualization and all audit/shadow variants are disabled.

Run:

```bash
bash development/final_exact_paired_efficiency_20261008/run_official100_prewarm.sh
```
