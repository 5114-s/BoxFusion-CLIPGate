# Exact final no-prewarm paired efficiency benchmark

This benchmark starts after the cache-symmetric paired benchmark completes.
It compares the same frozen original BoxFusion and snapshotted final ReCaR-3D
sources without an explicit scene-cache prewarm.  The first arm alternates by
scene so the report can quantify order/cache bias.

This run is intended as a paper-like absolute-FPS sensitivity check.  The
cache-symmetric run remains the primary measurement of relative method
overhead.
