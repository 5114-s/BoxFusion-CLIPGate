# CA-1M final base anchor

The final base namespace is `ca1m_c4_final_base_g0_clip_topk3_fixed10_v1`.
It preserves the CA-1M score-0.4/gap-20 geometry contract and applies:

1. immutable CA CuTR proposal replay;
2. Selective Boxer G0 (`center <= 0.10 m`, volume ratio `[0.50, 2.00]`);
3. frozen OpenCLIP appearance-aware association;
4. deterministic reliable-view fusion with `Top-K = min_views = 3`.

The CLIP gate and reliable-view selector have no fitted parameters and read no
category or 3D ground-truth annotations.  `sensor_info.gt.depth.K` in the
legacy loader name is the observed depth-camera calibration bucket, not an
annotation input.  CA boxes remain world-space OBB corners / center-size plus
a separate rotation matrix; no ScanNet axis alignment is enabled.

No ScanNet learned B6, quality gate, or TR3D benefit gate is loaded.  The only
checkpoint inside the YAML is the generic frozen Boxer checkpoint; OpenCLIP
is supplied as a frozen runtime asset by the runner.

## Required order

Run the read-only checks first:

```bash
bash scripts/run_ca1m_c4_final_base_fixed10.sh --preflight 0,1
bash scripts/collect_ca1m_native_final_base_train100.sh --preflight 0,1
```

The explicit fixed10 run, when authorized, runs a G0-only control and the new
base from the same immutable proposal cache.  The new base finalizer creates
a hard-linked same-run identity prediction.  A no-GT audit must verify this
byte/semantic identity and report the G0-to-new-base geometry/score changes
before the paired evaluator is called:

```bash
bash scripts/run_ca1m_c4_final_base_fixed10.sh --run 0,1
```

Only after fixed10 passes should train100 collection be started:

```bash
bash scripts/collect_ca1m_native_final_base_train100.sh --run 0,1
```

The train100 runner reuses only the immutable CA train CuTR cache and writes
all new outputs under `ca1m_native_final_base_train100_v1`.  It never invokes
the evaluator or validation GT.  Its output is the source anchor for a new
native-B6 evidence collection and CA-train-only retraining; the older CA B6
checkpoint is not authorized on this changed anchor.
