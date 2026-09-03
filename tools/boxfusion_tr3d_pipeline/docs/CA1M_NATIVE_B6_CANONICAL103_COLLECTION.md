# CA-1M canonical103 native-B6 observer collection

This route is an isolated, ground-truth-free collection pass for the public
canonical 103-scene CA-1M validation subset.  It does not reuse the historical
C0 prediction as a baseline and does not reuse the fixed-ten proposal cache.

The frozen sequence is:

1. live CuTR, detector score threshold 0.4, frame gap 20;
2. create an immutable cache in
   `ca1m-native-b6-canonical103-score04-gap20-cutr-v1`;
3. replay the exact cached CuTR proposals;
4. apply Selective Boxer G0 (`center <= 0.10 m`, volume ratio `[0.50, 2.00]`);
5. observe every final OBB with the 14-D CA-1M native-B6 feature extractor;
6. save a same-process pre-observer anchor and require byte identity with the
   post-observer prediction.

The runner never imports or invokes `eval_ca1m.py`.  It opens RGB, depth,
intrinsics, poses, and gravity only.  `eval: true` in the two templates is the
existing `demo.py` switch for serializing a prediction, not an evaluator call.

## Commands

Static, non-GPU preflight (the default):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/collect_ca1m_native_b6_canonical103.sh --preflight
```

Future dual-GPU collection, explicitly authorized by `--run`:

```bash
bash scripts/collect_ca1m_native_b6_canonical103.sh --run 0,1
```

The run is resumable only from fully sealed per-scene artifacts.  A partial
permanent scene is rejected rather than overwritten.  Staged output is moved
into the formal namespace only after validation, then made read-only.

Primary output:

```text
reports/ca1m_port/ca1m_c3_native_b6_observer_canonical103_v1/identity_audit.json
```

## Limits

- This collection produces no AP result.
- It cannot be used to train, calibrate, tune a threshold, or authorize an
  active validation result; training must use the separately frozen train100
  route.
- Reported FPS is cache-assisted observer-path throughput, not live-CuTR
  end-to-end throughput.
- The four validation scenes without the public canonical filtered target are
  intentionally outside this 103-scene list, but no target file is read here.
