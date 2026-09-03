# Frozen CLIP vocabulary gate: isolated warm runtime

This is a runtime-only, no-GT/no-evaluator benchmark on physical GPU 1
(`NVIDIA GeForce RTX 3090`).  It replays all 62 valid shadow receipts (186
actual OWL crops) from `CLIP_VOCAB_SHADOW_FULL100.json`; every timed GPU call
contains exactly the three historical crops of one real receipt.

Command:

```bash
conda run -n boxfusion2 python tools/benchmark_scannet_clip_vocab_gate_runtime.py \
  --device cuda:1 --warmup 10 --repeats 100 \
  --output reports/clip_vocab_gate_runtime/RUNTIME_GPU1.json
```

## Results

| Timed unit (batch of 3 real crops) | p50 | p95 | max | samples |
|---|---:|---:|---:|---:|
| RGB disk decode + 960 resize + crop + CLIP preprocess | 37.82 ms | 45.13 ms | 48.07 ms | 62 |
| Crop + CLIP preprocess from already-resized RGB | 2.95 ms | 5.19 ms | 6.40 ms | 62 |
| ViT-H image encode + normalize + 473-way cosine score | 68.76 ms | 69.76 ms | 70.30 ms | 100 |

The native ViT-H checkpoint plus cached text features took 7.33 s to load once.
That cold-start cost is deliberately excluded from warm timings and must not be
paid for each receipt.  The GPU number also excludes host-to-device transfer;
the three preprocessed tensors, model, and 473 text features were held on GPU.
Inference used the sidecar's native float32 path without autocast.

With RGB already available/resized in the online pipeline, the isolated warm
gate is approximately 71.7 ms at the median (2.95 + 68.76 ms), or about 13.9
receipt gates/s if run serially.  Reading all three RGB files from disk raises
this rough serial sum to 106.6 ms.  These rates are **not end-to-end FPS**:
they exclude OWLv2/Boxer proposal generation, lifting/tracking, native
BoxFusion, synchronization with the RGB-D stream, and resource contention.
The gate is event-driven for confirmed birth receipts rather than a per-frame
operation, so a true online FPS claim requires an integrated replay benchmark.

Machine-readable results: `RUNTIME_GPU1.json`.
