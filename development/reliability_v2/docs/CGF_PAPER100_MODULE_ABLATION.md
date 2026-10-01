# CGF-paper-100 training-free online ablation

## Frozen protocol

- ScanNet official 100-scene list, with the same scene order for every arm.
- Frozen CuTR / Cubify Anything and frozen CLIP weights, vocabulary, and embeddings.
- `detection.score_thresh: 0.4`, `data.gap: 25`. The released inference code's
  unchanged first-frame safeguard retries at `0.4 / 4 = 0.1` only when frame 0
  has no surviving proposal; this behavior is identical in every arm.
- Official class-agnostic evaluator; prediction confidence is replaced by `1.0` in memory.
- No target-dataset training, fine-tuning, calibration, future-frame access, or ground-truth access.
- A module is retained only after AP, paired-scene uncertainty, runtime, causality, and bounded-memory checks.

## Factorial control before adding a new module

| Arm | Appearance gate | Reliable-view Top-K | Status |
| --- | --- | --- | --- |
| P0 | off | off | same-root full100 complete |
| P1 | on | off | full100 complete |
| T | off | on | same-root full100 running |
| P2 | on | on | full100 complete |

Official constant-score results already available:

| Arm | AP15 | AP25 | AP50 | boxes |
| --- | ---: | ---: | ---: | ---: |
| P0 | 28.2679 | 22.5423 | 5.9346 | 2279 |
| P1 | 28.0884 | 21.9982 | 6.1744 | 2331 |
| P2 | 28.6490 | 23.0029 | 6.8098 | 2343 |
| P2 - P1 | +0.5606 | +1.0047 | +0.6354 | +12 |

Paired scene bootstrap (10,000 resamples) for P2 - P1:

| Metric | delta (AP points) | 95% interval | positive resamples |
| --- | ---: | ---: | ---: |
| AP15 | +0.5606 | [-0.4202, 1.5002] | 84.90% |
| AP25 | +1.0047 | [0.0014, 1.8461] | 97.51% |
| AP50 | +0.6354 | [0.1983, 1.2822] | 99.60% |

Current conclusion: Reliable-view Top-K has the clearest evidence at AP50, positive but weaker evidence at AP25, and insufficient evidence for a robust AP15 gain. The P0/T arms are required to estimate its effect without the appearance-gate interaction.

The module decision is frozen before P0/T results are available:

- Judge Reliable-view Top-K only from `T - P0`; `P2 - P1` is a conditional
  replication, not a substitute for the direct effect on the released base.
- Judge the appearance gate only from `P1 - P0`.
- A module passes accuracy only when AP25 and AP50 are positive, AP15 is at
  least -0.10 AP, the mean of AP15/AP25/AP50 improves by at least +0.20 AP,
  and, across AP25/AP50, at least one paired-bootstrap 95% lower bound is
  nonnegative while the other has at least 90% positive resamples.
- Start from P0. Retain each module that independently passes accuracy,
  causality, bounded-state, and runtime gates. Use P2 only if both modules
  independently pass and their interaction does not make P2 fail the same
  gates versus P0. Prefer the simpler arm when evidence is inconclusive.

Completed-arm runtime:

| Arm | mean scene FPS | frame-weighted FPS |
| --- | ---: | ---: |
| P0 | 14.1314 | 12.8699 |
| P1 | 14.1604 | 12.0890 |
| P2 | 13.9116 | 12.0548 |

For paired scenes, the P2/P1 FPS ratio has mean 0.9883 and median 0.9990;
the frame-weighted throughput ratio is 0.9972. Thus the measured incremental
Top-K overhead is small on the completed gate-on pair. T versus P0 will test
the same claim without the appearance gate.

Across the completed full runs, frame-weighted P1/P0 and P2/P0 throughput
ratios are 0.9393 and 0.9367. These are below the frozen 0.95 retention gate;
the four-arm paired analysis is still required before attributing the slowdown,
and T/P0 remains the decisive Top-K runtime contrast.

## Ordered module tests

1. Complete P0/T and choose the stronger causally valid base arm.
2. SMOV-lite fragment shadow: depth-edge cleanup and bounded 5 cm voxel fragments only; native predictions must remain unchanged.
3. Group3D-lite unmatched-only secondary association:
   - base versus Group3D with raw fragments estimates the association effect;
   - Group3D with cleaned fragments versus raw fragments estimates the isolated SMOV cleanup effect.
4. MV3DIS-lite past/current depth-consistency weighting.
5. OnlineAnySeg-lite bounded voxel-hash object memory.
6. Zoo3D0 past-only observer/supporter novelty gate for new proposals.
7. Append only high-precision gated births; the official constant-score evaluator does not preserve a low-score suffix.
8. Keep native CLIP classes, vocabulary, and embeddings unchanged.

## Stage gates

- Shadow isolation: native prediction count, values, and serialized result hash unchanged.
- Accuracy: report AP15/AP25/AP50 and paired-scene intervals against the immediately preceding arm.
- Online causality: current and past frames only; terminal handling cannot replay a stale proposal as a new observation.
- Resource bounds: fixed caps for proposals, views, points, tracks, and pending observations.
- Runtime: full-pipeline FPS ratio at least 0.95 against the paired base, plus per-module latency statistics.
- Training-free addition: fixed geometry rules only; no learned weights introduced by these added modules.

## Frozen-proposal replay and numerical floor

All experiments after the gate x Top-K selection use one new v3 proposal
cache. The producer run is not an accuracy control. E2 replay 1, E2 replay 2,
SMOV shadow, Group3D raw, and Group3D clean must all consume the same cache;
replay errors are fatal and never fall back to live CuTR.

- Per-frame CuTR rows, order, dtype, shape, content digest, filtering attempt,
  and input signature must match exactly.
- Shadow callbacks must leave an in-process snapshot of native proposals and
  BoxFusion state bit-exact.
- Independent GPU runs are judged against an E2 replay-repeatability floor,
  not by final-pickle byte identity. Initial limits are p95 corner drift at
  most 1 mm, global maximum at most 15 mm, and each AP drift at most 0.05 AP.
- An active module's AP gain must exceed both 0.10 AP and twice the measured
  repeat-run AP noise before paired-bootstrap evidence is interpreted.
- Cache-replay FPS is not a realtime claim; every retained module is also
  profiled on the live frozen-CuTR path.

## SMOV-lite staging status

The audited portable core is staged in `boxfusion/smov_fragments.py` but is
not imported by inference and has no enabled configuration. Its source hash is
`9086634ac218da3bc81537fb4009c1f814dc23024f6fa021f5b654a23ddab55f`.
The repository test suite for this module passes 30/30 cases; an independent
12-case adversarial audit also passed configuration, cap, transaction, alias,
edge, registration, and pose checks.

This is a correctness-only staging approval, not a realtime approval. A
480x640 large-crop CPU stress test measured median prepare latency of about
12.60/87.35/353.72/695.40 ms for 1/8/32/64 proposals. The module therefore
remains disconnected until real-scene profiling and optimization or bounded
asynchronous execution demonstrate a paired full-pipeline FPS ratio of at
least 0.95. No AP claim is possible in observer-only shadow mode; output
identity is the required result.

A first barrier-prefix optimization candidate reduced the 8-proposal synthetic
median to about 21 ms, but independent audit rejected that candidate: masked
depth subtraction changed NaN/Inf failure semantics under strict NumPy error
handling, and 2048x2048 peak memory increased materially. It remains outside
the repository until exact error parity, memory review, and independent
regression tests pass.

The repaired candidate, SHA-256
`f2d3bd1e15ef0fea28bdb3ffad55c383f5ad50b490afeee5e84f3b9a66fe544c`,
restores the audited IEEE failure behavior and uses chunked adaptive-width
prefix arrays. It passed 48 release tests plus an independent 207-case
random/adversarial parity audit and about 3.74 million direct path checks. Its
independent 480x640 full-crop median was about 19.10 ms for eight proposals;
2048x2048 isolated peak RSS was about 109.73 MiB versus 113.09 MiB for the
audited oracle. This approves the extraction core only. It remains disconnected
until observer wiring, output non-interference, and live full-pipeline FPS pass.

The separate stable-track observer registry is also approved for staging after
25 regression tests and an independent adversarial audit. Disabled hooks do not
evaluate or consume native inputs; exceptions detach only the observer; aliases
target active canonical tracks; row-count handshakes cover both native reindex
sites; and literal-backed physical caps remain 1024 active rows, 4096 current
proposals, and 5120 temporary rows even under forced attribute/constant
rebinding. Root `BoxManager` remains unmodified until the P0/T run completes.
