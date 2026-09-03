# Frozen Boxer + unexplained-depth shadow/oracle preflight

Date: 2026-08-23

## Decision

Do **not** enable birth from the current hard unexplained-depth gate.  On the
sealed three-scene preflight, its incremental missed-object recall headroom is
21.4286 / 17.8571 / 3.5714 points at IoU 0.15 / 0.25 / 0.50.  The AP50
necessary condition for a +10-point gain therefore fails.

Keep the frozen OWLv2 + Boxer proposal source in shadow.  Before the depth
gate, its incremental missed-object recall headroom is 25.0000 / 21.4286 /
17.8571 points, so the universal proposal geometry itself has enough
class-agnostic coverage to justify testing a different past-only confirmation
mechanism.

This is a three-scene rejection/preflight result, not a full100 accuracy claim.

## Frozen protocol

- Native arm: T05 (`score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K3, frozen CuTR and CLIP).
- Formal score mode: every evaluated prediction has score 1.0.
- Scenes: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Universal proposal source: frozen OWLv2 with 1,220 released LVIS+ prompts,
  followed by frozen BoxerNet; no target-data training or optimizer.
- Proposal thresholds: 2D 0.25, per-class 2D NMS 0.50, 3D 0.50.
- Online schedule: the sealed T05 gap-25 schedule, with 66 / 112 / 30 valid
  keyframes.  Invalid-pose frame 1325 is an abstention; no future frame is
  substituted.
- The final v5 run forces `annotation_path=None` and guards against opening
  `full_annotations.json`.  DINOv3, OWLv2, its fixed LVIS+ text cache, Boxer,
  source, prompt and schedule hashes are sealed.
- Shadow only: no native box, score, order, CLIP category or embedding is
  changed; birth is false.

The depth gate uses the preregistered constants: stride 4, depth 0.10--8.00 m,
5 cm signed-floor voxels, native AABB expansion 5 cm, at least 16 candidate
voxels, at least 16 unexplained voxels and unexplained ratio at least 0.50.
Because it explains depth using final T05 boxes, this gate is an offline oracle
filter and is not itself deployable.

## Output and baseline checks

- GT objects: 28; native T05 predictions: 46.
- Raw per-view Boxer candidates: 3,085 (941 / 1,802 / 342).
- Depth-gated candidates: 624 (227 / 234 / 163).
- Terminal causal tracker candidates: 57 (24 / 32 / 1).
- Native prediction SHA-256 values before and after shadow/oracle are identical
  for all three scenes.
- v5 raw, 2D and tracked CSVs are byte-identical to v4, while v5 supplies the
  strict no-GT access evidence missing from v4.

## Geometry and AP oracle

| Pool | Metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---|---:|---:|---:|
| T05 | Official AP, score=1.0 | 27.7670 | 27.7670 | 16.2060 |
| T05 | Native maximum matching | 18 | 18 | 14 |
| Raw Boxer | Native-union maximum matching | 25 | 24 | 19 |
| Raw Boxer | Additional recoverable GT | 7 | 6 | 5 |
| Raw Boxer | Incremental recall headroom, points | **25.0000** | **21.4286** | **17.8571** |
| Raw Boxer | GT-selected fixed-suffix delta AP | +12.5744 | +14.4172 | +9.9254 |
| Depth-gated | Native-union maximum matching | 24 | 23 | 15 |
| Depth-gated | Additional recoverable GT | 6 | 5 | 1 |
| Depth-gated | Incremental recall headroom, points | **21.4286** | **17.8571** | **3.5714** |
| Depth-gated | GT-selected fixed-suffix delta AP | +13.5623 | +9.6279 | +2.0070 |
| Terminal tracks | Additional recoverable GT | 1 | 0 | 2 |
| Terminal tracks | Incremental recall headroom, points | 3.5714 | 0.0000 | 7.1429 |

“Incremental recall headroom” is the strict maximum-matching difference
between `native union pool` and native geometry, divided by 28 GT objects.  It
is the relevant necessary ceiling for gains coming from recovered misses.

The GT-selected suffix is only a constructive counterfactual.  It is not a
mathematical AP upper bound.  With every score fixed to 1.0, NumPy's default
unstable tie sort can change the global evaluation order when a suffix changes
length; AP deltas therefore contain a tie-order component.  Geometry
maximum-matching counts are the primary diagnostic for this stage.

## Realtime observation

The frozen proposal branch is causal and processes only past/current frames.
In the conservative v4 timing pass, excluding one warm-up frame per scene, the
aggregate keyframe latency was 192.0 ms median and 345.8 ms p95.  A gap-25
branch on a 30 Hz stream has an 833.3 ms budget; 204/205 keyframes met it.  The
v5 warm-cache pass was faster and all keyframes met the budget.

This establishes isolated asynchronous throughput only.  Co-running the
branch with native CuTR/BoxFusion on one GPU has not yet been measured, so the
complete integrated system is not yet proven realtime.

## Artifacts

- Preregistration: `docs/UNEXPLAINED_DEPTH_BOXER_ORACLE_PREREGISTRATION.md`
- Launcher: `scripts/run_scannet_boxer_unexplained_shadow_preflight.sh`
- Sealer: `tools/seal_boxer_shadow_candidates.py`
- Oracle: `tools/audit_scannet_boxer_unexplained_oracle.py`
- Sealed candidate JSON SHA-256:
  `84eb4f2c62d1573d9e9f1ec4c3df5a6cac16ad10c8cece0989d37dd97b734e9e`
- Sealed candidate NPZ SHA-256:
  `c1a921d70de447bf528711a71deb34cf93a9bf671d3514baafa42b7b1b8b4a6c`
- Final oracle JSON SHA-256:
  `576f18af67484393544f4937fd6d99d1a2fce40fe3d308d79d074c2212b4db8e`

## Recommended next experiment

Do not tune the 0.50 depth-ratio threshold on these three GT-inspected scenes
and do not enable active birth.  The next output-inert experiment should keep
the raw frozen Boxer proposals, add causal past-only multi-view association and
support confirmation, and use unexplained depth only as a soft reliability
feature or veto.  Measure candidate survival, false-positive oracle and
integrated latency before any low-score append path is implemented.
