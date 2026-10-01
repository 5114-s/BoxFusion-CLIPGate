# S2 YOLOE-direct dev3 result and stopping decision

Date: 2026-08-23

## Decision

Reject the preregistered S2 terminal policy. Do not activate its 17-box suffix,
do not open H10 for S2, and do not run this branch on full100. The fixed suffix
added no true positive at strict IoU 0.15, 0.25, or 0.50 and reduced AP at all
three thresholds.

Do not discard the frozen YOLOE producer entirely. The separately sealed
complete universe of 284 causal confirmed tracks has an optimistic additional
native-union matching ceiling of 6/3/3 GT at IoU 0.15/0.25/0.50. This is enough
for a necessary +10-recall-point ceiling on these 28 dev3 GT, but it is not a
measured AP gain and provides no deployable way to identify the useful tracks.
Keep that universe only as a future no-GT selector research branch.

The next single-variable experiment is **S3a Boxer Top4 -> MobileSAM per-view
masked lifting**. It must remain separate from any YOLOE-track selector or
active birth so that its contribution is identifiable.

## What was measured and what was only a ceiling

Three different results must not be conflated:

1. **Formal measured AP:** the preregistered 17-box suffix was appended after
   the byte-identical frozen T05 prefix and evaluated with every row at score
   `1.0`. This is an actual counterfactual prediction and an actual AP result.
2. **102-row raw terminal ceiling:** all terminal producer outputs were sealed
   before GT was read. Post-hoc maximum-cardinality matching asks how many GT
   could be covered if an oracle selected among them. It is not an AP result,
   ranking policy, or permitted gate.
3. **284-track complete-universe ceiling:** every active or archived causal
   track that reached three views was exported before the terminal extent,
   projection, deduplication, native-overlap, NMS, and cap gates. Post-hoc
   matching and geometry oracles are still only optimistic recall ceilings.

Maximum matching ignores the false-positive ordering and precision cost that
determine AP. A statement that a pool “supports +10” below therefore means
only that its additional native-union recall is at least 10 points on dev3. It
does not mean that appending the pool, or any currently implemented no-GT
selector, improves AP by 10 points.

## Formal S2 terminal AP result: failure

The frozen T05 prefix contains 46 predictions across the three scenes. The S2
materializer appended 17 fixed candidates (6/6/5 by scene), preserving the
native rows, classes, geometry, scores, order, and on-disk payload as an exact
prefix. Formal evaluation used constant score `1.0` for both prefix and suffix;
stored detector scores were provenance only.

| Metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| Native T05 AP | 27.7670 | 27.7670 | 16.2060 |
| T05 + fixed 17-box suffix AP | 21.4403 | 21.4403 | 13.5429 |
| AP delta | **-6.3267** | **-6.3267** | **-2.6632** |
| Native greedy TP | 18 | 18 | 14 |
| Native + suffix greedy TP | 18 | 18 | 14 |
| Candidate maximum-matching TP | 0 / 17 | 0 / 17 | 0 / 17 |
| Additional native-union matches | 0 | 0 | 0 |
| Native evaluator FP | 28 | 28 | 32 |
| Native + suffix evaluator FP | 45 | 45 | 49 |

The suffix therefore contributed 17 false positives and no recall at every
threshold. It failed both preregistered dev3 promotion requirements: positive
AP delta and at least one additional native-union match at all thresholds.

## Raw 102-row terminal ceiling

The 102 terminal rows are 37 from `scene0568_00`, 51 from `scene0606_01`, and
14 from `scene0377_02`. Using their reported producer q02/q98 AABBs without
selection or mutation gives:

| Ceiling metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| Candidate-only maximum matches | 20 | 16 | 11 |
| Native maximum matches | 18 | 18 | 14 |
| Native + candidate union maximum matches | 22 | 19 | 16 |
| Additional union matches over native | **4** | **1** | **2** |
| Additional union recall points over 28 GT | **+14.29** | **+3.57** | **+7.14** |

This pool does contain useful geometry, but its ceiling is below +10 at IoU
0.25 and 0.50. It therefore cannot support the requested all-threshold +10
target even with a perfect selector over only these 102 rows.

The official-baseline-unmatched recovery audit found the same 4/1/2 maximum
recoveries. At IoU 0.15, three of the four useful rows were later rejected by
the native-overlap gate and one by the output cap. The single IoU-0.25 recovery
and both IoU-0.50 recoveries were rejected by native overlap. This is post-hoc
diagnosis, not authorization to invert or tune that gate on dev3.

## Complete 284-track universe ceiling

The complete no-GT universe contains all confirmed active and archived tracks:
88/155/41 by scene, or 284 total. Candidate membership was sealed before GT
access, and normal terminal arrays plus deterministic 512-point samples were
identical to the frozen S2 replay.

### Fixed producer geometry

| Ceiling metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| Candidate-only maximum matches | 22 | 18 | 12 |
| Native maximum matches | 18 | 18 | 14 |
| Native + candidate union maximum matches | 24 | 21 | 17 |
| Additional union matches over native | **6** | **3** | **3** |
| Additional union recall points over 28 GT | **+21.43** | **+10.71** | **+10.71** |

The fixed q02/q98 producer geometry narrowly clears the preregistered necessary
+10 recall ceiling at IoU 0.25 and 0.50: three additional GT correspond to
10.71 points on this small 28-GT development set. This one-GT-count margin is
too small to claim robust full100 behavior.

### Post-hoc geometry ceilings

| Geometry ceiling | IoU 0.15 additional | IoU 0.25 additional | IoU 0.50 additional |
|---|---:|---:|---:|
| Best bounded per-view box per track | 6 (+21.43) | 4 (+14.29) | 4 (+14.29) |
| Best preregistered geometry per track | 6 (+21.43) | 5 (+17.86) | 4 (+14.29) |

These oracle rows show that per-view geometry contains more AP25/AP50
headroom than the terminal q02 box. They use GT after sealing to choose a box
or quantile for each track and are explicitly nondeployable.

## Funnel diagnosis: 284 -> 102 -> 17

| Stage or disposition | Count | Meaning |
|---|---:|---|
| Three-view causal confirmed tracks | 284 | Complete active + archived universe |
| Rejected by terminal minimum extent | 139 | Failed the fixed all-axis `0.30 m` extent gate |
| Rejected by projection consistency | 36 | Failed the fixed mean projection-IoU gate |
| Removed by terminal deduplication | 7 | Redundant under the producer terminal deduplication |
| Terminal producer rows | 102 | Survived the first terminal funnel |
| Rejected by native AABB overlap | 63 | Maximum native AABB IoU was at least `0.10` |
| Removed by candidate self-NMS | 6 | Suppressed at candidate AABB IoU `0.25` |
| Removed by per-scene output cap | 16 | Fell below the fixed cap of six per scene |
| Fixed materialized suffix | 17 | 6/6/5 rows evaluated at score `1.0` |

The first funnel retains only 102/284 tracks. Its producer-geometry ceiling
contracts from 6/3/3 additional union matches to 4/1/2, a re-optimized matching
loss of 2/2/1 at IoU 0.15/0.25/0.50. Counts from different maximum matchings
must not be interpreted as a one-to-one attribution to individual tracks.

The second funnel retains 17/102 terminal rows, or 17/284 of the full universe.
It contracts the remaining 4/1/2 additional-match ceiling to 0/0/0. The native
overlap gate accounts for 63 of the 85 removals and post-hoc contains every
raw AP25/AP50 recovery. A replacement needs object identity or stronger causal
support; simply lowering the overlap threshold on these opened scenes would be
GT-conditioned tuning.

## Online, no-target-training, and data-access contract

The producer satisfies the intended experimental constraints:

- The exact sealed gap-25 RGB/depth/pose/calibration stream contains 209
  scheduled keyframes in fixed scene order.
- Frozen YOLOE-11s prompt-free weights were not trained or fine-tuned on the
  target ScanNet split for this experiment. There was no train command,
  gradient update, learned target quality head, or online model update.
- Processing was causal and past/current-only. Track memory, per-observation
  points, per-track points, and TTL were bounded; no future frame or terminal
  backfill was used.
- Detector labels, CLIP features, ScanNet labels, GT boxes, and oracle outputs
  were unavailable to candidate membership and ranking. The original frozen
  CLIP/native prefix remained unchanged.
- S2 and S3 were shadow diagnostics: `birth=false` or
  `active_birth_authorized=false`. Native predictions were never modified.
- Dev3 GT was opened only after each candidate pool was create-only sealed.
  S2 failed dev3, so H10 was not opened. The raw ceiling records
  `h10_gt_accessed=false`; the complete-universe report records
  `H10_accessed=false` and `full100_accessed=false`.

## Isolated replay and runtime caveat

The frozen isolated S2 replay logs report:

| Scene | Scheduled keyframes | Wall-clock cost | Reported average FPS |
|---|---:|---:|---:|
| `scene0568_00` | 66 | 187.91 s | 8.65 |
| `scene0606_01` | 113 | 255.58 s | 10.98 |
| `scene0377_02` | 30 | 56.22 s | 13.07 |

The producer's internal provider-plus-geometry timers sum to 39.934 s over 209
scheduled keyframes, or 0.191 s per scheduled keyframe. Those component timers
can exclude surrounding native inference, data loading, synchronization, and
materialization costs and must not be reported as end-to-end FPS.

The replay used isolated runtime and artifact roots. Nested CuTR inference
existed only to drive observer lifecycle; its slightly nondeterministic final
boxes were discarded, and the formal byte-identical T05 prefix was loaded
separately. Consequently the experiment establishes causal bounded execution
and gives an isolated throughput observation, but it does not establish an
integrated same-GPU BoxFusion latency delta or production real-time compliance.
That measurement remains mandatory before any future selector is activated.

## Recommendation and next experiment

1. Archive the S2 terminal result as a negative result. Do not tune its extent,
   projection, native-overlap, NMS, or cap thresholds on these opened scenes.
2. Preserve the sealed 284-track universe only as evidence that a future
   causal no-GT selector could be worth studying. It is not permission to append
   all tracks, use an oracle geometry, or start an active birth.
3. If that selector is revisited, preregister a new branch and untouched split.
   It must recover the useful native-overlap tracks while controlling the much
   larger false-positive pool, then pass measured constant-score AP and
   integrated runtime tests.
4. The current next single-variable test is **S3a Boxer Top4 -> MobileSAM
   per-view lifting** in shadow mode. Do not combine it with a YOLOE selector,
   changed terminal gate, geometry oracle, or birth in the same test.

## Frozen artifacts and hashes

| Artifact | SHA-256 |
|---|---|
| `docs/S2_YOLOE_DIRECT_PREREGISTRATION.md` | `fc737deec401de54845a0b7d8cb1152443203d55eef335b0ca99270396f663f3` |
| S2 dev3 stream seal | `cf363f9d92bd5b0c1aaa51ee6c200744fbf60404d671320c75df96ae20128655` |
| S2 terminal shadow JSON | `0f15ee414003139a6b59e2092d8a0d73897acecba06132213fd7394f93cd5017` |
| S2 terminal shadow NPZ | `7fbe7ea7e550071efeb2c371475f2d848492c1eb32d46b6c74ad85fb7e257656` |
| S2 fixed candidate content digest | `18859ede7f421a4344e88eb05cc211874cada212e876e8c51b98414630f19297` |
| S2 formal AP/oracle report | `30423798c2631f244c557d232bdbd188ebf9c72901124a19d0ceee6faac6a799` |
| S2 102-row raw-ceiling report | `2cb338ff37b909adc70d395cb31d8478acfe3af8f3a0587dfdbaea45d5811fd3` |
| `docs/S3_YOLOE_CONFIRMED_UNIVERSE_PREREGISTRATION.md` | `d95454d79f4c64d3a4be241ec2152082ac442cf89a394d5964780bd388ef139d` |
| S3 complete-universe aggregate seal | `8a298f805002c5b9b8554bf52b2e4c6ad78bfebdd0b8113bed1d41ed6df16434` |
| S3 complete-universe ceiling report | `f7cb8855a7698c03983bc8cbb44e7a62aba2777bd95e6d1ae2f12c9ee41930b5` |

Counterfactual prediction pickle SHA-256 values are:

| Scene | SHA-256 |
|---|---|
| `scene0568_00` | `d1b4c9a87e7a674cebc4e84763fc8549a71c56c03badda3888408a62f9d6ca93` |
| `scene0606_01` | `795b960e1d224e0e2335e8ea0b87519960e200ac5bd54c0412ebd53e8ad86d19` |
| `scene0377_02` | `e726581776fbb7468b4281c187bdd7eb9579f9c4065bccf1505a42285fbf2a37` |

Complete-universe scene artifacts are:

| Scene | Manifest SHA-256 | NPZ SHA-256 |
|---|---|---|
| `scene0568_00` | `aa0e49c2f3dddbf407ffd2970157f27f8cb33bfc8822cc27ace484bb8b9ad237` | `544a7e252300608a69e8764de925f5648b394e0c012164a7d60c789e9c91c3c7` |
| `scene0606_01` | `46d5905fd5cf1fbe3c301e7a0615bd99ff81b3c4abe503bba69fe6f8f012d69f` | `e58b6a306b2006d426b63503bbf4e0239207b90120b582720a352aba2e2e98f4` |
| `scene0377_02` | `e46a0626561f9ebe6fefbfa5adf47039e5f0c44f2c4556327fafca58cefdd6dd` | `e6880498c37a6bc5126c33c08b804a9230f5f12bed6a58eda6f10b44ca6e239e` |

Frozen implementation source SHA-256 values are:

| Source | SHA-256 |
|---|---|
| `tools/materialize_s2_yoloe_direct_shadow.py` | `884de97bcb0d05238a636e6ea73a5a955ac13180aad32c9f8ac43721ae54fc62` |
| `tools/audit_scannet_s2_yoloe_direct_oracle.py` | `6a76f6cd187038c5f50949962a406548cfc362c51d823011a35ab64432a636ae` |
| `tools/audit_scannet_s2_yoloe_raw_ceiling.py` | `3024e1562ed6e485af96386338ce1571fd14ebfab666a70327f2ea7bb5dd86e0` |
| `tools/export_s3_yoloe_confirmed_universe.py` | `25e1444c70613161054684789601c154787f6b1075bace02a114db7ed448aa9c` |
| `tools/audit_scannet_s3_yoloe_confirmed_universe.py` | `3be484a4b5328b65baee4c8341485640a68a9d642cd5ec75a24981421e5c6610` |

Runtime-log SHA-256 values are:

| Scene | SHA-256 |
|---|---|
| `scene0568_00/smoke.log` | `0bae39546532740b407f8979968a313a54674649e4b794bcad01c41de9f4e8f8` |
| `scene0606_01/smoke.log` | `9c56a7ea9754dea4f08b31c048820b1ed576a30ffe9e636c15ca963d672fd590` |
| `scene0377_02/smoke.log` | `dfa98f00e7b51f774ac88ece0e4c2e6ad9da0b19824dcdde37338905b0dbc358` |

## Test ledger

The four focused suites were rerun together with pytest plugin autoload
disabled and passed `22/22`:

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/admin1/miniconda3/envs/boxfusion-online/bin/python -m pytest -q \
  tests/test_materialize_s2_yoloe_direct_shadow.py \
  tests/test_audit_scannet_s2_yoloe_direct_oracle.py \
  tests/test_audit_scannet_s2_yoloe_raw_ceiling.py \
  tests/test_s3_yoloe_confirmed_universe.py

22 passed in 0.62s
```

| Test file | Tests passed | SHA-256 |
|---|---:|---|
| `tests/test_materialize_s2_yoloe_direct_shadow.py` | 5 | `6fda1c982ae7ceee149b143aacd0f03bd03013b15e269469999012b391911eec` |
| `tests/test_audit_scannet_s2_yoloe_direct_oracle.py` | 7 | `004e429a2884deeb472d49e1df0212ac97e0fad56dd248ddfcaa601093bb1f11` |
| `tests/test_audit_scannet_s2_yoloe_raw_ceiling.py` | 5 | `f480c03f1d1227188339bf4c5f200c91b91080ad07270eb0d43bcd2fb2ff54b0` |
| `tests/test_s3_yoloe_confirmed_universe.py` | 5 | `57650f4224593d9f76e891702ede57ec74913ce3bedd7fe8454b6419ca21f5b0` |
