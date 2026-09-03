# T05 full100 error-budget audit

Date: 2026-08-23

## Frozen protocol

- ScanNet official ordered 100-scene list, SHA256
  `4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5`.
- CuTR threshold `0.5`, keyframe gap `25`, appearance gate disabled.
- The evaluator overwrites every prediction score with `1.0`.
- Ground truth is read only by this offline audit; it is not available to
  online inference.

## Official AP result

| arm | AP15 | AP25 | AP50 | boxes |
| --- | ---: | ---: | ---: | ---: |
| B05 | 29.7643 | 25.2746 | 7.9653 | 1795 |
| T05 | 30.0506 | 25.4093 | 8.5448 | 1801 |
| T05 - B05 | +0.2863 | +0.1348 | +0.5794 | +6 |

The paired 10,000-scene-bootstrap 95% intervals in AP points are
`[-0.6748, 1.1539]`, `[-0.6088, 0.9380]`, and `[0.0682, 1.3236]`.
Only AP50 has a clearly positive interval.

An absolute +10 target over fresh B05 is `39.7643 / 35.2746 / 17.9653`.
After T05, the remaining gaps are `9.7137 / 9.8652 / 9.4206` AP points.

## E0: T05 final-box ceiling

There are 1,433 GT boxes and 1,801 T05 final boxes. Matching uses strict
`IoU > threshold`. Maximum one-to-one matching is a GT-only offline recall
ceiling and is not reported as deployable AP.

| IoU | official greedy TP | official recall | maximum-match TP | maximum-match recall | official FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| .15 | 863 | 60.2233% | 864 | 60.2931% | 938 |
| .25 | 791 | 55.1989% | 791 | 55.1989% | 1010 |
| .50 | 461 | 32.1703% | 461 | 32.1703% | 1340 |

The maximum-recall values of the final pool already exceed the three target
AP values (`39.7643 / 35.2746 / 17.9653`). Therefore missing proposals are
not an information-theoretic barrier to +10, although a deployable selector
still has to realize enough of that ceiling. The large AP-to-recall gaps also
show that low-quality/false-positive selection under constant scores is a
major error category.

| IoU | GT with any covering final box | one-to-one conflict | predictions covering any GT | redundant covering predictions | below-threshold/background predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| .15 | 910 | 46 | 902 | 38 | 899 |
| .25 | 807 | 16 | 801 | 10 | 1000 |
| .50 | 462 | 1 | 461 | 0 | 1340 |

The best-IoU band counts over GT are: `IoU=0: 207`, `(0,.05]: 171`,
`(.05,.15]: 145`, `(.15,.25]: 103`, `(.25,.50]: 345`, and `>.50: 462`.
The 345 GT boxes in the `.25-.50` band are the clearest AP50 geometry
headroom. T05 versus B05 increases maximum matches by `+2 / +3 / +19`, so
Reliable-View Top-K primarily improves high-IoU geometry.

## E2-cache: independent proposal-pool evidence

The sealed cache
`/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2`
contains 23,651 post-filter proposals for the same official 100 scenes and
protocol. Its manifest is byte-paired to the separate X0 CuTR result, not to
the live B05/T05 runs above. Consequently this section is independent
same-protocol evidence, not a causal T05 decomposition.

| maximum one-to-one TP | IoU .15 | IoU .25 | IoU .50 |
| --- | ---: | ---: | ---: |
| sealed X0 final | 862 | 789 | 440 |
| final union all per-frame proposals | 1098 | 1024 | 708 |
| union gain ceiling | +236 | +235 | +268 |
| final union proposals with all AABB extents >= 0.3 m | 992 | 919 | 648 |
| extent-filtered union gain ceiling | +130 | +130 | +208 |

Using all raw proposals, `188 / 214 / 267` GT boxes are covered by a
per-frame proposal but not by the final X0 pool at IoU `.15 / .25 / .50`.
This shows substantial post-proposal association/fusion/survival headroom,
but the cache lacks proposal-to-track-to-final provenance, so these losses
cannot yet be split rigorously into association and geometry.

Naively appending raw proposals is invalid: the pool contains many duplicates
and low-quality boxes, and constant scoring provides no low-score suffix.
Any recovery must use causal support and conservative selection.

## Decision

1. Retain T05 provisionally for its AP50 effect.
2. Before an active Graw run, add an observer-only trace containing stable
   proposal IDs, raw world corners, native match/unmatch decisions, terminal
   track membership, and final-row-to-track mapping.
3. Replay B05, T05, and Graw from one sealed proposal stream. Replay failure
   must be fatal; observer mode must preserve output digests exactly.
4. Test Graw only on native-unmatched proposals, with no birth, no score or
   CLIP change. This is the next active module because it can reduce
   fragmentation/duplicates and recover proposal evidence without adding a
   high-FP constant-score suffix.
5. Run fixed-10 first; promote Graw to full100 only after paired AP and runtime
   gates pass.

The existing `analyze_fused_oracle.py --constant-score` AP field is not used
as official AP here: it uses stable tie sorting and non-strict thresholding,
whereas the official evaluator uses NumPy default quicksort and strict
`IoU > threshold`.
