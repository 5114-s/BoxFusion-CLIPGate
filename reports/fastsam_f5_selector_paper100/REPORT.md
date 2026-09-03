# F5 GT-free past-only geometry selector — paper100 result

## Decision

`discard_f5_geometry_selector_for_plus10_route`

F5 passes the frozen no-GT integrity, causality, determinism and online-runtime
gates, but fails the preregistered `+10 AP points at every IoU threshold`
capacity gate.  Active birth is not authorized.

The AP values below are a GT-selected constructive suffix and are oracle-only.
They are not deployable F5 AP.  F5 remained shadow-only, so the actual native
output and its constant-score AP are unchanged.

## Paper100 evaluation

| Metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| Native AP, score=1.0 | 31.0130 | 26.7911 | 12.0669 |
| GT-selected suffix AP | 50.6149 | 40.3964 | 15.7056 |
| Oracle delta AP | +19.6019 | +13.6052 | +3.6387 |
| Extra union matches | +384 | +273 | +91 |
| Required extra matches | 144 | 144 | 144 |
| F4 G4 extra-match capacity | 473 | 421 | 260 |
| F5 retained F4 capacity | 81.18% | 64.85% | 35.00% |

The decisive failure is IoU 0.50: only 91 extra maximum-matching GT instances
remain, and even a GT-selected suffix improves AP50 by only 3.6387 points.

## Selector census and online gates

- 100 scenes, 6,817 scheduled keyframes, 6,726 successful frames.
- 52,299 unique frozen F4 sources; exactly one selected geometry per source.
- H0 / HL / HLG / HB: 1,149 / 5,281 / 44,205 / 1,664.
- HB appears in 94 scenes and is 3.1817% of all sources.
- F5 incremental warm p95: 18.6660 ms.
- Composed warm p95 / max: 301.4659 / 462.6084 ms per source frame.
- Composed warm mean amortized over the 25-frame stride: 8.5031 ms/raw frame.
- Warm deadline misses: 0; CUDA peak: 632,576,000 bytes.

At IoU 0.50, the 1,664 selected HB sources recover only 37 native-unmatched GT
instances (2.22% candidate match fraction).  HLG contributes most of the
remaining capacity, but the fixed selector retains only 35% of F4 G4's AP50
geometry capacity.

## Integrity

- No-GT merge receipt SHA-256:
  `9e8bd97a2e29a6f5b68fc1532ef1ff1f4c5698115df0631db38486269bea300e`
- Evaluation report SHA-256:
  `d6e66adca31503361df9d5c451c97cdd0ed89b3dfc90fdf5b25ae3681ca13fa6`
- Selector protocol SHA-256:
  `2a6d62fa9d5912dc3871bbc485f44987565bda61b818722b3a4e6577d34a6afc`
- Evaluation protocol SHA-256:
  `5eb3120808ff61fcc2ffabb3b2912f1057a82c3af4222a8d7018f767901b07f7`
- Evaluator source SHA-256:
  `e7c7284c9d1751d33c807b84f23711c1853ae208f7010d6f2abba9dce908bb0e`

The evaluator pins the exact F4/F5 receipts, historical F4 report, scene list,
native predictions, GT, axis alignments and official evaluator sources.  It
executes the sealed official `eval_det.py` matching/AP kernel and cross-checks
it against the independent constant-score implementation at every threshold.
