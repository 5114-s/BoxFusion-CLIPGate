# Boxer-Past3 S1 depth-receipt development result

Date: 2026-08-23

## Decision

Reject S1 active birth.  Do not evaluate S1 on H10 and do not run S1 on
full100.

S1 froze each OBB at its first stable three-frame receipt and then required a
causal depth/view support component.  On the three already-open S0 development
scenes it emitted seven fixed candidates.  Only one was an additional match at
IoU 0.15; none matched at IoU 0.25 or 0.50.  The fixed suffix therefore failed
the preregistered all-threshold recovery and nonnegative-AP gates.

## Constant-score result

| Metric | IoU 0.15 | IoU 0.25 | IoU 0.50 |
|---|---:|---:|---:|
| Native T05 AP | 27.7670 | 27.7670 | 16.2060 |
| T05 + fixed S1 suffix AP | 28.4006 | 26.5463 | 14.8732 |
| AP delta | **+0.6335** | **-1.2207** | **-1.3328** |
| Candidate maximum-matching TP | 1 / 7 | 0 / 7 | 0 / 7 |
| Additional native-union matches | 1 | 0 | 0 |
| Added evaluator false positives | 6 | 7 | 7 |

All rows, including the native prefix and candidate suffix, use constant score
`1.0`.  Native scene files remained byte-identical.

## No-GT funnel and bounded runtime

| Scene | Raw per-view | Frozen receipts | Depth-qualified | Native-overlap rejected | Final |
|---|---:|---:|---:|---:|---:|
| scene0377_02 | 342 | 7 | 2 | 1 | 1 |
| scene0568_00 | 941 | 26 | 19 | 16 | 3 |
| scene0606_01 | 1,802 | 59 | 17 | 14 | 3 |
| **Total** | **3,085** | **92** | **38** | **31** | **7** |

The S1 materializer used an 11-keyframe global RGB-D/pose ring and a maximum
five-node graph per receipt.  Mean/p95/max materializer time was approximately
9.53/15.95/58.58 ms per processed keyframe, excluding frozen OWLv2 + Boxer
inference.  This is an observer-branch measurement, not an integrated
same-GPU end-to-end FPS claim.

## Interpretation

Depth consistency and viewpoint separation verify that the same stable 3D
region was observed; they do not establish that the region is a benchmark
object or that its OBB is accurate.  S1 therefore fixes the terminal-geometry
drift in S0 but still lacks a strong objectness/semantic-consistency test and a
box-geometry refinement step.

The already generated H10 proposal pool remains no-GT.  H10 annotations have
not been opened for S1, so it can be preserved for a separately preregistered
S2 gate.  The preliminary H10 proposal runner was later found to enumerate one
future pose metadata entry per scene before truncation and to run an unused
inline tracker.  Its predictions did not consume the future pose, but a strict
past-only S2 should regenerate the pool with manifest-only frame enumeration,
verified stream-input hashes, and inline tracking disabled.

## Artifacts

- S1 preregistration:
  `docs/BOXER_PAST3_S1_DEPTH_PREREGISTRATION.md`
- S1 shadow JSON SHA-256:
  `4d691fdbef0dbd52ca6421a1ad524d7ba7a97b2ef5718abf67d3d241dc15c1dd`
- S1 shadow NPZ SHA-256:
  `d246b20afa5bd085c701f7b64053cca3a69ffd84a158f5a72b6e22dfb9bd5540`
- S1 oracle JSON SHA-256:
  `9d25d81385e00c38abd59501b6a4aa7865043732d7c320ddfab45ac340ed3a27`
- S1 materializer source SHA-256:
  `c90250afc2d56e81f8ef2c23a3b65bde69303f54b5a633e98ee46a05a3fb4874`
- Extended fixed-candidate oracle source SHA-256:
  `a84b0dd81678fe4dcf927f7d8acaf4a0dd7dd3c9126976de53e7b40f556a379a`
- H10 no-GT raw-candidate JSON SHA-256:
  `c493a5d88dcaee81a3f707d2b4d47dda62e7329e7fcb091e6738000f5448e6ff`
- H10 no-GT raw-candidate NPZ SHA-256:
  `fc448a668f2d45196ccc1840b397238f4e818812b5390b8b0cf148f23a638376`
