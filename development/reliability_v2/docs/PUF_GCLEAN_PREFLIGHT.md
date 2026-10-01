# PUF-geometry-shadow / Gclean preflight

Date: 2026-08-23 (Asia/Shanghai)

## Decision

`PUF-active` remains disabled and no full100 run is authorized.  The fixed
three-scene preflight is mechanically valid and fast, but the six active-safe
association suggestions produce **zero terminal-actionable changes** and
therefore **zero AP gain** under both evaluation score protocols.

This experiment implements only the training-free voxel association
normalization from the [PUF paper](https://arxiv.org/html/2607.07170), checked
against its [official repository](https://github.com/yyyyangyi/PUF).  It is not
the full scene-graph model: semantic Dirichlet updates, learned relationship
priors, relationship inference and birth are all absent.

## Frozen protocol

- T05 base: CuTR `score_thresh=0.5`, appearance gate off, Reliable-View Top-K3
  on, CLIP frozen.
- Ordered scenes: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Sealed cache namespace:
  `scannet-graw-e2-score05-preflight3-v3-r1`.
- Producer fingerprint:
  `457c997631cd71a83b6480a6e45e103e273ac5ed2d1488252790549c2e2b3504`.
- Inputs: native-unmatched SMOV-clean 5 cm voxel fragments and bounded
  begin-keyframe-past tracks not already reserved by native association.
- PUF rule: `Lk=intersection/current_voxels`, `lambda_null=0.4`, joint
  normalization over every positive-overlap candidate in the frozen Top-8
  broad phase.
- Raw paper directive: `beta_null <= 0.5`, followed by stable-ID argmax.
- Active-safe subset: selected track must beat null (`margin>0`) and the same
  past track may not be selected by another proposal in that keyframe.
- Birth, live merge, geometry change, score change, class change and CLIP
  update are all disabled.

The active-safe rule was frozen during pre-AP code audit.  Gclean matcher
failure, evidence failure, cap violation, or probability normalization error
over `1e-12` fails the entire PUF keyframe open.

## Mechanics and non-interference

All three observer traces are valid; 199 keyframes and 1,163 sealed proposals
were processed with no fail-open event.

| quantity | result |
|---|---:|
| PUF candidate proposals | 199 |
| positive candidate pairs | 47 |
| null-only proposals | 167 |
| raw paper directives | 6 |
| active-safe associations | 6 |
| agreeing with Gclean | 4 |
| new relative to Gclean | 2 |
| same-track conflict groups | 0 |
| maximum normalization error | 1.11e-16 |

The PUF-shadow prediction root has the same three scenes, 46 rows, row order,
class order, and exact saved scores as T05.  Replay optimization produces only
the already-known stochastic geometry floor: corresponding-corner error versus
T05 is 0 mm at p50, 5.330 mm at p95 and 14.306 mm maximum, within the 15 mm
preflight cap.  PUF never touches these native boxes.

## Association audit

| scene / frame | current -> past | beta track | beta null | margin | Gclean | terminal class |
|---|---:|---:|---:|---:|---|---|
| scene0568_00 / 50 | 7 -> 3 | 0.6045 | 0.3955 | 0.2091 | agree | target dropped |
| scene0568_00 / 375 | 124 -> 19 | 0.5634 | 0.2958 | 0.2676 | agree | target dropped |
| scene0568_00 / 850 | 299 -> 226 | 0.5512 | 0.3832 | 0.1680 | PUF-only | candidate dropped |
| scene0606_01 / 325 | 62 -> 56 | 0.5096 | 0.4904 | 0.0191 | agree | candidate dropped |
| scene0606_01 / 700 | 176 -> 77 | 0.4111 | 0.3595 | 0.0516 | PUF-only | candidate dropped |
| scene0606_01 / 1700 | 360 -> 292 | 0.6177 | 0.3823 | 0.2354 | agree | later native same |

Terminal classification totals are three candidate-dropped, two
target-dropped, one later-native-same and **zero both-survive-distinct**.  The
fail-closed materializer consequently deletes zero rows.  This is not an
executed merge; it is a create-only terminal duplicate-suppression
counterfactual.

## AP

The headline evaluator is an isolated copy of the published evaluator whose
validation list is replaced in `/tmp` by the sealed three-scene list.  It
forces every prediction score to `1.0`.  The native-score evaluator is reported
separately and never mixed with the headline.

| score protocol / arm | boxes | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|---:|
| constant 1.0 / T05 | 46 | 27.7670 | 27.7670 | 16.2060 |
| constant 1.0 / PUF-shadow native | 46 | 27.7670 | 27.7670 | 16.2060 |
| constant 1.0 / PUF counterfactual | 46 | 27.7670 | 27.7670 | 16.2060 |
| constant counterfactual - native | 0 | **+0.0000** | **+0.0000** | **+0.0000** |
| native score / T05 | 46 | 26.8814 | 26.8814 | 18.1330 |
| native score / PUF-shadow native | 46 | 26.8814 | 26.8814 | 18.1330 |
| native score / PUF counterfactual | 46 | 26.8814 | 26.8814 | 18.1330 |

The AP15/AP25/AP50 recalls are unchanged at 64.2857/64.2857/50.0000 percent.

## Online cost

| component | p50 ms/keyframe | p95 ms/keyframe | max ms/keyframe |
|---|---:|---:|---:|
| pair evidence | 0.021 | 0.096 | 0.504 |
| probability and safety audit | 0.052 | 0.121 | 0.282 |
| PUF incremental total | 0.076 | 0.209 | 0.604 |
| SMOV + Gclean + PUF observer | 22.816 | 52.263 | 207.467 |

At the fixed gap of 25, PUF incremental p95 amortizes to about 0.0084 ms per
input frame.  The whole observer p95 amortizes to about 2.091 ms per input
frame.  PUF itself comfortably passes the preregistered 2 ms/keyframe
incremental limit.

## Gate result and next step

The trace, probability, non-interference and latency gates pass.  The scientific
gate fails because terminal-actionable associations are 0 (minimum 5), AP25
and AP50 improvements are not positive, and mean AP improvement is 0.  No
threshold is retuned, `PUF-active` is not implemented, and full100 would only
repeat an output-inert observer at substantial runtime cost.

The route advances to `MV3DIS-shadow`.  It should remain a verifier/re-ranker
over secondary candidates first; it must not be treated as a path to the +10
AP objective unless it exposes terminal-actionable corrections.  This PUF
result also reinforces that association-only modules cannot supply the missing
recall needed for +10 points; a later universal-proposal/birth branch remains
the stage with the appropriate ceiling.

Artifacts:

- preregistration: `docs/PUF_GCLEAN_PREREGISTRATION.md`
- PUF diagnostics: `logs/scannet_puf_gclean_shadow_replay_score05/diagnostics`
- native identity audit: `logs/scannet_puf_gclean_shadow_identity_score05.json`
- terminal audit:
  `results/scannet_puf_gclean_counterfactual_score05_preflight3/puf_gclean_counterfactual_audit.json`
- strict AP logs: `logs/scannet_puf_gclean_formal3_score05`
- machine summary: `logs/scannet_puf_gclean_preflight_score05.json`

