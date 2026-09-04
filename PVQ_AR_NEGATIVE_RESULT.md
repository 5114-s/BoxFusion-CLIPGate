# PVQ-AR @ correspondence — negative result record

**Status: falsified. Do not tune thresholds; do not keep as an accuracy
module. Code stays as reusable infrastructure only.**

---

# Addendum: PVQ-AR contrast-rule @ 3D NMS (CLIP) — negative result

**Status: falsified with CLIP embeddings. The absorbed-TP channel (88
recoverable GT) is real, but CLIP crop appearance is the wrong instrument
for it. Do not run active with these features.**

## Hypothesis under test

> A negative-prototype contrast score (T-Rex-Omni style: parent claim vs
> best rival claim over view-compatible historical prototypes) can refuse
> wrong native NMS absorbs and recover the 88 absorbed true positives
> measured by `tools/audit_scannet_pvq_nms_headroom.py`.

Shadow run: `scannet_t05_boxer_pvq_nms_ar_shadow_score05`
(config `scannet_t05_boxer_pvq_nms_ar_shadow_score05.yaml`, driver
`scripts/run_scannet_t05_boxer_pvq_ar_full100.sh`). Output-inert:
AP 35.03/31.46/15.75 vs Cbest 34.99/31.41/15.67 (noise envelope).
6,018 adjudicated events: 3,202 parent-owns, 1,379 orphan-guard,
62 no-rival, **1,375 would-refuse**. Retrieval cost ~10 ms/scene.

## Evidence (`tools/audit_scannet_pvq_nms_ar.py` + offline sweep)

- would-refuse labelling (IoU 0.15): 246 correct (17.9%),
  **311 wrong — parent and child are the same GT object, refusal breaks a
  correct dedup (22.6%)**, 818 unscored. Recoverable coverage: 41 of 88.
- Offline threshold sweep over logged (sim_parent, sim_rival, IoU): best
  precision 26.9% at margin 0.3 / sim 0.7 / IoU<0.3, coverage drops to
  33. No configuration approaches usable precision.
- Decisive split: when the parent HAS view-compatible prototypes ("true
  contest"), precision is **9.5%** (200 wrong vs 59 correct) — CLIP
  actively points to same-category look-alike rivals over the co-located
> parent. When the parent has no compatible prototype (orphan path),
  precision is 24.7% with 33 recoverable. The look-alike failure is
  structural for category-level embeddings on ScanNet (identical
  chairs/tables), so a model swap is not guaranteed to fix it.

## Not falsified

- The 88-GT absorbed-TP recovery channel (geometry-side evidence).
- Candidate instruments ranked by prior: (1) refuse when merging degrades
  the parent track's multi-view geometric consistency — no appearance
  needed; (2) instance-level embeddings (DINOv2/FG-CLIP) — weak prior
  given identical-instance clutter.

## Artifacts

- arbitration logs: `diagnostics/pvq_nms_ar_shadow_score05/*_pvq_nms_ar.jsonl`
- audit: `logs/pvq_nms_ar_audit.json`
- unit tests: `tests/test_pvq_ar.py` (NMS-stage contrast tests included)


## Hypothesis under test

> View-indexed multi-prototype historical CLIP queries (PVQ-AR), used to
> locally rearrange Top-1/Top-2 ambiguous `proposal -> track` edges of the
> native 2D matching (correspondence) stage, improve official100 AP on top
> of the Cbest route (Top-K3 + Boxer active + native real-score).

## Implemented module (phase-1 contract, all enforced)

- `boxfusion/pvq_ar.py`: view-indexed retrieval of `K <= 4` view-compatible
  historical CLIP prototypes per track (angular view gate, default 60 deg),
  bounded per-track memory, query-before-commit (snapshots committed at
  keyframe start), abstain-safe decision rule (both tracks must expose a
  compatible prototype; low similarity is never negative evidence).
- Hooks: `correspondence_association` ambiguity extraction (accepted Top-1
  vs different-track Top-2, margin gap <= 0.10) and demo.py keyframe
  lifecycle. Native 3D NMS, scores, proposals, Boxer corners untouched.
- Driver asserts every phase-1 prohibition (no FastSAM/SAM, no TSDF, no
  covariance fusion, no global Hungarian, no birth modules).

## Evidence (three arms, official100, real-score evaluator)

| arm | AP@0.15 | AP@0.25 | AP@0.50 | events | rearranged |
|---|---|---|---|---|---|
| Cbest baseline | 34.9863 | 31.4140 | 15.6662 | — | — |
| + PVQ-AR shadow | 35.0729 | 31.4954 | 15.7160 | 95 | 0 (22 would) |
| + PVQ-AR active  | 35.0347 | 31.4604 | 15.7001 | 97 | 25 applied |

- The active delta (+0.05/+0.05/+0.03) is inside the measured GPU
  non-determinism envelope (~3 cm corner drift between identical reruns).
- 6,242 keyframes produced only 95-97 ambiguity events; 25 reassignments.
- Choice-set oracle (`tools/audit_scannet_pvq_ar_choice_set.py`): the
  proposals at ambiguity events have **max GT IoU 0.010** — they are
  `<0.35 m` clutter that the ScanNet detection GT (furniture classes,
  min dimension 0.34 m) does not score. Even a perfect rearranger cannot
  move AP from this hook point.

## What is and is not falsified

Validated: multi-prototype view-indexed retrieval works mechanically; it is
safe (abstain-dominant: 63 native-better / 8 missing-prototype /
2 low-similarity) and effectively free (retrieval ~0.3 ms/scene; FPS
median 17.7 active vs 17.8 shadow).

Falsified: **only** "PVQ-AR at the correspondence hook point improves AP".

Not falsified: historical-prototype retrieval at other decision points
(e.g. 3D NMS/spatial merges of scored furniture proposals). That is a new
hypothesis and must pass its own choice-set/oracle gate (~+1 AP bar)
before shadow/active work — see `tools/audit_scannet_pvq_nms_headroom.py`.

## Artifacts

- predictions: `results/scannet_t05_boxer_pvq_ar_{shadow,active}_topk3_real_score05`
- scene logs: `logs/scannet_t05_boxer_pvq_ar_{shadow,active}_topk3_real_score05`
- eval logs: `logs/scannet_official100_real_score/scannet_t05_boxer_pvq_ar_*.log`
- choice-set audit: `logs/pvq_ar_choice_set_audit.json`
- unit tests: `tests/test_pvq_ar.py`
- configs: `config/scannet_t05_boxer_pvq_ar_{shadow,active}_topk3_real_score05.yaml`
- driver: `scripts/run_scannet_t05_boxer_pvq_ar_full100.sh`
