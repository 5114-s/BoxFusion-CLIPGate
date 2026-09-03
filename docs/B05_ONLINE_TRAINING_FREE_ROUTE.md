# B05 online no-target-training route

## Frozen protocol

- Official ScanNet 100-scene list and order.
- `detection.score_thresh: 0.5`, native keyframe gap 25.
- The released frame-0 retry remains identical in every arm: retry at
  `score_thresh / 4` only when frame 0 has no proposal.
- The official evaluator replaces every prediction confidence by `1.0`.
- CuTR / Cubify Anything, CLIP, its vocabulary, class features, and embeddings
  remain frozen.
- Frozen broadly pretrained foundation models are allowed, but no component
  may be trained, fine-tuned, calibrated, or threshold-selected on the target
  evaluation scenes.
- Added active logic must be causal (current/past only), bounded, and retain at
  least 95% of B05 live frame-weighted throughput.

The frozen historical B05 reference is:

| Metric | B05 | Absolute +10 AP target | Required AP-point gain |
| --- | ---: | ---: | ---: |
| AP15 | 29.6907 | 39.6907 | +10.0000 |
| AP25 | 24.8289 | 34.8289 | +10.0000 |
| AP50 | 7.7049 | 17.7049 | +10.0000 |

The target is an absolute gain of ten AP points at every IoU threshold.

## Corrected online pipeline

```text
RGB-D + pose
  -> frozen CuTR / Cubify Anything proposals (threshold 0.5)
  -> auxiliary current-frame 5 cm depth fragments
  -> native BoxFusion association
       -> unmatched-only Group3D-lite secondary association
  -> query-before-commit bounded past voxel memory update
  -> retained reliable-view Top-K / depth-consistent BoxFusion geometry
  -> optional robust OBB candidate, accepted only by past/current depth support
  -> optional unexplained-depth candidate
       -> past-only observer/supporter high-precision birth gate
  -> frozen native CLIP semantics
```

Fragment extraction must precede Group3D matching, while committing the
current observation to memory must happen after every current-frame decision.
This prevents current-to-current matching and future leakage.

## Ordered single-variable tests

1. **T05 Reliable-View Top-K.** Fresh same-root B05 and T05 differ only in
   output directory and `reliable_views.enabled`. Appearance gating is off.
2. **Offline error-budget audit.** Measure existing-box one-to-one recall,
   duplicate/fragmentation burden, IoU distribution, and a proposal/geometry
   oracle ceiling. This uses GT only after inference for diagnosis and never
   feeds GT into the online model. If the ceiling is below an absolute +10 AP
   target, a new frozen high-recall proposal source is mandatory.
3. **Proposal replay and Sshadow.** Qualify the v3 cache on three real scenes,
   then run SMOV fragment extraction as observer-only. Native output must be
   exactly unchanged; this stage has no AP claim.
4. **Graw.** Run the frozen unmatched-only Group3D matcher using raw 5 cm
   fragments. `Graw - E2` is the secondary-association effect.
5. **Gclean.** Use the identical matcher and proposal membership with
   SMOV-clean fragments. `Gclean - Graw` isolates depth-edge cleanup.
6. **Dassoc.** Use MV3DIS-inspired sparse-depth consistency only to verify or
   veto Group3D secondary matches. It cannot override native association.
7. **Mshadow then Mobb.** First attach a bounded voxel-hash object memory with
   no output change. Then test a same-yaw robust OBB alternative accepted only
   when held-out past/current depth support dominates the native box.
8. **Rshadow then Zbirth.** Observe unexplained depth components in a bounded
   asynchronous queue. Only candidates confirmed by a fixed past-only
   observer/supporter rule may become boxes.

At every active step, retain only the passing arm; do not stack a failed or
inconclusive module.

## Module suitability

| Candidate | Use | Reason |
| --- | --- | --- |
| Reliable-View Top-K-lite | First active test | Fixed NumPy geometry, causal, no new weights, low measured overhead |
| SMOV depth cleanup | Shadow/infrastructure | Can improve fragment quality, but cannot change AP alone |
| Group3D-lite voxel overlap | Highest-priority next active module | Unmatched-only association can remove fragmentation/duplicates without births or semantic changes |
| MV3DIS-lite depth consistency | Conservative verifier | Useful for occlusion-aware abstention; full scene-level mask matching is not copied |
| OnlineAnySeg-lite voxel memory | Observer, then geometry support | Full learned query tracker is not used; bounded hash memory alone has no AP effect |
| Robust OBB dual hypothesis | Conditional later test | May improve AP50, but the previous OpenBox-SMOV R2 rule already failed full100 |
| Zoo3D0 past-only supporter | Last-stage birth verifier | Full observer set is noncausal online; only past observers are allowed |
| OpenM3D-like residual proposals | Research risk, last | Full OpenM3D is learned; only a fixed bounded residual-depth idea could comply |
| B6-ID | Invariant only | Scores are overwritten to 1.0, so it cannot improve AP |
| Low-score append | Excluded | The evaluator destroys low-score suffix protection and tie sorting can reorder all boxes |

## Retention gates

For each active arm versus the immediately preceding retained arm:

- AP25 and AP50 deltas must be positive; AP15 must be at least -0.10 AP.
- Mean AP15/AP25/AP50 gain must be at least +0.20 AP point.
- Gain must exceed `max(0.10 AP, 2 * replay/repeat noise)`.
- In 10,000 paired scene bootstrap replicates, one of AP25/AP50 must have a
  nonnegative 95% lower bound and the other at least 90% positive replicates.
- Cumulative live frame-weighted FPS relative to B05 must remain at least 0.95;
  per-step ratios alone are insufficient because overhead accumulates.
- No future frame, GT, optimizer, target-scene checkpoint, unbounded state, or
  stale terminal proposal may be used.

The final absolute-10-AP claim is made only if all three AP targets are met on
a single retained full100 arm. Existing evidence shows Reliable Top-K is a
sub-one-to-low-one-point module and the previous OpenBox-SMOV geometry rule was
neutral/negative. Therefore the complete ten-point target is unlikely from
post-processing alone. It probably requires both a substantially stronger
high-precision frozen proposal source for AP15/AP25 and a new robust geometry
estimator for AP50; neither is yet qualified under constant-score evaluation.
