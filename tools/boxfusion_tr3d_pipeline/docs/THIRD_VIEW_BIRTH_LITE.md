# Third-View Birth Lite (shadow only)

## Scope

This is a training-free, causal confirmation observer for future supplemental
births. It does **not** filter CuTR/BoxFusion anchors. The native keep mask is
always identity, and the exported prediction remains the native output.

Applying a three-view filter to all native boxes is unsafe: in the current
fixed-10 evidence only 79 of 355 final native fusion groups contain at least
three source proposals, while 152 boxes survive native post-processing. In
`scene0598_01`, only four final native groups have three sources although 16
boxes are exported. A global filter would therefore destroy recall.

## Frozen causal rule

- Observe only real CuTR keyframe commits; never observe the terminal branch
  when it reuses stale proposals.
- Map every source init ID to its recorded source frame and collapse repeated
  proposals from the same frame.
- A lineage is probationary with one or two distinct source frames and becomes
  confirmed on the third distinct source frame.
- Confirmation is sticky across an unambiguous native merge.
- Stable IDs come from `CausalFusionIdRegistry`, not a stateless minimum-ID
  reconstruction.
- Invalid source mappings, duplicate stable IDs, ambiguous splits, or resource
  bound violations fail closed during shadow testing.

The current core is only native witness memory. The companion side-birth
probation ledger enrolls PUF arbitration `birth` recommendations and tests
whether their committed causal identity is observed in three distinct true
CuTR keyframes. Native target labels are recorded only after association for
offline diagnostics and never influence the state transition.

The ledger therefore records `puf_event_source=true`: its caller supplies
already-screened birth events. `puf_access=false` and `puf_state_access=false`
mean that the ledger itself neither imports nor queries PUF state; they do not
mean that the seed events were produced independently of PUF.

## Safety contract

- no training, learned parameters, GT, detector scores, CLIP, or online update;
- at most 1,024 native witness tracks and five source IDs per group;
- `min_distinct_source_frames=3` is frozen when enabled;
- `observer_only=true`, `active_authorized=false`;
- `native_filter_applied=false`, `native_outputs_mutated=false`;
- future output, if ever authorized, must be append-only: all native rows first
  and unchanged, followed only by confirmed, independently novel side boxes.

Use
`config/scannet_qim_puf_arbitration_third_view_shadow.yaml` for the isolated
real-stream ablation. The log contains the machine-readable prefix
`Third-view-birth-lite shadow JSON | `.

## What this can establish

The shadow run can establish causal persistence, bookkeeping latency, and
whether a bad proposed birth would have remained on probation. It cannot
establish an AP improvement because it deliberately changes no prediction.
Actual recall gains require the later asynchronous missed-object proposer plus
an append-only novelty check; the third-view rule protects precision in that
branch.

## Real S1 shadow result

On `scene0598_01` (829 input frames and 34 true CuTR keyframes), the ledger
observed 25 PUF-arbitration birth events. Native association later classified
24 as births and one as unique history. Three events reached a third distinct
CuTR keyframe and all three were native-relative births, but this retains only
3/24 events (12.5%), or 3/22 unique stable identities (13.64%). These are
native-association proxy labels, not ground truth.

The known frame-175/proposal-18 history duplicate was joined exactly once to
stable ID 15, had only the frame-175 event-relative witness, never confirmed,
and retired at frame 450 after the frozen ten-keyframe TTL. Thus the rule has a
useful precision-protection signal, but it is too strict to gate every PUF
birth: it would suppress 87.5% of the native-relative birth events in this
scene. It remains restricted to future asynchronous supplemental candidates.

The observer ran at 31.80 FPS versus 31.39 and 32.35 FPS for two same-code
controls. Relative to the warm control, labels and scores are exact and the
maximum corner difference is 3.624e-5 m. A control-to-control repeat differed
by 2.918e-3 m, establishing native cross-process numerical nondeterminism; no
bitwise identity is claimed. Deterministic single-scene evaluation produced
identical AP15/AP25/AP50 (0.277389/0.277389/0.030303), so the shadow-stage AP
gain is exactly zero as intended. The machine-readable paired report is
`reports/side_birth_probation/scene0598_s1_v1.json`.
