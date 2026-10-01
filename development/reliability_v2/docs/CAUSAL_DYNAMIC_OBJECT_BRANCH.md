# Causal Dynamic Object Branch

## Scope

This branch addresses the released BoxFusion limitation that a moving object is
repeatedly fused into a static world-space map. It remains online,
reconstruction-free, training-free, and bounded. Native 3D NMS and explicit 2D
matching are retained.

```text
CuTR + Boxer proposal
  -> native 3D NMS + 2D Matching
  -> one representative observation per native row
  -> bounded causal dynamic object state
       geometry/optional cached CLIP association + explicit null
       constant-velocity current state
       static / uncertain / dynamic posterior
       latched dynamic identity after confirmed motion
       visible / occluded / out-of-view / unknown evidence
       tentative / confirmed / occluded / dormant / retired lifecycle
  -> static track: native all-history PFO
  -> confirmed dynamic track: bypass all-history PFO, use W<=5 current state
  -> deduplicated persistent-anchor map + separate current-state map
```

The implementation is in:

- `boxfusion/causal_dynamic_objects.py`: independent bounded state machine;
- `boxfusion/causal_dynamic_branch.py`: BoxFusion observation, visibility,
  alias, output, and audit adapter;
- `demo.py`: keyframe and terminal wiring;
- `boxfusion/box_fusion.py`: dynamic-only PFO bypass.

## Causal and memory contract

- Query reads only committed keyframes. The current keyframe is installed only
  by the exact-token `commit_frame` call.
- Raw frame IDs such as `0,25,50` are recorded for audit, but motion uses the
  continuous processed-keyframe ordinal `0,1,2`.
- Observation history is at most 5 rows per logical object.
- View-compatible appearance memory is at most 4 descriptors per object.
- Native-row aliases are at most 16 per logical object.
- Tracks and observations per keyframe have explicit hard caps.
- Exact appearance-aware association is restricted to the bounded union of
  symmetric geometry Top-4 and appearance Top-4 neighbourhoods; assignment
  uses the CPU SciPy solver when present and a deterministic NumPy fallback
  otherwise.
- When full, only an unconfirmed tentative or retired track can be evicted;
  a current-frame matched or newly created track is protected.
- Occlusion and out-of-view status do not reduce existence. Only a conservative
  RGB-D free-space contradiction counts as visible negative evidence.
- Motion extrapolation is capped at three unobserved processed keyframes. The
  track then becomes dormant without score decay, avoiding unbounded drift
  while retaining its persistent-map identity.

## Output contract

`data.output_dir` is the persistent/static-map view. Native association, row
labels, and every static/unclaimed row are retained. In active mode, a latched
dynamic identity stops accepting all-history PFO updates; multiple native
aliases are collapsed to one row whose geometry comes from the last trusted
pre-dynamic anchor and whose score is the maximum real native observation
score. Consequently, active persistent output can change row count, geometry,
and score for dynamic identities only. Shadow and disabled modes remain
output-inert.

`causal_dynamic_branch.current_output_root` is the current-state map. In active
mode it:

- replaces center/size only for a confirmed dynamic track using its W<=5 state;
- preserves the latest observation's full rotation, with only the estimated yaw
  delta applied;
- keeps one latest native alias when a moving object created several static-map
  rows;
- removes dormant/retired dynamic identities and rows rejected by visible
  free-space evidence, while retaining static objects that are merely outside
  the current camera view;
- leaves every static or unclaimed native row unchanged.

Shadow mode computes and logs decisions but never bypasses PFO or changes a
materialized output. Disabled mode is the default and does not request extra
appearance features.

## Reusing open-vocabulary evidence

When `use_appearance: true`, the branch reuses the frozen CLIP crop descriptor
already supported by BoxFusion. It does not add SAM, optical flow, a detector,
training, or dense reconstruction. To preserve online cost, normal continuing
observations use geometry; CLIP features are reused from the native semantic
forward only for the first frame and NMS-retained new rows, where they are
needed for dormant-track relocation. The dynamic branch therefore does not add
a pre-NMS all-proposal CLIP pass.

## Running the synthetic dynamic set

The checked-in active configuration is
`config/scannet_dyn_causal_dynamic_active.yaml`. Run all scenes declared by the
manifest with:

```bash
bash scripts/run_scannet_causal_dynamic_full75.sh
```

The launcher writes persistent and current predictions plus one JSONL identity
ledger per scene. It then invokes a fail-closed coverage check. Missing, extra,
duplicate, empty, or truncated ledgers without a matching terminal summary
cause a non-zero exit instead of being silently treated as complete output.

Run the implementation tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n boxfusion2 \
  pytest -q tests/test_causal_dynamic_objects.py \
  tests/test_causal_dynamic_branch.py \
  tests/test_validate_dynamic_run_coverage.py
```

## Evaluation boundary

Static ScanNet class-agnostic AP should be reported on the persistent output.
The current output needs a dynamic benchmark with time-indexed ground truth and
must report at least current-state AP, stale-old-location rate, duplicate rate,
same-ID relocation recovery, ID switches, fragmentation, retirement latency,
and end-to-end FPS/P95 latency.

The existing synthetic removal/move benchmark is useful for regression testing,
but it is not evidence of robustness to naturally moving or non-rigid humans.
That claim requires a complete benchmark run and natural dynamic sequences.
Likewise, the bounded-core microbenchmark is not an end-to-end FPS result; FPS
must be measured by the complete launcher on the target GPU and scene manifest.
