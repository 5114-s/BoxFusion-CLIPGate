# P1-v2 snapshot-target / native-sparse ablation

This checkout isolates two changes to the frozen B6 residual-proposal
observer.  Neither stage may change formal BoxFusion boxes, labels, scores,
count, or order.

| Stage | Head | Train-only target scope | Only change from prior stage |
|---|---|---|---|
| P1 | `per_voxel_mlp` | `scene_global` | Historical residual observer |
| P1R | `per_voxel_mlp` | `snapshot_inside_only` | Snapshot-local target assignment |
| P1S | `native_sparse_context_v1` | `snapshot_inside_only` | Native sparse spatial context |

P1R exists because historical P1 assigns Top-K after concatenating every
provider step in a scene.  Repeated views of the same GT compete for a
scene-global maximum of K positives, so otherwise valid observations in later
steps become negatives.  P1R changes only that target contract.  P1S must use
the exact P1R dataset/loss/decoder/NMS contract and changes only the head.

## Frozen safety contract

Both stages require:

```text
observer_only=true
mutation_enabled=false
applied_count=0
uses_ground_truth=false
class_agnostic=true
regression_dim=6
```

P2 occupancy, P2V2 local geometry, P2V3 fusion, refit, supplemental output,
Soft-NMS, and every older experimental mutation are disabled.

The checkpoint and every diagnostic must expose the same exact fields:

```text
model_config.head_architecture       # per-voxel P1R
model_config.architecture            # strict native sparse P1S
training_config.target_assignment_scope
p1_head_architecture
p1_target_assignment_scope
```

The run manifest binds those fields, B6/P1 checkpoint hashes, the canonical
forbidden validation-list hash, assets, code tree, scene list, and output
roots.  Resume is rejected if any bound input changes.

P1R/P1S checkpoint loading also recomputes
`train_scene_ids ∩ scannetv2_val`; an empty `forbidden_overlap` value stored
inside a checkpoint is not trusted by itself.

## Run order

Train P1R using only ScanNet train diagnostics and its fixed train-only
scene-disjoint development split.  Then run:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_p1v2_sparse_dev

BOXFUSION_P1V2_CHECKPOINT="$PWD/models/scannet_p1r_snapshot_inside.pt" \
  bash scripts/run_scannet_p1v2_ablation.sh P1R 0,1

bash scripts/audit_scannet_p1v2.sh P1R
```

Only if P1R passes the frozen gate, train P1S against the identical P1R
targets and run:

```bash
BOXFUSION_P1V2_CHECKPOINT="$PWD/models/scannet_p1s_native_sparse.pt" \
  bash scripts/run_scannet_p1v2_ablation.sh P1S 0,1

bash scripts/audit_scannet_p1v2.sh P1S
```

Full validation is opt-in and is forbidden before the fixed10 gate passes:

```bash
BOXFUSION_P1V2_FULL100=1 \
BOXFUSION_P1V2_CHECKPOINT=/absolute/checkpoint.pt \
  bash scripts/run_scannet_p1v2_ablation.sh P1S 0,1
```

## Pre-registered fixed10 gate

All conditions must pass:

- exact 10-scene prediction/diagnostic set;
- no failed observer step;
- mutation disabled and applied count zero;
- `B6 ∪ candidate` novel recall gain at IoU 0.25 at least 3 pp;
- novel recall gain at IoU 0.50 at least 1 pp;
- P1R: AP25 novel TP no lower than P1 and at least two additional AP50 TP;
- P1S: AP25 novel TP no lower than P1R and at least one additional AP50 TP;
- no more than 256 candidates/scene;
- observer runtime no more than 0.80 seconds/scene.

Failure returns `STOP_P1R` or `STOP_P1S`.  Do not run 100 scenes and do not
tune thresholds or retrain from the fixed validation-10 report.

Cross-run pickle byte identity is recorded but is informational because the
upstream CUDA B6 path has an already measured P0-repeat drift.  Safety is
established by the in-process observer contract and artifact validator, not
by forgiving a mutation as “nondeterminism.”

## Frozen fixed10 result (2026-07-30)

Both observers completed the pre-registered ten-scene protocol without
mutating a B6 prediction.  The standard ScanNet output was identical for
P1R and P1S (`42.1274 / 36.7839 / 18.0457`); this is an observer-safety
check, not proposal-head AP.

| Stage | Candidates / scene | Head seconds / scene | Novel TP @0.15 | Novel TP @0.25 | Novel TP @0.50 | Decision |
|---|---:|---:|---:|---:|---:|---|
| P1R | 201.7 | 0.5409 | 21 | 10 | 0 | `STOP_P1R` |
| P1S | 219.3 | 0.6088 | 25 | 15 | 0 | `STOP_P1S` |

P1S therefore improved `B6 ∪ P1` recall at IoU 0.25 by 10.07 percentage
points on this diagnostic set, compared with 6.71 points for P1R, while
neither head produced a new TP at IoU 0.50.  Among the 105 GT boxes missed
by B6 at IoU 0.50, P1S had 23 best candidates at IoU at least 0.25, eight
at least 0.35, three at least 0.40, and one at least 0.45; the maximum was
0.4806.  This identifies centre/extent regression quality, rather than
candidate recall or score thresholding, as the next bottleneck.

Per the frozen protocol, neither stage advances to the 100-scene validation
run and no threshold is tuned from this ten-scene result.  The complete
machine-readable reports are:

- `reports/p1v2_ablation/p1r_ablation10_b6frozen_v1/recall.json`
- `reports/p1v2_ablation/p1s_ablation10_b6frozen_v1/recall.json`

## Core-field dependency

The outer protocol assumes the residual core/trainer provide the four
architecture/target fields listed above and set `p1_stage/p1_profile` to the
active P1R/P1S values.  The validator fails closed if these fields are absent;
it does not infer P1R/P1S from a checkpoint filename.
