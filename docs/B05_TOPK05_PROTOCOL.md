# Reliable Top-K at score_thresh=0.5: B05/T05 audit protocol

## What is already established

The historical score-0.5 output is
`upstream_clean/scorefix_results/scannet`, with 100 official scenes and 1,786
predictions.  The frozen constant-score evaluator reports:

| arm | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| historical B05, score forced to 1.0 | 29.690655 | 24.828889 | 7.704887 |
| same rows, disk scores retained | 32.5338 | 27.1139 | 9.4641 |

The paired-bootstrap candidate replayed the historical B05 and obtained
`0.296906550288 / 0.248288894196 / 0.077048873804`, exactly matching the
official constant-score log to its printed six decimals.  The official
constant evaluator is
`upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py`, SHA256
`aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27`.

## Formal arms

Do not compare a newly run T05 only against the July historical output for a
final module claim.  Run both arms with the same current code revision:

* **B05**: score threshold 0.5, appearance gate off, Reliable Top-K off.
* **T05**: score threshold 0.5, appearance gate off, Reliable Top-K3 on.

The formal `config/scannet_b05_control_score05.yaml` and
`config/scannet_topk_fusion_score05.yaml` files must differ semantically only
in `data.output_dir` and `box_fusion.reliable_views.enabled`. In particular,
both contain the same entire frozen Reliable-TopK parameter block, so an
omitted-default difference cannot contaminate the comparison.

Freeze before B05 begins:

1. ordered official 100-scene list and its SHA256;
2. both YAML files and a semantic-diff receipt;
3. `demo.py`, `boxfusion/box_manager.py`, `boxfusion/instances.py`,
   `boxfusion/box_fusion.py`, `boxfusion/reliable_views.py`;
4. CuTR and CLIP checkpoint hashes, Python environment, CUDA/device mapping;
5. the constant-score evaluator hash above.

Use the same GPU list and the same modulo scene-to-GPU assignment for both
arms.  Reject any pre-existing prediction in either output root, rather than
silently resuming artifacts produced by a different code/config hash.  Run B05
then T05 without editing the frozen files.  Each arm must have exactly one
regular prediction file for every ordered scene and no extra scene file.

The preferred design replays one exact proposal cache into both arms.  If
proposal replay is not yet available, rerunning CuTR separately means the AP
contrast includes detector/GPU nondeterminism.  In that case a second current
B05 repeat is required for a strict causal claim; otherwise label T05-B05 as a
conditional single-run result.

## Evaluation and statistics

Run `scripts/eval_scannet_cgf_paper100_constant_score.sh` separately on B05
and T05.  Disk scores may remain available for audit, but the headline result
must use the evaluator's in-memory score 1.0.

Then run 10,000 paired scene-bootstrap replicates with seed 20260822:

```bash
/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  /data/ZhaoX/BoxFusion/tools/scannet_b05_t05_paired_bootstrap.py \
  --b05 /data/ZhaoX/BoxFusion/results/scannet_b05_control_score05 \
  --t05 /data/ZhaoX/BoxFusion/results/scannet_topk_fusion_score05 \
  --replicates 10000 --seed 20260822 \
  --out /data/ZhaoX/BoxFusion/logs/cgf_score05_topk/scannet_b05_t05_paired_bootstrap_10000.json
```

`--baseline/--treatment` are exact aliases for `--b05/--t05`, matching the
automated continuation queue.

This bootstrap preserves the official global all-equal-score quicksort tie
behavior.  A per-scene AP bootstrap would be a different metric and must not
be substituted.

Pre-registered module-effect gate (AP values are percentage points):

* observed AP25 and AP50 deltas are positive;
* AP15 delta is at least -0.10;
* mean of AP15/AP25/AP50 deltas is at least +0.20;
* for AP25/AP50, at least one 95% percentile lower bound is nonnegative and
  the other has at least 90% positive bootstrap probability.

For separately rerun proposals, define each threshold's empirical inference
noise as the largest absolute B05-to-B05-repeat AP drift.  Require the active
gain to exceed `max(0.10 AP point, 2 * noise)` before attributing it to Top-K.
The paired bootstrap captures scene sampling uncertainty, not GPU inference
nondeterminism.

The user's overall target is an absolute +10 AP points at AP15, AP25, and
AP50.  Relative to historical B05, the route-level targets are
39.690655/34.828889/17.704887.  This is not the retention gate for this single
module.  Report both absolute AP-point and relative-percent changes, while
judging the final route against the absolute targets.

## Online/realtime gate

Parse one final `Cost: ... Average FPS: ...` record from each of the 100 scene
logs.  Primary throughput is frame-weighted FPS, computed as
`sum(cost_i * fps_i) / sum(cost_i)`; report mean-scene FPS and paired scene
ratios as diagnostics.

Retain the module only if:

* T05/B05 frame-weighted FPS is at least 0.95;
* the median paired scene FPS ratio is at least 0.95;
* no scene errors/OOMs occur and all Top-K summaries have finite statistics;
* inference remains causal and no GT is read before offline evaluation.

Historical B05 throughput was approximately 15.1564 frame-weighted FPS and
15.7809 mean-scene FPS.  It is a context value, not the formal denominator;
the fresh current-code B05 is the runtime denominator.
