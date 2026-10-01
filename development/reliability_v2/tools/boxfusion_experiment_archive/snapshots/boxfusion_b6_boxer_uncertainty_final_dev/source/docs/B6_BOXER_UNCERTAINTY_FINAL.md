# B6 + fixed-Top-K final-only Boxer uncertainty

This isolated ablation removes the two confounders observed in the previous
U2 route: uncertainty is no longer allowed to change Top-K membership, and no
uncertainty geometry is visible to online association or B6 scoring.

## Frozen data flow

```text
Selective Boxer G0 online association/fusion
  -> record immutable G0 Top-K recipe
  -> B6 quality score
  -> baseline ScanNet minimum-extent filter
  -> fixed-member Boxer confidence reweighting
  -> final geometry only
```

For a selected Boxer view, its G0 weight is multiplied by
`q = 1 / (1 + sigma^2)`. CuTR fallback rows and invalid confidence values use
a neutral factor of one. The selected G0 source rows and their order are
immutable. The adjusted selected weights are normalized to unit mean before
the released projection objective is evaluated.

The final optimizer starts from the final G0 box tensor and its frozen
rotation. Search state is local to the final candidate; it cannot write
`all_pred_box`, `BoxManager`, the online controller, or G0 statistics.

## Profiles

- `f0_control`: final module disabled; frozen B6 + Selective Boxer G0.
- `f1_observer`: calculate candidates and diagnostics, export G0 geometry.
- `f2_active`: export accepted candidate geometry only.

All profiles keep `score_thresh=0.4`, `minimum_extent=0.4`, Selective Boxer G0
gate `center<=0.10 m`, volume ratio `[0.50, 2.00]`, B6 detector blend `0.40`,
and reliable-view `Top-K=3`.

## Protected-field contract

The baseline geometry determines the final minimum-extent mask before this
module runs. The module is then forbidden to filter, append, sort, suppress,
or rescore detections. Every observer/active scene records and checks:

- identical prediction count;
- byte-identical B6 scores;
- byte-identical source-index order;
- byte-identical stable-ID order;
- zero Top-K selection or ranking changes;
- observer corners identical to the baseline.

Any row-level mapping or numeric failure keeps the baseline row. A scene-level
contract failure restores the complete baseline geometry.

## Fixed-10 protocol

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_boxer_uncertainty_final_dev

bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f0_control 0,1
bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f1_observer 0,1
bash scripts/audit_scannet_b6_boxer_uncertainty_final.sh f1_observer
bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f2_active 0,1
bash scripts/audit_scannet_b6_boxer_uncertainty_final.sh f2_active
```

Proceed to 100 scenes only if the pre-registered fixed-10 gate is met:

```text
delta AP50 >= +0.5
delta AP25 >=  0.0
delta AP15 >= -0.3
all protected-field audits pass
```

Then run:

```bash
BOXFUSION_B6_BOXER_FINAL_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
  bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f2_active 0,1

BOXFUSION_B6_BOXER_FINAL_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
  bash scripts/audit_scannet_b6_boxer_uncertainty_final.sh f2_active
```

This implementation establishes a clean causal ablation; it does not imply
that AP will exceed B6 until the fixed-10 GPU result is measured.
