# ScanNet MVSR design ablation

All arms use the same strictly causal online evidence pass and the exact same native boxes, labels, and box count; only native confidence scores differ.

## Composite-support factors

| Arm | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| native | 34.9925 | 31.4534 | 15.7216 |
| only_2d_iou | 37.3040 | 33.5839 | 17.0794 |
| no_proposal_score | 36.1136 | 32.6752 | 16.6207 |
| no_visibility | 38.9071 | 35.3360 | 18.5878 |
| no_center_consistency | 38.2784 | 34.9088 | 18.1850 |
| no_scale_consistency | 38.2995 | 34.9309 | 18.1978 |
| no_overlap_consistency | 38.2947 | 34.8870 | 18.0981 |
| full | 38.3048 | 34.9355 | 18.1718 |

## Temporal aggregation

| Arm | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| latest | 35.5928 | 32.0937 | 16.4855 |
| mean | 35.7507 | 32.3700 | 16.4876 |
| ema | 35.7589 | 32.3841 | 16.5145 |
| top3_mean | 38.0895 | 34.6832 | 18.1449 |
| maximum | 38.3048 | 34.9355 | 18.1718 |
