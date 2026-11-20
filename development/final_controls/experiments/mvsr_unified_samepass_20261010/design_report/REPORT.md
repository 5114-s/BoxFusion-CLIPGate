# ScanNet MVSR design ablation

All arms use the same strictly causal online evidence pass and the exact same native boxes, labels, and box count; only native confidence scores differ.

## Composite-support factors

| Arm | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| native | 34.9264 | 31.4995 | 15.8463 |
| only_2d_iou | 37.2778 | 33.6238 | 17.2112 |
| no_proposal_score | 36.0862 | 32.7230 | 16.7938 |
| no_visibility | 38.8707 | 35.3637 | 18.7320 |
| no_center_consistency | 38.2454 | 34.9455 | 18.3209 |
| no_scale_consistency | 38.2671 | 34.9669 | 18.3333 |
| no_overlap_consistency | 38.2611 | 34.9203 | 18.2346 |
| full | 38.2735 | 34.9719 | 18.3073 |

## Temporal aggregation

| Arm | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| latest | 35.5538 | 32.1450 | 16.6005 |
| mean | 35.7252 | 32.4180 | 16.6092 |
| ema | 35.6939 | 32.4226 | 16.6363 |
| top3_mean | 38.0778 | 34.7406 | 18.2829 |
| maximum | 38.2735 | 34.9719 | 18.3073 |
