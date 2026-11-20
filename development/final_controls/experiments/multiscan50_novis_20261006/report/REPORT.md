# MultiScan-50 AP

## nonstructural

GT boxes: 1614. all annotated object instances except beam/ceiling/floor/pillar/remove/wall/windowsill.

| Arm | AP15 | AP25 | AP50 | Predictions |
|---|---:|---:|---:|---:|
| previous_four_factor | 41.65 | 36.25 | 18.99 | 27182 |
| paired_four_factor | 41.79 | 36.38 | 18.90 | 27183 |
| base | 24.80 | 23.36 | 13.72 | 621 |
| plr | 32.81 | 29.39 | 15.61 | 1221 |
| calr | 39.35 | 34.57 | 17.99 | 26583 |
| mvsr | 25.52 | 24.10 | 14.67 | 621 |
| plr_calr | 41.22 | 35.72 | 18.16 | 27183 |
| plr_mvsr | 33.53 | 30.13 | 16.57 | 1221 |
| calr_mvsr | 40.07 | 35.32 | 18.95 | 26583 |
| full_no_visibility | 41.96 | 36.49 | 19.12 | 27183 |

## scannet18

GT boxes: 441. conservative MultiScan-label mapping to ScanNet18 object classes.

| Arm | AP15 | AP25 | AP50 | Predictions |
|---|---:|---:|---:|---:|
| previous_four_factor | 41.21 | 37.63 | 23.60 | 27182 |
| paired_four_factor | 41.58 | 38.02 | 23.18 | 27183 |
| base | 33.35 | 30.33 | 18.79 | 621 |
| plr | 38.26 | 34.11 | 20.07 | 1221 |
| calr | 38.80 | 34.89 | 21.79 | 26583 |
| mvsr | 35.93 | 33.40 | 21.49 | 621 |
| plr_calr | 40.36 | 36.18 | 21.68 | 27183 |
| plr_mvsr | 40.84 | 37.18 | 22.77 | 1221 |
| calr_mvsr | 41.38 | 37.96 | 24.49 | 26583 |
| full_no_visibility | 42.95 | 39.28 | 24.40 | 27183 |

