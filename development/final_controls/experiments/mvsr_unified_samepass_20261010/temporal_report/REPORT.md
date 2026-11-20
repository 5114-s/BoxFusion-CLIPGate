# ScanNet final three-factor MVSR temporal aggregation

All arms use proposal score, 2D overlap, and 3D geometry consistency;
visibility is excluded. Native geometry, labels, and box count are fixed.

| Aggregation | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| native | 34.9264 | 31.4995 | 15.8463 |
| latest | 36.1718 | 32.7843 | 17.1284 |
| mean | 36.2440 | 32.8768 | 16.9330 |
| ema | 36.3351 | 32.9833 | 17.1437 |
| top3_mean | 38.6848 | 35.1753 | 18.7043 |
| maximum | 38.8707 | 35.3637 | 18.7320 |
