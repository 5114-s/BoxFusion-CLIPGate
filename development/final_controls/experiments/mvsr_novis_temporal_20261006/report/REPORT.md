# ScanNet final three-factor MVSR temporal aggregation

All arms use proposal score, 2D overlap, and 3D geometry consistency;
visibility is excluded. Native geometry, labels, and box count are fixed.

| Aggregation | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| native | 34.8964 | 31.4686 | 15.7577 |
| latest | 36.1398 | 32.7542 | 17.0465 |
| mean | 36.2053 | 32.8403 | 16.8557 |
| ema | 36.2999 | 32.9526 | 17.0667 |
| top3_mean | 38.6402 | 35.1309 | 18.6252 |
| maximum | 38.8230 | 35.3167 | 18.6460 |
