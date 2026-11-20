# ScanNet-100 MVSR distinct-view audit

All arms use the final no-visibility support. Native geometry and row count are fixed.
The shuffled negative permutes the single-view logit corrections among native identities within each scene.

| Arm | Activated native rows | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| native | 0 | 35.1552 | 31.5693 | 15.8213 |
| single_view | 1284 | 39.0928 | 35.4750 | 18.7296 |
| two_view | 1252 | 39.1173 | 35.5083 | 18.6961 |
| three_view | 1176 | 39.1568 | 35.4878 | 18.7125 |
| shuffled_negative | 1284 | 35.7854 | 31.9421 | 16.4433 |
