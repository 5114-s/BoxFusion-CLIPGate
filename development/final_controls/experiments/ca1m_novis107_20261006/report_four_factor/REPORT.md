# CA-1M-107 strict-causal online no-child factorial

All eight arms are materialized from one complete online run. PLR uses
post-NMS proposals only; NMS-suppressed children are disabled.

| Configuration | Boxes | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| Base | 16,009 | 44.8850 | 36.7506 | 15.4454 |
| PLR | 16,729 | 45.7130 | 37.6202 | 15.8191 |
| CALR | 26,093 | 46.5258 | 38.2410 | 16.3106 |
| MVSR | 16,009 | 45.0886 | 37.1276 | 16.3049 |
| PLR+CALR | 26,813 | 46.8835 | 38.6844 | 16.5324 |
| PLR+MVSR | 16,729 | 45.9313 | 38.0087 | 16.6839 |
| CALR+MVSR | 26,093 | 46.7293 | 38.6179 | 17.1700 |
| PLR+CALR+MVSR | 26,813 | 47.1002 | 39.0711 | 17.3967 |

## Conditional effects

| Effect | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| PLR|base | +0.8281 | +0.8696 | +0.3737 |
| CALR|base | +1.6408 | +1.4904 | +0.8652 |
| MVSR|base | +0.2036 | +0.3770 | +0.8595 |
| MVSR|PLR | +0.2182 | +0.3885 | +0.8648 |
| MVSR|CALR | +0.2035 | +0.3769 | +0.8594 |
| MVSR|PLR+CALR | +0.2167 | +0.3868 | +0.8643 |
| CALR|PLR+MVSR | +1.1690 | +1.0625 | +0.7128 |
| PLR|CALR+MVSR | +0.3709 | +0.4532 | +0.2267 |
| full|base | +2.2153 | +2.3205 | +1.9513 |
