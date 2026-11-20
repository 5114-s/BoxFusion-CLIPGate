# CA-1M-107 strict-causal online no-child factorial

All eight arms are materialized from one complete online run. PLR uses
post-NMS proposals only; NMS-suppressed children are disabled.

| Configuration | Boxes | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| Base | 16,009 | 44.8850 | 36.7506 | 15.4454 |
| PLR | 16,729 | 45.7130 | 37.6202 | 15.8191 |
| CALR | 26,093 | 46.5258 | 38.2410 | 16.3106 |
| MVSR | 16,009 | 45.0503 | 37.0796 | 16.3050 |
| PLR+CALR | 26,813 | 46.8835 | 38.6844 | 16.5324 |
| PLR+MVSR | 16,729 | 45.8852 | 37.9564 | 16.6840 |
| CALR+MVSR | 26,093 | 46.6910 | 38.5700 | 17.1701 |
| PLR+CALR+MVSR | 26,813 | 47.0550 | 39.0203 | 17.3972 |

## Conditional effects

| Effect | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| PLR|base | +0.8281 | +0.8696 | +0.3737 |
| CALR|base | +1.6408 | +1.4904 | +0.8652 |
| MVSR|base | +0.1653 | +0.3290 | +0.8596 |
| MVSR|PLR | +0.1721 | +0.3362 | +0.8648 |
| MVSR|CALR | +0.1652 | +0.3290 | +0.8595 |
| MVSR|PLR+CALR | +0.1715 | +0.3360 | +0.8649 |
| CALR|PLR+MVSR | +1.1698 | +1.0639 | +0.7133 |
| PLR|CALR+MVSR | +0.3640 | +0.4503 | +0.2271 |
| full|base | +2.1700 | +2.2697 | +1.9518 |
