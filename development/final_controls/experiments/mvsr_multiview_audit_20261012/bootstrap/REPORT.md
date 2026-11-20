# scannet final paired scene bootstrap

B=2000, seed=20261012; values are AP points.

| Contrast | IoU | Point | 95% CI | P(Δ>0) |
|---|---:|---:|---:|---:|
| single_view − base | 0.15 | +3.9376 | [+2.9073, +4.8458] | 1.0000 |
| single_view − base | 0.25 | +3.9057 | [+2.8881, +4.7757] | 1.0000 |
| single_view − base | 0.50 | +2.9083 | [+2.1452, +3.6922] | 1.0000 |
| two_view − base | 0.15 | +3.9621 | [+2.9695, +4.8515] | 1.0000 |
| two_view − base | 0.25 | +3.9390 | [+2.9654, +4.8050] | 1.0000 |
| two_view − base | 0.50 | +2.8748 | [+2.1185, +3.6673] | 1.0000 |
| three_view − base | 0.15 | +4.0016 | [+2.9956, +4.8676] | 1.0000 |
| three_view − base | 0.25 | +3.9184 | [+2.9173, +4.8056] | 1.0000 |
| three_view − base | 0.50 | +2.8912 | [+2.1236, +3.6555] | 1.0000 |
| shuffled_negative − base | 0.15 | +0.6302 | [-0.1932, +1.4416] | 0.9375 |
| shuffled_negative − base | 0.25 | +0.3728 | [-0.4196, +1.2571] | 0.8280 |
| shuffled_negative − base | 0.50 | +0.6220 | [+0.0660, +1.2669] | 0.9840 |
