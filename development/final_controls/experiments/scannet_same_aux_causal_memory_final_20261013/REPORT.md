# ScanNet-100 same-auxiliary-stream causal score-memory baseline

All arms use the canonical 1,787 native boxes and the exact frozen WeDetect-Uni/Boxer candidate stream used by ReCaR-3D.

| Arm | Native/Proposal/Anchor | Total | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|---:|
| base | 1,787/0/0 | 1,787 | 34.9264 | 31.4995 | 15.8463 |
| simple_matched_native | 1,787/1,101/23,172 | 26,060 | 42.6814 | 38.3050 | 19.0707 |
| simple_matched_latest | 1,787/1,101/23,172 | 26,060 | 43.8897 | 39.5696 | 20.3619 |
| simple_matched_maximum | 1,787/1,101/23,172 | 26,060 | 46.6025 | 42.1615 | 21.9726 |
| simple_fixed_caps_latest | 1,787/1,200/53,524 | 56,511 | 44.0824 | 39.6938 | 20.3568 |
| full_recar | 1,787/1,101/23,172 | 26,060 | 47.0282 | 42.6101 | 22.1437 |

The score-memory state is strictly causal and bounded. The matched arm uses only ReCaR's per-scene branch output counts as a terminal evaluation budget; it does not reuse ReCaR-selected boxes or ground truth. The fixed-cap arm is fully standalone and uses the deployment caps 12/640.
