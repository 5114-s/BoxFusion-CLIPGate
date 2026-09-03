# R5: true SPGroup3D local-grouping observer

R5 evaluates whether official SPGroup3D grouping evidence can identify harmful
or beneficial R3 local box replacements.  It is isolated from the live
BoxFusion trees and is strictly observer-only.

## What is genuinely reused

- Official non-overlapping `segmentator.segment_mesh` superpoints.
- Official pretrained `BiResNet` backbone.
- Official geometry-aware voting.
- Official `k=8` superpoint attention.
- Official three-stage superpoint/voxel fusion, producing 390-D group features.

The source is pinned to commit
`181283547323d3bd54d0e9f58baf0cd413ccc107`; the checkpoint is pinned to
SHA256 `cabd9f88da3bf41dcb8aa46696d47aaa7c94913a3086f9404374a0b149714edf`.
The mesh segmentator is pinned to commit
`4c6126551685166c6c300551e9ad63db988928c4` and its locally compiled binding
to SHA256 `41e0ba70e8cbdd771aecad6157d5c671327c12a56e672af80496aa34b54f4cc8`;
the cache manifest discloses the Python-3.10/C++17 compatibility patches.
The ScanNet-18 `SPHead` is deliberately not loaded.  CLIP remains the semantic
head in any future BoxFusion route.

## Safety and interpretation

R5 does not write prediction files and verifies the R3 prediction-tree hash
before and after feature extraction and pair observation.  Its normal path has
no ground-truth or CLIP access.  Ground truth is confined to a separate offline
counterfactual audit.

This first implementation consumes the reconstructed ScanNet triangle mesh to
obtain the exact official partition.  It is therefore an offline experiment,
not an online-speed claim.  A positive result must later be reproduced with an
incremental causal superpoint builder before activation.

The official repository is CC BY-NC 4.0; this isolated route is for
non-commercial research and preserves source/checkpoint attribution.

## Fixed-10 command

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_spgroup_dev
bash scripts/run_tr3d_r5_spgroup_fixed10.sh 0 r5_spgroup_fixed10_v1
```

The two veto rules are pre-registered zero-threshold rules.  Neither rule is
authorized for active predictions by the fixed-10 audit; held-out confirmation
is mandatory.
