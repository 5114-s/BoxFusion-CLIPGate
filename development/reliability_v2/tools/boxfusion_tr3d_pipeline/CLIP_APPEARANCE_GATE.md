# Score-preserving CLIP instance-appearance gate

This root-level BoxFusion experiment is based on the completed
score-preserving ScanNet reproduction:

```text
AP@0.15 = 32.53
AP@0.25 = 27.11
AP@0.50 =  9.46
```

The baseline evidence remains at:

- predictions: `upstream_clean/scorefix_results/scannet`
- driver log: `upstream_clean/scorefix_logs/full_driver.log`
- input frames: `upstream_clean/scannet_readme_frames`

The CLIP-gate experiment deliberately reuses the same input-frame pipeline,
detection threshold, keyframe gap, fusion settings, real Cubify scores, and
score-aware evaluation. Its outputs are isolated under:

- predictions: `results/scannet_clip_gate_scorefix`
- driver log: `logs/scannet_clip_gate_scorefix/driver.log`
- scene logs: `logs/scannet_clip_gate_scorefix/scenes`
- evaluation log: `logs/scannet_clip_gate_scorefix/eval_stdout.log`

## Association change

Each new proposal crop is encoded once by CLIP. The normalized image feature
is stored as `appearance_features`. Original 3D overlap and projected 2D IoU
still generate candidates first. CLIP cosine similarity then continuously
adjusts the required geometric IoU:

```text
effective threshold =
    original threshold
  + confidence * dissimilarity penalty
  - confidence * similarity bonus
```

The first ablation uses `max_iou_bonus: 0.0`; appearance may protect against a
suspicious merge but cannot create a merge rejected by original geometry.
Low-confidence detections fall back toward the original threshold, and
`hard_geometry_iou` lets strong geometry override a poor image crop.

This first version keeps one surviving observation as the global appearance
feature. Multi-view feature memory is a separate future ablation.

## Commands

Tests:

```bash
cd /data/ZhaoX/BoxFusion
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/admin1/miniconda3/envs/boxfusion2/bin/python \
  -m pytest -q tests/test_clip_appearance_gate.py
```

Single GPU:

```bash
cd /data/ZhaoX/BoxFusion
bash scripts/run_scannet_clip_gate_scorefix.sh 0
```

Two independent scene workers on GPU 0 and GPU 1:

```bash
cd /data/ZhaoX/BoxFusion
bash scripts/run_scannet_clip_gate_scorefix.sh 0,1
```

The two-GPU mode splits the fixed 100-scene list by index, so workers never
write the same scene. Evaluation runs once after both workers finish. Existing
prediction files are skipped for safe resume.
