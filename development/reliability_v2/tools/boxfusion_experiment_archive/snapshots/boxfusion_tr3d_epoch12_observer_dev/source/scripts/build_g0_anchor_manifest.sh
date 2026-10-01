#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
G0_ROOT="${BOXFUSION_G0_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev}"
RESULT_ROOT="${BOXFUSION_G0_RESULT_ROOT:-$G0_ROOT/results/b6_selective_boxer/s1_selective/scannetv2_val-4b18fc586f7a}"
SCENE_LIST="${BOXFUSION_G0_SCENE_LIST:-$G0_ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
OUTPUT="${BOXFUSION_G0_MANIFEST:-$ROOT/manifests/frozen_g0_selective_boxer_full100.json}"

python "$ROOT/tools/build_frozen_anchor_manifest.py" \
  --anchor-name G0-Selective-Boxer \
  --reference-root "$RESULT_ROOT" \
  --scene-list "$SCENE_LIST" \
  --artifact "quality_checkpoint=$G0_ROOT/models/scannet_b6_iou_mlp.npz" \
  --artifact "yoloe_checkpoint=$G0_ROOT/models/yoloe-11s-seg-pf.pt" \
  --artifact "selective_config=$G0_ROOT/config/scannet_b6_selective_boxer.yaml" \
  --artifact "launcher=$G0_ROOT/scripts/run_scannet_b6_selective_boxer.sh" \
  --artifact "boxer_lifter=$G0_ROOT/boxfusion/boxer_lifter.py" \
  --artifact "online_refinement=$G0_ROOT/boxfusion/online_refinement.py" \
  --artifact "quality_score=$G0_ROOT/boxfusion/quality_score.py" \
  --artifact "demo=$G0_ROOT/demo.py" \
  --ap15 40.2787 \
  --ap25 35.4508 \
  --ap50 15.2181 \
  --metadata-json '{"profile":"s1_selective/g0","score_threshold":0.4,"minimum_extent_m":0.4,"quality_detector_blend":0.4,"selective_gate":{"max_center_shift_m":0.1,"min_volume_ratio":0.5,"max_volume_ratio":2.0},"class_agnostic":true}' \
  --required-scene-count 100 \
  --output "$OUTPUT"

python "$ROOT/tools/verify_frozen_anchor_manifest.py" --manifest "$OUTPUT"
