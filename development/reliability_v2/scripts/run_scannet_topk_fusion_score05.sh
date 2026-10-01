#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
export BOXFUSION_EXPERIMENT_NAME=scannet_topk_fusion_score05
export BOXFUSION_CONFIG="$ROOT/config/scannet_topk_fusion_score05.yaml"
export BOXFUSION_STAGE1_EXPERIMENT=scorefix_baseline_score05
export BOXFUSION_REFERENCE_TEXT="T05: official 100 scenes, score_thresh=0.5, no appearance gate, Reliable-View Top-K enabled"

exec bash "$ROOT/scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh" "${1:-0,1}"
