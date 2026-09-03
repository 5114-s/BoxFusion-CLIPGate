#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
export BOXFUSION_EXPERIMENT_NAME=scannet_clip_gate_topk_fusion_score04
export BOXFUSION_CONFIG="$ROOT/config/scannet_clip_gate_topk_fusion_score04.yaml"
export BOXFUSION_STAGE1_EXPERIMENT=scannet_clip_gate_score04
export BOXFUSION_REFERENCE_TEXT="Paired Stage-1 control: CLIP gate with score_thresh=0.4"

exec bash "$ROOT/scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh" "${1:-0}"
