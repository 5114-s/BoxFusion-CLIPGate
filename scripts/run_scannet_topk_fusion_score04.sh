#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
export BOXFUSION_EXPERIMENT_NAME=scannet_topk_fusion_score04
export BOXFUSION_CONFIG="$ROOT/config/scannet_topk_fusion_score04.yaml"
export BOXFUSION_STAGE1_EXPERIMENT=scannet_cgf_paper100_score04
export BOXFUSION_REFERENCE_TEXT="Top-K-only CGF arm: official 100 scenes, score_thresh=0.4, no appearance gate"

exec bash "$ROOT/scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh" "${1:-0,1}"
