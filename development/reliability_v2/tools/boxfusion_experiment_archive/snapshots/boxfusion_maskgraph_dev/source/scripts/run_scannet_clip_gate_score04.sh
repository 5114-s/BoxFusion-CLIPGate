#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
export BOXFUSION_EXPERIMENT_NAME=scannet_clip_gate_score04
export BOXFUSION_CONFIG="$ROOT/config/scannet_clip_gate_score04.yaml"
export BOXFUSION_REFERENCE_TEXT="Threshold control: Stage-1 CLIP gate with score_thresh=0.4"

exec bash "$ROOT/scripts/run_scannet_clip_gate_scorefix.sh" "${1:-0}"
