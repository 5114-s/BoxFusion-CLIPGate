#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
export BOXFUSION_EXPERIMENT_NAME=scannet_b05_control_score05
export BOXFUSION_CONFIG="$ROOT/config/scannet_b05_control_score05.yaml"
export BOXFUSION_REFERENCE_TEXT="B05-current: official 100 scenes, score_thresh=0.5, no appearance gate, Reliable-View Top-K disabled"

exec bash "$ROOT/scripts/run_scannet_clip_gate_scorefix.sh" "${1:-0,1}"
