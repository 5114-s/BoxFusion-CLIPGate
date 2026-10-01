#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
export BOXFUSION_EXPERIMENT_NAME=scannet_cgf_paper100_score04
export BOXFUSION_CONFIG="$ROOT/config/scannet_cgf_paper100_score04.yaml"
export BOXFUSION_REFERENCE_TEXT="Strict CGF baseline: official 100 scenes, score_thresh=0.4, no appearance gate, no Reliable Top-K"

exec bash "$ROOT/scripts/run_scannet_clip_gate_scorefix.sh" "${1:-0,1}"
