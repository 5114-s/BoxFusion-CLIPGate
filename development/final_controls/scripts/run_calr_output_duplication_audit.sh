#!/usr/bin/env bash
set -euo pipefail
MAIN=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
FACTORIAL_COMPONENTS="$MAIN/development/final_controls/experiments/mvsr_design_20261005/components"
KEY_CONTROL_COMPONENTS="$MAIN/development/final_controls/results/scannet_recar3d_key_control_components"
OUTPUT="$MAIN/reports/scannet_calr_output_duplication_20261012"
if [[ ! -s "$OUTPUT/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/analyze_calr_output_duplication.py" \
    --components "$FACTORIAL_COMPONENTS" \
    --native-components "$FACTORIAL_COMPONENTS" \
    --plr-components "$FACTORIAL_COMPONENTS" \
    --calr-components "$KEY_CONTROL_COMPONENTS" \
    --native-branch native_factor_no_visibility \
    --plr-branch births_plr_v1 \
    --calr-branch births_calr_v1 \
    --output "$OUTPUT"
fi
