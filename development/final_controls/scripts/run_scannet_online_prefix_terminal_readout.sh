#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
SOURCE_RUN="$MAIN/development/final_controls/experiments/mvsr_multiview_audit_20261012"
OUTPUT="$MAIN/reports/scannet_online_prefix_terminal_readout_20261014"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
EVAL_LOG_ROOT="$MAIN/logs/scannet_official100_real_score"
LABEL_PREFIX=recar3d_scannet_online_prefix_terminal_readout_20261014

mkdir -p "$OUTPUT/logs"
exec > >(tee -a "$OUTPUT/logs/driver.log") 2>&1

if [[ ! -s "$OUTPUT/replay_results.json" ]]; then
  "$PYTHON" "$MAIN/tools/replay_scannet_online_prefix.py" \
    --scene-list "$SCENE_LIST" \
    --config "$SOURCE_RUN/config.yaml" \
    --native-prefix-root "$SOURCE_RUN/native_prefix" \
    --evidence-root "$MAIN/development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence" \
    --terminal-readout \
    --native-min-extent-m 0.30 \
    --output "$OUTPUT"
fi

for system in base recar; do
  for tag in p25 p50 p75 p100; do
    bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
      "${LABEL_PREFIX}_${system}_${tag}" \
      "$OUTPUT/predictions/${system}_${tag}"
  done
done

"$PYTHON" "$MAIN/tools/summarize_scannet_online_prefix.py" \
  --replay-results "$OUTPUT/replay_results.json" \
  --eval-log-root "$EVAL_LOG_ROOT" \
  --label-prefix "$LABEL_PREFIX" \
  --output "$OUTPUT/report"

echo "[$(date '+%F %T')] corrected ScanNet-100 prefix terminal-readout experiment complete"
