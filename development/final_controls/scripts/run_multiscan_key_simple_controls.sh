#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
PREPARED="$MAIN/data/multiscan_boxfusion_eval50"
EXPERIMENT="$MAIN/development/final_controls/experiments/multiscan50_novis_20261006"
PATCH_ROOT="$MAIN/development/final_controls/experiments/multiscan50_mvsr_temporal_patch_20261012"
OUTPUT="$MAIN/reports/multiscan50_key_simple_controls_20261012"
ARMS="$OUTPUT/predictions"
mkdir -p "$OUTPUT/logs"
exec > >(tee -a "$OUTPUT/logs/driver.log") 2>&1

bash "$MAIN/development/final_controls/scripts/run_multiscan50_mvsr_temporal_patch.sh"

if [[ ! -s "$ARMS/manifest.json" ]]; then
  "$PYTHON" "$MAIN/tools/materialize_multiscan_key_simple_controls.py" \
    --prepared-root "$PREPARED" --experiment-root "$EXPERIMENT" \
    --mvsr-patch-root "$PATCH_ROOT" --output "$ARMS"
fi

"$PYTHON" "$MAIN/tools/eval_multiscan_boxfusion.py" \
  --prepared-root "$PREPARED" \
  --prediction full_reference "$ARMS/full_reference" \
  --prediction plr_direct "$ARMS/plr_direct" \
  --prediction calr_score_memory "$ARMS/calr_score_memory" \
  --prediction mvsr_max "$ARMS/mvsr_max" \
  --prediction mvsr_latest "$ARMS/mvsr_latest" \
  --output "$OUTPUT/evaluation"

if [[ ! -s "$OUTPUT/bootstrap_plr/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_multiscan_pairs.py" \
    --prepared-root "$PREPARED" --base-root "$ARMS/plr_direct" \
    --treatment "plr_causal=$ARMS/full_reference" \
    --replicates 2000 --seed 20261012 --output "$OUTPUT/bootstrap_plr"
fi
if [[ ! -s "$OUTPUT/bootstrap_calr/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_multiscan_pairs.py" \
    --prepared-root "$PREPARED" --base-root "$ARMS/calr_score_memory" \
    --treatment "calr_three_frame=$ARMS/full_reference" \
    --replicates 2000 --seed 20261012 --output "$OUTPUT/bootstrap_calr"
fi
if [[ ! -s "$OUTPUT/bootstrap_mvsr/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_multiscan_pairs.py" \
    --prepared-root "$PREPARED" --base-root "$ARMS/mvsr_latest" \
    --treatment "mvsr_historical_max=$ARMS/mvsr_max" \
    --replicates 2000 --seed 20261012 --output "$OUTPUT/bootstrap_mvsr"
fi
echo "[$(date '+%F %T')] MultiScan-50 key simple controls complete"
