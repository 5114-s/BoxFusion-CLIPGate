#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
MVSR="$MAIN/development/final_controls/experiments/mvsr_multiview_audit_20261012"
SCANNET_FACTORIAL="$MAIN/development/final_controls/experiments/mvsr_design_20261005/factorial_no_visibility"
CA1M_FACTORIAL="$MAIN/development/final_controls/experiments/ca1m_novis107_20261006/factorial_no_visibility"
OUT="$MAIN/reports/supplement_offline_statistics_20261012"
mkdir -p "$OUT/logs" "$OUT/mplconfig"
exec > >(tee -a "$OUT/logs/driver.log") 2>&1

MPLCONFIGDIR="$OUT/mplconfig" "$PYTHON" "$MAIN/tools/plot_mvsr_support_ranking.py" \
  --scene-list "$SCENE_LIST" --components "$MVSR/components" \
  --reranked-component native_factor_no_visibility_max \
  --bootstrap-iterations 10000 --bootstrap-seed 20261012 \
  --expected-native-boxes 1787 --output-dir "$OUT/mvsr_support_ranking_1787"

"$PYTHON" "$MAIN/tools/audit_mvsr_label_definitions.py" \
  --components "$MVSR/components" --output "$OUT/mvsr_label_audit_1787"

if [[ ! -s "$OUT/mvsr_max_vs_top3/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_final_recar_pairs.py" \
    --dataset scannet --scene-list "$SCENE_LIST" \
    --base-root "$MVSR/components/native_factor_no_visibility_top3_mean" \
    --treatment "maximum=$MVSR/components/native_factor_no_visibility_max" \
    --replicates 2000 --seed 20261012 --output "$OUT/mvsr_max_vs_top3"
fi

if [[ ! -s "$OUT/scannet_factorial_all_vs_base/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_final_recar_pairs.py" \
    --dataset scannet --scene-list "$SCENE_LIST" \
    --base-root "$SCANNET_FACTORIAL/base" \
    --treatment "plr=$SCANNET_FACTORIAL/plr" \
    --treatment "calr=$SCANNET_FACTORIAL/calr" \
    --treatment "mvsr=$SCANNET_FACTORIAL/mvsr" \
    --treatment "plr_calr=$SCANNET_FACTORIAL/plr_calr" \
    --treatment "plr_mvsr=$SCANNET_FACTORIAL/plr_mvsr" \
    --treatment "calr_mvsr=$SCANNET_FACTORIAL/calr_mvsr" \
    --treatment "full=$SCANNET_FACTORIAL/full" \
    --replicates 2000 --seed 20261012 --output "$OUT/scannet_factorial_all_vs_base"
fi

if [[ ! -s "$OUT/ca1m_factorial_all_vs_base/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_final_recar_pairs.py" \
    --dataset ca1m \
    --scene-list "$MAIN/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt" \
    --ca1m-data-root /extra/ZhaoX/boxfusion_ca1m \
    --base-root "$CA1M_FACTORIAL/base" \
    --treatment "plr=$CA1M_FACTORIAL/p" \
    --treatment "calr=$CA1M_FACTORIAL/a" \
    --treatment "mvsr=$CA1M_FACTORIAL/m2" \
    --treatment "plr_calr=$CA1M_FACTORIAL/p_a" \
    --treatment "plr_mvsr=$CA1M_FACTORIAL/p_m2" \
    --treatment "calr_mvsr=$CA1M_FACTORIAL/a_m2" \
    --treatment "full=$CA1M_FACTORIAL/p_a_m2" \
    --replicates 2000 --seed 20261012 --output "$OUT/ca1m_factorial_all_vs_base"
fi

touch "$OUT/DONE"
echo "[$(date --iso-8601=seconds)] supplementary offline statistics complete"
