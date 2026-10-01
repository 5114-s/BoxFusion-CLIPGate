#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/final_controls"
PIPE="$MAIN/development/final_pipeline"
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
SCENES="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
COMPONENTS="$DEV/results/scannet_recar3d_key_control_components"
EVIDENCE="$DEV/evidence/scannet_recar3d_key_controls"
KEY_REPORT="$DEV/reports/scannet_recar3d_key_controls_official100_20260926/REPORT.md"
FACTORIAL="$PIPE/results/scannet_recar3d_final_factorial100"
FACTORIAL_REPORT="$PIPE/reports/scannet_recar3d_final_factorial100"
AUDIT_REPORT="$PIPE/reports/scannet_recar3d_candidate_audit100"
MVSR_REPORT="$PIPE/reports/scannet_recar3d_mvsr_analysis100"
LOG="$PIPE/logs/scannet_postprocess.log"
EVAL_LOG_ROOT="$MAIN/logs/scannet_official100_real_score"

mkdir -p "$PIPE/logs" "$PIPE/results" "$PIPE/reports"
exec > >(tee -a "$LOG") 2>&1
[[ -s "$KEY_REPORT" ]] || { echo "Missing completed key-control report" >&2; exit 1; }
[[ ! -e "$FACTORIAL/manifest.json" ]] || { echo "Factorial already exists" >&2; exit 1; }

"$PYTHON" "$PIPE/materialize_factorial.py" \
  --scene-list "$SCENES" --expected-scenes 100 \
  --components "$COMPONENTS" --output "$FACTORIAL"
for arm in base p a m2 p_a p_m2 a_m2 p_a_m2; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "recar3d_chain_factorial_${arm}" "$FACTORIAL/$arm"
done
"$PYTHON" "$PIPE/summarize_scannet_factorial.py" \
  --manifest "$FACTORIAL/manifest.json" --log-root "$EVAL_LOG_ROOT" \
  --log-prefix recar3d_chain_factorial --output "$FACTORIAL_REPORT"
"$PYTHON" "$PIPE/audit_candidate_evidence.py" \
  --scene-list "$SCENES" --components "$COMPONENTS" \
  --evidence "$EVIDENCE" --factorial "$FACTORIAL" --output "$AUDIT_REPORT"
"$PYTHON" "$PIPE/analyze_mvsr_ranking.py" \
  --scene-list "$SCENES" --components "$COMPONENTS" --output "$MVSR_REPORT"
sha256sum "$FACTORIAL/manifest.json" "$FACTORIAL_REPORT/results.json" \
  "$AUDIT_REPORT/results.json" "$MVSR_REPORT/results.json" \
  > "$PIPE/reports/scannet_final_evidence.sha256"
echo "[$(date '+%F %T')] ScanNet final postprocessing complete"
