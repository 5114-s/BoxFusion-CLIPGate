#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
LOG_ROOT="$MAIN/reports/requested_audits_20261012"
STATUS="$LOG_ROOT/status.tsv"
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/resume_stage3_stage4.log") 2>&1

run_stage() {
  local index="$1" name="$2" script="$3"
  printf '%s\t%s\tRESTARTED\t%s\n' "$index" "$name" "$(date --iso-8601=seconds)" >> "$STATUS"
  bash "$script"
  printf '%s\t%s\tCOMPLETED\t%s\n' "$index" "$name" "$(date --iso-8601=seconds)" >> "$STATUS"
}

run_stage 3 scannet100_online_prefix_quality \
  "$MAIN/development/final_controls/scripts/run_scannet_online_prefix_quality.sh"
run_stage 4 scannet100_calr_duplication_audit \
  "$MAIN/development/final_controls/scripts/run_calr_output_duplication_audit.sh"

printf 'all\tcomputational_queue\tCOMPLETED\t%s\n' "$(date --iso-8601=seconds)" >> "$STATUS"
