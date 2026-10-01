#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
PIPE="$MAIN/development/final_pipeline"
STATUS="$PIPE/reports/final_evidence_chain"
KEY_SESSION=recar3d_key_controls100
KEY_REPORT="$MAIN/development/final_controls/reports/scannet_recar3d_key_controls_official100_20260926/REPORT.md"

mkdir -p "$STATUS"
exec 9>"$STATUS/run.lock"
flock -n 9 || { echo "Final evidence chain is already active" >&2; exit 1; }
exec > >(tee -a "$STATUS/driver.log") 2>&1

stage() {
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" | tee -a "$STATUS/stages.tsv"
}
failed() {
  code=$?
  stage FAILED "exit=$code command=${BASH_COMMAND}"
  exit "$code"
}
trap failed ERR

stage WAITING "ScanNet key controls"
while tmux has-session -t "$KEY_SESSION" 2>/dev/null; do
  sleep 30
done
[[ -s "$KEY_REPORT" ]] || { echo "ScanNet key-control run ended without REPORT.md" >&2; exit 1; }
stage COMPLETE "ScanNet key controls"

stage RUNNING "ScanNet factorial + no-child audit + MVSR behavior"
bash "$PIPE/postprocess_scannet.sh"
stage COMPLETE "ScanNet factorial + no-child audit + MVSR behavior"

stage RUNNING "CA-1M-107 final configuration + factorial"
bash "$PIPE/run_ca1m_final107.sh"
stage COMPLETE "CA-1M-107 final configuration + factorial"

stage RUNNING "matched single-GPU FPS/latency/memory/deadline"
bash "$PIPE/run_final_fps100.sh"
stage COMPLETE "matched single-GPU FPS/latency/memory/deadline"

stage RUNNING "selected-scene process evidence + qualitative figure"
bash "$PIPE/run_final_qualitative.sh"
stage COMPLETE "selected-scene process evidence + qualitative figure"

cat > "$STATUS/REPORT.md" <<EOF
# Final evidence chain complete

- ScanNet key controls: \`development/final_controls/reports/scannet_recar3d_key_controls_official100_20260926\`
- ScanNet factorial: \`development/final_pipeline/reports/scannet_recar3d_final_factorial100\`
- Candidate audit: \`development/final_pipeline/reports/scannet_recar3d_candidate_audit100\`
- MVSR analysis: \`development/final_pipeline/reports/scannet_recar3d_mvsr_analysis100\`
- CA-1M factorial: \`development/final_pipeline/reports/ca1m_recar3d_final107\`
- Efficiency: \`development/final_pipeline/reports/scannet_recar3d_final_fps100\`
- Qualitative figure: \`development/final_pipeline/reports/qualitative_recar3d_final\`
EOF
stage COMPLETE "all required evidence"
