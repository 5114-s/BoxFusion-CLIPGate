#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SCENE="${2:-scene0568_00}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_LIST="/tmp/boxfusion_boxer_${SCENE}_smoke.txt"

printf '%s\n' "$SCENE" >"$SMOKE_LIST"
export BOXFUSION_BOXER_SCENE_LIST="$SMOKE_LIST"

bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x0_cutr "$GPU"
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x0_replay "$GPU"
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x1_observer "$GPU"
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x2_active "$GPU"

"${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}" \
  "$CODE_ROOT/tools/audit_boxer_lifting_contract.py" \
  --scene-list "$SMOKE_LIST" \
  --baseline-root "$CODE_ROOT/results/boxer_lifting/x0_cutr_replay" \
  --observer-root "$CODE_ROOT/results/boxer_lifting/x1_boxer_observer" \
  --observer-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/x1_boxer_observer" \
  --observer-box-atol 0.0001 \
  --active-root "$CODE_ROOT/results/boxer_lifting/x2_boxer_active" \
  --active-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/x2_boxer_active" \
  --proposal-cache-root "$CODE_ROOT/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2" \
  --proposal-cache-source-root "$CODE_ROOT/results/boxer_lifting/x0_cutr" \
  --output "$CODE_ROOT/reports/boxer_lifting/${SCENE}_smoke_contract.json"

echo "One-scene Boxer lifting smoke completed: $SCENE"
