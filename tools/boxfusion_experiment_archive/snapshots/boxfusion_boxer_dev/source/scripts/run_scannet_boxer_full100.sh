#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="$CODE_ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
LIST_SHA256="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
LIST_SCOPE="$(basename "$SCENE_LIST" .txt)-${LIST_SHA256:0:12}"
FIXED10_SUMMARY="$CODE_ROOT/reports/boxer_lifting/fixed10_summary.json"
FIXED10_LIST="$CODE_ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
CURRENT_ROUTE_REVISION="$(bash "$CODE_ROOT/scripts/print_boxer_route_revision.sh")"
EXPECTED_FIXED10_LIST_SHA256="$(sha256sum "$FIXED10_LIST" | awk '{print $1}')"

if [[ ! -s "$FIXED10_SUMMARY" ]]; then
  echo "Missing fixed-10 decision report: $FIXED10_SUMMARY" >&2
  echo "Run scripts/run_scannet_boxer_fixed10.sh first." >&2
  exit 1
fi

recommend_full100="$(
  "$PYTHON" - "$FIXED10_SUMMARY" "$CURRENT_ROUTE_REVISION" "$EXPECTED_FIXED10_LIST_SHA256" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    report = json.load(handle)
valid = (
    report.get("recommend_full100", False)
    and report.get("route_revision") == sys.argv[2]
    and report.get("scene_list_sha256") == sys.argv[3]
)
print("1" if valid else "0")
PY
)"
if [[ "$recommend_full100" != "1" && "${BOXFUSION_BOXER_FORCE_FULL100:-0}" != "1" ]]; then
  echo "Fixed-10 gate rejected the 100-scene run." >&2
  echo "Inspect $FIXED10_SUMMARY; set BOXFUSION_BOXER_FORCE_FULL100=1 only for a documented override." >&2
  exit 1
fi

export BOXFUSION_BOXER_SCENE_LIST="$SCENE_LIST"

# Existing fixed-10 artifacts are reused only when their code/config/weight
# fingerprint matches.  The runner fails closed on stale predictions.
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x0_cutr "$GPU_SPEC"
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x0_replay "$GPU_SPEC"
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x1_observer "$GPU_SPEC"

"$PYTHON" "$CODE_ROOT/tools/audit_boxer_lifting_contract.py" \
  --scene-list "$SCENE_LIST" \
  --baseline-root "$CODE_ROOT/results/boxer_lifting/x0_cutr_replay" \
  --observer-root "$CODE_ROOT/results/boxer_lifting/x1_boxer_observer" \
  --observer-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/x1_boxer_observer" \
  --observer-box-atol 0.0001 \
  --proposal-cache-root "$CODE_ROOT/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2" \
  --proposal-cache-source-root "$CODE_ROOT/results/boxer_lifting/x0_cutr" \
  --output "$CODE_ROOT/reports/boxer_lifting/full100_identity.json"

bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x2_active "$GPU_SPEC"

"$PYTHON" "$CODE_ROOT/tools/audit_boxer_lifting_contract.py" \
  --scene-list "$SCENE_LIST" \
  --baseline-root "$CODE_ROOT/results/boxer_lifting/x0_cutr_replay" \
  --observer-root "$CODE_ROOT/results/boxer_lifting/x1_boxer_observer" \
  --observer-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/x1_boxer_observer" \
  --observer-box-atol 0.0001 \
  --active-root "$CODE_ROOT/results/boxer_lifting/x2_boxer_active" \
  --active-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/x2_boxer_active" \
  --proposal-cache-root "$CODE_ROOT/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2" \
  --proposal-cache-source-root "$CODE_ROOT/results/boxer_lifting/x0_cutr" \
  --output "$CODE_ROOT/reports/boxer_lifting/full100_contract.json"

if [[ "$(bash "$CODE_ROOT/scripts/print_boxer_route_revision.sh")" != "$CURRENT_ROUTE_REVISION" ]]; then
  echo "Route source/config changed during the full-100 run; refusing summary." >&2
  exit 1
fi

"$PYTHON" "$CODE_ROOT/tools/summarize_boxer_lifting_ablation.py" \
  --baseline-log "$CODE_ROOT/logs/boxer_lifting/x0_replay/$LIST_SCOPE/eval_stdout.log" \
  --observer-log "$CODE_ROOT/logs/boxer_lifting/x1_observer/$LIST_SCOPE/eval_stdout.log" \
  --active-log "$CODE_ROOT/logs/boxer_lifting/x2_active/$LIST_SCOPE/eval_stdout.log" \
  --contract-report "$CODE_ROOT/reports/boxer_lifting/full100_contract.json" \
  --active-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/x2_boxer_active" \
  --scene-list "$SCENE_LIST" \
  --route-revision "$CURRENT_ROUTE_REVISION" \
  --phase full100 \
  --output "$CODE_ROOT/reports/boxer_lifting/full100_summary.json"

echo "Full-100 paired Boxer lifting ablation completed."
