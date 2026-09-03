#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="$CODE_ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
LIST_SHA256="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
LIST_SCOPE="$(basename "$SCENE_LIST" .txt)-${LIST_SHA256:0:12}"
CONTROLLED_SUMMARY="$CODE_ROOT/reports/boxer_lifting/fixed10_summary.json"
CURRENT_ROUTE_REVISION="$(bash "$CODE_ROOT/scripts/print_boxer_route_revision.sh")"

if [[ ! -s "$CONTROLLED_SUMMARY" ]]; then
  echo "Run the controlled X0/X1/X2 fixed-10 ablation first." >&2
  exit 1
fi
controlled_positive="$(
  "$PYTHON" - "$CONTROLLED_SUMMARY" "$CURRENT_ROUTE_REVISION" "$LIST_SHA256" <<'PY'
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
if [[ "$controlled_positive" != "1" && "${BOXFUSION_BOXER_FORCE_PRE_FILTER:-0}" != "1" ]]; then
  echo "Controlled post-filter Boxer lifting was not positive." >&2
  echo "The full pre-filter route is therefore stopped by default." >&2
  exit 1
fi

export BOXFUSION_BOXER_SCENE_LIST="$SCENE_LIST"

# X0 is reused as the exact original downstream reference.
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" x0_cutr "$GPU_SPEC"
bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" f1_pre_observer "$GPU_SPEC"

"$PYTHON" "$CODE_ROOT/tools/audit_boxer_lifting_contract.py" \
  --scene-list "$SCENE_LIST" \
  --baseline-root "$CODE_ROOT/results/boxer_lifting/x0_cutr" \
  --observer-root "$CODE_ROOT/results/boxer_lifting/f1_pre_observer" \
  --observer-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/f1_pre_observer" \
  --output "$CODE_ROOT/reports/boxer_lifting/full_lifting_fixed10_identity.json"

bash "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" f2_pre_active "$GPU_SPEC"

"$PYTHON" "$CODE_ROOT/tools/audit_boxer_lifting_contract.py" \
  --scene-list "$SCENE_LIST" \
  --baseline-root "$CODE_ROOT/results/boxer_lifting/x0_cutr" \
  --observer-root "$CODE_ROOT/results/boxer_lifting/f1_pre_observer" \
  --observer-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/f1_pre_observer" \
  --active-root "$CODE_ROOT/results/boxer_lifting/f2_pre_active" \
  --active-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/f2_pre_active" \
  --allow-active-schedule-change \
  --output "$CODE_ROOT/reports/boxer_lifting/full_lifting_fixed10_contract.json"

if [[ "$(bash "$CODE_ROOT/scripts/print_boxer_route_revision.sh")" != "$CURRENT_ROUTE_REVISION" ]]; then
  echo "Route source/config changed during the pre-filter run; refusing summary." >&2
  exit 1
fi

"$PYTHON" "$CODE_ROOT/tools/summarize_boxer_lifting_ablation.py" \
  --baseline-log "$CODE_ROOT/logs/boxer_lifting/x0_cutr/$LIST_SCOPE/eval_stdout.log" \
  --observer-log "$CODE_ROOT/logs/boxer_lifting/f1_pre_observer/$LIST_SCOPE/eval_stdout.log" \
  --active-log "$CODE_ROOT/logs/boxer_lifting/f2_pre_active/$LIST_SCOPE/eval_stdout.log" \
  --contract-report "$CODE_ROOT/reports/boxer_lifting/full_lifting_fixed10_contract.json" \
  --active-diagnostics "$CODE_ROOT/diagnostics/boxer_lifting/f2_pre_active" \
  --scene-list "$SCENE_LIST" \
  --route-revision "$CURRENT_ROUTE_REVISION" \
  --phase fixed10 \
  --output "$CODE_ROOT/reports/boxer_lifting/full_lifting_fixed10_summary.json"

echo "Fixed-10 full pre-filter Boxer lifting ablation completed."
