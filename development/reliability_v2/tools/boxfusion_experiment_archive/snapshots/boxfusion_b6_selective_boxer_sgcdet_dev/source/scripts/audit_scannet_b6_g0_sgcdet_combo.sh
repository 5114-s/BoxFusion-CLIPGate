#!/usr/bin/env bash
set -euo pipefail

# CPU-only audit for G0, sparse observer, and sparse identity controls.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_COMBO_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
G0_TAG="${BOXFUSION_COMBO_G0_TAG:-g0_frozen_fixed10_v1}"
OBSERVER_TAG="${BOXFUSION_COMBO_OBSERVER_TAG:-g0_sgcdet_observer_fixed10_v1}"
IDENTITY_TAG="${BOXFUSION_COMBO_IDENTITY_TAG:-g0_sgcdet_identity_fixed10_v1}"

g0_pred="$ROOT/results/b6_g0_sgcdet/$G0_TAG/$list_scope"
observer_pred="$ROOT/results/b6_g0_sgcdet/$OBSERVER_TAG/$list_scope"
identity_pred="$ROOT/results/b6_g0_sgcdet/$IDENTITY_TAG/$list_scope"
g0_boxer="$ROOT/diagnostics/b6_g0_sgcdet/$G0_TAG/$list_scope/boxer"
observer_boxer="$ROOT/diagnostics/b6_g0_sgcdet/$OBSERVER_TAG/$list_scope/boxer"
identity_boxer="$ROOT/diagnostics/b6_g0_sgcdet/$IDENTITY_TAG/$list_scope/boxer"
observer_online="$ROOT/diagnostics/b6_g0_sgcdet/$OBSERVER_TAG/$list_scope/online"
identity_online="$ROOT/diagnostics/b6_g0_sgcdet/$IDENTITY_TAG/$list_scope/online"
report_root="$ROOT/reports/b6_g0_sgcdet/$list_scope"
mkdir -p "$report_root"

echo "Auditing frozen Selective-Boxer G0 in all three control stages"
"$PYTHON" "$ROOT/tools/audit_g0_boxer_active.py" \
    --scene-list "$SCENE_LIST" \
    --stage "g0=$g0_boxer" \
    --stage "observer=$observer_boxer" \
    --stage "identity=$identity_boxer" \
    --json-output "$report_root/boxer_g0_contract.json"

echo "Auditing strict same-run sparse observer/identity contracts"
BOXFUSION_SGCDET_S0_PRED_ROOT="$g0_pred" \
BOXFUSION_SGCDET_S1_PRED_ROOT="$observer_pred" \
BOXFUSION_SGCDET_S2_PRED_ROOT="$identity_pred" \
BOXFUSION_SGCDET_S1_DIAGNOSTICS_ROOT="$observer_online" \
BOXFUSION_SGCDET_S2_DIAGNOSTICS_ROOT="$identity_online" \
BOXFUSION_SGCDET_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_SGCDET_AUDIT_JSON=1 \
    bash "$ROOT/scripts/audit_scannet_sgcdet_sparse_identity.sh" \
    > "$report_root/sparse_identity_contract.json"

"$PYTHON" - "$report_root/sparse_identity_contract.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    report = json.load(handle)
print(json.dumps(report, indent=2, sort_keys=True))
if not report.get("ok", False):
    raise SystemExit("Sparse observer/identity audit failed")
PY

echo "Combined G0/SGCDet control audits passed"
echo "Reports: $report_root"
