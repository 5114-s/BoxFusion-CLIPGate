#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_B6_BOXER_FINAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
PYTHON="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}/bin/python"

case "$PROFILE" in
    f1_observer)
        EXPECTED_MODE="observer"
        ;;
    f2_active)
        EXPECTED_MODE="active"
        ;;
    *)
        echo "Profile must be f1_observer or f2_active" >&2
        exit 2
        ;;
esac

if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing scene list: $SCENE_LIST" >&2
    exit 1
fi
list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
diagnostics_root="$ROOT/diagnostics/b6_boxer_uncertainty_final/$PROFILE/$list_scope/final"
report="$ROOT/reports/b6_boxer_uncertainty_final/${PROFILE}_${list_scope}_audit.json"

exec "$PYTHON" "$ROOT/tools/audit_final_boxer_uncertainty.py" \
    --diagnostics-root "$diagnostics_root" \
    --expected-mode "$EXPECTED_MODE" \
    --scene-list "$SCENE_LIST" \
    --output "$report"
