#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAIR_RUN_ID="${BOXFUSION_RETRAINED_PAIR_RUN_ID:-v2}"

if [[ ! "$PAIR_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid BOXFUSION_RETRAINED_PAIR_RUN_ID: $PAIR_RUN_ID" >&2
    exit 2
fi

G0_TAG="g0_retrained_frozen_fixed10_${PAIR_RUN_ID}"
OBSERVER_TAG="g0_retrained_observer_fixed10_${PAIR_RUN_ID}"
IDENTITY_TAG="g0_retrained_identity_fixed10_${PAIR_RUN_ID}"
ACTIVE_TAG="g0_retrained_active_fixed10_${PAIR_RUN_ID}"
COUNTERFACTUAL_TAG="g0_retrained_active_fixed10_identity_${PAIR_RUN_ID}"

echo "G0 retrained paired fixed10 run namespace: $PAIR_RUN_ID"
echo "  g0/observer/identity/active: $G0_TAG / $OBSERVER_TAG / $IDENTITY_TAG / $ACTIVE_TAG"

export BOXFUSION_RETRAINED_RUN_TAG="$G0_TAG"
bash "$SCRIPT_DIR/run_scannet_b6_g0_sgcdet_retrained.sh" g0 "$GPU_SPEC"
export BOXFUSION_RETRAINED_RUN_TAG="$OBSERVER_TAG"
bash "$SCRIPT_DIR/run_scannet_b6_g0_sgcdet_retrained.sh" observer "$GPU_SPEC"
export BOXFUSION_RETRAINED_RUN_TAG="$IDENTITY_TAG"
bash "$SCRIPT_DIR/run_scannet_b6_g0_sgcdet_retrained.sh" identity "$GPU_SPEC"

BOXFUSION_COMBO_G0_TAG="$G0_TAG" \
BOXFUSION_COMBO_OBSERVER_TAG="$OBSERVER_TAG" \
BOXFUSION_COMBO_IDENTITY_TAG="$IDENTITY_TAG" \
    bash "$SCRIPT_DIR/audit_scannet_b6_g0_sgcdet_combo.sh"

export BOXFUSION_RETRAINED_RUN_TAG="$ACTIVE_TAG"
bash "$SCRIPT_DIR/run_scannet_b6_g0_sgcdet_retrained.sh" active "$GPU_SPEC"

BOXFUSION_COMBO_ACTIVE_TAG="$ACTIVE_TAG" \
BOXFUSION_COMBO_COUNTERFACTUAL_TAG="$COUNTERFACTUAL_TAG" \
    bash "$SCRIPT_DIR/evaluate_scannet_b6_g0_sgcdet_same_run.sh" "${GPU_SPEC%%,*}"
