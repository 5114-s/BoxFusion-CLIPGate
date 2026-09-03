#!/usr/bin/env bash
set -euo pipefail

# Strict same-run identity audit plus report-only cross-run drift analysis.
# This script is CPU/read-only and never starts inference.  Independent GPU
# runs are not raw-byte comparable because optional model construction and
# CUDA kernels may change floating-point/RNG state.  Mutation is instead
# proven against the exact pre/post rows stored by each run's diagnostics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_ROOT="${BOXFUSION_SGCDET_S0_PRED_ROOT:-$ROOT/results/sgcdet_sparse_s0_frozen_b6_fixed10_v1}"
OBSERVER_ROOT="${BOXFUSION_SGCDET_S1_PRED_ROOT:-$ROOT/results/sgcdet_sparse_s1_observer_fixed10_v1}"
IDENTITY_ROOT="${BOXFUSION_SGCDET_S2_PRED_ROOT:-$ROOT/results/sgcdet_sparse_s2_identity_fixed10_v1}"
OBSERVER_DIAGNOSTICS_ROOT="${BOXFUSION_SGCDET_S1_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/sgcdet_sparse_s1_observer_fixed10_v1}"
IDENTITY_DIAGNOSTICS_ROOT="${BOXFUSION_SGCDET_S2_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/sgcdet_sparse_s2_identity_fixed10_v1}"
SCENE_LIST="${BOXFUSION_SGCDET_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="${BOXFUSION_SGCDET_AUDIT_PYTHON:-$ENV_ROOT/bin/python}"
AUDIT_TOOL="$ROOT/tools/audit_sgcdet_sparse_identity.py"
JSON_OUTPUT="${BOXFUSION_SGCDET_AUDIT_JSON:-0}"

for file in "$PYTHON" "$AUDIT_TOOL" "$SCENE_LIST"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing sparse identity audit dependency: $file" >&2
        exit 1
    fi
done

if [[ "$JSON_OUTPUT" != "0" && "$JSON_OUTPUT" != "1" ]]; then
    echo "BOXFUSION_SGCDET_AUDIT_JSON must be 0 or 1" >&2
    exit 1
fi
arguments=(
    "$AUDIT_TOOL"
    --baseline-root "$BASELINE_ROOT"
    --observer-root "$OBSERVER_ROOT"
    --identity-root "$IDENTITY_ROOT"
    --observer-diagnostics-root "$OBSERVER_DIAGNOSTICS_ROOT"
    --identity-diagnostics-root "$IDENTITY_DIAGNOSTICS_ROOT"
    --scene-list "$SCENE_LIST"
)
if [[ "$JSON_OUTPUT" == "1" ]]; then
    arguments+=(--json)
fi

exec "$PYTHON" "${arguments[@]}"
