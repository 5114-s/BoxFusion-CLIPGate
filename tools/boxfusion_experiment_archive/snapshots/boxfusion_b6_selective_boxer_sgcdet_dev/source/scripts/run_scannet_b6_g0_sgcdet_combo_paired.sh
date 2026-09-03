#!/usr/bin/env bash
set -euo pipefail

# Complete fixed10 protocol. Existing fingerprint-matched scenes resume safely.

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT"
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh g0 "$GPU_SPEC"
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh observer "$GPU_SPEC"
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh identity "$GPU_SPEC"
bash scripts/audit_scannet_b6_g0_sgcdet_combo.sh
bash scripts/run_scannet_b6_g0_sgcdet_combo.sh active "$GPU_SPEC"
bash scripts/evaluate_scannet_b6_g0_sgcdet_same_run.sh "${GPU_SPEC%%,*}"
