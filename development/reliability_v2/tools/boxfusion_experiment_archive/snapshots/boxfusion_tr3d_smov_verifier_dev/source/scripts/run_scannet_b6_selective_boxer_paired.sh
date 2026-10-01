#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_SPEC="${1:-0,1}"

echo "Running frozen B6 control, non-mutating observer, and Selective Boxer"
bash "$ROOT/scripts/run_scannet_b6_selective_boxer.sh" s0_control "$GPU_SPEC"
bash "$ROOT/scripts/run_scannet_b6_selective_boxer.sh" s0_observer "$GPU_SPEC"
bash "$ROOT/scripts/run_scannet_b6_selective_boxer.sh" s1_selective "$GPU_SPEC"

echo "Running mandatory identity and row-wise fallback audit"
exec bash "$ROOT/scripts/audit_scannet_b6_selective_boxer.sh"
